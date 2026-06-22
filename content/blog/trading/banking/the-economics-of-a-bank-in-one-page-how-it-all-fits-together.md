---
title: "The Economics of a Bank in One Page: How It All Fits Together"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The integrated model of a bank: how cheap funding, lending, fees, risk, and a thin capital cushion connect into one machine that turns a tiny return on assets into a double-digit return on equity, and the levers management actually pulls."
tags: ["banking", "bank-economics", "return-on-equity", "net-interest-margin", "leverage", "capital", "credit-risk", "bank-business-model", "roa", "funding"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank is one connected machine: cheap, sticky deposits fund earning assets, the spread plus fees minus costs and loan losses becomes a tiny profit on assets, and leverage on a thin equity cushion multiplies that into a double-digit return on equity. Pull any lever to raise the return and you almost always thin the cushion that lets the bank survive.
>
> - A typical large bank earns about **1.1% on assets (ROA)** and turns it into about **12% on equity (ROE)** only because it funds roughly **\$12 of assets with \$1 of equity** — leverage of about 12 times.
> - The whole income statement is four lines: **net interest income + fee income − operating cost − loan provisions**, then tax. Everything else in the bank exists to move one of those four numbers.
> - The same leverage that multiplies a 1.1% ROA into a 12% ROE multiplies a 1% asset loss into a **12% hit to equity** — which is why an 8% cushion can vanish in one bad year.
> - Management has only a handful of real levers — grow loans, reprice deposits, take more or less credit risk, lever up or down, cut costs, buy back stock — and **every lever except cost-cutting trades higher ROE for more fragility.**
> - The one number to remember: **ROE = ROA × leverage.** A bank is a low-margin business made profitable by borrowing, and made fragile by the same.

Why does a business that earns barely a penny of profit on every dollar it manages get treated as one of the most important institutions in the economy? A grocery store that made 1% on the goods on its shelves would be considered a disaster. A bank that makes 1.1% on its assets is considered healthy, pays its staff well, and is worth more than its book value. The trick is not in the 1.1%. The trick is that the bank does not own most of the dollars it is earning that 1.1% on. It borrowed them — from you, mostly — and it borrowed about twelve dollars for every one dollar of its own. A thin margin on a borrowed mountain is a fat return on the sliver of equity underneath it.

That sentence is the entire business, and it is also the entire danger. The same mountain that turns 1.1% into 12% turns a small loss into a catastrophe. Across this series we have pulled the bank apart function by function — the [maturity-transformation trade at its heart](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread), the [balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity), the [income statement](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions), the [spread](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained), the [four risks](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational), the [capital cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion), and the great failures. This post puts every piece back together into a single picture you can hold in your head.

![Bank economics on one page from funding to assets to income to ROA to leverage to ROE](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-1.png)

The diagram above is the mental model for the whole post: funding flows into earning assets, the assets throw off a spread, fees add a second income stream, costs and loan losses come out, what is left is profit on assets, and leverage on a thin equity cushion turns that profit-on-assets into return on equity. Read it left to right and you have read a bank's annual report. Everything that follows is just zooming into one box at a time and then connecting them back up.

## Foundations: a bank is one machine, not five departments

Before we can connect the pieces, we have to name them. A bank looks, from the outside, like five separate businesses: a place that takes deposits, a place that makes loans, a place that runs your cards and payments, a risk department that says no, and a finance team that talks to investors. The most important idea in this post is that those are not five businesses. They are five stages of **one machine**, and the output of the machine is a single number: **return on equity**, the profit the bank earns for each dollar its shareholders have put in.

Let me build the machine from zero, defining each term the first time it appears.

**Funding** is where the bank gets its money. A bank is unusual: almost all the money it works with belongs to other people. The biggest source is *deposits* — the balances in your checking and savings accounts, which the bank legally owes back to you but in practice can lend out because not everyone withdraws at once. A smaller source is *wholesale funding* — money the bank borrows from other financial institutions, in the *repo market* (borrowing overnight against securities as collateral) or by issuing *bonds* (longer-term IOUs to investors). And a sliver — usually about 8% — is *equity*, the shareholders' own money, which is not borrowed and never has to be paid back. Equity is special: it is the only money that can absorb a loss without the bank owing anyone.

**Assets** are where the bank puts that money to work. The largest bucket is *loans* — to households (mortgages, cards, auto, personal) and to companies (term loans, revolving credit lines). The second bucket is *securities* — bonds the bank buys, mostly government and mortgage-backed, that pay interest and can be sold for cash. A third bucket is *cash and reserves* — money parked at the central bank or in the vault, earning little but instantly available. Loans and securities are the *earning assets*; cash is mostly a liquidity buffer.

**Income** is what those assets earn, net of what the funding costs. *Net interest income* (NII) is the single biggest line: the interest the bank collects on its loans and securities minus the interest it pays on its deposits and borrowings. That gap, expressed as a percentage of earning assets, is the *net interest margin* (NIM) — usually around 3% for a US bank. On top of NII sits *fee income* — money the bank earns without lending: card interchange, account fees, advisory fees, asset-management fees, payment fees. Fees are *capital-light*, meaning they do not require the bank to hold much equity against them, which makes them prized.

**Costs and losses** are what comes out. *Operating expense* (opex) is the cost of running the bank — staff, branches, technology, compliance. The ratio of opex to revenue is the *efficiency ratio*, and lower is better. Then there are *provisions* — money the bank sets aside for loans it expects to go bad. A provision is a charge against profit today for losses the bank forecasts tomorrow. In a good year provisions are small; in a downturn they explode.

**Profit, ROA, and ROE** are the output. Subtract costs and provisions from income, take off tax, and you get *net profit*. Divide profit by total assets and you get *return on assets* (ROA) — the margin the machine earns on everything it manages, typically around 1%. Multiply ROA by *leverage* — the ratio of assets to equity, usually about 12 — and you get *return on equity* (ROE), the headline number investors judge the bank on. This identity, **ROE = ROA × leverage**, is the spine of bank economics and we will return to it constantly. (We unpack it in full in [ROE, ROA, and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged); here we connect it to everything else.)

**The cushion.** Underneath the whole machine sits the equity cushion. Because the bank owes its depositors and bondholders their money back in full, any loss on its assets comes straight out of equity. With equity at 8% of assets, the bank can lose 8% of its assets before it is *insolvent* — owing more than it owns. That 8% is the entire margin of safety. The machine's job is to earn a return on that 8%; the cushion's job is to keep the machine from dying when it doesn't.

Hold those six words in your head — **funding, assets, income, costs, profit, cushion** — and you can reconstruct any bank's economics from a blank page. The rest of this post walks each one, then connects them.

## The funding engine: cheap, sticky money is the whole franchise

Start where the money starts. A bank's first and most important advantage is not that it lends well — anyone with capital can lend. It is that it funds itself with money that is *cheaper and stickier* than anyone else can get. That is the franchise. Strip it away and a bank is just a slow, regulated hedge fund.

![The two sides of a bank balance sheet funding and assets](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-2.png)

The left bar above is the funding side. About **71% of the balance sheet is deposits**. Most of those deposits pay very little — a checking account might pay nothing, a savings account 1% to 2% even when the central bank's rate is 5%. The reason banks can pay so little is that deposits are *sticky*: people leave money in their accounts for convenience, for the payment services attached, for the deposit insurance, and out of sheer inertia. A bank's deposit base is built over decades and does not walk out the door when a competitor offers ten basis points more. (A *basis point* is one hundredth of a percent — 0.01%.)

That stickiness is worth real money, and we can measure exactly how much. The key concept is *deposit beta* — the fraction of a central-bank rate increase that a bank passes through to depositors. If the central bank raises rates by 5 percentage points and the bank only raises deposit rates by 1.5 points, its deposit beta is 0.30. The lower the beta, the more of the rate rise the bank keeps as extra spread.

#### Worked example: how much a cheap deposit base is worth

Take two banks, each with \$100 billion of deposits. Bank A has a sleepy, loyal retail base with a deposit beta of 0.30; Bank B funds itself with rate-sensitive corporate cash with a beta of 0.70. The central bank raises rates by 5 percentage points.

- **Bank A's deposit cost rises by** 5% × 0.30 = **1.5 points.** Its annual interest bill on \$100 billion of deposits rises by \$100bn × 1.5% = **\$1.5 billion.**
- **Bank B's deposit cost rises by** 5% × 0.70 = **3.5 points.** Its bill rises by \$100bn × 3.5% = **\$3.5 billion.**

The difference is **\$2.0 billion a year**, on the funding side alone, for two banks that look identical on the asset side. If both lend that money out at the same rate, Bank A's profit is \$2 billion higher purely because its depositors don't demand a competitive rate. *That \$2 billion is the cash value of a cheap, sticky deposit base — it is the franchise that everything else is built on.*

This is why the deepest survival question for any bank is not "how good are its loans?" but "how loyal is its funding?". The wholesale slice on the funding bar — about 10% repo and 7% long-term debt — is the opposite of sticky. Wholesale lenders price every day and flee at the first whiff of trouble. A bank that funds itself mostly with deposits has a fortress; a bank that funds itself with hot wholesale money is renting its funding and can be evicted overnight. We saw this break [Northern Rock](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) and Lehman: the asset losses were the kindling, but the wholesale-funding run was the fire.

So the first stage of the machine sets the tone for the whole thing. Cheap funding is the raw material; how cheap and how sticky it is determines both how much profit the rest of the machine can make and how easily the whole thing can die. It is worth pausing on how strange this is. In almost every other business, the cost of your raw material is set by a market you cannot control — a steelmaker pays the going price for iron ore. A bank's biggest cost, its funding, is set partly by *loyalty* — by how little its depositors demand to leave their money where it is. That loyalty is built over decades, through branches and brands and the simple friction of switching accounts, and it cannot be bought quickly. It is the one part of the machine a competitor cannot replicate by spending money, which is exactly why it is called the franchise and why banks defend their deposit base far more fiercely than their loan book.

## The asset engine: turning borrowed money into a spread

The right bar in the figure above is the asset side — where the funded money goes. About **52% is loans, 22% is securities, 13% is cash, and 13% is trading and other.** The job of the asset engine is to earn more on these assets than the funding cost, and to do it without taking so much risk that the losses eat the spread.

Here is the central mechanic, the *maturity transformation* this whole series is built on. The bank borrows short — deposits can be withdrawn tomorrow — and lends long — a mortgage runs 30 years, a corporate term loan 5. Long assets pay more than short funding, so the bank earns the gap. That gap, the *spread*, is the engine's output. But the gap exists precisely because the bank is taking two risks the depositor isn't: the risk that rates move against it (the long asset is locked in while the short funding reprices), and the risk that the borrower doesn't pay back.

#### Worked example: the spread on one dollar of deposit

Follow a single dollar through the asset engine. You deposit \$1.00. The bank pays you 1.5% on it, so it owes you 1.5 cents a year.

- The bank cannot lend all of it — it must hold some as cash and reserves and some in liquid securities. Say it lends 80 cents at a 6% loan rate and puts 20 cents in securities at 4%.
- **Interest earned:** \$0.80 × 6% + \$0.20 × 4% = 4.8 cents + 0.8 cents = **5.6 cents.**
- **Interest paid:** \$1.00 × 1.5% = **1.5 cents.**
- **Net interest income on the dollar:** 5.6 − 1.5 = **4.1 cents**, before any losses.

That 4.1 cents, divided by the dollar of earning assets, is roughly the net interest margin. *Every dollar you deposit becomes about four cents of gross spread for the bank — small, but multiplied across trillions of dollars, it is the largest profit line in banking.* Now hold onto that 4.1 cents, because in a moment we will watch costs and losses eat into it, and then watch leverage multiply what survives.

The asset mix is itself a risk dial. Loans pay the most but carry the most credit risk and tie up the most capital. Securities pay less but are safer and can be sold. Cash pays the least but is instantly available. A bank that shifts its mix toward loans raises its spread and its risk together; a bank that piles into long-dated securities — as [Silicon Valley Bank](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) did — can earn a bit more yield but takes on enormous interest-rate risk if it doesn't fund those securities with equally long money. The asset engine and the funding engine are not independent: the bank's survival depends on the *match* between them, which is why treasury and asset-liability management exist as a function at all.

## The fee engine: income without a balance sheet

Not all of a bank's income comes from the spread. A growing share comes from *fees* — money earned for services rather than for lending. This is the second income stream feeding the machine, and it has a special property: it is *capital-light*.

Recall that lending requires the bank to hold equity against the loan in case it goes bad. A \$100 loan might require \$8 to \$10 of equity behind it. Fee income requires almost none — when a bank earns a card interchange fee or an advisory fee, it isn't risking its balance sheet, so regulators don't make it hold much capital against that income. The result is that a dollar of fee income generates a *higher return on equity* than a dollar of spread income, because there is less equity in the denominator.

This is why the most admired banks have large fee businesses. Payment processing, wealth management, asset management, card networks, advisory and underwriting — these throw off income year after year without consuming the cushion. They also tend to be *stickier* across the credit cycle: when a recession hits and loan losses spike, fee income from running people's payments and managing their money holds up far better than lending profit. A bank that is half fees is far more resilient than a bank that is all spread.

#### Worked example: why a dollar of fees beats a dollar of spread

Suppose a bank can earn \$1 of pre-tax profit two ways: from lending, or from fees.

- **From lending:** the \$1 of spread profit comes from a loan book that requires \$10 of equity behind it (10% capital). The return on that equity is \$1 ÷ \$10 = **10%** (before tax and costs).
- **From fees:** the \$1 of fee profit requires, say, \$2 of equity for operational risk and infrastructure. The return on that equity is \$1 ÷ \$2 = **50%.**

Same dollar of profit, five times the return on equity, because the fee dollar barely touches the cushion. *This is why "fee income" is a magic phrase in bank analysis — it lets a bank lift its ROE without taking more credit risk or thinning its capital, the one part of the machine that doesn't trade return for fragility.* It is the closest thing a bank has to a free lever, and we will see it again when we get to the management dials.

The catch is that fee income is competitive and, increasingly, regulated. Interchange fees get capped; overdraft fees get banned; advisory fees get squeezed by competition. So while fees are the highest-quality income, they are also the hardest to grow. A bank cannot simply decide to earn more fees the way it can decide to make more loans.

## The income build: four lines and a tax

Now we assemble the income side. The bank's entire profit-and-loss statement, stripped to its bones, is four lines and a tax. Get these four numbers and you have valued the machine's annual output.

![Income build net interest income plus fees minus costs minus provisions equals profit](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-3.png)

The pipeline above is the build, scaled to a bank with \$1,000 of assets so the numbers are easy to carry. Read it top to bottom:

- **Net interest income: +\$28.** The spread on roughly \$920 of earning assets at a ~3% NIM.
- **Fee income: +\$12.** The capital-light second stream.
- **Operating expense: −\$23.** Staff, branches, technology, compliance — the cost of running the bank. With \$40 of total revenue and \$23 of opex, the efficiency ratio is \$23 ÷ \$40 ≈ **58%**, a respectable figure.
- **Provisions: −\$3.** The set-aside for expected loan losses, small in a normal year.
- **Pre-tax profit: +\$14.** What's left for the tax authority and the shareholders.

#### Worked example: a full bank P&L from the line items

Let's build a complete, if simplified, bank from these per-\$1,000 figures and scale it to a \$1 trillion bank. Multiply every line by 1 billion (since \$1 trillion ÷ \$1,000 = 1 billion):

| Line | Per \$1,000 | On \$1 trillion of assets |
|---|---|---|
| Net interest income | +\$28 | +\$28.0bn |
| Fee income | +\$12 | +\$12.0bn |
| **Total revenue** | **+\$40** | **+\$40.0bn** |
| Operating expense | −\$23 | −\$23.0bn |
| Pre-provision profit | +\$17 | +\$17.0bn |
| Provisions | −\$3 | −\$3.0bn |
| **Pre-tax profit** | **+\$14** | **+\$14.0bn** |
| Tax at ~21% | −\$2.9 | −\$2.9bn |
| **Net profit** | **+\$11.1** | **+\$11.1bn** |

So a \$1 trillion bank earns about **\$11.1 billion of net profit.** Divide by \$1 trillion of assets and the **ROA is 1.11%** — right in the normal range. *Notice that the entire annual profit of a trillion-dollar institution is just \$11 per \$1,000 of assets — a margin so thin that a single bad provisions year, doubling that −\$3 to −\$17, would wipe out the whole pre-tax profit.* That fragility is not a bug in the example; it is the defining feature of the business.

The four-line structure tells you exactly where to look when a bank's profit changes. Profit up? It is one of four things: wider spread, more fees, lower costs, or lower provisions. Profit down? Same four, in reverse. There is nowhere else for the number to come from. This is the discipline the income build imposes: it forces every story about a bank's earnings — "the bank had a great quarter," "the bank is in trouble" — back onto one of four concrete levers. We connect this to the full income statement in [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).

## The spread engine over the cycle: why NIM is the heartbeat

The biggest of those four lines — net interest income — is not a constant. It breathes with the rate cycle, and watching it breathe is the single best way to understand why bank earnings are so cyclical. The net interest margin is the heartbeat of the machine.

![US commercial bank net interest margin 2010 to 2024](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-4.png)

The chart traces US commercial-bank NIM from 2010 to 2024. It opens at **3.76% in 2010**, grinds down through the long zero-rate decade to a **trough of 2.56% in 2021**, then snaps back to **3.30% in 2023** as rates rose. That shape is not random — it is the funding and asset engines pulling against each other as rates move.

Here is the mechanism. When the central bank cuts rates to zero, a bank's loan and securities yields fall, but deposit rates are already near zero and can't fall much further — you can't pay a depositor a negative rate without driving them out. So the asset side falls faster than the funding side, and the margin compresses. That is the 2010-2021 decline. When rates rise, the reverse happens, but with a twist: loan yields reprice up quickly (especially floating-rate loans), while deposit rates lag because of that sticky, low deposit beta we measured earlier. The margin widens fast. That is the 2022-2023 snap-back.

#### Worked example: the spread × volume × leverage chain

The reason a small move in NIM matters so much is that it sits at the front of a chain that leverage multiplies at the back. Let me run one move all the way through.

Take our \$1 trillion bank with \$920 billion of earning assets. Suppose NIM widens by just **20 basis points**, from 2.80% to 3.00%, as rates rise.

- **Extra net interest income:** \$920bn × 0.20% = **\$1.84 billion** per year, straight to pre-tax profit.
- After tax at 21%, that is \$1.84bn × 0.79 = **\$1.45 billion** of extra net profit.
- On \$1 trillion of assets, that lifts **ROA by about 0.15 points**, from (say) 1.11% to 1.26%.
- With leverage of 12.5×, the **ROE lift is 0.15% × 12.5 = 1.9 points** — from ~13.8% to ~15.7%.

So a 20-basis-point move in the spread — a number most people would round to zero — becomes nearly **two full points of ROE** by the time it has run through volume and leverage. *That is the spread-volume-leverage chain: a tiny margin, applied to an enormous asset base, then multiplied by leverage, is how banks turn invisible rate moves into visible swings in profitability.* It also runs in reverse, which is why a flattening or inverting yield curve — which squeezes NIM — is a genuine threat to bank earnings, not a market curiosity.

NIM is therefore the number to watch every quarter. It tells you whether the front of the machine is producing more or less raw spread, and because everything downstream multiplies it, a small change there is a large change at the output.

## The risk drains: where the spread leaks away

So far the machine only earns. But every dollar of spread is exposed to losses, and the risk functions exist to make sure those losses stay smaller than the spread. There are four risks a bank runs, and each one drains the machine in a different way. (We map the full taxonomy in [the four risks every bank runs](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational); here we connect them to the income build.)

**Credit risk** is the big one: the risk that borrowers don't repay. It shows up as *provisions* — the −\$3 line in our build. The key insight is that credit risk is *pro-cyclical and lagging*. In good years, almost no loans default, provisions are tiny, and profit looks fantastic. In bad years, defaults cluster all at once, provisions spike, and the same loan book that printed money the year before now bleeds it. The expected loss on any loan is **PD × LGD × EAD** — the *probability of default* times the *loss given default* (the fraction not recovered) times the *exposure at default* (how much is owed). A bank prices its loans to cover the average expected loss, but it survives the downturn only if its capital covers the *unexpected* loss when defaults run far above average.

#### Worked example: how a provision spike eats the profit

Our \$1 trillion bank has, say, \$520 billion of loans. In a normal year its provision charge is \$3 per \$1,000 of total assets = \$3 billion, an annual loss rate on loans of \$3bn ÷ \$520bn ≈ **0.58%.**

Now a recession hits. The annual loss rate on the loan book triples to **1.75%.**

- **New provision charge:** \$520bn × 1.75% = **\$9.1 billion** — up from \$3bn, an extra **\$6.1 billion** charge.
- Recall pre-tax profit was \$14bn. The extra \$6.1bn of provisions cuts it to **\$7.9 billion** — pre-tax profit nearly halved by one line moving.
- If losses run worse — a 3% loss rate, \$15.6bn of provisions — pre-tax profit goes *negative*, and the loss starts eating the equity cushion directly.

*This is why bank earnings boom late in an expansion and crater early in a downturn: the credit-risk drain is nearly dry when times are good and gushes exactly when the bank can least afford it.* The provisions line is the machine's most volatile part, and it is the one that turns a profitable bank into a failing one.

The other three risks drain differently. **Market risk** is the risk that the value of the bank's traded positions and securities falls; it hit SVB through its bond portfolio, where rising rates cut the market value of long-dated securities. **Liquidity risk** is the risk that the bank can't fund itself — that depositors and wholesale lenders pull their money faster than the bank can raise cash — and this is the risk that actually kills banks, often while they are still technically solvent. **Operational risk** is the risk of loss from failed processes, fraud, or cyber-attacks — Barings, the rogue trader, the AML fines. Each of these can punch a hole in the income build or, worse, in the cushion directly. The risk functions are the machine's drainage control: their job is to keep the four leaks smaller than the spread, in every year, including the bad ones.

## The capital constraint: the thin cushion under everything

Now we reach the part that determines whether the machine survives the drains: the *equity cushion.* This is the most counterintuitive piece of bank economics, because more equity makes a bank safer but lowers its ROE — the two goals point in opposite directions.

Recall the funding bar: equity is about **8% of the balance sheet.** That means the bank funds \$92 of every \$100 of assets with other people's money and only \$8 with its own. The \$8 is the cushion — the buffer that absorbs losses before depositors and bondholders are touched. Because the bank owes its creditors in full, every dollar of asset loss comes straight out of that \$8. Lose \$8 of asset value and the cushion is gone; the bank is insolvent.

The arithmetic is brutal in its simplicity. With an 8% cushion, **an asset loss of more than 8% wipes out the bank's equity.** With the thinner cushions some banks ran — Lehman at about 3% equity, [30 times leverage](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — a loss of just **3% of assets** is fatal. This is the whole reason regulators obsess over capital ratios and why Basel rules force minimum cushions: the cushion is the only thing standing between an ordinary bad year and a bank failure. We go deep on this in [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) and on the rules in [BIS and Basel regulation](/blog/trading/finance/bis-and-basel-bank-regulation).

But here is the tension that makes bank management an art. That same thin cushion is *why ROE is high.* If a bank held 50% equity — funded half its assets with shareholder money — it would be incredibly safe, but its ROA of 1.1% would translate into an ROE of just 1.1% ÷ 50% = 2.2%, a terrible return that no investor would fund. The thinness of the cushion is not an accident or a sin; it is the source of the bank's profitability. Management is perpetually choosing how thin to make the cushion — thin enough to earn an attractive ROE, thick enough to survive a downturn. That single choice, more than any other, separates a fortress bank from a fragile one.

## The amplifier: ROA × leverage = ROE

We have now built every piece. It is time to connect the income build to the cushion through the one identity that runs the whole machine: **ROE = ROA × leverage.** This is the amplifier, and it is where a low-margin business becomes a high-return one.

![Return on assets multiplied by leverage equals return on equity amplifier](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-5.png)

The stack above shows the amplification on a clean \$100 of assets. The bank funds \$100 of assets with \$92 of deposits and debt plus \$8 of equity — leverage of \$100 ÷ \$8 = 12.5×. The income machine earns \$1.10 of profit on the \$100, so ROA = 1.1%. But the shareholders only put in \$8, so *their* return is \$1.10 ÷ \$8 = **13.8%.** Same profit, measured against the thin cushion instead of the whole balance sheet, is a double-digit return. That is the entire reason the bank exists as a business.

#### Worked example: pull one lever — more leverage — and watch ROE and fragility move together

Take our bank earning a steady 1.1% ROA. Management decides to lever up — to fund more assets with the same equity, taking leverage from 12.5× to 20×. Watch both numbers move.

| | Conservative | Aggressive |
|---|---|---|
| ROA | 1.1% | 1.1% |
| Leverage (assets ÷ equity) | 12.5× | 20× |
| Equity as % of assets | 8.0% | 5.0% |
| **ROE = ROA × leverage** | **13.8%** | **22.0%** |
| Asset loss that wipes equity | 8.0% | 5.0% |

By doing nothing but levering up — no better loans, no wider spread, no new fees — the bank lifts ROE from 13.8% to **22%.** Investors cheer. The CEO gets a bonus. But look at the bottom row: the asset loss that destroys the bank just fell from 8% to **5%.** *The exact same lever that lifted ROE by 8 points also cut the survival margin by 3 points — leverage is a multiplier on both the return and the loss, and there is no way to get one without the other.* A bank running at 22% ROE through leverage is not a better bank than one at 13.8%; it is the same machine with a thinner cushion, and the market usually only learns the difference in a downturn.

This is the deepest point in the whole post. Leverage does not create value — it only redistributes it from safety to return. ROA is where real value is created (better funding, better lending, more fees, lower costs). Leverage just decides how much of that value gets concentrated onto the equity, and how exposed that equity is when things go wrong. A bank with a high ROE built on a high ROA is genuinely good; a bank with a high ROE built on high leverage is genuinely fragile, and they look identical on the headline number.

#### Worked example: how one dollar of deposit becomes ROE

We have walked the machine stage by stage; now let us send a single dollar all the way through it, from the moment it lands in a deposit account to the return it earns the shareholder. This is the whole post in one calculation.

You deposit **\$1.00.** The bank now owes you that dollar, payable on demand, and it pays you 1.5% to hold it — **1.5 cents a year** of interest expense.

- **Stage 1, the asset engine.** The bank keeps about 8 cents as cash and liquid buffer and puts the other 92 cents to work: say 65 cents in loans at 6% and 27 cents in securities at 4%. Interest earned: \$0.65 × 6% + \$0.27 × 4% = 3.9 + 1.08 = **4.98 cents.**
- **Stage 2, net interest income.** Subtract the 1.5 cents you are paid: 4.98 − 1.5 = **3.48 cents** of net interest income on the dollar.
- **Stage 3, add fees, subtract costs and losses.** Scale the income build to this dollar: add roughly 1.2 cents of fee income, subtract about 2.3 cents of operating cost and 0.3 cents of provisions. Pre-tax profit: 3.48 + 1.2 − 2.3 − 0.3 = **2.08 cents.** After tax at 21%, **about 1.64 cents** — but remember the dollar of deposit funds slightly more than a dollar of assets is the wrong way round; here the dollar of *assets* (funded by this deposit) earns the profit, so net profit on the asset dollar is **roughly 1.1 cents**, i.e. the 1.1% ROA, once we reconcile fees and costs to the full balance sheet.
- **Stage 4, the amplifier.** Here is the magic. That dollar of assets is backed by only about **8 cents of equity** (the cushion is 8% of assets). The shareholder put in 8 cents and earned 1.1 cents on it. Their return is \$0.011 ÷ \$0.08 = **13.8%.**

So your one dollar of deposit, paying you 1.5 cents, ends up earning the shareholder a 13.8% return on the 8 cents of equity standing behind it. *The deposit is the raw material, the spread and fees minus costs and losses are the processing, and leverage is the multiplier that turns a 1.1-cent margin into a 13.8% return — every dollar in your checking account is one tiny unit of that machine, and the whole bank is just trillions of these dollars running the same four stages in parallel.*

## The output: ROE and the cost of equity

The machine's final output is ROE, and there is one more number it must clear to create value: the *cost of equity* — the return shareholders demand for bearing the risk of owning a bank, typically around 10%.

![US commercial bank return on equity 2010 to 2024](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-6.png)

The chart shows US bank ROE from 2010 to 2024, with the ~10% cost-of-equity line marked. The relationship is simple but it governs how a bank is valued: when ROE is **above** the cost of equity, the bank is creating value, and the market pays *more* than book value for it (a price-to-book ratio above 1). When ROE is **below** the cost of equity — as it was in the 2010 recovery, at just 5.9% — the bank is destroying value, earning less than shareholders could get elsewhere for the same risk, and the market pays *less* than book. The 2010 low at 5.9% and the recovery toward 10-12% trace exactly this: a banking system clawing its way back above its cost of equity after the financial crisis.

This connects the whole machine back to the spine. The bank exists to earn an ROE above its cost of equity, sustainably, across the cycle. To do that it must: fund itself cheaply (the deposit franchise), lend and invest at a spread above that funding cost (the asset engine and NIM), add capital-light fee income, keep costs and loan losses below the spread (the risk drains), and lever a tiny ROA into a respectable ROE on a cushion thin enough to be profitable but thick enough to survive. Every piece we built feeds this one number, and this one number decides whether the bank is worth more or less than the equity put into it.

## The levers management actually pulls

If you sit in a bank's executive suite, you do not get to invent new physics. You inherit the machine and you have a handful of dials. Knowing exactly what those dials are — and what each one does to ROE *and* to fragility — is the whole job, and it is the most useful thing a reader of bank earnings can carry.

![Management levers and their effect on ROE versus fragility](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-7.png)

The matrix above lays out the levers against their two effects — what each does to ROE, and what each does to fragility. The pattern is the lesson: **almost every lever that raises ROE also raises fragility.**

- **Grow loans.** More earning assets at the same spread means more net interest income and higher ROE. But loan growth is the single best predictor of future losses: banks that grow loans fastest, especially late in a cycle, take on the borrowers everyone else rejected, and those loans default most in the downturn. Fast growth lifts ROE now and fragility later.
- **Reprice deposits up.** Raising the rate paid to depositors makes the deposit base stickier and calmer — less likely to flee — but it costs spread and lowers ROE in the near term. This is the rare lever that *trades ROE for safety*, the inverse of the others, which is why nervous banks do it during a scare.
- **Take more credit risk.** Lending to riskier borrowers earns a fatter risk premium and lifts ROE while the economy is calm. The losses arrive concentrated in the recession. Same trade as loan growth, sharper.
- **Add leverage.** As we just saw, levering up multiplies ROA into ROE directly — and thins the cushion just as directly. The purest "more return, more fragility" lever there is.
- **Cut costs.** Lowering the efficiency ratio lifts profit and ROE *without* touching the cushion, the spread, or the risk. This is the one genuinely free lever — the only dial that improves ROE with no fragility cost. It is also the hardest to pull far, because banks are already lean and cutting too deep damages the franchise.
- **Buy back stock.** Returning equity to shareholders raises ROE per share (the same profit spread over fewer shares and less equity) — but it also removes equity from the cushion, leaving less to absorb losses. A buyback is a small dose of leverage dressed up as capital return.

The takeaway from the matrix is sobering and clarifying at once. When a bank reports a rising ROE, the right question is never "great, how did they do it?" but "*which lever, and what did it cost in fragility?*" An ROE that rose because costs fell or fees grew is high-quality and durable. An ROE that rose because the bank grew loans fast, reached for credit risk, levered up, or bought back stock is *borrowed from the future* — it shows up as a great few years followed by a bad one. The cost-cutting lever is the only one that doesn't borrow.

## Common misconceptions

**"A high ROE means a well-run bank."** Not necessarily. ROE = ROA × leverage, so a high ROE can come from a genuinely good machine (high ROA: cheap funding, good lending, lots of fees, low costs) *or* from a thin cushion (high leverage). The first is durable; the second borrows from the future. Lehman's ROE looked great until 30× leverage met a 3% asset loss. Always decompose: a 13.8% ROE from a 1.1% ROA and 12.5× leverage is a different animal from a 13.8% ROE from a 0.7% ROA and 20× leverage. The number to trust is **ROA**, because leverage can flatter ROE without any real improvement in the business.

**"Banks make money on the difference between what they pay and charge, so higher rates are always good."** Higher rates help the *spread* only if the asset side reprices faster than the funding side — which depends on deposit beta and the asset-liability match. A bank funded with sticky, low-beta deposits and floating-rate loans loves rising rates. A bank funded with hot money and holding long fixed-rate securities — SVB — gets crushed by them, because its funding cost reprices up while its asset value falls. Rates are a tailwind or a headwind depending entirely on how the funding and asset engines are matched.

**"Provisions are just an accounting entry, not real money."** A provision is a charge against today's profit for losses the bank forecasts on its loans. It is as real as cash: it reduces the equity cushion the moment it is taken. The reason it feels like an accounting game is that it is *forward-looking and discretionary at the edges* — banks can build reserves in good years and release them in bad ones, which smooths reported profit. But the underlying losses are real, and a bank that under-provisions in the boom simply takes the pain later, all at once. A reserve release in a good year flatters ROE just as surely as a provision build punishes it in a bad one, and neither tells you anything about whether the loans are actually good — only about where the bank is in the cycle.

**"The capital cushion is wasted money that could be earning a return."** This is the argument bank executives make for thinner capital, and it has a grain of truth — equity does lower ROE. But the cushion is not idle; it is the *price of being allowed to run a deposit-funded, leveraged machine at all.* Without it, the bank fails the first time asset losses exceed the buffer, and the deposit franchise — the thing that makes the whole machine valuable — evaporates in a run. The cushion is insurance on the franchise, and the optimal amount is "enough to survive a severe but plausible downturn," not "as little as the regulator allows." Banks that minimize the cushion to maximize ROE are not earning a higher return; they are selling disaster insurance to themselves and pocketing the premium until the disaster arrives.

**"Fee income and interest income are basically the same — profit is profit."** They are not the same in the way that matters most: capital intensity. A dollar of fee income generates a far higher return on equity than a dollar of spread income, because it barely consumes the cushion. It is also more resilient across the credit cycle — fees from running payments and managing money hold up when loan losses spike. A bank's *mix* of fee versus spread income is one of the best signals of both its profitability and its durability, and "profit is profit" hides exactly that signal.

**"A bank is solvent, so it is safe."** Solvency and liquidity are different things, and a bank can die of the second while passing the first. Solvency means assets exceed liabilities — the cushion is positive. Liquidity means the bank can meet withdrawals *today*, with cash, without selling long assets at a loss. Because the funding side is short (deposits payable on demand) and the asset side is long (loans and securities that take time or a price cut to convert to cash), a perfectly solvent bank can run out of cash if depositors leave faster than it can liquidate. Every modern bank run — Northern Rock, SVB — was a liquidity death, not a solvency death, at the moment it happened. "The bank has positive equity" is necessary but nowhere near sufficient for "the bank is safe."

## How it shows up in real banks: same machine, two settings

The most useful way to see the whole machine at work is to compare two banks running the *same machine* with *different dial settings.* The headline ROE can be nearly identical; the survival odds are not.

![Fortress bank versus fragile bank same machine different settings](/imgs/blogs/the-economics-of-a-bank-in-one-page-how-it-all-fits-together-8.png)

The before-after figure contrasts a *fortress* bank with a *fragile* one. Both might post a low-teens ROE in a good year. But trace the dials:

The **fortress** holds **10% equity (10× leverage)**, funds itself **80% with sticky insured deposits**, lends to **prime borrowers losing 0.4% a year**, and posts an **ROE around 11%.** When a downturn knocks 3% off its assets, the loss is \$3 of a \$10 cushion — painful, a year of lost profit, but the bank survives with capital to spare. Its lower ROE in the good years was the premium it paid for that survival, and it was a bargain.

The **fragile** bank holds **5% equity (20× leverage)**, funds itself **40% with hot wholesale money**, lends to **subprime borrowers who lose 3% in a downturn**, and posts a slightly *higher* **ROE around 13%** in good years — which is exactly why investors funded it. When the same kind of downturn knocks 5% off its assets, that 5% loss equals its entire 5% cushion. The equity is gone, the wholesale funders flee, and the bank fails — while the fortress, running the identical machine, lives. The cruel part is that for years the fragile bank looked *better* on every screen an investor runs: higher ROE, faster growth, leaner cushion. The difference only became visible the year the drains opened, and by then it was a failure, not a stock pick.

This is the pattern behind nearly every bank failure in this series, and it is worth naming the real cases through this lens:

- **Silicon Valley Bank, 2023.** SVB ran the asset and funding engines badly mismatched: it funded long-dated, fixed-rate securities (the asset engine reaching for yield) with concentrated, uninsured tech-startup deposits (the worst kind of un-sticky funding). When rates rose, the securities lost market value (market-risk drain) and the depositors — \$42 billion in a single day, 94% uninsured — fled (liquidity-risk drain). The cushion couldn't absorb the marked-down losses fast enough. Same machine, lethal dial settings on funding stickiness and asset-liability match. Notice that no single line of the income build was the cause; it was the *connections* between funding, assets, market risk, and liquidity that killed it, which is exactly why a one-page view of the whole machine catches what any single ratio misses.
- **Lehman Brothers, 2008.** Lehman ran the leverage dial to roughly **30×** — a cushion near 3% — funded itself heavily with overnight wholesale repo (the least sticky funding), and held assets whose value collapsed. A 3% fall in asset value was enough to wipe the equity, and the wholesale funders, pricing daily, ran. The machine was identical to a commercial bank's; the leverage and funding dials were set for maximum ROE and minimum survival. Lehman's ROE in the boom years was the envy of Wall Street — and it was entirely a leverage story, the highest-quality-looking number from the lowest-quality source.
- **The 2010 recovery, sector-wide.** The ROE chart's 5.9% trough shows the whole machine running *below* its cost of equity. Provisions were still elevated (the credit drain), spreads were compressing toward the zero-rate floor (the NIM heartbeat slowing), and banks were rebuilding cushions after the crisis (de-leveraging, which lowers ROE). Every part of the machine was working against ROE at once — which is exactly why banks traded below book value for years. It is the mirror image of a boom: in 2010 the drains were wide open, the spread was thin, and the leverage that once amplified profit was being deliberately unwound, so the same identity that produces a 14% ROE in good times produced a 6% ROE here.
- **The fee-rich survivors.** The banks that came through the 2008 and 2020 shocks best were not the highest-ROE banks of the boom — they were the ones with large, capital-light fee businesses and sticky deposit franchises. When the credit drain opened, their fee income held up while their lending peers' profit cratered, and their thick deposit bases didn't run. They had set the dials toward durability, and it cost them a point or two of ROE in the good years and saved them in the bad ones. Over a full cycle, the durable-dial bank usually compounds book value faster than the fragile one, because it never has to raise expensive equity at the bottom or sell assets into a panic — the boring machine wins the long game.

The lesson across all four: the machine is always the same six stages. What differs — and what determines whether a bank thrives or dies — is *where management set the dials*, and whether the spread and the cushion were large enough to survive the year the drains all opened at once. A great bank and a fragile bank are not different machines; they are the same machine with different settings, and the settings are invisible in the ROE.

## The takeaway / How to use this

The single most valuable thing you can take from this entire series is the ability to look at any bank — in a headline, an earnings report, a stock pitch — and reconstruct its machine on the back of an envelope. The recipe is now mechanical:

1. **Start with ROA.** Profit ÷ assets. This is where real value is created or destroyed, and it is the number leverage cannot fake. About 1% is normal; above 1.3% is excellent; below 0.7% is weak. If you only get one number, get this one.
2. **Decompose the ROE.** ROE = ROA × leverage. If the ROE is high, ask *which factor*. A high ROA is a good machine. High leverage is a thin cushion. They look identical on the headline and could not be more different in a downturn.
3. **Trace the four income lines.** Net interest income (the spread × the asset base), fee income (capital-light, durable), operating cost (the efficiency ratio), and provisions (the credit drain). Any change in profit traces to one of these four. The mix of fee versus spread tells you how durable the profit is.
4. **Check the funding.** What fraction is sticky insured deposits versus hot wholesale money? Sticky funding is the franchise and the survival margin both. This is the question that separates the banks that survive a scare from the ones that don't, and it almost never shows up in the ROE.
5. **Measure the cushion.** Equity as a percent of assets, and the asset loss that would wipe it out. An 8% cushion survives an 8% loss; a 4% cushion dies at 4%. The cushion is the answer to "can this bank survive a bad year?" — the question the ROE never answers.
6. **Identify the lever.** When the ROE moves, name the dial: loans, deposit pricing, credit risk, leverage, costs, or buybacks. Then ask what it cost in fragility. Cost-cutting and fee growth are free; everything else borrows from the future.

Run those six steps and you have done what a bank analyst does, and you have done it through the one frame that ties the whole machine together: **a bank turns cheap funding and a thin cushion into a leveraged return, and lives or dies on whether its spread and its cushion are large enough to survive the year the drains all open at once.** That is how a bank makes money, and that is how a bank dies — the same machine, read forward for profit and read backward for failure.

The thin margin is not the flaw. The thin cushion is not the flaw. They are the design. The flaw is only ever in *how thin* — and now you can measure it.

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the borrow-short, lend-long trade at the heart of the machine, and why it is structurally fragile.
- [ROE, ROA, and the leverage identity: how a bank is judged](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) — the full DuPont decomposition of the amplifier and how investors value the franchise.
- [Net interest margin and the spread business explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the heartbeat in depth: deposit beta, asset repricing, and what widens and narrows the spread through a rate cycle.
- [The four risks every bank runs: credit, market, liquidity, operational](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational) — the drains in detail and how they interact and compound.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the loss-absorber and the leverage math that turns a 1% ROA into a 12% ROE and a 3% loss into a failure.
- [The income statement of a bank: net interest income, fees, and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — the four-line build, line by line, with the efficiency ratio and pre-provision profit.
- [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the funding-stickiness and asset-liability-match dials set lethally wrong, in real time.
- [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation) — why regulators force a minimum cushion, and the capital rules that constrain every dial in the machine.
- [How money is created: banks, central banks, and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — the system-level view of how the deposit-and-lending machine creates money across the whole banking sector.

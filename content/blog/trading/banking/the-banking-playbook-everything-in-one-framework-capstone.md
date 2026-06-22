---
title: "The Banking Playbook: Everything in One Framework"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The capstone of the series: one durable mental model that ties the balance sheet, every operation, risk, capital, regulation, the great failures, and the modern frontier into a single way to read any bank."
tags: ["banking", "bank-balance-sheet", "net-interest-margin", "bank-capital", "liquidity-risk", "bank-failures", "bank-valuation", "framework", "capstone"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short, lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. Everything else in banking is a detail of that one fragile trade.
>
> - **The whole machine is one chain.** Funding (cheap, sticky deposits) → assets (loans and securities) → the spread (a net interest margin around 3%) → a thin equity cushion (about 8% of assets, so roughly 12x leverage). Every function, control, and failure plugs into a link in this chain.
> - **There are only four risks and four ways to die.** Credit, market, liquidity, and operational risk. A bank dies when losses eat the cushion (solvency) or when funding flees before assets can be sold (liquidity). Capital answers the first; liquidity buffers and trust answer the second.
> - **Reading any bank is four questions.** How does it fund itself? How thick is the cushion versus what regulators demand? How runnable is it, right now? And what is the franchise worth — is high return real, or just leverage in disguise?
> - **The one number to remember:** an 8% equity cushion means a bank can lose roughly 8% of its assets before it is insolvent — and in 2023, Silicon Valley Bank lost depositor confidence and watched \$42 billion try to leave in a single day, against a cushion that small.

In the early hours of March 10, 2023, the staff of Silicon Valley Bank watched a number climb on a screen. The night before, depositors had tried to pull \$42 billion. By morning, another \$100 billion was queued to leave. The bank held \$209 billion in assets and looked, on paper, perfectly solvent a week earlier. It was dead within thirty-six hours. No fraud, no exotic derivative, no rogue trader. Just the oldest trade in finance — borrow short, lend long — running in reverse at the speed of a smartphone.

Over the last sixty-one posts, this series has taken that one trade apart, piece by piece. We read the balance sheet. We followed a deposit into a loan and a loan into a security. We watched payments cross the world, traced how treasury manages the gap between assets and funding, counted the capital regulators demand, and walked through the failures and scandals that teach how a bank actually breaks. This capstone does the opposite job. It puts every piece back together into a single framework — one mental model you can carry into any bank's annual report, any headline, any crisis, and use to ask the right questions in the right order.

![The whole bank in one frame showing funding and assets feeding the spread, then risk and capital, then the bank lives or dies](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-1.png)

The figure above is the entire series compressed to one frame. Funding and assets feed the spread; the spread, after losses, lands on a thin equity cushion; and that cushion — together with the trust of depositors — decides whether the machine lives or dies. Every track in this series is a zoom into one of those boxes. By the end of this post, you should be able to look at any bank and place everything you read about it onto this diagram.

## Foundations: the one machine, restated from zero

Let us rebuild the whole thing from nothing, assuming you have read none of the earlier posts.

Start with the everyday version. Imagine a corner shop whose owner notices that the cash in the till mostly sits there. Customers pay in, customers take change out, but on any given day the till never fully empties. So the owner starts lending the idle cash to a neighbour who needs it for a year, charging interest. As long as not every customer wants their change back on the same day, this works — and the owner pockets the difference between what the neighbour pays and what (if anything) the owner pays the customers for leaving cash in the till.

That is a bank. A bank takes money that is *repayable on demand* (your deposit, which you can withdraw whenever you like) and turns it into money that is *locked up for years* (a mortgage, a business loan). This trick has a name: **maturity transformation** — turning short-term money into long-term money. It is genuinely useful: it is how a 30-year mortgage can exist even though no saver wants to lock up cash for 30 years. But it is also structurally fragile, because the short money can leave faster than the long money comes back. We unpack this opening idea in [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).

Three more terms, defined once and used everywhere:

- A **liability** is something the bank owes. Your deposit is the bank's liability — the bank owes it to you. (This is why your "savings" are, on the bank's books, a debt.)
- An **asset** is something the bank owns or is owed. The loan it made is the bank's asset — the borrower owes the bank.
- **Equity** (also called *capital*) is the difference: assets minus liabilities. It is the owners' stake, and crucially, it is the cushion that absorbs losses. If assets fall in value, equity shrinks first; only when equity is exhausted do the depositors start to lose money.

A *basis point* is one hundredth of one percent (0.01%). A bank's **net interest margin** (NIM) — the spread between what it earns on assets and what it pays on funding, as a percentage of its assets — is typically around 300 basis points, or 3%. Hold that number; it is the engine.

It is worth pausing on *why* a bank is fragile rather than treating fragility as a flaw to be engineered away. The fragility is not a bug; it is the same mechanism as the usefulness, viewed from the other side. The economy *wants* maturity transformation — it wants 30-year mortgages and 10-year business loans funded by savers who keep their money available on demand. Only an institution that bridges that gap can provide it, and bridging the gap *necessarily* means owing money that can be called faster than the money you are owed comes back. You cannot have the benefit without the exposure. That is why the whole apparatus of capital rules, liquidity buffers, deposit insurance, and a lender of last resort exists: not to remove the fragility, which is impossible, but to *manage* it — to make the panic equilibrium rare and survivable. Keep this in mind whenever someone proposes a banking innovation that promises the upside with none of the risk: if it still does maturity transformation, it still has the fragility, no matter what it is called.

Now the spine of the whole series, in one sentence:

> A bank borrows short (deposits), lends long (loans), earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive.

Every later track is a detailed answer to a question this sentence raises. *Where does the cheap short money come from?* — the deposit franchise (Track B). *How does the long lending actually happen and how is it priced?* — lending and credit (Track B). *How does money physically move so the deposits exist at all?* — payments (Track C). *Who manages the gap between short funding and long assets?* — treasury (Track D). *What can go wrong, and how much cushion is required?* — risk, capital, regulation (Track E). *What does it look like when the trade fails?* — the great failures and scandals (Tracks F and G). *How is the machine changing?* — the modern frontier (Track H). *And how do you read all of this in a real bank?* — the analyst's playbook (Track I, where this capstone sits).

## The balance sheet: the anchor image for everything

If you remember one picture from this entire series, make it the balance sheet — because every other concept is a movement on it.

![Bank balance sheet with assets on the left and funding plus an eight percent equity cushion on the right](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-2.png)

A bank's balance sheet is two columns of exactly equal height. On the left, the **assets**: roughly 52% loans, 22% securities, 13% cash and reserves, and 13% trading and other. On the right, the **funding plus equity**: roughly 71% deposits, 10% wholesale and repo borrowing, 11% long-term debt and other, and — the sliver that decides everything — about 8% equity. The two columns must balance, because equity is *defined* as the plug that makes them balance.

This is the inverse of your household balance sheet, and noticing that is half the insight. Your deposit is your asset and the bank's liability. The bank's loan to you is the bank's asset and your liability. A bank is, structurally, the mirror image of its customers — which is exactly why it can transform maturity, and exactly why it is fragile. We build this from scratch in [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity).

The thin right-hand sliver is the most important 8% in finance. It means that for every \$100 of assets, the bank has put up only about \$8 of its own money; the other \$92 belongs to depositors and creditors. That ratio — \$100 of assets on \$8 of equity — is **leverage of about 12.5 times** (100 ÷ 8). Leverage is a magnifier: it multiplies returns on the way up and losses on the way down. Read the full mechanics in [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).

#### Worked example: how a 4% loss becomes a 50% wipe

Take a small bank: \$100 of assets, \$92 of deposits and debt, \$8 of equity. Now its loan book sours and 4% of the assets — \$4 — turn out to be worthless.

Assets fall from \$100 to \$96. The depositors and creditors are still owed \$92; that does not change. So equity, the plug, falls from \$8 to \$96 − \$92 = \$4. A **4% fall in assets just cut equity in half** — from \$8 to \$4. That is the leverage multiplier (about 12.5x) working in reverse: a 4% asset loss × 12.5 ≈ a 50% equity loss.

Push it further. If 8% of assets — \$8 — go bad, assets fall to \$92, equity falls to zero, and the bank is exactly insolvent: it owes its funders precisely what it owns. Anything worse, and depositors are looking at a loss. The intuition: *the equity cushion measures, almost literally, the percentage of its assets a bank can lose before it fails.* For a typical commercial bank that number is around 8%. That is not much margin for a portfolio of thousands of loans in a recession — which is the entire reason regulation exists.

## The spread engine: how the machine earns its living

A bank's profit, stripped down, has two engines. The bigger one is **net interest income**: interest earned on assets minus interest paid on funding. The smaller one is **fee income**: payments, cards, advisory, asset management. Out of the total, the bank pays its staff and systems (the *efficiency ratio* measures costs as a share of revenue), sets aside money for loans it expects to go bad (*provisions*), and what is left is profit. We lay out the full income statement in [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).

The net interest margin is the heartbeat. Watch what it does through a rate cycle.

![US bank net interest margin from 2010 to 2024 falling to a 2021 trough then rising after rate hikes](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-3.png)

Through the zero-rate years after 2010, the margin compressed — falling from 3.76% in 2010 to a trough of 2.56% in 2021, because the bank could not pay less than zero on deposits but the yield on its assets kept sliding. Then the Federal Reserve hiked hard in 2022–2023, asset yields jumped, and because deposit rates lag (the *deposit beta* — the share of a rate rise passed to depositors — is well below one), the margin sprang back to around 3.2% by 2024. This is the spread business breathing. Read the deep version in [net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained).

#### Worked example: the full spread-to-ROE chain

This is the chain the whole series builds toward, so we walk every step. Take a clean, mid-sized bank with \$100 billion in assets.

- **Step 1 — the spread.** It earns a 3.2% net interest margin on its assets: 3.2% × \$100bn = \$3.2bn of net interest income.
- **Step 2 — add fees, subtract costs.** Add \$1.0bn in fee income, for \$4.2bn of revenue. It runs at a 60% efficiency ratio, so costs are 0.60 × \$4.2bn = \$2.52bn. That leaves \$1.68bn of *pre-provision operating profit* — what the bank earns before loan losses.
- **Step 3 — subtract provisions.** It sets aside \$0.4bn for expected loan losses. Pre-tax profit is \$1.68bn − \$0.4bn = \$1.28bn.
- **Step 4 — tax to net income.** At a 22% tax rate, net income is \$1.28bn × 0.78 ≈ \$1.0bn.
- **Step 5 — ROA.** Return on assets = net income ÷ assets = \$1.0bn ÷ \$100bn = **1.0%**. This is the famous "1% ROA" rule of thumb for a healthy bank.
- **Step 6 — ROE via leverage.** With 8% equity, the bank has \$8bn of equity. Return on equity = net income ÷ equity = \$1.0bn ÷ \$8bn = **12.5%**. Equivalently, ROE = ROA × leverage = 1.0% × 12.5 = 12.5%.

The intuition that ties this series together: *a bank turns a tiny 1% return on its assets into a respectable 12.5% return on its equity purely by being levered 12.5 times.* The spread is thin; leverage makes it matter. And that is exactly why the same leverage makes losses lethal — the magnifier runs both ways.

![Return on assets and return on equity for US banks showing about one percent ROA becomes about twelve percent ROE](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-9.png)

The chart above is the rule of thumb in real industry data: ROA hovers around 1% and ROE around 10–12%, and the gap between them is leverage. When ROA collapsed to 0.72% in the 2020 shock, ROE collapsed with it to 6.65% — same leverage, smaller spread. We formalize this in [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).

## The one trade, drawn as a pipeline

Before we walk the operational tracks, here is the spine itself as a process — the chain that every later function attaches to.

![The bank's core trade as a pipeline from borrow short to lend long to earn the spread to absorb losses to keep trust](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-4.png)

Read it left to right. **Borrow short** — deposits, repayable on demand. **Lend long** — loans that run for years. **Earn the spread** — the yield on the long assets minus the cost of the short funding. **Absorb losses** — on an 8% equity cushion, about 12x levered. **Keep trust** — or face a run. Every operational track in this series is a deep zoom into one of those links. The deposit franchise is the first box. Lending and credit are the second. Treasury manages the relationship between the first two. Risk and capital govern the fourth. And the failures all happen when the last box breaks.

## The operations, track by track

### Deposits: the franchise (Track B)

The first box in the pipeline is the most undervalued. The cheap, sticky deposit base *is* the franchise — it is the reason a bank is worth more than a finance company that has to borrow its money in the market. A *current account* (or checking account) pays little or no interest and can be withdrawn anytime; a *term deposit* locks money up for a set period at a higher rate. The share of cheap current-and-savings money in the funding mix is the **CASA ratio**, and a high one is the difference between a great bank and a mediocre one, because it holds the cost side of the spread down. Deposit *stickiness* — how reluctant depositors are to leave — is the hidden asset that doesn't appear on the balance sheet. We make the case in [retail deposits, the funding base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).

Why does this matter so much to the framework? Because the cost of funding is half the spread, and the deposit franchise is the only part of a bank's cost base that is genuinely hard to copy. Anyone with capital can buy a loan book or hire bond traders; almost no one can quickly assemble millions of households who leave their salaries in a current account paying near-zero because switching banks is a chore. That inertia is worth real money. A bank funding itself at 0.5% instead of 4% on \$71 billion of deposits saves 3.5% × \$71bn = about \$2.5 billion a year — which is most of the difference between a great franchise and a struggling one. It is also why the same deposit base is the prize that fintechs, stablecoins, and neobanks are all circling: whoever holds the cheap money holds the spread. And it is why *who* the depositors are matters as much as how many dollars they hold — a base of insured, diversified retail savers is a fortress; a base of large, connected, uninsured corporates is a fuse.

### Lending and credit: the core asset (Track B)

The second box is where the money is made and lost. A loan moves through a lifecycle: origination, underwriting, approval, disbursement, monitoring, and — if it goes wrong — collection. The discipline that decides who gets a loan is *credit analysis*: the classic **five Cs** (character, capacity, capital, collateral, conditions), formalized today into credit scores, debt-service-coverage ratios, and loan-to-value limits. The end-to-end mechanics are in [the lending business](/blog/trading/banking/the-lending-business-how-a-bank-underwrites-a-loan-end-to-end), and the decision discipline in [credit analysis and the five Cs](/blog/trading/banking/credit-analysis-the-five-cs-and-how-a-loan-gets-approved).

How does a bank price a loan? Not by guessing. It stacks the costs: cost of funds + expected loss + a capital charge + operating cost + a target margin. *Expected loss* is the engine of credit pricing, and it has three factors we will meet again under risk: the probability the borrower defaults (PD), the fraction lost if they do (LGD), and the amount exposed at default (EAD). The pricing build-up is in [loan pricing](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge), and when loans go bad, the [non-performing loan workout](/blog/trading/banking/non-performing-loans-and-the-workout-process) and [provisioning under IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) take over.

#### Worked example: pricing a loan to survive the cycle

A bank lends \$1,000,000 to a mid-rated company for one year. Walk the build-up:

- **Cost of funds.** The bank funds the loan at 3.0%, so \$30,000.
- **Expected loss.** The borrower's one-year default probability (PD) is 1.2%; if it defaults, the loss given default (LGD) is 45% of the \$1,000,000 exposure (EAD). Expected loss = PD × LGD × EAD = 0.012 × 0.45 × \$1,000,000 = \$5,400.
- **Capital charge.** Regulators make the bank hold equity against the loan. Suppose the loan carries an 8% capital requirement on its risk weight, so \$80,000 of equity is tied up, and shareholders want a 12% return on it: 0.12 × \$80,000 = \$9,600.
- **Operating cost.** Servicing the loan costs \$3,000.

Add them: \$30,000 + \$5,400 + \$9,600 + \$3,000 = \$48,000. To break even *and* pay shareholders their required return, the bank must charge at least 4.8% on the loan. To actually profit, it charges more — say 6.0%, earning \$60,000, of which \$12,000 is true economic profit after the capital charge. The intuition: *a loan's interest rate is not a number plucked from the air; it is the sum of every cost the loan imposes on the machine, including the cost of the equity it forces the bank to hold.* This is why riskier borrowers pay more — their expected-loss term is larger — and why a bank that underprices risk is quietly burning its own cushion.

### Payments: the plumbing that makes deposits possible (Track C)

Deposits do not exist in a vacuum; they exist because money can move. Payments are the unglamorous plumbing under the whole franchise, and they generate fee income and sticky low-cost balances on the side. When you pay someone at another bank, no cash crosses the street — the two banks settle through accounts they hold with each other or with the central bank (*nostro* and *vostro* accounts; correspondent banking). Domestic rails range from real-time gross settlement (RTGS) for big transfers to batch systems (ACH) and card networks; cross-border payments hop through correspondent banks and SWIFT messages. The starting map is [the payments business](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks), and the card economics — the four-party model and the interchange split — are in [the cards business](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split). Trade finance, where a bank's promise de-risks global commerce, lives in [letters of credit and guarantees](/blog/trading/banking/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance).

Two things connect payments back to the spine. First, payments are where a bank's *fee income* — the second profit engine alongside net interest income — is largely earned: a slice of every card swipe (interchange), a fee on every wire, a margin on every cross-border conversion. On a \$100 card purchase, roughly \$1.75 of interchange flows to the card issuer, a few cents to the network, and a markup to the acquiring bank — small per transaction, enormous across billions of swipes. Second, and more subtly, running a corporate's payments captures its *operating deposits* — the cash a company must keep in its bank to make payroll and pay suppliers. Those balances are sticky and cheap precisely because the company cannot easily move them without rewiring its whole payments machinery. So payments quietly feeds box one of the pipeline: the franchise. A bank that owns a corporate's transaction banking owns its deposits, and a bank that owns the deposits owns the cheap side of the spread. This is also why the geopolitics of payments — the ability to cut a country off from the dollar plumbing — is a weapon, a theme the series links out to in [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments).

### Treasury and ALM: the cockpit (Track D)

Here is the function that actually manages the maturity-transformation trade. The treasury, governed by an asset-liability committee (ALCO), watches the *gap* between the bank's long assets and its short funding, manages liquidity, and hedges interest-rate risk. The single most important — and most lethal — concept in the whole series sits here: **interest-rate risk in the banking book** (IRRBB) and the *duration gap*.

*Duration* is the sensitivity of a bond's price to interest rates: a bond with a duration of 5 falls about 5% in price when rates rise 1 percentage point. When a bank funds long, fixed-rate assets (high duration) with short deposits (near-zero duration), it has a large positive duration gap — and a jump in rates slashes the value of its assets while its funding cost rises. That is not a textbook hazard; it is the exact mechanism that broke SVB. The cockpit is described in [bank treasury and asset-liability management](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit), the duration trap in [interest-rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap), and the funding side in [the funding stack](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds). Liquidity — the rules that try to ensure a bank can survive a 30-day stress — is in [liquidity management, LCR and NSFR](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer), which makes the series' sharpest distinction: *liquidity is not solvency, and a solvent bank can still die.*

Treasury also reaches into the overnight money market through *repo* — selling a security today with a promise to buy it back tomorrow, which is really a collateralized loan. Repo is how banks and dealers fund their securities cheaply, and its sudden freezes were central to both 2008 and the 2019 funding scare; the mechanics are in [the repo market](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight). And [securitization](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — packaging loans into securities to sell off — is the lever that lets a bank make loans without holding them all on its own balance sheet, the originate-to-distribute model whose abuse helped detonate 2008.

#### Worked example: the duration-gap hit that opens the trapdoor

This is the treasury risk that broke SVB, in numbers. A bank holds \$80 billion of long-dated bonds with an average duration of 6 years, funded by deposits with effectively zero duration. Its equity cushion is \$8 billion.

Now market rates rise by 2 percentage points (200 basis points), exactly what the Fed delivered in 2022–2023. The change in the bond portfolio's value is approximately −duration × rate change × portfolio: −6 × 0.02 × \$80bn = **−\$9.6 billion**.

That \$9.6 billion loss is *larger than the entire \$8 billion equity cushion.* On a pure mark-to-market basis, this bank is already insolvent — its bonds are worth \$9.6bn less, and only \$8bn of equity stood behind them. Accounting may let the bank classify the bonds as "held to maturity" and not book the loss, so the cushion looks intact on the printed balance sheet. But the economic hole is real, every analyst can compute it from the footnotes, and the moment the bank is forced to *sell* those bonds to meet withdrawals, the paper loss becomes a cash loss and the trapdoor opens. The intuition: *a duration gap is a hidden short position against rising rates, and a positive gap large enough means a rate shock can vaporize the cushion before a single borrower has defaulted.* Credit risk is slow; market risk through the duration gap is fast — and treasury is the function whose entire job is to keep that gap from becoming a death sentence.

## Risk, capital, and regulation: the cushion and the rules around it

Everything above describes how the machine earns. This track describes how it is kept from blowing up. And the elegant thing about banking risk is that there are only four kinds.

![A matrix of the four bank risks credit market liquidity operational and how each one kills the bank](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-5.png)

- **Credit risk** — borrowers default; loan losses eat the cushion.
- **Market risk** — rates or prices move; the value of assets falls, and equity follows.
- **Liquidity risk** — funding flees faster than assets can be sold; a run drains cash before the (solvent) assets can be turned into money.
- **Operational risk** — fraud, cyber, a control failure, a rogue trader; a loss event or, worse, lost trust.

These four, plus the *three lines of defence* that govern them (the business that takes risk, the risk function that limits it, and internal audit that checks both), are the whole risk taxonomy. We lay it out in [the four risks every bank runs](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational). Credit risk gets its own engine in [PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss); market risk and value-at-risk in [market risk, VaR and the trading limits](/blog/trading/banking/market-risk-var-stressed-var-and-the-trading-limits); and operational risk, the catch-all for everything human, in [operational risk, fraud, cyber and loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events).

Against these risks stands capital. Regulators do not just want *some* equity; they want it measured against the *riskiness* of the assets. **Risk-weighted assets** (RWA) re-weight each asset by how dangerous it is — a government bond might get a 0% weight, a corporate loan 100%, so \$100 of government bonds and \$100 of corporate loans demand wildly different amounts of capital. The Basel framework then layers minimum ratios and buffers on top of RWA. The capital stack and the Basel evolution are in [Basel I, II, III](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank), the RWA machinery in [risk-weighted assets and how capital ratios work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work), the forward-looking exams in [stress testing and CCAR](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills), and the safety net that backstops the whole system in [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard). For the system-level view of why these rules exist at all, the series links out to [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation).

#### Worked example: does the capital cushion pass the test?

Take a bank with \$100 billion in total assets, of which \$80 billion are risk-weighted assets (RWA) after applying Basel weights, and \$8.5 billion of common equity tier 1 (CET1) capital — the highest-quality loss absorber.

Its **CET1 ratio** = CET1 ÷ RWA = \$8.5bn ÷ \$80bn = **10.6%**.

Now compare it to what a large systemically important bank is actually required to hold: a 4.5% minimum, plus a 2.5% capital-conservation buffer, plus (say) a 1.5% surcharge for being systemically important. That sums to an effective demand of about **8.5% of RWA**. Our bank's 10.6% clears it with about 2 percentage points to spare — \$1.7bn of buffer above the requirement (2.1% × \$80bn).

The intuition: *a bank's safety is not the raw size of its equity but the size of its equity relative to the riskiness of its assets and the bar regulators set.* The same \$8.5bn would look strong against \$60bn of RWA (14.2%) and dangerously thin against \$120bn (7.1%, below the demand). This is why "how big is the cushion" must always be asked as "how big *relative to RWA and the requirement*" — and why banks fight so hard over how RWA is calculated.

## How banks die: the failures (Track F)

Now the payoff of the whole framework: when you understand the machine, every failure stops being a mystery and becomes a predictable break in one of the links. And failure is not rare. It is recurring.

![US bank failures per year from 2005 to 2025 with a large 2008 to 2012 wave and a 2023 spike](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-6.png)

The chart is a reminder against complacency: 157 US banks failed in 2010 alone, in the long tail of the financial crisis, and the count never stays at zero for long. 2023 brought only five failures by count — but three of them (Silicon Valley Bank, Signature, First Republic) were among the largest in US history by assets.

A bank dies in one of two ways, and both map onto the framework:

- **Insolvency** — losses (credit or market) exhaust the equity cushion. The right-hand sliver of the balance sheet goes to zero. The S&L crisis of the 1980s killed over a thousand thrifts this way, through an interest-rate mismatch; see [the savings and loan crisis](/blog/trading/banking/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures).
- **Illiquidity** — funding flees before the (perhaps still solvent) assets can be sold for cash. This is the classic *run*, and in the digital age it happens in hours, not weeks. The mechanism is dissected in [the anatomy of a bank run](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse).

The run itself deserves a moment, because it is the single most counterintuitive thing about banking. A run is *self-fulfilling*: if you believe others will withdraw, the rational thing is to withdraw first, because the bank pays out in the order people arrive and the latecomers get nothing. That logic holds even for a perfectly healthy bank — which is the unsettling insight of the Diamond–Dybvig model. Two equilibria exist side by side: a calm one where nobody runs and everybody is fine, and a panic one where everybody runs and the bank dies, and the only difference between them is what depositors *believe* the others will do. Deposit insurance exists precisely to delete the panic equilibrium for small savers — if your money is guaranteed, you have no reason to join the queue. But for large uninsured balances the panic equilibrium is alive and well, and in the smartphone era the queue forms in minutes, not days. That is why the 2023 runs were faster than anything in history: the depositors were connected, the balances were uninsured, and the withdrawal was a few taps away.

The modern case studies are simply this framework playing out:

- **Silicon Valley Bank (2023)** is the duration trap plus a digital run. It funded itself with concentrated, 94%-uninsured tech deposits and parked the money in long-dated bonds. When rates rose, those bonds lost value (a duration-gap loss of about \$17 billion of unrealized losses on a thin cushion), depositors noticed, and \$42 billion tried to leave in a day. Market risk created the hole; liquidity risk delivered the death blow. Full autopsy in [Silicon Valley Bank 2023](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).
- **Credit Suisse (2023)** is the slow death of trust. A decade of scandals eroded confidence until clients pulled CHF 110 billion in a single quarter, and the bank was force-married to UBS — wiping out CHF 16 billion of AT1 bondholders along the way. Operational and conduct failures, compounding into a liquidity run. See [Credit Suisse 2023](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout).
- **Lehman Brothers (2008)** is leverage and wholesale funding. Levered about 30.7 times, funded by short-term repo it had to roll every day, Lehman could not survive the moment lenders stopped rolling. Detailed in [Lehman Brothers 2008](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank), with [Northern Rock](/blog/trading/banking/northern-rock-2007-the-first-bank-run-of-the-modern-era), [Continental Illinois](/blog/trading/banking/continental-illinois-1984-and-the-birth-of-too-big-to-fail), and [Washington Mutual](/blog/trading/banking/washington-mutual-and-the-2008-mortgage-bank-failures) rounding out the wholesale-funding-run family. The system-level take on the 2023 episode is in [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

#### Worked example: the survival test (solvency, then liquidity)

This is the test the framework is built to run. Take a bank under stress and ask the two questions in order.

**Question 1 — solvency.** The bank has \$100bn of assets and \$8bn of equity. A recession hits and losses arrive: \$3bn of credit losses on the loan book and \$2bn of mark-to-market losses on its bond portfolio (a duration-gap hit). Total losses \$5bn. Equity falls from \$8bn to \$3bn. The bank is *still solvent* — it has \$95bn of assets against \$92bn owed, and \$3bn of equity left. On paper, it survives.

**Question 2 — liquidity.** But the losses are now public, and 20% of its \$71bn of deposits — \$14.2bn — try to leave in a week. The bank holds \$13bn of cash and reserves and can quickly sell \$8bn of high-quality securities, for \$21bn of available liquidity. \$21bn covers the \$14.2bn outflow — *this* bank survives the run, with \$6.8bn to spare.

Now change one number. Suppose those same deposits were 94% uninsured and concentrated among a few thousand panicking depositors who all leave at once: not \$14.2bn but \$42bn flees in a day. The \$21bn of liquidity is overwhelmed, the bank is forced to dump its remaining bonds at fire-sale prices (crystallizing more losses, which now *do* threaten solvency), and it fails — even though it started the week solvent. That is SVB in two paragraphs. The intuition: *solvency and liquidity are two separate tests, and a bank must pass both; a solvent bank with runnable funding can die before lunch.* Always run the test in that order, and always look hard at *who* the depositors are, not just how many dollars they hold.

## Conduct and governance: when the people break (Track G)

Not every failure is a balance-sheet failure. Some are failures of behaviour — the operational-risk box in the matrix, scaled up to existential. These matter to the framework because they destroy the one thing the machine cannot run without: *trust*.

- **Wells Fargo (2016 onward)** turned sales incentives into a machine for opening millions of fake accounts, drawing roughly \$4.9 billion in fines across 2016, 2020, and 2022, and lasting reputational damage. The lesson is that incentives are a risk control; set them wrong and the staff will manufacture the disaster for you. See [the Wells Fargo fake-accounts scandal](/blog/trading/banking/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong).
- **Barings (1995)** shows a single rogue trader, Nick Leeson, destroying a 233-year-old bank with about £827 million (\$1.3 billion) of hidden losses — because the same person controlled both the trading and the settlement of his own trades. Segregation of duties is not bureaucracy; it is structural defence. See [Barings and Nick Leeson](/blog/trading/banking/barings-and-nick-leeson-1995-how-one-trader-broke-a-300-year-old-bank).
- The **LIBOR scandal** (rigging the benchmark behind trillions of dollars of contracts, about \$9 billion in industry fines), the **AML failures** at HSBC and Danske, the **1MDB** heist that ensnared Goldman, and the **Wirecard** fraud (€1.9 billion of cash that never existed) complete the conduct gallery. Each is the operational-risk corner of the matrix, turned lethal. See [the LIBOR scandal](/blog/trading/banking/the-libor-scandal-rigging-the-worlds-most-important-number), [money laundering and AML failures](/blog/trading/banking/money-laundering-and-the-aml-failures-hsbc-danske-and-the-compliance-machine), [the 1MDB scandal](/blog/trading/banking/the-1mdb-scandal-and-an-investment-banks-role-in-a-heist), and [Wirecard 2020](/blog/trading/banking/wirecard-2020-the-collapse-of-a-fintech-darling-and-the-missing-billions).

The unifying thesis of Track G: *a bank's balance sheet can be perfect and the bank can still die, because the asset that doesn't appear on the balance sheet — trust — is the one that funds everything.* Credit Suisse had capital; it ran out of trust.

There is a second, quieter lesson that ties conduct back to the spine. Each of these scandals was, at root, a *control* failure — and controls are not red tape, they are the immune system that protects the cushion. Barings died because one man controlled both his trades and their settlement, so there was no second pair of eyes; that is why *segregation of duties* is a structural rule, not a suggestion. Wells Fargo's fake accounts grew because the incentive system rewarded volume without a control that checked whether the accounts were real; incentives, badly set, become a risk *factory*. AML failures persisted because screening systems were underfunded relative to the flows passing through them. The framework's verdict on conduct is therefore the same as its verdict on capital: the cheap thing to skimp on in a boom — a control, a buffer, a check — is exactly the thing whose absence is fatal in the bust. When you read a bank, the conduct record is not gossip; it is direct evidence about whether the immune system works.

## The modern frontier: the same machine, new pipes (Track H)

The last operational track asks whether technology changes the spine. The answer is consistent: the pipes change, the trade does not.

Neobanks and challenger banks rebuilt the front end but discovered the hard truth that the deposit franchise — cheap, sticky money — is what makes a bank profitable, and most digital banks struggle to win it ([digital banking and the neobank model](/blog/trading/banking/digital-banking-and-the-neobank-business-model)). Open banking and banking-as-a-service unbundle and rebundle the functions ([open banking and embedded finance](/blog/trading/banking/open-banking-apis-banking-as-a-service-and-embedded-finance)). The real strategic question is whether stablecoins and central bank digital currencies will *disintermediate the deposit base itself* — pulling the cheap funding out of banks and into narrow-bank-like instruments, attacking the first box of the pipeline directly ([stablecoins, CBDCs and the threat to bank deposits](/blog/trading/banking/stablecoins-cbdcs-and-the-threat-to-bank-deposits)). For the system-level views the series links out to [central bank digital currencies](/blog/trading/finance/central-bank-digital-currencies-cbdc) and the crypto-side [stablecoins explainer](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar). Even custody — the invisible banks that simply hold the world's assets — is, at bottom, a different way to earn a fee on the same plumbing ([custody and securities services](/blog/trading/banking/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets)).

The takeaway for the framework: *when you read a fintech story, find the box of the pipeline it is attacking — funding, lending, payments, or trust — and ask whether it actually escapes the maturity-transformation trade or just repaints it. Almost always, it just repaints it.*

## How to analyze a bank: the four-question playbook (Track I)

Everything now collapses into a workflow. When you pick up any bank — its annual report, its regulatory disclosures (the Pillar 3 report), a news headline — you read it in four questions, in this order.

![A flow of the four questions for reading any bank funding cushion liquidity and value feeding one verdict](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-7.png)

1. **Funding — how does it pay for itself?** Sticky retail deposits, or hot uninsured money and wholesale borrowing that can flee? Look at the CASA ratio, the uninsured-deposit share, the concentration. SVB's 94% uninsured base was the red flag visible a year ahead. Start at the funding footnotes — they answer box one of the pipeline, and the value of a [cheap, sticky deposit base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) is the whole reason one bank trades richer than another.
2. **Cushion — how thick is the capital, relative to the demand?** Compare the CET1 ratio to the regulatory requirement (around 8.5% for a big bank). A ratio barely above the minimum, or one propped up by aggressive [risk-weighted-asset modelling](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work), is a thin cushion in disguise. The denominator is as gameable as the numerator, so always sanity-check the capital ratio against the plain leverage ratio (equity to total assets, no risk-weighting).
3. **Liquidity — how runnable is it, right now?** Look at the liquidity coverage ratio, the size of the high-quality liquid asset buffer, and — the lesson of 2023 — the *unrealized* losses on the [held-to-maturity book and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap), which can turn a paper cushion into a fire-sale trap the moment the bank is forced to sell.
4. **Value — what is the franchise worth?** Compare price-to-book to return-on-equity. A bank earning an ROE above its cost of equity deserves to trade above book value; one earning below deserves to trade below. A high ROE that comes entirely from [leverage rather than a genuine franchise](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) is fragile, not valuable.

The whole machine in one page — the levers management actually pulls — and the integrated economic model live in the Track I posts; this capstone is their synthesis.

#### Worked example: the red-flag scan and valuing the franchise

Run the playbook on two hypothetical banks, side by side.

*Bank A* reports a 14% ROE — impressive at first glance. But the scan shows: funding is 70% uninsured corporate deposits from one industry; CET1 is 8.7%, barely above the 8.5% demand; and the held-to-maturity book carries \$6bn of unrealized losses against \$7bn of equity. The 14% ROE is leverage and duration risk, not franchise. On the valuation: its cost of equity is 11%, and a bank earning 14% might seem to deserve a premium — but because that 14% is fragile, a sober analyst marks it down. Using a simple warranted multiple, price-to-book ≈ (ROE − growth) ÷ (cost of equity − growth); if the *sustainable* ROE through a cycle is really only 9% (once the duration luck reverses) against an 11% cost of equity, the warranted price-to-book is *below* 1.0 — the "cheap-looking high-ROE bank" is a value trap.

*Bank B* reports a steadier 12% ROE: 80% sticky insured retail deposits, CET1 of 13%, a deep liquid buffer, negligible unrealized losses. Its 12% sits comfortably above its 10% cost of equity, and it is sustainable. Warranted price-to-book ≈ (12% − 3%) ÷ (10% − 3%) ≈ 1.3 — it deserves to trade *above* book.

The intuition: *return on equity is meaningless until you know where it comes from. The same headline ROE can mark a fortress or a powder keg; the four-question scan tells you which.* This is the difference between [valuing a bank properly](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) and being seduced by a number.

## A great bank versus a doomed one, across the framework

Lay the two banks from the worked example side by side and the framework does all the talking.

![A doomed bank and a durable bank compared across funding cushion liquidity and value](/imgs/blogs/the-banking-playbook-everything-in-one-framework-capstone-8.png)

The doomed bank funds itself with concentrated, uninsured, runnable money; carries a thin cushion riddled with unrealized losses; cannot sell its held-to-maturity assets without crystallizing those losses; and shows a high ROE that is really just leverage. The durable bank funds itself with diversified, sticky retail deposits; holds capital comfortably above the requirement; sits on a deep buffer of genuinely liquid assets; and earns a steady ~1% ROA through the cycle. Same four dimensions, opposite answers — and the framework predicts which one wakes up dead after a bad headline.

## Common misconceptions

**"A bank keeps your deposit in a vault."** No. Your deposit is a *loan from you to the bank*, which it lends onward. At any moment the bank holds only a fraction of total deposits as cash and reserves — around 13% of assets in our composite. That is the entire point of a bank, and the entire reason it is fragile. If everyone asked for their money at once, no solvent bank on earth could pay, because the money is out on long loans.

**"A profitable bank is a safe bank."** Profitability and safety are different axes. A bank can post a 14% ROE right up to the week it fails, because high returns often come from high leverage or duration risk — the very things that kill it in a downturn. SVB was profitable until it wasn't. Solvency (is the cushion intact?) and liquidity (can it meet withdrawals today?) are the safety tests, not the income statement.

**"Liquidity and solvency are the same thing."** They are the most important distinction in the series. *Solvency* means assets exceed liabilities — the cushion is positive. *Liquidity* means the bank can turn enough assets into cash *right now* to meet outflows. A perfectly solvent bank with \$50bn of good but slow-to-sell loans can be killed by \$10bn of withdrawals it cannot meet today. Most modern failures are liquidity events that happen to solvent-on-paper banks — which is exactly why the lender of last resort exists.

**"More capital just makes banks less profitable, so less is better."** More equity does lower ROE for a given ROA (less leverage), but it raises survival odds enormously, and the asymmetry is brutal: the cost of a little extra capital is a slightly lower return in good years; the cost of too little is the whole bank in a bad one. The 4%-loss-halves-equity example shows why a cushion that looks "wastefully large" in a boom is exactly right for a bust.

**"Deposit insurance means runs can't happen anymore."** Insurance (\$250,000 per depositor in the US) stops *small* depositors from running, which is most retail customers. It does nothing for large *uninsured* balances — and a bank funded by big uninsured corporate deposits, like SVB, is as runnable as any 1930s bank, only faster. The 2023 runs proved that insurance changed the *who*, not the *whether*.

## How it shows up in real banks: one integrated walk-through

Let us read one bank end to end, the way the framework wants — using Silicon Valley Bank as the worked specimen because every link in the chain is visible in it.

**Funding (box one).** SVB's deposits ballooned during the 2020–2021 tech boom to about \$175 billion, of which roughly 94% were *uninsured* — above the \$250,000 limit — and concentrated among venture-backed startups who all banked with the same handful of venture firms and talked to each other constantly. By the four-question scan, this is the brightest possible red flag: hot, concentrated, runnable money. A reader doing the scan in 2022 would have stopped here and worried.

**Assets and the spread (boxes two and three).** Flush with deposits and few loans to make, SVB poured the money into long-dated, fixed-rate securities — high duration. In a low-rate world this earned a modest spread. But it built an enormous positive duration gap: long fixed assets funded by deposits that could leave tomorrow.

**The cushion meets market risk (box four).** When the Fed hiked rates through 2022, the value of those long bonds fell. By early 2023, SVB carried roughly \$17 billion of unrealized losses — a number comparable to its entire equity cushion. On a held-to-maturity basis it could pretend the losses weren't real, but everyone could read the footnotes. Market risk had quietly hollowed out the cushion. *Solvency* was already in question.

**Trust and the run (box five).** SVB tried to plug the hole by selling securities at a loss and raising equity. The announcement, instead of reassuring, screamed the problem out loud. The concentrated, connected, uninsured depositors did the math in unison: \$42 billion fled on March 9, with \$100 billion more queued for March 10. *Liquidity* failed. The bank that had been "solvent on paper" the week before was seized within thirty-six hours.

Every link in the chain we drew at the top of this post is visible in SVB: cheap money came in (box one), got lent long into bonds (box two), earned a thin spread (box three), the cushion got eaten by market risk (box four), and trust evaporated into a run (box five). Nothing about it was novel. It was the oldest trade in finance, run in reverse, at the speed of a group chat. The system-level companion read is [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs); the operational autopsy is [Silicon Valley Bank 2023](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).

Now contrast the same walk-through with JPMorgan, the durable specimen. Its funding is enormous, diversified, and dominated by sticky retail and operating deposits across millions of customers and dozens of industries — no single group can run it. Its CET1 ratio sits comfortably above its requirement. Its liquidity buffer is vast, and it actively hedges its duration gap rather than betting on it. Its ~1% ROA and low-teens ROE come from a genuine deposit-and-fee franchise, not from a single concentrated bet. Same framework, same four boxes — opposite reading at every one. That is why JPMorgan *bought* failed banks in 2008 and 2023 rather than becoming one.

## The takeaway: how to use this

You now hold a complete framework, and it fits on a napkin.

Every bank, anywhere, in any era, is the same machine: it borrows short, lends long, earns a spread of about 3%, runs on a cushion of about 8% equity (leverage near 12x), and lives or dies on solvency and trust. Sixty-one posts of detail — deposits, lending, payments, treasury, risk, capital, regulation, failures, scandals, fintech, valuation — are all zooms into one of the five boxes of that chain. When you understand the chain, the details stop being a list to memorize and become a map you can navigate.

So when you next read about a bank — in a 10-K, a headline, a friend's question about whether their money is safe — run the four questions in order:

1. **How does it fund itself?** Find the deposit mix and the uninsured share. Hot, concentrated money is the first and loudest red flag.
2. **How thick is the cushion, relative to the demand?** Compare CET1 to the requirement; distrust capital that depends on flattering RWA models.
3. **How runnable is it right now?** Check the liquidity buffer *and* the unrealized losses lurking in the held-to-maturity book — the trap that turns a paper cushion into a fire sale.
4. **What is the franchise worth?** Decompose the ROE. Return that comes from a real deposit franchise is durable; return that comes from leverage and duration is a powder keg with a high coupon.

Then ask the two survival questions — *is it solvent?* and *is it liquid?* — in that order, and remember that the answer to the second can be "no" even when the answer to the first is "yes."

The durable lesson, the one this whole series exists to leave you with, is this: **a bank is a confidence machine wearing a balance sheet.** The numbers tell you whether the cushion can absorb the losses; but the cushion only ever gets tested when confidence cracks, and confidence is the one input no regulator can mandate and no spreadsheet can show. Read the numbers to know whether a bank *can* survive a shock. Read the funding base, the concentration, and the conduct record to guess whether it will ever be *asked* to. Hold both in your head at once, and you can read any bank in the world.

## Further reading & cross-links

This capstone is the map; each track below is the territory. Start anywhere and the framework will tell you which box you are standing in.

**Track A — the business model & balance sheet:** [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) · [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) · [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) · [net interest margin and the spread](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) · [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) · [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).

**Tracks B–C — operations: deposits, lending, payments:** [retail deposits, the funding base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) · [the lending business](/blog/trading/banking/the-lending-business-how-a-bank-underwrites-a-loan-end-to-end) · [credit analysis and the five Cs](/blog/trading/banking/credit-analysis-the-five-cs-and-how-a-loan-gets-approved) · [loan pricing](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) · [the payments business](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) · [the cards business](/blog/trading/banking/the-cards-business-issuing-acquiring-interchange-and-the-mdr-split) · [trade finance](/blog/trading/banking/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance).

**Track D — treasury, ALM & the markets side:** [bank treasury and ALM](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) · [interest-rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) · [the funding stack](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds) · [liquidity management, LCR and NSFR](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) · [the repo market](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight) · [securitization](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities).

**Track E — risk, capital & regulation:** [the four risks every bank runs](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational) · [credit risk: PD, LGD, EAD](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) · [market risk and VaR](/blog/trading/banking/market-risk-var-stressed-var-and-the-trading-limits) · [operational risk](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events) · [Basel I, II, III](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) · [risk-weighted assets](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) · [stress testing and CCAR](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills) · [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).

**Tracks F–G — how banks die, and conduct:** [the anatomy of a bank run](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) · [Silicon Valley Bank 2023](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) · [Credit Suisse 2023](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout) · [Lehman Brothers 2008](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank) · [the savings and loan crisis](/blog/trading/banking/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures) · [the Wells Fargo fake-accounts scandal](/blog/trading/banking/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong) · [Barings and Nick Leeson](/blog/trading/banking/barings-and-nick-leeson-1995-how-one-trader-broke-a-300-year-old-bank) · [the LIBOR scandal](/blog/trading/banking/the-libor-scandal-rigging-the-worlds-most-important-number) · [money laundering and AML failures](/blog/trading/banking/money-laundering-and-the-aml-failures-hsbc-danske-and-the-compliance-machine).

**Track H — the modern frontier:** [digital banking and neobanks](/blog/trading/banking/digital-banking-and-the-neobank-business-model) · [open banking and embedded finance](/blog/trading/banking/open-banking-apis-banking-as-a-service-and-embedded-finance) · [stablecoins, CBDCs and bank deposits](/blog/trading/banking/stablecoins-cbdcs-and-the-threat-to-bank-deposits) · [custody and securities services](/blog/trading/banking/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets).

**The system-level companions (outside this series):** [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) · [BIS and Basel regulation](/blog/trading/finance/bis-and-basel-bank-regulation) · [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) · [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) · [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) · [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments) · [central bank digital currencies](/blog/trading/finance/central-bank-digital-currencies-cbdc) · [credit rating agencies](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch) · [stablecoins explainer](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) · [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).

This is educational, not investment advice: it explains how banks work and how they break, not which to buy or avoid.

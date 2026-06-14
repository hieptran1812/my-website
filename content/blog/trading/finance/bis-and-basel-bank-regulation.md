---
title: "The BIS and Basel: How the Central Bank of Central Banks Writes the Rules for Every Bank"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How an obscure institution in Switzerland and a committee that meets behind closed doors decide how much loss every bank on Earth must be able to absorb, why those rules exist, and why banks fight them so hard."
tags: ["banking", "bank-regulation", "basel", "bis", "capital-ratio", "risk-weighted-assets", "tier-1-capital", "leverage-ratio", "stress-tests", "systemic-risk", "financial-crisis"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The Bank for International Settlements is the central bank of central banks, and through the Basel Committee it writes the capital rules that quietly decide how much loss every bank on Earth must be able to absorb before it fails.
>
> - A bank's **capital** is the sliver of money funded by its owners, not its depositors; it is the buffer that soaks up losses first, and when it runs out the bank is insolvent.
> - The central rule is a **capital ratio**: capital divided by risk-weighted assets must be at least 8%, plus extra buffers that push the real-world minimum closer to 10.5% to 13% for big banks.
> - **Risk-weighting** lets a bank hold less capital against assets it claims are safe (a mortgage counts as half, a government bond often as zero), which is both clever and the source of decades of gaming.
> - Before 2008, many giant banks ran on 2% to 3% real capital; a 3% loss on assets was enough to wipe them out, which is why the crisis was catastrophic rather than merely painful.
> - The Basel rules are **soft law**: the committee in Switzerland agrees a standard, and only when your national regulator writes it into local law does your bank have to obey it, which is why the same accord lands differently in the US, the EU, and the UK.

Here is a number almost nobody outside finance knows, and it explains more about the modern world than most headlines. For every \$100 a typical large bank holds in loans and bonds, only somewhere between \$10 and \$13 is money the bank's owners actually put at risk. The other \$87 to \$90 is other people's money: your deposits, money borrowed from other banks, bonds the bank sold to investors. The bank is, in the most literal sense, a highly leveraged bet placed mostly with money that is not its own. The thin slice that *is* its own is called capital, and the single most consequential question in all of bank regulation is: how thin is too thin?

The diagram above is the mental model: a bank is a tall column of other people's money with a thin band of the owners' own money sitting on top, and that thin band is the only thing standing between an ordinary loss and a failure that hits depositors, taxpayers, and the wider economy. The institution that decides how thick that band has to be is not your government, not your central bank exactly, and not any elected body. It is a committee that meets a few times a year in a quiet Swiss city, hosted by an organization most people have never heard of, and its decisions ripple into the cost of your mortgage, the safety of your savings, and the odds of the next financial crisis.

![A bank capital structure with equity on top and deposits below](/imgs/blogs/bis-and-basel-bank-regulation-1.png)

This post takes that hidden machinery apart, slowly and from zero. We will build the handful of ideas you need: what bank capital actually is and why it is not a pile of cash in a vault; what "risk-weighted assets" means and why a dollar of one asset can demand far more capital than a dollar of another; how a capital ratio works and where the famous 8% comes from; what the leverage ratio adds as a blunt backstop; what the liquidity rules do; and what a stress test is. Then we will meet the institutions themselves: the Bank for International Settlements (the BIS) and the Basel Committee, and walk through the accords they have written from 1988 to today's bitter "Endgame" fight. We will see exactly why thin capital made 2008 so destructive, why banks lobby so hard against higher requirements, and how the rules behaved under live fire in March 2023. By the end you should be able to explain, with real numbers, why the most powerful financial rule-maker on Earth is one almost no voter has ever heard of.

## Foundations: capital, risk-weights, and the ratio that runs banking

Before any of the institutions make sense, six ideas have to be solid. None is hard, but the whole subject lives in the relationships between them. This section is the load-bearing wall; the rest of the post leans on it.

### What "capital" actually means (and what it is not)

The word *capital* causes more confusion than any other term in banking, so let us nail it down. **Capital is not cash.** It is not a reserve of money the bank keeps in a drawer for emergencies. Capital is a *source of funding* — specifically, the portion of a bank's money that comes from its owners rather than from people the bank owes.

Think of it through a simple household analogy. Suppose you buy a \$500,000 house. You put down \$50,000 of your own money (your equity) and borrow \$450,000 as a mortgage (your debt). The house is your asset, worth \$500,000. The mortgage is your liability, \$450,000. The \$50,000 gap between them is your equity — your capital. If house prices fall 5%, your house is now worth \$475,000, but you still owe \$450,000, so your equity has dropped from \$50,000 to \$25,000. The loss came out of *your* slice, not the bank's. That is exactly how a bank works, with the roles flipped: the bank is the owner with the thin equity slice, and its depositors and bondholders are the lenders.

A bank's balance sheet is a list with two sides that must add up:

```
Assets        =   Liabilities       +   Equity (capital)
(loans, bonds,    (deposits,            (the owners'
 reserves)         borrowings)           loss-absorbing slice)
```

**Equity, also called capital, is what is left for the owners after every depositor and creditor is paid.** It is the shock absorber. When the bank's assets lose value, that loss is subtracted from equity first. Depositors only start losing money once equity has been completely wiped out — once the bank is **insolvent**, meaning its assets are no longer worth enough to cover what it owes everyone else. Capital is therefore the *measure of how much loss a bank can take before it stops being able to pay people back*. More capital means more room to absorb losses. That is the entire point of capital regulation: force banks to keep enough of this loss-absorbing buffer that ordinary bad luck does not become a failure.

### Leverage: the multiplier on both gains and losses

The flip side of thin capital is high **leverage** — the ratio of a bank's total assets to its capital. If a bank has \$100 of assets and \$5 of capital, its leverage is 20-to-1: it controls \$20 of assets for every \$1 of its own money. Leverage is what makes banking profitable in good times and lethal in bad ones. A 1% return on \$100 of assets is a 20% return on \$5 of capital — wonderful. But a 1% *loss* on \$100 of assets is a 20% loss of capital, and a 5% loss wipes the capital out entirely. The higher the leverage, the smaller the asset loss it takes to destroy the bank. Capital and leverage are two ways of saying the same thing: a bank with 5% capital is levered 20-to-1; a bank with 10% capital is levered 10-to-1.

### Risk-weighted assets: why a dollar is not a dollar

Here is the idea that makes bank regulation genuinely subtle. Not all assets are equally risky. A \$100 loan to a shaky startup is far more likely to default than \$100 lent to the US government. It would be crude to demand the same capital against both. So Basel does not ask banks to hold capital against their *total* assets. It asks them to hold capital against their **risk-weighted assets**, or **RWA** — each asset's dollar value multiplied by a "risk weight" that reflects how dangerous regulators judge it to be.

The risk weights are set by category. A loan to a highly rated government often gets a 0% weight (regulators treat it as riskless). A residential mortgage typically gets a 35% to 50% weight. An ordinary corporate loan gets 100%. A risky, unsecured exposure can get more than 100%. You multiply each asset by its weight and add them up to get total RWA. We will work a full example shortly, but the headline is this: **risk-weighting means the same \$100 asset can require wildly different amounts of capital depending on what kind of asset it is.** This is simultaneously the cleverest and the most-gamed feature of the whole system, and we will return to both sides of that.

### The capital ratio: the rule at the center of everything

Now we can state the rule that runs banking worldwide. A bank's **capital ratio** is its capital divided by its risk-weighted assets:

```
capital ratio = capital / risk-weighted assets (RWA)
```

The foundational Basel requirement is that this ratio must be **at least 8%**. A bank with \$8 of capital for every \$100 of RWA just clears the bar. On top of that 8% sit several *buffers* — extra layers added after 2008 — that in practice push the real-world minimum for a large bank closer to 10.5%, and for the biggest globally important banks toward 13% or more. The buffers exist so that a bank can dip into them in a downturn without immediately breaching the hard 8% floor. Keep the structure in mind: an 8% legal minimum, plus a stack of buffers on top.

### Tier 1 versus Tier 2: not all capital absorbs loss equally

Just as not all assets are equally risky, not all capital is equally good at absorbing losses. Regulators rank capital by quality.

- **Tier 1 capital** is *going-concern* capital — it absorbs losses while the bank is still alive and operating. Its core is **Common Equity Tier 1 (CET1)**: ordinary shares plus retained earnings (profits the bank kept instead of paying out). This is the purest, most reliable buffer, because shareholders have no claim to be repaid — their money is simply at risk. A sub-layer called **Additional Tier 1 (AT1)** consists of special perpetual bonds designed to convert to equity or be written off if capital falls too far; they are loss-absorbing but more fragile and more legally contentious, as 2023 would dramatically show.
- **Tier 2 capital** is *gone-concern* capital — it absorbs losses only once the bank has failed and is being wound down. It is mostly **subordinated debt**: bonds that get repaid only after depositors and senior creditors, so they take losses in a liquidation. It is a real cushion for creditors but does nothing to keep a struggling bank open.

The post-2008 rules deliberately pushed banks toward more CET1 — the highest-quality, most genuinely loss-absorbing capital — because the crisis revealed that fancy hybrid instruments counted as capital on paper but evaporated when it mattered. The structure of regulatory capital, ranked by how readily each piece soaks up a loss, is the subject of figure 7 later in the post.

### Systemic risk and the central-bank coordinator

Two last foundational ideas. **Systemic risk** is the danger that one bank's failure cascades into others — through direct exposures, fire-sale price drops, or sheer panic — and brings down the whole financial system rather than just one firm. A single bakery failing is sad; a single *bank* failing can be contagious, because banks owe each other money and because confidence is fragile. Capital rules exist largely to contain systemic risk: a well-capitalized banking system can absorb a shock without the dominoes falling.

A **central bank** is the public institution that issues a country's currency, sets its interest rates, and acts as lender of last resort to the banking system (the Federal Reserve in the US, the European Central Bank in the eurozone, the Bank of England in the UK). For more on how that money-creation machinery works, see [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier). A *coordinator* of central banks, then, is a body that gets these national institutions into one room so their rules do not contradict each other and so a bank cannot escape tough rules simply by booking its business in a laxer country. That coordinator is the BIS, and the rule-writing arm it hosts is the Basel Committee. Now we can meet them.

## The BIS: the bank that only central banks may use

The Bank for International Settlements sits in Basel, Switzerland, in a curved high-rise nicknamed "the Tower." It is the oldest international financial institution in the world, founded in 1930, and it has one of the strangest customer lists imaginable: its clients are not people or companies but *other central banks*. You cannot open an account there. Roughly 60-plus central banks and monetary authorities — covering the overwhelming majority of world GDP — are its members and customers. This is why it is so often called "the central bank of central banks," and the label is not loose marketing; it is close to a literal job description.

The BIS does three distinct things, and conflating them is the most common source of confusion about it.

### A banker to central banks

First, it is a genuine bank, but only for central banks and a handful of official institutions. When the central bank of, say, Korea wants to hold some of its foreign-exchange reserves safely, earn a return on them, or settle a transaction in gold or dollars with another central bank, it can do that through the BIS. The BIS holds deposits, manages reserves, conducts gold and foreign-exchange operations, and provides short-term credit to central banks. A meaningful share of the world's official currency reserves are managed through or at the BIS. It is, in effect, the place where the institutions that stand behind every national currency keep some of their own money and do business with one another — a layer of plumbing above the layer most people ever see.

### A forum where central bankers meet privately

Second, and arguably more important than its banking, the BIS is a *forum*. Central-bank governors and senior officials gather in Basel regularly — most famously at the bimonthly meetings of central-bank governors — to talk, candidly and off the record, about the state of the world economy, the risks they see building, and how to respond. These are some of the most consequential conversations in global finance, and they happen with no cameras and no minutes released. The value is precisely the privacy: officials can be frank about fragilities they would never say aloud in public for fear of triggering the very panic they are trying to avoid. The BIS hosts the rooms, the secretariats, and the standing committees where this coordination happens.

### A research hub and standard-setting host

Third, the BIS is a research powerhouse and the host of the bodies that write global financial standards. Its economists produce widely read analysis on financial stability, debt, and the plumbing of markets, and its quarterly and annual reports are read closely by anyone who runs money. Crucially, the BIS *hosts* several standard-setting committees as independent bodies under its roof — the most important of which, for our purposes, is the Basel Committee on Banking Supervision. The BIS provides the building, the secretariat, the meeting rooms, and the institutional continuity; the committees do the rule-writing. Keeping this distinction straight matters: the BIS is the host and the banker; the Basel Committee is the rule-maker. People say "Basel says…" when they mean the committee, and "the BIS" when they mean the host institution, and the two get blurred constantly.

It is worth being clear about what the BIS is *not*. It is not a world government for money. It cannot compel any country to do anything. It has no army of inspectors marching into banks. Its power is entirely indirect: it convenes the people who *do* have power in their own countries, and it lends those gatherings legitimacy, continuity, and a neutral venue. That indirectness is the key to understanding everything that follows.

## The Basel Committee and how a rule becomes binding

The Basel Committee on Banking Supervision (often just "the BCBS," or sloppily "Basel") was created in 1974, in the wake of a messy cross-border bank failure (the collapse of West Germany's Bankhaus Herstatt, which left counterparties around the world holding losses because of time-zone settlement gaps). Its founding purpose was to make sure that internationally active banks were supervised somewhere, by someone, to a common standard — so that no bank could fall through the cracks between national regulators, and so that banks from lax jurisdictions could not undercut banks from strict ones.

The committee's membership is the central banks and bank supervisors of the major economies — around 28 jurisdictions today, including the US Federal Reserve and the Office of the Comptroller of the Currency, the European Central Bank, the Bank of England, and the supervisors of Japan, China, Canada, and the rest of the large economies. They meet, debate, and publish standards — capital rules, liquidity rules, supervisory guidance. The way those standards actually reach your bank is the single most misunderstood thing about Basel, so let us trace it carefully.

![A flow from the Basel Committee through national regulators to a bank](/imgs/blogs/bis-and-basel-bank-regulation-6.png)

The figure above shows the chain, and the crucial feature is the gap in the middle. The Basel Committee writes a standard, but **that standard is not law anywhere.** It is what lawyers call *soft law* — an agreed international norm with no direct legal force. For it to bind a single bank, each member country must take the standard home and write it into its *own* binding regulation: in the United States through rules issued by the Federal Reserve, the OCC, and the FDIC; in the European Union through the Capital Requirements Regulation and Directive (CRR/CRD); in the United Kingdom through the Prudential Regulation Authority's rulebook. Only at that final step does your bank actually have to comply.

This structure has two enormous consequences. First, **the same Basel accord lands differently in different places.** Countries can be stricter than Basel (gold-plating) or implement it late, or carve out exceptions. The US, for instance, historically applied the toughest Basel rules only to its largest banks, exempting thousands of community banks — a choice that mattered enormously in 2023. Second, **the rules are powerful precisely because everyone agrees to follow them even though no one is forced to.** The committee has no enforcement arm. Its leverage is peer pressure, reputation, and the simple fact that a country whose banks ignore Basel finds those banks shut out of international markets and treated as risky by everyone else. It is a club whose rules bite because membership is valuable, not because a sheriff enforces them.

#### Worked example: how the 8% minimum sizes a bank's capital

Let us make the central rule concrete. Suppose a bank has built up \$2,000 of risk-weighted assets — its total assets, each multiplied by its risk weight and summed. The Basel minimum capital ratio is 8% of RWA.

Step 1: compute the required capital. 8% of \$2,000 is 0.08 × \$2,000 = \$160. So to satisfy the bare minimum, the bank must hold at least \$160 of regulatory capital.

Step 2: check a bank that holds exactly that. If the bank has \$160 of capital against \$2,000 of RWA, its capital ratio is \$160 / \$2,000 = 8% — it just clears the legal floor.

Step 3: add the buffers. Post-2008, on top of the 8% sits a "capital conservation buffer" of 2.5%, and for many banks a "countercyclical buffer" and a surcharge for being systemically important. Stack a 2.5% conservation buffer on the 8% and the practical minimum becomes 10.5% of RWA, or 0.105 × \$2,000 = \$210. A big global bank with an additional 1% to 3.5% surcharge might need 11.5% to 13%, i.e. \$230 to \$260.

Step 4: see what "breaching a buffer" means. If this bank's capital falls from \$210 to \$180, it is still above the hard 8% floor (\$160) but inside its buffer zone. It is not shut down, but regulators restrict it from paying dividends or bonuses until it rebuilds the buffer. The buffer is a usable cushion, not a second cliff.

The one-sentence intuition: the 8% is a hard floor you must never cross, and the buffers above it are a soft cushion you are allowed to dip into in bad times, at the price of being told to stop paying out cash.

## How risk-weighting actually works

Risk-weighting deserves its own section because it is where the cleverness and the gaming both live. The promise is elegant: instead of treating every asset the same, tie the capital requirement to how risky each asset really is, so banks are not punished for holding safe things and are forced to hold more cushion against dangerous things. The reality is that "how risky" is a judgment, and once there is a judgment, there is a number to be argued over and engineered.

### The standardized weights

Under the simplest, regulator-set ("standardized") approach, assets fall into buckets with fixed weights. The exact numbers have shifted across accords, but the canonical picture looks like this:

| Asset | Typical risk weight | RWA per \$100 |
|---|---|---|
| Cash, reserves at the central bank | 0% | \$0 |
| Highly rated government bonds (own currency) | 0% | \$0 |
| Residential mortgage (prime, low loan-to-value) | 35%–50% | \$35–\$50 |
| Ordinary corporate loan | 100% | \$100 |
| Lower-rated / unsecured corporate exposure | 150% | \$150 |

A dollar of cash demands no capital at all. A dollar of mortgage demands capital against only \$35 to \$50 of RWA. A dollar of corporate loan demands capital against the full \$100. The bank's *total* required capital is 8% (plus buffers) of the *sum* of these weighted amounts, not of its raw assets.

#### Worked example: a mortgage at a 50% risk weight needs half the capital

Take a bank that makes a single \$100 residential mortgage with a 50% risk weight. How much capital must it hold against that loan?

Step 1: compute the RWA. \$100 of mortgage × 50% weight = \$50 of risk-weighted assets.

Step 2: apply the minimum ratio. The bank must hold 8% of that RWA in capital: 0.08 × \$50 = \$4 of capital.

Step 3: compare to a corporate loan. Now suppose instead the bank makes a \$100 loan to a company, weighted 100%. RWA = \$100, and required capital = 0.08 × \$100 = \$8. The same \$100 of lending demands twice the capital purely because of the risk weight.

Step 4: see the incentive this creates. Capital is expensive for a bank (shareholders demand a high return on it). If two loans earn similar interest but one demands half the capital, the bank is steered, hard, toward the low-weight asset. This is why banks love mortgages and government bonds: not only are they "safe," they are *capital-cheap*. The risk weight is a price signal, and banks respond to it exactly as you would expect.

The one-sentence intuition: risk-weighting means the capital a loan costs a bank depends as much on its regulatory weight as on its actual interest rate, which quietly shapes what banks choose to lend against.

### The two approaches, and the conflict between them

There are two ways to get the weights. The **standardized approach** uses the regulator's fixed buckets, as above. The **internal-ratings-based (IRB) approach**, introduced in Basel II, lets sophisticated banks use their *own* statistical models to estimate the riskiness of their assets and thus their own risk weights. The logic was that a big bank knows its loan book better than a generic regulatory table does, so its own models should be more accurate.

The problem is obvious in hindsight: if a bank's own model produces the risk weight, and a lower weight means less expensive capital, the bank has every incentive to build a model that produces low weights. Two banks holding economically identical assets could report very different RWA because their models disagreed — and the bank with the more flattering model looked better capitalized while being no safer. The gaming of internal models is one of the central reasons Basel III and the "Endgame" later clamped down, introducing an **output floor** that says a bank's model-based RWA can be no lower than a set percentage (72.5% in the final standard) of what the standardized approach would produce. The floor is a leash on the models: use your own numbers, but not if they let you claim your assets are dramatically safer than the regulator's table says.

## The leverage ratio: a blunt backstop the models cannot game

Risk-weighting has a fatal vulnerability: if the weights are wrong — because a model is gamed, or because everyone agreed an asset was safe when it was not — then the capital ratio looks healthy while the bank is actually fragile. Before 2008, banks loaded up on assets that carried low risk weights but turned out to be deeply risky (highly rated mortgage securities, in particular), and their risk-weighted capital ratios looked fine right up until the assets blew up. The whole apparatus of risk-weighting failed at exactly the moment it was needed.

The fix Basel III added is a **leverage ratio**: capital divided by *total* assets, with no risk-weighting at all (a few off-balance-sheet items are added in, but the spirit is "everything, weighted at 100%"). The minimum is 3% — capital must be at least 3% of total exposure — and the biggest global banks carry a higher leverage requirement on top. The leverage ratio is deliberately crude. It does not care whether an asset is a government bond or a junk loan; it counts every dollar the same. That crudeness is the point: a measure you cannot game by re-weighting is a useful backstop to one you can.

![Two banks with the same loss, one wiped out and one still standing](/imgs/blogs/bis-and-basel-bank-regulation-3.png)

The before-and-after above is the heart of why capital matters at all, and it is worth dwelling on. Two banks hold the same \$100 of assets and take the same \$3 loss. The thinly capitalized bank, with only \$3 of capital, is wiped out — its equity goes to zero and it is insolvent. The well-capitalized bank, with \$12 of capital, simply absorbs the hit and walks away with \$9 of capital still standing. The loss is identical; the outcome is the difference between a failure and a bad quarter. Everything regulators do is, at bottom, an argument about how thick that buffer should be.

#### Worked example: the leverage ratio versus the risk-weighted ratio on one balance sheet

Consider a bank with \$1,000 of total assets, made up mostly of government bonds (0% weight) and mortgages (50% weight), so that its risk-weighted assets come to only \$300. It holds \$24 of capital. Let us compute both ratios.

Step 1: the risk-weighted capital ratio. Capital / RWA = \$24 / \$300 = 8%. By the headline rule, this bank looks fine — it exactly meets the 8% minimum.

Step 2: the leverage ratio. Capital / total assets = \$24 / \$1,000 = 2.4%. Against the actual size of its balance sheet, the bank holds only 2.4% capital — *below* the 3% leverage minimum.

Step 3: see the conflict. The same bank passes the risk-weighted test and fails the leverage test. The risk-weighted ratio says "you have plenty of capital for how safe your assets are"; the leverage ratio says "but if those weights are wrong, you are levered more than 40-to-1 and have almost no cushion." The bank must satisfy *both*, so here the leverage ratio is the binding constraint — it forces the bank to raise more capital despite passing the headline test.

Step 4: see why both are needed. A bank stuffed with genuinely safe assets is well served by the risk-weighted ratio and over-penalized by leverage; a bank whose "safe" assets are secretly dangerous is caught by leverage when the risk-weighted ratio is fooled. Requiring both means a bank cannot escape capital simply by claiming everything it holds is low-risk.

The one-sentence intuition: the risk-weighted ratio asks "enough capital for your risk?" and the leverage ratio asks "enough capital for your size?", and a safe banking system needs the answer to both to be yes.

## The liquidity rules: surviving a run, not just a loss

Capital answers one question — can the bank absorb losses? — but 2008 and 2023 both showed a second, distinct way a bank dies: it runs out of *cash* even while still solvent. A bank can have plenty of capital and still fail if depositors flee faster than it can turn its assets into money to pay them. Capital is about *solvency*; liquidity is about *survival in the moment*. Basel III added two liquidity rules to address the second.

The **Liquidity Coverage Ratio (LCR)** requires a bank to hold enough high-quality liquid assets — cash and assets that can be sold instantly without a fire-sale loss, mostly government bonds — to cover its expected net cash outflows over a 30-day stress scenario. The idea is that if a run starts, the bank can meet a month of withdrawals out of its own liquid stockpile, buying time for the panic to subside or for an orderly resolution. The rule is, roughly, "hold a month's worth of escape money in instantly sellable form."

The **Net Stable Funding Ratio (NSFR)** works over a one-year horizon and addresses a structural mismatch: it requires that a bank's long-term, illiquid assets be funded by stable, long-term sources (equity, long-term debt, sticky deposits) rather than by flighty short-term borrowing that can vanish overnight. It is the rule that says "do not fund a 10-year mortgage book with money you have to roll over every week." Where the LCR is about surviving the first 30 days of a run, the NSFR is about not building a balance sheet that is a run waiting to happen.

These rules matter because the fastest bank failures are liquidity failures, not capital failures. Silicon Valley Bank in 2023 was, on paper, not catastrophically undercapitalized at the start of its run; it died because it could not raise cash fast enough to meet withdrawals without crystallizing losses on bonds it had been carrying at face value. The full anatomy of that collapse is in the [SVB and Credit Suisse 2023 case study](/blog/trading/finance/svb-credit-suisse-2023-bank-runs); for our purposes, it is the clearest modern proof that capital rules alone are not enough, which is exactly why Basel III paired them with liquidity rules.

## Stress tests: rehearsing the disaster on paper

The capital ratio, the leverage ratio, and the liquidity rules are all *static* — they measure a bank as it is today. But the question regulators really care about is forward-looking: would this bank still be standing after a severe recession? That is what a **stress test** answers. A stress test is a structured "what if" exercise in which regulators take a deliberately harsh hypothetical scenario and run it through a bank's actual books to project what would happen to its capital.

![A pipeline showing the stages of a supervisory stress test](/imgs/blogs/bis-and-basel-bank-regulation-5.png)

The pipeline above shows the mechanism. Regulators design a **severe but plausible scenario** — for example, unemployment jumping to 10%, house prices falling 30%, the stock market halving, a sharp recession lasting two years. They then **apply that scenario to each bank's specific portfolio**: how many of its mortgages would default at 30% lower house prices, how much its trading book would lose, how its revenue would shrink. The result is a stream of **projected losses** that get deducted from the bank's capital. Finally they compute the bank's **post-stress capital ratio** and compare it to the minimum. A bank that stays above the floor passes; one that falls below must raise capital, cut its dividend, or shrink its balance sheet until it would survive the scenario.

In the US this exercise is run annually by the Federal Reserve under the name CCAR / DFAST; in Europe the European Banking Authority runs EU-wide stress tests; the Bank of England runs its own. They are now a central pillar of supervision, and they are powerful for a reason that is easy to miss: a stress test is not about whether the bank passes *today's* rules, but about whether it would survive *tomorrow's* disaster, which is the only thing that ever actually matters.

#### Worked example: a stress-test shortfall that forces a capital raise

Suppose a bank holds \$40 billion of capital against \$400 billion of risk-weighted assets — a 10% capital ratio today, comfortably above the minimum. The regulator's severe scenario then projects that, over the two-year stress horizon, the bank would suffer \$22 billion of losses (defaults plus trading losses) while earning only \$4 billion of net income to offset them.

Step 1: project the post-stress capital. Start with \$40 billion, subtract \$22 billion of losses, add \$4 billion of retained earnings: \$40 − \$22 + \$4 = \$22 billion of capital remaining after the stress.

Step 2: project the post-stress RWA. In a downturn, risk weights rise as assets deteriorate; say RWA grows to \$420 billion. The post-stress ratio is \$22 billion / \$420 billion ≈ 5.2%.

Step 3: compare to the required minimum. Suppose this bank must maintain a 7% minimum (the 4.5% CET1 floor plus its specific buffers and surcharge) even after stress. At 5.2%, it falls short of 7%.

Step 4: size the capital raise. To hit 7% on \$420 billion of RWA, the bank needs 0.07 × \$420 billion = \$29.4 billion of capital. It has \$22 billion. The shortfall is \$29.4 − \$22 = \$7.4 billion. The bank must raise roughly \$7.4 billion of new capital — by issuing shares, retaining earnings, or cutting its dividend — or shrink its balance sheet so the required amount falls. Until it does, the regulator can block its payouts.

The one-sentence intuition: a stress test converts a hypothetical recession into a precise dollar figure of capital the bank must add today, which is how regulators force resilience before the storm rather than after.

## A capsule history: Basel I, II, III, and the Endgame

With the mechanics in hand, the accords themselves tell a clean story — each one a patch on the failure the last one revealed.

![A timeline of the Basel accords from 1988 to the Endgame](/imgs/blogs/bis-and-basel-bank-regulation-4.png)

The timeline above traces forty years of rule-making, and the through-line is simple: every accord raised both the *amount* and the *quality* of capital banks had to hold, usually right after a crisis proved the previous rules too lax.

### Basel I (1988): one rule, one ratio

The first accord was a marvel of simplicity. It said: hold capital equal to at least 8% of your risk-weighted assets, where assets fall into a handful of crude risk buckets (0%, 20%, 50%, 100%). That was essentially it. Basel I's genius was that a single, simple, common rule was vastly better than the patchwork of inconsistent national rules that preceded it, under which a Japanese bank, an American bank, and a German bank could all claim to be adequately capitalized by completely different yardsticks. Its weakness was that the buckets were so crude they invited gaming: every corporate loan got the same 100% weight whether the borrower was rock-solid or nearly bankrupt, so banks were nudged to make the *riskiest* loan in each bucket (more interest, same capital).

### Basel II (2004): let the banks model their own risk

Basel II tried to fix the crudeness by making risk-weighting far more granular and, fatefully, by letting sophisticated banks use their own internal models (the IRB approach) to set their risk weights. The intent was accuracy. The effect, as we have seen, was that banks engineered their models toward low weights, and the headline capital ratios drifted away from economic reality. Basel II also leaned heavily on credit-rating agencies to set weights — and those agencies were, notoriously, rating the very mortgage securities at the heart of the coming crisis as ultra-safe. Basel II was being rolled out in the years right before 2008, and it is fair to say it was overtaken by events almost immediately.

### Basel III (2010 onward): the post-crisis overhaul

After 2008, the Basel Committee tore up large parts of the framework and rebuilt it. Basel III did not change the headline 8% much; instead it attacked every weakness the crisis had exposed. It demanded far more *high-quality* capital — raising the minimum CET1 (pure common equity) component and adding the capital conservation buffer (2.5%), the countercyclical buffer, and surcharges for systemically important banks, so the real minimum for a big bank rose toward 10.5%–13%. It introduced the **leverage ratio** as the non-gameable backstop. It introduced the **LCR** and **NSFR** liquidity rules. And it tightened what could count as capital, throwing out hybrid instruments that had failed to absorb losses in the crisis. Basel III is the regime banks largely operate under today, phased in over more than a decade.

### The "Endgame" (sometimes called "Basel IV"): the final fight

The last major piece of Basel III — finalized in 2017 but still being implemented in the late 2020s — is contentious enough that it has earned its own name. The industry calls it the **"Basel III Endgame"** (US regulators' term) or **"Basel IV"** (the banks' term, used to argue it is really a whole new accord, not a finishing touch). Its centerpiece is the **output floor**: the rule that a bank's model-based RWA cannot fall below 72.5% of the standardized (regulator-table) RWA. This directly caps the gaming of internal models that Basel II had unleashed. It also overhauls the standardized approaches for credit, market, and operational risk. Because the output floor forces some banks — especially European banks that had leaned hard on low internal weights — to recognize meaningfully more RWA and thus hold more capital, it has been ferociously contested. We will return to that fight as a live case study.

Laid side by side, the four accords show a clear arc. Basel I and II shared the same 8%-of-RWA headline, no leverage backstop, and no liquidity rule; their difference was that Basel II let banks model their own risk weights. Basel III kept the 8% but stacked buffers on top, pushed capital toward the high-quality CET1 core, and added the two missing backstops — a 3% minimum leverage ratio and the LCR/NSFR liquidity rules. The Endgame then bolted on the output floor and tighter G-SIB requirements to stop the model-gaming Basel II had unleashed.

![A matrix comparing Basel I, II, III, and the Endgame on four dimensions](/imgs/blogs/bis-and-basel-bank-regulation-2.png)

The matrix above makes the convergence visible: read down any column and you watch the framework acquire, accord by accord, every backstop the previous version lacked.

## Why thin capital made 2008 catastrophic

It is worth stating plainly why all of this matters, using the crisis that reshaped it. In the years before 2008, many of the world's largest banks and investment banks were running on astonishingly thin capital. Reported risk-weighted ratios looked acceptable — often comfortably above 8% — but those ratios relied on risk weights that treated mortgage-backed securities as nearly riskless. Measured against their *total* assets (the leverage view that did not yet have a binding rule), several major institutions held capital of only 2% to 3%. They were levered 30-to-1, 40-to-1, and in some cases higher.

#### Worked example: a 3% loss wipes out a 33-to-1 bank

Take a stylized pre-crisis investment bank with \$1,000 billion of assets funded by just \$30 billion of capital — a leverage of about 33-to-1, and capital of 3% of total assets.

Step 1: identify the fatal threshold. Equity is \$30 billion. Any loss on assets larger than \$30 billion makes the firm insolvent, because the loss is subtracted from equity first.

Step 2: convert that to a percentage. \$30 billion of capital on \$1,000 billion of assets is 3%. So a loss of just **3% of assets** — \$30 billion — is enough to wipe out the entire capital base.

Step 3: feel how small 3% is. A 3% fall in the value of a portfolio is an ordinary bad year, not a once-in-a-century event. When the mortgage securities these banks held lost far more than 3% of their value, the equity did not shrink — it vanished, and then some. The firm owed more than it owned.

Step 4: contrast with a 12%-capital bank. A bank with \$120 billion of capital on the same \$1,000 billion of assets (12%) could absorb a \$120 billion loss — a 12% asset hit — before insolvency. It survives the same shock that destroys the 3% bank. This is precisely the before-and-after picture from figure 3, now at crisis scale.

The one-sentence intuition: at 3% capital, a perfectly ordinary 3% loss is a death sentence, which is why the crisis turned a housing downturn into a global collapse — and why every reform since has been an argument for a thicker buffer.

When Lehman Brothers failed in September 2008, the proximate cause was a liquidity run, but the deep cause was that thin capital meant losses quickly exceeded equity, so creditors faced real losses, so everyone stopped lending to everyone, and the system seized. The full mechanics of that collapse are in the [Lehman Brothers 2008 case study](/blog/trading/finance/lehman-brothers-2008-financial-crisis). The lesson the regulators took away — and built into Basel III — was blunt: the buffer was too thin, made of the wrong stuff, and measured by a yardstick (risk weights) that could be fooled. Fix all three.

## The perennial debate: how much capital is enough?

If thicker capital makes banks safer, why not require a lot more of it — 20%, 30%? This is the central, unresolved argument of bank regulation, and it is worth understanding both sides honestly, because it is not a battle between good and evil but a genuine tradeoff.

**The banks' case against higher capital** runs like this: capital is expensive. Shareholders demand a high return on the equity they put at risk — far higher than the interest paid on deposits or bonds. If you force a bank to fund more of its assets with expensive equity and less with cheap deposits, its overall funding cost rises, and it passes that on by lending less, charging more for loans, or earning lower returns. Banks argue that excessive capital requirements choke lending, slow economic growth, and push activity into the less-regulated "shadow banking" system, where the same risks fester out of sight. They also argue that very high requirements at home, when competitors abroad face lower ones, simply hand business to foreign banks.

**The regulators' case for higher capital** is that the banks systematically understate the cost of their own failures. When a thinly capitalized bank fails, the losses do not stop at its shareholders — they hit depositors, taxpayers (through bailouts), and the whole economy (through frozen credit and recession). Those costs are *externalities*: real costs the bank does not pay but society does. A bank optimizing for its own return on equity will always prefer to run thinner than is socially optimal, because it keeps the upside of leverage while the public absorbs the tail risk. From this view, capital requirements are not a tax on banking; they are the price of not socializing bank losses. Many academic economists argue the social optimum is far higher than current rules — well into the high teens or twenties as a percentage — and that the "capital is expensive" argument largely ignores that more capital makes a bank safer and therefore *cheaper* to fund, partially offsetting the cost.

The truth sits in the tension. Some capital is unambiguously good; the first jump from 3% to 10% bought enormous resilience at modest cost. Whether the next jump from, say, 13% to 18% is worth its cost is a genuinely hard empirical question, and it is exactly what the Endgame fight is about. The honest summary is that the level is contested, the *direction* (more and higher-quality capital than the pre-2008 world) is settled, and the lobbying never stops because billions of dollars of bank profit ride on every percentage point.

## Common misconceptions

**"A bank's capital is a pile of cash it keeps in reserve."** No. Capital is a source of *funding* — the share of the bank's money that comes from owners rather than creditors — not a stash of assets. A bank can hold lots of capital and very little cash, or lots of cash and little capital; they are different things. "Capital" is about who bears losses; "cash" and the liquidity rules are about meeting withdrawals. Conflating them is the single most common error, and it makes the whole subject incomprehensible.

**"The BIS is a global regulator that controls the banks."** The BIS controls nothing. It is a banker to central banks and a host for the committees that write standards. Even the Basel Committee, which does write the rules, cannot enforce them — its standards are soft law that only bind a bank once a *national* regulator enacts them. The system's power comes from voluntary agreement and peer pressure, not from a global enforcement authority.

**"An 8% capital ratio means the bank holds 8% of its assets in capital."** Not quite — it holds 8% of its *risk-weighted* assets. Because risky assets are weighted up and safe ones weighted down (often to zero), a bank can satisfy an 8% risk-weighted ratio while holding far less than 8% of its *total* assets in capital. A bank reporting a healthy 12% risk-weighted ratio might have only 4% or 5% capital against its actual balance sheet. This gap is exactly why the leverage ratio exists as a separate backstop.

**"Higher capital requirements mean banks have less money to lend."** This sounds obvious but is mostly a confusion. Raising capital requirements changes the *mix* of how a bank funds its loans — more equity, less debt — not the total amount of funding available. A bank can hold more capital and lend exactly the same amount; it just has a thicker owner-funded cushion under those loans. There is a real second-order effect (equity funding can be costlier, which can nudge lending down at the margin), but the crude "more capital = less lending" framing is wrong and is a favorite rhetorical move in the lobbying fight.

**"Risk weights are objective measures of how risky an asset really is."** They are regulatory judgments, and frequently wrong ones. Government bonds carry a 0% weight by rule even though governments do default (Greece, Argentina). Highly rated mortgage securities carried low weights right up until they detonated in 2008. Where banks set their own weights via internal models, the numbers bend toward whatever lets the bank hold less capital. A risk weight is a policy choice dressed as a measurement, which is why the leverage ratio (which ignores weights entirely) had to be invented as a check.

**"Basel III made another banking crisis impossible."** It made the system far more resilient — banks today hold several times the high-quality capital they did in 2007 — but it did not abolish bank failures, as 2023 proved. Basel manages the risks it was designed for (loss-absorption, basic liquidity) and is regularly outflanked by risks it underweighted (the speed of a digital deposit run, the concentration of uninsured depositors, interest-rate risk on "safe" bonds held to maturity). Rules fight the last war; the next crisis usually arrives through a door the rules left ajar.

## How it shows up in real markets

### 2008: the buffer that was not there

The crisis is the founding trauma of modern capital regulation, and its mechanism was exactly the thin-capital story above. Major institutions ran on 2%–3% capital against total assets while reporting acceptable risk-weighted ratios, because the assets sinking them — highly rated mortgage-backed securities — carried low risk weights. When those securities lost far more than a few percent of their value, equity was obliterated, and because every big bank was simultaneously exposed and simultaneously thin, the failures threatened to cascade. Governments stepped in with bailouts precisely because the alternative — letting an undercapitalized, interconnected system fail — was judged catastrophic. The bailouts proved the regulators' core point: when a bank's buffer is too thin, the public, not the shareholder, ends up holding the loss. Every dollar of higher capital required since has been justified by reference to this episode.

### The Basel III rollout: a decade of phasing in

Basel III was agreed in 2010 but deliberately phased in over more than a decade, with full implementation of the final pieces stretching into the late 2020s. The slow rollout was itself a policy choice: forcing banks to raise hundreds of billions of dollars of new capital overnight would have crushed lending in a fragile post-crisis economy, so the rules ratcheted up gradually to let banks build capital out of retained earnings rather than emergency share sales. By the early 2020s, large banks held common-equity capital several times the level of 2007 — a genuine, measurable increase in resilience. The rollout is the rare example of a major regulatory overhaul that largely achieved its stated goal: the banking system entered the 2020 pandemic shock far better capitalized than it entered 2008, and it absorbed that shock without a banking crisis.

### March 2023: the rules under live fire

The collapse of Silicon Valley Bank and the forced rescue of Credit Suisse stress-tested the Basel framework in real time, and the results were mixed. SVB exposed a gap: it had loaded up on long-dated government bonds (a 0% or low risk weight, capital-cheap, and supposedly "safe"), but those bonds lost value as interest rates rose, and SVB was not subject to the full liquidity rules because US regulators had exempted mid-sized banks. The capital framework had not flagged the danger because interest-rate risk on bonds held to maturity sat in a blind spot. Credit Suisse, by contrast, *was* a fully Basel-regulated global bank with strong reported ratios — and it still failed, because capital cannot stop a crisis of confidence once depositors and counterparties decide to leave. Its rescue also detonated the AT1 bond market: roughly \$17 billion of Credit Suisse's Additional Tier 1 bonds were written to zero while shareholders received some value, inverting the loss hierarchy investors thought they understood and forcing a re-pricing of that entire instrument class. The episode (dissected fully in the [2023 bank-runs case study](/blog/trading/finance/svb-credit-suisse-2023-bank-runs)) showed both that the rules had genuinely strengthened the core and that they could still be outrun by speed, concentration, and a risk the weights ignored.

### The US "Endgame" lobbying fight

When US regulators proposed their version of the Basel III Endgame in 2023, the banking industry mounted one of the most aggressive lobbying campaigns in regulatory memory — including, unusually, prime-time television advertisements warning ordinary Americans that the rules would raise the cost of their mortgages and small-business loans. The original US proposal would have increased aggregate capital requirements for the largest banks by an estimated high-teens percentage, and the banks argued this was unjustified gold-plating that would choke lending and hand business to foreign competitors and shadow banks. Regulators countered that the increase merely closed gaps the 2008 and 2023 episodes had exposed. The fight dragged on, and by 2024 regulators signaled they would substantially scale back the proposed increase — a vivid, public demonstration of the core dynamic: the *direction* of reform (more capital) is settled, but the *level* is fought over relentlessly, and the banks have enormous resources to deploy in that fight. It is the clearest recent proof that capital rules are not handed down from a technocratic mountaintop; they are negotiated, contested, and shaped by power.

### How risk-weighting steered banks into sovereign bonds

A quieter but profound effect of the framework is the way the 0% risk weight on government bonds shapes the entire financial system. Because a bank holds *no* required capital against highly rated government debt in its own currency, those bonds are the most capital-efficient asset a bank can own — they earn interest while costing nothing in capital. This creates a powerful, permanent incentive for banks to stuff their balance sheets with government bonds, which in turn gives governments a captive buyer for their debt and ties the health of banks tightly to the health of their sovereign. In the European debt crisis of 2010–2012, this "doom loop" became acute: banks held huge quantities of their own government's bonds at a 0% weight, so when a government's creditworthiness wobbled, its banks — supposedly well-capitalized — were suddenly sitting on losses the risk weights had said were impossible. SVB in 2023 was the same pattern in miniature: "safe," capital-cheap government bonds whose market value fell, with the capital framework looking the other way. The 0% sovereign weight is one of the most consequential and most criticized design choices in all of Basel, precisely because it is a policy decision masquerading as a statement of fact about risk.

### The components of capital, made concrete

![A tree of regulatory capital from Tier 1 down to subordinated debt](/imgs/blogs/bis-and-basel-bank-regulation-7.png)

The tree above is the taxonomy we built in the foundations, now drawn out: total regulatory capital splits into Tier 1 (going-concern, able to absorb losses while the bank operates) and Tier 2 (gone-concern, absorbing losses only in a wind-down). Tier 1's core is CET1 — ordinary shares and retained earnings, the purest loss absorber — with AT1 perpetual bonds as a more fragile addition. Tier 2 is mostly subordinated debt plus certain loan-loss reserves. The reason this hierarchy is load-bearing rather than bureaucratic is that 2008 and 2023 both turned on *quality*: instruments that counted as capital on paper but failed to absorb losses when the moment came. Basel III's central capital reform was to push the requirement down toward the top of this tree — more CET1, less reliance on the fragile lower layers — because the only capital that reliably protects depositors is the kind that takes the loss without anyone having to argue about it in court. For where banks sit in the wider ecosystem of financial firms, see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions).

## When this matters to you and further reading

This machinery feels remote, but it touches your life at several specific points. When you deposit money, the reason you can treat a bank account as "safe" is a combination of deposit insurance and the capital and liquidity rules that make the bank unlikely to need it — the thicker buffer Basel III forced is part of why your savings are safer in 2026 than they were in 2007. When you take out a mortgage, the rate you pay is shaped in part by the risk weight regulators assign to mortgages: a capital-cheap asset is one banks compete to lend against, which holds rates down. When a politician argues about "burdensome bank regulation" or "letting banks lend more," they are arguing about exactly the capital ratios in this post, and you now know what is actually at stake — resilience against the next crisis on one side, the cost of credit and the competitiveness of domestic banks on the other. And when the next banking scare hits the news, you will be able to read past the panic to the real question: was the buffer thick enough, was it made of the right stuff, and did the rules measure the risk that actually mattered?

If you want to go deeper, the natural next steps are the case studies where these rules met reality: the [Lehman Brothers 2008 collapse](/blog/trading/finance/lehman-brothers-2008-financial-crisis) for the thin-capital crisis that built Basel III, and the [SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) for the rules under modern live fire. To understand the broader system the banks sit inside — how money itself is created and how the different kinds of financial firms fit together — read [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) and the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions). The primary sources themselves are surprisingly readable: the Basel Committee publishes its standards openly, and the BIS annual reports are among the clearest writing in all of finance about the risks building under the surface of the global financial system. None of this is investment advice; it is the structural knowledge that lets you understand why the financial world is shaped the way it is, and why the most powerful rule-maker in it is one almost no one has ever heard of.

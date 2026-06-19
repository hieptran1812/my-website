---
title: "Basel III/IV: How Bank Capital Rules Set the Price of Credit"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How the Basel capital framework decides how much equity a bank must hold against each loan or trade, and how that single rulebook throttles the cost and quantity of credit across the whole economy."
tags: ["regulation", "basel", "bank-capital", "credit", "leverage-ratio", "liquidity", "rwa", "macro", "geopolitics", "banking", "treasury-market", "policy"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A global rulebook called Basel decides how much shareholder equity a bank must hold behind every loan, bond, and trade, and that one set of rules quietly sets the price and quantity of credit for the entire economy.
>
> - Banks must fund a slice of every asset with their own equity (a loss-absorbing cushion). The required slice depends on the asset's official *risk weight* — and a low weight on government bonds versus a high weight on small-business loans steers banks toward holding bonds and away from lending.
> - On top of risk-weighted rules sits a blunt *leverage ratio* (the Supplementary Leverage Ratio, SLR, in the US) that counts every asset equally. When it binds, dealers stop warehousing Treasuries — which is exactly what broke the world's safest market in March 2020.
> - More capital makes a bank safer but lowers its return on equity, so the system supplies less credit at a higher spread. "Safer with no cost" is a myth: tighter capital is a tax on lending that shows up as wider loan spreads.
> - The one number to remember: an 8% total capital requirement means roughly **\$1 of bank capital can support about \$12.50 of risk-weighted assets** — change that 8% and you change how much credit the system can create.

On the morning of March 9, 2020, the safest, deepest, most liquid market on Earth — the \$20-trillion market for US Treasury bonds — stopped working. Spreads between nearly identical bonds blew out. Prices that should never gap, gapped. Investors trying to *sell* the safest asset in the world, to raise cash, found no one on the other side. The "dash for cash" forced the Federal Reserve to buy roughly \$1 trillion of Treasuries in three weeks to glue the market back together.

Why did the dealers — the big banks whose job is to stand in the middle and absorb selling — step back at the worst possible moment? Not because they were scared of Treasuries. They were scared of a rule. A regulatory ratio called the Supplementary Leverage Ratio charged them capital for *every* dollar of assets on their balance sheet, even risk-free Treasuries. As clients dumped bonds onto dealers, the dealers' balance sheets ballooned, the ratio tightened toward its limit, and the cheapest response was to *refuse to buy more*. A capital rule designed to make banks safer made the safest market unsafe.

This is the hidden hand we will trace in this post. Behind almost every question about how much credit exists, how cheap it is, how leveraged the financial system can get, and where liquidity disappears in a crisis, sits a single rulebook: the Basel framework. It is not a law passed by any one legislature — it is a set of standards written by central bankers in a Swiss city — yet it dictates the price of credit more directly than almost any statute. Let us build it from the ground up.

![Bank balance sheet showing assets funded by deposits debt and a thin equity cushion that absorbs losses](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-1.png)

## Foundations: a bank balance sheet, capital, and risk-weighted assets

Before we can talk about Basel, we need to understand what a bank *is* in accounting terms. A bank is a balance sheet — a list of what it owns and what it owes — and everything about capital rules is a constraint on the shape of that list.

Start with an everyday comparison. You buy a \$500,000 house with a \$100,000 down payment and a \$400,000 mortgage. The house is your *asset*, the mortgage is your *liability*, and your \$100,000 down payment is your *equity* — your skin in the game. If house prices fall 10%, the house is now worth \$450,000 but you still owe the bank \$400,000, so your equity has dropped from \$100,000 to \$50,000. You absorbed the entire \$50,000 loss; the lender lost nothing, because your equity stood between the loss and their loan. Now suppose you had put only \$20,000 down and borrowed \$480,000: the same 10% price drop wipes out your \$20,000 *and* leaves the lender \$30,000 short. A bank works exactly the same way, only the "house" is a portfolio of loans and securities, the "mortgage" is depositors and bondholders, and the "down payment" is the shareholders' equity. Capital rules are simply a regulator telling banks: *put more down.*

### A bank is three stacks

Every balance sheet has three parts, and they obey one iron rule: **assets = liabilities + equity**.

- **Assets** (what the bank owns and earns on): loans it has made (mortgages, business loans, credit-card balances), securities it holds (government bonds, corporate bonds), cash, and reserves at the central bank. This is the left side.
- **Liabilities** (what the bank owes): mostly *deposits* (your checking account is the bank's IOU to you), plus borrowed money — bonds the bank issued, money borrowed overnight. This is the top-right.
- **Equity** (also called *capital*): the difference between assets and liabilities. It is the shareholders' stake — the money put in by owners plus profits retained. This is the bottom-right.

A simple example. Suppose a bank holds \$100 of assets, funded by \$92 of deposits and debt and \$8 of equity. If \$5 of its loans go bad and are written off, assets drop to \$95. The bank still owes depositors their \$92 in full — that promise does not shrink. So equity absorbs the loss: it falls from \$8 to \$3. The depositors are untouched. **Equity is the shock absorber that stands between a loss and the depositors.** That is the entire point of capital.

Now flip it. Suppose the bank had funded the same \$100 of assets with \$98 of deposits and only \$2 of equity. The same \$5 loss now wipes out the \$2 of equity *and* eats \$3 into what is owed to depositors. The bank is insolvent — its assets are worth less than its debts. The thinner the equity cushion, the smaller the loss it takes to push the bank under. This is why regulators care so much about how much equity a bank holds: equity is what keeps a loss from becoming a run, and a run from becoming a crisis.

### "Capital" is not a pile of cash

The single most common confusion: people hear "the bank must hold more capital" and assume a vault of cash sits idle. Wrong. **Capital is a source of funding, not a use of it.** It sits on the right side of the balance sheet (how the bank is funded), not the left (what the bank does with the money). Telling a bank to "hold more capital" means "fund more of your assets with shareholders' equity and less with borrowing and deposits." The bank still lends the money out. It just has a bigger owner-funded buffer behind those loans. We will hammer this point again, because nearly every misunderstanding of capital rules starts here.

### Risk-weighted assets: not all assets are equally risky

If equity must absorb losses, then the amount of equity a bank needs should depend on how *risky* its assets are. A bank stuffed with risk-free government bonds needs a thinner cushion than a bank stuffed with speculative real-estate loans. Basel encodes this with **risk-weighted assets (RWA)**.

The idea is simple. Take each asset, multiply its dollar value by a **risk weight** — a percentage that says "how risky is this, relative to a benchmark business loan?" — and sum it up. The benchmark, a standard corporate loan, gets a 100% weight. Safer assets get lower weights; riskier assets get higher ones. Where do the weights come from? Two routes. Smaller banks use the **standardized approach**, where the regulator hands them a fixed table of weights (the one below). Large banks may apply to use the **Internal Ratings-Based (IRB) approach**, estimating their own probability-of-default and loss-given-default for each exposure and feeding those into a Basel formula that spits out a risk weight. The IRB route is where models can shave RWA — and it is exactly the loophole that the Basel IV output floor was built to close, as we will see.

The history here matters. The very first Basel Accord in 1988 (Basel I) had only a handful of crude buckets — 0%, 20%, 50%, 100%. Basel II (2004) added the internal-model route and far finer gradations, which let sophisticated banks shrink their RWA dramatically; that flexibility was a major reason banks entered 2008 with razor-thin capital against assets they had modeled as safe. Basel III (2010–2017, the framework we live under) was the post-crisis correction: more capital, better-quality capital, plus the leverage ratio, the liquidity ratios, and the buffers — all the rules we are about to walk through. So when you read "risk weight," remember it is the third generation of a thirty-year argument about how to measure bank risk, and every generation has been an attempt to plug the holes the previous one left.

A rough map of the standardized risk weights under Basel:

- **0%** — claims on your own government in its own currency (US Treasuries for a US bank, Bunds for a German bank). Treated as risk-free. *Zero capital required.*
- **20%** — high-grade exposures, certain interbank claims, agency mortgage-backed securities.
- **35–50%** — residential mortgages (low-LTV), depending on the loan-to-value ratio.
- **75%** — qualifying retail and small-business (SME) exposures.
- **100%** — standard corporate loans, the benchmark.
- **150%** — past-due loans, certain high-risk and speculative exposures.

So a \$100 million Treasury position has \$0 of RWA (100m × 0%). A \$100 million book of corporate loans has \$100 million of RWA. The *same dollar amount* of assets generates wildly different capital requirements depending on what kind of asset it is. Hold this thought — it is the lever that steers all bank lending.

### The capital ratios

Basel then sets the rule as a ratio: **capital ÷ risk-weighted assets ≥ a minimum percentage.** Capital comes in tiers, ranked by how good it is at absorbing losses:

- **CET1 (Common Equity Tier 1):** the best capital — common shares plus retained earnings. The cleanest, most loss-absorbing equity. Minimum: **4.5%** of RWA.
- **Tier 1:** CET1 plus certain perpetual instruments (Additional Tier 1, like contingent-convertible "CoCo" bonds that convert to equity in stress). Minimum: **6%** of RWA.
- **Total capital:** Tier 1 plus Tier 2 (subordinated debt that absorbs losses only in a wind-down). Minimum: **8%** of RWA.

That **8% total / 6% Tier 1 / 4.5% CET1** is the Basel III minimum. Memorize the 8%. It is the headline number from which the credit multiplier flows.

Why the tiers? Because not all "capital" absorbs losses equally well, and the regulator wants to make sure the bank has enough of the *good* kind. Common equity (CET1) absorbs losses instantly and continuously — when a loan goes bad, the loss flows straight through to common shareholders, who have no claim to be made whole. That is why CET1 is the gold standard and gets its own, highest, requirement. Additional Tier 1 instruments (the CoCo bonds) sit one notch down: they pay a bond-like coupon in good times but are contractually written down or converted into equity when the bank's CET1 ratio falls through a trigger — so they absorb losses while the bank is still a *going concern* (still operating). Tier 2 is subordinated debt that only takes losses in *gone-concern* — a wind-down or liquidation. The ladder, from CET1 down to Tier 2, is a ladder of how *early* in the bank's distress each instrument starts eating losses. Pre-2008, banks padded their ratios with low-quality "capital" that turned out not to absorb losses when it mattered; Basel III's insistence on a high CET1 floor is the direct fix.

One more subtlety worth knowing: the ratios are measured continuously and reported quarterly, and breaching even a *buffer* (not just a minimum) triggers automatic consequences. There is no grace period and no discretion — the rule is mechanical, which is precisely what makes it predictable enough to trade around.

![Risk weight matrix mapping asset classes to Basel risk weights and the capital required per one hundred million dollars](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-2.png)

### The buffers on top

The minimums are the floor below which a bank is in regulatory trouble. But Basel III stacks several **buffers** on top, designed to be drawn down in bad times and rebuilt in good times:

- **Capital conservation buffer:** +2.5% CET1, always on. Breach it and you can keep operating but face automatic limits on dividends and bonuses (the "Maximum Distributable Amount" brake) until you rebuild. This is the buffer's whole trick — it makes the bank cut payouts, not lending, when stressed.
- **Countercyclical capital buffer (CCyB):** 0–2.5% CET1, dialed up by national regulators when credit is growing too fast and released in a downturn. A *macroprudential* lever: lean against the credit cycle.
- **G-SIB surcharge:** an extra 1–3.5% CET1 for "Global Systemically Important Banks" — the JPMorgans and HSBCs whose failure would threaten the whole system. The bigger and more interconnected you are, the more capital you carry.

Add it up for a large US bank and the *effective* CET1 requirement is often 10–13%, not the bare 4.5%. The buffers do most of the real work.

![Capital stack ladder from CET1 to Tier 1 to total capital with conservation countercyclical and GSIB buffers above the minimum](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-3.png)

### The leverage ratio: the blunt backstop

Risk weights are clever, but cleverness is a weakness. A bank (or its regulator) can game the weights — declare assets safer than they are, pile into "low-risk" exposures that turn out to be toxic (as happened with AAA-rated mortgage tranches before 2008). So Basel III added a deliberately dumb backstop: the **leverage ratio**.

The leverage ratio ignores risk weights entirely. It is simply **Tier 1 capital ÷ total assets** (with off-balance-sheet exposures added in), and Basel sets a **3% minimum** — meaning total assets can be at most ~33× Tier 1 capital. The US gilds this with a higher bar for its biggest banks, the **Supplementary Leverage Ratio (SLR)**: 3% for large banks, plus a 2% "enhanced" buffer for the eight US G-SIBs, so **5%** at the holding-company level.

The leverage ratio is meant to be a *backstop* — a floor that only bites when the risk-weighted requirement has been gamed too aggressively. But when a bank loads up on genuinely low-risk-weight assets (Treasuries, reserves at the Fed, repo), the risk-weighted requirement stays loose while the leverage ratio tightens, because the leverage ratio counts those "safe" assets at full dollar value. When that happens, the leverage ratio *binds* — it, not the risk-weighted rule, becomes the constraint. And a binding leverage ratio is what makes dealers walk away from the Treasury market. We will return to this; it is the single most important market-plumbing fact in the whole framework.

The key mental model is that a bank faces *two* capital ratios at once — the risk-weighted ratio (8% of RWA) and the leverage ratio (3% or 5% of *total* assets) — and at any moment **one of them is tighter than the other**. Whichever requires *more* capital is the *binding* constraint; the other is slack. A bank full of risky, high-weight corporate loans is usually bound by the risk-weighted ratio. A bank full of safe, low-weight Treasuries and reserves is bound by the leverage ratio. The whole behavior of the bank — what it will hold, what it will refuse — flips depending on which ratio binds. A trader who does not first ask "which ratio is binding here?" is guessing. We will make this the first question of the playbook.

### The liquidity rules

Capital protects against *losses*; it does nothing for *liquidity* — the risk that a solvent bank simply runs out of cash because depositors flee faster than it can sell assets. SVB in 2023 was solvent on paper one week and seized the next. So Basel III added two liquidity ratios:

- **LCR (Liquidity Coverage Ratio):** the bank must hold enough *high-quality liquid assets* (cash, reserves, Treasuries) to cover 30 days of stressed outflows. LCR ≥ 100%. It answers: *can you survive a one-month run?*
- **NSFR (Net Stable Funding Ratio):** the bank's long-term assets must be funded by stable, long-term funding (not flighty overnight money). NSFR ≥ 100%. It answers: *is your funding structurally sound, not just for a month?*

These rules are *why* banks now hoard Treasuries and reserves: those assets count as high-quality liquid assets (HQLA) for the LCR. A liquidity rule, like a capital rule, steers what banks choose to hold. Keep that pattern in mind — every Basel rule is a thumb on the scale of bank behavior.

It is worth understanding *why* a liquidity rule was even necessary, because it is a different failure mode from the one capital protects against. Capital protects against **insolvency** — the bank's assets are worth less than its debts. Liquidity protects against **illiquidity** — the bank is perfectly solvent (assets exceed debts) but cannot turn its assets into cash *fast enough* to pay fleeing depositors. Northern Rock in 2007 and Lehman in 2008 were as much liquidity failures as solvency ones: they relied on short-term wholesale funding that vanished overnight, and they could not sell their illiquid assets quickly without crashing the price. A bank can be solvent on Friday and gone on Monday purely from a liquidity run, which is why Basel III added the LCR and NSFR alongside the capital rules. SVB in 2023 is the textbook modern case: it was arguably solvent at hold-to-maturity values, but 94% of its deposits were uninsured and flighty, and once they ran, no amount of "solvency" could save it. The liquidity rules try to make sure a bank can survive the *month* of a run, not just the *year* of a loss — and like every Basel rule, they do it by forcing banks to hold a particular kind of asset, which warps the whole market for that asset. The world's banks now own a structural mountain of Treasuries partly because Basel's liquidity rules make them the cheapest way to satisfy the LCR.

## How a risk weight steers lending: the zero-weight bias

Now the payoff of all that setup. The risk weights are not neutral accounting; they are a *price* that the regulator puts on holding each kind of asset, and banks optimize against that price. The cheapest asset to hold — in capital terms — is a zero-risk-weight government bond. The most expensive is a high-weight loan to a small business or speculative borrower. So, all else equal, the rules nudge a bank toward holding government bonds and away from lending to the real economy.

Let us make that concrete.

#### Worked example: required capital for a \$100m loan at 100% vs 20% risk weight

Take a bank with two ways to deploy \$100 million, and apply the 8% total-capital rule.

- **A corporate loan book**, risk weight 100%. RWA = \$100m × 100% = \$100m. Required total capital = 8% × \$100m = **\$8m**. The bank must put \$8m of its own equity behind this book.
- **A portfolio of high-grade exposures**, risk weight 20%. RWA = \$100m × 20% = \$20m. Required capital = 8% × \$20m = **\$1.6m**.
- **US Treasuries**, risk weight 0%. RWA = \$0. Required capital = **\$0**.

The same \$100 million of assets demands \$8m, \$1.6m, or \$0 of scarce equity depending only on the regulatory label. Equity is the bank's most expensive funding (shareholders demand a high return), so the bank will, at the margin, prefer the asset that ties up the least of it. *The risk weight is a tax on lending, and government debt is tax-free.*

This is not a hypothetical bias — it is visible in bank balance sheets worldwide, which have grown steadily more stuffed with sovereign bonds since Basel III. It also creates the so-called *sovereign-bank doom loop*: a domestic banking system that loads up on its own government's zero-weighted debt becomes fragile precisely when that government's solvency is in question, as Europe learned in 2011–2012. The rule that calls sovereign debt risk-free is the rule that chains banks to their sovereign.

The lending bias has a second, sharper edge for small business. SME loans carry a 75–100% weight while a mortgage carries 35–50% and a government bond 0%. A capital-constrained bank, told to shrink its RWA, will cut the highest-weight assets first — and those are loans to the productive, job-creating, but capital-hungry part of the economy. When you read that "banks are pulling back from small-business lending after a new capital rule," this is the mechanism. The rule did not ban the loan; it just made the loan the most expensive thing on the menu.

There is a subtle reason regulators tolerate the distortion: the alternative — treating every asset alike — is *worse*. A flat capital charge would make a risk-free Treasury and a junk-rated startup loan cost the same equity, which would push banks toward the *riskier* asset (it pays more for the same capital). Risk-weighting at least pushes the capital where the danger is. The trouble is that the weights are administered prices set by a committee, and any administered price creates arbitrage at its seams. Banks have spent decades hunting the gaps: turning a 100%-weight corporate loan into a 20%-weight securitized tranche; parking exposure with a low-weight counterparty; structuring around the boundaries of the IRB formula. Every reform closes some seams and opens others. The deep lesson for a market reader is that **the risk weights are not a neutral description of risk — they are a set of incentives, and capital flows toward whatever the rules under-price.** Find the under-priced corner of the Basel table and you have found where the next round of bank balance-sheet growth — and the next round of hidden risk — will pool.

A worked illustration of how a *small* weight change reshapes a balance sheet:

#### Worked example: a risk-weight change that shifts \$5bn of lending

A regional bank has \$10bn of CET1 and runs a tight 10% CET1 ratio, so it can carry \$100bn of RWA. It currently splits this as \$60bn of corporate loans (100% weight → \$60bn RWA) and \$80bn of residential mortgages (50% weight → \$40bn RWA), exactly hitting \$100bn of RWA.

Now suppose the regulator raises the mortgage risk weight from 50% to 65% (a real lever national regulators pull to cool a housing boom). The same \$80bn of mortgages now generates \$52bn of RWA, not \$40bn — \$12bn more, with no new lending at all. To stay at \$100bn of RWA against its \$10bn of capital, the bank must shed \$12bn of RWA somewhere, and the cheapest cut is its 100%-weight corporate book: it must reduce corporate loans by \$12bn (each \$1 of which is \$1 of RWA). A 15-point tweak to one weight just forced a \$12bn contraction in business lending.

The intuition: because capital is the binding budget, a risk-weight increase on *one* asset class forces the bank to cut lending in *another* — the weights are connected through the single capital constraint, so a regulator nudging one number ripples across the entire loan book.

## How the leverage ratio and SLR bind: why dealers abandon Treasuries

Here is the counterintuitive twist that surprises even finance professionals. We just said zero-risk-weight Treasuries are *capital-free* under the risk-weighted rule. That is true — for the risk-weighted rule. But the **leverage ratio counts them at full dollar value.** And for the largest US dealer banks, the leverage ratio (the SLR) is often the *binding* constraint, not the risk-weighted ratio.

Think about what a Treasury dealer does. It buys bonds from sellers, holds them briefly in inventory, and sells them to buyers, earning the tiny bid-ask spread. To do this it must *warehouse* the bonds on its balance sheet, even for a day. Each bond, no matter how safe, adds a dollar of total assets — and the SLR demands 5% capital against every dollar of total assets at a US G-SIB. The capital cost of a risk-free asset under the SLR is not zero; it is 5%.

In normal times this is a cost of doing business. In a panic, when everyone wants to sell Treasuries at once, the dealer's inventory swells, total assets balloon, and the SLR slams toward its limit. The marginal cost of buying one more bond — in scarce capital — explodes. The rational dealer response is to *stop buying*. That is precisely what happened in March 2020, and it is what makes a leverage ratio a market-structure event, not just a bank-safety rule.

The deep irony is worth sitting with. The risk-weighted rule says "Treasuries are risk-free, hold all you want, zero capital." The leverage rule says "every dollar of balance sheet costs you 5% capital, Treasuries included." So the *same regulatory framework* simultaneously tells dealers that Treasuries are free and that they are expensive — and which message governs depends entirely on which ratio is binding at that moment. In calm markets, dealers have spare leverage headroom and behave as if Treasuries are nearly free. In a flood, the leverage ratio binds and dealers behave as if Treasuries are the most expensive thing they could possibly hold. This regime-switch is the single most important thing to understand about the modern Treasury market: its liquidity is *state-dependent on a regulatory ratio*, abundant when the SLR is slack and gone when it binds, with very little in between. Markets that look bottomlessly liquid 99% of the time can vanish in the 1% precisely because of this nonlinearity.

#### Worked example: the leverage ratio binding on a \$ Treasury book

A US G-SIB has \$150 billion of Tier 1 capital and must satisfy a 5% SLR. Its maximum total balance sheet is therefore Tier 1 divided by the SLR: \$150bn ÷ 0.05 = \$3,000bn, or **\$3 trillion**. The leverage ratio puts a hard ceiling on how big the bank can get, regardless of how safe its assets are.

Suppose it is already running \$2.94 trillion of assets — comfortable on the risk-weighted ratio (mostly low-weight stuff), but only \$60bn of headroom under the SLR. A wave of clients now wants to dump \$200bn of Treasuries onto the dealer. To absorb them, the dealer's assets would have to rise to \$3.14 trillion — \$140bn *over* its \$3 trillion ceiling. It physically cannot, without raising new equity overnight (impossible in a panic) or dumping other assets. So it buys \$60bn, hits the wall, and refuses the rest. The Treasury market, starved of its shock absorber, gaps.

The intuition: under the SLR a risk-free Treasury costs the same balance-sheet capacity as a junk loan, so in a flood the dealer rations the scarce capacity and walks away from the safest asset of all.

This is why, in April 2020, the Fed and bank regulators temporarily **exempted Treasuries and reserves from the SLR denominator** — letting banks hold them "for free" again so dealers could re-enter the market. The exemption expired in March 2021, and the debate over whether to make some version of it permanent has run ever since. It is one of the most important live regulatory questions for anyone who trades rates: *if the SLR is reformed to exempt Treasuries, dealer balance-sheet capacity expands and Treasury-market liquidity improves; if it is not, the next dash-for-cash has the same fault line.* We will put this in the playbook.

![Causal flow showing client Treasury selling expands the dealer balance sheet which tightens the SLR until the dealer stops buying and the market gaps](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-4.png)

The same plumbing explains the **September 2019 repo spike**, when the overnight repo rate — the rate at which Treasuries are pledged for cash overnight — briefly leapt from ~2% to nearly 10%. Banks that could have stepped in to lend cash against Treasury collateral chose not to, in part because doing so would have expanded their balance sheets against binding leverage and liquidity ratios at quarter-end (when SLR is measured for some purposes). Reserves had grown scarce after years of quantitative tightening, and the regulatory cost of redeploying them was too high. A funding market for the world's safest collateral seized up — again, not because the collateral was bad, but because the rules made intermediating it expensive. (For the mechanics of repo and the plumbing of overnight funding, see the cross-links below.)

## How higher capital lowers ROE and credit supply

We have seen *which* assets the rules favor. Now the deeper question: does requiring *more* capital overall shrink the amount of credit, and at what cost? The answer is yes — with a real and measurable price — and understanding the mechanism kills the most persistent myth in the whole debate.

A bank's **return on equity (ROE)** is profit divided by equity. Banks are highly leveraged: a small return on a huge asset base, magnified by thin equity, becomes a large return on that equity. Force the bank to hold more equity against the same assets, and you de-magnify the return. Mechanically, more capital lowers ROE.

There is a famous academic counterargument here, and it is worth stating fairly because it shapes the whole policy debate. The **Modigliani–Miller theorem** says that, in a frictionless world, a firm's funding mix (debt versus equity) does not change its total value: more equity makes the equity *safer*, so shareholders should rationally accept a *lower* required return, exactly offsetting the de-leveraging. If MM held perfectly, forcing banks to hold more capital would be costless — the lower ROE would come with proportionally lower risk and no change in the cost of credit. Academics like Anat Admati have argued, on this basis, that bank capital can be raised far more than the industry claims at little real cost. The industry counters that MM does *not* hold for banks because of two big frictions: the **tax shield** on debt (interest is tax-deductible, equity dividends are not, so debt is genuinely cheaper after tax) and the **implicit government guarantee** on bank debt (deposits are insured and big banks are too-big-to-fail, so their debt is artificially cheap). Those frictions are real, so more capital *does* raise a bank's funding cost — but by *less* than the naive ROE arithmetic suggests, because part of the ROE drop is offset by lower risk. The truth sits between the two camps: capital is not free, but it is not as expensive as banks claim. Hold both ideas at once when you read the lobbying.

#### Worked example: bank ROE before vs after a higher CET1 requirement

A bank holds \$1,000m of risk-weighted assets that earn a 1.2% return on assets (ROA) after costs — so \$12m of profit per year.

- **Before:** required capital is 8%, so equity = \$80m. ROE = \$12m ÷ \$80m = **15.0%**.
- **After:** the regulator raises the effective requirement to 12% (minimum plus buffers plus surcharge). Equity must rise to \$120m. If profit is unchanged at \$12m, ROE = \$12m ÷ \$120m = **10.0%**.

A 4-percentage-point rise in the capital requirement cut ROE by a third, from 15% to 10%. Shareholders who demanded 15% are not satisfied with 10%, so the bank must respond — and the intuition is that there are only two levers it can pull to defend ROE: charge more for credit, or make less of it.

How does the bank push ROE back up? It has three moves, and all three tighten credit:

1. **Raise loan spreads** — charge borrowers more, lifting profit per dollar of asset. The cost of the rule is passed to borrowers as a higher price of credit.
2. **Shrink the balance sheet** — lend less, cut the highest-RWA (highest-weight) assets first. The quantity of credit falls.
3. **Reshuffle toward low-weight assets** — swap loans for government bonds, which is the zero-weight bias again.

Empirically, the research consensus (the Basel Committee's own studies, plus academic work) is that each 1-percentage-point rise in capital requirements raises loan spreads by very roughly 5–15 basis points and trims credit volume modestly, mostly in the transition years. The effect is real but not catastrophic — and crucially, it must be weighed against the enormous benefit of fewer crises. The point is not that capital is bad; it is that capital is *not free*, and anyone who claims otherwise is ignoring the ROE arithmetic.

It also matters *where* the credit goes when it gets squeezed — and the answer is "out of the regulated banks and into the shadows." When capital rules make a loan too expensive for a bank to hold, the loan does not always vanish; often it migrates to a non-bank lender — a private-credit fund, a business-development company, an insurer — that is not subject to Basel and can hold the asset more cheaply against its own (lighter) capital. This is the great unintended consequence of tightening bank capital: it pushes credit creation into the **shadow banking** system, which is less regulated, less transparent, and outside the lender-of-last-resort safety net. Post-2010, private credit exploded from a niche to a multi-trillion-dollar asset class precisely as bank capital rules tightened — the two trends are not a coincidence. A market reader should always ask, when bank capital tightens: *where did the credit go?* The answer is usually "somewhere you can see less of it," which is a feature for the regulated banks and a worry for systemic stability. (See the shadow-banking cross-link.)

#### Worked example: the credit multiplier — how much credit \$1 of capital supports

Flip the ratio around to see the macro picture. If total capital must be at least 8% of RWA, then \$1 of capital supports at most \$1 ÷ 0.08 = **\$12.50** of risk-weighted assets — the reciprocal of the capital ratio is the leverage the system is allowed to run.

So \$1 of bank capital backs **\$12.50 of risk-weighted assets** at the 8% minimum. But RWA depends on the risk weight, so the *credit* that \$1 of capital supports depends on what the bank lends to:

- Against 100%-weight corporate loans: \$12.50 of RWA = **\$12.50 of loans** per \$1 of capital.
- Against 50%-weight mortgages: \$12.50 of RWA = **\$25 of mortgages** per \$1 of capital (because each \$1 of mortgage only generates \$0.50 of RWA).
- Against 0%-weight Treasuries: *unlimited* under the risk-weighted rule — \$1 of capital backs an infinite quantity of RWA, which is exactly why the leverage ratio exists to cap it.

Now raise the requirement to 12%: \$1 of capital supports only \$1 ÷ 0.12 = \$8.33 of RWA, down from \$12.50. The same banking-system capital creates a third less risk-weighted credit. The intuition: the capital ratio is the dial that sets the gear ratio between the system's equity and the credit it can manufacture — turn it up and the whole economy's credit-creation capacity turns down.

This is the macro hand. Lending creates money (the act of making a loan credits a deposit into existence), so a constraint on how much banks can lend is a constraint on how much money the system creates. Basel sits upstream of the money supply just as much as the central bank's interest rate does. (See the cross-link on how credit creates money.)

![Higher capital requirement lowers bank ROE which the bank defends by widening spreads shrinking lending or shifting to government bonds](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-5.png)

## Basel IV, the output floor, and the endgame debate

"Basel IV" is the nickname for the final batch of post-2008 reforms — formally still called Basel III, finalized in December 2017 and phasing in through the late 2020s. Banks call it Basel IV because of how much it bites. The headline change is the **output floor**.

Recall that big banks are allowed to use their own *internal models* to estimate risk weights (the "Internal Ratings-Based" or IRB approach), rather than the standardized weights we listed. The problem: banks systematically modeled their risks as *lower* than the standardized approach implied, shrinking their RWA and their required capital. Two banks holding identical assets could report wildly different RWA depending on how aggressive their models were — undermining the whole point of a common rulebook.

The **output floor** says: a bank's model-based RWA can be no less than **72.5%** of what the standardized approach would produce. It is a floor under model optimism. For banks that had pushed their internal models hard — many large European banks — the floor forces RWA *up*, which forces required capital up, which (per the ROE arithmetic above) tightens their credit. This is why the European banking lobby fought it for years, and why the US version (the "**Basel III Endgame**") became a political battle: the initial 2023 US proposal would have raised large-bank capital requirements by roughly 16–19%, and after intense industry pushback regulators signaled in 2024–2025 a substantially watered-down "re-proposal" with a much smaller increase. The *level* of the endgame is itself a market-moving variable — every percentage point of required capital is a percentage point of bank-lending capacity and bank ROE.

Why would a bank fight so hard over a few percentage points? Run the multiplier backward. If the Endgame raises a large bank's effective requirement from 12% to 14% of RWA, then \$1 of its capital now supports \$1 ÷ 0.14 = \$7.14 of RWA instead of \$1 ÷ 0.12 = \$8.33 — a 14% drop in the credit each dollar of capital can carry. To keep its ratio without raising new equity (dilutive and unpopular with shareholders), the bank must shrink RWA by that same 14%, which means either pulling back lending or pushing some businesses out the door. So a "small" 2-point rule change translates into a *double-digit* swing in lending capacity. That is the leverage hidden inside the ratio, and it is why every basis point of the Endgame is fought line by line through the public-comment process.

The output floor also reshapes *which* banks are most affected, which matters for relative trades. European banks, with their heavy reliance on internal models and low average risk weights, face the biggest RWA inflation from the floor — some by 20% or more — so the floor is a bigger drag on European than on US bank capital, all else equal. US banks were already closer to standardized weights, so the floor bites them less; their bigger fight is over the market-risk and operational-risk charges in the rest of the Endgame package. The upshot: the same global standard lands with very different force across regions, creating durable differences in bank profitability and lending capacity that show up in relative bank-equity performance.

#### Worked example: the capital cost of a trading position under the buffers

Consider a bank's trading desk holding a \$500m corporate-bond inventory with a market-risk RWA of \$400m (after the Basel "Fundamental Review of the Trading Book" charges). Apply the full effective CET1 stack for a G-SIB: 4.5% minimum + 2.5% conservation buffer + 1.0% CCyB + 2.0% G-SIB surcharge = **10% effective CET1**.

Required CET1 against this position = 10% × \$400m = **\$40m**. If the desk's shareholders demand a 12% return on that equity, the position must earn at least 12% × \$40m = **\$4.8m per year** just to cover its capital cost — about 0.96% of the \$500m notional, *before* funding and operating costs. Any trade that cannot clear that hurdle is not worth doing.

The intuition: the buffers turn capital into a per-trade hurdle rate, so a desk facing a 10% capital charge silently abandons every low-margin business that cannot clear its cost of equity — which is how a buffer in Basel quietly shrinks the range of markets a bank will make.

This is the *pro-cyclicality* problem in microcosm. In a downturn, asset values fall and risk models flash red, so RWA mechanically *rises* exactly when banks can least afford to raise capital. To restore their ratios, banks shed assets and cut lending into the teeth of the recession — amplifying the downturn. The countercyclical buffer was bolted on specifically to counter this: regulators *release* the buffer in a bust (lowering the requirement) so banks have room to keep lending. Whether they actually do is one of the central macroprudential questions of the next downturn, and one worth watching.

There is a behavioral wrinkle that blunts the buffers' design. Even when a regulator *releases* a buffer, banks are often reluctant to dip into the freed-up capital, because doing so signals weakness to the market — investors and analysts treat a bank that runs near its minimum as a problem bank, regardless of what the regulator permits. So the *usable* buffer in practice is smaller than the *official* buffer: banks self-impose a "management buffer" above the regulatory one and defend it even in stress. This is why, during the COVID shock in 2020, regulators had to do more than just release buffers — they had to actively encourage and sometimes shame banks into using them, and many banks still cut lending and hoarded capital. The lesson for a market reader: do not assume that a released buffer translates one-for-one into more lending. The capital a bank *will* deploy is the capital above its self-imposed market-credibility threshold, which can be a percentage point or two above the regulatory floor. Read the bank's stated CET1 *target*, not just its regulatory minimum, to gauge real headroom.

## Common misconceptions

**Myth 1: "More capital makes banks safer at no cost, so just require a lot of it."** The safety is real; the "no cost" is not. As the ROE worked example showed, raising the requirement from 8% to 12% cut a sample bank's ROE from 15% to 10%, and the bank claws that back by widening loan spreads (the consensus estimate is roughly 5–15bp per extra point of capital) and trimming credit. Capital is genuinely valuable — fewer crises are worth a lot — but it is a tax on credit, paid by borrowers in wider spreads and by the economy in slightly less lending. The honest framing is a *trade-off*, not a free lunch. (The opposite extreme — "capital is pure deadweight" — is equally wrong; a 2008-scale crisis costs the economy multiples of any plausible spread widening.)

**Myth 2: "All assets need the same capital."** They emphatically do not. A \$100m Treasury position requires \$0 of capital under the risk-weighted rule; a \$100m corporate-loan book requires \$8m at the 8% minimum; a \$100m book of past-due loans (150% weight) requires \$12m. The entire architecture is built on *differentiating* assets by risk weight — which is exactly what creates the zero-weight bias toward government bonds and away from small-business lending. If you think all bank assets are treated alike, you will misread every story about banks "de-risking" their balance sheets.

**Myth 3: "The leverage ratio is a technicality that doesn't matter for markets."** It is the single most important rule for Treasury-market and repo-market liquidity. Because the SLR counts risk-free Treasuries at full value, it — not the risk-weighted ratio — binds the biggest US dealers, and a binding SLR is what made dealers refuse to buy Treasuries in March 2020 (forcing ~\$1tn of Fed buying) and contributed to the September 2019 repo spike (overnight rates to ~10%). If you trade rates, the SLR is not a technicality; it is the fault line under the world's most important market.

**Myth 4: "Basel is a law, so it's the same everywhere."** Basel is a set of *standards* agreed by the Basel Committee on Banking Supervision — a body that sits at the Bank for International Settlements in Basel, Switzerland, and has no power to bind anyone. It is a *club of central banks and supervisors* that agrees on a common rulebook, which each member then writes into its own domestic law. That implementation step is where the divergence creeps in: the US adds the enhanced SLR and a slower, watered-down "Endgame"; the EU softens the output floor's bite and phases it in over a longer window; the UK runs its own "Strong and Simple" regime for small banks; emerging markets often adopt a simplified version. The *level* and *timing* differ enough to create cross-border arbitrage (book the business where the capital is cheapest) and very different bank-lending capacity by region — which is precisely why the divergence is tradeable.

**Myth 5: "If a bank passes its capital ratio, it's safe."** The ratio is a necessary condition, not a sufficient one. SVB met its regulatory capital ratios right up until the week it failed — its problem was *liquidity* and *interest-rate* risk, not the credit risk the capital ratio measures. A bank can be perfectly capitalized against the risks Basel measures and still be killed by a risk Basel measures poorly (duration, deposit flightiness, concentration). The ratio tells you the bank can absorb the losses the *model* anticipates; it tells you nothing about the loss the model didn't see coming. Treat a healthy capital ratio as table stakes, then look separately at funding stability, deposit concentration, and asset duration.

## How it shows up in real markets

**The 2019 repo spike.** On September 17, 2019, the secured overnight repo rate spiked from around 2% toward 10% intraday. The Fed had to inject reserves through emergency repo operations. A major contributing factor: years of quantitative tightening had drained bank reserves, and the banks holding the remaining reserves were reluctant to lend them out against Treasury collateral because doing so expanded their balance sheets against binding leverage and liquidity constraints — especially around quarter-end reporting dates. A capital-and-liquidity rule, interacting with a shrinking reserve supply, made the plumbing of the safest collateral market seize. (See the money-market-plumbing cross-link for the repo mechanics.)

**The March 2020 SLR episode.** As COVID hit, investors worldwide sold even Treasuries to raise dollars. Dealers' balance sheets ballooned with the bonds they were absorbing, the SLR slammed toward its 5% limit, and dealers pulled back. The Fed bought roughly \$1 trillion of Treasuries in three weeks and — tellingly — the regulators *temporarily exempted Treasuries and reserves from the SLR* (April 2020 to March 2021) to free up dealer capacity. The market's recovery tracked the policy response. The episode is the cleanest natural experiment we have that the leverage ratio sets Treasury-market liquidity.

**Bank-multiple compression.** Equity investors price banks on ROE relative to cost of equity. As Basel III raised effective requirements through the 2010s, large-bank ROEs fell from the pre-crisis 15–20% range toward 8–12%, and bank price-to-book multiples compressed accordingly — many large banks traded *below* book value for years, something almost unheard of pre-2008. A bank trading at 0.8× book is the market saying "this bank earns less than its cost of equity," and a large part of *why* it earns less is the capital it must now hold. When you see a regional or European bank at a deep discount to book, the capital regime is usually a big part of the story.

**The European sovereign-bank doom loop, 2011–2012.** The zero risk weight on home-government debt encouraged Italian, Spanish, and Greek banks to load up on their own sovereigns' bonds — capital-free under the rules. When those sovereigns' creditworthiness came into question, the banks' "risk-free" holdings cratered in value, gutting bank capital exactly as the governments most needed healthy banks to keep lending. The banks' weakness then forced the governments toward bailouts, deepening the sovereign-debt crisis, which further hit the banks — a loop. A single risk weight (0% on the home sovereign) helped turn a fiscal crisis into a banking crisis and back again. Europe is still debating whether to put a positive risk weight on concentrated sovereign holdings to break the loop; the politics are brutal because it would force banks to hold less of their own government's debt at exactly the time governments want them to hold more.

The interest-rate backdrop matters too, because Basel's liquidity rules pushed banks to hold long-duration Treasuries and reserves — and when the Fed raised rates fast in 2022–2023, those "safe" holdings lost market value, which is the wound SVB died of. Basel's HQLA definition treats a long-dated Treasury as a high-quality liquid asset for *liquidity* purposes, but it carries enormous *interest-rate* risk that the LCR does not penalize and that, for many mid-size banks, escaped the capital charge entirely through an accounting election. Capital and liquidity rules made the holdings look safe; the rate cycle made them dangerous. The lesson regulators are now absorbing is that "safe for one rule" is not "safe overall" — an asset can be liquid and risk-free for credit purposes while being a duration time-bomb, and the framework's siloed rules can miss the combination.

![Federal funds target rate upper bound stepping from 0.50 to 5.50 percent across the 2022 to 2024 hiking cycle](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-6.png)

![Required total capital and the credit multiplier across Basel risk weights from zero to one hundred fifty percent](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-7.png)

![Bank return on equity falling as the CET1 capital requirement rises from six to fourteen percent](/imgs/blogs/basel-iii-iv-bank-capital-rules-and-the-price-of-credit-8.png)

## How to trade it: the playbook

Capital rules are slow-moving and public, which makes them one of the cleaner things to position around — the rule change is announced years before it bites, and the repricing drifts in. Here is how a practitioner reads the framework.

**Read capital ratios for lending capacity.** A bank's CET1 ratio versus its requirement tells you its *headroom* — the buffer above the minimum it can deploy into new lending or return to shareholders. A bank sitting at 11% CET1 against a 10% requirement has 1 point of buffer; a bank at 14% has room to grow the loan book or buy back stock. Aggregate this across a banking system and you have a read on the *credit cycle's* capacity: when system-wide buffers are thin, expect tightening lending standards and wider credit spreads regardless of what the central bank's policy rate is doing. The capital ratio is a leading indicator of credit supply.

**Watch the SLR/leverage-ratio debate for Treasury-market liquidity.** Any signal that US regulators will exempt Treasuries and reserves from the SLR denominator (or otherwise relax it) is bullish for Treasury-market liquidity and for dealer balance-sheet capacity: it lets dealers warehouse more bonds, tightens repo spreads, and dampens the next dash-for-cash. Conversely, a tightening of leverage rules, or simply the *absence* of reform as Treasury supply keeps growing, leaves the March-2020 fault line in place. If you trade rates, swap spreads, or repo, the SLR headline is a first-order catalyst. Track Fed/OCC/FDIC proposals and Treasury-market-resilience reports.

**Trade the Basel III Endgame level.** The size of the final US capital increase is a direct input to large-bank ROE and lending capacity. A *higher* final number compresses bank ROE and multiples, favors lower-capital business models, and tightens credit at the margin; a *watered-down* number (as signaled in 2024–2025) is a relief rally for bank equity and supports credit supply. The repricing happens on each proposal, re-proposal, and comment-deadline — a regulatory calendar you can position around. (See the regulatory-calendar cross-link.)

**Read the buffers for the cycle.** When a national regulator *raises* the countercyclical buffer, it is leaning against a credit boom — a late-cycle signal. When it *releases* the buffer, it is trying to keep credit flowing in a bust. The buffer setting is a regulator's published opinion on where the credit cycle stands; treat it as a macro indicator.

**The position.** Long banks with high capital buffers and clean balance sheets when the regulatory tide is *easing* (Endgame watered down, SLR reform, buffer release) and credit is cheap; underweight thinly-capitalized banks into a *tightening* regime; in a leverage-binding episode (quarter-end stress, a funding scare), expect repo and swap-spread dislocations and the Fed to backstop them. Pair any rates-liquidity view with the SLR state.

**What invalidates the view.** The capital-rules thesis breaks when: (1) regulators reverse course — a watered-down Endgame that suddenly re-tightens, or an SLR reform that gets pulled; (2) the binding constraint shifts — if a bank's risk-weighted ratio, not its leverage ratio, becomes the binding one, the Treasury-liquidity logic flips; (3) the central bank floods the system with reserves (QE), which loosens the leverage-and-liquidity bind temporarily and masks the underlying fragility; or (4) banks raise fresh equity, expanding capacity and resetting the headroom math. Always check *which* ratio is binding before trading the implication — the whole playbook hinges on it.

The throughline: Basel is a single rulebook, written by central bankers, that prices every asset a bank can hold and thereby sets the cost and quantity of credit for the whole economy. It decides how much the system can lend, how leveraged its dealers can be, and where liquidity vanishes in a crisis. It is, quite literally, a hidden hand on the price of money — and once you can read the ratios, you can read the credit cycle a step ahead of the central bank's rate.

## Further reading & cross-links

- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine: how any rule change discounts into prices before it bites.
- [The legal mandate of a central bank](/blog/trading/law-and-geopolitics/the-legal-mandate-of-a-central-bank) — the statute behind the regulator that writes (and enforces) capital rules.
- [Deposit insurance, lender of last resort, and the anatomy of a bank run](/blog/trading/law-and-geopolitics/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run) — the SVB 2023 case study and the liquidity side of bank fragility.
- [Dodd-Frank: the post-2008 rulebook](/blog/trading/law-and-geopolitics/dodd-frank-the-post-2008-rulebook) — the US legislative scaffolding that Basel III plugs into (stress tests, the SLR enhancement, living wills).
- [The regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock) — how to position around proposal and comment-deadline dates like the Basel III Endgame.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the non-bank plumbing that absorbs what capital-constrained banks won't.
- [Money-market plumbing: repo, collateral, and SOFR](/blog/trading/macro-trading/money-market-plumbing-repo-collateral-sofr) — the mechanics behind the 2019 repo spike.
- [How credit creates money: the lending channel](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles) — why a constraint on lending is a constraint on the money supply.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the policy rate that sits alongside capital rules as the second hand on the price of money.

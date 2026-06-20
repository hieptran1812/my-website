---
title: "Bank Capital and Leverage: Why Equity Is the Thin Cushion"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a bank funds itself with about ten times more borrowed money than its own, why that leverage means a small fall in asset value can wipe it out, and why equity is the thin cushion standing between a bank and insolvency."
tags: ["banking", "bank-capital", "leverage", "equity", "insolvency", "balance-sheet", "lehman-brothers", "silicon-valley-bank", "basel", "loss-absorption"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank funds roughly 90% of its assets with other people's money and only about 10% with its own equity, so it runs at around 10x leverage; that thin equity layer is the only thing absorbing losses before depositors and creditors take the hit.
>
> - Leverage is the multiplier: leverage = total assets ÷ equity. At 10x, the bank controls \$10 of assets for every \$1 of its own money. The same leverage that magnifies returns magnifies losses.
> - A bank at 10x leverage is wiped out by roughly a 10% fall in the value of its assets; at 30x it takes only about a 3.3% fall. Lehman Brothers ran at 30.7x.
> - Equity is a *cushion*, not a *reserve* — it is not a pile of cash sitting in a vault. It is the share of the assets that the owners, not the lenders, have a claim on, and it is the buffer that takes the first loss.
> - The one number to remember: a normal company might run at 1.5x–3x leverage; a bank runs at roughly 10x, and that single fact explains why banks fail in ways normal firms never do.

On the morning of 15 September 2008, Lehman Brothers filed for bankruptcy. At the end of 2007 — just twenty months earlier — Lehman had reported total assets of about \$691 billion supported by reported equity of around \$22 billion. That is leverage of roughly 30 to 1. Read that again slowly: for every dollar of its own money, Lehman was holding about thirty dollars of assets, and the other twenty-nine dollars were borrowed. When the value of those assets — mostly mortgage-related securities — fell by a few percent, there was no longer enough equity underneath them to make the firm worth more than it owed. A drop of a little over 3% in the value of its book was, arithmetically, enough to erase the entire company. Markets did the rest.

Now fast-forward to March 2023. Silicon Valley Bank looked nothing like a swashbuckling Wall Street trading house. It was a deposit-funded commercial bank lending to tech startups, and it had loaded up on safe, boring U.S. government bonds. But when interest rates rose sharply through 2022, the market value of those long-dated bonds fell, and by the end of 2022 SVB was sitting on roughly \$17 billion of unrealized losses on its securities — against common equity of about \$16 billion. The losses were quietly larger than the cushion meant to absorb them. Once depositors realised this, \$42 billion of deposits walked out the door in a single day. The bank was gone within 48 hours.

Two completely different banks, two completely different decades, one identical wound. Both were brought down not by a catastrophe but by a *small* move in the value of their assets — small relative to the assets, fatal relative to the sliver of equity beneath them. This post is about that sliver: what bank capital actually is, why banks run at leverage levels that would be reckless for almost any other business, and why equity is the thin cushion that decides, in the end, whether a bank lives or dies. The figure above is the mental model to keep in your head the whole way through: a tall block of assets, funded mostly by deposits and debt, with equity as a thin green strip at the top.

![A bank balance sheet with assets on the left and deposits debt and a thin equity sliver on the right](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-1.png)

## Foundations: equity, capital, leverage, and the thin cushion

Before we can talk about why a 3% fall matters, we need four words defined from absolute zero. None of them mean what everyday language suggests, and getting them wrong is the single most common source of confusion about banks.

### A balance sheet, in one sentence

Every business has a *balance sheet* — a snapshot, on a given day, of what it owns and what it owes. The left side lists **assets**: everything the business owns or is owed (cash, buildings, inventory, and for a bank, loans and securities). The right side lists **liabilities** (everything the business owes to others) plus **equity** (the owners' residual claim). The two sides always balance, by construction, because of a single accounting identity:

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

Here every symbol is a dollar total on a given date: *Assets* is what the firm owns, *Liabilities* is what it owes to outsiders, and *Equity* is what is left for the owners after the outsiders are paid. Rearranged, this identity tells you exactly what equity *is*:

$$\text{Equity} = \text{Assets} - \text{Liabilities}$$

Equity is not a thing the bank holds. It is a *difference* — the value of the assets minus the value of the debts. If the assets are worth more than the debts, the difference is positive and the owners have something. If the assets fall in value until they are worth less than the debts, the difference goes negative, and the firm is *insolvent*: it owes more than it owns. Hold onto that. Equity is the gap, and insolvency is what happens when the gap closes.

### What "capital" means (and what it does not)

In banking, the word **capital** is used almost interchangeably with **equity**, and this trips up nearly everyone. When a regulator says "the bank must hold more capital", a beginner pictures the bank stashing more cash in a safe. That is wrong. Capital is *not* an asset the bank parks somewhere; it is a *funding source* on the right-hand side of the balance sheet. Holding more capital means funding more of your assets with the owners' money (equity) and less with borrowed money (debt and deposits).

Think of it this way. Suppose you buy a \$500,000 house. You can fund it with a \$50,000 deposit (your own money) and a \$450,000 mortgage (borrowed). Your "capital" in the house is the \$50,000 of equity. If the regulator told you to "hold more capital", they would not mean "keep more cash in your wallet" — they would mean "put more of your own money down and borrow less". More capital = a bigger equity slice = less leverage. That is the entire idea, and it is why "raising capital" (issuing new shares) and "holding capital" (funding with equity) are two faces of the same coin.

### Leverage: the multiplier

**Leverage** is the ratio of total assets to equity:

$$\text{Leverage} = \frac{\text{Total assets}}{\text{Equity}}$$

The leverage ratio (here, the simple assets-to-equity multiple) tells you how many dollars of assets the firm controls for each dollar of its own money. A firm with \$100 of assets and \$10 of equity is at 10x leverage: it controls ten dollars of stuff for every one dollar that is genuinely its own. The other \$90 is borrowed.

Leverage is a multiplier in both directions. If your \$100 of assets earns 1%, that is \$1 — but \$1 on your \$10 of actual equity is a 10% return. Leverage turned a 1% asset return into a 10% equity return. That is the magic that makes banking profitable on razor-thin margins. But run the film backwards: if your \$100 of assets *loses* 1%, that is a \$1 loss — and \$1 against your \$10 of equity is a 10% hit to your own money. Leverage magnifies losses by exactly the same factor it magnifies gains. There is no free lunch hiding in the multiplier.

#### Worked example: computing the leverage multiple

Let's make this concrete. A bank reports \$1,000 billion (\$1 trillion) in total assets and \$80 billion in equity. What is its leverage?

$$\text{Leverage} = \frac{\$1{,}000\text{bn}}{\$80\text{bn}} = 12.5\times$$

So this bank controls \$12.50 of assets for every \$1 of its own money. Put differently, equity is \$80bn out of \$1,000bn, which is 8% of the balance sheet. Notice the relationship: the equity *share* of the balance sheet is just one divided by the leverage multiple. 1 ÷ 12.5 = 0.08 = 8%. Whenever someone tells you a bank "has an 8% capital ratio", they are telling you, in disguise, that it runs at about 12.5x leverage. The two numbers are the same fact wearing different clothes.

*Intuition:* leverage and the equity share are reciprocals — a thin equity slice and a high leverage multiple are literally the same sentence said two ways.

### The thin cushion: equity as the first loss-taker

Now the part that matters. Why does the size of equity decide whether a bank survives? Because of a hard rule about *who takes losses first*. When a bank's assets lose value — a loan goes bad, a bond falls in price — that loss has to land on *someone*. The accounting identity forces it: if assets fall and liabilities (what you owe depositors and lenders) stay fixed, then equity, the difference between them, must shrink by the full amount of the loss.

This is the cushion. Losses eat equity first, dollar for dollar. As long as there is equity left, depositors and lenders are made whole — they are still owed exactly what they were owed, and the assets are still worth at least that much. The owners absorb the pain. But the moment cumulative losses exceed the equity cushion, the assets are worth less than the debts, equity goes negative, and the loss spills past the owners onto the creditors and depositors. That is insolvency. Equity is, quite literally, the thickness of the buffer between "the owners lose money" and "the depositors lose money".

This is why we call equity the *loss-absorbing* layer, and why a regulator's obsession with capital is really an obsession with how much loss a bank can eat before the people who trusted it with their money get hurt.

### Why the order of losses is fixed: the hierarchy of claims

The reason losses fall on equity *first* and depositors *last* is not an accident of accounting; it is the legal **hierarchy of claims**. When a firm is wound up, its assets are paid out to claimants in a strict priority order, like a queue. At the front of the queue are secured creditors and depositors (and the deposit insurer standing in for them); behind them are senior bondholders; then subordinated bondholders; and dead last, at the very back of the queue, are the equity shareholders. Whoever is last in the queue is, by definition, first to take losses — because losses are just the queue running out of money before it reaches the back. Equity is last in line for payouts, which is exactly why it is first in line for losses. The shareholders signed up for that deal: in exchange for taking the first loss, they get all the upside above what is owed to everyone ahead of them. This is the bargain at the core of what equity *is*, in a bank or any other firm — it is the residual, the part that gets whatever is left after everyone else is paid, which is wonderful when assets are growing and catastrophic when they fall.

## Why a bank is a leverage machine (and your dentist is not)

A normal, healthy company — a software firm, a restaurant chain, a manufacturer — typically funds itself mostly with equity and a modest amount of debt. Its leverage might be 1.5x, 2x, maybe 3x in a debt-heavy industry. A bank routinely runs at 10x or more. Why is a bank built so differently? The answer is the single most important structural fact in all of banking, and it traces straight back to this series' spine.

### Deposits are debt — and that is the whole business

When you put \$10,000 in a bank, you experience it as "saving money". From the bank's point of view, you have *lent it \$10,000*. Your deposit is a liability of the bank — money it owes you, repayable on demand. A bank's primary product is gathering these deposits cheaply and lending them out at a higher rate, earning the spread. (We cover the spread itself in [net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained); here we only care that deposits are *debt*.)

This is what makes a bank a leverage machine by *design*. Its core business is borrowing — borrowing from millions of depositors — and using that borrowed money to buy assets (loans and securities). A bank with \$74 of deposits, \$18 of other debt, and \$8 of equity funding \$100 of assets is not being reckless; it is doing exactly the job it exists to do. The deposits are the raw material. The leverage is not a bug bolted onto the business; the leverage *is* the business. This is why you cannot understand a bank by analogy to a normal firm. A dentist who borrowed thirty times her own money to buy equipment would be insane; a bank that does the same with depositor money is, depending on the assets, simply a bank.

#### Worked example: a bank with 10% equity faces a 5% loan loss

Here is the cushion working in slow motion. A bank has \$100 billion of assets funded by \$90 billion of deposits and debt and \$10 billion of equity. That is exactly 10x leverage and a 10% equity ratio. Now a recession hits and 5% of its assets turn out to be worthless — \$5 billion of loans default and recover nothing.

Assets fall from \$100bn to \$95bn. The \$90bn it owes depositors and lenders does not change one cent — they are owed what they are owed. So:

$$\text{New equity} = \$95\text{bn} - \$90\text{bn} = \$5\text{bn}$$

Equity has gone from \$10bn to \$5bn. A 5% loss on assets caused a 50% loss of equity. The bank is still solvent — depositors are still fully covered, assets exceed debts — but half its cushion is gone in one blow, and its leverage has *risen* to \$95bn ÷ \$5bn = 19x, making it far more fragile against the next loss.

*Intuition:* at 10x leverage, every 1% of asset losses destroys 10% of equity — so a "small" 5% asset loss is a catastrophic 50% equity loss.

![Solvent bank with equity absorbing a loss next to the same bank insolvent after a larger loss](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-3.png)

### The same fragility, viewed as a fall in asset value

That worked example used a loss expressed as a percentage of assets. There is an even cleaner way to see the danger: ask how far the assets have to fall in value to erase *all* the equity. The answer is beautifully simple. If equity is a fraction $1/L$ of the balance sheet (where $L$ is leverage), then a fall of $1/L$ in asset value wipes out the entire equity cushion.

$$\text{Asset fall that wipes equity} = \frac{1}{\text{Leverage}}$$

At 10x leverage, equity is 1/10 = 10% of assets, so a 10% fall in asset value erases it. At 20x, only a 5% fall. At 30x, a fall of just 1/30 ≈ 3.3% is enough. The more leverage, the smaller the asset move that destroys the bank. This is not a metaphor; it is arithmetic, and it is the reason high leverage and fragility are the same thing. The chart below shows the relationship across leverage levels — notice how steeply the survivable cushion shrinks as leverage climbs.

![Bar chart of the asset fall that wipes out equity at five ten twenty and thirty times leverage](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-2.png)

#### Worked example: 30x leverage means a 3.3% asset fall = insolvency

Take an investment bank with \$300 billion of assets and \$10 billion of equity — 30x leverage, roughly Lehman's profile. How big a fall in the value of its \$300bn book wipes out the \$10bn cushion?

$$\text{Asset fall} = \frac{\$10\text{bn}}{\$300\text{bn}} = 0.0333 = 3.33\%$$

A 3.33% decline in the value of the assets — \$10bn of losses — and equity is gone. To feel how tiny that is: a portfolio of mortgage securities can easily move 3% in a bad week. The firm did not need a depression; it needed a slightly-worse-than-average month in the wrong assets. And here is the second-order trap: as losses mount and equity shrinks, leverage *rises*, so each subsequent percentage-point fall does even more damage. A bank near the edge is on a slope that gets steeper as it slides.

*Intuition:* at 30x leverage the entire margin for error is a 3.3% asset fall — less than the normal weekly wobble of a risky bond book.

### Why this is the series' spine

Step back. A bank borrows short (deposits, repayable on demand) and lends long (loans and bonds, locked up for years), earning the spread. To make that spread big enough to be a real business on assets that yield only a few percent, it must be heavily leveraged — equity has to be a small slice or the returns are too thin to bother. But the very leverage that makes the spread worth earning is what makes the bank fragile: it survives only as long as its thin equity cushion absorbs losses faster than they arrive, *and* only as long as depositors keep trusting it enough not to ask for their money all at once. Capital is the answer to the first fragility (can it absorb losses?); liquidity is the answer to the second (can it meet withdrawals?). This post is the capital half of that fragile trade. Everything else in banking — credit underwriting, provisioning, Basel rules, stress tests — exists to protect, measure, or rebuild this one thin cushion.

## The other side of leverage: why banks want the thin cushion

We have spent this far dwelling on the danger of leverage. To understand why banks run thin cushions *deliberately* — and why shareholders and management often push against holding more capital — we need to look at the side of the trade that pays. Leverage is not just a fragility; it is the engine of a bank's return on equity, and that tension between return and safety is the political and economic heart of every capital debate.

### The leverage identity: how a thin spread becomes a fat return

Banks earn a tiny margin on a huge balance sheet. The return on the *assets* — what the bank earns on every dollar of loans and securities it holds, after costs and losses — is small. A well-run bank earns a **return on assets (ROA)** of around 1%. One percent! On its own that would be a dismal business; almost any normal company earns far more on its assets. The trick that turns a 1% ROA into an attractive business is leverage, captured in the central identity of bank profitability:

$$\text{ROE} = \text{ROA} \times \text{Leverage}$$

Here *ROE* is return on equity (profit divided by the owners' equity), *ROA* is return on assets (profit divided by total assets), and *Leverage* is assets divided by equity. The identity is true by simple algebra — the assets cancel — but its meaning is profound: a bank converts a thin return on a big asset base into a respectable return on a small equity base by stacking enough assets on top of each dollar of equity. This is the same DuPont decomposition you would apply to any firm, applied to a bank, and it is the subject of [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) in full; here we need only the punchline that leverage is what makes the ROE worth having.

#### Worked example: turning a 1% ROA into a 12% ROE

A bank earns a 1% return on its \$1,000 billion of assets — that is \$10 billion of profit. Its equity is \$80 billion (so leverage is 12.5x). What is its return on equity?

$$\text{ROE} = \frac{\$10\text{bn profit}}{\$80\text{bn equity}} = 12.5\%$$

Or, using the identity directly: ROE = 1% × 12.5 = 12.5%. The bank earned a feeble 1% on its assets but a healthy 12.5% on its owners' money, and the entire gap was manufactured by leverage. Now suppose a regulator forces the bank to hold twice as much equity — \$160 billion, halving leverage to 6.25x. The same \$10bn of profit now divides into a bigger equity base: ROE = \$10bn ÷ \$160bn = 6.25%. The bank is far safer (a 16% asset fall now needed to wipe it out, versus 8% before) but its return to shareholders has halved.

*Intuition:* leverage is the lever that turns banking's tiny asset margin into a real equity return — which is exactly why shareholders resist holding more capital, even though more capital is what keeps the bank alive.

### The eternal tug-of-war: shareholders want it thin, regulators want it thick

This is the conflict that animates the entire history of bank capital regulation. Shareholders and management, paid in part on ROE, have a standing incentive to run the cushion as thin as the rules allow, because a thinner cushion (higher leverage) mechanically lifts ROE. Regulators, charged with protecting depositors and the financial system, want the cushion thick, because a thicker cushion absorbs more loss before anyone else gets hurt. Left entirely to its own incentives, a bank will tend to lever up until something stops it — which is precisely why minimum capital requirements exist. A bank's reported ROE can be flattered simply by shrinking the denominator (equity), and a high ROE achieved by high leverage is *more fragile*, not more impressive, than the same ROE achieved on a thicker cushion. This is the single most important thing to remember when comparing two banks' returns: ask how much of the ROE is real profitability and how much is just borrowed risk.

### Buybacks and dividends: how a bank thins its own cushion on purpose

There is a subtle, important way banks actively shrink the cushion even in good times: returning capital to shareholders through dividends and share buybacks. Every dollar a bank pays out as a dividend or uses to buy back its own stock is a dollar of equity leaving the balance sheet — the cushion gets thinner by exactly that amount, and leverage ticks up. In normal times this is healthy capital management; a bank that retains every dollar of profit forever would become over-capitalised and earn a poor ROE. But it means the cushion is not a passive thing that only shrinks when losses hit — management is constantly *choosing* how thick to keep it, trading safety against shareholder returns. When a regulator suspends a bank's buybacks during a stress test (as the Fed has done in crises), what they are really doing is forcing the bank to *stop thinning its own cushion* until the danger passes.

#### Worked example: a buyback thins the cushion

A bank has \$100 billion of assets and \$10 billion of equity — 10x leverage, a 10% cushion. It earns \$1.2 billion in a year and decides to return all of it to shareholders: \$0.5bn in dividends and \$0.7bn in buybacks. Assume assets stay flat. Equity falls from \$10bn to \$10bn − \$1.2bn + \$1.2bn earned = it would have *grown* to \$11.2bn if it retained everything, but by paying it all out, equity stays at \$10bn. Now suppose instead the bank buys back \$1.2bn of stock *beyond* its earnings, drawing the cushion down to \$8.8bn while assets hold at \$100bn.

Leverage rises from 10x to \$100bn ÷ \$8.8bn ≈ 11.4x, and the asset fall that wipes the bank drops from 10% to 1 ÷ 11.4 ≈ 8.8%. The shareholders got cash and a higher ROE; the depositors got a thinner cushion and a more fragile bank. Nothing illegal happened — this is ordinary capital return — but it shows that the cushion is a dial management turns, and turning it toward shareholders turns it away from safety.

*Intuition:* dividends and buybacks are deliberate cushion-thinning; the same act that rewards shareholders quietly raises the bank's leverage and lowers its margin for error.

## How losses actually flow: the absorption waterfall

Saying "losses eat equity" is the headline, but the mechanism has more steps, and understanding the order tells you a lot about how banks try to stay alive. Losses are absorbed in a strict sequence, like water cascading down a set of ledges. Only when one ledge overflows does the water reach the next.

### Step 1: this year's profit takes the first hit

Before a loss touches the equity cushion built up over years, it first eats into *this year's* earnings. A bank that earns \$8 billion of pre-provision profit in a year can absorb \$8 billion of loan losses that year and simply report zero profit — the cushion is untouched. This is why a profitable bank is a resilient bank: ongoing earnings are the first line of defence, refilling the bucket as fast as losses drain it. The income statement (covered in [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions)) is, in this sense, the cushion's daily top-up.

### Step 2: loan-loss reserves, the pre-funded buffer

Banks do not wait for loans to default to recognise losses. They estimate expected losses in advance and set aside a **loan-loss reserve** (also called the allowance for credit losses) — a contra-asset that pre-funds the expected bad debt. When a loan actually goes bad, the loss is charged against this reserve, not directly against equity. The reserve is itself rebuilt out of earnings through a **provision** expense on the income statement. So a bank that saw trouble coming and built a fat reserve has a second cushion *ahead* of equity. It is when losses run *past* the reserve — when reality is worse than the estimate — that they start chewing into capital.

### Step 3: equity absorbs the rest

Once this year's earnings are exhausted and the reserve is depleted, the remaining loss lands on the accumulated equity cushion — the retained earnings and paid-in capital of years past. This is the real buffer, and it is what regulators measure. Each dollar of loss past the first two ledges is a dollar straight off the owners' stake.

### Step 4: past equity, the creditors and depositors

When equity hits zero and keeps going, the loss has nowhere left to land but on the people the bank owes. In an orderly world, this happens through resolution: the bank is closed, the deposit insurer steps in to make insured depositors whole, and bondholders and uninsured depositors take losses according to the hierarchy of claims. This is the line that capital exists to keep the loss above. The whole point of the cushion is to make step 4 a rare event.

![Pipeline showing losses absorbed by earnings then reserves then equity then creditors](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-6.png)

#### Worked example: walking a loss down the waterfall

A bank has \$6 billion of annual pre-provision profit, \$10 billion of loan-loss reserves, and \$50 billion of equity. A severe recession produces \$20 billion of credit losses in one year. Where does the loss land?

First, the \$6bn of this year's profit absorbs \$6bn → \$14bn of loss remains. Next, the \$10bn reserve absorbs \$10bn → \$4bn remains. Finally, the residual \$4bn hits equity: \$50bn − \$4bn = \$46bn. The depositors and bondholders are never touched. The bank's leverage rises modestly, its capital ratio dips, but it is comfortably solvent — it had three full ledges of cushion before a single dollar reached the danger zone.

Now redo it with a thinly-capitalised bank: \$1bn of profit, \$2bn of reserves, \$5bn of equity, hit by the same \$20bn loss. Profit absorbs \$1bn → \$19bn left; reserves absorb \$2bn → \$17bn left; equity absorbs \$5bn and is *exhausted* with \$12bn of loss still unaccounted for. That \$12bn falls on creditors and depositors. The bank is insolvent by \$12bn. Same loss, totally different outcome — because the cushion was thin.

*Intuition:* it is not the size of the loss that kills a bank, it is the size of the loss *relative to the cushions stacked in front of equity* — earnings, reserves, then capital.

## Book capital versus regulatory capital: two different rulers

So far we have spoken of "equity" as if it were a single, clean number. In reality there are (at least) two important measures of a bank's capital, and they can differ by a lot. Confusing them is how analysts get blindsided.

### Book equity: the accounting number

**Book equity** is the equity figure on the audited balance sheet: paid-in capital plus retained earnings, adjusted for a few accounting items. It is what the accountants say the owners' stake is worth, using the values the bank carries its assets at. The trouble is that not every asset is carried at its current market value. A bank can hold a bond it bought for \$100 and still carry it at \$100 on the books even after rates rise and its market value drops to \$85 — *if* it classifies that bond as "held to maturity" and intends to hold it to the end. The \$15 of loss is real but does not show up in book equity. This is exactly the gap that hid inside SVB.

### Regulatory capital: the supervisor's number

**Regulatory capital** is the version banking supervisors care about, defined under the Basel framework. It starts from book equity but makes its own adjustments — deducting things that would not actually absorb losses in a crisis (like goodwill and some intangibles), and capturing some losses that book accounting lets banks defer. Crucially, regulators do not measure capital against *total assets*; they measure it against **risk-weighted assets** (RWA) — assets scaled by how risky each is, so a safe government bond carries less required capital than a risky corporate loan. We deliberately do not derive RWA or the Basel ratios here; that is the job of [risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) and the deeper [Basel I, II, III and the capital rules](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank). For now, the point is only that "capital" measured by accountants and "capital" measured by regulators are two different rulers, and a bank can look fine on one while being fragile on the other.

### Capital is not all the same quality

There is one more layer of subtlety that we will preview here and leave to the Basel post to derive fully. Not every form of capital absorbs losses equally well. Regulators rank capital into quality tiers:

- **Common Equity Tier 1 (CET1)** — ordinary shares plus retained earnings. This is the purest, most reliable loss-absorber: it takes losses first, always, while the bank is still a going concern. When people talk about "the cushion", they overwhelmingly mean CET1.
- **Additional Tier 1 (AT1)** — perpetual bonds (often called CoCos, contingent convertibles) that convert to equity or get written down if CET1 falls below a trigger. They absorb losses, but only once the bank is in trouble. The Credit Suisse collapse in 2023 made AT1 famous when \$17 billion of these bonds were wiped to zero.
- **Tier 2 (T2)** — subordinated debt that ranks above equity but below ordinary creditors. It mostly absorbs losses in a wind-down, not while the bank is still operating.

![Matrix of CET1 AT1 and Tier 2 capital tiers with what each is and when it absorbs losses](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-8.png)

The takeaway for this post: when you read that a bank has "12% capital", always ask *which* capital. A bank with 12% total capital but only 7% CET1 has far less of the good, always-on, going-concern loss-absorbing kind than the headline suggests. The thin cushion has layers, and only the bottom layer is truly always there.

#### Worked example: book capital versus mark-to-market capital

A bank reports \$100 billion of book equity. It also holds \$300 billion of bonds it bought when rates were low, classified as held-to-maturity and carried at cost. Rates have since risen, and the current market value of those bonds is \$270 billion — a \$30 billion unrealized loss that book accounting lets the bank keep off its equity line.

On paper, equity is \$100bn. But if you marked everything to market, true economic equity is \$100bn − \$30bn = \$70bn. The bank's reported leverage might look like a comfortable 12x, but its *economic* leverage is far higher, and its real cushion is 30% thinner than the headline. If depositors ever force the bank to sell those bonds (turning the paper loss into a realised one), the \$30bn hole becomes visible and the cushion shrinks for real. This is not theoretical — it is the precise mechanism, covered in depth in [interest rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap), that turned SVB's paper losses into a death spiral.

*Intuition:* the cushion that matters in a crisis is the *mark-to-market* one, and a bank can look well-capitalised on the accountants' ruler while being dangerously thin on the market's.

## Rebuilding the cushion — and why it is hardest exactly when you need it

A cushion that only ever shrinks would be a doomed defence. Banks rebuild capital constantly, and understanding the three ways they do it — and why each one fails at the worst moment — explains a great deal about how crises actually unfold.

### The three ways to rebuild capital

The first and healthiest way is **retained earnings**: a profitable bank simply keeps some of its profit instead of paying it all out, and the retained dollars add directly to equity. A bank earning a 12% ROE that pays out half its earnings is organically thickening its cushion by roughly 6% of equity a year. This is slow but free and reliable — in good times. The catch is that it depends on the bank being profitable, and banks are least profitable precisely when losses are mounting.

The second way is **raising new equity**: issuing new shares to investors and adding the proceeds to the cushion. This can thicken capital fast, by billions in a single offering. But it dilutes existing shareholders (more shares means each old share owns a smaller slice), which is why management hates doing it — and crucially, it is *cheap and easy when you don't need it* and *expensive or impossible when you do*. A healthy bank can raise equity at a good price; a bank that the market suspects is in trouble faces a brutal choice: issue shares at a crushed price (signalling weakness and diluting heavily) or not issue at all. This timing trap is at the centre of nearly every bank failure.

The third way is **shrinking the balance sheet** — selling assets or refusing to make new loans, so that the same equity covers a smaller asset base, lifting the capital ratio by shrinking the denominator. This is *deleveraging*, and it works arithmetically, but it has a vicious side effect: if many banks deleverage at once, they dump assets into a falling market (depressing prices further, creating more losses for everyone holding those assets) and choke off lending to the real economy (deepening the recession that caused the losses). This is the **fire-sale and credit-crunch** dynamic that turns one bank's problem into everyone's.

#### Worked example: raising equity before versus during a crisis

A bank with \$10 billion of equity needs to add \$2 billion of capital to restore its cushion. In calm times its shares trade at \$50, so it issues 40 million new shares (40m × \$50 = \$2bn). Painful but manageable.

Now the same bank tries to raise that \$2 billion *after* the market has lost confidence and the stock has fallen to \$10. It now needs to issue 200 million shares (200m × \$10 = \$2bn) — five times the dilution. Existing shareholders' ownership is gutted, the offering itself screams distress and can accelerate the very run it was meant to prevent, and investors may simply refuse to participate at any price. This is exactly the wall SVB hit: its attempt to raise \$2.25 billion of fresh capital in March 2023, far from reassuring the market, confirmed that something was badly wrong and triggered the run that killed it within two days.

*Intuition:* the cheapest moment to thicken the cushion is when you don't need to, and the moment you visibly need to is the moment the market makes it ruinously expensive or impossible — so capital must be built *before* the storm, not during it.

### Why regulators set a *floor*, and a leverage backstop

This timing trap is the deepest reason capital is *regulated* rather than left to bank management. Because raising capital is hardest exactly when it is most needed, and because thin cushions privatise the upside (shareholder ROE) while socialising the downside (taxpayer and depositor losses), supervisors impose minimum capital requirements that banks must hold *at all times*, in calm — so the cushion is already thick when the storm arrives. The modern framework, born from the failures this series studies, sets a minimum amount of capital as a percentage of risk-weighted assets, plus extra buffers that can be drawn down in stress and must be rebuilt afterward.

It also includes a deliberately crude backstop: a simple **leverage ratio** that caps total assets relative to capital *regardless of risk weights* — typically requiring capital of at least 3% of total assets (higher for the biggest banks), which is a hard ceiling on leverage of roughly 33x and, for systemic banks, closer to 20x or below. The leverage ratio exists precisely because the risk-weighted measure can be gamed — a bank can load up on assets that regulators deem "low risk" and run enormous leverage while still showing a healthy risk-weighted ratio. The crude backstop catches that. We deliberately stop here; the full derivation of the ratios, the buffers, the risk weights, and the leverage backstop is the work of [Basel I, II, III and the capital rules](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) and [risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work). The point for this post is only the *why*: the rules exist to force the cushion to be thick in calm weather, because no bank can reliably thicken it in a storm.

## The scale problem: thin cushions on enormous balance sheets

There is a reason bank fragility frightens regulators in a way that, say, a leveraged hedge fund usually does not: the balance sheets are colossal, and millions of ordinary people's savings sit on the liability side. A few percent of a number that large is an enormous amount of money, and the equity standing behind it is a thin slice of that same enormous number.

The biggest banks in the world hold assets measured in *trillions* of dollars. ICBC in China sits around \$6.3 trillion; JPMorgan Chase around \$4.0 trillion; Bank of America around \$3.3 trillion. At a typical capital ratio, the equity behind a \$4 trillion balance sheet is only a few hundred billion dollars — a number that sounds vast until you remember it is the cushion for *four thousand* billion of assets. The chart below shows the scale; mentally shave off the top few percent of each bar and you have the entire equity cushion for the whole institution.

![Horizontal bar chart of the largest global banks by total assets in trillions of dollars](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-7.png)

This scale is why the leverage math is not an academic curiosity. When a bank with \$4 trillion of assets runs at 10x leverage, a coordinated 10% fall in asset values across its book — the kind of thing a severe systemic crisis can produce — is a \$400 billion loss, which is its entire equity. That single institution failing can take counterparties, payment systems, and depositor confidence down with it. The thinness of the cushion, multiplied by the size of the balance sheet, is precisely what "too big to fail" is made of. (The deposit-insurance and lender-of-last-resort backstops that exist because of this are covered in [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).)

#### Worked example: the equity behind a trillion-dollar bank

A global bank has \$2 trillion of assets and runs at a 7% CET1-to-total-assets leverage ratio (a simple, non-risk-weighted measure). How much equity is that, and how big a loss erases it?

$$\text{Equity} = 0.07 \times \$2{,}000\text{bn} = \$140\text{bn}$$

The cushion is \$140 billion — a genuinely huge number. But against \$2 trillion of assets, a loss of just \$140bn — a 7% fall in asset values — wipes it out. In the 2008 crisis, losses on mortgage-related assets at some institutions exceeded that. The lesson scale teaches is uncomfortable: a bigger bank does not have a *thicker* cushion in percentage terms; it has the same thin slice spread over a far larger and more systemically dangerous balance sheet.

*Intuition:* equity scales with the balance sheet, so a giant bank's cushion is just as proportionally thin as a small one's — there is no safety in size, only more at stake.

## Common misconceptions

**"Equity is a pile of cash the bank keeps in reserve."** No. Equity is a *funding source* on the right side of the balance sheet, not an asset on the left. It is the difference between what the bank owns and what it owes — the owners' residual claim. A bank can have plenty of equity and very little cash, or lots of cash and almost no equity. When a regulator demands "more capital", they are not asking the bank to hoard cash; they are asking it to fund more of its assets with shareholders' money and less with borrowing. Confusing capital (a funding mix) with liquidity (cash on hand) is the single most common and most dangerous error in reading a bank.

**"A well-capitalised bank can't fail."** Capital protects against *solvency* (losses exceeding the cushion); it does nothing directly against a *liquidity* run (everyone demanding their money at once). A bank can be perfectly solvent — assets worth more than debts — and still die in a day because it cannot turn its long-term loans into cash fast enough to pay fleeing depositors. SVB was arguably solvent on a held-to-maturity basis until the run forced it to realise losses. Capital and liquidity are two different defences against two different deaths; this post covers only the first. (Liquidity gets its own treatment in [liquidity management, LCR, NSFR and the buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)

**"Higher leverage just means higher returns — banks lever up because they're greedy."** Leverage cuts both ways with mathematical symmetry: the same multiple that turns a 1% asset return into a 10% equity return turns a 1% loss into a 10% hit. Banks are not uniquely greedy; they are *structurally* leveraged because their core business is borrowing (deposits) to fund assets (loans). The fragility is built into the business model, not bolted on by recklessness — though banks absolutely can and do push leverage too far, as Lehman did.

**"A 3% fall in asset value is trivial — banks deal with bigger moves all the time."** A 3% fall in asset value is trivial *relative to the assets* and catastrophic *relative to the equity*, and which one matters depends entirely on leverage. At 30x leverage, 3.3% is the whole cushion. The mistake is anchoring on the wrong denominator: a move that is a rounding error to the asset book can be the entire net worth of the firm. Always measure losses against equity, never against assets.

**"If a bank's book equity is positive, it's solvent."** Only if the book values are honest. Book equity uses the values the bank carries its assets at, which for held-to-maturity securities can be well above market value. A bank can report positive book equity while being economically insolvent on a mark-to-market basis. The gap between accounting solvency and economic solvency is exactly where SVB's \$17 billion of hidden losses lived. Trust the mark-to-market number when a crisis is forcing sales.

**"More capital makes banks less profitable, so it's bad for the economy."** This is the banking industry's favourite argument against higher requirements, and it is half-true and half-misleading. Yes, all else equal, more equity lowers ROE (the leverage identity guarantees it). But a better-capitalised bank is also a *safer* bank, which lowers its cost of funding (lenders and depositors demand less compensation for risk) and makes it far less likely to need a taxpayer rescue. A large body of research suggests the economy-wide cost of modestly higher bank capital is small, while the benefit — fewer catastrophic crises like 2008, which cost trillions in lost output — is enormous. The "more capital hurts the economy" framing measures the cost to shareholders' ROE and quietly ignores the cost to everyone else of a bank that fails. A thicker cushion is a tax on bank shareholders' returns and an insurance policy for everyone else.

**"All of a bank's equity is available to absorb losses."** Not quite. Book equity includes items that would not actually be there to soak up a real loss — goodwill from past acquisitions, certain deferred tax assets, and other intangibles that have no recoverable value in a crisis. This is exactly why regulatory capital *deducts* these items from book equity to arrive at the loss-absorbing core (CET1). A bank can show \$100 billion of book equity of which \$15 billion is goodwill — leaving only \$85 billion of genuine, crisis-proof cushion. When you size up a bank's true buffer, strip out the intangibles; the cushion that absorbs losses is the tangible, common-equity part, not the headline equity line.

## How it shows up in real banks

The leverage math is not a textbook abstraction. It is the proximate cause of two of the most consequential bank failures of the modern era, separated by fifteen years but identical in their arithmetic.

### Lehman Brothers, 2008: 30.7x leverage and a 3.3% margin for error

Lehman Brothers entered the financial crisis as one of the most leveraged major institutions on Wall Street. At the end of 2007 its reported leverage was about 30.7 times — roughly \$30.70 of assets for every \$1 of equity, supporting a balance sheet that approached \$639 billion of assets. At that leverage, the asset fall needed to wipe out the firm's entire equity was 1 ÷ 30.7 ≈ 3.3%. Lehman's assets were heavily concentrated in mortgage-related securities and real-estate exposures — precisely the assets that fell far more than 3.3% as the housing market unwound.

There was a second, darker layer. Lehman used a technique nicknamed "Repo 105" to temporarily move about \$50 billion of assets off its balance sheet at quarter-end, making its reported leverage look lower than it truly was. So the *real* leverage, and the *real* margin for error, was even thinner than the headline 30.7x. When wholesale lenders — the firms Lehman borrowed from overnight to fund its book — lost confidence and refused to roll over the funding, Lehman could neither absorb the asset losses (the cushion was too thin) nor refinance its short-term debt (the run). It filed for bankruptcy on 15 September 2008. The deep dive on the mechanics lives in [Lehman Brothers 2008: leverage, Repo 105 and the run](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank); for our purposes, Lehman is the canonical demonstration that extreme leverage shrinks the survivable asset move to almost nothing.

![Bar chart comparing Lehman 30.7 times leverage with a ten times commercial bank peer](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-4.png)

### Silicon Valley Bank, 2023: when the hidden loss quietly exceeded the cushion

SVB is the more instructive case precisely because it was *not* a wild-eyed leverage story. It was a deposit-funded commercial bank that did something supposedly conservative: it invested its flood of tech-startup deposits in high-quality bonds, including a large held-to-maturity book of about \$91 billion. The problem was duration. When the Federal Reserve raised rates rapidly through 2022, the market value of those long-dated bonds fell. By the end of 2022, SVB was carrying roughly \$17 billion of unrealized losses across its available-for-sale and held-to-maturity books — against common equity of about \$16 billion.

Sit with that comparison. The hidden mark-to-market loss was *larger than the entire equity cushion*. On a book basis, SVB looked adequately capitalised, because the held-to-maturity losses were not flowing through equity. On an economic, mark-to-market basis, the cushion was already gone. The chart below puts the two side by side. When depositors and analysts did the same arithmetic in March 2023, confidence evaporated: \$42 billion of deposits were withdrawn on 9 March alone — with 94% of SVB's deposits sitting above the \$250,000 insurance limit and therefore acutely flight-prone — and an estimated \$100 billion more was queued to leave the next day before the FDIC seized the bank. SVB is the proof that the cushion that matters is the mark-to-market one, and that a bank can be killed by "safe" assets if leverage is high and the losses are hidden in the wrong accounting bucket. The full anatomy is in [the SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

![Bar chart of SVB seventeen billion dollar unrealized bond loss versus its sixteen billion dollar equity cushion](/imgs/blogs/bank-capital-and-leverage-why-equity-is-the-thin-cushion-5.png)

### The pattern across both

Strip away the surface differences — an investment bank versus a commercial bank, mortgage securities versus government bonds, 2008 versus 2023 — and the skeleton is identical. A leveraged balance sheet meant a small percentage fall in asset value was enough to exhaust a thin equity cushion. In Lehman's case the leverage was overt (30x) and the asset fall came from credit. In SVB's case the leverage was ordinary but the cushion was secretly already eaten by interest-rate losses hiding off the equity line. Both times, the moment the market realised the cushion was gone, funding fled and the bank died within days. Capital determined whether the bank *could* survive the losses; confidence and liquidity determined whether it got the chance to. The two halves of the fragile trade, one wound.

## The takeaway / How to use this

The deepest thing to internalise from this post is a reflex about *denominators*. When you read that a bank lost 2% on its assets, or that bond values fell 4%, your instinct should not be "that's small". Your instinct should be: *small compared to what?* Compared to the assets, yes. Compared to the thin equity cushion underneath — which at 10x leverage is only 10% of those assets, and at 25x is only 4% — a 4% asset fall is the *entire company*. Banking is the one business where you must always measure the loss against the sliver of equity, never against the mountain of assets it supports. Get the denominator right and the whole field stops being mysterious.

From that single reflex, three practical lenses follow. First, **leverage is the master fragility number.** Before you look at a bank's clever products or its growth story, compute assets ÷ equity. A bank at 10x will survive a recession that vaporises a bank at 25x, all else equal, because the survivable asset fall is two and a half times larger. Second, **always ask which capital and at what marks.** A headline capital ratio can flatter a bank that holds the wrong kind of capital (light on CET1) or carries underwater assets at cost (held-to-maturity losses off the equity line). The cushion that saves you in a crisis is the mark-to-market, common-equity cushion — so look there, not at the friendliest number on the front page. Third, **remember that capital answers only half the question.** It tells you whether a bank can *absorb* losses; it tells you nothing about whether it can *meet withdrawals* while it does. A bank dies of either, and the thin cushion only guards against the first.

If you take one image away, take the cover figure: a tall block of assets funded almost entirely by deposits and debt, with equity as a thin green strip at the very top. That strip is the whole margin between a bank that takes its losses quietly and a bank that takes the financial system down with it. Everything else in this series — credit underwriting, loan-loss provisioning, liquidity buffers, the Basel rulebook, the stress tests, the great failures — is, in the end, an elaborate effort to keep that strip thick enough, honest enough, and well-understood enough that it does its one job: absorbing the losses before they reach the people who trusted the bank with their money. A bank is a leveraged, confidence-funded machine, and equity is the thin cushion on which the whole fragile trade rests.

This is educational material about how banks are structured, not investment advice about any particular bank or security.

## Further reading & cross-links

- [Reading a bank balance sheet: assets, liabilities and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — the full anatomy of the balance sheet this post stands on, with every line item defined from zero.
- [Basel I, II, III and the capital rules that govern every bank](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) — where the CET1 / AT1 / Tier 2 tiers and the minimum ratios are derived in full.
- [Lehman Brothers 2008: leverage, Repo 105 and the run on an investment bank](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank) — the deep dive on how 30.7x leverage and a wholesale-funding run brought down a 158-year-old firm.
- [BIS and Basel: how banks are regulated](/blog/trading/finance/bis-and-basel-bank-regulation) — the system-level view of the global capital framework and the institutions that set it.

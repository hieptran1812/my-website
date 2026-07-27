---
title: "Alameda Research: The Cautionary Tale of Every Conflict in One Firm"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero autopsy of Alameda Research — how one owner running a market maker, a prop desk, an affiliated exchange, and the token used as collateral concentrated every conflict in crypto into a single balance sheet, and how the whole structure unwound in nine days. With worked dollar examples for the collateral loop, the liquidation gap, and the bank-run arithmetic."
tags: ["crypto", "market-makers", "alameda-research", "ftx", "ftt", "counterparty-risk", "proof-of-reserves", "crypto-players", "risk-management", "case-study"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 28
---

> [!important]
> **TL;DR** — Alameda Research was a crypto trading firm that, on paper, did nothing unusual: it made markets, it took directional bets, it invested. What made it the defining disaster of the last cycle was *structure*. One owner controlled the market maker, the proprietary trading book, the exchange that held customers' deposits, **and** the token those deposits were effectively lent against. When you put all four under one roof with no wall between them, every conflict that markets normally keep in separate companies runs through a single balance sheet.
>
> - **Four hats, one head.** Alameda was a market maker (earns the spread), a prop desk (bets its own money), the sister of an exchange (FTX, which held customer funds and saw every order), and a beneficiary of a token it helped create (FTT). Each role is legitimate alone; together they are a machine for turning other people's money into your own risk.
> - **The collateral was the problem, not the leverage.** Alameda borrowed against FTT, a token whose price it and FTX heavily influenced. Marking your own token up buys you borrowing power out of thin air — and when the token falls, the collateral and the borrower fail at the same instant, because they share a price.
> - **A mark is not a bid.** A balance sheet that values an illiquid token position at the last trade price assumes a buyer exists at that price for *every* unit. For a large position in a thin token, most of that "value" was never money.
> - **The run was arithmetic.** Once withdrawal requests exceeded liquid assets, the outcome was decided. Everything after that was a queue. Per the U.S. Department of Justice (2024), roughly **$8 billion** of customer funds had been misappropriated; Sam Bankman-Fried was convicted on seven counts in November 2023 and sentenced to 25 years.
> - **You could have seen the shape of it without insider access.** Five public questions — who owns both sides, what is the collateral, is it segregated, is it audited, is the token affiliated — would have flagged the risk in 2021.

Most of this series is about players who move price on purpose and mostly stay inside the rules: venture funds that time unlocks, market makers that quote spreads, exchanges that list tokens. Alameda Research is the chapter where the machinery breaks. It is worth studying not because it was exotic — it was, mechanically, a very ordinary crypto trading firm — but because it took every conflict of interest the earlier posts described, removed the walls between them, and let them compound until an estimated eight-billion-dollar hole opened under a million customers.

This is a post-mortem, and I am going to be careful about what is *established* versus what is *characterized*. Sam Bankman-Fried was **convicted** by a jury in November 2023 and **sentenced** in March 2024; those are facts of record. The internal balance sheet that lit the fuse was **leaked and reported** by CoinDesk; I will treat its numbers as reported, not gospel. Where I give dollar figures I will attach a source and a date, because in a story this large the false-precision temptation is enormous and the real lesson survives without it.

## Foundations: what Alameda actually was

Alameda Research was founded in **2017** by Sam Bankman-Fried (SBF) and a small group of traders, several of them from the effective-altruism and quantitative-trading worlds. Its original edge was boring and real: arbitrage. In 2017–2018, Bitcoin traded at persistently higher prices on Japanese and Korean exchanges than on U.S. venues — the famous "kimchi premium" and its Japanese cousin. A firm that could move dollars, yen, and coins across borders faster than everyone else could buy cheap on one venue and sell dear on another, pocketing the gap. That is honest market-making-adjacent arbitrage, and it is how Alameda first made money.

That origin matters for one reason: it built a reputation for competence that later carried far more weight than it should have. A firm that genuinely printed money on clean arbitrage in 2017–2018 earns the benefit of the doubt from lenders and investors in 2021. When the same firm later says "our balance sheet is fine," people who remember the arbitrage years are inclined to believe it. Competence in one era is not solvency in the next, but markets routinely confuse the two, and Alameda's early edge bought it years of unearned trust.

In **2019**, Bankman-Fried and Gary Wang founded **FTX**, a derivatives-focused crypto exchange. This is the fork in the road. Alameda did not stop trading; it became the *sister company* of an exchange. And FTX issued its own exchange token, **FTT**, which gave holders fee discounts and other benefits — and which Alameda held in enormous size from the beginning.

By 2021, at the top of the bull market, both companies were enormous. FTX raised money from blue-chip venture funds — Sequoia, Temasek, SoftBank, Ontario Teachers' Pension Plan, BlackRock — at a peak valuation of roughly **$32 billion** in early 2022, and Bankman-Fried became the public face of "the good guy in crypto," testifying to Congress, buying stadium naming rights, and pledging to give his fortune away. The polish was real and it was load-bearing: the more legitimate FTX looked, the less anyone probed the plumbing between it and Alameda. Reputation was doing the job that an audit and a board were supposed to do, and reputation cannot fail gradually — it fails all at once.

Hold those three facts together, because the entire tragedy is latent in them:

1. **Alameda traded** — as a market maker and a prop desk.
2. **FTX custodied customer money** — deposits sat on the exchange.
3. **FTT existed**, Alameda owned a lot of it, and its price was heavily influenced by the very people who owned it.

Nothing on that list is illegal. The disaster is what happens when one person controls all three and there is no independent board, no segregation of customer funds, and no audited accounts to force the truth into daylight.

![One owner controlled a market maker, a proprietary trading book, the exchange that held customer deposits, and the token pledged as collateral, with no wall between them, so losses flowed to customers, lenders, and equity investors.](/imgs/blogs/alameda-research-the-cautionary-tale-1.webp)

### The four hats

The cleanest way to see why the structure itself was the risk is to name each role Alameda (and its owner) played, and ask a simple question of each: *how does this hat earn, what does it privately know, and who eats its losses?*

- **Hat 1 — Market maker.** Quotes both sides of a book, earns the spread, and is supposed to be neutral on direction. Legitimate. But a market maker sees order flow, and a market maker that is *also* the exchange sees **all** of it.
- **Hat 2 — Prop trader.** Takes directional bets and makes venture investments with the firm's own capital. Also legitimate — as long as the capital is genuinely the firm's own.
- **Hat 3 — Sister exchange.** FTX held customer deposits and matched their orders. An exchange is supposed to be a neutral venue and a *custodian* — the one role where the money on the platform is emphatically **not** yours to trade.
- **Hat 4 — Token issuer.** FTT was printed by FTX, held in size by Alameda, marked at market, and posted as collateral for loans. The issuer of a token has every incentive to want it high; using it as collateral turns that wish into leverage.

![A matrix of the four roles Alameda played, showing that each earns differently, holds different private information, and pushes its losses onto a different group, so combining them concentrates every conflict.](/imgs/blogs/alameda-research-the-cautionary-tale-2.webp)

Each hat is worn by legitimate firms every day. Citadel Securities makes markets. Jane Street runs a prop book. Coinbase runs an exchange. Binance issues BNB. What no serious, well-governed firm does is wear **all four at once with the same pocket and no wall** — because the private information from one role and the customer money from another become irresistibly available to cover the losses of a third. That is not a hypothetical; it is, per the Department of Justice's case, precisely what happened.

## The engine: a token that collateralizes itself

To understand how Alameda got so big and so fragile at the same time, you have to understand the **circular-collateral loop**. This is the mechanism, and it is worth walking slowly because it looks like magic and is actually just arithmetic.

Suppose you issue a token — call it FTT — and you hold a large fraction of the supply. You can borrow against that token as collateral. Lenders will lend you, say, 50 cents for every dollar of FTT you pledge (a 50% loan-to-value ratio, which is generous for an illiquid affiliated token, but Alameda's lenders were not always careful). Now here is the loop: **you can spend money to push the token's price up, and every dollar the mark rises multiplies the collateral value of the tokens you already hold.**

### Worked example 1 — the circular-collateral loop

Say you hold **90 million FTT**. The market price is **$25**, so your position is marked at $2.25 billion, and at 50% LTV it supports **$1.125 billion** of borrowing.

Now you spend **$50 million** buying FTT in a thin market, and it lifts the price from $25 to **$30** — entirely plausible when the float actually trading is small and you are the dominant holder. Your 90 million tokens are now marked at 90m × $30 = **$2.70 billion**. At 50% LTV that is **$1.35 billion** of borrowing capacity.

You spent **$50 million** and your borrowing base grew by **$225 million** ($1.35bn − $1.125bn) — new credit conjured from a mark you moved yourself. Do this repeatedly and a modest amount of real capital inflates into an enormous balance sheet.

![The circular-collateral loop: spending fifty million dollars to lift a self-held token's mark from twenty-five to thirty dollars expands borrowing capacity by two hundred twenty-five million, credit created from nothing.](/imgs/blogs/alameda-research-the-cautionary-tale-3.webp)

The catch — and it is the whole catch — is that **the loop runs in reverse just as fast**. When the mark falls, the collateral shrinks, and the lender calls the loan. Because the collateral (FTT) and the borrower (Alameda) *share a price*, a decline in FTT doesn't just reduce Alameda's assets; it simultaneously triggers margin calls Alameda can only meet by selling the very asset that is falling. Both sides of the trade fail at the same moment. This is the structural bomb. Nothing about it is a crime by itself — plenty of companies pledge their own stock — but doing it with an illiquid token, at scale, while also holding customer deposits, is how you build something that can go from "solvent" to "gone" in days.

## The lie inside "mark to market"

There is a second, subtler mechanism that made Alameda's balance sheet a fiction long before anyone ran on it: the difference between a **mark** and a **bid**.

When you hold a position and value it at "the last traded price × the number of units," you are making a silent assumption: *that a buyer exists at that price for every single unit you hold.* For a liquid asset — a few thousand shares of Apple — that assumption is basically fine. For a huge position in a thin token, it is a lie, and the size of the lie grows with the size of the position.

### Worked example 2 — the mark says $500m, the book says $95m

You hold **20 million units** of an illiquid token. The last print was **$25**, so your balance sheet proudly reads **$500 million**. Then you actually try to sell.

- The first slice — say **3 million tokens** — clears near the top of the book, averaging maybe **$25.7** → about **$77 million**. Wait, the book was thin; realistically you get an average around **$25.7** only on the very first sliver, then it drops fast.
- The next slice walks the book down: the next few million tokens average perhaps **$5**, bringing in on the order of **$15 million**.
- By the time you are dumping the last **10 million tokens**, the bids have essentially evaporated; you average maybe **$0.30**, for about **$3 million**.

Add it up: roughly **$77m + $15m + $3m ≈ $95 million** actually realized. The balance sheet said $500 million. **The other ~$405 million was never money** — it was a mark that assumed depth that did not exist. (The exact numbers here are illustrative; the *shape* — a headline mark several times larger than the cash you could raise — is exactly what an affiliated-token-heavy balance sheet looks like.)

![Marking twenty million tokens at the last print of twenty-five dollars shows five hundred million on the balance sheet, but selling into a thin order book realizes only about ninety-five million, because past the first few million tokens there is almost no bid left.](/imgs/blogs/alameda-research-the-cautionary-tale-4.webp)

Combine the two mechanisms and you have Alameda's balance sheet in one sentence: **a large fraction of its assets were affiliated tokens (FTT, and a very large Solana position) marked at prices that could not survive being sold, pledged as collateral for real dollars that had to be repaid.**

## The spring that broke the book

Balance sheets do not leak at random moments; they leak after something has gone wrong. To understand why the fatal document was dated **30 June 2022**, you have to understand what happened to crypto in the spring of that year — because that is when Alameda's real losses were incurred, and, per the trial testimony, when the reliance on customer funds went from convenient to existential.

In **May 2022**, the **Terra/LUNA** ecosystem collapsed, wiping out roughly **$40 billion** of nominal value in days when its "algorithmic stablecoin" UST lost its dollar peg. That detonation cascaded through crypto's tightly interconnected credit markets. **Three Arrows Capital (3AC)**, a large hedge fund that had borrowed from nearly every lender in the industry, defaulted and entered liquidation in the summer. Lenders that had funded 3AC — **Celsius**, **Voyager**, **BlockFi**, **Genesis** — either failed or froze, and a broad **crypto credit crunch** set in: everyone wanted their loans back at once.

Alameda was a major borrower in exactly this market. As reported and as described at trial, the spring crash inflicted large losses on Alameda's book and triggered margin calls from its third-party lenders — the same lenders now demanding repayment across the whole industry. A firm with genuine, segregated, liquid capital absorbs this. A firm whose assets were disproportionately its own affiliated tokens (FTT, SOL), marked at prices that could not survive being sold, faced a gap between what it owed in real dollars and what it could actually raise. That gap is the reason the June balance sheet existed at all, and the reason it looked the way it did.

The honest response to that gap would have been to take the loss, wind down, and tell lenders the truth. What happened instead — per the DOJ's case and the cooperating witnesses' testimony — is that the gap was filled with **FTX customer money**.

### The line of credit

Mechanically, the customer funds reached Alameda through its account on FTX. Every trading firm on an exchange has some borrowing limit and gets liquidated if it breaches it. Alameda, per trial testimony, had two special privileges that no ordinary customer had: a **near-unlimited line of credit** (witnesses referenced a figure on the order of **$65 billion**) and an **exemption from the exchange's automatic liquidation engine** — reportedly implemented as a hidden "allow negative balance" setting in FTX's code. Together those meant Alameda could draw down customer deposits essentially without limit and never be forcibly closed out the way any other account would be.

### Worked example 5 — what an uncapped line of credit is worth

Consider two identical trading firms, each with **$1 billion** of genuine equity, each wanting to make a leveraged bet.

- **Firm A** trades on a normal exchange with a standard **5× maximum**. It can control **$5 billion** of positions. If the market moves **20%** against it, its equity is wiped and the liquidation engine closes it out. Its maximum loss to *others* is roughly zero — the margin system protects the venue and other customers.
- **Firm B** has an uncapped line against the exchange's customer deposits and no liquidation trigger. It can control **$50 billion** or more. A 20% adverse move does not close it out; it simply converts into a **$10 billion** hole in customer funds that no one can see, because the account is never forced to settle.

Same equity, same bet, radically different blast radius. The difference is not skill or leverage ratio in the ordinary sense; it is the **removal of the liquidation backstop** that exists precisely to stop one account's losses from becoming everyone else's. That removal — reported as a deliberate code change — is what turned a large trading loss into an $8 billion customer shortfall.

This is why "Alameda was just a hedge fund that made bad bets" is the wrong lesson. Hedge funds make bad bets constantly; the ones that trade with their *own* money and face *real* margin calls blow up privately, and their customers — there are none — lose nothing. Alameda's losses became catastrophic only because the four hats let it reach into the one pot of money that was never supposed to be at risk.

## The leaked balance sheet

On **2 November 2022**, CoinDesk's Ian Allison published a story based on a **leaked internal Alameda balance sheet dated 30 June 2022**. As reported, it showed roughly **$14.6 billion** of assets against about **$8 billion** of liabilities — a headline coverage of **1.83×**, which sounds comfortable. The problem was the *composition* of the asset side. A very large share of it was **FTT** and **Solana (SOL)** — exactly the affiliated, illiquid, self-influenced positions from the two worked examples above.

### Worked example 3 — coverage collapses when you refuse to count the affiliated tokens

Take the reported figures at face value and then do the one thing a careful lender would do: strip out the positions you could not actually sell near their mark.

- **As presented:** $14.6bn assets ÷ $8.0bn liabilities = **1.83× coverage**. Looks solvent.
- **Excluding FTT and SOL:** remove roughly **$7.0 billion** of those two positions and you are left with about **$7.58bn** of assets. Now $7.58bn ÷ $8.0bn = **0.95× coverage**. **Insolvent** — the remaining assets no longer cover the debt.

And this was the *flattering* version, because the leaked sheet still did not surface the roughly **$8 billion** of FTX customer money that had flowed to Alameda. A liquid, arm's-length balance sheet with 1.83× coverage is fine. A balance sheet whose "coverage" depends entirely on two tokens the firm itself influenced, and which omits the customer liability, is a countdown timer.

![A bar chart of Alameda's leaked 30 June 2022 balance sheet, showing headline coverage of 1.83 times collapsing to 0.95 times once the FTT and SOL positions that could not be sold near their mark are stripped out.](/imgs/blogs/alameda-research-the-cautionary-tale-5.webp)

Once that story was public, the market did the stripping-out in real time. **Changpeng Zhao (CZ)**, CEO of Binance — an early FTX investor that had been paid out partly in FTT — announced on **6 November 2022** that Binance would sell its remaining FTT holdings "due to recent revelations." That announcement, from the largest exchange in the world, was the match. FTT began to fall; the circular-collateral loop ran in reverse; and customers, now reading the same balance sheet, started trying to get their money off FTX.

## Nine days

What is stunning about the collapse is how *fast* it was once it started — and how little new information any of it required. The balance sheet had been public since 2 November. The run began days later. From the leak to the Chapter 11 filing was **nine calendar days**.

- **2 Nov 2022** — CoinDesk publishes the leaked Alameda balance sheet.
- **6 Nov** — CZ announces Binance will sell its FTT.
- **7 Nov** — Bankman-Fried publicly insists "FTX is fine. Assets are fine." (They were not.)
- **8 Nov** — FTX halts customer withdrawals. Binance signs a non-binding letter of intent to acquire FTX.
- **9 Nov** — Binance walks away after due diligence, citing reports of mishandled customer funds and investigations.
- **11 Nov 2022** — FTX, Alameda Research, and roughly 130 affiliated entities file for **Chapter 11 bankruptcy**. Bankman-Fried resigns; restructuring specialist **John J. Ray III** — who had overseen the Enron liquidation — takes over as CEO.

![A nine-day timeline from the leaked balance sheet on 2 November 2022 to the Chapter 11 filing on 11 November, showing that nothing new was discovered after the balance sheet went public four days before the run began.](/imgs/blogs/alameda-research-the-cautionary-tale-6.webp)

Ray's first-day declaration in the bankruptcy proceeding is one of the most quoted documents in crypto for a reason. He wrote that "never in my career have I seen such a complete failure of corporate controls and such a complete absence of trustworthy financial information," from a group that used software "to conceal the misuse of customer funds." Whatever else is contested, the *absence of controls* is not — it is the sworn statement of the man brought in to clean it up.

## The run was arithmetic, not panic

People describe bank runs as psychological — a stampede, a loss of confidence, animal spirits. That framing is comforting and wrong. A run is **arithmetic**. The only number that matters is *liquid assets divided by claims already in the queue*, and once that ratio drops below 1, the outcome is decided regardless of anyone's mood.

### Worked example 4 — the coverage ratio in a run

Imagine an exchange that owes customers **$10 billion** but can lay hands on only **$2 billion** of genuinely liquid assets today (the rest being illiquid tokens, venture stakes, and loans to an affiliate). Watch what happens as withdrawal requests pile up:

| Day | Cumulative claims filed | Liquid assets | Coverage |
|-----|------------------------|---------------|----------|
| 1   | $1bn                   | $2bn          | **2.00×** |
| 2   | $3bn                   | $2bn          | **0.67×** |
| 3   | $6bn                   | $2bn          | **0.33×** |
| 4   | $8bn                   | $2bn          | **0.25×** |

The line is crossed on **Day 2**. From that instant, the venue is technically unable to honor all requests; every withdrawal it *does* pay makes the position of the remaining customers worse. The only honest response once coverage drops below 1 is to close the gate — which is exactly what "withdrawals halted" means, and why it is always the tombstone. (Figures illustrative and rounded; the mechanism is what generalizes.)

![A line chart showing that in a run, cumulative withdrawal requests quickly exceed the fixed pool of liquid assets, so the coverage ratio falls below one on the second day and everything after is a queue.](/imgs/blogs/alameda-research-the-cautionary-tale-7.webp)

The deep reason the gate had to close is that the money was not there in the first place. Per the DOJ's 2024 case, roughly **$8 billion** of customer deposits had been used by Alameda — for venture investments, loan repayments, political donations, real estate, and to plug trading losses. The exchange's promise ("your deposits are yours, held for you") and the reality ("your deposits are working capital for our affiliated hedge fund") had diverged years earlier. The run just revealed it.

## What was proven, and what it cost

Here I want to be precise, because this is where careless writing turns allegation into asserted fact. The following are matters of **court record**, not characterization:

- **Sam Bankman-Fried was convicted** on **2 November 2023** — one year to the day after the CoinDesk story — on **seven counts**, including wire fraud, conspiracy to commit wire fraud, and conspiracy to commit securities and commodities fraud.
- He was **sentenced on 28 March 2024 to 25 years** in federal prison and ordered to forfeit approximately **$11 billion**.
- Three of his closest colleagues — **Caroline Ellison** (Alameda's CEO), **Gary Wang** (FTX co-founder and CTO), and **Nishad Singh** (FTX head of engineering) — pleaded guilty and cooperated with prosecutors. Ellison was sentenced in September 2024 to two years.
- Per the DOJ, the fraud involved roughly **$8 billion** of misappropriated customer funds, over **$1.3 billion** taken from lenders, and over **$1.7 billion** from equity investors.

There is a genuine, still-debated question about *recoveries*: the FTX bankruptcy estate, benefiting partly from a large recovery in the value of its Solana holdings and Anthropic stake, has said it can repay creditors their claims valued in dollars **as of the November 2022 petition date**, plus interest. That is a real and unusual outcome — most frauds return cents on the dollar. But it does **not** mean "no one was hurt": customers were locked out for years, were repaid the *dollar* value of assets that in some cases multiplied several-fold afterward, and bore enormous uncertainty. "The estate recovered a lot" and "it was a multi-billion-dollar fraud" are both true.

### Three misconceptions worth clearing up

Because the story is so large, it has accreted a set of comfortable explanations that are mostly wrong. Clearing them up is where the transferable lesson lives.

**"It was a fiendishly sophisticated fraud."** It was not. The trustee's own sworn words describe an *absence* of controls, not a clever circumvention of them: no reliable list of bank accounts, no complete list of employees, approvals conducted by emoji in group chats, corporate funds used to buy personal real estate. Sophistication would have required systems to defeat. There were no systems. The danger of Alameda was not genius; it was the vacuum where governance should have been. That should *raise* your guard, not lower it — the next one will look just as unremarkable from the outside.

**"FTT was a scam token."** Also not quite. FTT's mechanics — fee discounts, buy-and-burn, staking benefits — were ordinary exchange-token design, the same family as Binance's BNB. A token is not fraudulent for existing. The fraud was in the *use*: marking a self-held, thinly traded token at prices the firm influenced, pledging it as collateral for real dollars, and building an $8 billion customer hole behind it. The lesson is not "avoid exchange tokens"; it is "an affiliated token on the issuer's own balance sheet, marked at the issuer's own price, is a closed loop you cannot verify from outside."

**"Only insiders could have known."** This is the most dangerous misconception, because it excuses not looking. The affiliation was public. The FTT-as-collateral relationship was reported. The absence of an audit was knowable. The concentration of the balance sheet was in the leaked document for four days before the run. What insiders had was *confirmation and timing* — not the shape of the risk, which was legible to anyone who asked the five questions and was willing to act on disliking the answers.

![A checklist graph of five public questions about ownership, collateral, segregation, disclosure, and affiliated tokens that a customer could have asked of any venue without insider access.](/imgs/blogs/alameda-research-the-cautionary-tale-8.webp)

## What changed afterward

The collapse did change behavior, and the change is instructive precisely because it is incomplete. Within weeks, major exchanges rushed to publish **proof-of-reserves (PoR)** — cryptographic attestations, often using a Merkle tree of customer balances, meant to show that on-chain assets match what customers are owed. That is a genuine improvement: it makes the "is the money actually there" question checkable in a way it was not for FTX.

But PoR as practiced has a hole big enough to drive another Alameda through, and it is worth naming so you are not falsely comforted. A proof of *reserves* shows assets; it does not show **liabilities** or **off-balance-sheet obligations**. An exchange can prove it controls a billion dollars of coins on Tuesday while owing two billion, or while those coins are borrowed for the snapshot and returned the next day. Proof of reserves without a matching, audited proof of *liabilities* is half a fraction — the numerator without the denominator. It is better than nothing, and it is not solvency. The durable lesson is the same one the whole episode teaches: **an attestation you cannot independently verify, from a party that controls both sides, is a promise wearing the costume of a proof.**

Regulators moved too — more enforcement, more scrutiny of affiliated trading and custody, more pressure toward the separation of exchange, broker, and custodian functions that traditional finance mandates for exactly these reasons. Whether that hardens into durable structure or fades with the next bull market is, as of this writing, genuinely unsettled. The rules that would have stopped Alameda are not exotic; they are the boring separations — custody here, trading there, an auditor in between — that crypto spent a decade treating as legacy friction rather than hard-won protection.

## What it means if you're on the other side

The uncomfortable truth is that most of the red flags were **visible without insider access**. You did not need the leaked balance sheet to be nervous; you needed to ask five questions and dislike the answers.

**1. Who owns both sides?** If the same person controls the exchange holding your deposit *and* a large trading firm, the wall you are relying on is a promise, not a structure. Alameda and FTX shared an owner; that was public.

**2. What is the collateral, really?** If a firm's borrowing is secured by a token it issued or heavily influences, its "solvency" is a mark it can move — and that mark can move against it just as fast. FTT's role as collateral was reported well before the collapse.

**3. Is customer money segregated?** A custodial exchange should hold your deposits *separately* and be able to prove it. "Not your keys, not your coins" is not a slogan; it is a description of exactly this failure mode.

**4. Is it audited — by someone you'd trust?** FTX had no audit of the kind a public financial institution would face. Absence of a real audit is not neutral; it is the absence of the one mechanism that forces the truth out.

**5. Is the token affiliated?** A large position in a venue's own token, on that venue's balance sheet, marked at that venue's price, is a closed loop. Closed loops are where fraud hides, because there is no outside price to check against.

None of these questions require you to be a forensic accountant. They are the retail-defense version of the entire "how the players move price" series: when the same hand controls the quote, the venue, the collateral, and the disclosure, you are not a customer — you are inventory. The defense is not cleverness; it is refusing to leave more on any single venue than you are willing to see frozen, and preferring venues that can *prove* segregation and *survive* an audit.

### Worked example 6 — sizing your exposure to a venue

The practical version of "don't be inventory" is a sizing rule, and you can make it concrete. Suppose you actively trade and genuinely need funds sitting on exchanges to do it. Decide first how much *total* loss to a single venue failure you could absorb without it changing your life — say **$4,000**. Then treat each venue's on-exchange balance as if there were some probability it goes to zero this year.

If you assign even a modest **2% annual "venue failure or freeze"** probability to a mid-tier exchange, then leaving **$50,000** there carries an expected loss of 0.02 × $50,000 = **$1,000 per year** — and, more importantly, a **$50,000** worst case. Cap the *worst case*, not the expected value: if your tolerance is $4,000 of unrecoverable loss and you use three venues, hold no more than roughly **$1,300** on any one at rest, and move the rest to self-custody or to venues that publish real proof-of-reserves and segregation. The numbers are yours to set; the discipline is to size to the **frozen-forever** scenario rather than the everything-is-fine scenario, because Alameda's customers were, overwhelmingly, sized to the second.

This is the same rehypothecation antipattern that took down **Celsius** and **Voyager** in the same year: customer assets that were promised to be held turned out to be lent, leveraged, or pledged to affiliated risk. FTX was the largest and most spectacular instance, but it was not unique, and it will not be the last. The pattern — *custody plus affiliated trading plus an affiliated token, minus segregation and audit* — is the thing to recognize. When you see it, the correct exposure is the amount you would shrug at losing.

Alameda is the cautionary tale because it is not exotic. It is the ordinary crypto trading firm — market maker, prop desk, exchange, token — with the walls removed. Every earlier post in this series describes one of those hats operating within the rules. This post is what happens when one person wears all four and asks you to trust that they will keep the promises the structure was built to let them break.

There is one last thing worth sitting with. The people running Alameda and FTX were not caricature villains at the start; by most accounts they believed, for a long time, that they would trade their way out of the hole and no one would ever be hurt. That is exactly why the structure matters more than the intentions. Good intentions inside a machine built with no walls still route customer money to cover trading losses, because that is what the machine makes easy. You cannot protect yourself by judging whether the people are trustworthy — you protect yourself by checking whether the *structure* would let them hurt you even if they wanted to. Alameda is the proof that the second question is the only one that reliably works.

If you want to see how these same wallets and flows can be reconstructed *from the outside* — how an affiliated-token collateral loop or a diversion of customer funds leaves an on-chain signature you can trace — that is the subject of the forensic-lab posts later in this series, on [tracing a firm's on-chain footprint](/blog/trading/crypto-players/tracing-a-market-makers-onchain-footprint) and [following token flows from insiders to exit liquidity](/blog/trading/crypto-players/following-token-flows-from-insiders-to-exit-liquidity).

## Sources & further reading

- Ian Allison, "Divisions in Sam Bankman-Fried's Crypto Empire Blur on His Trading Titan Alameda's Balance Sheet," *CoinDesk*, 2 November 2022 — the leaked balance sheet story that began the collapse.
- *United States v. Samuel Bankman-Fried*, U.S. District Court, Southern District of New York — indictment, trial record, verdict (2 November 2023), and sentencing (28 March 2024).
- U.S. Department of Justice press releases on the conviction and sentencing of Samuel Bankman-Fried (2023–2024), for the customer, lender, and investor loss figures.
- John J. Ray III, Declaration in Support of First Day Motions, *In re FTX Trading Ltd.*, U.S. Bankruptcy Court, District of Delaware, 17 November 2022 — the "complete failure of corporate controls" statement.
- Guilty pleas and cooperation of Caroline Ellison, Gary Wang, and Nishad Singh (DOJ, 2022–2024).
- FTX Debtors' bankruptcy filings and reorganization plan disclosures (2023–2024) for recovery and repayment figures.
- Related posts in this series: [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), [The Crypto VC Operating Model](/blog/trading/crypto-players/the-crypto-vc-operating-model), and the hub, [Crypto VCs and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers).

*Nothing here is legal or investment advice. Adjudicated facts are cited to the court record; reported figures (notably the leaked balance sheet) are attributed to their source and date and should be read as reported, not independently verified.*

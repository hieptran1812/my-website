---
title: "Washington Mutual and the 2008 Mortgage Bank Failures: When the Loans Were the Problem"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How the largest US bank failure ever happened — WaMu's option-ARM machine, the credit losses that ate its capital, the ten-day deposit run, and the overnight FDIC seizure-and-sale that wiped its owners but spared every depositor."
tags: ["banking", "washington-mutual", "bank-failure", "subprime", "option-arm", "fdic", "indymac", "2008-crisis", "credit-risk", "deposit-run", "receivership"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Washington Mutual was a credit-driven failure: it lost so much money on bad mortgages that a deposit run finished a bank its own loan book had already mortally wounded, making it the largest US bank failure ever at about \$307 billion in assets.
>
> - WaMu's growth engine was the **option-ARM** — a mortgage whose balance could *grow* while the borrower paid a teaser rate, engineered to look cheap up front and then reset into a payment the borrower could never make.
> - When those loans went bad, the losses ate through the bank's thin equity cushion. That is the defining feature of a **credit-driven** death: the assets themselves were worth less than the bank paid for them, and the loss was permanent.
> - The run was the *symptom*, not the disease — depositors pulled about **\$16–17 billion over roughly ten days** in September 2008, and the regulator seized WaMu on September 25 and sold it to JPMorgan the *same day*.
> - The resolution spared every depositor and the FDIC insurance fund; shareholders and bondholders absorbed the entire loss. The one number to remember: **\$307 billion** — still the biggest bank to die on American soil.

In the third week of September 2008, the United States banking system was already on fire. Lehman Brothers had filed for bankruptcy on the 15th. The insurer AIG had been nationalised the next day. Money-market funds were "breaking the buck." And inside a thrift headquartered in Seattle — Washington Mutual, "WaMu," a bank older than the state of Washington itself — depositors were quietly emptying their accounts. Not in a single dramatic line out the door, the way a 1930s run looks in the photographs. They were doing it by phone, by web transfer, by walking up to the counter and asking for a cashier's cheque. Over about ten days, roughly \$16–17 billion left the building.

On the evening of Thursday, September 25, 2008, the regulator that supervised WaMu — the Office of Thrift Supervision, the OTS — declared the bank unsafe and unsound, seized it, and placed it into the receivership of the Federal Deposit Insurance Corporation. The FDIC did not spend a single dollar of its insurance fund. It had already lined up a buyer. That same night, JPMorgan Chase agreed to buy WaMu's banking operations — its branches, its deposits, its loans — for about \$1.9 billion. By Friday morning, WaMu's depositors were JPMorgan customers and didn't lose a cent. WaMu's shareholders and its bondholders were wiped out. With about \$307 billion in assets, it was, and remains, the largest bank failure in American history.

The diagram below is the mental model for the whole story: a slow credit collapse that played out over more than a year, then a fast funding collapse that finished it in ten days, then a seizure-and-sale that happened literally overnight. Hold that shape in your head — *slow poison, fast death, instant resolution* — because everything else in this post hangs off it.

![Timeline of Washington Mutual from credit cracks in 2007 to the September 2008 seizure and JPMorgan sale](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-1.png)

## Foundations: the words you need before the story makes sense

This post is going to use a handful of terms that the financial press throws around as if everyone already knows them. We don't assume that. Let's build each one from zero, with a plain-money explanation first, before we put it to work.

### What a bank actually is, in one sentence

A bank borrows money short and lends it long. It takes your deposit — money you can demand back *today* — and lends it out as a 30-year mortgage you can't get back for decades. It earns the gap between the low rate it pays you and the higher rate it charges the borrower. That trick has a name: **maturity transformation** (turning short-term money into long-term loans). It is the most useful and the most fragile thing in finance, because it only works as long as two things hold: depositors trust the bank enough to leave their money, and the bank's loans are actually good enough to be worth what it lent. WaMu broke the second condition first, and that broke the first. (We unpack the core machine in [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).)

### Equity: the thin cushion that absorbs losses

A bank funds its loans mostly with other people's money — deposits and borrowed funds. The slice that belongs to the bank's *owners*, the part that isn't owed to anyone, is called **equity** or **capital**. It is the shock absorber. If the bank's loans lose value, the loss comes out of equity first, before any depositor or lender is touched. The catch is that this cushion is *thin*. A typical large bank funds itself with about 8% equity and 92% other people's money, which means it is levered roughly 12.5 times (1 ÷ 0.08 ≈ 12.5). Lose more than that 8% to bad loans and the bank is, by definition, insolvent — it owes more than it owns. (For the full account of why this cushion is deliberately thin, see [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).)

### Subprime: lending to weak borrowers

A **subprime** borrower is one with a poor credit history — late payments, defaults, low credit scores — who a careful lender would charge more, or refuse. Subprime mortgages are loans to exactly these people. They're not inherently evil; charging a higher rate for a higher risk is normal banking. The disaster comes when a bank makes *huge volumes* of them, prices them as if the risk were small, and assumes house prices will keep rising forever to bail out any borrower who can't pay.

### Option-ARM: the loan whose balance can grow

This is the one to slow down on, because it is the engine of the whole WaMu story. An **ARM** is an *adjustable-rate mortgage* — the interest rate changes over time, instead of being fixed for 30 years. An **option-ARM** (also called a "pick-a-payment" loan) goes one disastrous step further: each month the borrower can *choose* how much to pay. They can pay the full amount, an interest-only amount, or — and here is the trap — a **minimum payment** that is *less than the interest owed*.

When you pay less than the interest, the unpaid interest doesn't vanish. It gets added to the loan balance. Your debt *grows* while you make your payments. That phenomenon — a loan balance rising because you're underpaying the interest — is called **negative amortization** ("neg-am" for short). Normal loans amortize *down*; an option-ARM in minimum-payment mode amortizes *up*. We'll do the arithmetic on this shortly, because it's the heart of why these loans were guaranteed to fail.

### Originate-to-distribute: making loans to sell, not to keep

The old model of banking was **originate-to-hold**: a bank made a loan and kept it on its books, so it cared deeply whether the borrower would repay. The model that took over in the 2000s was **originate-to-distribute**: a lender made the loan, then quickly sold it on to Wall Street, which bundled thousands of loans into bonds and sold *those* to investors. This bundling is called **securitization** (we cover the mechanics in [securitization](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities)). The poison in originate-to-distribute is the incentive: if you're going to sell the loan next week, you don't actually care whether it repays over 30 years. You care about *volume*. That broke the most important discipline in lending — the lender's own skin in the game.

### Credit-driven versus rate-driven failure

This is the single most important distinction in the post, so let's be precise. A bank can die two very different ways.

A **rate-driven** failure happens when a bank holds *good* assets that simply *lost market value* because interest rates rose. Bond prices fall when rates rise; a bank that loaded up on long-dated bonds at low rates can see a huge paper loss without a single borrower defaulting. That was Silicon Valley Bank in 2023: its assets were US Treasuries and government-guaranteed mortgage bonds — almost no credit risk — but rising rates created a \$17 billion unrealised loss, and a run forced it to crystallise that loss. (The full SVB and Credit Suisse story is in [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

A **credit-driven** failure is different and, in a sense, more fundamental: the assets themselves are *bad*. The borrowers won't repay. The loss isn't a paper mark that might recover if rates fall — it's permanent, because the money is genuinely gone. That was WaMu. Its problem wasn't that good loans lost market value; its problem was that the loans were worth less than it lent because the borrowers were defaulting and the homes were worth less than the mortgages. Keep this distinction live — we'll come back to it with a side-by-side figure.

Here is the distinction in a table, because it is the backbone of the whole post:

| | Credit-driven failure (WaMu) | Rate-driven failure (SVB) |
|---|---|---|
| What the assets were | Bad loans — option-ARMs, subprime | Good bonds — Treasuries, agency MBS |
| Why they lost value | Borrowers defaulted | Rates rose, bond prices fell |
| Is the loss permanent? | Yes — the money is gone | No — it reverses if rates fall |
| Where you see it coming | Rising loan-loss provisions | Rising unrealised loss footnote |
| How fast the run was | Slow (~10 days, retail deposits) | Fast (~36 hours, uninsured deposits) |

Both ended in a run and a seizure, but the *cause* — and therefore the early-warning signal an analyst should watch — was opposite.

### The seizure-and-sale: how a US bank actually dies

When a bank fails in the US, it doesn't file for bankruptcy like a normal company. A bank's *primary regulator* (for WaMu, the OTS) declares it insolvent and appoints the **FDIC as receiver**. **Receivership** means the FDIC takes legal control of the failed bank to wind it down in an orderly way. The FDIC's mandate is twofold: protect insured depositors (up to \$250,000 per depositor per bank) and minimise the cost to its insurance fund. Its preferred tool is a **purchase and assumption** transaction — it finds a healthy bank to *buy* the good parts (deposits, branches, performing loans) and *assume* the deposit liabilities, usually arranged in secret over a weekend (or in WaMu's case, a single evening) so customers wake up to a new name on the door but the same money in their account. (Deposit insurance and the resolution machinery are covered in [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).)

With those eight terms in hand — maturity transformation, equity cushion, subprime, option-ARM, originate-to-distribute, credit-vs-rate failure, receivership, and the seizure-and-sale — the rest of the story reads cleanly.

## The mortgage machine: how WaMu built itself to fail

To understand the failure, you have to understand how WaMu made money on the way up, because the same machine that grew it killed it.

Washington Mutual was a *thrift*, also called a savings institution — a bank historically focused on taking household deposits and making home loans. For most of its life it was a sleepy, conservative Seattle mortgage lender. In the late 1990s and 2000s, under a growth-obsessed strategy, it transformed itself into a national consumer-lending machine, acquiring lenders, opening branches aggressively, and — this is the part that mattered — leaning ever harder into high-margin, high-risk mortgages.

The crown jewel of that strategy was the option-ARM. WaMu loved option-ARMs for a reason that tells you everything about how a bank's accounting can lie to it. Remember negative amortization: when a borrower pays the minimum, the unpaid interest is added to the loan balance. Under the accounting rules of the time, the bank could *book that unpaid interest as income* — as if it had actually received the money. So the more borrowers underpaid, the more "earnings" WaMu reported, even though no cash was coming in and the loan was quietly getting riskier. The bank was, in effect, recognising profit on payments it hadn't received and might never receive. By the mid-2000s, a large share of WaMu's option-ARM borrowers were making only the minimum payment, which means a large share of WaMu's reported mortgage "income" was phantom — interest accrued onto balances that were already too big.

Let's walk the actual machine, step by step.

![Pipeline showing the option-ARM machine from teaser rate to negative amortization to reset shock to default and loss](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-3.png)

A borrower walks in. They want a house they can't really afford. The option-ARM lets them qualify on the *minimum* payment — a teaser as low as 1–2% — so a household that could only truly afford a \$200,000 house gets approved for a \$400,000 one. For the first stretch, they pay the minimum, the balance creeps up, and everyone is happy: the borrower has their house, WaMu books the accrued interest as profit, and the loan gets sold or held depending on the year. Then the loan hits its **recast** trigger — typically when the balance reaches 110–125% of the original amount, or after five years, whichever comes first. At recast, the minimum-payment option disappears. The payment jumps to the fully-amortising amount at the now-higher rate. For many borrowers, the monthly payment *doubled* overnight. They couldn't pay. They defaulted. And because house prices had peaked and were now falling, the home securing the loan was worth less than the ballooned balance — so when WaMu foreclosed and sold, it took a loss on every one.

#### Worked example: how an option-ARM turns into a guaranteed loss

Let's make the negative-amortization trap concrete. Suppose a borrower takes a \$400,000 option-ARM. The "real" interest rate is 7%, so the true monthly interest is \$400,000 × 0.07 ÷ 12 = \$2,333. But the teaser minimum payment is set at an effective 2%, so the borrower is allowed to pay only \$400,000 × 0.02 ÷ 12 = \$667 per month.

Each month, the borrower pays \$667 but owes \$2,333 in interest. The shortfall — \$2,333 − \$667 = \$1,666 — gets added to the loan balance. After one year of this, the balance has grown by roughly \$1,666 × 12 = \$20,000, so the borrower now owes about \$420,000 on a house they bought for \$400,000. They've been "paying their mortgage" faithfully and their debt went *up* \$20,000.

Now run it forward. After about three years of neg-am, the balance crosses the 110% cap — around \$440,000. The loan recasts. The payment jumps from \$667 to the full amortising payment on \$440,000 at 7% over the remaining term, which is roughly \$3,100 a month. The borrower's payment more than quadrupled. Meanwhile the house, bought at the 2006 peak for \$400,000, is now worth maybe \$300,000 in the 2008 market. The borrower owes \$440,000 against a \$300,000 house — they are \$140,000 "underwater." They walk away. WaMu forecloses, sells the house for \$300,000 (less, after selling costs — say \$270,000 net), and books a loss of \$440,000 − \$270,000 = **\$170,000** on a single loan.

The intuition: an option-ARM wasn't a loan that *might* go bad — once house prices stopped rising, it was a loan *engineered* to go bad, because the product itself made the balance grow into a payment the borrower was never able to make.

Multiply that single-loan loss across a book of tens of billions of dollars of option-ARMs and subprime mortgages, and you can see the shape of the iceberg WaMu was steering into.

### Why the machine looked so profitable right up to the end

The cruelest feature of the option-ARM model is that it produced its best-looking financials *just before* it blew up. Three things conspired to make a doomed loan book look like a growth story.

First, the **phantom-income accounting** we mentioned: neg-am interest the borrower never paid was booked as revenue, so the more the loans deteriorated (more borrowers paying the minimum), the better current "earnings" looked. Second, **rising house prices masked every bad loan** — as long as homes kept appreciating, a borrower who couldn't pay could simply refinance or sell at a profit, so defaults stayed artificially low and WaMu's loss rates looked benign. Third, **the loans were young**. A mortgage rarely defaults in its first year or two; the trouble comes at reset, three to five years in. So a bank that was *originating faster every year* always had a book dominated by fresh, not-yet-defaulting loans, which kept reported delinquency low even as the embedded risk piled up. The faster you grew, the healthier you looked — until growth stopped.

This is the trap of judging a lender by its current loss rate during a boom. WaMu's reported credit metrics looked tolerable into 2007 not because the loans were good but because they hadn't aged into their default window yet and house prices were still bailing out the early cracks. The instant both of those supports gave way — prices fell, and the 2005–2006 vintages hit their resets — the losses arrived all at once. A lender whose earnings *improve* as it makes riskier loans, in a market that *only* works if asset prices rise, is not a growth company; it is a time bomb with a good income statement.

## The credit deterioration: when the losses started eating capital

Through 2007 and into 2008, the housing market turned. House prices, which the entire option-ARM model implicitly assumed would keep rising, started falling — eventually more than 30% from peak in the worst markets. Every assumption baked into WaMu's loan book reversed at once. Borrowers couldn't refinance out of their resetting loans because they were underwater. Defaults climbed. Foreclosures climbed. And the losses started landing on WaMu's income statement as **provisions** — money the bank sets aside to cover loans it now expects to go bad. (For how provisioning works under modern accounting, see [collateral, security and loan-loss provisioning](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl).)

Here is where the *credit-driven* nature of the failure becomes a balance-sheet death sentence. Recall the equity cushion: a bank funds itself with roughly 8% equity. That is its entire margin for error. If loan losses exceed that 8% of assets, the bank is insolvent — full stop. WaMu's loan losses were not a paper mark that might recover; they were realised, permanent credit losses chewing directly through that cushion.

In April 2008, WaMu tried to plug the hole. It raised about \$7 billion in new equity from investors, led by the private-equity firm TPG, to absorb the mounting mortgage losses. This is the classic move of a credit-stressed bank: when losses eat your capital, you sell new shares to rebuild it. But raising \$7 billion to cover losses that the market increasingly believed would run *far* higher was like bailing a boat with a teacup. Within months it was clear the losses kept coming, the new capital was being burned, and — critically — the \$7 billion injection diluted existing shareholders and signalled how deep the trouble ran. WaMu's stock, which had traded above \$40 in 2007, collapsed toward a few dollars.

The credit deterioration also poisoned WaMu's funding. A bank with a deteriorating loan book pays more to borrow in the wholesale market (from other banks and money-market funds) because lenders demand a premium for the risk — and eventually they stop lending to it at all. So the credit problem on the *asset* side started squeezing the *funding* side even before depositors moved. The bank was being attacked from both ends: its loans were worth less, and its money was getting more expensive and harder to roll over.

#### Worked example: credit losses eating the capital cushion

Let's quantify how fast credit losses can kill a bank, using realistic proportions. Take a simplified bank with \$300 billion in assets (roughly WaMu's scale) funded by 8% equity. That's \$24 billion of equity standing between the bank and insolvency.

Now suppose its mortgage book — say \$150 billion of the \$300 billion in assets — is the toxic option-ARM-and-subprime pile, and the market comes to believe lifetime losses on that book will run 15%. That's \$150 billion × 0.15 = **\$22.5 billion** of expected credit losses.

Compare the two numbers. The capital cushion is \$24 billion. The expected losses are \$22.5 billion. The losses very nearly *equal* the entire equity base. There is essentially no cushion left over for anything else to go wrong — and plenty else was going wrong. Even the \$7 billion capital raise only lifts the cushion to about \$31 billion against losses that, once you add fear and a falling-housing tail, the market feared could run well past \$30 billion. At that point the bank isn't bankrupt on paper *yet*, but it is *visibly* insolvent in the market's eyes — and a bank that the market believes is insolvent cannot fund itself.

The intuition: a credit-driven failure isn't subtle. You can almost watch it on a single line — expected loan losses creeping up toward the size of the equity cushion. When the loss estimate crosses the capital, the bank is dead; the only question left is the date.

This is the moment to look at WaMu in context. It became the largest bank failure in US history — bigger than every other name in the modern record, including the three big 2023 failures.

![Horizontal bar chart of the biggest US bank failures by assets with Washington Mutual largest at 307 billion dollars](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-2.png)

At about \$307 billion in assets, WaMu dwarfs even First Republic (\$229 billion) and SVB (\$209 billion) from 2023. Sit with that ranking for a second: of the six biggest US bank failures ever, the *largest* is the credit-driven one, and it has held that record for over fifteen years. Size, in banking, is not safety — it's just a bigger thing to fall over.

## The run: when the credit problem became a funding crisis

A bank can limp along while insolvent for a surprisingly long time, *as long as its depositors don't notice or don't move*. WaMu had been losing on its loans for a year. What killed it in days was the run.

To see why a run is fatal, you have to remember what a deposit *is* from the bank's side. A deposit is a loan from you to the bank that you can call in *at any moment*. The bank doesn't keep your money in a vault; it has lent it out (in WaMu's case, into 30-year mortgages it could not sell quickly, and certainly not at full value in a crashing market). So a bank holds only a thin sliver of its assets as cash and liquid securities it can turn into cash fast. When too many depositors ask for their money at once, the bank simply doesn't have it on hand — not because it's a fraud, but because the whole *business model* is to lend the deposits out. (We cover why the funding base is the franchise in [retail deposits and the funding base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).)

For most of 2008, WaMu's deposits held up reasonably well — retail depositors are "stickier" than wholesale lenders, and most accounts were under the insured limit. But two events in September lit the fuse. IndyMac had already failed in July (more on that below), planting the idea that a big mortgage lender could actually go down and that uninsured money might not be safe. Then Lehman Brothers collapsed on September 15, and the entire financial system convulsed. Suddenly the abstract worry — *is my bank safe?* — became a concrete, urgent question for millions of WaMu customers. Over the roughly ten days from Lehman's failure to the seizure, depositors pulled about \$16–17 billion.

![Step chart of the WaMu deposit run reaching about 16 to 17 billion dollars over ten days in September 2008](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-4.png)

The chart shows the cumulative outflow climbing day after day (the exact daily path is illustrative; the endpoint of about \$16–17 billion and the roughly ten-day window are the cited facts from the OTS and FDIC). This was not a frenzied 1930s queue, and it wasn't the 36-hour, social-media-driven, \$42-billion-in-a-day digital sprint that took down SVB fifteen years later. It was a steady, grinding bleed — fast enough to overwhelm a bank that was already weakened, slow enough that the regulators could see it coming and prepare the sale.

Why is a \$16–17 billion outflow lethal to a \$300 billion bank? Because a bank can only meet withdrawals from its *liquid* assets — cash and securities it can sell immediately. The rest is locked up in illiquid mortgages it can't sell at par in a crashing market. Once the liquid buffer is gone, the bank must either borrow against its remaining assets (from other banks or the central bank) or fail. By late September, WaMu had drawn down its liquidity, the wholesale market had effectively shut to it because everyone knew it was wounded, and the run showed no sign of stopping. The OTS later cited the deposit outflow explicitly as the trigger for the seizure: the bank had become unable to meet its obligations.

#### Worked example: the deposit run versus the cash on hand

Let's see why a run is a *liquidity* execution even of a bank that is also insolvent. Take our \$300 billion bank again. Suppose it holds, generously, 10% of assets as liquid resources it can actually turn into cash quickly — that's \$30 billion. The other \$270 billion is mortgages and other loans that cannot be sold fast at full value, especially not in September 2008.

Now the run hits: \$16–17 billion walks out over ten days, and the pace is *accelerating* because every day's headline scares the next wave of depositors. Of the \$30 billion liquid buffer, more than half is already gone. The bank tries to raise more cash, but it can't sell its mortgages near par (the market has collapsed), it can't borrow in the wholesale market (lenders won't touch it), and the run is still running. Project the trend forward and the \$30 billion buffer empties in another week or two. At that point the bank physically cannot return the next depositor's money.

Note the cruel double-bind. The bank is *insolvent* (loan losses ≈ its capital) *and* about to be *illiquid* (run > liquid buffer). Either one alone can kill a bank; WaMu had both. The insolvency meant no one would lend to it or buy its equity to save it; the illiquidity meant it would run out of cash within days regardless.

The intuition: a run doesn't have to drain the *whole* bank to kill it. It only has to drain the thin liquid buffer faster than the bank can refill it. For WaMu, \$16–17 billion against a buffer of maybe \$30 billion, with no way to refill, was plenty.

## The seizure: the night the regulator pulled the plug

By Thursday, September 25, 2008, the OTS had seen enough. The bank's deposit run, its collapsing share price, its inability to fund itself, and the market's verdict that its losses exceeded its capital all pointed the same direction. The OTS declared Washington Mutual Bank unsafe and unsound, closed it, and appointed the FDIC as receiver.

Here is the part that surprises people: the FDIC didn't pay out depositors from its insurance fund, and it didn't liquidate the bank piece by piece. It executed a **purchase and assumption** — it had been quietly running an auction for WaMu's banking operations in the days before, and JPMorgan Chase won. The deal was signed and announced the same night. JPMorgan paid about \$1.9 billion to acquire WaMu's banking subsidiary: its roughly 2,200 branches, its deposit base, and its loan book. By the time WaMu's branches opened Friday morning, they were JPMorgan Chase branches in all but signage. No depositor lost access to their money for even a single business day.

This is the FDIC's resolution model working exactly as designed, and it's worth dwelling on the choreography because it's genuinely elegant. The regulator times the seizure for after business hours (close of business Thursday). The buyer is pre-arranged. The transfer is legal and instantaneous: the deposits and the good assets move to the acquirer; the holding company and its obligations stay behind in the receivership to be wound down. The public sees continuity; the loss is allocated *inside* the structure, to the people who legally agreed to bear it.

The crucial structural point — and the source of endless confusion and litigation afterwards — is the distinction between **Washington Mutual Bank** (the operating bank that the OTS seized and the FDIC sold) and **Washington Mutual Inc.** (the *holding company* that owned the bank and that had issued bonds and stock to investors). The FDIC sold the *bank* to JPMorgan. The *holding company* was left with almost nothing — its main asset, the bank, had been taken and sold — so it filed for bankruptcy the next day. That is why the bank's depositors were perfectly fine while the holding company's shareholders and bondholders were destroyed: the regulator rescued the *functioning bank* and let the *investment vehicle that owned it* collapse.

#### Worked example: the resolution waterfall — who gets what

Let's trace the dollars through the seizure, because the "waterfall" — the order in which claims get paid in a failure — is the whole point of a bank's capital structure.

When a bank is resolved, claims are paid in a strict priority. Depositors and the FDIC sit at the *top* (most protected); then secured creditors; then senior bondholders; then subordinated ("junior") bondholders; and at the very *bottom*, taking losses first, the shareholders. Imagine WaMu's resolution as a bucket of recovered value poured from the top.

Start with the depositors. There were roughly \$188 billion of deposits. JPMorgan *assumed* all of them as part of the purchase — both insured and uninsured. So depositors recover **100 cents on the dollar**. The FDIC insurance fund pays out **\$0**, because the sale covered the deposits; this was the rare large failure that cost the fund nothing.

Now the holding company's investors. WaMu Inc. had issued senior debt, subordinated debt, preferred stock, and common stock to raise money over the years — call it, in round numbers, billions across those layers. When the bank was sold to JPMorgan and the holding company went bankrupt, the recovered value flowed up the priority ladder: secured and senior claims at the holding company recovered *some* cents on the dollar in the lengthy bankruptcy that followed, subordinated bondholders recovered a sliver, preferred shareholders recovered next to nothing, and common shareholders — who had owned a stock worth over \$40 a year earlier — recovered **essentially zero**.

So the waterfall did its job: every depositor whole, the deposit-insurance fund untouched, and the entire loss landing on exactly the people who had signed up to bear it — the bank's owners and its junior lenders. That allocation is not an accident or an injustice; it is the *purpose* of having an equity-and-bond cushion under the deposits in the first place. The capital stack exists to be wiped out so the depositors don't have to be.

![Graph of the WaMu resolution showing seizure then sale to JPMorgan then depositors and FDIC spared while shareholders and bondholders wiped](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-6.png)

The intuition: in a clean bank resolution, the depositors and the insurance fund are the protected class and the owners are the sacrificial one. WaMu is the textbook case — the largest failure ever, resolved at zero cost to the fund, with the loss perfectly contained to shareholders and bondholders.

## IndyMac: the run that came first

WaMu wasn't the only mortgage bank to fail in 2008, and it wasn't even the first big one. To understand WaMu's run, you have to understand IndyMac, which failed about ten weeks earlier and showed everyone — depositors and regulators alike — exactly how a mortgage bank dies.

IndyMac was a California-based thrift, originally spun out of the mortgage lender Countrywide, that had built itself into one of the largest savings institutions in the country on the back of **Alt-A** and **stated-income** loans. Alt-A loans sit between prime and subprime — borrowers with decent credit but little documentation. "Stated-income" loans, nicknamed "liar loans," let borrowers simply *declare* their income without proof. IndyMac was a high-volume originate-to-distribute machine: it made loans largely to sell them on. When the market for those loans froze in 2007–2008, IndyMac was stuck holding loans it couldn't sell, against a deteriorating housing market, funded by deposits.

In June 2008, a US senator publicly released a letter questioning IndyMac's solvency. That letter, widely reported, set off a depositor panic. Over about eleven days, customers withdrew roughly \$1.3 billion. On July 11, 2008, the OTS seized IndyMac and handed it to the FDIC. With about \$32 billion in assets, it was one of the largest failures up to that point — and, importantly, it was a *messier* resolution than WaMu's would be. The FDIC could not immediately arrange a buyer for the whole bank, so it set up a temporary "bridge bank" (IndyMac Federal) to run the institution while it sorted out the assets and reimbursed depositors. Crucially, IndyMac's failure *did* cost the FDIC insurance fund — billions of dollars — and it left some uninsured depositors only partially repaid.

That last point is what made IndyMac so dangerous to the rest of the system, and especially to WaMu. The images of customers queuing outside IndyMac branches, and the news that *uninsured* deposits above the limit might not come back in full, told every depositor in America a chilling lesson: a big, familiar bank can fail, and if you're over the insured limit, your money is at risk. IndyMac didn't just fail; it *taught depositors to run*. Ten weeks later, that lesson was fresh in the minds of WaMu's customers when Lehman fell.

IndyMac also illustrates the *originate-to-distribute* failure mode in its purest form, which is worth pausing on because it's slightly different from WaMu's. WaMu was primarily a *portfolio* lender for its option-ARMs — it held a lot of them on its own balance sheet — so its death was a classic asset-quality death: the loans it kept went bad. IndyMac, by contrast, lived more on the *conveyor belt*: it made loans largely to sell them, earning a fee on each one as it passed through. That model has a hidden fragility. As long as Wall Street keeps buying the loans, the conveyor belt spins and the fees roll in. But the moment the buyers vanish — as they did in 2007 when investors realised these loans were toxic — the lender is suddenly stuck holding a pile of freshly-made loans it *intended* to sell but now cannot, against a funding base (deposits and short-term borrowings) that assumed the loans would move quickly. The originate-to-distribute lender doesn't just suffer credit losses; it suffers a *warehouse* problem — inventory it can't move, financed by money it has to repay. That double-hit is why the pure originate-to-distribute lenders died fastest of all.

#### Worked example: insured versus uninsured in a messy resolution

IndyMac is the cleanest illustration of why the \$250,000 insurance limit matters so much. Suppose a small business kept \$500,000 in an IndyMac account to make payroll. The FDIC insurance limit at the time was \$100,000 per depositor (it was raised to \$250,000 a few months later, in October 2008, partly *because* of episodes like this).

When IndyMac failed, this depositor was insured for \$100,000 — that came back immediately, in full. The remaining \$400,000 was *uninsured*. The depositor became a creditor of the failed bank for that \$400,000, recovering only whatever the FDIC eventually realised from selling IndyMac's assets — reported at around 50 cents on the dollar for the uninsured portion in the early going. So on \$400,000 uninsured, the depositor might recover roughly \$200,000 over time, taking a real loss of around \$200,000 — plus the cash-flow nightmare of having that money frozen exactly when payroll was due.

The intuition: deposit insurance protects the *small* depositor completely and leaves the *large* one exposed. That asymmetry is why the run on WaMu — and later on SVB, whose deposits were 94% uninsured — was driven by businesses and wealthy customers with balances far above the limit. They had the most to lose and the most reason to move first. WaMu's saving grace versus IndyMac was that its sale to JPMorgan covered *all* depositors, insured and not; IndyMac's depositors weren't so lucky.

## The mortgage-lender wipeout: the wider casualty list

WaMu and IndyMac were the two big *deposit-taking banks* to fall, but they were part of a much larger massacre of mortgage originators. The originate-to-distribute model created a whole ecosystem of lenders whose entire business was making mortgages and selling them onward — and when the buyers for those mortgages disappeared in 2007–2008, the lenders died in a wave.

The most important of these was **Countrywide Financial**, which by the mid-2000s was the largest mortgage lender in the United States, originating roughly one in five US home loans at its peak. Countrywide was the purest expression of the originate-to-distribute model: it made enormous volumes of subprime and option-ARM loans, sold most of them to Wall Street to be securitized, and lived on the fees. But it was not primarily a deposit bank — its funding came heavily from the wholesale and securitization markets, not sticky retail deposits. So when the securitization market seized up in 2007, Countrywide's funding evaporated almost overnight, and it spiralled toward collapse. It didn't get *seized* like WaMu or IndyMac; instead, in early 2008, Bank of America bought it in a fire sale (a deal BofA would spend years and tens of billions of dollars regretting, as Countrywide's bad loans generated waves of lawsuits and settlements). Countrywide's near-death by funding-market freeze, rather than deposit run, is itself a useful contrast: a non-bank lender dies when the *capital markets* stop funding it, not when retail depositors flee.

Beyond the big three, dozens of mortgage specialists — New Century, American Home Mortgage, and many others — filed for bankruptcy in 2007–2008 as the loans they'd made went bad and the market to sell them vanished. This is the **shadow banking** dimension of the crisis: a vast amount of mortgage credit had been created by entities that *looked* like banks (they made loans) but funded themselves in the wholesale and repo markets rather than with insured deposits, which meant they had no deposit insurance, no central-bank backstop, and no cushion when their funding ran. (For how this parallel, unprotected banking system funds itself overnight and why that's so fragile, see [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).)

Let's lay the three deposit-and-mortgage casualties side by side, because their *differences* are as instructive as their common cause.

![Matrix comparing Washington Mutual IndyMac and Countrywide by size lending model and how each one ended](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-8.png)

All three died of the same disease — aggressive mortgage lending undone by falling house prices — but they met three different ends. WaMu, the biggest and a true deposit bank, was seized and cleanly sold to JPMorgan with the loss contained to its owners. IndyMac, a mid-sized deposit bank, was seized in a messier resolution that cost the insurance fund and burned some uninsured depositors. Countrywide, the volume king but not a deposit bank, was rescued by an acquirer when its capital-markets funding died. Same poison, three different funerals — and the difference in each case traces back to *how the institution was funded* and *who its creditors were*.

## How the failures connect back to what a bank is

Step back and the whole episode is one long demonstration of the spine of banking: a bank is a leveraged, confidence-funded maturity-transformation machine, and it survives only as long as its thin equity cushion absorbs losses faster than they arrive *and* its depositors keep trusting it.

WaMu broke the first condition. Its loans — option-ARMs and subprime — were so bad that the losses came faster than the 8% cushion could absorb. Once the loss estimate approached the capital, the bank was a dead institution walking. The credit problem then broke the second condition: a visibly insolvent bank cannot retain the trust of its depositors, so the run came, and the run executed a bank the loan book had already condemned.

This is also a story about *incentives all the way down*. The originate-to-distribute model removed the lender's skin in the game; the option-ARM's neg-am accounting let the bank book phantom profits on loans that were silently getting worse; the volume-obsessed growth strategy rewarded making more loans rather than better ones; and the assumption of ever-rising house prices papered over all of it. None of these were secret. They were the *business model*. The failure wasn't an accident that befell a sound bank; it was the predictable end state of a bank built to maximise mortgage volume in a rising market, run in reverse.

## Common misconceptions

**"WaMu failed because of the bank run."** No — the run was the *symptom*, not the disease. WaMu was already insolvent on a credit basis: its expected loan losses had grown to roughly the size of its entire equity cushion, which is why it had to raise \$7 billion in April 2008 and why its stock had collapsed to a few dollars. The run *finished* a bank that bad loans had already mortally wounded. A solvent bank can usually survive a run (it borrows against its good assets); an insolvent one cannot, because no one will lend against assets worth less than the loans. The run set the date of death; the loan book set the verdict.

**"WaMu and SVB failed the same way — both had runs."** They had runs, but for opposite reasons. SVB was a *rate-driven* failure: its assets were high-quality Treasuries and agency mortgage bonds with almost no credit risk, but rising rates created a \$17 billion unrealised loss, and a 36-hour, \$42-billion digital run forced it to crystallise that loss. WaMu was a *credit-driven* failure: its assets were bad loans whose losses were permanent regardless of what rates did. SVB's loss might have reversed if rates fell; WaMu's never could, because the money was genuinely gone.

**"The government bailed out WaMu."** It did not. There was no taxpayer money in WaMu's resolution, and the FDIC insurance fund spent nothing — the JPMorgan purchase covered the deposits. What the government *did* was facilitate an orderly seizure-and-sale that protected depositors. WaMu's shareholders and bondholders were *not* bailed out; they were wiped out. (IndyMac, by contrast, *did* cost the FDIC fund billions — not every failure is cost-free.)

**"Option-ARM borrowers were just irresponsible."** Some were stretching, certainly, but the product was engineered to be misunderstood. A loan that lets you pay so little that your balance *grows* — while your "payment" feels like a normal mortgage payment — is designed to obscure the eventual reset. Many borrowers genuinely did not understand that minimum payments were adding to their debt, or that the payment would multiply at recast. The asymmetry of understanding between the lender that designed the product and the household that took it is a big part of why the conduct fallout (and the lawsuits) ran for years.

**"WaMu was a small, reckless outfit that nobody should have trusted."** It was the sixth-largest bank in the United States, with about \$307 billion in assets and roughly 2,200 branches — a household name with a century-plus history. Its failure is a reminder that *size is not safety*. The single largest bank failure in American history was a big, familiar, mainstream institution. Scale just meant there was more of it to fall over.

![Before-and-after comparison contrasting a credit-driven failure like WaMu with a rate-driven failure like SVB](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-5.png)

## How it shows up in real banks

**The credit cycle versus the rate cycle — two ways to die.** The deepest lesson WaMu teaches is that banks die from *both* the asset-quality cycle and the interest-rate cycle, and the two look completely different from the outside. A credit-cycle death (WaMu 2008, the S&L crisis, most emerging-market banking crises) shows up as *rising loan-loss provisions* eating capital over quarters — you can watch it coming in the provisioning line of the income statement. A rate-cycle death (SVB 2023, much of the S&L crisis's *first* phase, Silicon Valley's duration trap) shows up as *unrealised losses on securities* from rising rates — you watch the held-to-maturity footnote, not the loan book. A complete bank analyst watches both: are the *loans* going bad (credit), and is the *value of the safe assets* falling because rates rose (rate)? WaMu and SVB are the two pure archetypes, fifteen years apart.

**Reading provisions as a leading indicator.** In WaMu's final year, the single most informative number was its loan-loss provision — the amount it set aside each quarter for expected bad loans. That number climbed quarter after quarter through 2007 and 2008, which was the market's early warning that credit losses were heading toward the capital base. Today, anyone reading a bank's results during a downturn should watch provisions the way a doctor watches a fever: a rising provision line on a mortgage-heavy lender in a falling-housing market is the credit-driven-failure signature. (The mechanics of how provisions are estimated under current rules are in [collateral, security and loan-loss provisioning](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl), and what happens to a bad loan once it defaults is in [non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process).)

**The deposit insurance limit, raised because of 2008.** The IndyMac and WaMu runs were driven heavily by *uninsured* depositors — businesses and wealthy individuals with balances above the limit, who had every reason to move first. In October 2008, partly in response, the US raised the FDIC insurance limit from \$100,000 to \$250,000 per depositor, where it remains. That single number change was designed to shrink the pool of "runnable" money. It worked imperfectly: in 2023, SVB's deposits were *94% uninsured* (a base of tech companies with huge balances), which is exactly why its run was the fastest in history. The lesson banks took from WaMu and IndyMac — that uninsured-deposit concentration is run-fuel — was a lesson the 2023 failures had to relearn.

**Who actually got wiped, in human terms.** It's worth being concrete about the resolution waterfall, because it determines who feels the loss. WaMu's depositors: untouched, became JPMorgan customers overnight. WaMu's roughly 43,000 employees: many lost their jobs as JPMorgan integrated the bank. WaMu's common shareholders — including a large number of ordinary retail investors and the bank's own employees who held stock — lost essentially everything. WaMu's bondholders: senior holders recovered a fraction in the years-long holding-company bankruptcy; subordinated holders recovered almost nothing. The TPG-led investors who had put in \$7 billion in April 2008: gone in five months. The structure protected the public and sacrificed the owners — exactly as the capital stack is designed to.

**Why bondholders, not just shareholders, mattered to the system.** WaMu's resolution sent one more signal that rippled far beyond Seattle: *senior bank-holding-company bondholders could lose money*. For years, the market had assumed that the bonds of a big bank were nearly as safe as deposits — that the government would never let large creditors take a loss. The WaMu seizure-and-sale, by wiping the holding company's bondholders, challenged that assumption hard. In the panicked days that followed, that contributed to the freezing of *all* bank funding markets, because if WaMu's senior bonds could go to pennies, so could anyone's. This is the uncomfortable tension at the heart of every resolution: imposing losses on bondholders is *correct* (they signed up for the risk and it disciplines lending), but doing it in the middle of a system-wide panic can deepen the panic. Regulators have wrestled with that trade-off ever since — it's the same tension that made the Credit Suisse AT1 wipeout in 2023 so controversial.

**The conduct tail that outlived the bank.** A credit-driven mortgage failure doesn't end with the seizure. The bad loans WaMu and Countrywide made generated *years* of litigation: investors who bought the mortgage bonds sued over how the loans were represented; regulators pursued the executives; the acquirers (JPMorgan and Bank of America) inherited tens of billions of dollars of legal liability for loans they hadn't even made. Bank of America's purchase of Countrywide is widely cited as one of the worst acquisitions in corporate history precisely because the legal tail was so long and so expensive. The lesson for any bank buying a failed lender: you're not just buying the assets, you're buying the *conduct history* that made them.

## The takeaway: how to read a credit-driven failure

If you take one thing from the WaMu story, make it the distinction between *how the assets are funded* and *how the assets can go wrong*, because those two axes between them explain almost every bank failure.

How the assets can go wrong gives you the *credit-versus-rate* axis. A credit-driven failure (WaMu) means the loans are genuinely bad and the loss is permanent — watch the provisioning line and ask whether expected losses are approaching the equity cushion. A rate-driven failure (SVB) means good assets lost market value when rates rose — watch the unrealised-loss footnote and ask whether a forced sale would crystallise more than the capital. The two demand completely different diagnostics, and conflating them (the way the financial press often does, lumping all failures together as "a run") blinds you to which one you're looking at.

How the assets are funded gives you the *runnability* axis. A bank funded by sticky, insured retail deposits (most of WaMu) runs *slowly* — its run took ten days. A bank funded by uninsured, concentrated, sophisticated depositors (SVB) runs in *hours*. A lender funded by wholesale and securitization markets (Countrywide) doesn't have a deposit run at all — it dies when the capital markets stop funding it. The funding base sets the *speed* of death; the asset quality sets the *certainty* of it.

Put the two together and you can place any failed bank on a map. WaMu: bad assets (credit), slow funding (retail deposits) — a slow, certain death finished by a grinding run. SVB: good assets that lost value (rate), fast funding (uninsured deposits) — a fast death triggered by the fastest run on record. Countrywide: bad assets (credit), market funding — death by funding freeze, no deposit run needed. The same two questions — *are the loans bad, and how runnable is the money?* — locate them all.

And the very last lesson is the one the \$307 billion headline carries: the biggest bank failure in American history was not exotic. It was a mainstream household-name bank that made too many bad mortgages, booked phantom profits on them, ran out of cushion when they went bad, and lost its depositors' trust at the worst possible moment. It is the spine of banking — borrow short, lend long, survive on confidence and a thin cushion — failing in the most ordinary way imaginable, just on the largest scale ever recorded. The next one will rhyme: watch the loan losses against the capital, and watch how runnable the money is, and you'll see it coming the way the OTS and the market saw WaMu coming — quarter by quarter, then all at once.

![Bar chart of FDIC-insured bank failures per year showing the 2008 to 2012 wave peaking at 157 in 2010](/imgs/blogs/washington-mutual-and-the-2008-mortgage-bank-failures-7.png)

WaMu didn't fail alone — it opened a wave. The chart shows FDIC-insured bank failures per year: a handful before 2008, then 25 in 2008, 140 in 2009, and a peak of **157 in 2010** before the cleanup tapered off. WaMu was the largest single casualty, but it was the leading edge of a credit-cycle massacre that took out hundreds of mostly smaller banks that had made the same bet on rising house prices. That is what a credit-driven banking crisis looks like at the system level: not one spectacular failure, but one giant one followed by years of smaller ones, all dying of the same loans going bad at the same time.

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the core machine that WaMu broke, explained from zero.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why the 8% cushion that WaMu's losses ate through is so thin by design.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the originate-to-distribute machine that removed the lender's skin in the game.
- [Non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process) — what happens to a mortgage once the borrower defaults, the other side of WaMu's losses.
- [Collateral, security and loan-loss provisioning (IFRS 9 and CECL)](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) — how the provisioning line that signalled WaMu's death is calculated.
- [Deposit insurance, the lender of last resort and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — why depositors were spared and the FDIC fund untouched.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — why a deposit run is the thing every bank fears most.
- [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the rate-driven failure to contrast with WaMu's credit-driven one.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — how the non-deposit lenders like Countrywide funded themselves and why that funding vanished.

*This is educational material about banking history and mechanics, not financial advice.*

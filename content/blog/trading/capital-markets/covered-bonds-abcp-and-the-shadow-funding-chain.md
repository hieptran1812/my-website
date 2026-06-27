---
title: "Covered Bonds, ABCP, and the Shadow Funding Chain: A Bank Built Outside the Bank"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How covered bonds, asset-backed commercial paper, and the conduit-and-money-fund chain create long-term credit from short-term savings — and why that chain freezes the moment trust breaks."
tags: ["capital-markets", "covered-bonds", "abcp", "shadow-banking", "securitization", "maturity-transformation", "repo", "money-market-funds", "liquidity-risk", "structured-finance"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Covered bonds and asset-backed commercial paper are two more ways to turn loans into tradable funding, and the conduits they feed quietly rebuild an entire bank — maturity transformation and all — outside the deposit-insured system.
>
> - A **covered bond** keeps its collateral pool on the issuer's balance sheet and gives investors **dual recourse** (to the bank *and* a ring-fenced pool); securitization sells the pool to an SPV and gives **single recourse** to that SPV alone.
> - **ABCP** (asset-backed commercial paper) is short-term paper a **conduit** issues to fund **long-term** assets — the classic maturity mismatch, with a sponsor bank's **liquidity line** as the only safety net.
> - Money funds → ABCP/repo → conduits → long assets is a **bank with no deposit insurance and no lender of last resort**; it works only while short-term lenders keep rolling.
> - The one number to remember: in August 2007 the US ABCP market shrank from about **\$1.2 trillion** toward **\$800 billion** in a matter of months as the roll stopped — the run that started the crisis.

On the morning of 9 August 2007, a quiet corner of the money market stopped working. BNP Paribas froze three funds that held US mortgage paper, saying it could no longer value them. Within hours, the buyers of asset-backed commercial paper — the safest, most boring short-term IOU in finance — simply stopped showing up. Programs that had rolled \$50 million of paper every morning for years suddenly could not place \$5 million. Nothing about the underlying mortgages had changed overnight. What changed was *trust* in the funding chain, and that was enough.

This is a post about the plumbing behind that morning. Most people learn the 2008 story as "subprime mortgages went bad." That is the spark, not the bomb. The bomb was a **funding structure** — a way of financing long-term credit with short-term money that had been assembled, piece by piece, outside the regulated banking system. To see why it was so fragile, you need to understand three things that rarely make the headlines: **covered bonds**, **ABCP**, and the **shadow funding chain** that strings money funds, conduits, and long-dated assets into a bank that nobody called a bank.

The thread running through all of it is the spine of this whole series: **securitization and its cousins are primary-market technologies for creating tradable funding, and they work only as long as the secondary and funding markets trust them.** When the trust is there, savings flow into 30-year mortgages through a chain of overnight promises. When it breaks, the chain seizes — and the long assets are still there, but nobody will fund them.

![Shadow funding chain from money funds through ABCP and repo to a conduit holding long-term assets](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-1.png)

## Foundations: maturity transformation, recourse, and what a bank actually does

Before the instruments, the core idea — because everything in this post is a variation on it.

A **bank** does something genuinely magical and genuinely dangerous: it takes money you can withdraw *today* (your deposit) and lends it out for *thirty years* (a mortgage). Your deposit is short. The mortgage is long. The bank bridges the gap and pockets the difference between the low rate it pays you and the higher rate the borrower pays. This is called **maturity transformation** — turning short-term savings into long-term loans — and it is the central trick of finance.

It is dangerous because of one word: **a run.** If everyone wants their short money back at once, the bank cannot call in a 30-year mortgage by lunchtime. The assets are long; the funding is short; the mismatch is structural. Society's answer to this danger was a three-part safety net wrapped around regulated banks: **deposit insurance** (the government guarantees your deposit, so you have no reason to run), a **lender of last resort** (the central bank lends against good collateral when private funding dries up), and **regulation** (capital and liquidity rules that limit how thin the bank can run).

Hold that picture, because the entire shadow funding chain is the same maturity transformation — *without* any of the three safety nets.

Two more terms you need:

- **Recourse** means: if the thing you lent against goes bad, *who* can you chase for your money? **Single recourse** = you can only go after one specific pool of assets. **Dual recourse** = you can go after two separate things, in sequence.
- **On vs. off balance sheet.** An asset is *on* an entity's balance sheet if that entity legally owns it and bears its gains and losses. It is *off* the balance sheet if it has been sold to a separate legal entity (a "special purpose vehicle," or **SPV**) that the original owner does not control. The whole point of moving assets off balance sheet is to make them — and their risk — *somebody else's problem*, at least on paper.

A useful way to keep the safety nets straight: a regulated bank has three independent defenses against a run, and each one corresponds to something the shadow chain lacks. Deposit insurance removes the depositor's *reason* to run (your money is guaranteed, so why panic). The lender of last resort removes the bank's *inability* to meet a run (the central bank lends against good collateral so the bank doesn't have to fire-sell). And capital-and-liquidity regulation reduces the *probability* of a run by keeping the bank from running too thin. Strip all three away and you have not removed the maturity transformation — you have only removed everything that made it survivable. That is the shadow chain in one sentence.

With those three ideas — maturity transformation, recourse, on/off balance sheet — the rest of this post is just bookkeeping with very high stakes.

If you want the base case before the variations, the sibling post on [securitization from first principles](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) builds the off-balance-sheet SPV machine from zero, and the [banking-side treatment](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) shows why a bank wants to do it in the first place.

## Covered bonds: a bond with two backstops

Start with the most conservative member of the family, because it is the clearest contrast.

A **covered bond** is a bond a bank issues to fund itself — say, to fund a book of mortgages. The bank tags a specific set of high-quality loans (the **cover pool**) and *ring-fences* them: legally segregates them so that if the bank fails, those loans are reserved first for covered-bond holders, not for general creditors. But — and this is the crucial part — **the loans stay on the bank's balance sheet.** The bank still owns them, still collects the payments, still bears the credit risk. It has simply promised that a particular pile of its assets stands behind a particular set of bonds.

So what does the investor get? **Dual recourse.** First, the bank owes you the money directly — a covered bond is a senior obligation of the issuer, like any other bond it sells. Second, *if the bank defaults*, you have a priority claim on the ring-fenced cover pool. You get two shots: the issuer, then the pool. And there is a third feature that securitization lacks — the pool is **dynamic**: if a mortgage in it goes delinquent or prepays, the bank is contractually required to *swap in a healthy loan* to keep the pool over-collateralized. The bank is on the hook to keep the collateral good.

![Covered bond keeps the pool on balance sheet with dual recourse versus securitization selling the pool to an SPV](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-2.png)

Compare that to **securitization**, the off-balance-sheet route. There, the bank *sells* the loans to an SPV. The SPV issues bonds backed only by those loans. The bank washes its hands: it has the cash, the loans are gone from its books, and if the SPV's pool defaults, investors have **single recourse** to the SPV — and *nothing* against the originating bank. The whole design is to isolate the bank from the pool. (When you then slice that SPV's bonds into risk tranches, you get the [CDOs and CLOs](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches) covered elsewhere in this series.)

That single difference — does the originator stay on the hook? — drives almost everything about how the two instruments behave in a crisis.

#### Worked example: dual recourse saving a covered-bond holder

You hold \$1,000,000 of covered bonds issued by a European mortgage bank, backed by a cover pool of prime mortgages. The pool is over-collateralized: the bank pledged \$1,250,000 of mortgages behind your \$1,000,000 (a 125% collateralization, or 25% over-collateralization).

Now a housing downturn hits and the pool underperforms. Suppose 10% of the pool's loans default and recover only half their value. The pool's value falls by roughly `\$1,250,000 × 10% × 50% = \$62,500`, to about \$1,187,500 — still comfortably above your \$1,000,000 claim, because of the cushion. You are made whole from the pool alone.

But here is the dual-recourse magic: *before* you ever touch the pool, the **bank itself** owes you the \$1,000,000 and keeps paying. The pool is the backstop to the backstop. For you to lose money, you need *both* the bank to fail *and* the ring-fenced pool to fall below your claim after over-collateralization. That double condition is why covered bonds have a near-spotless default history stretching back to 18th-century Prussia. **The intuition: two independent things both have to go wrong, so the bond is far safer than either backstop alone.**

### Why Europe loves covered bonds and the US leaned on securitization

This is one of the great structural divides in global finance, and it is not an accident.

**Europe** built its mortgage funding on covered bonds (German *Pfandbriefe* date to 1769). Banks kept the loans, kept the risk, and funded them with a transparent, dual-recourse bond governed by tight legal frameworks. Because the originator stays on the hook, the incentive to underwrite carefully never disappears — you cannot sell the loan and forget it. The market is large and, even through 2008, stayed open: covered bonds kept trading when securitization markets shut.

**The US** went the other way. With government-sponsored entities (Fannie Mae, Freddie Mac) standing behind conforming mortgages, and a deep appetite for tradable MBS, the American machine bolted toward **securitization** — sell the loan, move it off balance sheet, recycle the capital, originate the next one. That "originate-to-distribute" model is wildly efficient at moving credit. It is also, as the next sections show, structurally prone to a particular kind of failure, because the originator's skin in the game can evaporate the moment the loan is sold.

Neither model is "right." Covered bonds keep risk on the bank's balance sheet — safer for investors, but it ties up the bank's capital. Securitization frees the capital — efficient, but it can sever the link between *who makes the loan* and *who bears the loss*. Both are primary-market funding technologies. They differ in where the risk ends up sitting.

There is also a regulatory and legal reason for the divide. Covered bonds depend on a specific *legal framework* that gives bondholders their priority claim on the cover pool the instant the bank fails — without that statute, the "ring-fence" is just a contractual promise that a bankruptcy court might unwind. Continental Europe wrote those statutes centuries ago (Germany's Pfandbrief Act, Denmark's mortgage-credit system, France's *obligations foncières*). The US, lacking a comparable federal covered-bond law and *with* Fannie and Freddie already providing a deep, liquid outlet for conforming mortgages through securitization, simply never needed to build the covered-bond rails. The market structure followed the legal structure, which followed history. This is a recurring truth in capital markets: **the instruments a country uses are downstream of the laws it happened to write**, not of which design is theoretically superior.

A second-order consequence worth noticing: because the covered-bond issuer keeps the loans, it keeps a powerful incentive to underwrite them well. A bank that sells a loan and forgets it (the originate-to-distribute model) can grow sloppy — it earns its fee at origination and bears none of the default. A covered-bond bank earns nothing by lending to someone who will default, because the loan stays on its books and, worse, must be *swapped out of the cover pool* with a healthy loan if it goes bad. The funding structure quietly disciplines the lending decision. That alignment is a big part of why covered-bond pools have historically performed so well — the structure selects for careful lending.

### Where covered bonds sit relative to senior unsecured debt

One more nuance, because it explains who *loses* when a covered-bond bank fails. A bank's creditors stand in a queue. Covered-bond holders are near the front *and* have the pool; general ("senior unsecured") bondholders and uninsured depositors are further back with no pool. So when a bank fails, ring-fencing the best mortgages for covered-bond holders means there is *less* good collateral left for everyone else — a feature regulators call **asset encumbrance**. The more a bank funds itself with covered bonds, the more its remaining creditors are subordinated to that encumbered pool. This is the cost of the covered bond's safety: it is safe partly *because it pushes the risk onto the bank's other creditors*. Safety is rarely created; it is usually just relocated, and the skill is seeing where it went.

## ABCP: borrowing for 90 days to hold for 30 years

Now the dangerous cousin.

**Commercial paper** (CP) is the simplest corporate borrowing there is: a short-term IOU, usually 1–270 days, that a creditworthy issuer sells to investors who want a safe place to park cash for a few weeks. **Asset-backed** commercial paper (ABCP) is the same instrument, but issued not by an operating company — issued by a **conduit**: a special-purpose vehicle whose only job is to buy a pool of longer-term assets (mortgages, auto loans, credit-card receivables, even tranches of other securitizations) and fund them by continuously selling short-term paper.

Read that again, slowly, because it contains the whole problem. The conduit **holds assets that mature in 5, 10, even 30 years.** It **funds them with paper that matures in 30 to 90 days.** Every time a batch of paper comes due, the conduit must sell *new* paper to repay the old — it must **roll** the funding. As long as new buyers keep showing up each morning, the machine hums. The conduit earns the spread between the long asset yield and the short CP rate, exactly like a bank earns the spread between mortgages and deposits.

It *is* a bank. It is just a bank with no deposits, no deposit insurance, no banking license, and — until 2008 taught everyone otherwise — no lender of last resort.

![US securitization issuance by year showing the 2008 collapse](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-4.png)

The one thing standing between an ABCP conduit and disaster is the **liquidity backstop**: a committed credit line from a **sponsor bank**, promising that if the conduit ever can't sell new paper, the bank will lend it the cash to repay maturing paper. That backstop is what lets the conduit's paper earn a top short-term rating in the first place — investors are really lending against the *bank's* promise, not the conduit's assets. Which means the risk was never actually off the bank's balance sheet. It was off the balance sheet *until the day it wasn't*.

#### Worked example: a conduit funding \$10B of 5-year assets with 90-day paper

A conduit buys \$10,000,000,000 of 5-year asset-backed securities yielding 5.5%. It funds them by selling 90-day ABCP at 5.0%. The spread is 0.5%, so on \$10B the conduit nets roughly `\$10,000,000,000 × 0.5% = \$50,000,000` per year — paid to the sponsor and the program's equity holders for running the machine.

But the asset lasts 5 years and the paper lasts 90 days. Over 5 years there are about `5 × (365 / 90) ≈ 20` rollovers. Twenty separate mornings on which \$10B of new buyers must materialize to repay \$10B of departing buyers. If *nineteen* of those rolls go fine and *one* fails, the conduit is short \$10B in cash against assets it cannot sell that day. **The intuition: a 0.5% spread is paid for taking roll risk twenty times, and the twentieth roll is just as dangerous as the first.**

![Roll-risk timeline showing a 5-year asset refinanced about twenty times with 90-day paper](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-8.png)

That \$50M/year looks like easy money in the nineteen good years. It is the premium for a fat-tailed risk that shows up all at once. Repo financing — the other short-term funding rail, covered in the sibling post on [securities lending and repo](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — has the exact same shape: borrow overnight against long collateral, roll every morning, pray the haircut doesn't jump.

### Conduits, SIVs, and the spectrum of backstops

Not all of these vehicles were equally reckless, and the differences are exactly where the 2007 failures clustered. It helps to lay them on a spectrum by how much liquidity backstop they carried.

A **fully-supported conduit** had a liquidity line covering essentially 100% of its outstanding paper. If it ever failed to roll, the sponsor bank's committed line repaid every maturing note in full. Investors in that paper were really taking *bank* credit risk, lightly dressed as asset risk. These were the safest — and, predictably, the ones whose losses came straight back to the sponsor.

A **partially-supported conduit** carried a line covering only some fraction of its paper — the sponsor reasoned that asset quality and over-collateralization would cover the rest, so why pay for a full backstop. This is cheaper in good times and lethal in a run, because the uncovered slice has no backstop at all.

A **SIV (structured investment vehicle)** went furthest. It ran with only a *thin* liquidity line — often well under 10% of outstandings — on the theory that it could always sell its high-quality assets to raise cash if paper buyers vanished. That theory had one fatal assumption: that the assets stay liquid and fairly priced in a stress. They do not. In 2007, SIV assets became unsellable at any sane price exactly when the SIVs needed to sell them, and the thin backstops were instantly overwhelmed. SIVs were the first vehicles to die.

The lesson generalizes well beyond 2007: **a liquidity backstop sized for normal times is no backstop at all, because the only time you ever draw it is precisely when normal times have ended.** A safety net you can use only when you don't need it is not a safety net.

#### Worked example: how a thin SIV backstop runs out

A SIV holds \$20,000,000,000 of assets funded by ABCP and medium-term notes, with a liquidity line covering just 5% — \$1,000,000,000. Management's plan for a roll failure: draw the \$1B line and sell assets to cover the rest. Now a run hits and \$4B of paper comes due that the SIV cannot roll. The \$1B line covers a quarter of it; the SIV must sell \$3B of assets the same week. But every SIV is selling the same assets into a market with no buyers, so the \$3B clears at perhaps 85 cents, a `\$3,000,000,000 × 15% = \$450,000,000` loss — which erodes the SIV's thin equity and triggers covenants that force *more* selling. **The intuition: a backstop sized as a fraction of the funding is useless against a run on the whole funding, because runs take all of it at once.**

## The shadow funding chain: a bank assembled from spare parts

Step back and look at the whole chain, because the conduit is only the middle link.

On one end sit **money-market funds (MMFs)** — funds that hold savers' cash and promise it back on demand at a stable \$1.00 per share. To a saver, an MMF *feels* exactly like a bank deposit: safe, liquid, slightly higher yield. But an MMF is not a bank. It has no deposit insurance. It survives by holding only the safest, shortest paper — Treasury bills, repo, and... **ABCP**. Trillions of dollars of "cash" sit in money funds, and a big slice of it gets lent, every morning, into the short-term funding market.

On the other end sit **long-term assets** — mortgages, auto loans, student loans, CLO tranches — that need funding for years.

In the middle sit the **conduits and SIVs** (structured investment vehicles, a more aggressive conduit cousin that ran even thinner liquidity backstops). They borrow short from the money funds via ABCP and repo, and hold the long assets.

String it together and you have rebuilt a bank out of spare parts: **money funds = the depositors, ABCP/repo = the deposits, the conduit = the bank, the long assets = the loan book.** The maturity transformation is identical. The spread capture is identical. The run risk is identical. What is *missing* is identical too: no deposit insurance to stop the depositors (MMF investors) from fleeing, no lender of last resort to fund the conduit when private money vanishes, and — because it all happened off balance sheet — far lighter regulation than a real bank faced. This is what people mean by **"shadow banking"**: bank-like maturity transformation conducted outside the regulated, insured banking perimeter.

The genius of it, when trust holds, is real: it channels savers' idle overnight cash into 30-year credit at low cost, greasing the whole economy. This is the series spine in its purest form — **funding-market plumbing that turns short-term savings into long-term investment, as long as everyone trusts the chain.** The flaw is that it has all of a bank's fragility and none of a bank's protections.

![US debt issuance by type in 2023 on a log scale highlighting MBS and ABS](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-6.png)

The chart above puts the structured-funding pieces in context: Treasury issuance dwarfs everything, but MBS and ABS — the assets that conduits and SIVs loved to hold — are still a \$1.5T-plus and \$280B annual machine. That is a lot of long credit being created and needing somewhere to be funded.

### Why the chain forms at all: the demand for "safe" short assets

It is worth pausing on *why* this chain ever assembled itself, because it was not an accident or a fraud — it was the market answering a genuine demand. Corporations, pension funds, and money funds sit on enormous piles of cash that need a home for a few days or weeks: safe, liquid, and yielding a touch more than a bank account. The supply of genuinely safe short paper — Treasury bills — is finite and set by government borrowing. The gap between the *demand* for safe short assets and the *supply* of them is huge, and ABCP and repo grew to fill it. Structured finance was, in effect, *manufacturing* safe-looking short-term assets out of risky long-term ones. The conduit took a pool of 30-year mortgages and emitted a 30-day note that looked, smelled, and was rated like a T-bill.

That manufacturing is the deep magic and the deep danger at once. It is magic because it lets the economy fund far more long-term investment than the supply of patient long-term savings would otherwise allow. It is dangerous because the "safety" of the manufactured short asset is not intrinsic — it depends on the chain continuing to roll, which depends on confidence, which is exactly the thing that evaporates in a panic. A T-bill is safe because the government will print the money to repay it. ABCP only *looked* as safe as a T-bill. The difference between "is safe" and "looks safe" is invisible for years and then, on one morning, is the only thing that matters.

### Money funds: the depositor end of the chain

Look harder at the money-fund end, because that is where the run actually begins. A money fund promises its investors a stable \$1.00 net asset value and same-day redemption. To deliver that, it must hold only assets it believes are money-good and instantly saleable. ABCP, top-rated and short, qualified — so money funds became the single largest buyers of it. The investor in the money fund believed they held "cash." The money fund believed it held a safe short asset. The conduit believed it held a fundable long asset. *Every link in the chain believed it held something safer than the link below it.* That stacked belief is what made the chain so large and so cheap — and it is also why the unwind was so violent, because the moment any link doubted, the doubt propagated up *and* down simultaneously.

When a money fund grows nervous about a conduit, it does the only prudent thing: it lets its ABCP mature and does not buy more, shifting into Treasury bills instead. Multiply that across every money fund at once and the conduit's entire buyer base has vanished in days — not because the mortgages defaulted, but because the funders, each acting safely for their own investors, collectively pulled the funding. The run is an *emergent* property of everyone being individually prudent. There is no villain required.

#### Worked example: the funding hole when ABCP can't be rolled

Take our \$10B conduit. On a normal morning, \$2B of its paper matures and it sells \$2B of fresh paper to repay it — net cash movement zero. Now take a roll morning where, spooked by bad headlines, buyers will only absorb \$1.2B of new paper. The conduit owes \$2B and has raised \$1.2B. It is **\$800,000,000 short** by the close, against \$10B of assets it cannot sell in a day without taking a brutal loss.

Its options are ugly. Fire-sell assets: if a forced sale of \$800M of 5-year ABS clears at, say, 92 cents on the dollar instead of 100, that is an instant `\$800,000,000 × 8% = \$64,000,000` loss — and the fire sale pushes prices down further for everyone holding the same paper. Or draw the sponsor's liquidity line for the \$800M. Either way the loss or the asset lands back where it started. **The intuition: a short-term funding gap forces a long-term asset to be sold at a short-term price, and the gap between those two prices is the cost of the mismatch.**

## Why the mismatch is fragile: the anatomy of a run

A bank run and an ABCP run are the same physics. The trigger is doubt. Once a CP buyer suspects the conduit's assets might be worth less than face — or simply suspects that *other* buyers suspect it — the rational move is to not roll. Refuse to buy the new paper, take your cash back when the old paper matures, and wait. The problem is that *everyone* reasons this way at once. There is a first-mover advantage to running: the early refusers get repaid in full from the liquidity line or the first fire sales; the stragglers are left holding a conduit with a hole in it. So the run is self-fulfilling, exactly like a deposit run.

![Run dynamics where commercial-paper investors refuse to roll forcing fire sales or a sponsor draw](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-3.png)

Watch what happens next, following the figure. CP investors won't roll, so the conduit needs cash to repay maturing paper. Two branches. **Branch one: fire-sell the long assets.** But every conduit hit by the same shock is selling the same assets into the same frightened market at the same time, so prices gap down, the losses crystallize, and the funding hole gets *worse*, not better. **Branch two: draw the sponsor bank's liquidity line.** The bank now has to find real cash to honor a promise it hoped never to keep — and the conduit's long assets come *back onto the bank's balance sheet*, exactly the risk the off-balance-sheet structure was supposed to remove.

There is a vicious feedback loop hiding in branch one that deserves spelling out. When one conduit fire-sells assets, the market price of those assets drops. But other conduits and SIVs hold the *same* assets and mark them to that new, lower market price — which erodes their over-collateralization, spooks *their* CP buyers, and pushes *them* toward not rolling or fire-selling too. One forced seller manufactures the doubt that creates the next forced seller. This is why funding runs are not gradual: the fire-sale channel turns a localized shock into a market-wide one through the price of the shared collateral. Economists call it a "fire-sale externality" — each seller, acting rationally for itself, imposes a loss on every other holder, and no one internalizes the damage. A market that was deep and liquid on Monday can be a vacuum by Friday, with no change in the underlying loans at all.

That second branch is the cruel punchline of shadow banking. The whole architecture existed to move assets and risk *off* the bank. But the liquidity backstop is a tether. When the run comes, the tether yanks the assets straight back, and the bank discovers it owned the risk all along — now crystallizing at the worst possible moment, when its own funding is also under pressure and its capital is least able to absorb the hit.

#### Worked example: the sponsor bank's liquidity line being drawn

A sponsor bank backstops three conduits totaling \$30B of ABCP. It carries this as an *off-balance-sheet commitment* — a footnote, costing almost nothing in good times. In the run, all three conduits fail to roll simultaneously and draw their lines. The bank must suddenly fund \$30B in cash and take \$30B of long assets onto its balance sheet.

If those assets are now marked at 90 cents, the bank books a `\$30,000,000,000 × 10% = \$3,000,000,000` loss the instant they arrive — and \$30B of new assets demands new regulatory capital the bank had not reserved, against a \$30B funding need it must meet while its *own* short-term funding is freezing. A footnote became a balance-sheet-threatening event over a single weekend. **The intuition: an off-balance-sheet promise is free until the day it isn't, and that day it can be the largest single liability the bank has.**

This is why what looks like a clever way to economize on capital is really a way to *hide* a contingent liability — and contingent liabilities have a habit of all coming due in the same week.

## How it shows up in real markets: the 2007 ABCP run

This is not a thought experiment. It happened, and it is the prologue to the case study in the sibling post on [2008, when the securitization machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study).

By mid-2007 the US ABCP market was roughly **\$1.2 trillion** — bigger than the entire stock of Treasury bills at the time. A meaningful slice of those conduits and SIVs held, directly or indirectly, subprime-mortgage exposure. And subprime origination had exploded in the run-up, as the chart below shows: from under \$200B a year in 2001 to over \$600B by 2005–06, before collapsing.

![US subprime mortgage origination peaking before 2008 then collapsing](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-5.png)

When BNP Paribas froze its funds on 9 August 2007, the doubt arrived. Buyers could no longer tell which conduits held the bad paper, so — rationally — they refused to roll *any* of it. The ABCP market contracted by hundreds of billions of dollars within weeks, sliding from about \$1.2T toward \$800B over the following months. SIVs, which ran with the thinnest liquidity backstops, were the first to die; most were wound down or hauled back onto sponsors' balance sheets by late 2007.

The contagion path was precisely the second branch of the run figure. Sponsor banks were forced to draw or honor liquidity lines, taking tens of billions of assets back onto their books. That consumed capital and cash at the exact moment interbank funding was freezing. The line from "a French bank can't value three funds" to "global banks are capital-impaired" ran straight through the ABCP conduit. The assets themselves — the actual mortgages — defaulted only gradually over the following two years. The *funding* collapsed in weeks. The run came first.

And the contrast with covered bonds is the tell. Through the same period, European **covered-bond** markets stayed open and kept funding mortgages, precisely because their dual-recourse, on-balance-sheet structure gave investors no reason to run: the bank was still on the hook, the pool was still ring-fenced and over-collateralized, and there was no overnight roll to refuse. Same underlying asset class — mortgages — wildly different funding fragility. The difference was entirely in the *structure of the promise*, which is the whole lesson of this post.

![Global capital-market size comparing equity and bond markets](/imgs/blogs/covered-bonds-abcp-and-the-shadow-funding-chain-7.png)

For scale: global bond markets run to roughly \$140 trillion, larger than global equity. A huge share of that is funded, refinanced, and warehoused through exactly the short-term plumbing this post describes. When that plumbing trusts itself, it is invisible. When it doesn't, it is the whole story.

### The official-sector response, and why it had to happen

The run could not stop itself, because no private actor had both the incentive and the capacity to be the buyer of last resort. So the official sector stepped in — and the *form* of its intervention tells you exactly which safety net the shadow chain had been missing. The Federal Reserve created facilities whose names read like a list of the broken links: the AMLF (to lend against ABCP so money funds could sell it), the CPFF (to buy commercial paper directly when no one else would), the MMIFF, and a guarantee program for money-fund balances after the Reserve Primary Fund broke the buck. Each one was the central bank improvising the *lender-of-last-resort* function for a part of the financial system that had grown bank-like without ever getting a banking system's protections.

That is the deep verdict on shadow banking. The chain had recreated a bank's maturity transformation outside the perimeter, capturing the spread in good times. When it ran, the public sector had to extend the bank safety net to it anyway — because letting a \$1.2T funding market collapse would have taken the real economy with it. The privatized-gains, socialized-losses asymmetry that critics describe is not a moral accusation so much as a structural inevitability: **if you let bank-like risk grow outside the safety net, you will end up extending the safety net to it in the crisis, having collected none of the insurance premium in advance.** The post-crisis reforms — consolidating conduits onto sponsor balance sheets, money-fund floating-NAV rules, liquidity-coverage ratios — are all attempts to either bring the risk inside the perimeter or stop it forming. They work until the next mismatch finds a corner the rules don't reach.

### A Vietnam-scale footnote on the same shape

The shape is universal, even where the labels differ. Vietnam's corporate-bond episode of 2022 rhymed with it: property developers had funded long-dated projects with short-dated bonds sold heavily to retail investors, often rolled rather than repaid. When confidence cracked after high-profile arrests and a regulatory tightening, the roll stopped, issuers could not refinance, and a funding freeze hit assets that were fundamentally illiquid (half-built projects) rather than fundamentally worthless. Different country, different instrument, no money funds or conduits — but the same maturity mismatch and the same run dynamic: long assets, short funding, and confidence as the load-bearing wall. The series' sibling on [foreign flows in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) traces how external funding can amplify exactly this kind of local fragility.

## Common misconceptions

**"Off-balance-sheet means the risk is really gone."** No. The liquidity backstop is a tether that pulls the assets back onto the sponsor's balance sheet in a run. In 2007–08, banks took tens of billions of "off-balance-sheet" conduit assets back on, plus they often held the equity or first-loss piece of their own securitizations. The risk was relocated, relabeled, and lightly capitalized — not removed.

**"ABCP is safe because it's top-rated and short-term."** The top rating came largely from the *sponsor bank's liquidity promise*, not the conduit's assets. "Short-term" is the source of the danger, not a comfort: it means the funding must be rolled constantly, and a single failed roll is a crisis. Short maturity reduces *interest-rate* risk and amplifies *rollover* risk.

**"A covered bond is just a securitization that stayed on the balance sheet."** The staying-on-the-balance-sheet *is* the whole difference, and it changes the recourse, the incentive to underwrite well, the dynamic over-collateralization, and the behavior in a crisis. Single recourse to an isolated SPV versus dual recourse to a bank-plus-pool is not a detail — it is the entire risk profile.

**"Money-market funds are basically insured savings accounts."** They are not insured at all. They hold short paper and *aim* to keep a stable \$1.00 share price, but when the Reserve Primary Fund's Lehman holdings went bad in September 2008, it "broke the buck" (fell below \$1.00), triggering a classic run on money funds. That is the depositor end of the shadow chain running.

**"This was a one-off that's been regulated away."** The specific SIV structure is mostly dead, and post-crisis rules (consolidation accounting, liquidity coverage, money-fund reform) tightened the worst of it. But maturity transformation outside the banking perimeter is a permanent feature of finance — it migrates to wherever the rules are lightest. The shape recurs; only the labels change.

## The takeaway: trust is the load-bearing wall

The lesson of covered bonds, ABCP, and the shadow chain is not "structured finance is bad." It is that **every funding structure is a particular arrangement of promises, and its fragility lives entirely in how those promises behave when people stop trusting them.**

A covered bond is a strong promise: the issuer owes you, *and* a ring-fenced pool backs you, *and* there's no overnight roll to refuse. It is hard to run on. An ABCP conduit is a chain of weak, renewable promises: I'll fund you today and decide again tomorrow. It is trivially easy to run on, because not-rolling requires no action at all — you just don't show up. Same underlying loans; opposite fragility. The difference is structural, and it is knowable *in advance*, before any crisis, by anyone who reads the funding side of the balance sheet rather than just the asset side.

That is the practical skill this post hands you. When you look at any financial institution — a bank, a fund, a conduit, a stablecoin issuer, a fintech lender — don't just ask "what does it own?" Ask **"how is it funded, how short is that funding, and who is promised to step in when the funding refuses to roll?"** If the answer is "long assets, very short funding, and a backstop that only works if it's never needed all at once," you are looking at a bank built outside the bank — and you know exactly how it ends.

This is the series spine stated in its sharpest form: secondary and funding markets are what make primary issuance possible, because **nobody funds a 30-year asset unless they believe they can refinance their short claim on it tomorrow morning.** The shadow chain is that belief, industrialized. When the belief holds, savings flow into long-term credit and the economy grows. When it breaks — on a single August morning — the assets are still there, still paying, and the funding is simply gone. That gap, between an asset that's fine and funding that's vanished, is the most dangerous space in all of finance.

## Further reading & cross-links

- [Securitization from first principles: turning loans into bonds](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) — the off-balance-sheet SPV base case this post contrasts against.
- [CDOs, CLOs, and the tranching of tranches](/blog/trading/capital-markets/cdos-clos-and-the-tranching-of-tranches) — how the bonds these conduits held got sliced into risk layers.
- [2008: when the securitization machine broke (case study)](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study) — the full crisis this funding chain set off.
- [Money market vs. capital market: where short meets long](/blog/trading/capital-markets/money-market-vs-capital-market-where-short-meets-long) — the short-funding / long-asset boundary that the shadow chain straddles.
- [Securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — the other short-term funding rail with the same roll-risk shape.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the commercial-bank-side view of why a bank wants to move loans off balance sheet.

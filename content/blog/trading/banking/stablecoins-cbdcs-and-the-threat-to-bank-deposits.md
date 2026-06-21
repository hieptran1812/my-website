---
title: "Stablecoins, CBDCs and the Threat to Bank Deposits: Where the Funding Goes When Money Goes Digital"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How fully-reserved stablecoins and central-bank digital money could drain the cheap deposits that fund bank lending, why a digital dollar that does not lend shrinks credit, and how design choices decide the damage."
tags: ["banking", "stablecoins", "cbdc", "deposit-disintermediation", "narrow-banking", "bank-funding", "credit-creation", "digital-dollar", "run-risk", "fractional-reserve"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank deposit is not just money you parked; it is the raw material a bank lends out to create credit. A fully-reserved stablecoin or a central-bank digital currency is money that *cannot* be lent. If the public moves savings from one to the other, the banking system loses its cheapest funding and the economy loses the loans that funding supported.
>
> - A typical large bank funds itself about **71% from deposits**, the cheapest money it has; it lends out roughly **90 cents of every deposit dollar** and keeps the rest as reserves.
> - A stablecoin issuer holds its backing **1:1 in Treasury bills and bank reserves** and makes **no loans** — it is a "narrow bank" by construction, so a dollar that moves there stops creating credit.
> - If **15% of deposits** migrated to digital dollars, a bank funding 71% from deposits would have to replace that gap with **costlier wholesale money** (its deposit share could fall to about 56%), squeezing its margin; on a \$1,000bn deposit base, a 15% migration removes roughly **\$135bn of lending capacity**.
> - **CBDC design decides the damage**: a low per-person **holding cap** (the ECB has discussed figures around €3,000) and a **non-remunerated** design (pays no interest) keep deposits mostly safe; an uncapped, interest-paying CBDC could trigger deposit flight.
> - The one number to remember: a deposit dollar funds about **\$0.90 of loans**; a stablecoin or CBDC dollar funds **\$0**.

In March 2023, a stablecoin meant to always be worth exactly one dollar briefly fell to 88 cents. USDC, the second-largest dollar stablecoin, held part of its cash reserves at Silicon Valley Bank — and over the weekend that SVB failed, traders feared that money was trapped. For about 48 hours, a token that exists *because* it promises to be a perfect digital dollar was trading like a distressed asset. It snapped back to a dollar only after US regulators guaranteed SVB's deposits. The episode looked like a crypto story. It was really a banking story: a stablecoin's safety is exactly as good as where its reserves sit, and where its reserves sit is a bank.

That single weekend contains the whole subject of this post in miniature. Digital dollars — privately issued stablecoins and the central-bank digital currencies (CBDCs) that governments are piloting — are colliding with the oldest machine in finance: the deposit-funded, loan-making bank. The collision matters because of one structural fact most coverage skips. When you hold money as a bank deposit, the bank lends most of it out and creates credit. When you hold the same money as a stablecoin or a CBDC, by design it is *not* lent out — it sits in reserves and Treasury bills. Move enough money from the first form to the second, and you are not just changing how people pay. You are quietly defunding the credit machine.

The figure below is the mental model for the entire post: follow one \$1,000, first into a bank deposit, then into a stablecoin, and watch what happens to the lending it used to support.

![A bank deposit funds a loan while a stablecoin sits in reserves and funds nothing](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-1.png)

This is the deposit-disintermediation threat, and it traces straight back to the spine of this whole series: a bank is a leveraged, confidence-funded maturity-transformation machine. It borrows short (your deposits) and lends long (mortgages, business loans), earning the spread in between. Take away the cheap short borrowing, and the long lending has to shrink or get more expensive — either way the machine throttles down. Stablecoins and CBDCs threaten the bank at the most fundamental layer there is: its funding base.

## Foundations: deposits, full reserves, narrow banks, and digital dollars

Before we can see the threat, we need to define the pieces from zero. None of this requires prior banking knowledge — just patience to build the ideas one at a time.

### What a bank deposit really is

When you put \$1,000 into a checking account, you have not stored \$1,000 in a vault with your name on it. You have *lent* \$1,000 to the bank. The bank now owes you \$1,000 that you can demand back at any moment, and in exchange it can use that money. This is the single most counterintuitive fact in banking and everything below depends on it: **your deposit is the bank's liability and the bank's funding all at once.** It is the cheapest money the bank will ever get, because most depositors accept little or no interest in return for safety and convenience.

A bank does not keep all your \$1,000 idle. It keeps a fraction — call it 10% — as **reserves**, which are cash plus money held at the central bank, available to meet withdrawals. The rest it lends out. This is *fractional-reserve banking*: the bank holds only a fraction of deposits in reserve and puts the rest to work. We will use a clean rule of thumb throughout: a bank lends out about **\$0.90 of every \$1.00** of deposits and holds about \$0.10 as reserves and cash. (Real reserve ratios vary, but 90% lent is a good, honest illustration.)

### What "full reserve" means

A **fully-reserved** institution does the opposite. For every \$1.00 it holds for you, it keeps \$1.00 in safe, liquid assets — cash, central-bank reserves, or short-term government bills — and lends out **\$0.00**. It cannot fail to give you your money back, because it never let go of any of it. The catch: it creates no credit and earns no lending spread. It is a coat-check that genuinely keeps every coat, never lending a single one out.

### What a narrow bank is

A **narrow bank** is the formal name for a fully-reserved deposit institution: it takes deposits, holds them entirely in safe assets, and makes no loans. The idea is old — economists proposed it in the 1930s (the "Chicago Plan") to make banking run-proof. The whole appeal of a narrow bank is that it can *never* suffer a run, because it can always pay everyone back at once. The whole problem with a narrow bank is that it funds nothing: take the lending out of a bank and you have removed the reason an economy has banks. Keep this term in your pocket — because **a fully-reserved stablecoin is, economically, a narrow bank that happens to live on a blockchain.**

### What a stablecoin is

A **stablecoin** is a privately issued digital token designed to always be worth a fixed amount of an underlying currency — almost always one US dollar. The relevant kind here is the *fully-reserved* (or "fiat-backed") stablecoin: for every token in circulation, the issuer claims to hold one dollar of safe assets in reserve. USDC (issued by Circle) and USDT (issued by Tether) are the two giants; together they account for the large majority of the roughly \$160–250bn of stablecoins outstanding in 2024–2025. You hand the issuer a dollar, it mints you a token; you hand back the token, it burns it and returns your dollar. The token can move on a blockchain 24/7, instantly, anywhere. (For the full mechanics of how these are minted, redeemed and audited, see the crypto deep-dive linked at the end — here we care only about what they do to *banks*.)

The crucial property: **a fully-reserved stablecoin does not lend.** Its reserves sit in Treasury bills and bank reserves. It is a narrow bank wearing a crypto costume.

### What a CBDC is

A **central-bank digital currency (CBDC)** is digital money issued directly by a central bank — a digital version of physical cash. Unlike a deposit, which is a claim on a *commercial* bank, a CBDC is a direct claim on the *central bank* itself, the safest counterparty in the currency. Over 130 countries representing most of world GDP have explored a CBDC; China's e-CNY is the largest live pilot, and the European Central Bank has been designing a "digital euro." A retail CBDC, like a stablecoin, would be money the central bank does not lend out — it sits as a liability on the central bank's own balance sheet, backed by the central bank's assets. (The systemic, monetary-policy view of CBDCs is covered in the finance one-pager linked below; here we stay on the bank-funding angle.)

### What deposit disintermediation is

Put these together and you get the threat. **Disintermediation** means cutting out the middleman. **Deposit disintermediation** is the public moving money *out of* bank deposits — the funding the bank uses to make loans — and *into* something that sits outside the lending system, like a stablecoin or a CBDC. The bank, the "intermediary" between savers and borrowers, gets cut out of the loop. The money still exists; it just stops doing the one thing that creates credit.

### A maturity-transformation recap

One more recap, because it is the heartbeat of the whole series. **Maturity transformation** is the bank's core trick: it borrows *short* (deposits you can withdraw any second) and lends *long* (a 30-year mortgage, a 5-year business loan). It profits from the gap between the low rate it pays on short money and the higher rate it earns on long lending — the spread. This trick is what makes a bank both essential (it turns idle savings into long-term credit) and fragile (it can never repay all the short borrowing at once, because the money is tied up in long loans). Stablecoins and CBDCs attack the *short borrowing* side of this trade. That is why they are dangerous to banks even though they touch no loan directly.

### The three things, side by side

All three instruments are dollars you can spend. From the spender's seat they feel almost identical — a balance in an app, a number you can send to someone else. The differences that matter are invisible to the spender and decisive for the banking system: *who* holds your money, *whether* that holder lends it, whether the government insures it, and how fast it can run. The matrix below lays out the comparison that the rest of this post unpacks.

![Matrix comparing a bank deposit a stablecoin and a CBDC across who holds it lending insurance and run risk](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-4.png)

Read down the second row — *does it fund lending* — and you already have the thesis. A bank deposit recycles roughly \$0.90 of every dollar into loans. A stablecoin and a CBDC recycle \$0.00. The first row tells you *why*: a deposit sits with a commercial bank whose business is lending, while the digital dollars sit with an issuer or a central bank whose reserves are, by rule, parked in safe assets. The third and fourth rows are the price of that safety and the source of the next crisis, and we will come to both. For now, fix the second row in your mind: **only the bank deposit funds credit, and that is the entire reason moving money out of deposits matters.**

Now we can go deep.

## How one deposit funds lending — and why that is the thing at risk

Let us make this concrete with the simplest possible bank. Take a single bank that has just opened, with one customer and \$1,000 in equity (the owners' own money). We will watch a deposit turn into a loan, step by step, the way the cover figure sketched it.

#### Worked example: a deposit becomes a loan becomes more deposits

You deposit \$1,000 into the bank. The bank's balance sheet now shows:

- **Assets:** \$1,000 cash.
- **Liabilities:** \$1,000 it owes you (your deposit).

The bank keeps \$100 (10%) as reserves and lends \$900 to a local bakery. Now:

- **Assets:** \$100 reserves + \$900 loan = \$1,000.
- **Liabilities:** \$1,000 deposit (still owed to you).

Here is the magic. The bakery spends that \$900 paying a supplier, and the supplier deposits the \$900 into a bank. That bank keeps \$90 and lends \$810. The \$810 gets spent, redeposited, and \$729 gets lent again. Each round, 90% of the prior round flows on as new lending. Sum the whole infinite series:

$$\text{total lending} = 900 + 810 + 729 + \dots = \frac{900}{0.10} \times 0.90 = 9{,}000 \times 0.90 = \dots$$

Cleaner to compute the *deposits* created: the original \$1,000 supports total deposits of $1{,}000 \div 0.10 = \$10{,}000$, of which \$9,000 is new loans. Your single \$1,000 deposit, recycled through the banking system, becomes \$10,000 of deposits and \$9,000 of credit. **One sentence of intuition:** the deposit dollar does not just sit — it is the seed of a multiplier, and the more money is held as deposits, the more credit the system can create.

(This is the *money multiplier* in its textbook form. Modern banks are constrained more by capital and demand than by reserves, so the multiplier is looser than the arithmetic suggests — for the full system view of money creation, see the finance link at the end. But the direction is exactly right: deposits fund lending; remove deposits and you remove lending.)

### Why deposits are the cheapest funding a bank has

A bank can fund a loan from several sources, but they are not equal. The cheapest is a checking deposit, on which the bank often pays close to 0%. A savings deposit costs a little more. Beyond deposits, the bank can borrow in the *wholesale* market — from other banks, money-market funds, or by issuing bonds — but that money is expensive and skittish: it reprices to the market rate immediately and flees at the first sign of trouble.

This is why the entire series keeps insisting that **a cheap, sticky deposit base is the franchise.** A bank that funds itself at 0.5% and lends at 6% earns a fat 5.5-point spread. Force it to replace those deposits with wholesale money at 5%, and the spread collapses to 1 point. Same loans, a fraction of the profit. The deposit base is not just *a* funding source; it is *the* source of the bank's profitability and resilience. (We covered this in depth in [retail deposits, the funding base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — the cheap-money-is-the-franchise post.)

There are two distinct properties that make a deposit valuable, and it helps to separate them, because stablecoins and CBDCs threaten both:

- **It is cheap.** A bank pays little or nothing on a checking balance. The depositor is effectively lending the bank money for free in exchange for safety and convenience. That free funding is pure margin.
- **It is sticky.** Most people leave their salary account at the same bank for years. They do not move it overnight chasing a few basis points. That stability lets the bank fund a 30-year mortgage with money that, contractually, could leave tomorrow — the whole maturity-transformation trick depends on the deposit *behaving* long even though it is legally short.

A stablecoin attacks *cheapness* — it offers the same spendable dollar without the bank capturing the float, and it pulls the dollar out of lending. A frictionless, app-based digital dollar also attacks *stickiness* — the easier it is to move money with a tap, the less inertia protects the deposit. The deeper danger is not that a competitor pays a higher rate; it is that money which used to *sit* now *moves*, and a deposit base that moves is no longer the stable, long-behaving funding the maturity-transformation trade requires.

A typical large bank's funding looks like this: about 71% deposits, 10% wholesale and repo, 7% long-term debt, 4% other liabilities, and 8% equity — which is roughly 12.5× leverage (one dollar of equity holding up \$12.50 of assets). The chart below shows that mix today, and what happens to it if a chunk of deposits walks out the door into digital dollars and the bank has to backfill the hole with costlier money.

![Bar chart of a bank funding mix today versus after deposits migrate to digital dollars](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-3.png)

Notice what the scenario bar does: deposits fall from 71% to 56% of funding, and the bank has to make up the missing 15 points with wholesale funding (up from 10% to 19%) and long-term debt (up from 7% to 13%). The bank's books still balance — but its *funding cost* just jumped, because the cheapest 15 points got swapped for the most expensive money on the menu. That is the disintermediation threat showing up as a number on the liability side.

## What stablecoins and CBDCs do to that machine

Now drop a stablecoin into the picture. The contrast is stark, and it is worth being precise about, because the popular framing — "crypto is competing with banks" — misses the actual mechanism.

When you move \$1,000 from your bank deposit into a stablecoin, here is the literal sequence of balance-sheet events:

1. You instruct the issuer to mint \$1,000 of stablecoin.
2. Your \$1,000 leaves your bank account and is paid to the issuer.
3. The issuer takes that \$1,000 and buys Treasury bills and/or parks cash in *its own* bank account (held as reserves, often at a few large banks).
4. You now hold a \$1,000 token. The issuer holds \$1,000 of safe assets. **No loan was made.**

The pipeline below traces this flow all the way to its consequence for the bank.

![Pipeline showing money leaving deposits into a stablecoin or CBDC and draining bank funding](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-2.png)

The key move is step 3. Your money did not vanish — it still exists as Treasury bills and reserves. But it left the *fractional-reserve* system (where 90 cents of it would have been lent) and entered a *fully-reserved* system (where 0 cents of it is lent). The credit that your deposit would have seeded is gone. This is the precise sense in which **a stablecoin is a narrow bank**: it intermediates dollars into safe assets, never into loans.

### The aggregate picture: where do the reserves go?

There is a subtlety worth pausing on, because it determines who actually loses. When you move \$1,000 to a stablecoin, the issuer might just redeposit that \$1,000 in *a* bank as reserves. If so, the deposit didn't leave the banking system — it just moved from your bank to the issuer's bank. Has anything changed?

Yes, two things. First, the *composition* changed: a sticky retail deposit (cheap, stable, the franchise) was replaced by a concentrated institutional deposit from one stablecoin issuer (large, lumpy, prone to leaving en masse when redemptions spike). Banks value that swap poorly. Second, and more importantly, issuers increasingly hold reserves in **Treasury bills**, not bank deposits — both because bills are safer and because regulators want it that way. When a stablecoin issuer buys a T-bill, the dollar leaves the commercial-banking system entirely and lands with the government. *That* dollar is now unavailable for lending by anyone. As stablecoins scale, the share of their reserves in bills rather than bank deposits is exactly the share that is genuinely disintermediated.

A CBDC is even cleaner in its effect: a retail CBDC is a direct claim on the central bank, so a dollar moved into a CBDC leaves the commercial-banking system by definition and sits on the central bank's balance sheet. There is no "issuer redeposits it" step. It is gone from lending, full stop.

#### Worked example: the lending multiplier lost when money moves to a full-reserve token

Let us quantify what a single \$1,000 migration costs in credit. In the bank, your \$1,000 seeded \$9,000 of new loans through the multiplier (computed above). Move that same \$1,000 into a fully-reserved stablecoin whose reserves go into T-bills, and:

- New lending seeded: \$0 (a narrow bank lends nothing).
- Multiplier effect: gone (there are no successive rounds of redeposit-and-lend).
- Credit created: \$0 instead of \$9,000.

Even using the conservative, capital-constrained view where the realistic multiplier is far smaller — say each deposit dollar supports only \$0.90 of *direct* first-round lending — the migration still destroys 90 cents of lending per dollar moved. **One sentence of intuition:** the damage from disintermediation is not the \$1,000 — the money still exists — it is the *lending the \$1,000 would have funded*, which evaporates the moment the dollar enters a system that does not lend.

Scale that one dollar up to a banking system and the loss becomes a number that shows up in the real economy. The chart below uses the conservative first-round measure — each deposit dollar funds about \$0.90 of direct lending — applied to a \$1,000bn deposit base, and asks: how much lending disappears as a rising share of those deposits migrates into digital dollars that do not lend?

![Bar chart of lending lost as a rising share of deposits migrates to digital dollars](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-7.png)

The line is brutally simple because the mechanism is simple. At a 10% migration, \$90bn of lending is no longer funded; at 15%, \$135bn; at 30%, \$270bn — and that is only the *direct* first-round loss, before any multiplier amplification. These are not small numbers in a system where credit growth tracks economic growth fairly closely. A persistent drain of this size does not announce itself as a crisis; it shows up quietly as loans that never got made — the mortgage the bank declined, the business line that got cut, the small firm that could not refinance. That is the texture of disintermediation: not a bang, but a credit channel that narrows year by year.

#### Worked example: the bank's funding-cost hit

Take a bank with \$1,000bn of deposits, paying an average 0.5% on them. Suppose 15% — \$150bn — migrates to stablecoins and CBDCs. The bank still wants to hold its assets, so it must replace \$150bn of funding. The cheapest replacement available is wholesale money at, say, 5%.

- Old funding cost on that slice: $150 \times 0.5\% = \$0.75bn$ per year.
- New funding cost on that slice: $150 \times 5\% = \$7.5bn$ per year.
- Extra cost: **\$6.75bn per year**, straight off pre-tax profit.

If the bank earned, say, \$20bn pre-tax before, this single funding swap cuts it by a third — without a single loan going bad. **One sentence of intuition:** disintermediation hurts the bank twice, once by shrinking how much it can lend, and again by raising what it pays for whatever funding it keeps; the spread, the bank's whole reason to exist, gets squeezed from both ends.

## A stablecoin's reserves and where the income goes

If the dollar you moved into a stablecoin earns interest in Treasury bills, who keeps that interest? Not you. **The issuer does.** This is the business model, and understanding it explains both why stablecoins exploded when rates rose and why they are a structural competitor to bank deposits.

The chart below shows, on the left, the rough shape of a large stablecoin's reserves (modeled on what Circle has disclosed for USDC: the bulk in short-dated government securities via a government money-market fund, a slice in cash at banks), and on the right, the income that book throws off.

![Stablecoin reserve composition and the issuer reserve income it earns](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-5.png)

#### Worked example: a stablecoin issuer's reserve income

Take a stablecoin with \$60bn of tokens outstanding, fully backed by reserves, mostly in Treasury bills yielding 5%.

- Gross reserve income: $60 \times 5\% = \$3.0bn$ per year.
- Less operating, custody and audit costs of, say, \$0.6bn.
- Net to the issuer: **\$2.4bn per year.**

The token holder — you — earns **0%**. You handed over the right to the interest in exchange for a stable, fast, programmable dollar. **One sentence of intuition:** a stablecoin is a machine for capturing the interest on your money while paying you nothing for it, which is exactly the deal a checking account already offers — so the moment stablecoins offer something a checking account cannot (instant global settlement, 24/7, programmability), they compete head-on for the same balances banks rely on, while pulling the underlying money out of lending.

There is a regulatory wrinkle that *amplifies* the bank threat. In the US, the 2025 GENIUS Act framework for payment stablecoins (and similar rules elsewhere) prohibits issuers from paying interest to holders. The intent is to keep stablecoins as payment instruments, not savings products, precisely to protect bank deposits. But it has a paradoxical edge: it pushes issuers to attract balances on convenience alone and keeps the entire reserve yield with the issuer — which makes issuing stablecoins enormously profitable and incentivizes them to grow the float as large as possible. The bigger the float, the more deposits are disintermediated.

The no-interest rule also has an enforcement leak worth understanding, because it shapes how the threat actually evolves. While the *issuer* may not pay interest on the token, third parties — exchanges, wallets, DeFi protocols — can and do offer "rewards," lending yields, or staking returns on stablecoin balances held with them. So a holder can often earn a return on a stablecoin even though the issuer pays zero, by parking it somewhere that lends it on. That re-introduces the savings-substitute dynamic through a side door: the dollar leaves a bank deposit, becomes a stablecoin, then gets lent through a non-bank channel that the deposit-protection rules were never designed to police. The money still left the regulated banking system; it just took a slightly longer path out.

### The banks' counter-move: tokenized deposits

Banks are not standing still, and their main defensive idea reveals what they actually fear. Several large banks are building **tokenized deposits** — a deposit that lives on a blockchain and moves like a stablecoin, but is still a *deposit*, still a claim on the bank, and still lent out. The pitch to customers is "you get the speed and programmability of a stablecoin without leaving the bank." The pitch to regulators is "the money stays inside the regulated, insured, lending banking system." If tokenized deposits win, the bank keeps its funding and its lending while matching the technology — disintermediation is neutralized. If stablecoins and CBDCs win the user's preference instead, the funding walks. The contest between a "tokenized deposit" (lends, insured, a bank liability) and a "stablecoin" (does not lend, not insured, an issuer liability) is, underneath the jargon, the same contest the cover figure drew: the question of whether the digital dollar you choose to hold quietly keeps funding loans or quietly stops.

## Run risk: the narrow bank that can still break

Here is the trap people fall into. "A stablecoin is fully reserved, so it can never have a run." That is half true and dangerously misleading. A *narrow* bank cannot have an *insolvency* — its assets always cover its liabilities. But it can absolutely have a **run**, and a stablecoin's run risk is in some ways *worse* than a bank's.

A run is when holders all try to get out at once. For a normal bank, the run is fatal because the bank lent the money long and cannot call the loans back fast enough. For a stablecoin, the reserves are liquid, so in principle everyone can be paid. But three things make stablecoin runs sharp and sudden:

- **The reserves are not perfectly liquid in real time.** Treasury bills must be sold; cash sits in banks that have their own hours, settlement times, and — as 2023 proved — their own failure risk. A redemption wave can outrun the issuer's ability to convert reserves to cash *today*.
- **There is no deposit insurance.** A US bank deposit is insured up to \$250,000 by the FDIC. A stablecoin has *no* government backstop — if the reserve is short or frozen, you are an unsecured creditor of a private company. This is the single sharpest difference: the FDIC turns "is my bank safe?" into a question most depositors never have to ask, while a stablecoin holder has to trust a private issuer's reserve attestations in real time.
- **It moves at the speed of code.** A blockchain token can be dumped on an exchange in seconds, 24/7, with no branch to close and no friction to slow the panic. A whisper becomes a crash before a regulator is even awake.
- **The "first-mover" incentive is brutal.** Because redemptions are processed in order and reserves may not all be instantly available, the rational move when you smell trouble is to redeem *first*. Everyone knows everyone else knows this, so a small doubt can detonate into a full run in minutes — the same self-fulfilling logic that drives a classic bank run, but compressed from days into a single trading session.

The cruel irony is that *full reserves do not save you from a run* — they only save you from insolvency. A run is a liquidity event, not a solvency event: it is about whether you can pay *right now*, not whether your assets ultimately cover your liabilities. A bank with perfectly good loans can die in a run because the loans cannot be sold today; a stablecoin with perfectly good Treasury bills can depeg because the bills cannot all be converted to cash today and one of its reserve banks just failed. This is the deepest connection back to the series spine: liquidity is not solvency, and confidence is the only thing standing between an orderly redemption queue and a stampede.

#### Worked example: the USDC depeg, March 2023

This is not hypothetical. Circle held about \$3.3bn of USDC's roughly \$40bn reserve as cash at Silicon Valley Bank. When SVB failed on Friday March 10, 2023, that \$3.3bn was suddenly of uncertain value — the FDIC insurance limit is \$250,000, so \$3.3bn at one bank was overwhelmingly *uninsured*.

- Reserve exposure at risk: \$3.3bn out of about \$40bn = roughly **8% of the reserve** potentially impaired.
- Market reaction: USDC, which should trade at \$1.00, fell to about **\$0.88** over the weekend — a 12% discount, far more than the 8% reserve hole, because fear overshoots.
- Resolution: on Sunday March 12, US regulators guaranteed all SVB deposits, the \$3.3bn was made whole, and USDC climbed back to \$1.00 within days.

**One sentence of intuition:** a stablecoin is only as safe as the banks holding its reserves, so the supposedly "bank-free" digital dollar turned out to be acutely *bank-dependent* — its worst crisis was a bank run that ran *through* it. (For the SVB story itself, see the [SVB and Credit Suisse bank-runs post](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

There is a second-order irony worth naming. Stablecoins were sold partly as an escape from fragile banks. But by concentrating tens of billions of reserves in a handful of banks, they *create* concentration risk for those banks — a single issuer's redemption wave can yank a huge, lumpy deposit out overnight. The stablecoin's run risk and the host bank's run risk become two ends of the same rope.

## CBDC design choices: the knobs that decide the damage

A CBDC is potentially a bigger threat than any private stablecoin, for one simple reason: it is *the safest possible asset.* A direct claim on the central bank cannot default. In a panic, money does not just trickle toward a CBDC — it floods toward it, because depositors would rationally flee a wobbling commercial bank for the one liability in the system that can never fail. A poorly designed retail CBDC could turn every bank wobble into an instant, system-wide deposit run *into* the central bank. Central banks know this, which is why CBDC design is obsessed with one question: how do we let people use digital central-bank money without hollowing out the banks?

The answer is two design knobs. The graph below shows how they interact to set the threat level.

![Graph of CBDC design choices holding cap and remuneration and their effect on banks](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-6.png)

### Knob 1: the holding cap

A **holding cap** limits how much CBDC any one person may hold. The European Central Bank, in designing the digital euro, has discussed an individual cap in the region of **€3,000** (figures have shifted through consultation, but the order of magnitude is the point). With a cap, a CBDC is fine for everyday payments — you keep a few thousand for spending — but you cannot park your life savings in it. The bulk of your money stays in bank deposits, where it keeps funding lending. The cap is, explicitly, a fence around the banking system's funding base.

### Knob 2: remuneration

**Remuneration** means whether the CBDC pays interest. A **non-remunerated** CBDC (pays 0%) behaves like cash: useful for payments, useless for saving. It does not compete with an interest-bearing savings account, so it does not pull savings out of banks. A **remunerated** CBDC that paid a competitive rate would be devastating — it would be a risk-free, government-issued savings account, and rational savers would drain their bank deposits into it. Almost every serious retail-CBDC design leans non-remunerated or zero/low-rate precisely to avoid this.

#### Worked example: CBDC holding-limit math

Suppose a country adopts a retail CBDC with a €3,000 per-person holding cap, and there are 50 million adults.

- Maximum possible CBDC in circulation: $50{,}000{,}000 \times €3{,}000 = €150bn$.
- Compare to a banking system with, say, €4,000bn of household deposits.
- Maximum deposit drain if every adult maxed out their CBDC: $150 \div 4{,}000 = 3.75\%$ of household deposits.

A 3.75% migration is manageable — uncomfortable, but survivable, and far from the 15% scenario that squeezed our earlier bank. **One sentence of intuition:** the holding cap is a deliberate brake; it converts a potentially unbounded deposit run into the central bank into a bounded, modeled outflow the banking system can plan around. Remove the cap, and the brake is gone — which is why the cap, not the technology, is the load-bearing design decision.

There is a third knob worth a sentence: **intermediation.** Most CBDC designs are *intermediated* — you hold the CBDC through your existing bank's app, the bank does the customer service and compliance, and the central bank just runs the ledger. This keeps banks in the customer relationship even if the money itself is a central-bank liability. A *direct* CBDC, where citizens hold accounts straight at the central bank, would cut banks out entirely and turn the central bank into a retail bank — an idea almost no central bank actually wants, because running 50 million retail accounts is not their job and would politicize credit allocation.

### The "waterfall" trick: keeping a cap usable

A holding cap creates an obvious user problem. If your digital-euro wallet is capped at €3,000 and your salary of €3,500 lands in it, where do the extra €500 go? The elegant answer most designs adopt is a **reverse waterfall** (and its mirror, a *waterfall* on spending). You link your CBDC wallet to your normal bank account. When an incoming payment would push you over the cap, the overflow automatically sweeps into your bank deposit. When you try to spend more than your CBDC balance, the shortfall is pulled *up* from your bank account in real time to complete the payment. The cap stays binding on *stored* value, but the CBDC stays usable for any size of payment.

This is not a technical footnote — it is the whole game. The waterfall lets a central bank offer a frictionless digital payment instrument *and* keep the banking system's funding base mostly intact, because money never piles up in the CBDC beyond the cap. It is the clearest illustration of the central design philosophy: **a retail CBDC should be a great way to pay and a deliberately bad way to save.** Every choice — the cap, the zero interest, the waterfall, the intermediation — bends toward that one goal of protecting deposits.

## Who actually loses, and the second-order effects

So far the threat has read like one undifferentiated squeeze on "the banking system." In reality, disintermediation does not fall evenly, and the second-order effects ripple in directions that are not obvious at first. It is worth being precise about who gets hurt, because the answer tells you which banks to watch and why this is not a problem you can wave away with "banks will adapt."

### Not all deposits are equally at risk

A bank's deposit base is a mix, and the parts behave very differently when a shiny new digital dollar appears:

- **Non-interest-bearing transaction balances** (checking accounts, the money you keep for spending) are the most exposed to *stablecoins and CBDCs*, because those instruments compete precisely on the dimension transaction balances care about: speed, convenience, programmability. They are also the cheapest deposits the bank has — so the bank loses its *best* funding first.
- **Rate-sensitive savings and term deposits** are the most exposed to a *remunerated* CBDC or to interest-bearing tokenized money, because savers chase yield. This is why the no-interest design rule matters so much: it walls the digital dollar off from the savings balances rather than the transaction balances.
- **Deeply sticky relationship deposits** — the payroll account a small business has held for fifteen years, the household that keeps everything at one bank for convenience — are the most insulated. A token cannot easily replicate the inertia of a primary banking relationship.

The upshot: the banks most exposed are those whose funding leans on rate-sensitive, low-relationship, easily-moved deposits — which, not coincidentally, describes the kind of bank that already runs fastest in a crisis. SVB's deposits were 94% uninsured and concentrated in a single, networked industry; that is the profile that runs at the speed of a group chat. Stablecoins and CBDCs do not create a new vulnerability so much as sharpen an existing one.

### The central bank's balance sheet swells

Here is a second-order effect almost no one notices. When money flows out of bank deposits and into a CBDC (or into stablecoin reserves held as central-bank reserves), it does not just leave the banks — it *lands on the central bank's balance sheet* as a new liability. To hold that liability, the central bank holds assets against it. In effect, the central bank is now doing some of the financial intermediation the commercial banks used to do: it is taking in "deposits" (the CBDC) and holding assets (whatever backs it). The bigger the digital dollar grows, the bigger and more entangled in private finance the central bank's balance sheet becomes — which is exactly the role most central banks have spent a century trying *not* to take on, because it puts them in the business of allocating credit and exposes them to political pressure over who gets funded.

### Lending does not vanish — it migrates

The most important second-order effect is the one the narrow-banking debate (next section) turns on. If banks lose deposit funding and pull back on lending, the demand for credit does not evaporate. A business that needs a loan still needs the loan. So the lending migrates — to private-credit funds, fintech lenders, money-market-fund-financed channels, and the broader shadow-banking system. This sounds like a wash ("someone still lends"), but it is not. Those lenders fund themselves with flightier money, sit outside deposit insurance and the discount window, and are less visible to supervisors. You have not removed the maturity-transformation risk; you have pushed it into a corner where the safety nets do not reach. That is the recurring lesson of financial history: risk that is regulated out of one place reappears, usually larger and less watched, somewhere else.

## The narrow-banking debate: is disintermediation a bug or a feature?

We have been treating disintermediation as a threat. But there is a serious, century-old argument that says it is the *cure*, not the disease. This is the **narrow-banking debate**, and stablecoins have dragged it back to center stage.

The case *for* narrow banking goes like this. Fractional-reserve banking is inherently unstable: a bank promises to repay deposits on demand while holding assets it cannot sell on demand. That maturity mismatch is what causes runs, panics, and the need for deposit insurance and bailouts. A narrow-banking system — where "money" is fully reserved and "lending" is done separately by investment funds that take genuine investment risk — would be run-proof. Money would be safe; lending would be honest about its risk; no taxpayer would ever bail out a "bank" again. In this view, stablecoins are an accidental step toward a safer monetary system: they make the safe-money function (full reserves) explicit and separate from the risky-lending function.

The case *against* is the one this whole series has been making. Maturity transformation is not a bug to be engineered away — it is a service the economy needs. Someone has to turn short-term savings into long-term loans. If banks cannot do it because their funding has been narrow-banked away, the lending does not disappear; it migrates to less-regulated, less-transparent corners — the shadow-banking system of funds, fintech lenders, and private credit (see the [shadow banking and repo](/blog/trading/finance/shadow-banking-and-the-repo-market) post). You have not removed the maturity-transformation risk; you have moved it somewhere with no deposit insurance, no discount window, and no regulator watching. The before-and-after below frames the trade exactly.

![Before and after comparison of a fractional-reserve bank versus a full-reserve narrow bank](/imgs/blogs/stablecoins-cbdcs-and-the-threat-to-bank-deposits-8.png)

The honest answer is that the debate is unresolved, and the right framing is a trade-off, not a verdict. A fully narrow-banked world is run-proof and credit-starved. A fully fractional world is credit-rich and run-prone. Every real system, including the one we have, sits somewhere in between, propped up by deposit insurance and a lender of last resort that *make* the fragile maturity-transformation trade survivable. Stablecoins and CBDCs are pushing the dial toward the narrow end — and the question regulators are wrestling with is how far is too far before credit creation visibly shrinks.

It is worth noticing *why* the US has historically blocked even pure narrow banks. Several fintech firms have applied for "narrow bank" charters that would take deposits and park 100% of them at the Federal Reserve, paying depositors the risk-free Fed rate. The Fed has resisted granting them full access, and the stated reason is exactly our subject: a perfectly safe, Fed-rate-paying account would be so attractive that money would flood out of ordinary banks in any moment of stress, destabilizing the funding of the credit system. In other words, regulators have *already* decided that an uncapped, remunerated narrow bank is too dangerous to bank funding to allow — which is precisely the verdict they are now trying to encode into stablecoin rules (no interest) and CBDC design (holding caps). The narrow-banking debate is not academic; it is being adjudicated, one charter and one rulebook at a time.

## Common misconceptions

**"A fully-reserved stablecoin is risk-free, so it cannot hurt the banking system."** Two errors in one sentence. It is not risk-free — it carries issuer risk and reserve-bank risk, as the USDC depeg to \$0.88 proved. And even if it *were* perfectly safe, that safety is exactly the problem for banks: a perfectly safe, fully-reserved dollar is a dollar that creates no credit. The damage to the banking system comes precisely *because* the stablecoin doesn't lend, not despite it.

**"The money is still there, so nothing is lost."** The money is indeed still there — but the *lending* is not. When \$1,000 of deposits becomes \$1,000 of stablecoin reserves in Treasury bills, the \$9,000 of credit that \$1,000 would have multiplied into the banking system simply never gets created. Disintermediation destroys credit capacity, not money.

**"Stablecoins compete with crypto, not with banks."** Stablecoins compete with *deposits*. They hold the same thing a checking account holds — dollars you can spend — and they pull those dollars into reserves. The fact that they live on a blockchain is incidental to the funding threat. A stablecoin is a deposit substitute that doesn't lend; that is the whole story for a bank.

**"A CBDC is just a digital version of cash, so it changes nothing."** Physical cash is clumsy, capped by what you can carry, and earns nothing — so it never threatened deposits at scale. A digital central-bank claim is frictionless, potentially uncapped, and is the single safest asset in the currency. *That* is why CBDC design fights so hard over holding caps and remuneration: an uncapped, interest-paying CBDC is nothing like cash — it is a risk-free government savings account that could empty the banks.

**"If banks lose deposits, they will just borrow the money back, no big deal."** They can borrow it back — at a price. Deposits are the cheapest money a bank has (often near 0%); wholesale funding costs the market rate (5% in a high-rate world) and runs at the first sign of stress. Swapping deposits for wholesale funding does not restore the bank; it shrinks the spread that is the bank's entire reason to exist and makes the bank more fragile, as our \$6.75bn-a-year funding-cost example showed.

**"Disintermediation is a fast, dramatic event you would see coming."** Usually the opposite. A panic run, like USDC's weekend or SVB's 36 hours, is the rare, visible version. The more likely shape is a slow structural drift: a few percent of transaction balances migrating to digital wallets each year because they are simply more convenient, compounding quietly. Banks adapt to a slow drift by paying up for deposits and tightening lending standards — which is exactly why you would *not* see a headline. The damage is diffuse: a slightly higher cost of funds across the system, a slightly tighter credit channel, loans that quietly do not get made. The dangerous case for credit creation is not the spectacular run; it is the unspectacular, persistent leak.

## How it shows up in real banks

**USDC and USDT reserves: the disintermediation, made visible.** The two largest stablecoins together hold well over \$150bn in reserves as of 2024–2025, the overwhelming majority in short-dated US Treasury bills and repo, with a slice as cash at banks. Tether alone reported holding more than \$100bn in US Treasuries, which would rank it among the larger foreign holders of US government debt. Every one of those dollars is a dollar that left the fractional-reserve banking system and is now financing the US government instead of a mortgage or a small-business loan. That is deposit disintermediation you can see on a balance sheet: the credit that money used to seed is simply not being created.

**The deposit-flight scenario regulators war-game.** Supervisors run exactly the scenario in our funding-mix chart. The fear is not a steady 1% drift; it is a *stress* event in which a chunk of deposits flees to digital safe assets all at once — say 10–15% in a panic — forcing banks to fire-sale assets or scramble for wholesale funding at the worst possible moment. This is why the Basel liquidity rules (the LCR, covered in the [liquidity-management post](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer)) and the US stablecoin framework both treat large, runnable deposit substitutes as a systemic concern. The 36-hour digital run that killed SVB — \$42bn out in one day, another \$100bn queued — was a preview of how fast money moves when it can move with a tap. The arithmetic is sobering when you scale it: applied to a large bank with \$1,000bn of deposits, the 15% flight in our funding example is \$150bn walking out, against a Basel liquidity buffer that is sized for a 30-day stress, not a same-day digital stampede. The combination of frictionless digital safe assets and a 30-second exit is exactly the cocktail that turns a survivable liquidity dip into a fatal one — and it is why "where could the money instantly go instead?" has become a question on every supervisor's checklist. Before stablecoins and CBDCs, the honest answer was "nowhere very fast"; now it is "to a wallet, in seconds."

**CBDC pilots: design as defense.** China's e-CNY, the largest live retail CBDC, deliberately limits balances and pays no interest, and is *intermediated* through commercial banks — every design choice aimed at keeping it a payment tool, not a savings drain. The ECB's digital-euro work has revolved around the holding cap precisely because internal modeling showed that an uncapped digital euro could pull a destabilizing share of deposits out of euro-area banks. The Federal Reserve has been the most cautious of all, publishing research but no commitment, explicitly citing the risk to bank funding and credit. The common thread: every serious CBDC project treats *not hurting the banks* as a primary constraint, which tells you how real the funding threat is taken to be.

**The 2023 USDC–SVB scare, in full.** We used it as a worked example; here is the systemic lesson. A "decentralized" digital dollar nearly broke because \$3.3bn of its reserves sat at one failing bank — and it was saved only by the same government backstop (a deposit guarantee) that exists to save *banks*. The episode demolished the idea that stablecoins are a clean alternative to the banking system. They are deeply entangled with it: they hold reserves at banks, they buy the government debt banks compete with, and when a bank fails, the stablecoin shakes. The future of digital dollars and the stability of banks are not separate questions. They are the same question.

**The GENIUS Act and the regulatory bargain (2025).** The US payment-stablecoin law that took shape in 2025 made the bank-protection logic explicit: issuers must be fully reserved in safe assets, must *not* pay interest to holders, and face bank-like supervision. The deal on offer is "you may issue a digital dollar, but only as a narrow bank, and you may not turn it into a savings product that competes with deposits for yield." Whether that fence holds — or whether convenience alone pulls enough balances out of deposits to matter — is the open question the next decade will answer.

## The takeaway / How to use this

If you remember one thing, make it the asymmetry at the center of this post: **a deposit dollar funds about \$0.90 of lending; a stablecoin or CBDC dollar funds \$0.** Everything else is a consequence of that single line. Digital dollars are not dangerous to banks because they are risky — the fully-reserved ones are, in narrow terms, *safer* than a bank. They are dangerous because they are *inert*: they take the raw material of credit creation and lock it in Treasury bills where it makes no loans.

So when you read the next headline about a stablecoin crossing some new size milestone, or a central bank launching a CBDC pilot, ask the three questions that actually matter for the banking system:

- **Does it lend?** A fully-reserved digital dollar does not. Every dollar in it is a dollar pulled out of the credit machine.
- **Where do its reserves sit?** In bank deposits, the disintermediation is partial (the money is still in *a* bank). In Treasury bills or central-bank reserves, it is total (the money has left commercial banking entirely).
- **What are the brakes?** For a stablecoin, the brake is regulation (full reserves, no interest, supervision). For a CBDC, the brake is design (a holding cap and no remuneration). Weak brakes mean fast disintermediation; strong brakes mean a slow, manageable drift.

For anyone trying to read a bank's future, this connects straight back to the spine of the series. A bank lives or dies on cheap, sticky, abundant deposits — that is the short borrowing that funds the long lending, the maturity-transformation trade that is the bank's whole reason to exist. Stablecoins and CBDCs attack that funding base directly, not by being better banks, but by being *non-banks* that hold money without lending it. The threat is real but it is not unstoppable: it is exactly as large as the share of deposits that migrates, and that share is governable by the regulatory and design brakes we just listed. The banks most exposed are the ones with the thinnest, most rate-sensitive deposit bases; the ones most insulated are those with deeply sticky retail relationships that a token cannot easily replicate. Watch the deposit base, and you are watching the one number that decides whether digital dollars are a curiosity or a slow-motion squeeze on credit itself.

## Further reading & cross-links

- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — why the deposit base is the thing stablecoins and CBDCs threaten.
- [Deposit insurance, the lender of last resort, and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — the backstops that make fractional-reserve banking survivable, and which stablecoins lack.
- [Central-bank digital currencies (CBDC)](/blog/trading/finance/central-bank-digital-currencies-cbdc) — the systemic and monetary-policy view of CBDCs, beyond the bank-funding angle.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — the full mechanics of how stablecoins are minted, reserved, and redeemed.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — where maturity transformation migrates if narrow banking pushes it out of regulated banks.

*This is educational material about how banks fund themselves and how digital money interacts with that, not investment advice.*

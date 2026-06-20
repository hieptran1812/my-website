---
title: "The Repo Market and How Banks Fund Overnight: Pawning a Bond Until Tomorrow"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A plain-English deep dive into the repo market: how a repurchase agreement works as a collateralized overnight loan, what haircuts and the repo rate really mean, why rehypothecation multiplies one bond into many, and how repo froze in 2008 and spiked in September 2019."
tags: ["banking", "repo", "repurchase-agreement", "collateral", "haircut", "rehypothecation", "tri-party-repo", "wholesale-funding", "liquidity", "money-markets", "2008-crisis", "secured-funding"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A repo is a collateralized overnight loan dressed up as a sale-and-buyback: a bank sells a bond today for cash and promises to buy it back tomorrow at a slightly higher price, and that tiny price gap is the interest. It is the cheapest, biggest, and most fragile way a bank funds itself.
>
> - A repo turns a safe bond sitting on the balance sheet into cash overnight, secured by the bond itself, so it is far cheaper than borrowing on trust alone.
> - The lender protects itself with a *haircut* — lending less than the bond is worth — and the gap is the repo borrower's first loss. When haircuts jump, the same bond raises less cash, and that is a funding squeeze.
> - Rehypothecation lets one bond be re-pledged down a chain, so a single \$100 bond can quietly support \$400 or \$500 of financing — efficient on the way up, terrifying on the way down.
> - Repo is where 2008 actually broke (a silent run on repo as haircuts spiked) and where funding seized again in September 2019, when the overnight rate briefly leapt from about 2.25% toward 10%.

In the early hours of September 17, 2019, something happened in a corner of the financial system that almost nobody outside it watches, and it scared the people who do. The rate at which big banks and dealers borrow cash overnight against the safest collateral on earth — US Treasury bonds — did not drift up a tenth of a percent the way money-market rates normally move. It *exploded*. The overnight repo rate, which had been sitting calmly near the Federal Reserve's target of about 2.25%, spiked intraday toward 10%. The secured overnight financing rate, the official benchmark, printed 5.25% that day — more than double its level 48 hours earlier. For a few hours, the most liquid market in the world, the market that is supposed to make cash and Treasuries practically interchangeable, simply could not clear at a sane price.

This was not a crisis of solvency. No bank was bust. There were plenty of Treasuries, and plenty of cash in the system overall. It was a crisis of *plumbing*: too much cash had to flow to too few places at the same moment, the usual lenders did not show up, and the price of overnight money went vertical. The New York Fed had to step in within hours, injecting tens of billions of dollars to drag the rate back down, and it kept those operations running for months.

That episode is the perfect doorway into one of the least understood and most important mechanisms in all of finance: the repurchase agreement, or *repo*. Most people have never heard of it, yet repo is how a huge share of the world's banks and securities dealers fund themselves every single night — trillions of dollars rolled over, every day, in loans that mature tomorrow morning. It is the beating heart of short-term funding. And because it beats so fast and so quietly, when it skips, the whole body convulses. The diagram above is the mental model we will build from: cash moves one way today against a bond, and the trade reverses tomorrow at a slightly higher price.

![A repo as cash today against a bond reversed tomorrow at a higher price](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-1.png)

This post goes deep on the mechanics — what a repo actually is, how the haircut works, what the repo rate measures, why rehypothecation is both clever and dangerous, how tri-party repo bolts a clearing bank into the middle, and how this market froze in 2008 and seized in 2019. For the wider *system* view — how repo connects to the shadow-banking universe of money funds, hedge funds, and securities lenders — we link out to [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) rather than re-deriving it here. Our job is the machinery itself.

## Foundations: a repo, collateral, the haircut, the repo rate, and the rest of the vocabulary

Before any of the drama makes sense, we need the words. Repo has a reputation for being impenetrable, but it is impenetrable only because of jargon. Strip the jargon and it is one of the most intuitive deals in finance. So let us build it from a thing everyone understands.

### Start with a pawn shop

You need \$98 in cash for one night. You own a watch worth \$100. You walk into a pawn shop and you do *not* sell the watch — you pledge it. The pawnbroker hands you \$98 and keeps the watch. Tomorrow you come back, hand over \$98 plus a small fee, and walk out with your watch. You got your cash; the pawnbroker earned the fee; and the whole time the broker was holding something worth more than the cash they lent, so if you had vanished, they could have sold the watch and been made whole.

That is a repo. Almost exactly. A *repurchase agreement* is a deal where one party sells a security — usually a high-quality bond — to another party for cash today, with a binding promise to buy it back at a set price on a set future date, usually the very next morning. Legally it is dressed up as a sale and a repurchase, two trades. Economically it is a *secured loan*: you are borrowing cash, and the bond is the collateral. The pawnbroker is the lender; you, pledging the bond, are the borrower.

Why bother with the sale-and-buyback fiction at all, rather than just writing "secured loan" on the contract? Because the legal structure of a true sale gives the lender a powerful protection: if the borrower goes bankrupt, the lender already *owns* the collateral outright and can sell it immediately, without getting tangled in the slow, contested bankruptcy process that an ordinary secured creditor would face. Repo is carved out of normal bankruptcy rules in most major jurisdictions precisely so the lender can grab the collateral and walk. That legal certainty is a big part of why lenders are willing to accept such low rates — and, less happily, it is part of why a failing firm's collateral can be seized and dumped so fast that the failure becomes a fire sale. The same feature that makes repo safe for the individual lender makes it dangerous for the system, because everyone gets to grab the collateral at once.

A handful of terms now fall out naturally:

- **Collateral** is the asset you pledge to secure the loan — the watch, or in repo the bond. If you fail to pay back, the lender keeps and can sell the collateral. Good collateral is something safe and easy to sell: US Treasuries are the gold standard, agency mortgage bonds are common, and lower-quality bonds (corporate debt, structured products) are accepted at worse terms.
- **The haircut** is the difference between what the collateral is worth and the cash the lender hands over. The watch is worth \$100; the broker lent only \$98. That \$2 gap — 2% — is the haircut. It is the lender's cushion against the collateral falling in price before they could sell it. We will spend a whole section on this, because the haircut is where the action is.
- **The repo rate** is the interest on the loan, expressed as an annual percentage. It is *not* quoted as a fee; it is baked into the buyback price. You sell the bond for \$98 today and agree to buy it back for, say, \$98.0134 tomorrow. That \$0.0134 over one night, annualized, *is* the repo rate. We will compute one exactly below.
- **The tenor** is how long the loan lasts. The overwhelming majority of repo is *overnight* — borrow tonight, repay tomorrow morning, then often roll it over into a fresh overnight repo and do it all again. There is also *term repo* (a week, a month, three months) where the cash and collateral stay put for longer.
- **A reverse repo** is the same trade seen from the *lender's* side. If I am putting up cash and taking your bond, I have done a reverse repo — I am the cash provider, the secured lender. One party's repo is always the other party's reverse repo. The word just tells you which seat you are in: repo = "I need cash, here's my bond"; reverse repo = "I have cash, give me your bond."

### Two more terms you must meet now: rehypothecation and tri-party

These two come up so much that we define them up front and then unpack each in its own section.

- **Rehypothecation** is when the party *holding* your pledged collateral re-pledges it to someone else. The pawnbroker, holding your watch, walks next door and uses your watch to borrow cash for *their* own needs. In repo this is routine and legal: the lender who received your Treasury bond can turn around and repo it out to a third party. One bond ends up backing a chain of loans. This is why repo is so capital-efficient — and why one default can ripple down a chain.
- **Tri-party repo** is a plumbing arrangement where a third party — a *clearing bank* (in the US, essentially Bank of New York Mellon) — sits between borrower and lender, holds the collateral, prices it every day, and applies the haircut. The two trading parties never touch each other's assets directly; the custodian does the housekeeping. The alternative, *bilateral repo*, is a direct deal between two parties who handle delivery themselves.

That is the whole vocabulary. Repo, collateral, haircut, repo rate, tenor, reverse repo, rehypothecation, tri-party. Everything that follows is just these eight ideas interacting under stress. Now we make each one precise — and we start with the deceptively simple question of how the interest actually works.

## The repo rate: how a sale-and-buyback hides an interest rate

The thing that confuses people first about repo is that nobody writes "interest" anywhere. There are two prices: the price you sell at today, and the price you buy back at tomorrow. The interest is the *difference between those two prices*, and you have to back it out.

Let us do it slowly with friendly numbers.

#### Worked example: a repo with a haircut, and the implied repo rate

You are a securities dealer. You own a US Treasury bond with a market value of exactly \$100. You need cash overnight, so you repo it out.

Step 1 — the haircut sets the cash. Suppose the haircut is 2%. The lender will hand you the bond's value *minus* the haircut: \$100 × (1 − 0.02) = **\$98**. You get \$98 in cash; the lender holds a \$100 bond. The lender is over-collateralized by \$2 — that is their protection.

Step 2 — the repo rate sets the buyback price. Suppose the overnight repo rate is quoted at 5.00% per year. Interest is charged on the *cash* you borrowed (\$98), for *one night*. By the money-market convention, one day of a 5% annual rate is:

$$
\text{interest} = \$98 \times 0.05 \times \frac{1}{360} = \$0.01361
$$

We divide by 360 because the money market quotes rates on a 360-day year (an old convention that makes the arithmetic round). So the interest for one night on \$98 at 5% is about **1.36 cents**.

Step 3 — write the buyback price. Tomorrow you must hand back the \$98 you borrowed plus the 1.36 cents of interest:

$$
\text{repurchase price} = \$98 + \$0.01361 = \$98.01361
$$

So the deal, written out, is: "I sell you this bond for \$98.00 today, and I will buy it back for \$98.01361 tomorrow." No line on the contract says "interest 1.36 cents." It is hidden in the gap between the two prices. If someone hands you the two prices and asks for the rate, you reverse the arithmetic:

$$
\text{repo rate} = \frac{\$98.01361 - \$98.00}{\$98.00} \times \frac{360}{1} = 0.0500 = 5.00\%
$$

The one-sentence intuition: a repo rate is just the annualized version of the tiny price increase between the sale and the buyback, charged on the cash you borrowed, not on the bond you pledged.

Two things worth pinning down here. First, the interest is on the cash (\$98), not the collateral (\$100) — the haircut shrinks how much you can actually borrow, but the rate clock runs on the borrowed cash. Second, because the loan is overnight and secured by safe collateral, the repo rate is one of the *lowest* borrowing rates in the entire economy. It sits a sliver below or around the unsecured overnight rate (the rate banks charge each other on trust), precisely because the lender has the bond in hand. Cheapness is the whole appeal: a bank that funds with repo is funding at almost the risk-free rate.

### Where the repo rate sits in the rate stack

It helps to anchor the repo rate against its neighbors. On a calm day in 2019, the rate stack looked roughly like this:

| Overnight rate | What it is | Roughly where it sat |
|---|---|---|
| Interest on reserves | What the Fed pays banks on cash parked at the Fed | ~2.10% |
| Repo (Treasury collateral) | Secured overnight loan against Treasuries | ~2.20% |
| Fed funds | Unsecured overnight loan between banks | ~2.13–2.25% |
| Unsecured commercial paper | Short-term corporate IOU, no collateral | higher, varies by issuer |

The repo rate normally hugs the unsecured rates from just below, because collateral makes the loan safer. When repo trades *above* unsecured rates — as it did spectacularly in September 2019 — that is a screaming signal that something has gone wrong with the *supply of cash*, not with credit. We will return to exactly that.

### General collateral versus a special

One more distinction makes the rate readable, and it is the one most outsiders miss. Not all collateral is equal in the eyes of the cash lender, and the difference shows up in the rate.

Most repo is *general collateral*, usually shortened to **GC**. Here the lender does not care which specific bond it gets, as long as it falls into an accepted basket — "any on-the-run Treasury," say. The lender just wants safe collateral and a return on cash; the borrower just wants cheap cash and is happy to deliver whatever qualifying bond is convenient. The GC repo rate is therefore a pure *money* rate — it tells you the price of overnight cash secured by safe collateral, full stop. This is the rate that spiked in 2019.

Sometimes, though, a particular bond is in hot demand — perhaps everyone needs that exact security to settle short sales or to deliver into a futures contract. That bond goes *on special*: lenders will accept a *lower* repo rate to get their hands on it, because owning the right to that specific bond temporarily is worth something to them. In the extreme, a bond can go "special" all the way down to a near-zero or even negative repo rate — the cash lender is effectively paying for the privilege of borrowing that exact bond. The spread between the GC rate and a bond's special rate measures how scarce and sought-after that one security has become. For our purposes, the headline "repo rate" almost always means the GC rate — the price of cash — and that is the one whose spikes signal funding stress.

### Term repo and why overnight is the default

We have computed an overnight repo, but repo can run for any agreed tenor. The arithmetic generalizes cleanly: interest scales with the number of days.

#### Worked example: a 30-day term repo

Take the same \$98 of cash borrowed against a \$100 bond, but now the deal runs for 30 days at a 5% annual rate instead of one night. The interest is:

$$
\text{interest} = \$98 \times 0.05 \times \frac{30}{360} = \$0.408
$$

So you sell the bond for \$98.00 today and agree to buy it back for \$98.408 in 30 days. Thirty nights of interest is roughly thirty times one night's interest — \$0.408 versus \$0.0136 — exactly as you would expect.

Term repo locks in funding for longer, which is *safer* for the borrower (the lender cannot pull the loan or raise the haircut until it matures) but usually a touch more expensive, because the lender gives up flexibility and takes on more risk over a longer window. The one-sentence intuition: term repo is the borrower paying a small premium for the comfort of not having to refinance tomorrow morning.

Here lies a quiet but crucial choice every repo borrower makes. The cheapest funding is overnight, rolled fresh each day — but it is also the most fragile, because the borrower is at the mercy of the market every single morning. Term repo costs slightly more but buys insurance against a sudden funding cliff. Firms that funded long-dated, risky assets with the very cheapest overnight repo, rather than paying up for term, are exactly the firms that died in 2008 when the overnight roll stopped. The decision between overnight and term repo is, in miniature, the decision every bank makes about how much funding fragility to accept in exchange for a slightly fatter margin.

## The haircut: the lender's cushion and the borrower's leash

If the repo rate is the price of the loan, the haircut is the *size* of the loan. And the haircut, not the rate, is where repo gets dangerous. A small move in the rate is an annoyance; a move in the haircut can be a death sentence.

Recall the definition: the haircut is the percentage by which the lender discounts the collateral when deciding how much cash to lend. Pledge a \$100 bond at a 2% haircut and you get \$98. The haircut exists because the lender is not actually sure the bond is worth \$100 if they ever have to sell it in a hurry. Between the moment a borrower defaults and the moment the lender liquidates the collateral, the price could fall. The haircut is the lender's buffer against that fall. It is also, viewed from the other side, the borrower's *first loss* — the borrower's own money standing in front of the lender's.

So the haircut answers a precise question: how much can a given pile of collateral fund?

#### Worked example: how a \$X collateral pool funds \$Y of assets

Imagine a hedge fund or dealer that wants to own \$1,000,000,000 — one billion dollars — of US Treasury bonds, but it only has a little of its own cash. It funds the position in repo.

At a 2% haircut, every \$100 of bonds raises \$98 of cash. So \$1 billion of bonds raises:

$$
\$1{,}000{,}000{,}000 \times (1 - 0.02) = \$980{,}000{,}000
$$

The fund borrows \$980 million in repo against the bonds and only has to put up the other \$20 million of its own money. With \$20 million of equity it controls \$1 billion of bonds — that is **50× leverage** (\$1,000m ÷ \$20m). The haircut *is* the leverage limit: leverage equals 1 divided by the haircut. A 2% haircut permits up to 50×; a 4% haircut permits 25×; a 10% haircut permits only 10×.

Flip it around to size the funding from a fixed pool. If a dealer has \$50 billion of eligible collateral and the haircut is 2%, that pool funds \$50bn × 0.98 = **\$49 billion** of cash borrowing. The dealer uses that \$49bn to hold assets, earn the spread between what the assets yield and the repo rate they pay, and pocket the difference. The whole business model — borrow cheap and short in repo, hold higher-yielding assets — is maturity transformation, the same fragile trade that defines banking itself. (For the full statement of that spine, see [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).)

The one-sentence intuition: the haircut is the dial that converts a pile of bonds into a quantity of borrowed cash, and one over the haircut is the maximum leverage the collateral allows.

The chart below makes the relationship concrete: hold the bond at \$100 and crank the haircut up, and the cash it raises falls in lockstep.

![Bar chart of cash raised against a 100 dollar bond as the haircut rises](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-3.png)

Now sit with the dangerous implication. The borrower's leverage is set by someone else — the lender — and the lender can change the haircut whenever they get nervous. If you are running 50× leverage on a 2% haircut and your lender wakes up and decides the haircut should be 4%, your funding for the same bonds just dropped from \$98 to \$96 per \$100. You must find the missing \$2 per \$100 *today* — or sell bonds. That is a margin call by another name, and it is the mechanism by which a calm repo market becomes a stampede.

#### Worked example: a rising-haircut funding squeeze

Let us walk a squeeze step by step, because this is the single most important dynamic in the whole post.

You hold \$1 billion of bonds, funded at a 2% haircut, so you borrowed \$980 million and put up \$20 million of your own equity. Comfortable.

Now stress hits. The market starts to doubt your collateral — maybe it is not pristine Treasuries but mortgage-backed bonds, and house prices are wobbling. Your repo lenders raise the haircut from 2% to **25%**. Look at what that does to your funding capacity for the *same* bonds:

$$
\$1{,}000{,}000{,}000 \times (1 - 0.25) = \$750{,}000{,}000
$$

Your bonds now support only \$750 million of borrowing. But you owe \$980 million. The gap is:

$$
\$980{,}000{,}000 - \$750{,}000{,}000 = \$230{,}000{,}000
$$

You must come up with \$230 million in cash — overnight — or sell roughly \$230 million-plus of bonds into a falling market to repay the loan you can no longer roll. Your \$20 million equity cushion is annihilated many times over. And the cruel twist: when you and everyone else in your position sell bonds at once to plug the gap, the bonds' price falls, which makes the collateral look worse, which makes lenders raise the haircut *further*. The squeeze feeds itself.

The one-sentence intuition: in repo, the lender controls your leverage through the haircut, so a haircut spike forces you to find cash or dump assets *exactly when* doing either is hardest.

The before-and-after below is the squeeze in one picture: the same \$100 bond, but the cushion the lender demands swings from trivial to crushing.

![Before and after of a normal two percent haircut versus a twenty five percent haircut funding squeeze](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-2.png)

## Why repo is the engine of bank and dealer funding

Step back and ask: why do banks and securities dealers love this thing so much? Why fund overnight, rolling the loan every single morning, instead of borrowing for a year and sleeping soundly? Three reasons, and together they explain why repo grew into a multi-trillion-dollar market.

First, **it is cheap.** Because the loan is secured by safe collateral, the lender accepts a low rate. A dealer holding a big inventory of Treasuries can fund that inventory at a hair above the risk-free rate. Borrowing the same money unsecured — on the strength of the dealer's name alone — would cost meaningfully more. For a business that lives on thin spreads, shaving the cost of funding is the difference between profit and loss.

Second, **it is flexible.** Overnight funding means the dealer can resize its borrowing every day to match exactly the inventory it holds that day. Buy more bonds, repo more; sell bonds, repo less. There is no awkward year-long loan sitting on the books when the position has changed. Repo is funding that breathes with the balance sheet.

Third, **it puts idle collateral to work.** A bank or fund holding a mountain of Treasuries for regulatory or trading reasons is otherwise sitting on a dead asset. Repo turns that mountain into a cash machine: pledge the bonds overnight, raise cash, do something with the cash, get the bonds back tomorrow. The collateral does double duty.

And on the *other* side of every repo is a lender who needs this market just as badly. Money-market funds, corporate treasurers, securities lenders, foreign central banks — all of them have piles of cash they need to park *safely* and *briefly*, earning a little interest, with no credit risk. An overnight repo against Treasuries is close to the perfect parking spot: you earn a return, you hold the collateral, and you get your cash back tomorrow. So repo is a two-sided love affair: borrowers want cheap secured cash, lenders want a safe short-term home for cash. The market clears trillions a night because both sides genuinely need it.

This is also where repo connects to the bank's broader liability structure. Repo is one rung on the *funding stack* — below cheap, sticky retail deposits but above long-term bonds — and a bank that leans too hard on it is leaning on funding that can vanish overnight. We go through the whole ladder in [the funding stack](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds); the key point for now is that repo is *wholesale* funding, which means it is professional money that flees at the first sign of trouble, unlike a granny's checking account that mostly stays put.

### Who is actually in this market

It helps to put faces to the two sides, because the cast tells you why the market behaves the way it does. The repo market is not a vague machine; it is a specific set of institutions, each with a specific need.

On the **cash-borrowing side** (the firms saying "here is my bond, give me cash") you find: securities dealers and the trading arms of big banks, funding their bond inventories; hedge funds, levering up positions in Treasuries, corporate bonds, and mortgage securities; and other leveraged investors who want to own more bonds than their own capital would allow. These are sophisticated, fee-sensitive, professional borrowers who will move their business for a basis point — and who, crucially, *run* the moment they smell trouble in their counterparty.

On the **cash-lending side** (the firms saying "here is cash, give me a bond to hold") the biggest players are money-market funds — pools of investors' cash that need a safe, liquid, interest-earning home for hundreds of billions of dollars and have absolutely no appetite for credit risk. Alongside them sit corporate treasurers parking operating cash, securities lenders reinvesting the cash collateral from stock loans, and foreign central banks managing their dollar reserves. What unites all of them is that they are *risk-averse cash holders*, not yield-chasers. They are in repo precisely because it looks safe — and that is exactly why they flee at the first hint that it is not. A money fund will never knowingly risk its principal to earn an extra few basis points; the instant a borrower looks shaky or the collateral looks doubtful, it simply does not roll the loan and parks its cash somewhere safer, even at zero.

This is the deep reason repo runs are so violent. The lenders are the most skittish money in finance — short-term, professional, and allergic to loss — and they are funding some of the most leveraged borrowers in finance. When the skittish money meets the leveraged borrower and something goes wrong, the lender's instinct to pull and the borrower's inability to survive the pull combine into a run that can play out in *hours*.

#### Worked example: a dealer's repo-funded carry trade and its P&L

Make the economics concrete. A government-bond dealer holds \$10 billion of 2-year Treasury notes yielding 4.30%. It funds them overnight in repo at 4.00%, with a 2% haircut.

The haircut means the dealer borrows \$10bn × 0.98 = \$9.8 billion in repo and funds the remaining \$200 million from its own capital.

The carry — the spread it earns — is the difference between what the bonds yield and what the repo costs, on the borrowed amount, plus the full yield on its own \$200 million:

$$
\text{annual interest earned} = \$10{,}000\text{m} \times 4.30\% = \$430\text{ million}
$$
$$
\text{annual repo cost} = \$9{,}800\text{m} \times 4.00\% = \$392\text{ million}
$$
$$
\text{net carry} = \$430\text{m} - \$392\text{m} = \$38\text{ million per year}
$$

On \$200 million of its own capital, \$38 million of net carry is a 19% return on equity — fat, for holding government bonds. *That* is the magic of repo leverage: a 30-basis-point spread becomes a 19% return when you borrow 49 dollars for every dollar of your own.

The one-sentence intuition: repo lets a dealer turn a tiny yield spread into a large return on its own capital — which is wonderful until the repo rate rises, the haircut jumps, or the bonds fall, any of which can flip that leverage from amplifier of gains to amplifier of losses.

## Tri-party repo: the clearing bank in the middle

So far we have imagined two parties dealing directly — borrower hands over a bond, lender hands over cash, and they sort out the housekeeping themselves. That is *bilateral repo*, and it works fine when the two parties trust each other and can manage the daily mechanics. But the daily mechanics are surprisingly heavy. Every morning, someone has to: value the collateral at current market prices, check the haircut still holds, swap the right bonds for the right cash, handle the case where the borrower wants to substitute one bond for another, and unwind and re-do the whole thing the next day. Multiply that by thousands of trades and you have an operational nightmare.

Enter **tri-party repo.** A third party — a specialized *clearing bank* — sits in the middle and does all the housekeeping for both sides. In the US, this role is concentrated almost entirely in one institution, Bank of New York Mellon. The borrower and lender agree the terms (amount, rate, eligible collateral, haircut), and the clearing bank does the rest: it holds the collateral in custody, prices it every day, applies the haircut, moves cash and securities between the parties' accounts, and manages substitutions. The two trading parties never have to touch each other's assets; they each just have an account at the clearing bank.

![Graph of tri-party repo showing the clearing bank between borrower and lender](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-4.png)

The benefits are real. Operationally it is far cheaper and less error-prone to let a specialist custodian manage collateral than to do it bilaterally. The lender — often a money-market fund with no desire to become a bond-trading operation — gets to lend cash against collateral without ever managing the collateral itself. And the standardization makes the market deeper and more liquid.

But tri-party has a structural quirk that mattered enormously in the crisis: the **unwind.** For years, the clearing bank would unwind every tri-party repo each morning — temporarily returning the cash to the lenders and the collateral to the borrowers — and then re-establish the repos in the afternoon once the day's trades settled. During those daytime hours, the clearing bank itself was effectively financing the dealers' entire collateral books with its *own* intraday credit, often hundreds of billions of dollars. That made the clearing bank a giant, concentrated point of fragility, and it gave money-fund lenders a daily off-ramp: every morning their cash was handed back, and on any morning they could simply decline to re-lend. Reforms after 2008 largely eliminated the daily full unwind, precisely because regulators realized this daily reset was an accelerant for a run. The clearing bank had become a single chokepoint through which the entire dealer-funding market had to pass each day.

The structural lesson: tri-party makes repo efficient and scalable in normal times, but it also concentrates the entire market's plumbing in one institution and one daily ritual — and concentration is exactly what you do not want when the thing you are concentrating is the funding that thousands of firms depend on to open for business each morning.

## Rehypothecation: how one bond backs many loans

Now we reach the most mind-bending and most consequential feature of repo: the same bond can secure more than one loan at a time. This is *rehypothecation*, and once you see it, you understand why repo is both extraordinarily efficient and prone to chain-reaction collapse.

Go back to the pawn shop. You pledge your watch for \$98. The pawnbroker is now holding a \$100 watch. If the broker is allowed to *re-pledge* it — to walk next door and use your watch as collateral to borrow cash for the broker's own purposes — then your one watch is now securing two loans: yours, and the broker's. The watch next door's broker can re-pledge it again. One physical watch, a chain of loans.

In repo this is routine. When you repo a bond out, the lender typically receives full title to the bond and is free to re-use it — to repo it out again to a third party, to deliver it to settle a short sale, to post it as margin somewhere else. The collateral does not sit in a vault; it circulates. The technical name for how far a single piece of collateral travels through the system is *collateral velocity*, and in the years before 2008 a single high-quality bond might be re-used three or four times over.

#### Worked example: one collateral pool, multiplied down a chain

Take one \$100 Treasury bond and a 2% haircut at every link. Watch the financing pile up.

- **Link 1.** The original owner repos the \$100 bond and raises \$100 × 0.98 = **\$98**.
- **Link 2.** The lender who now holds the bond re-pledges it. The bond is worth \$100, but conventionally each re-pledge applies the haircut again, so this link raises about \$98 × 0.98 ≈ \$96. Running total of cash created off the one bond: \$98 + \$96 = **\$194**.
- **Link 3.** Re-pledged again, raising about \$94 more. Running total: **\$288**.
- **Link 4.** About \$92 more. Running total: **\$380**.
- **Link 5.** About \$91 more. Running total: **\$471**.

So one \$100 bond, re-pledged five times, has quietly supported roughly **\$471 of financing** across the chain. The collateral has not multiplied — there is still only one bond — but the *credit* built on it has nearly quintupled.

The one-sentence intuition: rehypothecation means a single safe bond can underpin a tall stack of loans, which makes the system marvelously efficient at turning scarce collateral into abundant funding — and means a single failure anywhere in the chain can leave several lenders all reaching for the *same* one bond.

![Line and bar chart of rehypothecation multiplying one collateral pool down a chain](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-8.png)

That last clause is the whole danger. As long as everyone keeps rolling their loans and nobody defaults, the chain holds and the system enjoys the efficiency. But if one link breaks — a dealer fails, or simply *might* fail — every lender up and down the chain suddenly wants the actual bond back, and there is only one bond. The chain that created \$471 of funding on the way up has to be *unwound*, and the unwinding is a scramble. The collateral that flowed so freely on the way up freezes solid on the way down. This is precisely what happened to firms that had re-pledged client collateral when they failed: clients discovered their "safe" collateral was tangled three loans deep in a chain and could not simply be handed back. After the crisis, rules tightened sharply on how much client collateral could be rehypothecated, which is one reason collateral velocity fell after 2008 and never fully recovered.

There is a deeper systemic point hiding here that is worth drawing out. Economists who study this call the number of times a single piece of collateral is re-used the *collateral multiplier*, and it behaves uncannily like the money multiplier in ordinary banking — the mechanism by which a banking system turns a small base of reserves into a much larger stock of deposits and loans. (We cover that cousin process in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).) Just as banks create money by lending out the same reserves many times over, the repo system creates *funding* by re-pledging the same bonds many times over. Both are wonderful machines for converting something scarce into something abundant. And both share the same flaw: the abundance is built on confidence, and when confidence cracks, the multiplier runs in reverse. Deposits and loans contract when banks stop lending; repo funding contracts when collateral stops circulating. A falling collateral multiplier is, in effect, a tightening of financial conditions that no central bank explicitly set — which is one reason the post-2008 contraction in collateral re-use made the system feel starved of funding even when reserves were ample.

The practical takeaway for reading any leveraged firm is this: do not just ask how much collateral it holds, ask how many times that collateral has already been promised to someone else. A firm sitting on a billion dollars of bonds that are unencumbered — pledged to nobody — is genuinely a billion dollars safer than a firm whose billion in bonds is the third link in three different repo chains. The same nominal asset can be a fortress or a trap depending entirely on whether anyone else has a prior claim on it.

## Repo versus the other ways a bank funds overnight

A bank that needs cash overnight has three doors it can knock on, and the choice among them tells you a lot about how stressed the bank — or the system — is.

The first door is **repo**: secured, cheapest, the everyday workhorse. The second is the **unsecured interbank market**, where one bank lends another bank cash overnight on nothing but the borrower's name and creditworthiness — no collateral. Because the lender is exposed to the borrower's default, the unsecured rate is higher than the repo rate; the gap between them is essentially the market's price of bank credit risk. The third door is the **central bank's standing facility** — the Fed's discount window, or the equivalent at other central banks — where a bank can borrow cash against eligible collateral directly from the central bank itself. This is the backstop of last resort: it is always there, but it is priced at a penalty above market rates, and historically banks have been reluctant to use it for fear of looking desperate (the "stigma" problem).

![Matrix comparing repo unsecured interbank and central bank facility funding](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-6.png)

Reading the three together is a diagnostic tool. In calm times, a bank funds mostly in repo and a bit unsecured, and the central-bank window goes unused. When stress rises, you see the symptoms in sequence: first the spread of unsecured rates over repo widens (credit fear); then haircuts in repo rise and term repo dries up (collateral fear); and finally, when even secured overnight funding becomes unreliable, banks crawl to the central-bank window despite the stigma. The progression from repo to unsecured to the discount window is, in effect, a thermometer for how badly a bank's normal funding has broken down.

#### Worked example: the cost of losing repo access

Suppose a mid-size dealer funds \$20 billion of assets overnight. In normal times it funds them in repo at 4.00%. A scare hits and its repo lenders walk; to keep the lights on, it must replace that funding unsecured at, say, 4.80% — an 80-basis-point jump.

$$
\text{extra annual cost} = \$20{,}000\text{m} \times 0.80\% = \$160\text{ million}
$$

Eighty basis points may sound trivial, but on \$20 billion it is \$160 million a year — likely far more than the dealer earns on the spread it was running in the first place. And that is the *good* outcome, where unsecured funding is still available. If it is not, the dealer must sell \$20 billion of assets into a stressed market, crystallizing losses, or go to the central bank and signal distress to the world. The one-sentence intuition: losing cheap secured funding does not just raise a bank's costs — past a certain point it removes the bank's ability to exist at its current size at all.

## Common misconceptions

**"Repo is a sale, so the borrower no longer owns the bond."** Legally a repo *is* structured as a sale and a repurchase, and title to the bond does pass to the lender. But economically the borrower keeps the bond's upside and downside: they get it back tomorrow at a price set today, so any change in the bond's market value belongs to them, not the lender. The lender holds the bond purely as security and earns only the repo rate. Treating repo as a real sale of the economic position is the single most common beginner error; it is a *financing*, not a divestment.

**"Treasuries are safe, so a Treasury repo can't blow up."** The collateral being safe does not make the *funding* safe. September 2019 was a freeze in repo against pristine US Treasuries — the safest collateral on earth — and the rate still leapt toward 10%. The risk in repo is rarely the collateral's credit; it is the risk that the *cash* stops showing up, or that the haircut jumps, or that the plumbing jams. A market can seize even when every bond in it is perfectly money-good.

**"A haircut just protects the lender; it doesn't really cost the borrower much."** A 2% haircut sounds small, and on a single trade it is. But the haircut *is* the borrower's leverage limit (leverage = 1 ÷ haircut) and the borrower's first loss. A move from 2% to 4% halves the implied leverage and, for a fully-extended borrower, forces an immediate cash call. Haircuts are not a static fee; they are a dynamic dial the lender turns, and turning it is how a calm market becomes a run.

**"Rehypothecation is a shady loophole."** It is neither shady nor a loophole — it is a deliberate, disclosed feature that makes scarce high-quality collateral go further, and the modern financial system genuinely depends on it to function. The problem is not that it exists but that it lengthens the chain of dependency: when a chain of re-pledged collateral has to unwind in a panic, many parties find they are all relying on the same underlying bond. The fix after 2008 was to *limit* how much client collateral can be re-used, not to ban the practice.

**"The 2008 crisis was about subprime mortgages defaulting."** Defaulting mortgages were the spark, but the thing that turned a bad-loan problem into a system-wide seizure was the run on repo. Firms like Bear Stearns and Lehman did not fail mainly because their assets defaulted overnight; they failed because their *overnight funding* evaporated as repo lenders raised haircuts and refused to roll. The crisis was, mechanically, a wholesale-funding run — and repo was where it ran.

## How it shows up in real banks

### 2008: the silent run on repo

The 2007–08 crisis is usually told as a story about subprime mortgages, but the part that actually broke the financial system was a run — not the old-fashioned kind with depositors queuing outside a branch, but an invisible run on repo. Economists Gary Gorton and Andrew Metrick later named it exactly that: "the run on repo."

Here is the mechanism. The big investment banks and dealers — Bear Stearns, Lehman Brothers, Merrill Lynch — funded enormous balance sheets with short-term, largely overnight, repo. Much of the collateral was not Treasuries but mortgage-backed securities and structured products. As house prices fell and doubts spread about what those bonds were really worth, repo lenders did two things. They raised haircuts on anything that smelled of mortgages — for some structured products, haircuts went from a couple of percent to 25%, 45%, or higher, meaning the same bond raised far less cash. And then, for the worst collateral, they simply refused to lend against it at all.

A firm running on overnight repo cannot survive that. Every morning it has to roll its funding; if lenders raise the haircut, it must instantly find more cash or sell assets; if lenders refuse, it has hours, not days, to live. To plug the gap, firms sold assets into a market where everyone else was selling the same assets for the same reason, so prices fell further, which made the collateral look worse, which justified higher haircuts still. That self-reinforcing spiral — doubt → higher haircuts → forced sales → lower prices → more doubt — is the anatomy of a repo run.

![Pipeline of a run on repo showing doubt haircuts pulled cash and fire sales in 2008](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-7.png)

Bear Stearns died of it in March 2008: its repo lenders pulled back over a matter of days and the firm, unable to fund itself, was sold to JPMorgan in a Fed-backed rescue. Lehman died of it in September 2008. Lehman's case is doubly instructive because it also used an accounting trick called Repo 105 — booking around \$50 billion of repos as *sales* rather than financings at quarter-end, to flatter its reported leverage — which delayed the market's recognition of how fragile its funding was. When the recognition came, the run was merciless. We cover that failure in full in [Lehman Brothers 2008](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank). The system-wide lesson is brutal and simple: an institution can be nominally solvent and still die overnight if its repo funding runs, because no amount of long-term assets helps you when you cannot fund them until morning.

### September 2019: the spike

The 2019 episode is the opposite kind of failure — not a credit panic but a pure plumbing jam — and it is instructive precisely because nothing was actually wrong with anyone's solvency.

On September 16–17, 2019, two ordinary cash drains coincided. US corporations pulled tens of billions out of money markets to pay their quarterly taxes, and a large batch of new Treasury debt settled, requiring dealers to come up with cash to pay for the bonds they had bought. Both drains hit the same overnight repo market on the same day. Normally, banks awash in reserves would step in and lend the cash, smoothing the spike. But after years of the Fed shrinking its balance sheet, the level of reserves in the system had fallen to the point where the biggest banks were no longer comfortable lending out their cash freely overnight. Demand for overnight cash surged; the usual supply did not show up.

The result was the spike in our opening: the overnight repo rate, normally near the Fed's 2.25% target, leapt intraday toward 10%, and the official secured overnight benchmark printed 5.25% — more than double its prior level.

![Line chart of the September 2019 overnight repo rate spike](/imgs/blogs/the-repo-market-and-how-banks-fund-overnight-5.png)

The New York Fed responded within hours, injecting cash through its own repo operations — lending against Treasuries to flood the market with the reserves it lacked — and it kept those operations running for months, eventually resuming outright purchases of Treasury bills to rebuild the level of reserves. The rate came back to earth almost as fast as it had spiked. The lesson here is subtle and important: even with perfect collateral and no credit fear anywhere, the repo market can seize simply because there is not enough cash in the right place at the right moment. Repo's smoothness depends on a deep, reliable pool of reserves sitting behind it; let that pool get too shallow and the most liquid market in the world can briefly stop working.

### Central-bank repo facilities: turning the freeze into a faucet

The deepest lesson of both episodes is that central banks have learned to treat the repo market as a control panel. They no longer just set an overnight target and hope; they actively use repo to manage the supply of cash.

On the lending side, a central bank can do *repo operations*: lend cash to the market against high-quality collateral, exactly as it did in September 2019, to put a *ceiling* on the overnight rate. The Fed has since made this permanent through a Standing Repo Facility, which stands ready to lend cash against Treasuries on demand at a set rate — a structural promise that the September 2019 spike will not be allowed to recur. On the *absorbing* side, a central bank can run *reverse repo operations*: take cash *in* from money funds and others against its own bonds, which puts a *floor* under the overnight rate by giving cash holders a guaranteed safe place to lend. In the years after the pandemic, the Fed's overnight reverse repo facility at times absorbed more than \$2 trillion of cash a night — a vast drain that kept short-term rates from collapsing when the system was flooded with reserves.

Read together, the central bank now brackets the overnight rate with repo on one side and reverse repo on the other — lending cash to cap the rate, absorbing cash to floor it. The market that froze in 2008 and seized in 2019 has become, by design, one of the main levers of monetary control. Which is fitting, because repo was never a sideshow; it was always the place where the price of overnight money is actually set.

## The takeaway: read the haircut, not the headline

If you remember one thing about repo, make it this: a repo is a pawn ticket on a bond, and the haircut — not the interest rate — is where the danger lives. The repo rate is just the small price of overnight cash; the haircut is the lender's cushion, the borrower's leverage limit, and the lever that turns a calm market into a stampede. When you read that "funding markets are stressed," the question to ask is not "did the rate tick up?" but "are haircuts rising and is anyone refusing to roll?" Those are the symptoms that precede a run.

This connects straight back to the spine of how a bank lives or dies. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long and survives only as long as its short funding keeps showing up each morning. Repo is the purest expression of that fragility. It is the cheapest funding precisely because it is the most willing to flee — wholesale, professional money that demands collateral and disappears the instant the collateral looks shaky or the cash gets scarce. A bank or dealer that funds long-dated, hard-to-sell assets with overnight repo is making the maturity-transformation bet at its most extreme, and it lives or dies on the willingness of strangers to roll a loan that matures every single morning.

So how do you *use* this? When you look at any bank or dealer, find out how much of its funding is overnight wholesale money like repo, and against what collateral. A firm funding pristine Treasuries overnight can probably ride out a storm; a firm funding illiquid, doubtful assets overnight is one haircut spike away from a run, no matter how solvent it looks on paper. The balance sheet tells you whether the firm is *solvent*; the funding profile tells you whether it can survive until *tomorrow morning* — and in a panic, tomorrow morning is the only deadline that matters. Repo is where you go to find out which kind of bank you are looking at.

And the next time the financial news mentions, almost in passing, that "repo rates spiked" or "funding markets tightened," you will know it is not a footnote. It is the system's pulse, taken at the one market where the price of overnight money is actually set — the market that quietly funds the world every night, and the first place that breaks when confidence does. Watch the haircut, watch who refuses to roll, and you are watching the early-warning system that most people never even know is there.

## Further reading & cross-links

- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the system-level view: how repo connects money funds, hedge funds, securities lenders, and the dealers into the non-bank credit machine.
- [The funding stack: deposits, wholesale funding, bonds and covered bonds](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds) — where repo sits on the ladder of how a bank funds itself, from sticky deposits to overnight wholesale money.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the rules that exist precisely because overnight funding like repo can vanish, and how banks are required to hold buffers against that.
- [Lehman Brothers 2008: leverage, Repo 105 and the run on an investment bank](/blog/trading/banking/lehman-brothers-2008-leverage-repo-105-and-the-run-on-an-investment-bank) — the case study of a repo run killing a firm, and the Repo 105 accounting trick that hid the fragility.
- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the spine of the whole series: why borrowing short and lending long is the fragile trade behind every bank.

*This is educational material about how the repo market works, not financial advice.*

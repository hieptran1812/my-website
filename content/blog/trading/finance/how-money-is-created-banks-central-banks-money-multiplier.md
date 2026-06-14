---
title: "How Money Is Actually Created: Banks, Central Banks, and the Money Multiplier"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles tour of where money really comes from: commercial banks create most of it when they make loans, the central bank supplies reserves to hit its rate, and money vanishes again when loans are repaid."
tags: ["money-creation", "banking", "central-banks", "money-multiplier", "monetary-policy", "fractional-reserve", "bank-capital", "quantitative-easing", "macroeconomics"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Most money in the modern economy is created by ordinary commercial banks the moment they make a loan, not by the central bank printing notes; the textbook "money multiplier" gets the direction backwards.
>
> - When a bank makes a loan it credits a new deposit into your account; both sides of its balance sheet expand and brand-new money exists that did not before.
> - The central bank does not pre-fund this. Banks lend first and find the reserves they need afterward, because the central bank supplies reserves on demand to keep its interest rate on target.
> - What actually limits money creation is loan demand, bank capital, regulation, and profitability — not the quantity of reserves.
> - Money is destroyed in mirror image: when you repay a loan, the deposit is deleted and that money ceases to exist.
> - The one number to remember: in most rich economies, somewhere around 90–97% of the money supply is bank deposits created by lending, and only a small slice is physical cash and central-bank reserves.

Here is a question almost everyone has wondered about but few can answer cleanly: when a bank gives you a \$10,000 loan, where does the \$10,000 come from? Does the bank reach into a vault of other people's savings and hand you their cash? Does it call the central bank and ask for a fresh stack of notes? Or does it simply *type the number into your account*?

The honest answer — the one a central-bank economist would give you and the one most textbooks still bury — is the third. The bank types the number in. No saver's money is moved. No notes are printed. Two new lines appear in the bank's ledger, and \$10,000 of money that did not exist a second ago now sits in your account, ready to spend. Money, in the modern economy, is mostly created by private banks at the stroke of a keyboard, and it is destroyed again just as quietly when the loan is paid back.

The diagram above is the mental model for this whole post: the comfortable textbook story (reserves arrive first, the bank lends them out, the system "multiplies" them) is on the left, and what actually happens (the loan comes first and creates the deposit; reserves are sourced later) is on the right. If you internalize the difference between those two pictures, everything else — why printing money does not automatically cause inflation, why the post-2008 explosion in central-bank reserves did *not* set off hyperinflation, what really stops banks from lending infinitely — falls into place.

![Textbook reserves-first money creation versus reality where a loan creates the deposit](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-1.png)

This is not a fringe view. The Bank of England published a paper in its 2014 Quarterly Bulletin titled "Money creation in the modern economy" that says it plainly: "The majority of money in the modern economy is created by commercial banks making loans." Deutsche Bundesbank said the same in 2017. The reason it still surprises people is that the simplified story is taught first, taught widely, and rarely un-taught. Let us build the correct picture from the ground up, defining every term as we go, and ground each step in arithmetic you can check on a napkin.

## The basics: what money even is

Before we can say where money comes from, we need to agree on what money *is*. Economists define money by what it does, through three jobs.

**Medium of exchange.** Money is the thing you hand over to get goods and services, so you do not have to barter. Without money, a baker who wants shoes must find a cobbler who happens to want bread (the "double coincidence of wants"). Money removes that friction: the baker sells bread to anyone, receives money, and buys shoes from anyone. Anything widely accepted in trade is serving as a medium of exchange.

**Unit of account.** Money is the ruler we measure value with. A coffee is "\$4," a car is "\$30,000," your salary is "\$70,000 a year." Prices, debts, and contracts are all denominated in this unit. In the United States that unit is the dollar; in the eurozone, the euro. The unit of account is defined by the state — the government decrees that taxes are owed in dollars, which is a big part of why everyone keeps their books in dollars.

**Store of value.** Money lets you move purchasing power across time: earn today, spend next year. It is an imperfect store — inflation eats at it — but a \$20 bill kept in a drawer is still worth roughly \$20 next month. (Contrast that with, say, fresh fish, which is a terrible store of value.)

Anything that does all three jobs reasonably well is money. Notice this definition says nothing about gold, government, or banks. Cigarettes have served as money in prisons; cattle did in the ancient world; large carved stones did on the Pacific island of Yap. What matters is acceptance, not material.

There is one more property worth naming because it explains why bank deposits can be money at all: **liquidity**, the ease with which something can be spent or converted to spending power without loss of value. Physical cash is perfectly liquid. A balance in your checking account is nearly as liquid — you can spend it instantly by card or transfer — which is exactly why it counts as money even though it is "only" a promise from a bank. A house is highly *valuable* but highly *illiquid*: you cannot buy groceries with a spare bedroom. The reason deposits dominate the money supply is that the banking system has made them as spendable as cash, so for everyday purposes a number in your account and a banknote in your wallet feel identical — even though, as we will see, they are fundamentally different kinds of promise from fundamentally different issuers.

### Base money versus broad money

Here is the first distinction that trips people up. There are two very different *kinds* of money in a modern economy, and conflating them is the root of most confusion.

**Base money**, also called M0, central-bank money, or "high-powered money," is money created directly by the central bank. It comes in exactly two forms: (1) physical cash — the notes and coins in your wallet — and (2) **reserves**, which are electronic balances that commercial banks hold in their accounts *at the central bank*. You and I cannot hold reserves; only banks have accounts at the central bank. Base money is the central bank's own liability, its IOU.

**Broad money** is the money the public actually holds and spends. It is measured in aggregates with names like M1 and M2:

- **M1** is the most liquid money: physical cash held by the public, plus the balances in checking and other instantly spendable deposit accounts.
- **M2** is M1 plus slightly less liquid balances: savings accounts, small time deposits (like a modest certificate of deposit), and retail money-market funds.

The crucial fact: the overwhelming majority of broad money is **commercial-bank deposits** — numbers in your bank account — not cash and not reserves. In the United States, M2 is roughly \$21 trillion, while physical currency in circulation is only around \$2.3 trillion. The rest, the great bulk, is deposits. Those deposits are created by banks when they lend. Hold that thought; it is the whole game.

To put a number on "the great bulk": across most advanced economies, physical cash is only on the order of 3–10% of the broad money supply, and central-bank reserves are another small slice. Roughly 90–97% of the money the public holds and spends is deposits at commercial banks — money that came into existence when a bank made a loan. That single statistic is the reason this post focuses almost entirely on commercial banks rather than the central bank: the central bank issues the small, foundational core, but private banks create the vast majority of the money that actually circulates in the economy. If you want one fact to anchor everything else, it is this: in the modern economy, most money is private, and it is born from lending.

### Reserves versus deposits — do not mix them up

This pairing causes more confusion than any other, so let us nail it down with a concrete picture.

- A **deposit** is what *you* have at *your bank*. It is the bank's promise to pay you. When your banking app says "\$3,000," that is a deposit — a liability of the bank to you.
- A **reserve** is what *your bank* has at the *central bank*. It is the central bank's promise to pay the bank. Reserves are the banks' own spending money, used to settle payments with each other.

These live on different ledgers and never directly become each other. When you "withdraw cash," your deposit (bank's IOU to you) shrinks and the bank hands you physical currency (central bank's IOU), which the bank gets by drawing down its reserves. But your everyday spending — card payments, transfers, direct debits — almost never touches physical cash at all. It moves deposits between accounts and, behind the scenes, moves reserves between banks to settle the net difference.

### Assets, liabilities, and double-entry bookkeeping in one paragraph

The last building block is the simplest accounting idea in the world. Every entity has a balance sheet with two sides that must always be equal: **assets** (things it owns or is owed — cash, loans it has made, buildings) and **liabilities plus equity** (things it owes — deposits, bonds it issued — plus the owners' stake). The rule of *double-entry bookkeeping* is that every transaction touches at least two entries so the two sides stay equal. If a bank gains an asset, it must simultaneously gain an equal liability (or lose another asset, or gain equity). This single rule — assets always equal liabilities plus equity — is the lever that lets a bank create money: it can grow an asset and a liability at the same instant, out of thin air, and its books still balance. We will see exactly how in a moment.

## The textbook story: the money multiplier

Almost every intro economics course teaches money creation through the **money multiplier**, also called fractional-reserve banking. The story is intuitive, internally consistent, and — as a literal description of how banks operate today — wrong. But you should understand it, both because you will hear it everywhere and because seeing precisely *where* it goes wrong is what makes the correct picture click.

The story goes like this. Suppose there is a **reserve ratio** — a rule (or habit) that a bank must keep, say, 10% of its deposits as reserves and may lend out the other 90%. Now suppose \$1,000 of fresh base money enters the banking system — say the central bank buys a bond from someone, who deposits the \$1,000 at Bank A.

#### Worked example: the \$1,000-to-\$10,000 multiplier

Let us walk the chain, keeping a running tally.

- **Round 1.** Bank A receives a \$1,000 deposit. It must keep 10% (\$100) as reserves and lends out \$900. That \$900 gets spent and ends up deposited at Bank B.
- **Round 2.** Bank B now has a \$900 deposit. It keeps 10% (\$90) and lends out \$810. That \$810 ends up at Bank C.
- **Round 3.** Bank C keeps \$81, lends \$729.
- **Round 4.** \$729 becomes \$65.61 reserves and \$656.10 lent. And so on.

Each round the new lending shrinks by 10%. Add up *all* the deposits created across infinitely many rounds: \$1,000 + \$900 + \$810 + \$729 + ... This is a geometric series with ratio 0.9, and it sums to \$1,000 ÷ (1 − 0.9) = \$1,000 ÷ 0.1 = \$10,000.

So the textbook claims that \$1,000 of base money, passed hand to hand and re-lent at a 10% reserve ratio, "multiplies" into \$10,000 of total deposits. The multiplier is 1 ÷ reserve ratio = 1 ÷ 0.10 = 10. The intuition is that reserves are the scarce raw material, the central bank controls how much of it exists, and banks mechanically multiply it up by a fixed factor.

**The takeaway of the worked example:** in the textbook, money creation is a passive, mechanical chain that turns a small amount of central-bank reserves into a much larger amount of bank deposits.

It is a beautiful little model. It is also misleading in three concrete ways:

1. **It gets the order backwards.** It says reserves come first and lending follows. In reality, banks decide to lend first, create the deposit immediately, and obtain the reserves afterward.
2. **Reserves are not the binding constraint.** Many countries (Canada, the UK, Australia, and since 2020 the United States) have a reserve requirement of *zero*, and banks still do not lend infinitely. If reserves were the brake, a zero requirement would mean infinite money. It does not.
3. **The central bank does not fix the quantity of reserves and let the multiplier run.** It fixes a *price* — an interest rate — and supplies whatever quantity of reserves the banking system needs at that rate. We will see why this completely changes the picture.

Keep the \$10,000 figure in your head, though, because the *correct* story produces a strikingly similar-looking outcome by a completely different mechanism — and that coincidence is exactly why the wrong model survived so long.

## How it really works: a loan creates a deposit

Here is the part that, once seen, cannot be unseen. Let us make a single, simple loan and watch the bank's balance sheet.

#### Worked example: a \$10,000 loan creates \$10,000 of new money

You walk into Bank A and are approved for a \$10,000 loan. The bank does the following, in one keystroke:

- It records a new **asset**: "Loan to you, \$10,000." You owe the bank \$10,000, and a debt owed *to* the bank is something it owns — an asset.
- It records a new **liability**: "Deposit in your account, \$10,000." The bank now owes *you* \$10,000 that you can spend — a deposit, which is the bank's IOU to you.

That is the entire transaction. Two new lines. Before the loan, neither line existed. After it, the bank's assets are \$10,000 larger and its liabilities are \$10,000 larger, so the books still balance — assets = liabilities + equity, just as before, only both sides are bigger. And critically: **the \$10,000 in your account is brand-new money.** It is part of the broad money supply (M1) that did not exist a moment ago. No other account anywhere went down. The bank did not take \$10,000 from a saver and pass it to you. It conjured a deposit by typing it, balanced by your promise to repay.

![A bank loan booked as an asset and an equal new deposit liability expanding both sides](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-2.png)

Read that again with the figure above. A loan does not move money from one pocket to another. It *expands the balance sheet on both sides simultaneously*. The asset (your debt) and the liability (your deposit) are created together, in equal amounts, from nothing but the loan agreement. This is what people mean by the slogan "loans create deposits." It is the exact reverse of the deposits-then-loans story you were taught.

"But surely," you object, "the bank can't just create money infinitely. There must be a constraint." There is — several, in fact — and we will get to all of them. But the constraint is *not* "the bank had to have the deposit first." It did not.

### Where do the reserves come from, then?

Here is the legitimate worry hiding behind the multiplier story. When you spend your new \$10,000 — say you wire it to a car dealer who banks at Bank B — Bank A has to *settle* that payment. Settlement happens in reserves: Bank A must transfer \$10,000 of reserves to Bank B. So the bank does need reserves; the multiplier story was not crazy to worry about them.

The error is in the *timing and the source*. Bank A does not need the reserves to *make* the loan. It needs them only *when the payment settles*, which can be moments or days later, and even then only the *net* amount after its incoming payments are subtracted. If Bank A made the loan and Bank B made a similar loan whose proceeds flowed back to Bank A, the two settlements net out and almost no reserves change hands.

And when a bank does come up short of reserves, it gets them — easily — in one of three ways:

1. It **borrows reserves from another bank** that has a surplus, in the overnight interbank market, at roughly the central bank's policy interest rate.
2. It **borrows directly from the central bank** at its lending facility (the "discount window" in the US), at a rate the central bank sets.
3. The **central bank supplies more reserves to the whole system** through open-market operations, precisely so that the interbank rate stays on its target.

This is the decisive point. The central bank targets an *interest rate*, not a *quantity* of reserves. If the banking system is short of reserves, the interbank rate would spike above target — so the central bank injects reserves until the rate falls back. If the system has too many reserves, the rate would fall below target, so the central bank drains them. Reserves are supplied *elastically, on demand, to hit the rate*. The quantity is whatever it takes. So a bank never has to worry "do enough reserves exist for me to lend?" — at the policy rate, they always do.

That is why the multiplier gets it backwards. Lending is not constrained by a fixed pool of reserves that gets divided up. Banks lend when they find a creditworthy borrower at a profitable rate; reserves to settle the resulting payments are sourced afterward at the going price. Loans drive deposits, deposits drive the *demand* for reserves, and the central bank supplies that demand to defend its rate. The causation runs almost exactly opposite to the textbook arrow.

#### Worked example: interbank netting shrinks the reserves a bank actually needs

The multiplier story makes it sound as if a bank must ship the full loan amount in reserves every time. It rarely does, because payments net out. Suppose on a given day Bank A's customers make \$50 million of payments to customers of Bank B, and Bank B's customers make \$48 million of payments back to Bank A's customers. At the end of the day, Bank A does not move \$50 million of reserves to Bank B — it moves only the *net*, \$50 million − \$48 million = \$2 million. The other \$48 million of flows cancel out against each other.

Now scale that up. A large bank processes billions of dollars of payments in both directions every day, and the vast majority offset. The reserves it actually needs to hold are tied to the small *net* imbalance and the regulator's liquidity rules, not to the gross volume of lending it does. A bank can create \$1 billion of new deposits through lending and, on a typical day, need only a few million dollars of reserves to settle the net flows that result. This is the second reason reserves are not the constraint: not only are they supplied on demand, but netting means the system needs far fewer of them than the gross lending would suggest.

**The takeaway:** because interbank payments net out, the reserves a bank needs to settle its lending are a small fraction of the loans it makes, so reserves can never be the thing that caps how much it lends.

## Central-bank money versus commercial-bank money

We now have two distinct moneys circulating, and it is worth being crystal clear about which is which, because your intuition probably treats "the money in my account" as if it were cash sitting in a vault with your name on it. It is not.

When your banking app shows \$3,000, you do not own \$3,000 of anything physical. You own a **promise** — the bank owes you \$3,000 and has agreed to pay it on demand, in cash or by transfer. Your deposit is **commercial-bank money**: an IOU issued by a private company. It is money because it is widely accepted and because the bank stands ready to convert it to cash or move it on your instruction. But it is the bank's liability, and its value rests on the bank's solvency (and, in practice, on deposit insurance and the central bank standing behind the system).

Physical cash and reserves, by contrast, are **central-bank money** — the IOU of the central bank, which can always honor IOUs denominated in its own unit because it issues that unit. This is the safest money there is.

The picture below shows how the layers stack up: a small core of central-bank base money (M0), with the much larger volume of commercial-bank deposits piled on top to form the broad money (M1, M2) that you actually spend.

![Money supply layers with small base money core under larger M1 and M2 deposits](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-3.png)

The reason this distinction matters in your life: in a bank run or a bank failure, the difference between central-bank money and commercial-bank money becomes very real. Cash in your hand is final; a deposit is a claim that can fail to pay if the bank collapses and is not backstopped. This is exactly why deposit insurance exists (the FDIC in the US insures deposits up to \$250,000 per depositor per bank) and why central banks act as "lender of last resort" — to make commercial-bank money trustworthy enough that we treat it as money. The system works precisely because, in normal times, you never have to think about the fact that your money is a private company's promise.

This two-tier structure — public central-bank money at the base, private commercial-bank money on top — is the architecture of every modern monetary system. If you want the broader map of who sits where in that system and who backstops whom, see [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system).

## What actually limits money creation

If banks can create deposits by typing them, why is the world not flooded with infinite money? Because lending is constrained — just not by reserves. The real brakes are loan demand, bank capital, regulation, and profitability. The graph below puts them together.

![Loan demand, bank capital, regulation, and profitability feeding into how much money is created](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-6.png)

### Loan demand

A bank cannot create money by lending if no one wants to borrow at the offered terms. Money creation requires a willing, creditworthy borrower on the other side of the table. In a recession, when households and firms are scared, paying down debt, and unwilling to take on more, banks can be desperate to lend and find few takers. Loan demand is the first and most underappreciated limit: the central bank can make borrowing cheap, but it cannot force anyone to borrow. This is the famous "pushing on a string" problem — you can push rates to zero and still get little new lending if demand is dead.

### Bank capital — the real ceiling

This is the constraint that has actually replaced reserves in the regulatory framework, and it is worth understanding precisely. **Capital** is the owners' stake in the bank — equity, the difference between its assets and its liabilities. It is the bank's own money at risk, the buffer that absorbs losses before depositors are touched. Regulators (under the international **Basel** accords) require banks to hold capital equal to at least a minimum percentage of their (risk-weighted) assets. A common simplified figure is an 8% capital ratio.

The key insight: every loan a bank makes is an asset, and assets carry a capital requirement. So *capital*, not reserves, sets the ceiling on how much a bank can lend. To grow its loan book, a bank must either have spare capital or raise more — by retaining profits or issuing shares.

#### Worked example: how \$1 of capital caps ~\$12.50 of assets

Suppose the rule is that a bank must hold capital equal to at least 8% of its assets. Turn that around: the maximum assets a bank can support with a given amount of capital is its capital divided by 0.08.

- With \$1 of capital: maximum assets = \$1 ÷ 0.08 = \$12.50.
- With \$1 billion of capital: maximum assets = \$1 billion ÷ 0.08 = \$12.5 billion.

So each \$1 of capital lets the bank carry up to \$12.50 of assets, the bulk of which will be loans — and each loan it makes creates a matching deposit. If the bank is already at its 8% limit and wants to lend another \$12.50, it must first find another \$1 of capital. If it cannot, it cannot lend, no matter how many reserves are available or how low the policy rate is.

**The takeaway:** in the modern system the binding constraint on money creation is bank capital, not reserves — \$1 of equity supports roughly \$12.50 of lending at an 8% ratio, and that is the real multiplier that matters.

Notice the structural similarity to the textbook multiplier: 1 ÷ 0.08 = 12.5 looks just like 1 ÷ 0.10 = 10. The arithmetic is the same shape. But the *thing being divided* is completely different — capital, not reserves — and the causation is reversed: the bank does not start with capital and mechanically lend it out; it lends to good borrowers and must hold enough capital against the resulting assets. The capital ratio is a ceiling the bank bumps into, not a pool it pours out.

### Regulation

Beyond capital, regulators impose **liquidity** requirements (banks must hold enough easily-sellable assets to survive a deposit outflow — the Liquidity Coverage Ratio under Basel III), leverage caps (a hard limit on assets relative to capital regardless of risk weights), and stress tests. Each of these can bind before capital does and limit how aggressively a bank expands its balance sheet. Regulation is the deliberate set of brakes society installs because the natural tendency of a profit-seeking banking system is to create *too much* credit in good times.

### Profitability

Finally, a bank only lends when lending is profitable. It must charge the borrower more than its own cost of funds plus the expected losses from default plus the cost of the capital it must hold. If the spread is too thin or the borrower too risky, the bank declines — and no money is created. Profitability is the day-to-day filter; demand and capital are the outer walls.

#### Worked example: the interest spread on a \$100,000 loan

Let us see why a bank wants to lend at all, and where its profit comes from. The business of banking is borrowing cheap and lending dear — the **net interest margin**.

Suppose a bank funds itself at 1% (it pays 1% interest on the deposits and short-term borrowing it uses) and lends at 5%. It makes a \$100,000 loan.

- Interest income on the loan: 5% × \$100,000 = \$5,000 per year.
- Interest cost of funding it: 1% × \$100,000 = \$1,000 per year.
- Gross spread (net interest income): \$5,000 − \$1,000 = \$4,000 per year, a 4-percentage-point margin.

Out of that \$4,000 the bank must cover operating costs, expected loan losses (if, say, 1% of such loans default and the bank loses half the balance, that is an expected \$500 a year), and a return on the capital it had to set aside. Whatever is left is profit. This is the engine: the spread between the rate it lends at and the rate it funds at, applied across its whole loan book, is where a bank's money comes from.

**The takeaway:** banks create money as a *byproduct* of a profit-seeking business — lending at a higher rate than they pay for funding — which is why they only create it when there is a creditworthy borrower and an adequate spread.

Notice what is *absent* from every one of these constraints: a fixed quantity of reserves to lend out. The brakes are demand, capital, regulation, and profit. Reserves are sourced on demand at the policy rate. That is the modern reality.

## Destroying money: repaying a loan deletes the deposit

If loans create money, what destroys it? The mirror image: **repaying a loan destroys money.** This is the half of the story almost no one is told, and it is essential — otherwise you would expect the money supply to grow forever, which it does not.

#### Worked example: repaying a \$10,000 loan destroys \$10,000 of money

Go back to your \$10,000 loan. A year later you pay it back. (For simplicity ignore interest for a moment and just trace the principal.) You instruct the bank to take \$10,000 from your account to settle the debt. On the bank's balance sheet:

- The **asset** "Loan to you, \$10,000" disappears — the debt is settled, so the bank no longer owns it.
- The **liability** "Deposit, \$10,000" disappears — \$10,000 is debited from your account to make the payment.

Both sides shrink by \$10,000. The books still balance. And the \$10,000 of deposit money that was created when the loan was made *no longer exists.* It was not transferred to the bank's profits or moved to a vault — it was extinguished. The broad money supply just fell by \$10,000.

**The takeaway:** money is created when banks lend and destroyed when borrowers repay, so the money supply is the *net* of new lending minus repayments across the whole economy.

![Lending creating a deposit and money versus repayment deleting the deposit and money](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-7.png)

This symmetry, shown above, has profound consequences. The total amount of money in the economy is not a fixed stock that the government doles out; it is a *flow balance*. When new lending exceeds repayments (a credit boom), the money supply grows. When repayments exceed new lending — when households and firms collectively "deleverage," paying down debt faster than they take on new debt — the money supply *shrinks*, even if the central bank does nothing. This is exactly what happened after 2008 in parts of the economy and is a big reason recoveries from debt-driven busts are so slow: as everyone repays, money is destroyed, spending falls, and the economy can grind down even with interest rates at zero.

(What about the interest you pay on top of the principal? The interest is income to the bank, not destroyed money — it becomes part of the bank's revenue, used to pay its costs and its own depositors and to build capital. Only the *principal* repayment destroys money; the interest redistributes existing money to the bank.)

## QE and money creation: reserves rise, but it is bank-system money

In 2008 and again in 2020, central banks did something that looked, to a casual observer, exactly like "printing money": **quantitative easing**, or QE. It is worth understanding precisely what QE does and does not create, because it is the single biggest source of "the Fed printed trillions and we'll get hyperinflation" confusion.

In QE, the central bank buys large quantities of bonds (government bonds, sometimes others) from banks and other financial institutions. It pays for them by creating new **reserves** — central-bank money — and crediting them to the seller's bank. So QE absolutely creates money. But *which* money? It creates **reserves**: base money sitting in banks' accounts at the central bank. It does **not** directly create deposits in your account. It swaps one asset (a bond) for another (reserves) on the banks' balance sheets.

This is the crux. QE pumps up the *base* (reserves), but reserves are not the money you spend, and — as we established — reserves do not get "multiplied" into deposits by some mechanical factor. For QE to raise the broad money you actually spend, it has to work *indirectly*: by pushing down interest rates and asset prices in a way that, hopefully, makes banks and borrowers want to do more lending. If that lending does not materialize — if loan demand is weak and banks are cautious — then reserves can balloon while broad money and inflation barely move. Which is precisely what was observed after 2008.

I cover the full mechanics, the intended channels, and the debates in [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) — but the one-line version for our purposes is: QE creates reserves (bank-system money), not deposits (your spending money), and the link between the two is the very link the money multiplier wrongly assumed was mechanical.

## The timeline: how the nature of money has changed

It helps to see money creation in historical context, because what "backs" money — and who creates it — has changed dramatically, and the changes explain a lot of today's confusion.

![Money regimes from gold standard to 1971 fiat to 2008 QE to CBDC pilots](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-4.png)

- **The gold standard (roughly the 19th century to 1971, with interruptions).** A dollar was a claim on a fixed amount of gold; the government promised to convert notes to gold at a set rate. This did *not* mean every dollar was a gold coin — even then, banks created deposits by lending, far in excess of the gold backing. The gold link constrained the *total* somewhat, and it constrained governments, but it also made the system rigid and crisis-prone (a run on gold could force brutal contractions).
- **1971, the Nixon shock.** President Nixon ended the dollar's convertibility to gold. From that point the dollar — like every major currency today — became **fiat money**: money that is money because the government says so and because everyone accepts it, *not* because it is backed by metal. This is the regime we live in now, and it is the source of the recurring complaint that money is "backed by nothing." It is backed by something — the productive economy, the legal system, and the requirement to pay taxes in it — just not by gold.
- **2008, the QE era.** After the financial crisis, central banks created reserves on a previously unimaginable scale. The Federal Reserve's balance sheet went from roughly \$0.9 trillion before the crisis to around \$2.7 trillion within a couple of years (and eventually far higher). This was central banks directly expanding base money — the closest thing to "printing" in the modern toolkit — and yet, as we will see, broad inflation stayed low for a decade afterward.
- **The 2020s, CBDCs.** Several central banks are now piloting **central bank digital currencies** — digital base money that the public could, in principle, hold directly. A CBDC would, for the first time in the modern era, let ordinary people hold central-bank money electronically rather than only as physical cash, potentially reshaping the two-tier structure we described. It is the next possible chapter in what money *is*.

## A field guide to the kinds of money

Putting the pieces together, it helps to lay out the distinct types of money side by side: who issues each, what backs it, and whose IOU it ultimately is. The matrix below is the reference card.

![Types of money by who issues them what backs them and whose IOU they are](/imgs/blogs/how-money-is-created-banks-central-banks-money-multiplier-5.png)

- **Physical cash** is issued by the central bank, backed by the credit of the government and the economy, and is the central bank's IOU. It is the only form of central-bank money the public can hold today.
- **Bank reserves** are also issued by the central bank and are its IOU, but only banks can hold them. They are the settlement money of the banking system.
- **Your deposit** is issued by your commercial bank, backed by the bank's assets (mostly its loan book), and is the *bank's* IOU to you — not the central bank's. This is the bulk of the money supply.
- **A stablecoin** (a crypto token pegged to a dollar) is issued by a private company, backed (if honest) by reserves the issuer holds, and is the *issuer's* IOU. It is a modern, unregulated cousin of a bank deposit, and it inherits the same risk: it is only as good as the issuer's solvency and the quality of its backing.

The single thread running through the whole table: almost no modern money is "backed by gold." Money is a web of IOUs, layered on top of each other, anchored at the bottom by the central bank's promise and the state's power to tax in the currency.

## Common misconceptions

Now that we have the correct model, let us correct the most common wrong beliefs head-on. Each is wrong in an instructive way.

**Misconception 1: "Banks lend out the deposits that savers put in."** This is the deposits-then-loans story, and it is backwards. Banks do not need a pre-existing deposit to lend; they create a *new* deposit when they lend. Aggregate deposits are mostly the *result* of bank lending, not the *source* of it. An individual bank does care about funding (it needs reserves to settle payments and deposits are a cheap funding source), which is why the intuition feels right at the level of one bank — but for the banking system as a whole, lending creates the deposits, not the other way around. Why it matters: if you believe banks merely intermediate savings, you will misunderstand both how credit booms happen and why the central bank cannot simply force more lending by adding reserves.

Here is the subtle reconciliation that confuses even careful people. From the viewpoint of a *single, small* bank, the deposits-then-loans story looks almost true: when that one bank makes a loan and the borrower spends the money to a customer of a rival bank, the small bank loses reserves and must fund them. So a single bank does behave as if it needs funding *before* it can comfortably lend, and it competes for deposits to get cheap funding. But step back to the *whole system*. When the borrower spends the new deposit, the money does not leave the banking system — it just moves from one bank to another. Across all banks together, the deposit created by the loan still exists; it has merely changed address. So at the system level, lending unambiguously creates deposits, and no pre-existing pool of savings was required. The single-bank view and the system view are both correct about different things — and mistaking the single-bank funding constraint for a system-level savings constraint is the precise error at the heart of the textbook model.

**Misconception 2: "Printing money always causes inflation."** Inflation comes from too much *spending* chasing too few goods. Creating reserves (QE) is not the same as creating spending power in the hands of people who will spend it. After 2008 the Fed created trillions in reserves and inflation stayed below target for years, because the reserves sat in the banking system and broad money and demand did not surge. Money creation *can* cause inflation — when it finances a surge in spending that outruns the economy's capacity — but the link is through spending, not through the act of creation itself. Why it matters: it explains why "money printing → hyperinflation" predictions repeatedly failed after 2008, and why the 2021–2022 inflation, which *did* involve money landing directly in people's spending accounts via fiscal transfers, behaved differently.

**Misconception 3: "Money is (or should be) backed by gold."** Modern money is fiat: it is not redeemable for gold and has not been since 1971. It holds its value because the issuing government is stable, the economy is productive, taxes must be paid in it, and the central bank manages its supply to keep inflation low. A gold link does not make money "real" — it just ties the money supply to the accidental rate at which gold is mined, which is why every major economy abandoned it. Why it matters: the "backed by nothing" complaint misdiagnoses both what gives money value and what could go wrong with it.

**Misconception 4: "The central bank controls the money supply by setting the quantity of reserves, which then get multiplied."** The central bank sets an interest *rate* and supplies whatever reserves the system demands at that rate. It influences the money supply *indirectly* — by making borrowing cheaper or dearer, which changes loan demand and bank lending — not by metering out a fixed amount of reserves that banks multiply. The money multiplier as a control lever does not describe how modern central banks operate. Why it matters: it is the difference between thinking the Fed "prints" the money supply and understanding that it nudges a privately-driven credit process.

**Misconception 5: "Reserves get multiplied into many times more deposits by a fixed factor."** The ratio of broad money to base money (the observed "multiplier") is not a fixed mechanical constant; it is just the *outcome* of how much banks happened to lend, divided by however many reserves happen to exist. After QE flooded the system with reserves, that ratio *collapsed* — banks did not lend ten times the new reserves; the reserves mostly just sat there. A genuine causal multiplier would not behave that way. Why it matters: it is the cleanest empirical refutation of the textbook model — the "multiplier" is a ratio you compute after the fact, not a machine that runs.

**Misconception 6: "Bank deposits are the bank holding your cash safely in a vault."** Your deposit is not stored cash; it is a *loan you have made to the bank* (an unsecured one, mostly), recorded as the bank's liability to you. The bank has used the funding to make loans and buy assets; only a small fraction is held as cash or reserves. This is why a bank can be solvent yet illiquid — perfectly good loans on its books but not enough ready cash if everyone demands their deposits at once. Why it matters: it is the entire reason bank runs exist and the reason deposit insurance and a lender of last resort are necessary.

## How it shows up in real markets

Theory is cheap. Here are concrete episodes where this mechanism — or its misunderstanding — drove real outcomes, with dates and numbers.

### Post-2008: reserves exploded, lending and inflation did not follow

This is the single most important real-world test, and it is the one the textbook fails. From 2008 through about 2014, the Federal Reserve created enormous quantities of reserves through successive rounds of QE. Bank reserves at the Fed went from a few tens of billions of dollars before the crisis to over \$2.5 trillion. If the money multiplier described reality, that flood of base money should have multiplied — at a 10x multiplier — into something like a \$25 trillion surge in broad money, and runaway inflation. Many prominent commentators predicted exactly that. It did not happen. Broad money (M2) grew far more modestly, bank lending recovered only slowly, and core inflation ran *below* the Fed's 2% target for most of the following decade. The mechanism from this post explains why cleanly: reserves do not get multiplied into deposits. Deposits are created by lending, lending depends on demand and capital, and after a balance-sheet recession both were weak. The reserves piled up as excess reserves, earning interest, doing very little. The lesson: base money and broad money are different animals, and you cannot reason about inflation from the size of the central bank's balance sheet alone.

### The 2021–2022 inflation: when money landed in spending accounts

Contrast 2008 with the COVID-era stimulus. In 2020–2021, governments did not just expand reserves; they sent money *directly into households' deposit accounts* — stimulus checks, expanded unemployment benefits, business grants. This was fiscal policy creating spending power in the hands of people who would spend it, layered on top of monetary easing and supply chains that had seized up. US M2 grew by roughly 25% over 2020 — an extraordinary jump — and this time the money was broad money in spendable accounts, not idle reserves. Inflation followed, peaking around 9% in mid-2022. The contrast with 2008 is the whole lesson of misconception 2: money creation causes inflation when it finances a surge in *spending* that outruns supply, not merely when a central-bank balance sheet grows. The *channel* — into spending accounts versus into bank reserves — is what mattered.

### Credit booms and busts: the money supply breathing

The Irish and Spanish property booms of the mid-2000s show money creation in fast-forward. Banks lent aggressively into rising property markets; each mortgage created a new deposit; the broad money supply ballooned; the new money chased property, pushing prices up, which justified more lending — a self-reinforcing credit boom. Irish private-sector credit roughly tripled in the decade to 2008. Then it reversed: when the bust came, new lending collapsed and borrowers (and failing banks) repaid or wrote off loans faster than new ones were made. Money was destroyed on a massive scale; the broad money supply contracted; spending cratered; the economy fell into a deep balance-sheet recession. This is the lending-creates / repayment-destroys symmetry from this post playing out at the level of a whole economy — the money supply expanding and contracting with the credit cycle, not with anything the central bank deliberately did.

### Weimar Germany and Zimbabwe: hyperinflation is fiscal, not banking

The textbook villain of money creation is hyperinflation, and the two canonical cases — Weimar Germany (1921–1923) and Zimbabwe (2007–2009) — are routinely cited as "what happens when you print money." But note the mechanism carefully: in both cases the money creation was the **government financing its own deficits** by having the central bank create money to hand directly to the state, which then spent it. Weimar Germany was printing notes to pay reparations and government wages amid collapsed production; Zimbabwe's central bank was monetizing government deficits after the productive economy (notably agriculture) had been destroyed. This is *fiscal* money creation — the state spending newly created money directly — not the *commercial-bank* lending mechanism this post is about, and crucially it happened against a backdrop of collapsing real output. The lesson is sharp: hyperinflation is what happens when a government creates money to fund spending while the economy cannot produce, not a generic consequence of banks making loans or central banks doing QE. Conflating the two is misconception 2 at its most dangerous.

### Negative or zero reserve requirements: the multiplier's quiet death

A subtle but decisive real-world fact: Canada has had a 0% reserve requirement since the early 1990s, the UK has no formal reserve requirement, and the United States cut its reserve requirement to *zero* in March 2020. Under the textbook multiplier (1 ÷ reserve ratio), a zero reserve ratio implies an *infinite* multiplier and therefore infinite money. Obviously that did not happen — the Canadian and US money supplies did not explode to infinity in 2020 for this reason. The reason is exactly the one in this post: reserves were never the binding constraint; capital, demand, regulation, and profitability are. The very existence of well-functioning banking systems with zero reserve requirements is, by itself, proof that the money multiplier is not how money creation works.

### Japan, 2001–2006: the first failed test of the multiplier

Long before the West tried QE, Japan ran the experiment. Facing deflation and a stagnant economy after its 1990s asset-bubble collapse, the Bank of Japan adopted "quantitative easing" in 2001, flooding the banking system with reserves and pushing the policy rate to zero. By the textbook multiplier, this should have multiplied into a large rise in broad money and ended deflation quickly. It did neither. Banks, sitting on bad loans and facing weak demand from over-indebted firms, simply held the extra reserves; broad money grew sluggishly; and Japan stayed in mild deflation for years. The reserves rose roughly fivefold without a corresponding surge in lending or prices. Japan was the canary: a full decade before 2008, it demonstrated that you can pump base money into the banking system and watch the "multiplier" collapse rather than multiply, because lending is driven by demand and bank balance-sheet health, not by the availability of reserves. The West relearned the same lesson the hard way after 2008.

### The Bank of England says so out loud

In 2014 the Bank of England's Monetary Analysis directorate published "Money creation in the modern economy" in its Quarterly Bulletin, stating that "the majority of money in the modern economy is created by commercial banks making loans" and explicitly that "the reality of how money is created today differs from the description found in some economics textbooks" — the multiplier story is "not an accurate description." When the institution that actually operates the monetary system writes a paper to correct the textbook, the textbook is wrong. The Bundesbank followed with a similar explainer in 2017. This is not a heterodox or fringe claim; it is the operating view of the people who run the plumbing.

### CBDCs: redesigning the two-tier system

Looking forward, central bank digital currencies could change *who* creates money. If the public could hold central-bank money directly (a CBDC) rather than only commercial-bank deposits, deposits might migrate from banks to the central bank, especially in a crisis (a digital bank run becomes one tap). That would shrink the deposit base from which banks lend and could constrain commercial money creation — which is exactly why most CBDC designs propose holding caps and a continued central role for commercial banks. The episode hasn't fully happened yet, but it is the clearest illustration that the architecture of money creation — public base layer, private credit layer — is a *design choice*, not a law of nature, and it can be redrawn.

## When this matters to you, and further reading

You will probably never need to make a journal entry on a bank's balance sheet. So why does any of this matter to you, concretely?

**It changes how you read the news.** When a headline screams "the Fed is printing trillions, hyperinflation is coming," you now know the right questions: Is this creating reserves or spending power? Is it landing in bank accounts at the central bank or in households' deposit accounts? Is the economy at capacity? Those questions separate the 2008 non-event from the 2021 inflation.

**It clarifies your own money.** The balance in your account is your bank's promise, not stored cash. That is why deposit insurance limits (\$250,000 per depositor per bank in the US) are worth knowing, why spreading large balances across institutions can matter, and why a bank's solvency is not a remote technicality but a direct claim on your money.

**It demystifies the credit cycle.** Booms and busts are, in large part, the money supply breathing — expanding as banks lend, contracting as loans are repaid or written off. Understanding that the money supply is a net flow of new lending minus repayments makes the slow, grinding recoveries from debt busts comprehensible rather than mysterious.

**It immunizes you against two opposite errors:** the gold-bug error that fiat money is fake and doomed, and the naive-MMT-caricature error that the government can create unlimited money with no consequences. The truth sits between: money is a network of IOUs that works as long as it is managed so that money creation roughly tracks the economy's capacity to produce — and breaks, in inflation or deflation, when it does not.

To go deeper into the system this sits inside, three companion pieces:

- [Who controls the world's money: the global financial system](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) — the map of central banks, commercial banks, and markets that this post zoomed into.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the mechanics of the policy rate that the central bank defends by supplying reserves on demand, which is *why* reserves are never the binding constraint on lending.
- [A field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) — what banks, shadow banks, and other intermediaries actually do, and where deposit creation fits among them.

And if you read one primary source, read the Bank of England's 2014 "Money creation in the modern economy." It is short, written for non-specialists by the people who run the system, and it says — from the inside — exactly what this post has argued: banks make a loan, a deposit appears, new money exists, and when the loan is repaid the money is gone again.

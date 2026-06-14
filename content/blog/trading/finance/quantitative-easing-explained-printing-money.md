---
title: "Quantitative Easing Explained: What Printing Money Actually Means"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-scratch guide to quantitative easing: why central banks create bank reserves to buy bonds, how that pushes long-term yields down and asset prices up, and why it skews toward the people who already own things."
tags: ["quantitative-easing", "central-banks", "monetary-policy", "bonds", "interest-rates", "federal-reserve", "inflation", "balance-sheet", "macroeconomics"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Quantitative easing is not the central bank printing cash and handing it to the public; it creates new bank reserves to buy bonds, which pushes long-term yields down and asset prices up once the short-term interest rate is already stuck at zero. It is powerful, blunt, and tilted toward whoever already owns assets.
>
> - QE expands both sides of the central bank's balance sheet at once: it buys bonds (an asset) and pays with newly created reserves (a liability). Nothing gets mailed to households.
> - Bond prices and yields move in opposite directions, so when the central bank buys bonds in size, their prices rise and the yields the whole economy borrows at fall.
> - It transmits through three channels: pushing investors out of bonds into riskier assets, signaling "rates low for long," and dragging down long-term borrowing costs like mortgages.
> - Reserves are not spendable money in the real economy by themselves, which is exactly why the 2010s saw enormous QE and almost no inflation. QE is also reversible: quantitative tightening, or QT, runs the process backward.
> - Remember one number: the Fed's balance sheet went from about \$0.9 trillion in 2007 to roughly \$9 trillion in 2022, and that growth happened by buying bonds, not by printing banknotes.

Here is a question almost everyone has asked at some point, usually in a slightly suspicious tone: *when the central bank "prints money," where does the money actually go, and why doesn't it cause runaway inflation every single time?* You have probably heard that after the 2008 crash, and again during the 2020 pandemic, central banks "printed trillions." You may also have noticed that for most of the 2010s, despite all that printing, prices barely moved at all. Those two facts seem to contradict each other. They do not. The reconciliation is the single most misunderstood policy in modern finance, and it has a clunky name: quantitative easing, or QE.

The diagram above is the mental model we will build the whole post on: QE expands a central bank's balance sheet on *both* sides at once. On the left, the bank buys bonds, and those bonds become its assets. On the right, it pays for them by creating new bank reserves, which are its liabilities. No truck of cash leaves the building. No money lands in your checking account. The "printing" is an electronic entry that swells the central bank's books and changes the price of one specific thing: long-term debt. Everything else QE does flows from that.

![Central-bank balance sheet before versus after QE, both sides expanding](/imgs/blogs/quantitative-easing-explained-printing-money-1.png)

*(Balance-sheet figures throughout this post are illustrative and approximate, as-of the dates noted; the U.S. Federal Reserve publishes the exact weekly numbers in its H.4.1 release.)*

By the end, you will understand exactly what QE is, why a central bank reaches for it, the three precise channels through which it works, the long list of things it does *not* do, how it gets unwound, who wins and who loses, and how it actually played out in Japan, the United States, and the COVID panic. We will ground every claim in dollar arithmetic you can check yourself. This is educational, not financial advice.

## Foundations: the four things you need before QE makes sense

QE sits on top of four ideas. If any one of them is fuzzy, QE will sound like magic or fraud. So we build each from zero. A reader who already trades bonds for a living can skim this section; a beginner cannot skip it.

### A central bank's balance sheet

A *balance sheet* is just a two-column list of what an institution owns and what it owes. The left column is *assets* — things of value the institution holds. The right column is *liabilities* — promises the institution owes to others — plus *equity*, which is whatever is left over for the owners. The defining rule, the one that gives the thing its name, is that the two sides must always be equal: total assets equal total liabilities plus equity. If you add something to one side, you must add something to the other.

A central bank, like the U.S. Federal Reserve (the "Fed"), the European Central Bank, or the Bank of Japan, has a balance sheet like any institution, but with an unusual pair of entries.

- On the **asset** side, the central bank mostly holds government bonds — IOUs issued by its own national treasury. (The Fed also holds some mortgage-backed securities, which we will define shortly.) These are things the central bank owns and that pay it interest.
- On the **liability** side sit two special kinds of money the central bank itself creates. The first is *currency* — the physical banknotes in your wallet, which are literally a liability of the central bank (look at a dollar bill: it is a "Federal Reserve Note"). The second, and the star of this whole story, is *reserves*.

A **reserve** is money that commercial banks hold in their accounts *at the central bank*. The structure is a hierarchy: ordinary people and businesses keep accounts at commercial banks; commercial banks keep accounts at the central bank. The balance in a bank's account at the central bank is called its reserves. Reserves are the safest, most final form of money in the system — when one bank pays another, what ultimately moves between them is reserves. Crucially, reserves are a kind of money that *only banks can hold*. You and I cannot have a reserve account at the Fed. This single fact is the key that unlocks most of the misconceptions about QE, so hold onto it.

The central bank has a superpower no one else has: it can create reserves out of nothing by typing a number into its own ledger. When it does, its liabilities go up (it now owes those reserves to the banks holding them), and it uses the freshly created reserves to buy something, so its assets go up too. Both sides grow together. That is QE in one sentence, and it is why our first figure shows the sheet ballooning on both sides at once.

### Bonds and yields, and why they move opposite

A **bond** is a loan you can buy. When a government needs to borrow, it sells bonds. A bond promises to pay the holder a fixed stream of cash: a regular *coupon* (an interest payment) and, at the end, the return of the *face value* (also called *par*, the original loan amount). If you own a \$1,000 bond with a 5% annual coupon, you receive \$50 every year, and at maturity you get your \$1,000 back.

Here is the idea that trips up almost everyone, and it is the engine of QE, so we go slowly. **A bond's price and its yield move in opposite directions.** When the price goes up, the yield goes down, and vice versa. To see *why*, you have to separate two numbers that beginners conflate: the coupon (fixed forever, printed on the bond) and the yield (what your money actually earns given the price you paid).

The bond above always pays \$50 a year. That \$50 never changes. But the *price* of the bond trades up and down in the market every day. Suppose you can buy that bond for \$1,000. You pay \$1,000, you get \$50 a year, so your yield is \$50 / \$1,000 = 5%. Now suppose demand for the bond surges and its price rises to \$1,100. The bond *still pays only \$50 a year* — that is fixed. But now you paid \$1,100 to get that \$50, so your yield is \$50 / \$1,100 = about 4.5%. The price went up; the yield went down. The fixed coupon is now spread over a larger purchase price, so each dollar you invested earns less. That inverse relationship is not a quirk; it is arithmetic, and it is the lever QE pulls.

A **yield** is just the return you earn on a bond expressed as a percent, given the price you paid for it. When you hear "the 10-year Treasury yield is 4%," that means a bond from the U.S. government maturing in ten years currently prices such that buyers earn about 4% a year. When commentators say "yields fell," they mean bond prices rose — the two are the same event seen from two sides.

### The yield curve and the term premium

Governments issue bonds at many different maturities: 3 months, 2 years, 10 years, 30 years. If you plot the yield of each maturity on a chart — short maturities on the left, long on the right — you get the **yield curve**, a line showing what interest rate the market demands for lending to the government over different lengths of time. The short end of the curve is controlled tightly by the central bank's main policy lever (more on that in a moment). The long end is set by the market, by millions of investors deciding what return they need to tie their money up for a decade or three.

Why would a 10-year bond usually yield *more* than a 3-month bond? Because lending for longer is riskier: more can go wrong over ten years (inflation could surge, the issuer's finances could deteriorate, you might need your cash back early), so investors demand extra compensation to hold the longer bond. That extra compensation, the bonus yield you get for bearing the risk of locking money up for a long time, is called the **term premium**. Keep this term close: QE's central trick is to *crush the term premium* — to buy so many long bonds that investors stop demanding much extra to hold duration, dragging long-term yields down.

### The policy rate and the zero lower bound

A central bank's normal, everyday tool is *not* QE. It is the short-term **policy rate** — in the U.S., the federal funds rate, the interest rate banks charge each other to borrow reserves overnight. By raising or lowering this one overnight rate, the central bank tugs the short end of the yield curve, and that pull ripples (imperfectly) out to the rest of the economy's borrowing costs. When the economy is too hot, the central bank raises the policy rate to slow borrowing; when the economy is weak, it cuts the rate to encourage borrowing. We cover this machinery in detail in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

But that lever has a floor. You cannot cut an interest rate much below zero. If banks were charged a steep negative rate, they would rather hold physical cash, which yields exactly 0%, than park reserves at a penalty. So the policy rate bottoms out around zero. This floor is the **zero lower bound** (often abbreviated ZLB): the point past which the central bank cannot stimulate the economy by cutting its normal rate any further, because the rate is already at (or near) zero.

And that is the trap QE was invented to escape. Suppose the policy rate is already at zero, the economy is still weak, and the central bank wants to deliver *more* stimulus. Its main lever is jammed against the floor. It cannot push the short rate lower, so it reaches for a different part of the curve entirely — the long end — using a different tool. That tool is quantitative easing.

### Two kinds of money: reserves versus deposits

There is one more distinction to nail down before we go further, because nearly every confused argument about QE comes from blurring it. There are *two different kinds of money* in a modern economy, and they live in two different places.

The first kind is **deposit money** — the balance in your checking account. This is the money you and businesses actually spend. When you pay for groceries, deposit money moves from your account to the shop's account. The total amount of deposit money in the economy is enormous (in the U.S., on the order of \$18 trillion), and almost all of it was created not by the central bank but by *commercial banks making loans*. When a bank approves a \$200,000 mortgage, it does not move \$200,000 from a vault; it simply types \$200,000 into the borrower's deposit account, creating new deposit money on the spot. That mechanism — banks creating spendable money by lending — is covered in detail in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).

The second kind is **reserves** — the bank money that lives in commercial banks' accounts *at the central bank*. Reserves are used only for one thing: settling payments *between banks*, and ultimately backstopping the deposit system. You cannot pay for groceries with reserves; the shop has no reserve account. Reserves are simply the plumbing the banks use among themselves, invisible to the public.

QE creates the *second* kind — reserves — not the first kind — deposits. This is the entire reason QE can be measured in trillions and still leave consumer prices flat: it floods the inter-bank plumbing with reserves, but it does not, by itself, put a single extra dollar into anyone's spendable checking account. Whether those reserves ever turn into spendable deposit money depends on whether banks then choose to lend — and in a damaged economy, they often do not. Keep these two kinds of money separate in your head and three-quarters of the mythology around QE dissolves.

## What QE actually is

Now we can state it precisely, with no hand-waving. Quantitative easing is the central bank **buying large quantities of longer-term bonds from banks and other private holders, paying for them with newly created reserves, in order to push down long-term interest rates and ease financial conditions when the short-term policy rate is already at the zero lower bound.**

Walk through the mechanics one transaction at a time. The figure below is the chain we are about to trace.

![The QE mechanism as a chain from bond purchase to higher asset prices](/imgs/blogs/quantitative-easing-explained-printing-money-2.png)

A bank (or a pension fund, or any private bondholder) owns a government bond. The central bank announces it will buy bonds. It contacts the market, agrees a price, and buys the bond. To pay, the central bank does not hand over cash or write a check drawn on someone else's account. It simply *credits the seller's reserve account* at the central bank with new reserves it created on the spot. The bond moves onto the central bank's asset side; the new reserves appear on its liability side. The seller's portfolio has changed: it used to hold a bond, now it holds reserves.

Notice what just happened to the broad money the public can spend: nothing, directly. The seller swapped one financial asset (a bond) for another (reserves). The seller did not get richer in net-worth terms — it traded a \$1,000 bond for \$1,000 of reserves. No new spending power was injected into the hands of households or shops. This is the heart of why QE is *not* "printing money for the public," and we will return to it repeatedly.

The central bank buys two main kinds of bonds in QE, and the choice is strategic. The first is plain government bonds — Treasuries, in the U.S. The second is **mortgage-backed securities** (MBS): bonds whose cash flows come from a pool of thousands of individual home mortgages bundled together, so that buying them directly subsidizes the cost of home lending. When the central bank wants to target housing specifically — as the Fed did when the mortgage market was frozen in 2008 — it buys MBS, which pulls mortgage rates down more directly than buying Treasuries would. When it just wants to lower the general level of long-term rates, it buys Treasuries. By 2022 the Fed held both in size: roughly \$5.7 trillion of Treasuries and \$2.7 trillion of MBS. The kind of bond it buys tells you which corner of the economy it is trying to cheapen.

#### Worked example: the Fed buys a \$1,000 bond from a bank

Let's do the balance-sheet entries explicitly, because seeing the numbers move is the whole point.

A commercial bank, call it Bank A, holds a U.S. Treasury bond with a face value of \$1,000 on the asset side of its own balance sheet. The Fed decides to buy it as part of QE.

Step 1 — Before the trade, Bank A's relevant assets look like this:

```
Bank A assets:
  Treasury bond ........ $1,000
  Reserves at the Fed .. $0
```

Step 2 — The Fed agrees to pay \$1,000 for the bond (we use par for simplicity). The Fed creates \$1,000 of brand-new reserves and credits Bank A's reserve account. The bond moves to the Fed.

Step 3 — After the trade, Bank A's assets look like this:

```
Bank A assets:
  Treasury bond ........ $0      (sold to the Fed)
  Reserves at the Fed .. $1,000  (newly created, credited by the Fed)
```

And the Fed's own balance sheet changed like this:

```
Fed assets:        Treasury bond +$1,000
Fed liabilities:   Reserves owed to Bank A +$1,000
```

Both sides of the Fed's sheet grew by \$1,000. Bank A's *total* assets did not change at all — it swapped a \$1,000 bond for \$1,000 of reserves. The intuition: QE is an asset swap, not a gift; it changes *what* the financial system holds (more reserves, fewer bonds), not how much it holds.

This is what people mean when they say QE "creates money." It creates *reserves* — bank money held at the central bank. It does not, by this act alone, create a single dollar in anyone's checking account.

## Why a central bank does it: reaching for the long end

If QE is just an asset swap that leaves total wealth unchanged, why bother? Because the swap is not neutral. It changes the *price of duration*, and that price matters enormously for the real economy.

Recall the trap: the policy rate is at zero, the economy is still weak, and the short end of the curve cannot be pushed lower. But most borrowing that actually matters to the economy is *long-term*. A 30-year mortgage is priced off long-term yields. A corporation issuing 10-year debt to build a factory borrows at long-term yields. A government financing infrastructure borrows long. The short overnight rate the central bank normally controls is almost irrelevant to those decisions. So even with the short rate pinned at zero, long-term rates can still be "too high" relative to what a weak economy needs.

QE attacks exactly that gap. By buying long-term bonds in enormous quantity, the central bank removes duration from the market. Investors who sell their long bonds to the central bank are left holding reserves earning almost nothing, and they go looking for yield elsewhere — bidding up the price of the remaining long bonds (pushing those yields down) and reaching into riskier assets. The net effect is to drag the *long* end of the curve lower even though the central bank never touched the short rate. That is the entire strategic purpose: when the short rate is stuck, push down the long rate directly.

It helps to break a long-term yield into its two parts, because QE targets the second one. A 10-year yield is, roughly, *the average short-term rate the market expects over the next ten years* plus *the term premium* — the extra compensation for locking money up that long. The central bank's normal policy rate only influences the first part (expected future short rates). QE works on *both*: through the signaling channel it lowers the expected-future-rates part (by convincing the market that rates will stay low for years), and through the portfolio-balance channel it directly compresses the term premium (by hoovering up the supply of long bonds, so investors stop demanding much extra to hold duration). Estimates of QE's effect vary, but a common finding is that the large U.S. programs lowered the 10-year Treasury yield by somewhere in the range of 0.5 to 1.0 percentage point versus where it would otherwise have been — a big move, achieved entirely without cutting the short rate, which was already zero.

There is also a quieter motive central banks rarely advertise but everyone understands: QE makes it cheaper for the *government* to borrow during a crisis. By holding yields down, the central bank lowers the interest the treasury pays on its new debt, which matters enormously when a government is running huge deficits to fight a recession or a pandemic. This is not the same as directly funding the government (we will be precise about that distinction later), but it is a real and intended side effect, and it is why critics worry that QE blurs the line between monetary and fiscal policy.

#### Worked example: a \$1,000 face, 5% coupon bond when yields fall from 5% to 4%

This is the most important piece of arithmetic in the post, because it shows precisely how QE lowering yields turns into a rising bond price — and why everyone who already owned bonds gets richer.

Take a bond with \$1,000 face value, a 5% coupon (so it pays \$50 a year), with, say, 10 years left to maturity. When the market yield for that kind of bond is 5%, the bond trades right at its face value of \$1,000, because the coupon rate equals the market yield. You pay \$1,000 and earn 5%; fair and square.

Now QE drives the market yield for 10-year bonds down to 4%. The bond still pays \$50 a year and still returns \$1,000 at the end — those are fixed and printed on the bond. But buyers now only *demand* a 4% return. To earn 4% (not 5%) on a bond paying a fixed \$50, you must pay *more than \$1,000* for it. How much more? You discount each future cash flow at 4% instead of 5% and add them up. Doing that calculation for a 10-year bond gives a price of roughly \$1,081.

Let's sanity-check the direction with the rough rule professionals use. A bond's price sensitivity to yield is captured by its *duration* — for a 10-year bond, duration is roughly 8 years. The rule of thumb: price change is approximately minus duration times the yield change. Here the yield fell by 1 percentage point (from 5% to 4%), so the price rises by roughly 8 × 1% = 8%. Eight percent of \$1,000 is \$80, taking the price to about \$1,080 — which matches our \$1,081 calculation almost exactly.

So a 1-percentage-point fall in yields handed the existing bondholder an instant capital gain of about \$80, or 8%, on every \$1,000 of bonds held. The intuition: QE lowers yields, and lower yields mechanically raise the price of every bond already outstanding, so QE is a direct wealth transfer to whoever held long bonds before it started.

## The three channels: how lower yields reach the real economy

QE does not move the economy by one route but by three at once. Naming them precisely is what separates someone who understands QE from someone who just repeats "money printer go brrr." The figure below shows all three fanning out from a single act of bond buying.

![Three transmission channels of QE fanning out into asset prices and cheaper borrowing](/imgs/blogs/quantitative-easing-explained-printing-money-3.png)

### Channel 1: portfolio balance

This is the channel we just walked through with the bond math. When the central bank buys up the supply of safe long-term bonds, the investors who sold them are left with reserves yielding almost nothing. They do not sit on that cash; their mandate is to earn returns. So they *rebalance their portfolios* toward whatever still offers yield: corporate bonds, equities, real estate, emerging-market debt. This buying pressure pushes up the prices of those riskier assets and pushes down *their* yields too. The mechanism is sometimes called "reaching for yield." The central bank, by removing safe assets, deliberately herds private money into riskier ones, lifting asset prices broadly. This is the channel most responsible for QE's effect on stock markets — and for the inequality critique we will get to.

### Channel 2: signaling

A central bank that has just committed to buying trillions of dollars of bonds over many months has, in effect, made a public promise: it does not intend to raise short-term rates any time soon, because doing so would undercut its own bond-buying program. The market reads QE as a credible signal that **rates will stay low for a long time** — "low for long." That expectation alone lowers long-term yields, because a 10-year yield is partly just an average of expected short-term rates over the next ten years. If everyone now believes short rates will stay near zero for years, the 10-year yield drops to reflect that, no further bond-buying required. The signal does part of the work.

### Channel 3: lower long-term yields feeding cheaper borrowing

The first two channels converge on a single observable outcome: long-term yields fall and the term premium shrinks. And because real-world borrowing costs are priced *off* those long-term yields, the cost of borrowing for households and businesses falls in lockstep. Mortgage rates, which track the 10-year Treasury yield closely, come down. Corporate borrowing costs come down. Cheaper borrowing, in theory, encourages more home-buying, more business investment, and more spending — which is the whole point of stimulus.

#### Worked example: a \$300,000 mortgage gets cheaper as the 10-year yield falls 1%

Let's make channel 3 concrete with the thing most people borrow the most for: a house.

Mortgage rates in the U.S. typically run a couple of percentage points above the 10-year Treasury yield. Suppose QE drags the 10-year yield down by 1 percentage point, and the 30-year mortgage rate falls right along with it — say from 6% to 5%.

Take a \$300,000, 30-year fixed-rate mortgage.

- At 6%, the monthly principal-and-interest payment is about \$1,799.
- At 5%, the monthly payment falls to about \$1,610.

That is a saving of roughly \$189 a month, or about \$2,270 a year, on the *same* \$300,000 loan. Over the full 30-year life of the mortgage, the difference in total interest paid is on the order of \$68,000. None of that came from anyone mailing the borrower cash. It came purely from QE pulling the long end of the yield curve down, which pulled mortgage rates down with it.

The intuition: QE's stimulus reaches you not as a deposit in your account but as a lower number on the loans you take out — and on a 30-year mortgage, a single percentage point is tens of thousands of dollars.

## What QE does NOT do

This is the section that separates understanding from mythology. Each of these is a thing QE is constantly, confidently said to do, and does not.

**QE does not put spendable money into the real economy by itself.** Reserves are bank money. Only banks hold them, and a bank cannot "spend" its reserves in the shops; it can only use them to settle payments with other banks or, if it chooses, to back new lending. The new money that ordinary people spend — deposits in checking accounts — is created when *commercial banks make loans*, not when the central bank creates reserves. (We unpack exactly how commercial banks create deposit money in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).) If banks do not lend, the reserves just sit at the central bank, and almost none of QE's money reaches the public's wallets. This is why QE can be huge and yet feel, on the ground, like nothing happened.

**QE does not guarantee inflation.** This is the big one, and the 2010s proved it. After 2008, the Fed, the ECB, and the Bank of England created trillions in reserves, and inflation in those years ran *below* central banks' 2% targets for most of the decade. Why? Because reserves are not spending power, and banks were not lending aggressively into a damaged, deleveraging economy. Money that does not get spent does not bid up prices. Inflation is too much *spending* chasing too few goods; QE creates reserves, not spending, so the link to inflation is indirect and conditional, not automatic.

**QE is not the government "printing money to spend."** When you hear "the government just prints money to pay its bills," that describes *fiscal* spending financed by money creation — a different thing. QE is a *monetary* operation: the central bank buys bonds *that already exist* in the secondary market, from private holders, not directly from the treasury. The government still has to issue and sell its bonds to investors at market prices first. QE can make that borrowing cheaper by lowering yields, but the central bank is not directly funding the government's spending. The legal separation matters: in most advanced economies the central bank is forbidden from buying bonds directly from the treasury, precisely to avoid the "print to spend" dynamic that historically ends in hyperinflation.

**Reserves "sitting idle" does not mean QE was useless.** A common jab is that QE created trillions in reserves that just "sit at the Fed doing nothing," so it accomplished nothing. But QE's main work — the portfolio-balance and signaling channels — operates through the *price of bonds and the level of long-term yields*, which changed regardless of whether the reserves were ever lent out. The reserves sitting idle is a feature, not a bug: it means the stimulus was delivered through asset prices and borrowing costs, not through a flood of bank lending that could have overheated things. Idle reserves are the visible by-product, not the mechanism.

## Quantitative tightening: running it backward

QE is reversible. The reverse is called **quantitative tightening**, or QT — the central bank shrinking its balance sheet after a period of QE. The two are mirror images, shown side by side below.

![QE easing versus QT tightening as mirror images of the balance sheet](/imgs/blogs/quantitative-easing-explained-printing-money-7.png)

There are two ways to do QT. The aggressive way is to *sell* bonds outright back into the market, which would push their prices down and yields up sharply. Central banks almost never do this, because dumping bonds can be disruptive. The gentle, standard way is *passive runoff*: when bonds the central bank holds mature, the treasury pays back the face value, and instead of using that cash to buy new bonds (which is what it does to keep the balance sheet flat), the central bank simply *lets the bonds roll off* — it does not replace them. As bonds mature and are not replaced, the central bank's assets shrink, and the reserves it created to buy them are extinguished. The balance sheet deflates from the bottom up.

QT pulls every QE channel in reverse: it drains reserves, nudges long-term yields up, and tightens financial conditions. It is, in effect, additional tightening layered on top of whatever the central bank is doing with its short-term policy rate.

#### Worked example: \$95 billion a month rolling off a \$9 trillion balance sheet

In 2022, with inflation surging, the Fed began QT by capping its monthly runoff at up to \$95 billion — about \$60 billion of Treasuries plus \$35 billion of mortgage-backed securities allowed to mature without replacement each month.

Let's see how slow a process that is on a \$9 trillion sheet.

- At \$95 billion per month, that is \$95B × 12 = \$1.14 trillion per year drained from the balance sheet.
- The balance sheet at peak was about \$9 trillion. To get back to the roughly \$4 trillion the Fed considered a plausible new floor, it would need to drain about \$5 trillion.
- At \$1.14 trillion a year, \$5 trillion takes roughly 4.4 years — and that is the *theoretical* pace, before any pause forced by market stress.

The intuition: QE can be deployed in a few panicked weeks, but QT is a multi-year crawl, because shrinking the balance sheet too fast risks draining reserves below the level banks need and seizing up money markets — exactly what nearly happened in late 2019 when an earlier, smaller round of runoff went too far.

#### Worked example: the same bond when yields rise from 4% back to 5%

The bond math runs both ways, and the reverse is what made 2022 the worst year for bonds in modern history. Take the same bond from earlier — \$1,000 face, 5% coupon, 10 years, currently priced at about \$1,081 because the market yield had fallen to 4% during QE. Now QT and rate hikes push the yield back up from 4% to 5%.

- The bond still pays \$50 a year and still returns \$1,000 at maturity — fixed.
- But buyers now demand 5%, not 4%. To earn 5% on a fixed \$50 stream, they will only pay about \$1,000 for the bond.
- The price falls from \$1,081 to \$1,000 — a loss of about \$81, or roughly 7.5%, on every bond held.

Apply the duration rule of thumb again: a 10-year bond's duration is about 8 years, the yield rose 1 percentage point, so the price falls by roughly 8 × 1% = 8%. The arithmetic matches. And remember, this loss is on what is supposed to be the *safest* asset there is — government bonds. In 2022, longer-dated Treasuries fell far more than this example, with the longest bonds losing well over 30% of their value as yields surged across the curve.

The intuition: rising yields are mechanically a loss for anyone already holding bonds, so the same lever that hands bondholders a windfall during QE hands them a brutal loss during QT — there is no free lunch, only a transfer that reverses.

## The distributional effects: who QE helps and who it hurts

Here is the uncomfortable part, and the one that has made QE politically radioactive. QE works *primarily by raising asset prices*. That is not an accident or a side effect; the portfolio-balance channel is *designed* to lift the price of stocks, bonds, and property. But the benefits of higher asset prices flow to whoever owns those assets — and asset ownership is extremely concentrated. So QE, whatever its macro merits, mechanically tilts toward the already-wealthy. The matrix below maps the winners and losers.

![Matrix of QE winners and losers across asset owners, borrowers, savers, and pensions](/imgs/blogs/quantitative-easing-explained-printing-money-6.png)

The big winners are **asset owners**. If you held a diversified stock portfolio and a house through the 2010s, QE was the wind at your back: it pushed valuations up year after year. The winners also include **borrowers** with long-term debt — homebuyers refinancing into cheaper mortgages, corporations issuing cheap bonds.

The losers are **cash savers** and the institutions that behave like them. If your wealth was in a savings account or short-term deposits, QE pinned the interest you earned near zero, so your money earned almost nothing for a decade while asset prices ran away from you. And **pension funds** and insurers, which must earn a certain return to fund the promises they made to retirees, were squeezed hard: when safe long-term yields collapse, the income they rely on collapses too, widening their funding gaps.

#### Worked example: \$10,000 in stocks during QE versus a \$10,000 saver earning ~0%

Let's quantify the gap that makes the inequality critique bite.

Two people each start with \$10,000 at the beginning of a QE decade.

- Person A puts it in a broad stock index. Suppose, helped along by QE lifting valuations, the index returns about 12% a year for ten years. Compounding, \$10,000 × (1.12)^10 ≈ \$31,100. Person A's \$10,000 nearly tripled.
- Person B, cautious, leaves it in a savings account. With the policy rate pinned near zero, the account pays roughly 0.1% a year. After ten years: \$10,000 × (1.001)^10 ≈ \$10,100. Person B earned about \$100 in a *decade*.

After ten years, Person A has about \$31,100 and Person B has about \$10,100 — a gap of roughly \$21,000 created largely by which side of the asset line each happened to be standing on when QE began. Worse, if inflation averaged even 2% a year over that decade, Person B's \$10,100 actually *lost* purchasing power: it buys less than the original \$10,000 did.

The intuition: QE rewards the owners of assets and penalizes the holders of cash, so its single largest real-world consequence is that it widens the wealth gap between people who already own things and people who do not.

This is the core critique of QE, voiced from both the left (it inflated the wealth of the rich) and parts of the right (it punished prudent savers and distorted markets). Defenders counter that the alternative — letting a depression run its course with no stimulus — would have hurt the poor and the jobless far more than it hurt savers, and that the job losses prevented were real. Both things can be true: QE may have been the least-bad available tool *and* deeply regressive in its distribution. Honest analysis holds both.

## Common misconceptions

**"QE causes hyperinflation."** The most persistent prediction about QE, and the one most thoroughly falsified by the evidence. After the 2008 QE programs, the U.S., U.K., eurozone, and Japan all ran *below-target* inflation for most of the following decade, despite trillions in reserves. Inflation is excess *spending* chasing goods; QE creates reserves, which are not spending power until banks lend and the public spends. The 2021–2022 inflation that did arrive came mostly from pandemic supply shocks and enormous *fiscal* (government cash) transfers landing directly in households' pockets — not from the reserve creation of QE per se. QE *can* contribute to inflation if it is sustained while the economy is already at full capacity, but "QE therefore hyperinflation" is simply wrong as a rule.

**"The Fed gives money to people during QE."** No money reaches the public directly. QE buys bonds from banks and other financial institutions and pays them in reserves. The public is affected only *indirectly*, through lower borrowing rates and higher asset prices. The thing people are often imagining — money deposited directly into citizens' accounts — is called "helicopter money" or a direct fiscal transfer (like the COVID stimulus checks), and that is a *fiscal* policy run by the government, not QE.

**"QE directly funds the government's deficit."** The central bank buys government bonds in the *secondary* market — from investors who already own them — not directly from the treasury at issuance. The government must still sell its new bonds to private buyers at market prices. QE makes that borrowing cheaper by holding yields down, which is a real and important effect, but it is not the same as the central bank handing the treasury freshly printed money to spend. The legal firewall against *direct* monetary financing of deficits exists precisely because crossing it has historically led to currency collapses.

**"All those reserves are sitting idle, so QE did nothing."** The reserves *do* largely sit at the central bank — but QE's main effects flowed through bond prices and long-term yields, which moved regardless. Idle reserves are evidence that the stimulus came through the *price* channel (cheaper borrowing, higher asset prices) rather than a lending boom, not evidence that nothing happened. The 8% jump in bond prices and the percentage point off mortgage rates from our worked examples were entirely real.

**"QE is the same as cutting interest rates."** Related but different. Cutting the policy rate works on the *short* end of the curve and is the normal tool. QE is what the central bank reaches for *after* the short rate hits zero and can go no lower; it works on the *long* end by buying duration. They are two distinct levers on two distinct parts of the yield curve, and QE exists specifically to operate when the rate lever is jammed against its floor.

**"QT will quickly undo QE."** QT is real but glacial. As the runoff arithmetic showed, draining a multi-trillion-dollar balance sheet at \$95 billion a month takes years, and central banks pause the moment money markets show stress. The balance sheet may never return to its pre-2008 size; the "new normal" floor is far higher than the old one, because the banking system now wants to hold far more reserves than it did in 2007.

**"QE means the central bank made a profit, so it was free."** During the low-rate years, QE actually *earned* the central bank money: it held trillions of bonds paying interest while paying almost nothing on the reserves it had created, and it remitted those profits to the treasury. But that flipped violently when rates rose. By 2023, central banks including the Fed were paying *more* interest on their reserves than they earned on their older, low-yielding bonds, swinging to large operating losses — tens of billions of dollars a year. So QE was never free; it was a bet that funded itself while rates were low and turned costly once rates rose. The taxpayer ultimately stands behind the central bank's balance sheet, which is one more reason QE is a fiscal question dressed as a monetary one.

## How it shows up in real markets

QE is not a textbook abstraction; it is the defining monetary story of the last two decades. Here are the episodes that matter, with the mechanism from this post visible in each. The timeline below traces the U.S. arc.

![Timeline of the Fed balance sheet from 2007 to 2022 through QE and QT](/imgs/blogs/quantitative-easing-explained-printing-money-5.png)

### Japan and the Bank of Japan, 2001: the original QE

Japan invented modern QE. After its asset bubble burst in 1990, Japan fell into a long deflationary slump, and by the late 1990s the Bank of Japan had already cut its policy rate to essentially zero — the first major economy to hit the zero lower bound in the modern era. With the rate lever jammed, in March 2001 the BOJ began explicitly targeting the *quantity* of reserves in the banking system rather than a price (an interest rate), buying government bonds to flood banks with reserves. That is QE, and the BOJ coined the framing. The lesson the rest of the world drew, and largely got wrong at first, was that QE seemed not to do much in Japan — inflation stayed near zero. The more careful lesson, learned later, was that QE in a balance-sheet recession where banks and firms are all trying to pay down debt is *pushing on a string* through the lending channel, even as it holds yields down. Japan would go on to push QE to its global extreme, eventually owning a large share of its entire government bond market and even buying equity ETFs directly — a reminder of how far the tool can be stretched.

### The Fed's QE1, 2008–2009: stopping the freefall

When the 2008 financial crisis hit and the Fed cut the federal funds rate to zero by December 2008, it ran out of conventional room with the economy still in freefall. So in late 2008 it launched its first large-scale asset purchase program, later called QE1, buying mortgage-backed securities (bonds whose payments come from pools of home mortgages) and Treasuries. The mortgage-bond buying was targeted: the mortgage market had seized, and by becoming a giant buyer the Fed pulled mortgage rates down and thawed home lending. QE1 took the balance sheet from about \$0.9 trillion to over \$2 trillion in a matter of months. The mechanism was textbook portfolio balance and channel 3: long yields fell, mortgage rates fell, and the freefall in asset prices was arrested. The lesson: QE's first and most defensible use is as a *crisis backstop*, restoring function to frozen markets, not as routine stimulus.

### QE2 and QE3, 2010–2014: stimulus by the trillion

With the acute crisis past but recovery weak and unemployment stuck near 10%, the Fed turned QE from a rescue tool into an ongoing stimulus tool. QE2, announced in November 2010, was a defined \$600 billion of Treasury purchases aimed squarely at pushing long-term yields lower. QE3, launched in September 2012, was open-ended — \$40 billion of mortgage bonds plus \$45 billion of Treasuries *per month*, "for as long as it takes," explicitly tied to improvement in the labor market. By the time purchases ended in late 2014, the balance sheet had reached about \$4.5 trillion. This is the era that produced the inequality critique: years of rising stock and house prices, a roaring bull market, and savers earning nothing. It is also the era that falsified the hyperinflation predictions — inflation stayed *below* the Fed's 2% target through almost all of it. Both the asset-price lift and the missing inflation are exactly what the channels in this post predict.

### The 2013 taper tantrum: the signaling channel in reverse

In May 2013, Fed Chair Ben Bernanke merely *mentioned* that the Fed might begin to slow ("taper") the pace of its QE3 purchases later in the year. He did not raise rates. He did not sell a single bond. He simply signaled that the flow of new buying might shrink. The market reaction was violent: the 10-year Treasury yield jumped from about 1.6% to nearly 3% over a few months, mortgage rates spiked, and emerging-market currencies and bonds sold off hard as the "reach for yield" money that QE had pushed abroad rushed back. This episode — the "taper tantrum" — is the cleanest real-world proof that QE's *signaling channel* is powerful and that *expectations* about future purchases, not just the purchases themselves, move yields. The lesson central banks took away was to communicate any wind-down with extreme, gradual, telegraphed care.

### COVID, March 2020: unlimited QE

When the pandemic hit in March 2020, financial markets seized in a "dash for cash" so severe that even the U.S. Treasury market — the deepest, safest market on earth — stopped functioning normally as everyone tried to sell at once. The Fed responded with the most aggressive QE in history. On March 23, 2020, it announced *unlimited* QE: it would buy Treasuries and mortgage bonds "in the amounts needed" to restore market function. It then bought at a staggering pace, taking the balance sheet from about \$4 trillion to roughly \$9 trillion within about two years. The immediate purpose was the original one — backstopping a frozen market — and it worked within days. But the scale, combined with massive *fiscal* stimulus (the trillions in direct payments and business support that, unlike QE, *did* land in people's pockets), set up the next episode.

### 2021–2022: inflation arrives, and QT begins

After a decade of QE without inflation, 2021 finally delivered it: U.S. inflation peaked above 9% in mid-2022, the highest in roughly forty years. The dominant causes were pandemic supply-chain breakage and the enormous *fiscal* transfers that put spending power directly into households — but the sheer scale of monetary accommodation, including QE, was part of the backdrop. The Fed reversed hard. It stopped buying in March 2022, began raising the policy rate at the fastest pace in decades, and in June 2022 started QT, capping runoff at up to \$95 billion a month. This is the full mirror image of everything above: yields rose sharply, bond prices fell (delivering 2022's brutal bond-market losses, the worst in modern history, as our bond math predicts in reverse), mortgage rates roughly doubled, and the asset-price party of the QE era partly unwound. The lesson, still being absorbed: QE deployed during a genuine crisis backstop is one thing; QE sustained into a recovery, alongside large fiscal stimulus, is part of how you eventually get inflation.

### The UK gilt crisis, September 2022: QE restarted mid-QT

A vivid illustration of how powerful and how double-edged the tool is came from Britain. In September 2022, the Bank of England was in the middle of *tightening* — it had ended QE and was about to start QT. Then a new UK government announced a large, unfunded package of tax cuts. Investors panicked about the resulting government borrowing, and UK government bonds (called *gilts*) sold off violently; the 30-year gilt yield spiked more than a full percentage point in days. That sudden price collapse triggered a doom loop in UK pension funds, which had used a strategy called "liability-driven investment" that required them to post more collateral as gilt prices fell. To raise that collateral they had to sell gilts — which pushed prices down further, demanding still more collateral. A self-reinforcing fire sale threatened to take down the pension system within hours.

The Bank of England's response is the instructive part: it *restarted QE* — announcing emergency, temporary, unlimited bond-buying to put a floor under gilt prices — even as it remained committed to tightening overall. The intervention worked within days and was wound down within weeks. The lesson is twofold. First, QE's market-backstop function is so powerful that a central bank will reach for it instantly when a core market threatens to seize, regardless of its broader stance. Second, the very pension funds that QE had squeezed for a decade (by holding yields down) were the ones whose strategies blew up when yields finally rose — a reminder that years of suppressed yields build hidden fragilities that only surface when the policy reverses.

### The composition of the peak balance sheet

It is worth pausing on *what* the Fed actually ended up holding at the 2022 peak, because it reveals the targeting. The stack below shows it.

![Composition of the Fed balance sheet at peak with Treasuries, MBS, reserves, and currency](/imgs/blogs/quantitative-easing-explained-printing-money-4.png)

Of the roughly \$9 trillion, about \$5.7 trillion was U.S. Treasuries and about \$2.7 trillion was mortgage-backed securities — that large MBS holding is the residue of deliberately targeting the housing market in QE1 and QE3 to pull mortgage rates down. On the funding side, about \$3.3 trillion was bank reserves and about \$2.3 trillion was physical currency in circulation. (These are illustrative, as-of 2022; the exact weekly figures are in the Fed's H.4.1 release.) The picture confirms the thesis of the whole post: the Fed's expansion was overwhelmingly about *buying bonds and creating reserves*, with physical cash a relatively stable, minority share. The "money printer" never printed much actual money; it bought a mountain of bonds and conjured a mountain of reserves to pay for them.

## When this matters to you and where to read next

You do not work at the Fed, so why should you care about the size of a central bank's balance sheet? Because QE quietly sets the backdrop for nearly every financial decision you make.

When QE is running, the message to your money is: *cash is trash, take risk*. Savings accounts pay nothing, so the rational response — and the one QE is explicitly engineered to provoke — is to push into stocks, bonds, and property, bidding their prices up. If you bought a home or invested in the market during a QE era, the policy was tailwind at your back, whether you knew it or not. When QT runs, the message reverses: cash finally pays something, borrowing gets more expensive, and the valuations QE inflated come under pressure. The brutal bond and stock losses of 2022 were, in large part, QE's tailwind turning into QT's headwind.

So the practical takeaways are these. First, watch the *direction* of the balance sheet and the long-term yield, not just the headline policy rate — the long end is where QE lives and where your mortgage is priced. Second, understand that "the Fed is printing money" almost never means what the phrase implies; it means the central bank is buying bonds and creating reserves, which lowers yields and lifts asset prices, with the effect on consumer inflation being conditional and lagged, not automatic. Third, recognize the distributional reality: QE is a powerful but blunt tool that reliably helps people who already own assets more than it helps everyone else, which is why it is as much a political question as an economic one.

To go deeper, read the companion pieces in this series. [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) explains the conventional short-rate lever that QE supplements once that lever hits zero. [How money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) explains the crucial difference between the reserves QE creates and the deposit money you actually spend — the distinction that resolves the inflation paradox. And [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) places the central bank inside the wider machine of savers, intermediaries, and markets, so you can see exactly where QE's effects enter the system and where they get stuck. None of this is advice on what to buy; it is the map you need to read the financial weather for yourself.

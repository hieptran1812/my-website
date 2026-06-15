---
title: "Treasury Issuance: How Bills, Coupons, and the Refunding Move Liquidity"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How the US Treasury's choice of how much to borrow — and whether in short bills or long coupons — quietly moves market liquidity, yields, and the reverse-repo cash pile, and how to read the Quarterly Refunding Announcement like a trader."
tags: ["macro", "treasury-issuance", "liquidity", "bills-vs-coupons", "quarterly-refunding", "reverse-repo", "bank-reserves", "tga", "net-liquidity", "rates"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The US Treasury borrows trillions every year, and the *form* it borrows in — short **bills** versus longer **coupons** — decides whether that borrowing merely shuffles idle cash around or actually drains money out of the markets you trade.
>
> - A budget deficit forces issuance: the Treasury spends more than it taxes, so it sells new debt at auction to cover the gap (roughly \$1.8 trillion a year in recent years).
> - **Bills** (debt under one year) tend to get bought by money funds paying out of the reverse-repo cash pile — that is **liquidity-neutral**, because cash just moves from one parking spot to another.
> - **Coupons** (notes and bonds, 2 to 30 years) tend to get bought by banks and bond funds paying out of **bank reserves** — that is **liquidity-negative**, because real money leaves the system that greases markets.
> - The **Quarterly Refunding Announcement (QRA)** is the calendar event that tells you the borrowing total and the bill-versus-coupon mix in advance. In 2023, the market panicked about a bill flood that turned out to be liquidity-neutral — and the lesson is the whole point of this post.
> - The one number to remember: in late 2022 the reverse-repo "shock absorber" peaked at **\$2.55 trillion**, and the 2023–24 bill flood drained it down to about **\$0.16 trillion** by December 2024 — without ever seriously denting the reserves that matter for funding markets.

In late July and early August of 2023, a lot of professional investors got scared of the same thing at the same time, and they got it wrong.

The story they told themselves went like this. The US government was running enormous deficits. The debt-ceiling standoff earlier that year had frozen new borrowing for months, and now that a deal was done, the Treasury had to catch up — it needed to issue something like a trillion dollars of new debt in a single quarter to refill its bank account and fund the government. A trillion dollars of fresh bonds has to be paid for with a trillion dollars of cash. Surely, the thinking went, that much cash being yanked out of markets and handed to the government would crush stocks, blow out yields, and tighten financial conditions hard. The bond market did wobble — the 10-year Treasury yield climbed from around 3.96% in late July toward 4.88% by October. Headlines screamed about a "refunding tantrum."

But the thing people feared most — a brutal drain of the cash that banks and markets run on — mostly did not happen. The reason is the single most important idea in this entire post: **not all Treasury issuance drains the same kind of money.** The Treasury issued that wall of new debt heavily in the form of short-term *bills*, and bills got soaked up by money market funds that paid for them by pulling cash out of an idle parking lot at the Federal Reserve called the reverse-repo facility. That cash had been sitting there earning interest, doing nothing for markets. When it moved into bills, the broad pool of *bank reserves* — the money that actually greases lending, repo, and risk-taking — barely moved. The "drain" everyone feared mostly hit a pile of cash that was already inert.

![Treasury issuance flow diagram from budget deficit to bills and coupons to who buys and the TGA to the liquidity effect](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-1.png)

That distinction — between draining inert cash and draining live money — is what separates traders who panic at every "wall of supply" headline from traders who can tell when a refunding actually matters. It is also a distinction almost nobody outside a rates desk makes, which is precisely why it is worth your time: when the consensus treats all issuance as identically scary, the trader who can sort it into "drains a parking lot" versus "drains the bloodstream" has a real, repeatable edge around four scheduled events a year. This post builds the whole machine from the ground up, assuming you have never bought a Treasury bond in your life. We will define what a bill is and what a coupon is, why a deficit forces the government to issue anything at all, who actually buys this debt, where the cash goes, and exactly why the bill-versus-coupon choice changes the effect on the liquidity that drives asset prices. Then we will turn it into a playbook: how to read the Quarterly Refunding Announcement, what to watch on the days around it, and when a supply scare is a fade versus when it is a real warning.

## Foundations: bills, coupons, the deficit, the QRA, and who buys

Let us define every moving part from zero, because the entire argument depends on getting the plumbing right.

### What the Treasury is doing, and why

The US federal government, like any household, has income and expenses. Its income is mostly taxes. Its expenses are everything from defense to Social Security to interest on past debt. When expenses exceed income in a given period, that shortfall is the **budget deficit**. In recent years the federal deficit has run on the order of \$1.8 trillion per year — meaning the government spends roughly \$1.8 trillion more than it collects.

A household that spends more than it earns has to borrow the difference. So does the government. The way the US Treasury borrows is by selling **Treasury securities** — IOUs that promise to pay the holder back, with interest. It sells them through public **auctions**: it announces it will sell, say, \$50 billion of a particular security on a particular date, investors submit bids, and the securities go to the winning bidders, who pay cash. That cash funds the government. The IOUs are the national debt.

There is one more wrinkle that surprises beginners: the Treasury must issue debt not only to cover the *new* deficit, but also to **roll over** old debt that is maturing. A bond issued three years ago that comes due today has to be paid off — and the Treasury pays it off by selling a brand-new bond to raise the cash. So the gross amount of issuance every quarter is much larger than the deficit alone. This rolling-over of maturing debt is exactly why there is a regular, scheduled event called the *refunding* — the Treasury is perpetually refunding (re-financing) its maturing obligations plus funding the new gap.

It helps to keep the two numbers distinct. *Net* issuance is the new debt the government adds — roughly the deficit plus any change in its cash balance. *Gross* issuance is net issuance plus all the maturing debt that must be replaced. Because the US carries a large stock of short bills that mature every few weeks, gross issuance runs into the tens of trillions per year even though net issuance is a fraction of that. When a headline shouts "the Treasury will auction \$7 trillion this quarter," it is almost always a gross number inflated by routine bill roll-overs — most of that cash is recycled from investors whose old bills just matured, not pulled fresh out of the market. This is the first reason "wall of supply" headlines mislead: the scary number usually counts money that is already circling in the front end. What you want is the *net* new cash need and, within it, the bill-versus-coupon split — the two numbers that actually move liquidity.

### Bills versus notes versus bonds — the maturity ladder

All Treasury securities are IOUs, but they differ by how long until they pay you back. That length of time is the **maturity**. The Treasury issues across a whole ladder of maturities, and the names matter:

- **Treasury bills** ("bills" or "T-bills") mature in **one year or less** — there are 4-week, 8-week, 13-week, 17-week, 26-week, and 52-week bills. Bills pay no periodic interest. Instead they are sold at a discount and redeemed at face value: you pay \$98 and get \$100 back in three months, and that \$2 is your interest. Because they are so short and so safe, bills behave almost like cash that pays a yield. This "cash-like" quality is central to everything below.
- **Treasury notes** mature in **2 to 10 years** — the 2-year, 3-year, 5-year, 7-year, and 10-year. Notes pay a fixed **coupon**: a set interest payment every six months, plus the face value back at maturity. The word "coupon" is historical — old paper bonds had detachable coupons you clipped and mailed in to collect interest.
- **Treasury bonds** mature in **20 or 30 years**. Same structure as notes — semiannual coupons plus principal at the end — just longer.

In market shorthand, everything that pays a periodic coupon — notes and bonds, anything 2 years and out — gets lumped together as **"coupons."** So the great divide of this post is **bills (under a year, cash-like, no coupon) versus coupons (2 to 30 years, pay interest, carry duration).** Hold that divide in your head; we are about to hang the entire liquidity argument on it.

One more term: **duration.** Duration measures how sensitive a bond's price is to a change in interest rates. A 30-year bond has high duration — if rates rise a little, its price falls a lot, because you are locked into the old rate for three decades. A 3-month bill has almost no duration — rates can move and the bill matures so soon that its price barely budges. This is why coupons are "risky" to hold and bills are "safe": the longer you lend, the more interest-rate risk you take. We will see that this is exactly why *different buyers* want bills versus coupons, which is why the two drain different pools of money.

There is a second reason the Treasury cares about the mix, beyond liquidity: **cost and rollover risk.** Bills are cheap to issue when the curve is normal, but they mature fast, so funding heavily in bills means coming back to market constantly — more frequent auctions, more exposure to a sudden jump in short rates. Coupons lock in a known rate for years, which is safer for the issuer but means paying a term premium (the extra yield investors demand to lend long). A formal advisory body, the **Treasury Borrowing Advisory Committee (TBAC)**, recommends keeping bills at a "prudent" share of the total — historically a rough guideline of around 15–20% of marketable debt. When the Treasury leans heavily on bills to fund a surge (as in 2023), it is borrowing short to avoid flooding the long end with duration, accepting more rollover risk in exchange for not pushing long yields up. That trade-off — bills to spare the long end versus coupons to lock in funding — is the lever behind every QRA, and it is why the mix is a *choice* the Treasury makes quarter by quarter rather than a fixed formula.

### The QRA — the Treasury's quarterly press release

Four times a year, the Treasury tells the market how it plans to borrow over the coming quarter. This is the **Quarterly Refunding Announcement (QRA)**, sometimes just called "the refunding." It is released at 8:30 a.m. Eastern on the first Wednesday of February, May, August, and November (with a related "borrowing estimate" released the Monday before). It is, for rates traders, one of the most-watched scheduled events on the calendar — on par with a Fed meeting or a payrolls report.

![Anatomy of a Quarterly Refunding Announcement showing what it contains and what it moves](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-5.png)

The QRA contains three things that move markets:

1. **The total borrowing need** for the quarter — how much net new cash the Treasury must raise (driven by the deficit, the roll, and how much it wants to rebuild its cash balance).
2. **The bill-versus-coupon mix** — how much of that borrowing will be done in short bills versus longer coupons. This is the number that decides the *liquidity* effect, and it is the number this post cares about most.
3. **The auction sizes per tenor** — exactly how big each upcoming 2-year, 5-year, 10-year, and 30-year sale will be. A surprise increase in long-end auction sizes means more duration the market must absorb, which can push long yields up.

We will return to the QRA as a tradeable event later. For now, just file away that the bill-versus-coupon mix is a published, scheduled number you can read in advance — you are not guessing.

### Who actually buys Treasuries — and why it matters

Here is where the liquidity story turns. Treasuries are not bought by one undifferentiated blob of "the market." They are bought by distinct types of investors who hold *different kinds of cash* and pay out of *different pools*. Three groups matter most:

- **Money market funds (MMFs).** These are funds that hold cash for individuals and corporations and invest only in ultra-safe, ultra-short instruments. They love bills — short, safe, cash-like, exactly their mandate. Critically, in 2021–2023 money funds parked huge amounts of spare cash in the Fed's **overnight reverse repo facility (ON RRP)**, which we will define in a moment. When bills offer a better yield, money funds pull cash *out of the RRP* and buy bills. That is the bill channel.
- **Banks and bond funds.** Banks, pension funds, insurers, and bond mutual funds want **duration** — they need longer maturities to match long-term liabilities or to express a view on rates. They buy coupons. When a bank buys a coupon bond, it pays out of its **reserves** — its account balance at the Fed. That is the coupon channel.
- **Foreign buyers.** Foreign central banks, sovereign funds, and overseas investors hold a large slice of US debt. They matter enormously for the *price* of Treasuries (a foreign buyers' strike can push yields up), but for the domestic *liquidity* plumbing we are tracing, their effect is more diffuse, so we will mostly set them aside and focus on the two domestic channels — RRP cash versus bank reserves.

The punchline is already visible: **bills get paid for out of the RRP cash pile; coupons get paid for out of bank reserves.** And those two pools of money are not equally important to markets. One is a parking lot for idle cash. The other is the lifeblood of the banking and funding system. Drain the first and markets barely notice; drain the second and funding tightens, repo rates jump, and risk assets feel it. Everything else is detail on top of that one sentence.

### Three accounts at the Fed you must know

To follow the cash, you need to know three balances that all sit on the Federal Reserve's balance sheet. The Fed is the government's bank and the banks' bank, so the key accounts live there.

**Bank reserves.** When a commercial bank has money, a chunk of it sits in an account at the Fed called its reserve balance. Reserves are the ultimate settlement money of the banking system — banks pay each other in reserves, and reserves back the whole apparatus of lending, repo, and market-making. When traders say "liquidity," reserves are usually the thing they ultimately mean. As of the data we will use, bank reserves were around \$3.0 to \$3.5 trillion (roughly \$3.03 trillion at end-2022, \$3.49 trillion at end-2023, \$3.27 trillion at end-2024). When reserves get scarce, funding markets seize up — that is the lesson of September 2019's repo spike, covered in [money-market plumbing](/blog/trading/macro-trading/money-market-plumbing-repo-collateral-sofr).

**The overnight reverse repo facility (ON RRP).** This is a facility where money market funds (and a few others) can lend cash to the Fed overnight, fully collateralized, and earn a set interest rate. In plain terms, it is a **parking lot** where money funds stash spare cash when they have nowhere better to put it. The cash in the RRP is *inert* — it is sitting at the Fed, not circulating in markets, not funding anyone. The RRP balance ballooned from near zero in early 2021 to a peak of **\$2.55 trillion in December 2022**, because the financial system was awash in cash with nowhere to go. That \$2.55 trillion is the "shock absorber" the whole story hinges on.

![ON RRP overnight reverse repo balance over time from 2.55 trillion peak down toward near zero](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-4.png)

Look at the shape of that chart. The RRP filled up to \$2.55 trillion at the end of 2022, then drained relentlessly: \$1.95 trillion by mid-2023, \$1.02 trillion by the end of 2023, \$0.66 trillion by mid-2024, and down to about **\$0.16 trillion by December 2024**, eventually approaching empty (\$0.01 trillion) by October 2025. That long slide is not random — it is largely the story of money funds emptying the parking lot to buy the bills the Treasury was flooding into the market. The bill flood had a built-in sponge.

**The Treasury General Account (TGA).** This is the **Treasury's own checking account at the Fed.** When the Treasury collects taxes or sells debt, the cash lands in the TGA. When it spends — pays a contractor, sends a Social Security check, pays interest — money flows out of the TGA into the economy. The TGA is the central node in our story, because *issuance proceeds land in the TGA*, and a rising TGA means cash has left markets and is sitting, idle, in the government's account. We will spend the next section on it.

## The TGA fills with proceeds — and that drains liquidity

Start with the simplest mechanical fact: when the Treasury sells a bond, the buyer pays cash, and that cash lands in the TGA. The TGA balance goes *up*. And here is the key accounting identity — because the TGA and bank reserves are both liabilities on the Fed's balance sheet, and the Fed's total balance sheet size is roughly fixed in the short run, **a dollar that moves into the TGA is a dollar that comes out of the rest of the system.** When the government's checking account fills up, money has been pulled out of private hands and frozen at the Fed until the government spends it back out.

Think of the Fed's balance sheet as a fixed-size pie. On the liability side, the big slices are bank reserves, the RRP, and the TGA. If the pie stays the same size and the TGA slice grows, one of the other slices must shrink. So when the Treasury raises cash and the TGA swells, either bank reserves shrink or the RRP shrinks. *Which one shrinks* is the entire ballgame — and it depends on who bought the debt, which depends on whether it was bills or coupons.

![TGA Treasury General Account balance over time with debt-ceiling floor and refunding rebuild marked](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-2.png)

The TGA chart shows how violently this account swings. In April 2020, at the height of the pandemic borrowing spree, the TGA ballooned to \$1.50 trillion and then \$1.80 trillion by July 2020 — the Treasury raised a war chest. Then look at June 2023: the TGA had been drawn all the way down to about **\$0.05 trillion** — \$50 billion, a rounding error for the US government. That near-zero point was not normal; it was the debt-ceiling standoff. By law the Treasury could not issue net new debt while the ceiling was binding, so to keep paying the bills it ran its checking account down to almost nothing. The moment the ceiling was lifted in early June 2023, the Treasury had to do two things at once: fund the ongoing deficit *and* rebuild its dangerously empty checking account. That double need is exactly what created the "wall of issuance" the market feared in the summer of 2023.

The rebuild from June to September 2023 took the TGA from \$0.05 trillion back up to about \$0.68 trillion — a swing of roughly \$630 billion drained out of markets and into the government's account in a single quarter. (The Treasury's own target at the time was around \$0.75 trillion, which it reached shortly after.) On paper, draining \$600–700 billion of cash in three months sounds catastrophic for liquidity. The reason it was not catastrophic is, again, *where* that cash came from — and that is the bill-versus-coupon mechanism we turn to now.

#### Worked example: a TGA rebuild from \$0.05T to \$0.75T drains \~\$700B

Walk the numbers. After the debt-ceiling deal in June 2023, the TGA sat at roughly \$0.05 trillion (\$50 billion). The Treasury's stated goal was to rebuild it toward \$0.75 trillion (\$750 billion). To get there, it had to raise the difference plus whatever it needed for the ongoing deficit and maturing-debt roll.

- Starting TGA: \$0.05 trillion.
- Target TGA: \$0.75 trillion.
- Net rebuild required: \$0.75T − \$0.05T = **\$0.70 trillion** of cash pulled into the account.

That \$0.70 trillion is a real liquidity drain *in the accounting sense* — \$700 billion left private hands and sat in the TGA. If that \$700 billion had been paid entirely out of **bank reserves**, reserves would have fallen by \$700 billion, and with reserves around \$3.0 trillion at the time, that is a 23% hit to the lifeblood of the funding system — genuinely scary. But because the Treasury raised most of it through **bills** bought by money funds emptying the RRP, the actual hit to reserves was far smaller; most of the \$700 billion came out of the \$1.9 trillion RRP parking lot instead. The intuition: a TGA rebuild is always a drain on paper, but the buffer it drains from determines whether markets feel it.

## Bills drain the RRP (neutral); coupons drain reserves (negative)

Now we put the two channels side by side, because this is the crux of the whole post. Same dollar of borrowing, opposite liquidity effect, depending purely on the instrument.

![Bills versus coupons two liquidity paths with money funds and RRP on one side and banks and reserves on the other](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-3.png)

### The bill channel: liquidity-neutral

When the Treasury issues a **bill**, the natural buyer is a money market fund. Bills are short, safe, and cash-like — exactly what an MMF holds. Where does the MMF get the cash to buy the bill? In the 2022–2024 environment, it got it by pulling cash *out of the RRP*. The money fund had been lending that cash to the Fed overnight at the RRP rate; now it instead lends it to the Treasury by buying a bill, because the bill yields a touch more.

Trace the cash:

1. Money fund pulls \$1 out of the RRP (RRP balance falls by \$1).
2. Money fund pays that \$1 to the Treasury for a bill.
3. The \$1 lands in the TGA (TGA balance rises by \$1).

Net effect on the rest of the system? The RRP fell by \$1 and the TGA rose by \$1. **Bank reserves did not change.** The cash simply moved from one parking lot at the Fed (RRP) to another parking lot at the Fed (TGA). Both were inert before and after. The money that markets actually run on — reserves — never moved. This is what "liquidity-neutral" means: the issuance happened, the debt got sold, the TGA filled, and the funding markets barely felt a thing because the drain hit a buffer of idle cash, not live reserves.

This is precisely why the 2023 H2 bill flood was a non-event for funding markets despite being enormous. The Treasury issued well over \$1 trillion in bills in that window, and money funds met it by emptying the RRP — which is why you see the RRP fall from \$1.95 trillion in June 2023 to \$1.02 trillion by December 2023 on the chart above. The bills drained the sponge, not the bloodstream.

#### Worked example: \$1T of bills absorbed by money funds shifting out of the RRP — liquidity-neutral

Suppose the Treasury issues exactly \$1 trillion of new bills in a quarter, and money market funds buy all of it by reallocating out of the RRP.

- RRP before: \$1.95 trillion (the June 2023 level).
- Money funds move \$1.00 trillion out of the RRP into bills.
- RRP after: \$1.95T − \$1.00T = **\$0.95 trillion**.
- TGA: rises by \$1.00 trillion as the proceeds land.
- Bank reserves: **unchanged**.

The \$1 trillion of issuance "drained" \$1 trillion — but only from the RRP, which had \$1.95 trillion of slack to give. Reserves, sitting around \$3.0 trillion, were untouched. Funding markets, repo rates, and risk assets feel essentially nothing. The intuition: when there is a deep RRP buffer, bill issuance is a transfer between two idle accounts and the broad market shrugs.

### The coupon channel: liquidity-negative

Now run the same \$1 trillion as **coupons** instead. The buyer of a 10-year note is not a money market fund — MMFs cannot hold 10-year duration; it would blow up their mandate. The buyer is a bank, a pension fund, an insurer, or a bond fund. When a bank buys a coupon bond, it pays out of its **reserve account** at the Fed. There is no idle RRP cash to tap here, because these buyers were not parking money in the RRP in the first place — money funds were.

Trace the cash:

1. A bank pays \$1 for a coupon bond out of its reserves (bank reserves fall by \$1).
2. The \$1 lands in the TGA (TGA rises by \$1).

Net effect: reserves fell by \$1, TGA rose by \$1, and the RRP did not change. This time the drain hit the **live** pool — the reserves that back lending, repo, and market-making. **That is liquidity-negative.** Real money left the banking system and froze in the government's account. If reserves are abundant, the system can absorb it; but if reserves are getting scarce, draining them tightens funding, lifts repo rates, and can pressure risk assets. The same \$1 trillion of borrowing has a genuinely different effect than it did as bills.

#### Worked example: the same \$1T in coupons bought by banks drains reserves — liquidity-negative

Same \$1 trillion of borrowing, but now issued entirely in coupons and bought by banks out of reserves.

- Bank reserves before: \$3.03 trillion (the end-2022 level from the data).
- Banks pay \$1.00 trillion for coupons out of reserves.
- Bank reserves after: \$3.03T − \$1.00T = **\$2.03 trillion**.
- TGA: rises by \$1.00 trillion.
- RRP: **unchanged**.

A \$1 trillion drain straight out of \$3.03 trillion of reserves is a 33% cut to the system's working money — exactly the kind of move that can push repo rates above the Fed's target, stress dealers' balance sheets, and force the Fed to react. The intuition: coupons reach past the RRP buffer and hit reserves directly, which is why a coupon-heavy refunding is the one that actually tightens financial conditions. Compare the two worked examples and you have the whole post in two numbers: the same \$1 trillion left reserves at \$3.03T in one case and dropped them to \$2.03T in the other.

### Why this asymmetry exists at all

You might ask: why don't banks just buy bills too, and why don't money funds buy coupons, so the channels blur together? Because of **mandates and duration**. Money funds are legally and structurally constrained to ultra-short, ultra-safe assets — under the rules that govern them, they must hold a heavily weighted-average-maturity that is measured in weeks, not years — so they cannot take duration risk, and they live in bills and the RRP. Banks and bond funds *want* duration — to earn higher yields, to match long liabilities, to express rate views — so they live in coupons. The buyer base for each instrument is genuinely different, holds a different kind of cash, and pays out of a different account. That structural separation is what makes the bill-versus-coupon distinction a real liquidity lever rather than an accounting curiosity.

It is worth being honest that the channels are not perfectly clean. Some banks do hold bills for their liquidity portfolios; some non-money-fund investors buy bills out of cash that is not RRP cash; foreign buyers sit across both. So the real world is a *tendency*, not a law: bills *lean* toward RRP-funded money-fund demand, coupons *lean* toward reserve-funded bank and bond-fund demand. The reason the tendency is strong enough to trade on is that the *marginal* buyer — the one whose demand has to be coaxed out to clear a surge in supply — is, for bills in 2022–2024, overwhelmingly money funds sitting on a mountain of RRP cash, and for coupons, the duration buyers paying from reserves. When you are sizing the liquidity effect of a refunding, you care about the marginal dollar, and the marginal dollar respected the split cleanly enough that 2023 played out almost exactly as the framework predicts.

There is also a crucial *conditional*: the bill channel is only liquidity-neutral **as long as the RRP has cash to give.** Look back at the RRP chart — by late 2024 the parking lot was nearly empty (\$0.16 trillion in December 2024, \$0.01 trillion by October 2025). Once the RRP is drained, the next wave of bill issuance can no longer be funded by emptying the parking lot, because the parking lot is empty. At that point even *bills* start pulling on reserves, and the comfortable "bills are neutral" assumption breaks. The buffer is what makes the rule true, and the buffer can run out. We will make this the heart of the playbook.

## The QRA as a market-moving event

We have established the mechanism. Now treat the Quarterly Refunding Announcement as what it really is to a trader: a scheduled event that prints three numbers, each of which moves a different part of the market. This is the section that turns the plumbing into something you can position around.

Recall the three things the QRA contains: the **total borrowing need**, the **bill-versus-coupon mix**, and the **auction sizes per tenor**. Each maps to a market reaction:

- **Total borrowing need → overall supply tone.** A bigger-than-expected total means more debt the market must absorb, which is, all else equal, mildly bearish for bond prices (higher yields). But "all else equal" is doing a lot of work — the *composition* matters more than the headline.
- **Bill-versus-coupon mix → the liquidity effect and the front end.** A bill-heavy mix means more front-end supply (which can lift short-term rates and pulls cash from the RRP) but is liquidity-neutral for reserves. A coupon-heavy mix means more duration to sell and a genuine reserve drain — liquidity-negative.
- **Auction sizes per tenor → long-end yields.** A surprise increase in 10-year and 30-year auction sizes means dealers and investors must absorb more long-duration paper. More duration to digest tends to push long yields *up*, steepening the curve. This is the channel that most directly hits the 30-year bond.

The reason the QRA is a genuine event — not just a data dump — is that the market trades the *surprise* relative to expectations. Going into each refunding, dealers publish forecasts of the borrowing total and the expected coupon-auction sizes. If the Treasury announces smaller coupon auctions than feared, long yields can *fall* on the news even though borrowing is huge, because the market had braced for worse. If it announces a tilt toward bills, the front end and the RRP react. The QRA is one of the rare macro events where the *composition* of a government action, not just its size, is the tradeable signal.

### The 2023 refunding sequence, told as an event

The clearest case study is the back-to-back refundings of 2023, which we will detail with numbers in the real-markets section. The compressed version:

- **August 2, 2023 QRA:** the Treasury announced it would increase coupon auction sizes for the first time in over two years, on top of a large bill program — and it raised its borrowing estimate sharply. The combination of "more coupons" (duration to sell) plus a credit-rating downgrade the same week sent the 10-year yield marching higher into the autumn.
- **November 1, 2023 QRA:** the Treasury *under-shot* the feared coupon increases — it tilted more of the borrowing back toward bills and made the long-end auction increases smaller than dealers expected. The market, which had braced for another duration deluge, rallied hard. The 10-year yield, which had touched 4.88% in October, fell back toward 3.88% by year-end.

The November refunding became a textbook example of the QRA moving markets through *composition*: same enormous borrowing need, but a less-duration-heavy mix, and bonds ripped. A trader who understood that the bill-heavy tilt was liquidity-neutral and the smaller coupon sizes meant less duration to absorb could have read the rally coming straight off the 8:30 a.m. release.

## Common misconceptions

A few beliefs about Treasury issuance are widespread, intuitive, and wrong. Each one is correctable with a number we have already met.

### Misconception 1: "More issuance always pushes yields up."

The reasoning behind this myth is that more supply of anything lowers its price, and a bond's price falling means its yield rising — so more issuance must mean higher yields. Sometimes true, often not, because *what* is issued matters more than *how much*. In November 2023 the Treasury's total borrowing need was enormous, yet the 10-year yield *fell* from 4.88% toward 3.88% into year-end — because the mix tilted toward bills and the coupon-auction increases were smaller than feared. Supply that lands in bills barely touches the long end; supply that lands in 30-year bonds does. The headline borrowing number is the least informative part of the QRA. Watch the mix and the per-tenor sizes, not the total. The same logic explains why the long end can sell off on a refunding even when *total* borrowing is unchanged: if the Treasury merely shifts the existing borrowing from bills into more 10s and 30s, it adds zero to the total but forces the market to digest more duration, and long yields rise on the composition shift alone.

### Misconception 2: "More government debt means an instant crisis."

Rising debt is a real long-run concern, but the leap from "debt is high" to "markets will break tomorrow" skips all the plumbing. The 2023 H2 episode is the rebuttal: the Treasury issued well over a trillion dollars in a quarter, rebuilt the TGA by roughly \$700 billion, and funding markets stayed calm — because the RRP, sitting at \$1.95 trillion, absorbed the bill flood. Whether issuance causes stress depends entirely on the buffer it drains and the form it takes, not on the debt level in the abstract. A trillion of bills against a full RRP is a non-event; a trillion of coupons against scarce reserves is a problem. The level of debt tells you almost nothing about the next month's liquidity.

### Misconception 3: "The Fed funds the government's deficit."

A very common belief is that when the government runs a deficit, the Fed simply "prints the money" to cover it. In the US, that is not how it works. The Treasury funds the deficit by **selling debt to private investors** — money funds, banks, pension funds, foreigners — who pay with existing money. The Fed does separately buy and hold Treasuries as part of monetary policy (quantitative easing), which does inject reserves, but QE is a *monetary-policy* decision aimed at interest rates and is mechanically distinct from funding the deficit. During the QT period we have been studying, the Fed was *shrinking* its Treasury holdings (its balance sheet fell from \$8.96 trillion in April 2022 toward \$6.66 trillion by mid-2025), not buying the new issuance. The private sector funded the deficit. Conflating the two leads you to expect reserves to rise with every deficit dollar, when in fact issuance during QT was *draining* the buffers. For how QE actually injects reserves, see [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money).

### Misconception 4: "Bills are always liquidity-neutral."

This is the subtle one, and getting it wrong is how you misread 2025 by applying the 2023 rule. Bills are liquidity-neutral *only while the RRP has cash to absorb them.* In 2023 the RRP held \$1.95 trillion — a deep sponge — so bills drained the sponge and spared reserves. But by December 2024 the RRP was down to \$0.16 trillion, and by late 2025 essentially empty. With no sponge left, fresh bill issuance has to be funded by someone other than money-funds-emptying-the-RRP, and the marginal cash increasingly comes out of bank reserves. The same instrument that was harmless in 2023 can be a reserve drain in a drained-RRP world. Always check the buffer before applying the rule.

### Misconception 5: "The TGA balance doesn't matter for trading."

It is easy to dismiss the government's checking account as accounting trivia. But the TGA is a direct, real-time liquidity lever. When the TGA *rises* (issuance proceeds piling up, or a tax date pulling cash in), liquidity is being drained from markets. When the TGA *falls* (the government spending, or running it down ahead of a debt-ceiling deadline), liquidity is being *released* into markets. The debt-ceiling episode of early 2023 is the vivid case: as the Treasury drew the TGA down from \$0.46 trillion in January 2023 to \$0.05 trillion by June, it was effectively *injecting* several hundred billion of liquidity into markets — a quiet tailwind for risk assets in the first half of 2023 that reversed hard when the TGA rebuilt in the second half. Traders who track the TGA path treat it as a leading indicator of liquidity direction.

## How it shows up in real markets

Now the case studies, with dates and numbers, so the mechanism is not abstract.

### Case study 1: the 2023 refunding "tantrum" and reversal

This is the canonical example, and we have been circling it; here is the full arc.

**The setup (early 2023).** A debt-ceiling standoff froze net new issuance for months. To keep paying the bills, the Treasury ran the TGA down from \$0.46 trillion in January 2023 to about \$0.05 trillion by June 2023. Draining the checking account *released* liquidity into markets — part of why risk assets were resilient in the first half of 2023.

**The deal and the catch-up (June 2023).** The ceiling was lifted in early June. Now the Treasury had to rebuild the TGA *and* fund the deficit *and* roll maturing debt — all at once. The market saw the size of the coming issuance — well over a trillion dollars in a quarter — and braced for a liquidity drain.

**The August 2 QRA.** The Treasury announced its first coupon-auction size increases in over two years, plus a heavy bill program, and raised its borrowing estimate. The same week, a major agency downgraded the US credit rating. The 10-year yield, around 3.96% in late July, began a relentless climb — to 4.05%-ish through August and ultimately a peak near **4.88% in October 2023.** Commentators called it a "refunding tantrum" and a "supply tantrum."

**What actually happened to liquidity.** Despite the yield spike, *funding* markets stayed calm. The bill-heavy mix meant money funds emptied the RRP to buy the bills: the RRP fell from \$1.95 trillion in June 2023 to \$1.02 trillion by December 2023. Bank reserves, far from collapsing, actually *rose* over 2023 (from \$3.03 trillion at end-2022 to \$3.49 trillion at end-2023). The drain hit the parking lot, not the bloodstream. The yield move was about *duration supply and term premium* at the long end — not a reserve crunch.

**The November 1 QRA and the reversal.** At the next refunding, the Treasury tilted *back* toward bills and made the coupon-auction increases smaller than dealers had feared. The market, braced for another duration deluge, rallied hard. The 10-year yield fell from its October peak near 4.88% back toward **3.88% by December 2023** — a roughly 100-basis-point round trip in two months, driven substantially by the *composition* of the refunding, not its size.

The lesson a trader takes from 2023: a "wall of supply" headline is not automatically a liquidity event. Decompose it. If the wall is bills and the RRP is full, fade the liquidity panic. If the long-end auction sizes are jumping, respect the duration/term-premium move at the long end — but recognize that the front end and reserves can be fine even as the 30-year sells off.

### Case study 2: debt-ceiling TGA swings as a liquidity see-saw

Debt-ceiling episodes are a recurring, scheduled-ish source of large TGA swings, and the TGA chart makes them visible.

The pattern is mechanical. As a debt-ceiling deadline approaches and binds, the Treasury cannot issue net new debt, so it **draws down the TGA** to keep paying obligations — injecting liquidity into markets as the account empties (the January-to-June 2023 drawdown from \$0.46T to \$0.05T is the textbook case). Then, the moment the ceiling is raised, the Treasury **rebuilds the TGA** by flooding the market with issuance — draining liquidity back out (the \$0.05T-to-\$0.75T rebuild). So a debt-ceiling cycle is a liquidity *see-saw*: a tailwind on the way down, a headwind on the way back up.

For a trader, the playbook is to map the see-saw in advance. The drawdown phase (ceiling binding) is a quiet liquidity tailwind — modestly supportive for risk. The rebuild phase (post-deal) is a liquidity headwind — but, crucially, *how much* of a headwind depends on the bill-versus-coupon mix of the rebuild and the state of the RRP buffer. In 2023 the rebuild was bill-heavy against a full RRP, so the headwind was soft. In a future episode where the RRP is empty (as it became by 2025), an equivalent rebuild would bite reserves directly and the headwind would be real. Same see-saw, different ground underneath it.

### Case study 3: net liquidity around the refundings

Traders bundle these moving parts into a single popular proxy: **net liquidity = Fed total assets − ON RRP − TGA.** The idea is to start from the gross size of the Fed's balance sheet (Fed assets) and subtract the two big "sterilizing" accounts that lock cash away from markets (the RRP and the TGA). What is left is a rough gauge of the liquidity actually available to the financial system. This is the central measure of the companion post, [the central-bank balance sheet and net liquidity](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga).

![Net liquidity chart with gross Fed assets falling under QT while net liquidity holds during the 2023 bill flood](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-6.png)

The chart tells the 2023 story in one picture. The dashed line is gross Fed assets, falling steadily through QT from \$8.96 trillion in April 2022 toward \$6.66 trillion by mid-2025 — the Fed shrinking its balance sheet, which mechanically pulls reserves out. You would think that, plus a wall of issuance, would crush net liquidity. But look at the solid net-liquidity line through the 2023 H2 refunding window (shaded): it *holds roughly steady* even as gross assets fall. Why? Because the bill flood drained the **RRP**, and since net liquidity already subtracts the RRP, a falling RRP and a rising bill-funded TGA partly offset inside the formula — the drain came out of an account that was already netted out. The liquidity that markets actually run on did not collapse.

The later window (shaded toward the right) shows the flip side: by 2024–2025, with the RRP nearly empty, there is no more sponge to drain. Further QT and TGA rebuilds now come out of *reserves*, so additional drains start to bite net liquidity directly. The chart is a single-image summary of Misconception 4: bills were neutral while the RRP had cash, and stop being neutral once it runs out.

#### Worked example: net liquidity before and after a refunding

Pin it down with the formula and real levels around the 2023 H2 refunding.

Take mid-2023 (June), near the start of the rebuild:

- Fed assets: \$8.34 trillion.
- ON RRP: \$1.95 trillion.
- TGA: \$0.05 trillion (the debt-ceiling floor).
- Net liquidity = \$8.34T − \$1.95T − \$0.05T = **\$6.34 trillion.**

Now take end-2023 (December), after the bill flood and TGA rebuild:

- Fed assets: \$7.68 trillion (QT has shrunk it by \$0.66T).
- ON RRP: \$1.02 trillion (drained by \$0.93T as money funds bought bills).
- TGA: \$0.72 trillion (rebuilt by \$0.67T).
- Net liquidity = \$7.68T − \$1.02T − \$0.72T = **\$5.94 trillion.**

Net liquidity fell from \$6.34T to \$5.94T — a drop of about \$0.40 trillion over six months. But notice what cushioned it: gross Fed assets fell \$0.66 trillion (pure QT drain) *and* the TGA rebuilt by \$0.67 trillion (issuance drain), which together is more than \$1.3 trillion of draining pressure — yet net liquidity only fell \$0.40 trillion, because the RRP simultaneously gave up \$0.93 trillion that was already netted out of the formula. The RRP absorbed roughly two-thirds of the combined drain. The intuition: with a deep RRP buffer, even simultaneous QT and a huge bill-funded TGA rebuild only nicked the liquidity that markets care about — which is exactly why the feared 2023 crunch never arrived.

### Case study 4: what happens when the buffer is gone

Project the same machine forward to the world of late 2024 and 2025, when the RRP had fallen to \$0.16 trillion (December 2024) and then near zero. In that world, run the bill-issuance example again: the Treasury issues \$1 trillion of bills, but money funds no longer have \$1 trillion sitting in the RRP to reallocate. The cash to buy those bills now has to come from somewhere closer to reserves. So the marginal bill issuance starts draining the same live pool that coupons drain. The neat 2023 rule — "bills neutral, coupons negative" — collapses into "everything is now somewhat negative," because the shock absorber is empty.

This is why, by 2025, the Fed slowed and then ended quantitative tightening and watched reserve metrics closely: with the RRP exhausted, every further drain — QT runoff, TGA rebuilds, bill issuance — landed on reserves, and the system had moved from "abundant reserves" toward "ample, watch carefully." The transmission from issuance to funding stress that was muted in 2023 became live again. The single most important variable across both regimes is the same: **how full is the buffer?** In 2023 it was full and bills were harmless; in 2025 it was empty and nothing was harmless. Read the buffer before you read anything else.

## How to trade it / The playbook

Everything above collapses into a four-step read you can run every refunding. The point is not to predict the deficit — that is given — but to translate the Treasury's published choices into a liquidity verdict and a position.

![Issuance playbook four reads QRA mix then TGA path then RRP buffer then net liquidity verdict](/imgs/blogs/treasury-issuance-bills-coupons-liquidity-drain-7.png)

### Step 1: Read the QRA mix

When the QRA prints at 8:30 a.m. on the first Wednesday of February, May, August, or November, go straight to the **bill-versus-coupon mix** and the **per-tenor coupon-auction sizes** — not the headline borrowing total. Ask:

- **Bill-heavy?** Expect more front-end supply (mild upward pressure on short rates and bill yields) and a pull on the RRP, but a *muted* reserve drain. Front-end and money-market trades are where the action is; the long end and reserves are relatively protected.
- **Coupon-heavy, with bigger 10s/30s auctions than expected?** Expect duration supply to push long-end yields up and the curve to steepen, *and* a genuine reserve drain — liquidity-negative. This is the mix that tightens financial conditions.
- **Smaller coupon increases than feared?** A bullish surprise for the long end even if total borrowing is huge — the November 2023 setup. Be ready to fade a pre-positioned bearish crowd.

The signal is always the *surprise versus the dealer-consensus mix*, because that is what the market trades.

### Step 2: Trace the TGA path

Pull up the Treasury's published TGA target and the recent balance. A **rebuild** (TGA rising toward target) means a drain is coming as proceeds pile up — a liquidity headwind in the weeks ahead. A **drawdown** (TGA falling, often into a debt-ceiling deadline or seasonal spending) means liquidity is being released — a tailwind. Also flag the big **tax dates** (mid-April especially), when cash floods into the TGA and temporarily drains markets regardless of issuance. The TGA path is your calendar of liquidity direction.

### Step 3: Check the RRP buffer — the decisive variable

This is the step that separates 2023 from 2025, and it is the one most casual readers skip. Look at the ON RRP balance:

- **Buffer full (RRP large, as in 2023, \$1.9T+):** bill issuance is absorbed by money funds emptying the RRP — **liquidity-neutral**. A bill-heavy refunding against a full RRP is a fade-the-panic setup. The drain hits idle cash, not reserves.
- **Buffer empty (RRP near zero, as in late 2024–2025):** there is no sponge left, so even bill issuance starts pulling on **reserves** — and *every* drain (QT, TGA rebuild, bills) is now liquidity-negative. Respect it. Watch repo rates (SOFR), the spread of repo to the Fed's reverse-repo rate, and any signs of funding stress, as covered in [money-market plumbing](/blog/trading/macro-trading/money-market-plumbing-repo-collateral-sofr).

The buffer is the conditional that makes every other rule true or false. Never apply "bills are neutral" without first confirming the RRP can pay for them.

### Step 4: Form the net-liquidity verdict and position

Combine the three reads into the proxy: **net liquidity = Fed assets − RRP − TGA.** Estimate its direction over the coming weeks:

- **Verdict: liquidity holds (full buffer + bill-heavy mix).** This is the 2023 H2 lesson. When the crowd is panicking about a "wall of supply" but the mix is bills and the RRP is deep, **fade the liquidity panic** — the drain hits inert cash and net liquidity barely moves. Stay constructive on risk; treat the supply scare in long-end yields as a term-premium move, not a reserve crunch (and a potential entry to fade once the duration is digested).
- **Verdict: liquidity drains (empty buffer + TGA rebuild or coupon-heavy mix).** **Respect the drain.** With no RRP sponge, a rebuild or a coupon-heavy refunding pulls real reserves out — trim risk, expect funding to tighten, watch repo stress and the front end, and lean toward a steeper curve as long-end supply lands. This is the configuration where a supply scare is *not* a fade.

### The invalidation — what tells you the read is wrong

A playbook without an exit is a gamble. Concrete invalidations for each side:

- If you faded a supply panic on a "full buffer + bills" read, the trade is **invalidated** if the RRP stops absorbing — i.e., the RRP balance flattens out near a floor while bills keep coming, or repo/SOFR starts pushing up toward or above the Fed's reverse-repo rate. That is the buffer running dry, and the neutral assumption is breaking. Cut the risk-on lean.
- If you respected a drain on an "empty buffer + rebuild" read but reserves prove sturdier than expected — repo stays calm, SOFR well-behaved, the front end orderly even as the TGA rebuilds — then the drain is being absorbed (perhaps by a still-elastic banking system) and the liquidity-negative thesis is weaker than you thought. Don't overstay the defensive posture.
- Across both: the QRA *composition surprise* is your fastest signal. If the Treasury's actual bill-versus-coupon mix diverges from what you priced — more coupons than you assumed, or a tilt back to bills — re-run all four steps from the 8:30 a.m. release, because the liquidity verdict can flip on the mix alone.

The whole discipline reduces to one ordered habit: **read the mix, trace the TGA, check the RRP buffer, then call net liquidity — in that order, every refunding.** Two numbers carry most of the weight, and they are both public and scheduled: the bill-versus-coupon split in the QRA, and how much cash is left in the RRP. Get those two right and you will know — before the crowd finishes reacting to the headline borrowing total — whether the next wall of Treasury supply is a liquidity non-event to fade or a real drain to respect.

## Further reading & cross-links

- [The central-bank balance sheet: net liquidity, reserves, RRP, and the TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) — the companion post that builds the net-liquidity proxy in full and tracks all three Fed accounts together.
- [Money-market plumbing: repo, collateral, and SOFR](/blog/trading/macro-trading/money-market-plumbing-repo-collateral-sofr) — what happens in the funding markets when reserves get scarce, and the signals (SOFR, repo spreads) that tell you the buffer is running dry.
- [Interest rates: the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — why duration, the yield curve, and the level of rates frame everything in this post.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the policy side: how the Fed's target rate, its reverse-repo rate, and reserve management interact with everything the Treasury does on the issuance side.
- [Quantitative easing explained: printing money](/blog/trading/finance/quantitative-easing-explained-printing-money) — the monetary-policy channel that injects reserves, and why it is mechanically distinct from the Treasury funding the deficit.

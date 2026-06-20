---
title: "Bank Treasury and Asset-Liability Management: The Balance-Sheet Cockpit"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank's treasury and its asset-liability committee actively steer the whole balance sheet's liquidity and interest-rate risk, and how funds transfer pricing decides who in the bank actually earns the spread."
tags: ["banking", "treasury", "asset-liability-management", "alm", "alco", "funds-transfer-pricing", "interest-rate-risk", "liquidity-risk", "repricing-gap", "maturity-transformation"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank's treasury is the cockpit that actively steers the whole balance sheet's two great risks: liquidity (can we fund ourselves tomorrow?) and interest-rate risk (what does a rate move do to our income and our equity?). The asset-liability committee (ALCO) sets the policy, and funds transfer pricing (FTP) is the internal price that makes every desk earn only the spread it actually added.
>
> - Treasury sits *between* the deposit business and the lending business, nets the two sides, and manages the leftover mismatch — the maturity-transformation trade — on behalf of the entire bank.
> - FTP is the hidden machine that charges the lender an internal cost of funds and pays the deposit-gatherer an internal credit. Lend at 6.50%, get charged 4.20% by treasury, and your real spread is 2.30% — not 5.00%.
> - The two numbers ALCO watches every month are **NII sensitivity** (how 12-month income moves if rates shift) and the **repricing gap** (how much of each side reprices, and when). A liability-sensitive bank can lose \$110 million of annual income from a single 1% hike.
> - The one fact to remember: a bank can be perfectly solvent and still die in 36 hours if its liquidity buffer is too small. Treasury's job is to make sure that never happens — and Silicon Valley Bank is what it looks like when it does.

In the first week of March 2023, the treasury team at Silicon Valley Bank was sitting on a balance sheet that, on paper, looked fine. The bank had roughly \$209 billion in assets, mostly long-dated US Treasuries and mortgage bonds it had bought when interest rates were near zero. It had about \$175 billion in deposits, most of them from venture-backed startups. The franchise was profitable. The loan book was clean. And yet within 48 hours the bank was gone — seized by the FDIC after depositors tried to pull \$42 billion in a single day, with another \$100 billion queued for the next morning.

What killed SVB was not a credit problem. Almost none of its loans defaulted. What killed it was the two risks that treasury exists to manage and that, at SVB, treasury had been allowed to ignore: interest-rate risk (the bank's long bonds had fallen in value as rates rose, leaving about \$17 billion of unrealized losses) and liquidity risk (when frightened depositors ran, the bank had to sell those bonds at a loss to raise cash, which crystallized the loss, which frightened depositors more). The maturity-transformation trade — borrow short, lend long — had been left un-steered, and the cockpit was empty.

The diagram below is the mental model for this whole post. Treasury is not a back-office function that moves cash around. It is the cockpit of the bank: it takes the short, cheap, flighty money the deposit desks raise and the long, fixed, locked-up loans the lending desks make, nets the two together, and actively manages the leftover liquidity and interest-rate risk so the bank survives whatever the next rate move or the next bad week throws at it.

![Graph showing treasury between deposits and loans managing liquidity risk and rate risk](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-1.png)

This connects straight to the spine of this whole series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. Treasury is the function that *consciously sizes* that fragile trade — how much mismatch to run, how much to hedge, how big a buffer to hold against a run. Get it right and the bank quietly earns its spread through every cycle. Get it wrong and the bank is one rate move or one rumor away from the exit.

## Foundations: treasury, ALCO, ALM, and the two risks

Before we go deep, let us build every term from zero, because the vocabulary here is the part that makes the subject feel harder than it is.

Start with the everyday version. Picture a corner shop that takes its customers' spare cash for safekeeping and, to earn a little, lends most of it out to the workshop down the street for a year. The shop pays its depositors a tiny rate and charges the workshop a bigger one; the difference is its living. But the shop has a problem nobody talks about until the day it bites: the depositors can ask for their cash *back at any time*, while the loan to the workshop is locked up for a year. If too many depositors show up at once, the shop has promised money it cannot instantly produce. And if interest rates jump, the shop may find itself paying more to keep its depositors than it is earning on a loan whose rate was fixed last year. Those are the two ancient risks of banking — running out of cash, and getting caught on the wrong side of a rate move — and a real bank does at industrial scale exactly what the corner shop does by instinct.

**Treasury** is the department inside a bank whose job is to manage those two risks for the entire institution at once. It is the central funding desk. Every other part of the bank — retail, corporate, mortgages, cards — generates either money coming *in* (deposits) or money going *out* (loans). Treasury collects the net of all of it and decides how to fund the gap and how to protect it against rate moves. The treasurer is, in effect, the pilot of the balance sheet.

**ALM** stands for *asset-liability management*. It is the discipline treasury practices: deliberately managing the relationship between what the bank owns (its assets — loans, bonds, cash) and what it owes (its liabilities — deposits, borrowings) so that the mismatch between them stays inside limits the bank can survive. ALM is not about any single loan or any single deposit; it is about the *whole balance sheet seen as one position*.

**ALCO** is the *asset-liability committee* — the senior body, usually chaired by the CFO or treasurer and including the heads of the business lines and risk, that meets monthly (sometimes weekly in a crisis) to look at the bank's mismatch, decide how much risk to run, and approve the limits treasury operates within. ALCO is where the maturity-transformation trade is steered. It is, quite literally, the room where someone decides how much of the bank's survival to bet on rates not moving.

Now the two risks, defined precisely, because people constantly conflate them.

**Liquidity risk** is the risk that the bank cannot meet its obligations *when they fall due* — that depositors or lenders want their money back faster than the bank can produce cash. It is a *timing* problem. A bank can be perfectly solvent (its assets are worth more than its liabilities) and still fail for lack of liquidity, because its assets are long-dated loans it cannot turn into cash today. Liquidity is about *can we pay right now*.

**Interest-rate risk** — in the banking book it is called **IRRBB** (interest-rate risk in the banking book) — is the risk that a change in interest rates hurts the bank, either by squeezing the income it earns (because funding reprices faster than assets) or by shrinking the value of its equity (because long fixed-rate assets fall in price when rates rise). It is a *pricing* problem, not a timing one. Interest-rate risk is about *how much do we earn, and how much is our equity worth, if rates move*.

The two are cousins and they often arrive together — a rate shock can trigger an outflow, as it did at SVB — but they are not the same risk and they have different cures. The cure for liquidity risk is a *buffer*: a stack of assets you can turn into cash fast. The cure for interest-rate risk is a *hedge*: a contract that pays you when rates move against you. Treasury runs both.

It is worth pausing on *why* they so often arrive together, because the link is the heart of modern bank fragility. Rising rates do two things at once to a liability-sensitive bank: they push up its funding cost (the interest-rate problem) *and* they make the long bonds it holds worth less on the market (a balance-sheet hole). When depositors notice both — that the bank is paying them too little and sitting on bond losses — they get nervous and start to leave (the liquidity problem). To meet the outflow, the bank must sell those very bonds, turning a paper loss into a realized one, which deepens the hole and frightens the next wave of depositors. Interest-rate risk and liquidity risk are not two separate accidents; in a crisis they become a single feedback loop, each one feeding the other. Treasury's job is to keep that loop from ever starting — by hedging the rate exposure so the bond losses never get large, and by holding a buffer so the bank never has to sell into a falling market to raise cash.

One more term, because the rest of the post leans on it. The **banking book** is the part of the bank's balance sheet held to *earn the spread and be kept to maturity* — the deposits, the loans, the bonds bought for yield. It is distinct from the **trading book**, which holds positions meant to be bought and sold for short-term profit. ALM and IRRBB are about the banking book. (The trading book has its own market-risk machinery, covered elsewhere in this series.)

That is the whole vocabulary. Treasury runs ALM, ALCO governs it, the two risks are liquidity and interest-rate, the arena is the banking book, and FTP — which we will meet next — is the internal price that ties it all together.

## What treasury actually does all day: four risk desks under one roof

People hear "treasury" and think of someone wiring money. The real job is closer to running four small risk desks at once, each watching a different way the balance sheet can hurt the bank. The matrix below lays them out: what can go wrong, how treasury measures it, and the tool it reaches for.

![Matrix of four ALM risks with how treasury measures and hedges each one](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-3.png)

The first desk is **liquidity**. The fear is that deposits flee and the bank cannot fund its loans. Treasury measures this with the liquidity coverage ratio (LCR), the net stable funding ratio (NSFR), and an internal count of "survival days" — how long the bank could meet outflows from its buffer alone. The tool is a stockpile of high-quality liquid assets (HQLA) plus term funding that cannot run.

The second desk is **interest-rate risk in the banking book**. The fear is that a rate move shrinks either income or equity. Treasury measures it with the repricing gap, with net interest income (NII) sensitivity to rate shocks, and with economic value of equity (EVE) sensitivity. The tool is the interest-rate swap and the deliberate choice of how much of the book to hold fixed versus floating.

The third desk is **foreign-exchange risk**. A bank with assets in one currency and funding in another carries a hidden bet on the exchange rate. Treasury measures the net open position in each currency and squares it with FX forwards and swaps so that the bank earns its banking spread, not a currency gamble.

The fourth desk is **capital**. This is the thin equity cushion that absorbs losses. Treasury watches the CET1 ratio (the core capital divided by risk-weighted assets) and the leverage ratio, and when either gets tight it either issues new capital or slows the growth of risk-weighted assets so the cushion is not stretched too thin.

Notice the pattern across all four: a clear fear, a number that measures it, and a specific instrument that treats it. That is the discipline of ALM. It turns vague dread ("what if rates move? what if depositors run?") into a quantity with a limit and a hedge. Everything that follows is the detail of how the two biggest of those desks — liquidity and interest-rate risk — actually work, and how FTP makes the whole machine accountable.

## Funds transfer pricing: the internal price that makes the spread honest

Here is the question that breaks most people's intuition about banking. When a branch raises a \$100 deposit at 1.50% and a corporate banker lends that \$100 at 6.50%, who earned the 5.00% spread — the branch or the banker?

The naive answer is "the banker, obviously — he made the loan." But that answer would ruin a bank, because it would tell management that lending is where all the money is and deposits are a cost center to be minimized. In fact the cheap, sticky deposit is often the more valuable half of the trade. The mechanism that gets the credit right is **funds transfer pricing**, and it is the most important piece of internal plumbing most people have never heard of.

FTP works like this: treasury becomes the internal bank for every desk. It *buys* every deposit and *sells* the funding for every loan, at an internal rate pegged to the market yield curve for that maturity. The deposit desk does not keep its deposits; it sells them to treasury and is paid an **FTP credit**. The loan desk does not fund its own loans; it buys the money from treasury and is charged an **FTP cost**. Each desk then earns only the spread between its customer rate and treasury's internal rate. The pipeline below traces a single \$100 through the machine.

![Pipeline showing FTP credit to the deposit desk and FTP charge to the loan desk through treasury](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-2.png)

Walk the numbers. Suppose the relevant point on the curve — the internal rate treasury uses for money of this maturity — is 4.20%. The deposit desk raises \$100 at 1.50% from a saver and sells it to treasury, which pays it an FTP credit of 4.20%. The deposit desk's profit is 4.20% − 1.50% = **2.70%** — that is the value of gathering cheap money. The loan desk lends \$100 at 6.50% and buys the funding from treasury at an FTP cost of 4.20%. Its profit is 6.50% − 4.20% = **2.30%** — that is the value of finding and pricing a good loan. Add them: 2.70% + 2.30% = 5.00%, the total spread. FTP has split the spread fairly between the two desks that earned it, and — crucially — treasury is left holding the *rate risk* in the middle, because it owes the deposit desk 4.20% on money that can leave tomorrow while it earns 4.20% on a loan locked for five years.

That last point is the whole reason FTP exists, beyond fair scorekeeping. By routing everything through treasury at a curve-based rate, the bank *centralizes* the interest-rate mismatch in one place where it can be measured and hedged, instead of leaving it scattered invisibly across a thousand loans and deposits. The desks earn clean, rate-neutral spreads; treasury owns the rate bet on purpose and manages it deliberately.

#### Worked example: the FTP charge and credit on a matched pair

Let us make this fully concrete with round numbers. A branch raises a \$10,000 one-year term deposit and pays the saver 1.50%, costing it \$150 a year. A small-business banker lends that \$10,000 as a one-year loan at 6.50%, earning \$650 a year. The naive view says the loan earned \$650 − \$150 = \$500 and the deposit was just a cost.

Now apply FTP at a one-year internal rate of 4.20%. Treasury pays the branch an FTP credit of 4.20% × \$10,000 = \$420. The branch's profit is \$420 − \$150 = **\$270**. Treasury charges the banker an FTP cost of 4.20% × \$10,000 = \$420. The banker's profit is \$650 − \$420 = **\$230**. The two add to \$500 — the same total — but now management can see that the *deposit* generated \$270 and the loan \$230. The intuition: FTP is the device that stops a bank from starving its most valuable franchise, the cheap deposit base, by accident.

#### Worked example: why the FTP curve, not a single rate, is the point

The previous example used one internal rate, 4.20%, because both legs were one-year. Real balance sheets are not matched. Suppose the branch raises a deposit that behaves like 3-month money (it can leave fast) while the banker makes a 5-year fixed-rate loan. Treasury must use the *3-month* point of the curve to pay the deposit desk — say 4.50% — and the *5-year* point to charge the loan desk — say 3.90% (in an inverted curve, short rates can sit above long rates).

The deposit desk earns 4.50% − 1.50% = 3.00%. The loan desk earns 6.50% − 3.90% = 2.60%. Both look healthy. But treasury is now *paying 4.50% on money that can vanish and earning 3.90% on money locked for five years* — a negative 0.60% carry on the mismatch, plus the risk that the 3-month rate it owes keeps climbing while the 5-year loan stays put. FTP has made the desks' spreads honest and dumped the ugly maturity mismatch squarely on treasury's desk, which is exactly where someone is paid to watch it. The intuition: FTP does not make the maturity-transformation risk disappear; it *concentrates* it where it can be seen and hedged.

### The liquidity premium: FTP also prices the run risk

There is a second charge buried inside a good FTP framework, and it is the part most outsiders miss. Beyond the *rate* charge — pegging the internal price to the yield curve — sophisticated banks add a **liquidity premium** to the FTP charge on loans, and a corresponding **liquidity credit** to sticky deposits. The logic is that a long-dated, hard-to-fund loan does not just consume rate-matched money; it consumes the bank's scarce *liquidity capacity*, because it has to be funded through good times and bad. So treasury charges the loan desk an extra few tenths of a percent for the liquidity it ties up, and pays a stable insured deposit an extra credit for the liquidity it provides.

This is how FTP quietly steers the bank's behavior toward safety. If a loan desk is charged a fat liquidity premium for funding a 10-year illiquid loan with overnight money, the loan stops looking profitable, and the desk is pushed to either price the loan higher or seek longer-term funding. If the branch network is paid a generous liquidity credit for gathering sticky insured deposits, suddenly the "boring" deposit franchise looks like the most valuable business in the bank — which, for a maturity-transformation machine, it is. FTP done well does not just keep score; it bends the whole organization toward the funding structure treasury wants.

Where FTP is done *badly*, the bank rots from the inside. If treasury uses a single flat internal rate regardless of maturity or liquidity, the loan desks are effectively subsidized to make long, illiquid, rate-mismatched loans — the desks book apparent profits while shoveling hidden rate and liquidity risk onto a treasury that is not charging for it. Many of the banks that blew up on duration mismatches had FTP systems that simply failed to charge for the risk being created, so the people creating it had every incentive to create more.

## The repricing gap: the simplest map of interest-rate risk

So treasury is left holding the rate mismatch. How does it measure it? The oldest and most intuitive tool is the **repricing gap** (also called the maturity-gap report). The idea is simple: sort everything on the balance sheet by *when its interest rate next resets*, and in each time bucket, compare how much of the asset side reprices against how much of the liability side reprices.

An asset "reprices" when its interest rate changes — a floating-rate loan reprices the moment its benchmark moves; a fixed 5-year loan does not reprice until it matures or is refinanced. A deposit reprices when the bank can change the rate it pays — overnight savings reprice almost immediately; a 2-year certificate of deposit does not reprice until it matures. Line up the two sides bucket by bucket and you can see, at a glance, whether the bank is exposed to rising or falling rates.

![Grouped bar chart of assets and liabilities repricing by time bucket with the gap labeled](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-4.png)

The chart above is an illustrative gap report for a bank whose funding is heavily short-dated. In the 0–3 month bucket, \$18 billion of assets reprice but \$40 billion of liabilities do — a gap of −\$22 billion. That negative near-term gap is the signature of a **liability-sensitive** bank: more of its funding reprices soon than its assets do. Further out, the picture flips — in the 3–5 year and 5-year-plus buckets, far more assets reprice than liabilities, because the bank holds long fixed-rate loans and bonds funded by deposits that have long since reset. The bank is short-funded against long assets, the classic maturity-transformation posture, just measured precisely.

The sign of the near-term gap tells you the bet. A bank with a *negative* short-term gap loses income when rates rise (its funding costs jump before its asset yields catch up) and gains when rates fall. A bank with a *positive* short-term gap — an **asset-sensitive** bank — is the mirror image: rising rates are a windfall, falling rates a squeeze.

#### Worked example: turning a repricing gap into a dollar of income

Take just the 0–3 month bucket from the chart: a gap of −\$22 billion (\$18bn of assets repricing against \$40bn of liabilities). Suppose rates rise by 1% (100 basis points; a basis point is one hundredth of a percent). The crude gap estimate of the hit to annual net interest income is the gap times the rate change: −\$22,000,000,000 × 1% = **−\$220 million**.

The logic: \$40 billion of funding now costs an extra 1%, which is +\$400 million of expense, while only \$18 billion of assets earn an extra 1%, which is +\$180 million of income. Net: \$180m − \$400m = −\$220 million of income, for as long as the rate change persists and before anything further reprices. The intuition: the repricing gap converts an abstract "we're liability-sensitive" into a hard number — a 1% move costs this bank \$220 million a year from the near bucket alone.

The repricing gap is beautifully simple, which is also its weakness. It treats every deposit in a bucket as if it reprices fully and instantly, which is rarely true — and that single assumption is where the real subtlety of ALM lives. We will fix it with deposit beta shortly. But first, the more rigorous successor to the gap report: NII sensitivity.

## NII sensitivity: what a rate shock does to next year's income

ALCO does not actually steer by the raw gap. It steers by **NII sensitivity** — a direct estimate of how much the bank's net interest income over the next 12 months changes under a defined rate shock. The bank builds a model of every asset and liability, applies a parallel shift in rates (typically ±1% and ±2%, sometimes a full ±3%), re-prices everything as it reprices through the year, and reports the change in projected income. It is the gap report, but dynamic and far more honest about *timing*.

![Bar chart of change in twelve month net interest income under parallel rate shocks](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-7.png)

The chart above shows an illustrative NII-at-risk profile for a liability-sensitive bank with a base-case 12-month net interest income of about \$2.0 billion. A +1% parallel shock costs it roughly \$110 million of income; a +2% shock costs about \$220 million. A −1% shock *helps* by \$110 million; a −2% shock by \$220 million. The downward-sloping profile — income falling as rates rise — is the visual fingerprint of liability sensitivity, and it is exactly the wrong shape to have when a central bank is hiking.

ALCO reads this chart as a *limit*. A bank might set a rule that NII-at-risk from a ±2% shock cannot exceed, say, 8% of projected income. If the +2% bar would breach that — here \$220m on a \$2.0bn base is 11%, over the limit — treasury must act: shorten the assets, lengthen the funding, or put on a swap that pays the bank when rates rise, until the modeled hit comes back inside the limit.

#### Worked example: sizing NII-at-risk against a limit

Suppose the board's risk appetite says the bank may lose at most 8% of its 12-month NII under a +2% parallel shock. Base-case NII is \$2.0 billion, so the limit is 8% × \$2,000,000,000 = \$160 million of allowed loss. The model shows the +2% shock would actually cost \$220 million. The breach is \$220m − \$160m = **\$60 million** over the limit.

To close it, treasury enters a "pay-fixed, receive-floating" interest-rate swap — a contract where the bank pays a fixed rate and receives a floating rate, so it *gains* when rates rise, offsetting the income it loses. If the swap is sized so that a +2% move produces a \$60 million gain, it neutralizes exactly the breach and brings NII-at-risk back to the \$160 million limit. The intuition: NII sensitivity is not a forecast, it is a control dial — ALCO reads the breach in dollars and treasury turns the dial with a swap until the bank is back inside its risk appetite.

A second measure travels alongside NII sensitivity: **EVE sensitivity** — the change in the *economic value of equity*. Where NII looks at next year's *income*, EVE looks at the present value of *all* the bank's future cash flows under a rate shock — essentially, what the bank's equity would be worth if you marked everything to market today. A bank can look fine on NII (short-horizon income holds up) and terrible on EVE (its long bonds have lost a fortune in market value). That gap between the two measures is precisely the trap SVB fell into, and we will return to it. For now, hold the pair in mind: **NII = income over the next year; EVE = value of equity right now.** ALCO watches both because a bank can die from either.

## Deposit beta: the one number that decides everything

The repricing gap and the NII model both rest on a single fragile assumption: how fast do deposit rates actually move when the central bank moves? This is captured by **deposit beta** — the fraction of a change in the policy rate that a bank passes through to the rates it pays depositors. A beta of 0.30 means that when the Fed raises rates by 1%, the bank raises its deposit rates by only 0.30%. A beta of 1.0 means full pass-through.

Deposit beta is the most important — and most uncertain — number in all of ALM, because it decides whether the bank's cheap funding stays cheap when rates rise. Low beta is a gift: the bank's funding costs barely move while its asset yields climb, and the margin widens. High beta is a curse: funding costs chase the policy rate up and the spread compresses. And beta is not constant — it starts low at the beginning of a hiking cycle, when depositors are slow to notice, and creeps up as they wake up and start shopping for yield.

![Step chart of cumulative deposit beta rising through the 2022 to 2024 hiking cycle](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-9.png)

The chart shows the cumulative deposit beta the US banking system actually experienced through the 2022–2024 hiking cycle: about 0.10 in mid-2022 (banks passed through almost nothing at first), rising to roughly 0.25 by the end of 2022, 0.40 by mid-2023 after the spring bank failures jolted depositors awake, and around 0.55 by mid-2024. That arc — slow then fast — is why bank margins jumped early in the cycle and then gave some of it back as betas caught up.

For treasury, the lesson is brutal: the bank's interest-rate risk is dominated by an assumption it does not control. A bank that modeled its NII sensitivity assuming a 0.30 beta, and then found betas running at 0.55 because depositors got nervous and yield-hungry, would discover its real income hit was nearly double what the model said. Estimating beta — by deposit type, by customer segment, by how the cycle is behaving — is a huge part of what an ALM team actually does.

#### Worked example: how deposit beta swings the spread

A bank holds \$100 billion of deposits and \$100 billion of floating-rate loans. The central bank hikes 2%. The loans, being floating, reprice fully: asset income rises by 2% × \$100bn = +\$2.0 billion. Now consider two beta worlds.

At a **low beta of 0.25**, deposit costs rise by 0.25 × 2% × \$100bn = +\$500 million. Net interest income improves by \$2.0bn − \$0.5bn = **+\$1.5 billion**. At a **high beta of 0.65**, deposit costs rise by 0.65 × 2% × \$100bn = +\$1.3 billion. Net interest income improves by only \$2.0bn − \$1.3bn = **+\$0.7 billion**. Same bank, same hike, same loans — but the income windfall is more than twice as large in the low-beta world. The intuition: deposit beta is the lever that turns a rate hike into either a feast or a shrug, and treasury's forecast of it is the single most consequential guess in the whole ALM model.

## Asset-sensitive or liability-sensitive: the same hike, opposite fates

Everything so far comes down to one question about a bank's posture: when rates rise, does its income go up or down? The answer splits all banks into two camps, and the before-and-after below shows the same hiking cycle producing opposite outcomes.

![Before and after diagram of an asset sensitive bank versus a liability sensitive bank in a hike](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-6.png)

An **asset-sensitive** bank has assets that reprice faster than its funding. Its loans are floating-rate, resetting to 7.5% as the central bank hikes, while its deposits are sticky and lag near 1.5%. When rates rise, its income rises — the margin widens, and the hike is a windfall. This is the comfortable posture, and most large US banks engineered themselves into it going into the 2022 cycle, which is why aggregate bank net interest margin jumped in 2022–2023.

A **liability-sensitive** bank is the mirror. Its assets are locked — long-dated fixed-rate bonds and mortgages stuck near 2.5% — while its funding (wholesale borrowing, large certificates of deposit, rate-sensitive corporate cash) reprices toward 5% as rates rise. When the central bank hikes, this bank's funding costs jump while its asset yields stay frozen. Its margin gets squeezed, and a big enough move can push its spread negative. This is the SVB posture, and it is the trap the rest of this post circles back to.

Crucially, neither posture is "wrong." A bank that *expects rates to fall* might deliberately run liability-sensitive, so the falling-funding-cost side of the cycle becomes its windfall. The sin is not the posture; the sin is running an *unintended, un-hedged, un-sized* posture — drifting into a giant rate bet without ALCO ever deciding to take it. Treasury's job is to make the bet a choice, with a limit and a hedge, not an accident.

## Liquidity: the buffer, the survival horizon, and why solvent banks die

Switch risks now, from rates to cash. A bank can have its interest-rate risk perfectly hedged and still fail overnight if it cannot produce cash when depositors want it. Liquidity is a *timing* problem, and it is the faster killer of the two — interest-rate risk grinds a bank down over quarters, but a liquidity crisis can end a bank in a day.

The first thing to internalize is that **liquidity is not solvency.** Solvency is about whether your assets are worth more than your liabilities over time. Liquidity is about whether you can turn assets into cash *today*. A bank stuffed with perfectly good 30-year mortgages is deeply solvent and, if everyone wants their deposits back this afternoon, completely illiquid — because nobody buys \$50 billion of mortgages in an afternoon at full price. The maturity-transformation trade is, by construction, a permanent liquidity mismatch: the deposits can leave faster than the loans can be called.

Treasury's defense is the **liquidity buffer**: a stockpile of assets that can be turned into cash quickly and without a big loss. The buffer is a ladder, with the most reliable money on top.

![Stack diagram of a layered liquidity buffer sized to survive a thirty day run](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-10.png)

At the top sits cash and reserves at the central bank — instantly available, no haircut, the first line against an outflow (say \$15 billion). Below it, **HQLA** (high-quality liquid assets) — government bonds and agency mortgage securities that can be sold or pledged for cash the same day with a tiny discount (say \$35 billion). Below that, pledgeable loans that can be posted as collateral at the central bank's discount window, slower and at a haircut (say \$20 billion). And the whole ladder is sized against a *modeled stress outflow* — under the Basel liquidity coverage ratio, the cash that could leave in a 30-day stress (say \$50 billion). The rule is blunt: the buffer must cover the modeled outflow.

#### Worked example: sizing the liquidity buffer with the LCR

The **liquidity coverage ratio (LCR)** is the formal rule, and it is just a fraction: HQLA divided by net cash outflows over a 30-day stress, which must be at least 100%. Take a bank with \$50 billion of HQLA. The regulator's stress scenario assumes specified run-off rates: suppose \$30 billion of "stable" insured retail deposits run off at 5% (= \$1.5bn), \$40 billion of "less stable" deposits at 10% (= \$4.0bn), and \$50 billion of flighty corporate and wholesale funding at 40% (= \$20bn). Net 30-day outflow = \$1.5bn + \$4.0bn + \$20bn = \$25.5 billion.

LCR = \$50bn ÷ \$25.5bn = **196%** — comfortably above the 100% minimum. The bank holds nearly twice the HQLA the 30-day stress demands. Now stress it harder: if a deposit panic pushes the corporate run-off rate from 40% to 90%, that bucket alone runs \$45 billion, total outflow jumps to about \$50.5 billion, and the LCR collapses to \$50bn ÷ \$50.5bn = **99%** — under water. The intuition: the LCR is only as safe as its assumed run-off rates, and a real panic blows through the assumptions — which is exactly why SVB's "fine" liquidity ratios meant nothing once 94% of its deposits, all uninsured and all flighty, decided to leave at once.

The deeper lesson is that liquidity risk is *behavioral*, and behavior changes in a crisis. The buffer is sized against a model of how depositors run, and the model is calibrated on normal times. In a true panic — especially the modern, smartphone-speed, social-media-amplified panic — depositors move faster and in greater numbers than any peacetime model assumed. This is why treasury does not stop at the regulatory LCR; it runs its own internal "survival days" stress, asking how long the bank lasts under outflows far worse than the rulebook requires.

## The funding stack: what treasury actually manages

Treasury cannot manage liquidity risk without understanding the *quality* of the bank's funding — not all money is equally likely to run. The funding stack is the menu treasury works with, and its composition is the single biggest determinant of how dangerous the bank's liquidity position is.

![Stacked bar of a large bank funding mix deposits wholesale debt and equity](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-8.png)

A typical large bank funds itself with roughly 71% deposits, 10% wholesale and repo borrowing, 7% long-term debt, 4% other liabilities, and about 8% equity. That 8% equity is the loss-absorbing cushion the whole series keeps returning to — it implies leverage of about 1 ÷ 0.08 = 12.5 times. But for liquidity, what matters most is the *stickiness* of the 71% deposits and the 10% wholesale.

Insured retail deposits — a household's checking account, protected up to \$250,000 by the FDIC — are the gold standard of funding: they barely move even in a crisis, because the depositor has no reason to run on insured money. Uninsured corporate deposits and especially wholesale funding (money borrowed from other banks and institutions) are the opposite: fast, rate-sensitive, and the first to flee at the first whiff of trouble. Treasury's craft is to keep the cheap, sticky base large and the flighty wholesale dependence small.

#### Worked example: why funding mix decides survival

Two banks each have \$100 billion of funding and identical \$50 billion of HQLA. Bank A is funded 90% insured retail / 10% wholesale. Bank B is funded 40% insured retail / 60% uninsured-and-wholesale. Apply a stress where insured deposits run at 5% and uninsured/wholesale runs at 50%.

Bank A's outflow: 5% × \$90bn + 50% × \$10bn = \$4.5bn + \$5.0bn = \$9.5 billion — its \$50bn buffer covers that more than five times over. Bank B's outflow: 5% × \$40bn + 50% × \$60bn = \$2.0bn + \$30bn = \$32 billion — its same \$50bn buffer covers it only 1.6 times, and a slightly worse panic breaks it. Same buffer, same size, wildly different survival, all because of funding *mix*. The intuition: the most important thing treasury manages is not how much funding the bank has but how *loyal* it is — a balance sheet funded by flighty money is a balance sheet living on borrowed time.

## How treasury hedges: swaps, term funding, and the two dials

Treasury has, in the end, two big dials to turn. The first dial is **interest-rate hedging**, and the workhorse instrument is the interest-rate swap. A swap is a contract to exchange one stream of interest payments for another — most commonly, paying a fixed rate and receiving a floating rate, or vice versa. A liability-sensitive bank worried about rising rates enters a "pay-fixed, receive-floating" swap: it now *receives* more as rates rise, offsetting the income it loses on its real balance sheet. The swap does not change a single loan or deposit; it bolts a synthetic offset onto the side of the book, converting the bank's effective rate posture without touching customer relationships.

To see why the swap works, walk one through. Say the bank holds \$10 billion of fixed-rate assets yielding 3% and funds them with deposits that reprice. It worries that rates will rise and its funding cost will climb while the 3% asset stays frozen. It enters a \$10 billion pay-fixed swap: it agrees to pay a counterparty a fixed 3.5% and receive a floating rate that tracks the market. If rates rise so the floating leg pays 5.5%, the bank *receives* 5.5% − 3.5% = 2% on \$10 billion = \$200 million from the swap — almost exactly offsetting the extra cost its repricing deposits now demand. The bank has, synthetically, converted its fixed-rate assets into floating-rate ones without selling a single bond. If instead rates *fall*, the swap costs the bank money — but that loss is offset by its now-cheaper funding. Either way, the swap has flattened the bank's exposure to the rate path. That is the whole trick: a swap is a side bet, sized and shaped so its payoff mirrors the loss treasury is trying to neutralize on the real balance sheet.

The second dial is the **funding term**. Treasury chooses how much of its funding is short (cheap but flighty) versus long (more expensive but locked in). Issuing a five-year bond instead of relying on overnight wholesale borrowing costs more in normal times — you pay a term premium — but it cannot run, which is exactly what you want before a storm. The net stable funding ratio (NSFR), the longer-horizon cousin of the LCR, formalizes this: it requires that long-dated assets be backed by funding that is itself reasonably long-dated, so the bank is not financing 30-year mortgages with money that can vanish overnight.

#### Worked example: the cost of buying safety

Suppose treasury wants to replace \$20 billion of overnight wholesale funding (costing 5.0%) with five-year bonds (costing 5.6%) so the money cannot run. The extra cost is the term premium: 5.6% − 5.0% = 0.6% × \$20,000,000,000 = **\$120 million a year**. That \$120 million is the *insurance premium* the bank pays for funding that survives a panic — a direct hit to net interest income in every calendar year, in exchange for not being forced to refinance \$20 billion in the middle of a run.

Was it worth it? If the bank's annual net interest income is \$2.0 billion, the safer funding costs it 6% of income (\$120m ÷ \$2.0bn) every year. That is real money, and a bank under pressure to hit return targets is tempted to skip it and run on the cheaper overnight money — right up until the week the overnight money refuses to roll. The intuition: safety is not free; the term premium and the swap cost are line items, and the perpetual temptation is to under-buy insurance to flatter the margin, which is precisely the temptation that ends banks.

The two dials trade off against each other and against profit. More hedging and more term funding mean a safer bank but a thinner margin — every swap has a cost, every long bond a term premium. Less hedging and more short funding mean a fatter margin and a more fragile bank. ALCO's monthly job is to set where those dials sit: how much rate risk to run, how much liquidity insurance to buy, and therefore how much of the maturity-transformation spread to keep versus how much to pay away for safety. That single set of choices is, more than anything else, what determines whether a bank quietly compounds for decades or dies in a weekend.

## Common misconceptions

**"Treasury is back-office — it just moves money around."** The opposite. Treasury runs the two largest risks on the balance sheet and decides, every month, how big a maturity-transformation bet the bank is making. The choices treasury and ALCO make about rate posture and liquidity buffer are more consequential to the bank's survival than almost anything the lending desks do — a clean loan book cannot save a bank whose treasury left it un-hedged and under-buffered, as SVB proved.

**"Liquidity and solvency are basically the same thing."** They are different risks with different cures, and conflating them is exactly how analysts miss failures. Solvency asks whether assets exceed liabilities over time; liquidity asks whether you can produce cash today. SVB was arguably solvent on a hold-to-maturity basis right up to the run — its bonds would have paid in full if held — but it was fatally illiquid, because it could not turn those bonds into cash fast enough at a price that did not crystallize a balance-sheet-wrecking loss. A solvent bank can die of illiquidity in 36 hours.

**"A high net interest margin means the bank is winning."** Sometimes a fat margin is just an un-hedged rate bet that happened to pay off — a liability-sensitive bank in a cutting cycle, or an asset-sensitive bank in a hiking cycle, will print a great margin until the cycle turns and the same posture flips to a loss. ALCO does not celebrate a high margin; it asks how much rate risk was taken to earn it. A durable margin earned with a hedged book is worth far more than a bigger margin riding an un-sized rate gamble.

**"The repricing gap tells you the rate risk."** The gap is a useful first sketch, but it assumes every deposit reprices fully and instantly, which is the assumption deposit beta exists to correct. A bank can show a frightening negative gap and have modest real rate risk because its deposits are sticky (low beta), or show a comfortable gap and carry huge hidden risk because its depositors will bolt the moment rates rise (high beta). The gap is the map; deposit beta is the terrain.

**"Hedging the rate risk also fixes the liquidity risk."** They are separate dials. A bank can swap away all of its interest-rate exposure and still be one rumor from a run if its funding is flighty and its buffer is thin. SVB's problem was *both* — an un-hedged rate posture *and* a buffer too small for its uniquely flighty, uninsured deposit base. Treasury has to turn both dials; pulling only one is a partial defense.

## How it shows up in real banks

**Silicon Valley Bank, March 2023 — the cockpit left empty.** SVB is the textbook case of every concept in this post failing at once. The bank had loaded up on long-dated Treasuries and mortgage bonds during the zero-rate years, making itself profoundly liability-sensitive into the fastest hiking cycle in forty years. As rates rose, those bonds lost market value — about \$17 billion of unrealized losses against a balance sheet of roughly \$209 billion. On NII it looked survivable; on EVE it was a disaster, but SVB had reportedly removed its interest-rate hedges to flatter near-term income, and its board had operated for months without a chief risk officer. When 94% of deposits turned out to be uninsured, concentrated, and venture-backed — the flightiest funding imaginable — a \$42 billion outflow on March 9 forced the bank to sell bonds at a loss, which crystallized the EVE hole, which triggered the next wave. Treasury's two dials had both been left in the wrong position, and the bank failed in 36 hours. It is the cleanest illustration ever recorded of what happens when the balance-sheet cockpit is unmanned.

**US banks and the 2022–2024 hiking cycle — asset sensitivity paying off.** The aggregate story ran the other way for most of the industry. Going into 2022, large US banks were broadly asset-sensitive, with floating-rate loans and a vast base of sticky, low-beta deposits. When the Fed hiked over five percentage points, their asset yields climbed while deposit costs lagged — exactly the asset-sensitive windfall. Industry net interest margin jumped from its zero-rate trough of about 2.56% in 2021 to roughly 3.30% in 2023. As the cycle matured and deposit betas crept from 0.10 toward 0.55, some of that windfall gave back, and margin eased to about 3.23% in 2024. The whole arc is treasury and ALCO decisions visible in the aggregate data: who was asset-sensitive, how sticky their deposits were, and how fast betas caught up.

![Line chart of US bank net interest margin from 2010 to 2024 with the trough and recovery marked](/imgs/blogs/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit-5.png)

**Credit Suisse, 2023 — when liquidity follows trust, not ratios.** Credit Suisse spent its final months with liquidity ratios that, on paper, looked compliant — and still suffered roughly CHF 110 billion of outflows in the fourth quarter of 2022 alone as confidence evaporated after years of scandals. It drew on a CHF 100 billion liquidity line from the Swiss National Bank and still could not stem the bleed, ending in a shotgun sale to UBS for about CHF 3 billion and a CHF 16 billion wipeout of its AT1 bonds. The lesson for treasury is sobering: the LCR is calibrated to a model of how depositors behave, and when *trust* breaks, behavior leaves the model behind. A buffer sized for a normal stress is no defense against a franchise-trust spiral.

**The 1980s savings-and-loan crisis — the original duration trap.** Decades before SVB, the US savings-and-loan industry ran the same un-hedged liability-sensitive bet at industry scale: thousands of thrifts funded long-dated, fixed-rate 30-year mortgages with short-term deposits. When Paul Volcker pushed rates toward 20% to break inflation, their funding costs exploded while their mortgage income stayed frozen at old low rates — the repricing gap turned lethal. More than a thousand thrifts failed, and the cleanup cost taxpayers on the order of \$124 billion. The S&L crisis is the historical proof that interest-rate risk in the banking book, left un-steered, is not a footnote — it can take down an entire sector. It is the reason IRRBB management and ALCO exist in the form they do today.

**2023 — the slow deposit migration that squeezed everyone.** Even banks that did not fail felt treasury pressure all year, through a quieter channel: deposit migration. As the policy rate climbed above 5% while many banks still paid almost nothing on checking accounts, depositors finally noticed the gap and moved cash into money-market funds and higher-paying online accounts. This is deposit beta arriving with a lag — and it forced treasuries across the industry to raise deposit rates to defend their funding base, compressing margins. Banks watched their cheap, sticky deposits slowly turn into expensive, rate-sensitive ones, exactly the dynamic the deposit-beta chart captures as the cumulative figure climbed past 0.50. No headline, no run, just a steady tax on the spread — and a reminder that the most valuable thing on a bank's balance sheet, its cheap deposit franchise, is also the thing that quietly erodes when treasury is slow to defend it.

**The everyday case — the monthly ALCO meeting.** Most of the time, none of this is dramatic. In a healthy bank, treasury arrives at the monthly ALCO with a gap report, an NII-sensitivity table, an EVE table, the latest deposit-beta estimates, and the LCR and NSFR. The committee looks at where the dials sit, checks every number against its limits, and makes small adjustments — add a few billion of swaps here, term out some funding there, nudge the buffer up before a known stress. The drama only comes when a bank lets the posture drift unmanaged between those meetings, or stops holding the meetings with teeth. The boring monthly discipline *is* the risk management; the failures are what happen in its absence.

## The takeaway / How to use this

If you want to read a bank's safety the way its own treasurer does, learn to ask the two questions ALCO asks every month, and ignore almost everything else.

First: **what is its rate posture, and is it on purpose?** Find the bank's NII sensitivity disclosure (it is in the annual report and the Pillar 3 filings) and look at how 12-month income moves under a ±1% or ±2% shock. A modest, bounded number means treasury is steering. A large number — especially one the bank does not appear to be hedging — means the bank is running a rate bet, and you should ask whether management chose it or drifted into it. Then check the EVE sensitivity alongside it: if NII looks calm but EVE shows a deep hole under rising rates, the bank is sitting on unrealized losses in long bonds — the SVB signature — and is far more fragile than its income statement admits.

Second: **how loyal is its funding, and how big is its buffer?** Look at the share of deposits that are insured versus uninsured, the dependence on wholesale funding, and the LCR. A bank funded by sticky, insured retail deposits with a fat HQLA buffer can survive a panic; a bank funded by uninsured corporate cash and overnight wholesale borrowing is living on the assumption that the panic never comes. The buffer's job is to make the bank's survival independent of whether depositors keep their nerve — and that is the difference between a bank that endures a scary week and one that does not see the following Monday.

Both questions trace straight back to the spine of this series. A bank is a leveraged, confidence-funded maturity-transformation machine, and treasury is the function that decides — consciously, with limits and hedges — exactly how much of that fragile trade to run. When the cockpit is staffed, the bank earns its spread quietly through every cycle and you never read about it. When the cockpit is empty, the same balance sheet that looked fine on Friday is gone by Sunday. The whole art of banking, in the end, is the active management of a mismatch that can never be fully closed — and that management has a name, an address, and a committee. It is treasury.

## Further reading & cross-links

- [Net Interest Margin and the Spread Business, Explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the margin that ALM exists to protect, and how it breathes through the rate cycle.
- [Interest-Rate Risk in the Banking Book (IRRBB) and the Duration Gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the deeper mechanics of the rate-risk dial, including EVE and the exact mechanism that broke SVB.
- [Liquidity Management: LCR, NSFR and the Liquidity Buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the full Basel liquidity rulebook and how the buffer is sized against a 30-day stress.
- [The Funding Stack: Deposits, Wholesale Funding, Bonds and Covered Bonds](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds) — the menu of funding treasury manages and the stickiness of each layer.
- [SVB and Credit Suisse 2023: The Bank Runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level one-pager on the 2023 failures this post dissects from the treasury's side.

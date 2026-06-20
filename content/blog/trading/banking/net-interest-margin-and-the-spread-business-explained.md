---
title: "Net Interest Margin and the Spread Business, Explained"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What a bank earns on its assets minus what it pays on its funding, divided by earning assets — and what makes that margin widen or narrow through a whole rate cycle."
tags: ["banking", "net-interest-margin", "nim", "deposit-beta", "asset-liability-management", "spread", "interest-rate-risk", "cost-of-funds", "repricing-gap"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank's net interest margin (NIM) is the spread between what it earns on its assets and what it pays on its funding, divided by its earning assets. It is the single number that tells you how much the bank's core lending machine is making per dollar it has lent or invested.
>
> - NIM = net interest income ÷ average earning assets. Earn 5.20% on assets, pay an effective 1.65% on funding, and you land around a 3.3% margin — the engine of bank profit.
> - The margin moves through the rate cycle because assets and deposits reprice at *different speeds*. The gap between those speeds — captured by *deposit beta* and the *repricing gap* — is what widens or narrows NIM.
> - A bank whose assets reprice faster than its funding is *asset-sensitive*: rising rates are a windfall, falling rates a squeeze. The opposite bank is *liability-sensitive*.
> - The one number to remember: US bank NIM bottomed near **2.56% in 2021** under zero rates and jumped to about **3.30% in 2023** as the Fed hiked — a roughly three-quarter-point swing worth tens of billions of dollars, all from the spread.

Why does a bank pay you almost nothing on your checking account but charge 7% on a car loan? That gap — not fees, not trading, not clever financial engineering — is where the overwhelming majority of a normal bank's money comes from. And the size of that gap is not fixed. It breathes in and out with the level of interest rates, with how fast loans and bonds reset their rates, and with how stubbornly depositors refuse to demand more for their savings.

In 2021, when the Federal Reserve had pinned its policy rate near zero for the better part of two years, the average US commercial bank earned a net interest margin of about 2.56% — the lowest in the modern record. Banks were drowning in deposits they could barely lend out profitably; the spread had been squeezed almost flat. Then the Fed hiked rates by more than five percentage points in eighteen months, the fastest tightening in forty years. By 2023, the industry's net interest margin had jumped to roughly 3.30%. That swing of about three-quarters of a percentage point sounds tiny. On the roughly \$24 trillion of assets the US banking system carries, it is worth on the order of \$150 billion a year in extra income. Same banks, same customers, same loans — a completely different profit, driven entirely by the spread business.

The diagram below is the mental model for this whole post: a bank earns a *yield* on its assets, pays a *cost* on its funding, the difference is the *spread*, and the spread earned across the whole asset book, divided by those assets, is the net interest margin. Everything else in this article is a deeper look at one of those four boxes and what makes the numbers inside them move.

![Pipeline showing asset yield minus cost of funds equals spread leading to net interest margin](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-1.png)

This connects directly to the spine of this whole series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short (deposits) and lends long (loans), and earns the spread between the two. Net interest margin is the *measurement* of that spread — the vital sign that tells you whether the core trade is healthy, anemic, or being squeezed to death. Read it right and you can see, before the headlines do, whether a bank's engine is gaining power or stalling.

## Foundations: yield, cost of funds, spread, and the margin

Let's build every piece from zero. By the end of this section you'll be able to compute a net interest margin from scratch and explain why it is *not* the same thing as the spread.

### Earning assets: the part of the bank that works for a living

A bank's balance sheet has assets (things it owns and lends out) and liabilities (money it owes). On the asset side, not everything earns interest. The branch buildings, the computer systems, and the goodwill from past acquisitions sit on the balance sheet but pay no interest. The parts that *do* earn interest are the **earning assets**: loans (mortgages, credit cards, business loans), securities (mostly government and mortgage bonds), interbank deposits, and cash parked at the central bank that earns interest on reserves.

A *basis point* is one hundredth of a percent — 0.01%. When you hear "rates rose 50 bps", that means half a percentage point. We'll use basis points and percentage points interchangeably, glossing as we go.

For a typical large US bank, earning assets are roughly 85-90% of total assets. The rest — premises, intangibles, and some operational cash — is "dead weight" that the earning assets have to carry. That is the first subtle point about NIM: it is measured against *earning* assets, not total assets, precisely so we're judging the productive core, not punishing a bank for owning a building.

### Yield on earning assets: what the asset side earns

The **yield on earning assets** is the interest income the bank collects, expressed as a percent of the average earning assets that produced it. If a bank holds \$100 billion of earning assets and collects \$5.2 billion of interest over a year, its asset yield is 5.2%.

Crucially, this is a *blended* yield. A bank holds a mix: credit cards yielding 18%, mortgages at 6%, Treasury bonds bought in 2021 at 1.5%, cash at the Fed earning the policy rate. The blended yield is the dollar-weighted average across all of it. A bank can have a high blended yield because it lends to risky borrowers at fat rates, or a low one because it's stuffed with old low-coupon bonds. The composition matters enormously — and it's one of the levers we'll return to.

### Cost of funds: what the liability side pays

The **cost of funds** is the interest the bank pays on its funding, as a percent of that funding. A bank funds its assets with deposits (checking, savings, certificates of deposit), borrowings (bonds it issues, money borrowed from other banks, repo), and a thin sliver of equity (shareholders' money, which pays no interest).

Deposits are the cheap, sticky core. A checking account often pays 0%; a basic savings account might pay 0.5% even when the Fed funds rate is 5%. Certificates of deposit (CDs — deposits locked up for a fixed term) pay more, maybe 4-5%, because the customer gives up access to the money. Bonds the bank issues to professional investors pay a market rate close to the policy rate plus a credit spread. So the *blended* cost of funds depends heavily on the **funding mix**: a bank full of free checking deposits has a much lower cost of funds than one relying on CDs and bond markets.

For US banks across the cycle, the funding mix runs roughly 74% deposits, 11% wholesale and repo, 7% long-term debt, 4% other liabilities, and 4% equity. The dominance of deposits — and how cheap they stay — is the whole franchise. (Why cheap deposits are *the* prize is the subject of [the retail deposits post](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).)

### The spread vs the net interest margin: not the same thing

Here is where beginners trip. The **spread** is simply asset yield minus cost of funds: 5.2% − 1.65% = 3.55 percentage points. It's the headline gap between the two rates.

The **net interest margin** is *net interest income* (NII — the actual dollars of interest earned minus interest paid) divided by *average earning assets*. In symbols:

$$\text{NIM} = \frac{\text{Interest income} - \text{Interest expense}}{\text{Average earning assets}} = \frac{\text{NII}}{\text{Avg earning assets}}$$

Why aren't these identical? Because the bank funds its earning assets with more than just interest-bearing money. A chunk of the asset book is funded by non-interest-bearing deposits (free checking) and by equity — money that costs nothing in interest. That free funding means the bank earns the *full* asset yield on those dollars while paying nothing, which pulls NIM *above* the raw spread.

Walk it through. Suppose the bank has \$100 of earning assets yielding 5.20%, so interest income is \$5.20. It funds those assets with \$90 of interest-bearing money at 1.65% (interest expense \$1.485) and \$10 of free money (zero-cost deposits plus equity). Net interest income is \$5.20 − \$1.485 = \$3.715. Divide by \$100 of earning assets: NIM = 3.72%. That's *higher* than the 3.55-point spread, because of the free funding. This free-funding kicker is why a deposit-rich bank's NIM beats a wholesale-funded bank's even at the same headline spread.

For the rest of this post we'll often use the simpler approximation — NIM ≈ spread — when the point is about *direction* (what makes the margin rise or fall), and the exact formula when we want the precise number. Both are correct; just know which one you're reading.

#### Worked example: computing a NIM from a yield and a cost of funds

You run a small bank. Over the year you hold an average of \$10 billion of earning assets. You collect \$520 million of interest income, so your yield on earning assets is \$520m ÷ \$10,000m = **5.20%**. You pay \$170 million of interest on your deposits and borrowings, so on the \$9 billion of interest-bearing funding that backs those assets, your cost of funds is \$170m ÷ \$9,000m ≈ **1.89%**. (The other \$1 billion of funding is free checking and equity.)

Your net interest income is \$520m − \$170m = **\$350 million**. Your net interest margin is \$350m ÷ \$10,000m = **3.50%**.

Notice the spread by itself is 5.20% − 1.89% = 3.31 points, but the NIM is 3.50% — higher, because \$1 billion of your assets is funded for free. The intuition: net interest margin rewards a bank twice for cheap funding — once through a lower cost of funds, and again through the free-funding lift that pushes NIM above the raw spread.

### Deposit beta: how fast funding chases the policy rate

When the central bank raises its policy rate, the rate a bank *could* pay on funding rises. But banks don't pass the full increase to depositors immediately — savers are slow to demand more, and switching banks is a hassle. **Deposit beta** measures how much of a rate move the bank actually passes through to its deposit rates.

If the Fed raises rates by 2 percentage points and the bank's average deposit rate rises by 1 point, the deposit beta is 1 ÷ 2 = 0.50, or 50%. A beta of 0 means deposit costs don't move at all (great for the bank); a beta of 1 means deposits track the policy rate point for point (terrible for the bank's margin). Deposit beta is *the* swing variable that decides how much of a rate hike a bank gets to keep.

### Asset repricing and the repricing gap

The asset side reprices too, and at its own speed. A floating-rate business loan tied to a benchmark reprices within days of a Fed move. A 30-year fixed mortgage doesn't reprice at all until it's refinanced or paid off — its rate is locked. A Treasury bond bought in 2021 keeps paying its old 1.5% coupon until it matures.

The **repricing gap** is the difference, over a given horizon, between how many of the bank's assets will reset their rate and how many of its liabilities will. If more assets reprice than liabilities in the next year, the bank is **asset-sensitive**: its income rises faster than its cost when rates go up. If more liabilities reprice, it's **liability-sensitive**: its cost rises faster than its income when rates go up.

There's a wrinkle in the word "reprice" that matters for deposits. A floating-rate loan reprices *contractually* — its rate is tied to a benchmark and resets on a schedule, whether the borrower likes it or not. A deposit reprices *behaviorally* — there's no contract forcing a checking account's rate up; the bank chooses what to pay, constrained only by competition and the fear that depositors will leave. This is why deposits are modeled with a *beta* (a behavioral pass-through) rather than a fixed reset date. Risk managers go further and assign deposits an "effective duration" — a checking balance that has sat untouched for years behaves, for repricing purposes, like a multi-year fixed-rate liability, even though contractually the depositor could withdraw it tomorrow. Getting that behavioral assumption right is half the art of measuring a bank's rate sensitivity, and getting it *wrong* is how banks are surprised when supposedly sticky deposits suddenly run.

That single distinction — asset-sensitive vs liability-sensitive — determines whether a rate hike is a gift or a punishment. We'll spend a whole section on it.

## Net interest income is the bank's heartbeat

Before going deeper into what moves NIM, anchor on why it matters so much. For a traditional commercial bank, net interest income is typically 60-75% of total revenue. Fees (account charges, card interchange, wealth management) make up most of the rest, and trading is small or zero for all but the biggest investment banks. (The full breakdown of how a bank books revenue and where provisions hit is in [the income statement post](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).)

So when NIM moves, the bank's whole profit moves with it — and it moves with leverage. A bank earning a 3.3% NIM on assets, against an equity base of roughly 10% of assets, is earning that spread on ten dollars of assets for every dollar of its own equity. A small change in the margin, multiplied by that leverage, is a large change in return on equity. This is why bank-stock investors are obsessed with NIM guidance: a 10 basis-point change in NIM, multiplied through, can move earnings per share by several percent.

Put the leverage in numbers. Take a bank with \$100 of assets funded by \$90 of liabilities and \$10 of equity — roughly 10x leverage, the normal shape of a bank. If its NIM rises by just 10 basis points (0.10%), that's \$0.10 of extra net interest income on \$100 of assets. But measured against the \$10 of equity that the shareholders actually put up, that \$0.10 is a full percentage point of additional pre-tax return on equity. The leverage that makes a bank fragile on the downside is the same leverage that turns a sliver of extra margin into a meaningful jump in shareholder return on the upside. NIM is small; multiplied by ten and read against a thin equity base, its movements are anything but.

The chart below is the headline picture — US bank NIM across a full cycle. Notice the long grind down through the 2010s low-rate era, the collapse to a 2.56% trough in 2021, and the sharp recovery to 3.30% by 2023 once the Fed hiked.

![US commercial bank net interest margin from 2010 to 2024 with trough and jump marked](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-2.png)

Read that chart as a story about the spine. The 2010s were a decade of cheap money: rates were low, so the *asset* yield a bank could earn was low, and there was a floor on how low deposit costs could go (you can't pay much less than zero). Squeezed from above by low asset yields and unable to push funding costs below zero, the spread compressed year after year. The 2021 trough is what *maximum* maturity-transformation stress looks like when rates are at zero: the bank is doing its job — borrowing short, lending long — but the job barely pays. The 2023 jump is the same machine suddenly given room to breathe.

## What actually moves NIM through a rate cycle

Now we get to the heart of it. NIM moves because the asset side and the funding side respond to rate changes at *different speeds and to different degrees*. Four forces dominate: the direction of rates, deposit beta, the asset and funding mix, and the shape of the yield curve.

### Force 1: the level and direction of rates

The single biggest driver of NIM over a cycle is the level of short-term rates, because of that free-funding kicker. Remember: a chunk of a bank's assets is funded by zero-cost deposits and equity. When rates are at zero, those free dollars earn nothing — the bank's "endowment" of free funding is worthless. When the policy rate is 5%, those same free dollars suddenly earn 5% at essentially no cost. This is why higher rates, all else equal, lift NIM: the free funding gets monetized.

That's the mechanism behind the chart you just saw. The dual-axis chart below makes the link explicit, plotting bank NIM against the Fed funds rate. NIM tracks the policy rate — with a lag, because deposit costs catch up slowly — but the relationship is unmistakable.

![Dual axis chart of bank NIM versus the Fed funds upper bound from 2019 to 2024](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-8.png)

But notice the lag and the imperfect tracking. NIM doesn't jump the instant the Fed moves, and it doesn't rise one-for-one. That's because the other three forces — beta, mix, and the curve — modulate how much of the rate move actually reaches the margin. The Fed sets the tide; these forces set how high the boat actually floats.

It's worth being precise about *why* the free-funding kicker exists, because it's the most counterintuitive part of the rate story. Think of the bank as having two pots of money funding its assets. One pot is interest-bearing — deposits that pay something, plus bonds the bank has issued. The other pot is free: non-interest-bearing checking accounts and the shareholders' equity. On the interest-bearing pot, a rate rise lifts both the income (from the assets that pot funds) and the cost (the interest the bank now pays), so the two largely offset — only the beta gap survives. But on the *free* pot, a rate rise lifts the income with no offsetting cost at all, because the funding was free to begin with and stays free. That asymmetry is pure margin. The bigger a bank's free pot — its non-interest deposits and its equity — the more its NIM benefits from every point of rate increase. This is also why the free-funding kicker was worth nothing in 2021: when the policy rate is zero, even free money earns zero, so the free pot contributes no margin. As rates climbed past 5%, that same free pot suddenly threw off 5% of pure spread. A bank with \$200 billion of free funding earns roughly \$10 billion a year on it at a 5% rate, and zero at a 0% rate — same deposits, same balance sheet, a \$10 billion swing in income from the level of rates alone.

There is a ceiling on the benefit, though. Push rates high enough for long enough and three things start to bite: deposit betas climb toward 1 as savers finally chase yield, depositors move money out of the bank entirely into Treasury bills and money-market funds (shrinking that precious free pot), and loan demand and credit quality deteriorate as borrowers struggle with higher payments. So NIM's relationship with rates is not a straight line up forever — it's a rise that fades and can reverse as the cycle matures. The early innings of a hiking cycle are the sweet spot; the late innings give some of it back.

### Force 2: deposit beta, the swing variable

Deposit beta is where a bank's franchise quality shows up most directly in the margin. The lower a bank's deposit beta, the more of a rate hike it gets to keep as wider spread.

Early in a hiking cycle, deposit betas are low — depositors are slow to notice rates have risen, and banks are in no hurry to raise what they pay. As the cycle matures, savers wake up, money flows toward higher-yielding accounts and money-market funds, and competition forces betas up. The chart below shows the *cumulative* deposit beta climbing through the 2022-24 cycle: from about 0.10 early on to roughly 0.55 by 2024 as deposits finally caught up.

![Step chart of cumulative deposit beta rising through the 2022 to 2024 hiking cycle](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-3.png)

This rising-beta dynamic is exactly why NIM in a hiking cycle tends to *expand first, then fade*. In the first year, assets reprice up while deposit costs lag — NIM widens fast. In the second and third years, deposit betas climb, funding costs catch up, and the margin gives some of the early gain back. The 2023 NIM print of 3.30% was the peak of that dynamic; the slight pullback to 3.23% in 2024 is deposit beta doing its slow work.

#### Worked example: a deposit beta of 0.5 on a 2-point hike

Your bank starts with a yield on earning assets of 4.00% and an average deposit cost of 0.50%. Your interest-bearing funding equals 90% of your earning assets (the other 10% is free checking and equity). Pre-hike, your net interest margin is approximately:

NIM = 4.00% − (0.50% × 0.90) = 4.00% − 0.45% = **3.55%**.

Now the Fed hikes by 2.00 points. Your assets are 90% sensitive (asset beta 0.90 — most of your loans and bonds eventually reprice), so your asset yield rises by 0.90 × 2.00 = 1.80 points, to 5.80%. Your *deposit* beta is 0.50, so your deposit cost rises by 0.50 × 2.00 = 1.00 point, to 1.50%. New NIM:

NIM = 5.80% − (1.50% × 0.90) = 5.80% − 1.35% = **4.45%**.

Your margin widened by 0.90 points — from 3.55% to 4.45% — purely because your assets repriced up by 1.80 points while your funding cost rose by only 1.00 point. The intuition: with a deposit beta below 1, every rate hike hands the bank a temporary windfall, and the lower the beta, the bigger the gift.

The chart below sweeps that same stylized bank across every possible deposit beta after a 2-point hike. At beta 0, NIM balloons to 5.35%; at beta 0.5 it's 4.45% (the case we just did); at beta 1.0 — full pass-through — NIM falls all the way back to the pre-hike 3.55%, as if the hike never happened for the margin. The downward slope *is* the value of a sticky, low-beta deposit base.

![Line chart of NIM after a two point hike as deposit beta rises from zero to one](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-5.png)

That chart is one of the most important pictures in banking. The entire competitive advantage of a great deposit franchise — a Bank of America, a Wells Fargo, a JPMorgan with tens of millions of sticky checking accounts — is the ability to sit at the *left* of that curve, with a low deposit beta, while competitors with hot, rate-shopping deposits are dragged to the right. Two banks can hold identical loans and identical bonds; the one with the better deposit base simply keeps more of every rate hike.

There's an asymmetry worth naming: deposit beta is usually higher on the way *up* than on the way *down*. When rates rise, depositors eventually demand more and betas climb; but when rates fall, banks are quick to cut what they pay while depositors are slow to accept it — nobody volunteers to earn less. This "sticky-up, slow-down" behavior on the funding side is a quiet friend of NIM at the start of a cutting cycle (deposit costs fall before they have to) but it's bounded, because betas can't fall below zero and deposit rates are already near a floor. Across a full cycle, the bank that wins on margin is the one whose depositors are slow to demand more when rates rise *and* content to accept less when rates fall — which is exactly the behavior of a customer who is there for the relationship, not the rate.

#### Worked example: when assets reprice faster than deposits

Let's isolate the *speed* effect, in dollars, over a single year. Your bank has \$10 billion of earning assets. Half of them — \$5 billion — are floating-rate loans that reprice immediately when the Fed moves. The other \$5 billion are fixed-rate (old mortgages and bonds) and don't reprice this year at all. On the funding side, you have \$9 billion of deposits; assume your deposit beta this year is just 0.30 (deposits are slow), so when the Fed hikes 2.00 points, your deposit cost rises 0.60 points across the whole \$9 billion.

Extra interest income from the asset side: the \$5 billion of floating loans reprice up by the full 2.00 points → +\$5,000m × 2.00% = **+\$100 million**. The fixed \$5 billion adds nothing this year.

Extra interest expense from the funding side: \$9,000m × 0.60% = **+\$54 million**.

Net change in NII: +\$100m − \$54m = **+\$46 million** of extra net interest income, in one year, from a single hike. On \$10 billion of earning assets, that's +0.46 points of NIM. The intuition: the margin windfall is exactly the gap between how fast your assets reprice up and how slowly your deposits chase them — asset repricing speed times the rate move, minus deposit beta times the rate move, all on their respective balances.

### Force 3: the asset and funding mix

The *composition* of the balance sheet sets the starting point for all of this. On the asset side, a bank tilted toward high-yield consumer loans (credit cards at 18%, auto at 8%) runs a structurally higher NIM than one stuffed with low-yield government bonds — but it also takes more credit risk, so part of that fatter margin is just compensation for expected loan losses. On the funding side, a bank with a high share of free checking and savings (a high CASA ratio — current and savings accounts as a share of deposits) has a structurally lower cost of funds than one leaning on CDs and wholesale borrowing.

Mix also creates traps. A bank that loaded up on long-dated, low-coupon bonds in 2020-21 — when it had excess deposits and nowhere to lend them — locked in a 1.5% asset yield for years. When rates jumped, that bank couldn't reprice those assets up; it was stuck earning 1.5% while its deposit costs rose. That is precisely the trap that broke Silicon Valley Bank, and we'll come back to it.

The mix question has a time dimension that's easy to miss. The *current* NIM reflects the asset book the bank built up over years, not the rates available today. A bank that originated a wall of 3% mortgages in 2021 will carry those low-yield assets for a long time — mortgages prepay slowly, and a homeowner with a 3% loan has every reason never to refinance. So even as new loans get written at 7%, the bank's *blended* asset yield rises only gradually, dragged down by the old book. This is the "back-book versus front-book" problem: the back book (existing loans) reprices slowly toward the front book (new loans) only as old loans mature or roll off. A bank with a fast-rolling book (credit cards, floating commercial loans) sees its asset yield catch up to new rates within a year or two; a bank loaded with 30-year fixed mortgages can take a decade. That difference in *book speed* is a structural determinant of how asset-sensitive a bank really is, separate from any hedging it does.

There's a deposit-mix mirror image of the same dynamic. A bank's funding mix doesn't just sit still through a cycle — it actively *deteriorates* as rates rise, in a process bankers call deposit migration. When rates are low, customers leave money in free checking because the alternative pays nothing. When rates climb, those same customers shift balances out of free checking into interest-bearing savings, then into CDs, then out of the bank entirely into money-market funds. Each step up that ladder raises the bank's cost of funds even if the *rate on each product* hasn't changed, simply because the *weighting* has shifted toward the costlier products. Deposit migration is a hidden tax on NIM that compounds the rise in deposit beta — and it's why a bank's realized cost of funds in the late innings of a hiking cycle can climb faster than its posted rates alone would suggest.

### Force 4: the shape of the yield curve

Because a bank borrows short and lends long, the *slope* of the yield curve — the difference between short-term and long-term rates — matters on top of the level. When the curve is steep (long rates well above short rates), the maturity-transformation trade is sweet: the bank funds at low short rates and earns high long rates, a wide spread baked into the curve itself. When the curve is flat or *inverted* (short rates above long rates, as in 2022-23), that built-in spread vanishes or goes negative, squeezing NIM on the margin even as the level of rates helps via the free-funding kicker. (For how and why the curve takes its shape, see [the yield curve in the fixed-income series](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance).)

This is the one force that worked *against* banks in 2022-23, which is why the NIM recovery, though large, was not as large as the level of rates alone would have predicted. The Fed pushed short rates above 5% while the 10-year Treasury sat well below that — a deeply inverted curve. For a bank that wanted to add new long-term assets (a 10-year fixed loan, say), the math was ugly: it would be funding at 5%-plus short rates to buy assets yielding less. The inversion punished new maturity-transformation even as the high level of short rates rewarded the existing free-funding base. The two effects pulled in opposite directions, and the net was a healthy but not euphoric NIM. The clean lesson: the *level* of rates monetizes a bank's existing free funding, but the *slope* of the curve determines whether adding new long assets is profitable. A bank reads both — and in an inverted curve, the smart move is often to keep new lending short and floating, not to lock in long assets at a negative carry.

There's a subtlety in how the slope reaches different parts of the book. The short end of the curve sets the bank's funding cost and the yield on its floating loans and its cash. The long end sets the yield on new fixed-rate loans and the bonds the bank buys for its securities portfolio. So a bank's NIM is sensitive to *which part* of the curve moves, not just to a single rate. A "bear flattener" — short rates rising while long rates stay put — squeezes a bank that funds short and holds long fixed assets, because its cost rises while its asset yield doesn't. A "bull steepener" — short rates falling while long rates hold — is a gift, because funding cheapens while long assets keep paying. Reading NIM well means reading the curve as a shape that's changing, not a single number going up or down.

The matrix below collects all of this into one reference: the levers that widen NIM on the left, the same levers turning against the bank on the right.

![Matrix of what widens versus what narrows net interest margin across six levers](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-6.png)

## Asset-sensitive vs liability-sensitive: the same balance sheet, two fates

This is the concept that ties everything together, so let's make it crisp. A bank is **asset-sensitive** if, over a given horizon, more of its assets reprice than its liabilities — so when rates rise, its income rises faster than its cost, and NIM widens. A bank is **liability-sensitive** if the reverse is true — more liabilities reprice than assets — so when rates rise, its cost rises faster than its income, and NIM narrows.

Most commercial banks are at least modestly asset-sensitive by design, and for a structural reason: their deposits behave as if they reprice very slowly (sticky checking and savings) while a good slice of their loans float. That makes the typical bank a *friend of higher rates and an enemy of cuts*. The before-and-after diagram below shows the same asset-sensitive bank in the two halves of a cycle.

![Before and after diagram of an asset sensitive bank in a cutting cycle versus a hiking cycle](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-4.png)

In a hiking cycle, the asset-sensitive bank's loans reprice up quickly while its deposit costs lag — spread widens, NIM jumps. In a cutting cycle, its loans reprice *down* quickly while its deposit costs, already near zero, can't fall much further — spread compresses, NIM falls. This asymmetry is why bank earnings tend to swell when the Fed is hiking and sag when it's cutting, and why bank-stock analysts watch the rate outlook as closely as they watch loan growth.

The graph below traces the exact transmission: a Fed move splits into two channels — the fast asset channel and the slow funding channel — and the gap between them lands on NIM.

![Graph showing a Fed rate move reaching NIM through fast asset repricing and slow deposit repricing](/imgs/blogs/net-interest-margin-and-the-spread-business-explained-7.png)

You don't have to guess whether a bank is asset- or liability-sensitive — every large bank discloses it. In their quarterly filings, banks publish a "net interest income sensitivity" table: how much their projected NII would change over the next twelve months given a +100 basis-point or −100 basis-point parallel shift in rates. A bank that reports, say, "+100 bps → +\$1.5 billion of NII" is telling you plainly that it's asset-sensitive, and by how much. Reading these tables across a few banks side by side is the fastest way to see who's positioned for higher rates and who's positioned for lower. The caveat: these projections rely on the bank's *own* deposit-beta and deposit-duration assumptions, which are exactly the assumptions that proved too optimistic for several banks in 2022-23. A bank can show modest, comfortable rate sensitivity on paper precisely because it assumed its deposits were stickier than they turned out to be.

There's a deep lesson hiding in that diagram about *which kind* of rate risk a bank is running. Being asset-sensitive feels great in a hiking cycle — until you realize you've made a bet on the direction of rates with your core franchise. A bank that gorges on asset-sensitivity to juice NIM in a hiking cycle has, by construction, set itself up to suffer when rates fall. And a bank that fights its natural asset-sensitivity by piling into long, fixed-rate bonds (to lock in yield) has converted a margin problem into a *capital* problem — because when rates rise, those bonds lose market value even as the rest of the bank's NIM improves. Managing that trade-off is the entire job of [interest-rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap), which is where the SVB story really lives.

It's worth dwelling on why most banks lean asset-sensitive in the first place, because it's not an accident — it's the natural shape of the business reinforced by a deliberate choice. The natural part: a typical bank's biggest liability is non-maturity deposits (checking and savings) that behave as if they barely reprice, while a meaningful slice of its assets float. That alone tilts it asset-sensitive. The deliberate part: banks generally *like* being asset-sensitive, because historically rate-cutting cycles have coincided with recessions, when the bank also faces rising loan losses. Being asset-sensitive means NIM rises when the economy is strong (and rates are climbing) and falls when the economy weakens (and rates are cut) — so the rate exposure is, in effect, a bet that's correlated with the rest of the bank's fortunes in a way management finds tolerable. The danger is overdoing it, or being asset-sensitive on the *income* statement while being dangerously long-duration on the *balance sheet* — the combination that turns a margin tailwind into a solvency event.

## A note on what NIM does and doesn't tell you

NIM is a powerful number, but it is a *gross* margin, not a profit. Three things stand between a healthy NIM and actual shareholder profit, and missing them is how people misread a bank.

First, **credit losses**. A bank earning a fat 5% NIM on subprime credit cards is not necessarily more profitable than a bank earning 3% on prime mortgages — because the subprime book will charge off a chunk of those loans to default. The right way to think about it is *risk-adjusted* margin: NIM minus the expected loss rate. A high NIM that exists only because the bank lends to risky borrowers is partly an illusion; the provisions for bad loans will claw it back.

#### Worked example: risk-adjusted margin, or why a fatter NIM can be the worse business

Compare two banks, each with \$10 billion of earning assets.

Bank A is a prime mortgage lender. It earns a yield of 4.30% and pays a cost of funds of 1.00% on funding equal to 90% of its assets, for a NIM of 4.30% − (1.00% × 0.90) = 4.30% − 0.90% = **3.40%**. Its loans are safe: expected annual credit losses are just 0.20% of assets.

Bank B is a subprime consumer lender. It earns a much higher yield of 9.00% and pays a cost of funds of 2.50% (it has to pay up for funding because investors see it as risky), again on 90% funding, for a NIM of 9.00% − (2.50% × 0.90) = 9.00% − 2.25% = **6.75%**. Its NIM is nearly double Bank A's. But its borrowers default: expected annual credit losses are 5.50% of assets.

Now subtract the losses to get the *risk-adjusted* margin. Bank A: 3.40% − 0.20% = **3.20%**. Bank B: 6.75% − 5.50% = **1.25%**. The bank with the gorgeous 6.75% NIM actually keeps far *less* per dollar of assets once you account for the loans that go bad — \$125 million of risk-adjusted income on \$10 billion, versus Bank A's \$320 million. The intuition: a wide NIM earned by lending to risky borrowers is mostly rented from the credit cycle, and the provisions come to collect — always read margin net of expected losses, never gross.

Second, **operating costs**. NIM is income before the bank pays its staff, runs its branches, and maintains its technology. A bank with a wide NIM but a bloated cost base can still earn a poor return. (The metric for that is the efficiency ratio, covered in the income-statement post.)

Third, **the asset on which it's measured**. Because NIM is income over *earning* assets, a bank can flatter its NIM by shrinking low-yield assets (selling off cash and bonds) so that the high-yield loans dominate the average. The margin goes up, but the absolute dollars of net interest income may not. Always read NIM alongside the dollar figure for net interest income and the size of the balance sheet — a rising margin on a shrinking book can be a warning, not a triumph.

## Common misconceptions

**"A wider spread always means a more profitable bank."** No. A wider spread often means more credit risk. Banks that lend to riskier borrowers charge more, which inflates the gross spread, but they also lose more to defaults. The 2008 subprime lenders had gorgeous gross margins right up until the loans went bad. Risk-adjusted margin — spread minus expected losses — is the honest number, and it can be *lower* for the high-spread lender once the cycle turns.

**"NIM and the spread are the same thing."** They're close, but not equal. NIM is net interest income over *earning assets*; the spread is just asset yield minus cost of funds. NIM is typically a touch higher than the spread because part of the asset book is funded by free, non-interest-bearing deposits and by equity. In our worked example the spread was 3.31 points but NIM was 3.50% — the 19 basis-point gap is the free-funding lift. A bank with a lot of free checking deposits enjoys a bigger gap.

**"Higher interest rates are bad for banks."** For the *core spread business*, higher rates are usually good, at least at first — they monetize the bank's free funding and let deposit betas lag, widening NIM. What higher rates *do* hurt is the market value of the bank's existing fixed-rate bonds and its loan demand. A bank can have rising NIM and falling bond values at the same time — and if it's forced to sell those bonds (as SVB was), the second effect can kill it even while the first looks healthy. The 2023 story was not "high rates crushed bank margins"; margins rose. It was "high rates crushed the value of bonds that a few badly-positioned banks were forced to sell".

**"A bank can just raise the rate it charges to widen its margin."** Only if it can do so without losing the loan to a competitor or driving the borrower into default. Loan pricing is set by competition and by the borrower's credit; a bank that unilaterally jacks up its loan rates either loses good borrowers (who refinance elsewhere) or keeps only the desperate ones (adverse selection). NIM is not a dial management can simply turn — it's an outcome of the rate environment, the bank's mix, and the competitive landscape.

**"Deposit beta is a fixed property of a bank."** It drifts through the cycle. Early in a hiking cycle, betas are low because savers are slow to react; late in the cycle, betas climb as competition heats up and depositors chase yield. The same bank that ran a 0.10 beta in early 2022 was running a 0.55 beta by 2024. A bank's *structural* beta (set by how sticky its franchise is) anchors the range, but the realized beta moves a lot — which is why NIM expands early in a hiking cycle and fades later.

## How it shows up in real banks

### The 2021 ZIRP trough: maximum maturity transformation, minimum reward

By 2021, the US banking system was awash in deposits. Pandemic stimulus and Fed asset purchases had pushed trillions of fresh dollars into bank accounts; total deposits surged. But the policy rate was pinned near zero, so banks couldn't earn much on the cash, and lending demand was soft. The result was the 2.56% NIM trough — the lowest in the modern record. Banks were doing maximum maturity transformation (sitting on enormous short-dated deposit funding) for minimum reward (asset yields near the floor). It was a vivid demonstration of the spine: the maturity-transformation machine was running flat out, and the spread it earned was almost nothing. The free-funding kicker — usually a bank's quiet gift — was worth essentially zero when the policy rate was zero.

### The 2022-23 hiking windfall: free funding suddenly monetized

When the Fed hiked from near zero to over 5% in eighteen months, the same enormous deposit base suddenly became a goldmine. Banks were paying near-zero on much of those deposits (low beta) while their floating-rate loans and their cash at the Fed repriced up immediately. NIM jumped from 2.56% in 2021 to about 3.30% in 2023. For the largest banks, this was the best net-interest-income environment in a generation — JPMorgan's net interest income rose tens of billions of dollars. The windfall was almost entirely a deposit-beta story: the asset side repriced up fast, the funding side lagged, and the gap fell straight to the margin. This is the asset-sensitive bank's dream half-cycle in action.

### Silicon Valley Bank: the mix trap that no NIM number warned about

SVB is the cautionary tale of confusing a healthy NIM with a safe bank. In 2020-21, flooded with deposits from its tech-startup clients, SVB invested an enormous share — roughly \$91 billion of its securities — into long-dated, fixed-rate bonds at the low yields then on offer. That locked in a low asset yield. When the Fed hiked, two things happened at once. SVB's NIM didn't collapse — it actually held up reasonably, because its deposits were nominally cheap. But the *market value* of those long bonds fell sharply, opening a huge unrealized loss. When SVB's concentrated, uninsured depositor base (94% of deposits were above the \$250,000 insurance limit) began to pull money — \$42 billion was requested for withdrawal on March 9, 2023 alone — SVB had to sell those bonds at a loss to raise cash, crystallizing the hole and triggering the run. The lesson for reading NIM: a clean margin can sit right next to a fatal asset-mix and duration problem. NIM measures the *income* statement; the bond losses lived on the *balance sheet*. You have to read both. (The system-level account is in [the SVB and Credit Suisse post](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

### The big-bank franchise advantage: low betas at scale

Through the 2022-24 cycle, the megabanks with the deepest pools of sticky retail checking deposits — JPMorgan, Bank of America, Wells Fargo — ran materially lower deposit betas than smaller and online-only banks. An online bank competing only on the rate it pays has, by definition, a high-beta deposit base: its customers are there *for* the rate and will leave the instant a competitor pays more. A megabank's customers are there for the branches, the app, the payroll direct deposit, the inertia. That stickiness let the big banks sit at the left of the beta curve and keep a larger share of the hiking windfall. It is the clearest demonstration that, in banking, *the quality of your funding is a bigger competitive advantage than the cleverness of your lending*.

### Regionals versus megabanks: the same hike, opposite outcomes

The 2023 cycle split the banking system in two, and NIM is the lens that explains the split. The megabanks — with their vast, diversified, sticky retail deposit bases — kept their deposit betas low and rode the hike to record net interest income. Many regional and community banks had a harder time. Their deposits were less sticky and more concentrated (commercial customers with large, rate-sensitive balances rather than millions of small inertial checking accounts), so their betas climbed faster and their funding costs caught up sooner. Worse, a number of them had, like SVB, parked excess pandemic-era deposits into longer-dated securities at low yields, so their asset side couldn't reprice up to meet the rising funding cost. The result was visible NIM compression at exactly the banks least able to absorb it — and, for the handful with the worst combination of concentrated funding and underwater bonds, outright failure. The same Fed hiking cycle was a windfall for one tier of banks and an existential threat to another, and the difference came down almost entirely to the quality of the deposit franchise and the discipline of the asset mix. NIM, read alongside the deposit base and the securities book, would have flagged the vulnerable banks before the runs did.

### The deposit war: when beta is forced up by competition

A subtler real-world dynamic is the deposit war. When one bank in a market starts aggressively raising the rate it pays to attract deposits — often a smaller bank scrambling for funding, or an online bank buying market share — it can drag the whole local market's deposit beta up. Rivals must either match the higher rate (raising their own cost of funds and compressing NIM) or watch their deposits walk out the door (losing the cheap funding that is the franchise). Through 2023, exactly this happened: online banks and money-market funds offering 4.5-5% forced traditional banks to raise savings and CD rates faster than they wanted to, pushing realized deposit betas above what banks had modeled. The lesson is that deposit beta is not purely a property of one bank's customers — it's partly set by the most desperate competitor in the market. A bank with genuinely loyal, low-rate-sensitivity depositors can resist the war; a bank whose customers are there only for the rate gets dragged into it and watches its margin compress.

### The cutting-cycle squeeze ahead

As the Fed began cutting in late 2024, the asymmetry started to bite in reverse. Asset-sensitive banks' floating loans began repricing *down* immediately, while their deposit costs — already elevated after years of competition — were sticky on the way down (depositors who fought to get 4.5% don't surrender it quietly). This is the cutting-cycle squeeze: the same asset-sensitivity that delivered the 2023 windfall threatens to compress NIM as rates fall. Watch the 2024 print already easing to 3.23% from 3.30% — and understand it as the front edge of that dynamic, with deposit beta continuing its slow climb even as the asset side gives ground.

### Net interest income in dollars: the leverage on the margin

To feel why a fraction of a point matters so much, consider the dollar scale. The US banking system carries on the order of \$24 trillion in assets, of which roughly \$20 trillion are earning assets. A 10 basis-point change in industry NIM is therefore about \$20 billion a year in net interest income. The swing from the 2.56% trough to the 3.30% peak — about 74 basis points — is on the order of \$150 billion of annual net interest income, appearing and disappearing purely from where rates sit and how fast deposits chase them. No fee initiative, no cost-cutting program, no new product moves a bank's profit like the spread does.

## The takeaway / How to use this

Net interest margin is the vital sign of a bank's core business — and now you can read it like a clinician rather than a tourist.

Start by separating the *level* effect from the *mix* effect. When you see a bank's NIM rise, ask: is this the rate cycle lifting all boats (the free-funding kicker monetizing as rates rise), or is this bank actually outperforming through a low-beta deposit base and a smart asset mix? The first is borrowed from the Fed and will reverse when rates fall; the second is a durable franchise advantage. The dual-axis NIM-vs-rates chart is your first check: if the bank's NIM just tracks the policy rate, it's riding the tide, not swimming.

Next, always hunt for the deposit beta. It is the most informative single number about a bank's funding quality. A bank running a low, slow beta in a hiking cycle has a genuinely sticky, valuable deposit base — the franchise the whole series keeps pointing at. A bank with a high beta is renting its deposits at market rates and has no real edge; its wide NIM in good times will evaporate fast in a deposit war. The beta curve we charted — where NIM at beta 0 is dramatically higher than NIM at beta 1 — is the entire value of a great franchise in one picture.

Then refuse to read NIM alone. Pair it with three things: the dollar figure for net interest income (a rising margin on a shrinking book can hide a falling business), the expected credit-loss rate (a high NIM bought with risky lending is partly an illusion that provisions will claw back), and — critically — the bank's interest-rate-risk position and any unrealized bond losses. SVB is the permanent reminder that a perfectly healthy-looking margin can sit directly on top of a fatal asset-mix and duration trap. The income statement and the balance sheet tell different halves of the same story, and the margin lives on the income side only.

Finally, locate the bank on the asset-sensitive/liability-sensitive spectrum and ask what the *next* leg of the rate cycle does to it. An asset-sensitive bank is making an implicit directional bet on rates with its core franchise: it wins when rates rise and loses when they fall. That's not free money — it's a position. Knowing which way a bank is leaning tells you whether the coming rate environment is a tailwind or a headwind for its single most important line of revenue.

Tie it back to the spine. A bank is a leveraged, confidence-funded maturity-transformation machine that earns the spread between short funding and long assets. Net interest margin is the *price tag* on that trade — the measurement of how much the fragile, essential act of borrowing short and lending long is paying right now. When you can read NIM and its drivers, you can see, quarter by quarter, whether the machine at the heart of every bank is gaining power or quietly running out of it.

*This is educational, not investment advice. It explains how a bank's margin works, not whether any particular bank stock is worth owning.*

## Further reading & cross-links

- [The income statement of a bank: net interest income, fees, and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — where net interest income sits in the full revenue picture, and how provisions and the efficiency ratio turn margin into profit.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — the deep dive on the cheap, sticky deposit base that gives a bank a low cost of funds and a low deposit beta.
- [Interest-rate risk in the banking book (IRRBB) and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the other side of rate sensitivity: how the same rate moves that lift NIM can sink the market value of a bank's assets, and the mechanism that broke SVB.
- [Silicon Valley Bank and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level account of how a margin that looked fine sat on top of a balance-sheet trap.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — why the slope between short and long rates, not just the level, shapes the spread a maturity-transforming bank can earn.

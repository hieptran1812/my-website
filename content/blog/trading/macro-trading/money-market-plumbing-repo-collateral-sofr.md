---
title: "Money-Market Plumbing: Repo, Collateral, SOFR, and the Early-Warning System"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the overnight repo market — how trillions get borrowed against collateral every night, what SOFR and the rate corridor actually measure, and how reading repo spreads gives a macro trader the earliest warning of funding stress."
tags: ["macro", "monetary-policy", "repo", "sofr", "collateral", "money-markets", "liquidity", "funding", "federal-reserve", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Underneath every asset price sits the **repo market**: the overnight plumbing where trillions of dollars get borrowed against collateral every single day. When repo seizes up, it is the canary in the coal mine — the first place funding stress shows. **SOFR** (the headline repo rate) and its spread to the policy rate are the macro trader's earliest warning that the financial system is running short of cash.
>
> - A **repo** is just a one-night secured loan: you hand over cash and take Treasuries as collateral, then unwind it the next morning with a sliver of interest. The rate on that loan, aggregated across the market, is **SOFR**.
> - The Fed boxes overnight rates inside a **corridor**: the **ON RRP** facility is the floor (no one lends cash for less), the **SRF** is the ceiling (no one pays more for repo), and **fed funds / SOFR** float in the middle. Watching SOFR climb toward the ceiling is watching stress build in real time.
> - Repo seizes when three things collide: **too much collateral to fund, too little cash (bank reserves) in the system, and a calendar pressure point** like quarter-end. That exact recipe produced the September 2019 spike — repo printed near **10%** while the policy rate was only **2.25%** — and the March 2020 dash-for-cash.
> - The one spread to watch: **SOFR minus IORB** (the interest the Fed pays on reserves). When SOFR pushes above IORB and stays there, cash is getting scarce. That is your funding-stress dial — slow to flash, but it flashes before equities do.

On the morning of September 17, 2019, something broke in the most boring corner of finance. The overnight repo rate — the cost of borrowing cash for one night against the safest collateral on earth, US Treasuries — did not tick up a few basis points the way it does on a busy day. It *exploded*. Intraday it touched roughly **10%**, more than four times the Federal Reserve's policy rate of about 2.25%. For a few hours, the biggest, most liquid, most heavily regulated funding market in the world simply could not find enough cash to clear, and the price of overnight money went vertical.

This was not a developing-market crisis or a crypto exchange blowing up. This was the plumbing under the US Treasury market — the deepest pool of safe assets in existence — failing to move cash from the people who had it to the people who needed it. The Fed, which had not run this kind of operation in over a decade, scrambled emergency repo facilities back to life within 24 hours and began pumping cash directly into the market. Within weeks it had restarted asset purchases. The spike was contained, but the message landed: **the overnight funding market is where systemic stress shows up first, and almost no one outside it is watching.**

That is the gap this post closes. Most traders track the S&P, the dollar, the 10-year yield, maybe the VIX. Very few track repo — and yet repo is *upstream* of every one of those. Asset prices are built on leverage, leverage is built on borrowed cash, and borrowed cash is priced in the repo market every night. When repo gets expensive, leverage gets expensive, and eventually something de-levers. The trader who reads repo saw September 2019 coming, sized the March 2020 funding panic, and front-ran the Fed's response each time. The trader who thought repo was "back-office stuff" got blindsided. We are going to build the entire picture from zero — what a repo even is, all the way to the exact spreads you watch and how you position around them.

![A repo trade showing cash and Treasury collateral swapping overnight then reversing with interest](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-1.png)

## Foundations: What repo, collateral, and SOFR actually are

Before any trading signal, you need to be able to explain the machinery to a ten-year-old. Almost every confused argument about "the repo market" comes from people using the word without knowing what one trade looks like. So we define every term from the ground up, with an everyday-money analogy first, then the precise mechanics.

### The everyday-money version: a pawn shop that resets every morning

Here is the whole idea in one sentence you already understand. **A repo is a pawn-shop loan that gets unwound the next day.**

You walk into a pawn shop with a gold watch. The pawnbroker hands you cash and keeps the watch. Tomorrow you come back, repay the cash plus a small fee, and get your watch back. While the watch is in the shop, the broker is protected: if you never return, they keep the watch and sell it. The cash they gave you was *less* than the watch is worth, so they have a cushion.

Now replace the watch with US Treasury bonds, the pawnbroker with a money-market fund, the borrower with a Wall Street dealer, and "tomorrow" with "literally tomorrow morning." That is a repo. The dealer needs cash overnight, owns a pile of Treasuries, and pledges them to a cash-rich lender in exchange for cash — promising to buy them back ("repurchase," hence *repo*) the next day at a tiny markup. The markup, annualized, is the interest rate on the loan.

That is the entire concept. Everything else is detail layered on top. Hold the pawn-shop picture in your head; we will keep returning to it.

### Repo and reverse repo: same trade, two seats

The word "repo" is short for **repurchase agreement**. Mechanically it is a sale of securities combined with a binding agreement to repurchase them at a set price on a set date. Economically — and this is the part that matters — **it is a collateralized loan of cash.** The "sale and repurchase" framing is mostly legal plumbing that makes the collateral easy to seize if the borrower defaults. Think of it as a loan, always.

Every repo has two sides, and which word you use depends on which seat you sit in:

- **Repo (from the cash *borrower's* seat):** "I am doing a repo" means *I am borrowing cash and posting my securities as collateral.* I am the dealer with bonds who needs money.
- **Reverse repo (from the cash *lender's* seat):** "I am doing a reverse repo" means *I am lending cash and taking securities as collateral.* I am the money-market fund with idle cash that wants a safe overnight return.

It is the **same transaction** viewed from opposite ends — like "I'm buying" versus "you're selling." When you read that the Fed's **reverse repo** facility (RRP) holds \$2.5 trillion, that means \$2.5 trillion of cash was *lent to the Fed* by money funds, with the Fed posting Treasuries as collateral. The Fed is the borrower of cash; the money funds are the lenders. We will unpack that facility shortly, but lock in the vocabulary now, because half of all repo confusion is people getting these two seats backwards.

### Collateral: why Treasuries are the watch

In our pawn-shop analogy, the watch is the **collateral** — the asset the lender holds to protect themselves. In the repo market, the overwhelmingly dominant collateral is **US Treasury securities**: bills, notes, and bonds issued by the US government. Why Treasuries?

- **They are the safest, most liquid asset in the world.** If the borrower defaults, the lender is holding something they can sell instantly to almost anyone, at a known price, in any market condition. Good collateral is collateral you can liquidate without taking a loss.
- **They are abundant.** The US government has issued trillions of dollars of them, so there is always a deep pool to pledge. (As we will see, *too much* new issuance is one of the things that breaks the market.)
- **They are homogeneous and easy to value.** A 10-year Treasury is a 10-year Treasury; everyone agrees what it is worth to the penny.

Other collateral exists — agency mortgage-backed securities, corporate bonds, even equities — and it trades in repo too. But it is riskier collateral, so it commands a worse deal for the borrower (a bigger cushion, a higher rate). When people say "the repo market" and "SOFR" without qualification, they mean the **Treasury repo** market, the cleanest and largest segment. That is our focus.

It is worth knowing one more wrinkle in collateral, because it occasionally drives strange rate moves: not all Treasuries are equally available. Most trade as **general collateral (GC)** — any Treasury will do, and the repo rate is the normal market rate. But a *specific* bond that everyone needs at once (often the newest, most-traded "on-the-run" issue, which short-sellers must borrow to deliver) can go **"special"**: so many people want to borrow that exact bond that lenders of it can charge a *lower* repo rate, sometimes even near zero or negative. A bond on "special" means its repo rate sits below GC because the collateral itself is in scarce demand. You do not need to trade this, but it explains why a single security's repo rate can detach from SOFR: SOFR is the *cash* price across GC, while a specific bond's rate reflects demand for *that bond*. When a benchmark issue goes deeply special, it is a sign of heavy short positioning or a collateral shortage in that maturity — a microstructure tell that occasionally precedes a squeeze.

### The haircut: the lender's cushion

The pawnbroker never hands you the full resale value of the watch. They give you less, so that if the watch's price drops or you vanish, they are still covered. That gap is the **haircut**.

In repo, the haircut is the percentage by which the collateral's value exceeds the cash loaned. A **2% haircut** means: to borrow \$100 of cash, you must post \$102 of Treasuries. The extra \$2 is the lender's safety margin. If you default and Treasury prices have slipped 1% overnight, the lender still has collateral worth more than the cash they are owed.

Haircuts are usually tiny for Treasuries — a fraction of a percent to a couple of percent — precisely because Treasuries barely move overnight and are trivial to sell. For riskier collateral, haircuts get fat: 5%, 10%, sometimes much more for volatile assets. **The size of the haircut is the lender's vote on how risky the collateral is.** When haircuts on a class of collateral suddenly widen, that is the funding market quietly downgrading that asset — a stress signal in its own right. In the 2008 crisis, exploding haircuts on mortgage collateral were a key channel through which the panic spread: borrowers had to post more and more collateral against the same cash, a brutal margin call run system-wide.

#### Worked example: a single overnight repo trade

Let us walk one trade end to end so the mechanics are concrete. A dealer needs \$100 million of cash for one night and owns Treasuries to pledge. A money-market fund has \$100 million of idle cash and wants a safe overnight return. They agree on a Treasury repo at a **2% haircut** and an overnight rate equal to SOFR, which we will say is **5.00%** (a realistic 2024 level).

- **Day 0, morning:** The dealer posts Treasuries with a market value of **\$102 million** (that is the \$100M loan grossed up by the 2% haircut: \$100M × 1.02 = \$102M). The money fund wires **\$100 million** of cash to the dealer. The fund now holds \$102M of Treasuries as collateral against a \$100M cash claim — a \$2M cushion.
- **Day 1, morning:** The dealer repays the **\$100 million** principal plus one night of interest. The interest is the rate times the principal times one day over 360 (money markets use a 360-day year): \$100,000,000 × 0.05 × (1 / 360) = **\$13,889**. The fund returns the \$102M of Treasuries. The dealer pays back \$100,013,889 and gets its bonds back.

The fund earned \$13,889 for taking essentially no risk for one night — it held more collateral than cash the entire time, and the collateral was the safest asset on earth. The dealer got the cash it needed and kept its bonds (it never *sold* them; it borrowed against them). The intuition: **a repo is a fully secured overnight loan, so its rate sits just a hair above the risk-free rate — and the haircut means the lender is over-collateralized the whole night, which is exactly why repo is "safe" until the day it isn't.**

### SOFR: the price of overnight cash, measured

We have been saying "the repo rate" as if there were one. In reality, thousands of repo trades happen every day at slightly different rates. **SOFR — the Secured Overnight Financing Rate — is the volume-weighted median rate across the Treasury repo market.** It is, quite literally, *the* answer to the question "what did it cost to borrow cash overnight against Treasuries today?"

A few things make SOFR important beyond just being a number:

- **It is based on real transactions, not quotes.** Every day the New York Fed collects data on roughly \$2 trillion of actual overnight Treasury repo trades and publishes the median. There is no panel of banks guessing what they *would* charge (the fatal flaw of the old LIBOR benchmark, which was manipulated for years). SOFR is a measurement of what actually happened.
- **It is secured.** Because every trade behind SOFR is collateralized by Treasuries, SOFR reflects the price of *cash*, stripped of credit risk. It is as close to a pure, market-measured risk-free overnight rate as exists.
- **It replaced LIBOR.** After the LIBOR scandals, regulators pushed the entire financial system onto SOFR. Trillions of dollars of loans, swaps, and derivatives now reference SOFR. So a move in repo plumbing is no longer an obscure back-office event — it reprices contracts across the whole economy.

The everyday intuition: **SOFR is the wholesale price of money for one night.** Just as there is a wholesale price of electricity or wheat, there is a wholesale price of overnight cash, and SOFR is it. When that price spikes, cash has become scarce — and scarce cash is the proximate cause of nearly every funding crisis.

One technical detail worth knowing, because it explains why SOFR sometimes looks "jumpy" even when nothing is wrong. SOFR is a *volume-weighted median*, which makes it robust to a few weird trades — but it also means that on dates when the mix of trades shifts (quarter-ends, big settlement days), the median can step up or down in a way that looks like stress but is partly compositional. The professional habit is to look not just at SOFR itself but at the **distribution behind it**: the New York Fed publishes the 1st, 25th, 75th, and 99th percentiles of the rate alongside the median. When the *top* of that distribution — the 99th percentile, the most desperate trades — pulls far away from the median, that tail is where stress shows up first, before the median itself moves. A widening tail with a calm median is an early whisper that some corner of the market is paying up for cash. The headline number is the summary; the percentiles are the early warning.

There is also a related benchmark you will hear about: **the SOFR average and the SOFR index**, which are just compounded-over-time versions of daily SOFR used to set the floating rate on actual loans and swaps. You do not need the math; you need the implication. Because so many real contracts reference compounded SOFR, a sustained move in the daily repo rate does not stay in the plumbing — it flows directly into the coupons on trillions of dollars of floating-rate debt and derivatives. The plumbing rate *is* the rate the broader economy pays.

### How SOFR relates to the fed funds rate

You may already know about the **fed funds rate** — the rate banks charge each other to borrow reserves overnight, and the rate the Fed "sets" when it raises or cuts. (We build that mechanism fully in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).) So how does SOFR relate to fed funds? They are siblings, not twins:

- **Fed funds is unsecured.** Banks lend each other reserves with no collateral, purely on trust. It reflects the price of *bank* credit overnight.
- **SOFR is secured.** Repo is collateralized by Treasuries. It reflects the price of *cash*, with credit risk stripped out.

Because they price almost the same thing — overnight money — they normally track each other within a few basis points. But they can diverge, and the *direction* of the divergence is informative. When SOFR rises **above** fed funds, it usually means cash is scarce in the collateralized market specifically — too many securities chasing too little cash. When SOFR sits **below** fed funds, cash is abundant and collateral is the scarce thing. A persistent gap between them is a tell about which side of the plumbing is tight. We will turn this into a tradeable signal later.

## The corridor: how the Fed boxes in overnight rates

Knowing what SOFR is leads to the obvious question: what stops it from going anywhere it wants? Why does it normally sit near the policy rate instead of swinging wildly every night? The answer is that the Fed builds a **corridor** — a floor and a ceiling — and overnight rates trade inside it. Understanding the corridor is the single most useful frame for reading funding stress, because *stress is just SOFR drifting toward, or punching through, the ceiling.*

![The overnight rate corridor with an RRP floor, fed funds and SOFR in the middle, and an SRF ceiling](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-2.png)

### The floor: the ON RRP facility

The **Overnight Reverse Repo facility (ON RRP)** is the Fed's floor on overnight rates. Here is how it works, in the seats we defined earlier: the Fed offers to **borrow cash** from a wide set of money-market players (money funds, government-sponsored enterprises, some banks) overnight, posting Treasuries as collateral, at a fixed rate the Fed sets. In other words, the Fed runs a reverse repo with the private sector — the Fed is the cash borrower, the money funds are the cash lenders.

Why is this a floor? Because **no rational cash lender would ever lend to a private borrower for *less* than they can get risk-free from the Fed.** If the Fed is paying, say, 4.55% to take your cash overnight with zero credit risk, you will not lend it to a dealer at 4.40%. You would just park it at the Fed. So the RRP rate becomes the minimum acceptable rate for overnight cash — the floor of the corridor. Any money fund with cash and nothing better to do dumps it at the RRP, and that option drags the whole market's floor up to the RRP rate.

The RRP is where you watch the **glut** side of the plumbing. When there is *too much* cash sloshing around relative to the available collateral and investments, money piles into the RRP. At its peak in December 2022, the ON RRP held **\$2.55 trillion** of parked cash — a staggering sum that represented money funds saying "we literally have nowhere better to put this." As that pile drains, it is a real-time gauge of cash leaving the system's safest parking lot and going to work (or being soaked up by Treasury issuance). We will chart that drain shortly.

### The ceiling: the Standing Repo Facility (SRF)

The **Standing Repo Facility (SRF)** is the Fed's ceiling on overnight rates, and it is the direct institutional answer to September 2019. Here the seats flip: the Fed offers to **lend cash** to dealers and banks overnight against Treasury collateral, at a fixed rate, on demand, in effectively unlimited size.

Why is this a ceiling? Because **no rational borrower would ever pay *more* in the open market than they can borrow from the Fed.** If the Fed will lend you cash at 4.75% against your Treasuries any time you ask, you will never pay 6% to a private lender. The SRF rate caps how high repo can go. The whole point is to prevent another 2019: if private cash dries up and repo starts spiking toward 10%, dealers can tap the SRF instead, and the spike never happens. The SRF turns "the Fed scrambled emergency operations together overnight" into "the facility is always open."

A crucial nuance: **the SRF is a backstop, not a subsidy.** Its rate is set slightly *above* the normal market rate, so in calm times nobody uses it — borrowing privately is cheaper. SRF usage only appears when private funding is tightening, which makes **any meaningful SRF usage itself a stress signal.** An empty SRF means the plumbing is fine; a busy SRF means cash got scarce enough that dealers came to the lender of last resort. Watching the SRF print is watching the ceiling get tested.

### The middle: fed funds, SOFR, and the IORB lever

Between the RRP floor and the SRF ceiling, the actual overnight rates — fed funds and SOFR — float around. The Fed nudges where they float using a separate lever: **IORB, the Interest on Reserve Balances.** This is the rate the Fed pays banks on the reserves they hold at the Fed. Because a bank can always earn IORB risk-free on its reserves, IORB acts as a powerful magnet for where overnight rates settle — banks will not lend reserves to anyone for less than they can earn just leaving them at the Fed.

So the corridor, from top to bottom, looks like this: **SRF ceiling → IORB / fed funds / SOFR clustered in the middle → RRP floor.** The Fed moves the whole corridor up or down by adjusting these administered rates (IORB, RRP, SRF) together when it hikes or cuts. The market rates live inside. And here is the payoff for a trader: **in calm times SOFR sits quietly near IORB, comfortably below the ceiling. As stress builds, SOFR drifts up, away from IORB and toward the SRF ceiling. That drift is the early-warning needle.** You do not need a crisis to read it — you read the *position of SOFR within the corridor* every single day.

#### Worked example: the RRP as the floor, in dollars

Let us make the floor concrete with the real peak. In December 2022, the ON RRP facility held its all-time high of **\$2.55 trillion** of cash parked by money-market funds, with the RRP rate around **4.30%** at that time.

- A money fund parks **\$50 billion** at the RRP overnight. Its return for one night: \$50,000,000,000 × 0.043 × (1 / 360) ≈ **\$5.97 million** for the night, fully risk-free, collateralized by the Fed's Treasuries.
- Why would a fund accept "only" the RRP rate when dealers might pay a touch more in private repo? Because in late 2022 the system was *drowning* in cash — there was more cash than good private opportunities to deploy it safely. The RRP became the default home for that excess. The fact that **\$2.55 trillion** chose the floor over private repo tells you the system had a *cash glut*, not a cash shortage.
- The mirror image is the lesson: the RRP rate sets the floor precisely because that \$2.55T was always willing to walk away from any private deal paying less. That standing alternative is what holds the floor up.

The intuition: **the RRP balance is a thermometer for excess cash. A swollen RRP means too much money and not enough collateral; a drained RRP means the cash has been pulled back into the system — often the prelude to scarcity on the other end.**

## Who plays: the cast of the money market

A market is its participants. To read repo, you need to know who is sitting at the table, what each one wants, and which way their cash flows. The plumbing is really a chain that routes idle cash from savers all the way to the leveraged investors who put it to work in the Treasury market.

![A flow map showing money-market funds routing cash through RRP and repo to dealers, Treasuries, and hedge funds](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-4.png)

### Money-market funds: the cash supply

**Money-market funds (MMFs)** are the great reservoir of cash in this system. They pool money from retail savers, corporations, and institutions who want a safe, liquid place to hold cash that earns a little yield — better than a checking account, nearly as safe. As of recent years, US money funds hold on the order of **\$6.5 trillion**. They are not allowed to take much risk, so what do they do with all that cash? They lend it overnight in repo (taking Treasuries as collateral) and they park it at the Fed's RRP. **MMFs are the dominant supplier of cash to the repo market.** When a flood of cash pours into money funds — as happened when the Fed hiked rates in 2022-23 and savers fled low-yielding bank deposits — that cash needs a home, and it shows up as repo lending and RRP balances.

### Primary dealers: the middlemen

**Primary dealers** are the large bank-affiliated broker-dealers (think the trading arms of the biggest banks) that the Fed transacts with directly and that make markets in Treasuries. They are the central intermediaries of the plumbing. A dealer's job is to hold inventory — it buys Treasuries from issuers and investors and sells them on — and holding inventory costs money, money the dealer borrows in repo. So dealers are the system's biggest *borrowers* of cash, pledging their Treasury inventory as collateral. When the government issues a wave of new bonds, dealers are the ones who absorb them initially, and they must fund that growing inventory in repo. That is why a surge in Treasury issuance translates directly into a surge in dealer repo borrowing — and pressure on rates.

### Hedge funds: the leverage

**Hedge funds** are major cash *borrowers* too, and they are increasingly central to the story. Many run a strategy called the **Treasury basis trade**: they buy cash Treasuries and short the corresponding Treasury futures to capture a tiny price gap, then lever the position enormously — sometimes 50-to-1 or more — to make the tiny gap worthwhile. That leverage is funded in repo. A basis-trade fund borrows cash overnight against its Treasuries, buys more Treasuries, borrows against those, and so on. The strategy is profitable in calm markets but lethally sensitive to funding costs: **if repo rates spike or the fund cannot roll its overnight borrowing, the whole levered position must be unwound at once.** This is precisely what amplified the March 2020 turmoil. Hedge-fund repo borrowing is the leverage that sits on top of the plumbing, and it is the thing most likely to break violently when the plumbing tightens.

### Banks: reserves and the other cash lender

**Banks** sit at the center too, but their key variable is **reserves** — the cash they hold at the Fed. Banks lend cash in repo and fed funds, and the amount they are willing to lend depends on how flush they are with reserves. When the banking system holds *ample* reserves, banks happily lend the excess into repo, keeping rates calm. When reserves get *scarce*, banks hoard them (they need reserves for their own regulatory and payment needs), and they pull back from lending into repo — which is exactly when repo rates jump. **The level of bank reserves is the master dial on repo conditions**, and it is the single most important number we have not yet charted. We get to it next, because it is the heart of why repo seizes.

## What makes repo seize: the three-body problem

Repo does not break randomly. It breaks when specific, knowable pressures converge. There are three of them, and when all three line up on the same day, you get a spike. Understanding this recipe is what turns repo-watching from trivia into a usable early-warning system.

![Three converging forces — collateral glut, scarce reserves, and quarter-end — producing a September 2019 repo rate spike](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-6.png)

### Force 1: collateral glut (too much to fund)

Repo is cash chasing collateral and collateral chasing cash. When the US Treasury issues a wave of new bonds — to fund deficits or rebuild its cash balance after a debt-ceiling fight — that new collateral has to be *funded*. Dealers buy the new issuance and must borrow cash in repo against it. Suddenly there is a pile of new Treasuries needing cash, and the demand for cash in repo jumps. More collateral chasing the same cash pushes the repo rate **up**. A heavy auction-settlement day, or a stretch of large deficits, floods the system with collateral and tilts the balance toward higher repo rates.

### Force 2: reserve scarcity (too little cash)

This is the master force. **Bank reserves are the ultimate source of cash in the system.** When reserves are *ample*, banks lend the excess freely into repo and rates stay calm and pinned near the corridor floor. When reserves fall toward *scarce*, banks stop lending the marginal dollar — they need it themselves — and the cash available to repo shrinks. Less cash chasing the same collateral pushes the repo rate **up**, and the system gets twitchy: small shocks that used to be absorbed now produce visible rate moves.

What drains reserves? Two big things. **Quantitative tightening (QT)** — the Fed shrinking its balance sheet — directly removes reserves from the banking system. And the **Treasury General Account (TGA)** — the government's checking account at the Fed — pulls reserves out when it fills up (tax payments and debt issuance move cash from banks to the government's Fed account). When the Fed is doing QT *and* the TGA is rising, reserves drain from both ends at once. This is the mechanism behind the great funding crunches, and it is the through-line connecting net liquidity, the balance sheet, RRP, and TGA into one picture. (We build that full net-liquidity framework in the companion post on the [central bank balance sheet, net liquidity, reserves, RRP, and TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga), and the broader funding-liquidity lens in [what liquidity means for market and funding](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders).)

![Bank reserves bar chart with a scarcity threshold marking the level below which repo gets twitchy](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-5.png)

The chart above is the single most important picture in this post. In September 2019, reserves had been drained to about **\$1.47 trillion** by a combination of QT and a rising TGA — a level the market later understood was below the comfortable minimum. Through 2020-21 the Fed flooded reserves back to over **\$4 trillion**, eliminating scarcity entirely. As QT resumed, reserves drifted back down toward **\$3 trillion**. The exact "scarcity line" is not a fixed number — it rises as the system grows — but the principle is permanent: **below some threshold of reserves, repo stops being calm and starts being reactive.** Watching reserves approach that zone is watching the fuel for a spike accumulate.

### Force 3: the calendar (quarter-end and tax dates)

The third force is timing. On specific calendar dates, cash gets pulled out of the system temporarily and balance-sheet space gets tight:

- **Quarter-end and especially year-end:** Banks face regulatory reporting on these dates and shrink their balance sheets to look safer, which means pulling back from repo lending right when it is least convenient. This is why you see reliable little repo rate bumps at the end of March, June, September, and December.
- **Corporate tax dates:** On quarterly tax-payment days, corporations move cash to the Treasury, which drains it from money funds and banks — pulling cash out of repo on a specific, predictable day.

None of these are crises by themselves. They are *pressure points*. But layer a calendar pressure point on top of a collateral glut on top of scarce reserves, and you get a spike. **September 17, 2019 was all three at once:** a corporate tax date pulled cash out, a wave of Treasury settlement dumped collateral in, and reserves had already been drained to that \$1.47T danger zone. The plumbing had no slack left, and the rate went to 10%. None of the three forces alone would have done it; the *convergence* did.

#### Worked example: reserve scarcity and the repo rate

Let us quantify how the level of reserves governs the spike risk, using the real reserve path. Think of it as how much "shock-absorbing cash" the system has.

- **September 2019:** reserves ≈ **\$1.47 trillion**. The system had almost no buffer. When the tax-date and collateral-settlement shock hit — call it a sudden \$100 billion of extra cash demand on one day — there was no slack to absorb it. Banks would not lend the marginal dollar, and the rate exploded to ~**10%**, roughly **775 basis points above** the ~2.25% policy rate.
- **December 2021:** reserves ≈ **\$4.19 trillion**, after the Fed's COVID-era flood. The system was awash. The same \$100 billion shock would barely register — there was \$4T of cash looking for a home, much of it sitting idle in the RRP. Repo stayed pinned near the floor. No spike was even conceivable.
- **December 2024:** reserves ≈ **\$3.27 trillion**, after two years of QT drained them back down. Still comfortable, but trending toward the zone where the Fed starts watching closely — which is exactly why the Fed slowed and then planned to end QT rather than risk a 2019 repeat.

The intuition: **the repo rate's sensitivity to any given shock is inversely proportional to the reserve buffer.** With \$4T of reserves a shock is absorbed silently; with \$1.5T the same shock detonates. The reserve level is not the trigger — it is the *amount of dynamite lying around* when a trigger arrives.

## Reading the spreads: turning plumbing into a signal

Now we convert all of this into the actual gauges a trader watches. The beauty of the corridor framing is that funding stress is *visible as a position* — where rates sit relative to the Fed's administered rates. You do not need inside information; the data is published daily by the New York Fed. Three spreads do most of the work.

### Spread 1: SOFR minus IORB — the master gauge

This is the one to internalize. **IORB** is what banks earn risk-free on reserves, and SOFR is the market rate for overnight cash. In calm times SOFR sits *at or slightly below* IORB, because banks have plenty of reserves and the marginal cash is abundant. As cash gets scarce, **SOFR climbs above IORB** — borrowers are now willing to pay more than the risk-free reserve rate to get cash, because cash is in short supply.

- **SOFR − IORB negative or near zero:** ample reserves, calm plumbing. Normal.
- **SOFR − IORB persistently positive (say +5 to +10 bps and rising):** reserves getting scarce, cash bid up. The early-warning needle is moving.
- **SOFR − IORB blowing out (tens of basis points, or worse):** acute stress, the kind that precedes Fed intervention.

The reason this is the *master* gauge is that it directly measures the thing that matters — whether cash is abundant or scarce relative to the system's needs — without you having to estimate reserve levels yourself. The market does the estimating for you and prints it as a spread every morning.

### Spread 2: RRP usage — the glut gauge

The RRP balance tells you the *opposite* corner of the picture: how much *excess* cash has nowhere to go. A swollen RRP (the \$2.55T peak) means a cash glut — there is more money than safe private opportunity, so it parks at the floor. A draining RRP means that excess is being pulled back into the system, usually by Treasury issuance soaking it up. **Watch the RRP for the transition.** As long as the RRP is fat, there is a giant buffer of cash that can be redeployed into repo the moment private rates rise above the floor — a shock absorber. When the RRP nears empty, that buffer is gone, and the next marginal stress has to come straight out of bank reserves. **A drained RRP plus falling reserves is the setup for a repeat of 2019.**

![Overnight reverse repo balance from its 2.55 trillion dollar peak down to near zero](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-3.png)

The chart shows the full arc: the RRP swelled to **\$2.55 trillion** in December 2022 (the cash glut of the rate-hiking cycle), then drained relentlessly as QT shrank the balance sheet and Treasury issuance soaked up the excess, falling toward near-empty by late 2025. That drain is the system *consuming its buffer*. While the buffer existed, repo could not spike — any rate rise just pulled cash out of the RRP. Once it is gone, the next shock lands directly on bank reserves, and the early-warning gauges (SOFR − IORB, SRF usage) become the only thing standing between calm and a spike.

### Spread 3: SRF usage — the ceiling gauge

The SRF, recall, is the ceiling. Its rate is set *above* the market, so in calm times it goes unused — borrowing privately is cheaper. **Any meaningful SRF usage is therefore a direct readout that private funding got expensive enough to drive borrowers to the lender of last resort.** It is the cleanest "the ceiling is being tested" signal there is. An empty SRF is reassuring; a busy SRF, especially on non-quarter-end days, says the plumbing is straining. Because the SRF exists now (it did not in 2019), a future scarcity event should produce *SRF usage* rather than a 10% spike — the facility converts a price blowout into a visible quantity. That makes SRF usage the modern equivalent of watching the 2019 spike, but earlier and cleaner.

#### Worked example: the September 2019 spike as a funding hit

Let us put a dollar figure on why a repo spike matters to a real book — this is what makes it a trader's concern and not an academic curiosity. Take a levered fund running the Treasury basis trade, funding **\$10 billion** of Treasury positions in overnight repo, rolling the loan every single night.

- **Normal night:** repo at the policy rate of ~**2.25%**. One night's funding cost: \$10,000,000,000 × 0.0225 × (1 / 360) ≈ **\$625,000**. Manageable — the basis-trade profit is designed to exceed this.
- **September 17, 2019:** repo spikes to ~**10%**. That same night's funding cost: \$10,000,000,000 × 0.10 × (1 / 360) ≈ **\$2.78 million**. The overnight funding bill **more than quadrupled** in a single day — an extra ~**\$2.15 million** of cost the fund did not budget for.
- **The second-order danger:** worse than the cost is the *availability*. If the fund cannot find anyone to lend it the \$10B at any reasonable rate, it cannot roll its position and is forced to sell Treasuries into a market where everyone else is also short cash. That forced selling pushes Treasury prices down and yields up — a funding problem becomes a price problem, and a price problem becomes everyone's problem.

The intuition: **a levered book lives or dies on overnight funding, so a repo spike is not a sideshow — it is a sudden, brutal margin call on the most leveraged players, and their forced de-levering is the channel that transmits a plumbing problem into asset prices.**

#### Worked example: sizing the SOFR minus IORB stress dial

The master gauge is a spread, so let us calibrate what its values *mean* in basis points, using realistic numbers from a tightening cycle. Suppose IORB — the rate banks earn on reserves — is set at **4.90%**, and we watch SOFR's relationship to it over three regimes.

- **Ample regime:** SOFR prints **4.88%**, i.e. **2 basis points below IORB** (SOFR − IORB = −2 bps). Cash is abundant; the marginal lender is happy to lend below the risk-free reserve rate just to put cash to work. This is the calm, risk-supportive state. Nothing to do.
- **Tightening regime:** SOFR prints **4.96%**, i.e. **6 basis points above IORB** (SOFR − IORB = +6 bps) and the spread has been positive and creeping for several weeks. Now borrowers are paying *more* than the risk-free reserve rate to get cash — a direct signal that reserves are getting scarce. On a \$10 billion overnight book, +6 bps versus IORB is an extra \$10,000,000,000 × 0.0006 × (1 / 360) ≈ **\$1,667 per night** — small in dollars, but the *signal* is the point: the funding layer has flipped from supportive to straining.
- **Stress regime:** SOFR prints **5.30%**, i.e. **40 basis points above IORB**. This is the kind of blowout that precedes or accompanies Fed intervention; cash is acutely scarce and the most leveraged players are getting squeezed.

The intuition: **the dollar cost of a few basis points is trivial, but the spread's job is not to cost you money directly — it is to tell you, before equities react, which of the three regimes the funding system is in, so you can size leverage accordingly.**

## Common misconceptions

A handful of wrong mental models keep traders from using repo as the signal it is. Each one is correctable with a number.

### "Repo is obscure back-office plumbing, irrelevant to my trading."

This is the costliest error. Repo is the **funding layer underneath all leverage**, and leverage underpins asset prices. Roughly **\$2-4 trillion** of overnight Treasury repo trades *every single day* — it is one of the largest markets on earth. When repo seized in September 2019, the Fed restarted operations within 24 hours and ultimately added hundreds of billions to its balance sheet; in March 2020 a repo-and-Treasury funding spiral was a core reason the Fed unleashed unlimited QE. Both events moved every asset class. Repo is not adjacent to your trading; it is *upstream* of it.

### "SOFR is just another name for the fed funds rate."

No. **SOFR is secured (collateralized by Treasuries); fed funds is unsecured (bank credit).** They normally track within a few basis points, but they measure different things, and the *gap* between them is informative. SOFR rising above fed funds signals collateralized-cash scarcity specifically. Treating them as identical throws away a real signal. And critically, SOFR can spike *while fed funds barely moves* — exactly what happened in 2019, when the stress was in the collateral market, not in bank credit.

### "The Fed always backstops repo instantly, so spikes can't happen."

Two corrections. First, in September 2019 the Fed did *not* have a standing facility — it had to improvise emergency operations, and the rate hit ~10% before they kicked in. The SRF was created *afterward* precisely because the backstop was not automatic. Second, even with the SRF, the backstop has a *price* (the SRF rate, set above market) and operational frictions, and it does nothing for the *forced de-levering* that a scarcity scare triggers among hedge funds. The Fed can cap the rate; it cannot prevent the de-risking that the scare itself sets off. "The Fed has it handled" is the kind of complacency that gets a levered book caught.

### "A big RRP balance means the system has lots of liquidity, so we're safe."

Subtle but important: a swollen RRP means there is excess cash *parked at the floor* — which is liquidity sitting idle, not liquidity working in the system. It is a buffer, yes, but it is also a sign that cash had *nowhere better to go*. And the RRP can drain fast. The danger is not the fat RRP; it is the *transition* from fat to empty, because that drain removes the shock absorber. By the time the RRP is empty, the next stress lands directly on bank reserves. A big RRP is comforting only until you watch how quickly it can disappear.

### "Quarter-end repo spikes are noise; ignore them."

Quarter-end bumps are *predictable* noise — and that predictability is the point. They tell you how much *slack* is in the system. In a flush system, quarter-end barely registers. As reserves get scarce, the quarter-end bumps get *bigger and last longer*. So the trend in quarter-end behavior is a leading indicator: **growing quarter-end spikes are the system telling you its buffer is shrinking, even before any crisis.** The bump itself is noise; the *trend in the bumps* is signal.

## How it shows up in real markets

The framework earns its keep in real episodes. Three stand out, each a different flavor of the same plumbing.

### September 2019: the classic three-force spike

We have referenced it throughout; here is the full anatomy. By mid-September 2019, **reserves had been drained to ~\$1.47 trillion** by QT and a rising TGA — below the system's comfortable minimum, though nobody knew the exact line. On September 16-17, a **corporate tax date** pulled cash to the Treasury *and* a wave of **Treasury settlement** dumped fresh collateral needing funding. All three forces converged with no slack left. Overnight repo printed near **10%**, and even the secured SOFR jumped well above the policy rate. The Fed restarted overnight repo operations within a day, began outright bill purchases in October (a "reserve management" program it was careful not to call QE), and the spike subsided. The lasting legacy was the SRF: a permanent ceiling so the improvisation would never be needed again.

![Step chart of the fed funds upper bound from 2019 to 2025 with the policy rate SOFR tracks](/imgs/blogs/money-market-plumbing-repo-collateral-sofr-7.png)

The policy-rate chart above is the backdrop SOFR lives against. In September 2019 the policy ceiling was just **2.25%** (the red dashed line) — so a 10% repo print was a rate nearly *five times* the policy rate, a violent dislocation. Trace the rest of the path: the emergency cut to 0.25% in March 2020, the aggressive 2022-23 hiking cycle to 5.50%, and the cuts that began in late 2024. SOFR shadows this entire staircase, normally sitting a few basis points off the line. The early-warning signal is never the level of the policy rate; it is **SOFR's distance from it** — and that distance going vertical is the alarm.

### March 2020: the dash for cash

The COVID shock was a different mechanism but the same plumbing. In March 2020, a global **dash for cash** hit every market at once: everyone wanted dollars, and they wanted them *now*. Investors sold even US Treasuries — the supposedly safest asset — to raise cash, which is the opposite of the usual flight-to-safety and a sign of genuine plumbing failure. Treasury market liquidity evaporated, repo strained, and the levered basis-trade funds were caught: they could not roll their funding and were forced to dump Treasuries, which deepened the price collapse in a vicious loop. The Fed's response was overwhelming and fast — unlimited QE, swap lines to foreign central banks, and direct backstops — precisely because it had learned from 2019 that funding markets can seize in hours, not days. The episode cemented that **the repo and Treasury funding markets are the load-bearing wall of the entire financial system**: when they crack, the Fed throws everything at them immediately.

### The quarter-end heartbeat and the QT drain

Between the dramatic episodes, the plumbing has a steady pulse you can read. Every quarter-end, especially year-end, repo rates bump up as banks shrink balance sheets for reporting — a reliable, predictable heartbeat. Through 2022-24, that heartbeat was muffled by the enormous cash glut: with **\$2.55 trillion** in the RRP and reserves above \$3 trillion, the system had so much slack that quarter-ends barely registered. But as the RRP drained toward empty and QT pulled reserves down, traders watched the quarter-end bumps for signs of *re-tightening* — bigger or longer bumps signaling the buffer was finally getting thin. This is the slow, unglamorous, *daily* version of repo-watching, and it is where the early-warning system actually lives most of the time: not in the spikes, but in the gradual changes in how the plumbing handles routine pressure.

## How to trade it: the playbook

Here is the part that pays. Repo is rarely a *direct* trade for most market participants — you are usually not lending overnight cash yourself. Its value is as the **earliest and most reliable funding-stress dial in macro**, and you use that dial to position the rest of your book. The signals are public, free, and updated daily by the New York Fed.

**The dashboard — check these in this order:**

1. **SOFR minus IORB (the master gauge).** This is your funding-stress needle. Negative or near-zero means ample cash, calm plumbing — risk-on conditions are *supported* by the funding layer. Persistently positive and rising means cash is getting scarce — the funding layer is starting to strain, and that is a yellow flag for every levered position in the market. A blowout means acute stress and likely Fed intervention. Watch the *trend*, not just the level.

2. **ON RRP balance (the glut/buffer gauge).** A fat RRP (trillions) is a big shock absorber — funding spikes are nearly impossible while it exists. Track the *drain*: as the RRP empties toward zero, the buffer is being consumed, and the system loses its ability to absorb shocks silently. A near-empty RRP shifts you to high alert on the other gauges.

3. **Bank reserves (the master dial).** Falling reserves — driven by QT and a rising TGA — move the system toward the scarcity zone. You will not always know the exact scarcity line, but you know the *direction*. Reserves falling while the RRP is already empty is the textbook setup for a 2019-style event. This is the slow-moving structural backdrop; the SOFR−IORB spread is the fast confirmation.

4. **SRF usage (the ceiling gauge).** Any meaningful, non-quarter-end SRF usage is a direct readout that private funding got expensive. An empty SRF is reassuring; a busy SRF says the ceiling is being tested.

**The positions these signals inform:**

- **When the funding layer is calm (SOFR ≤ IORB, fat RRP, ample reserves):** the plumbing is *supporting* risk. This is not itself a buy signal, but it removes a major tail risk — leverage is cheap and available, so a funding-driven crash is off the table for now. You can hold risk with less fear of a plumbing accident.
- **When the funding layer is tightening (SOFR pushing above IORB, RRP draining, reserves falling):** the plumbing is becoming a *risk*. Trim leverage, widen your stops, and be especially wary of the most funding-sensitive trades — anything that relies on cheap, available overnight financing (the basis trade is the canonical example). Rising funding stress is a headwind that hits levered and crowded positions *first*.
- **When the funding layer cracks (SOFR spike, busy SRF, repo dislocation):** this is the de-levering moment. Historically the Fed responds fast and hard, which means the *initial* spike is a panic-low for risk assets that the Fed then reverses. The aggressive playbook is to fade the panic *once the Fed signals it is stepping in* (emergency repo operations, RRP/SRF activity, asset purchases). The conservative playbook is simply to be the one who *already de-levered* before the crack, using the earlier signals — so you are a buyer of the dip, not a forced seller into it.

**The invalidation.** The whole framework assumes the corridor holds — that the Fed defends the floor and ceiling. The framework is *invalidated* if the Fed deliberately changes the regime: if it moves to a scarce-reserve "corridor" system on purpose, or if it removes a facility, the thresholds shift and you must re-anchor. It is also weakened in a true solvency crisis (2008-style), where the problem is *credit*, not cash, and even fully secured repo against Treasuries can strain because nobody trusts anybody — there, you watch credit spreads alongside repo. And the single most common way to be wrong is **complacency**: assuming "the Fed has it handled" and staying levered into a tightening funding picture. The Fed caps the *rate*; it does not cap the *forced de-levering* that a scare triggers among the most leveraged players, and that de-levering is what actually moves your P&L.

The one habit to build: **make the SOFR−IORB spread and the RRP balance part of your daily macro check, right next to the S&P and the dollar.** They are leading indicators of funding conditions, they are free, and they flash before equities do. Most traders ignore the plumbing until it bursts. The ones who read it are positioned before the burst — and that, in the end, is the entire edge of understanding money-market plumbing.

## Further reading & cross-links

- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the deeper structural picture of how non-bank lending and repo intertwine, and why this plumbing sits outside the traditional banking system. The natural next read after this post.
- [Central bank balance sheet: net liquidity, reserves, RRP, and TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) — how QT, the TGA, and the RRP combine into the net-liquidity framework that governs the reserve level we charted here.
- [What liquidity means for market and funding](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) — the broader map of funding liquidity versus market liquidity, and how the two reinforce each other in a crisis.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the mechanics of IORB, the corridor, and how administered rates steer the market rates SOFR and fed funds live among.

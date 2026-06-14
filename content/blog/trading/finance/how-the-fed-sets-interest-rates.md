---
title: "How the Fed Sets Interest Rates — and Why the Whole World Holds Its Breath"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into how the Federal Reserve pins one overnight rate and how that single number reaches your mortgage, your savings, and currencies worldwide."
tags: ["federal-reserve", "interest-rates", "monetary-policy", "fomc", "central-banking", "inflation", "bonds", "macro", "dollar", "fed-funds-rate"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The Fed does not decree all interest rates; it pins one overnight rate using a system of administered rates, and that single number propagates through markets to your mortgage, your savings, and currencies worldwide.
>
> - The **federal funds rate** is the rate banks charge each other for overnight loans. It is the only rate the Fed actually targets. Everything else follows from it.
> - Since 2008 the Fed pins that rate with a **floor system**: it pays interest on reserves (IORB) as an effective ceiling and offers an overnight reverse-repo (ON RRP) rate as a floor. The old method of "scarce reserves" is gone.
> - The Fed controls the **short end** of the rate spectrum. The bond market, betting on inflation and growth, sets the **long end** — which is why mortgages can rise even when the Fed cuts.
> - A rate hike is not "good" or "bad" — it **transfers income from borrowers to savers**, reprices every balance sheet, lifts the dollar, and can trigger crises in countries that borrowed in dollars.
> - The one number to remember: the policy rate moves in **basis points** (a basis point is one-hundredth of a percent), and a single 0.25% move ripples into trillions of dollars of repriced debt.

You have probably watched it happen. On a Wednesday afternoon, roughly every six weeks, a man in a suit walks to a podium, reads a few paragraphs, takes questions for forty-five minutes, and somewhere in there says whether a single number is going up, down, or staying put. By the time he finishes, the stock market has lurched, the value of the dollar against the euro has moved, mortgage rates have twitched, and a finance ministry in Jakarta or Buenos Aires is on the phone with its central bank. All of that, from one rate that almost no ordinary person ever pays directly.

How does a single overnight interest rate — one that applies to loans banks make to each other, often just for one night — end up steering the cost of a house in Ohio, the yield on a pension fund in Tokyo, and the solvency of a government in Africa? The diagram above is the mental model for the whole post: the Fed pins one rate, and that one number fans out through the banking system into every loan, every deposit, and every currency on the planet. Once you see the fan-out, the daily noise about "what the Fed will do" stops being mysterious and starts being mechanical.

![How one Fed rate reaches your mortgage and the dollar](/imgs/blogs/how-the-fed-sets-interest-rates-1.png)

This post builds that picture from the ground up. We start with what an interest rate even *is*, define the federal funds rate from zero, walk through who actually makes the decision, and then — the part most explainers skip — show the surprisingly plumbing-heavy machinery the Fed uses to *force* the rate to where it wants it. From there we trace transmission outward to your wallet, look at how the Fed manages expectations as a tool in itself, and end with the real episodes where this all played out, sometimes violently. No finance background assumed. Every term is defined the first time it appears.

## Foundations: what an interest rate actually is

Before the Fed, before banks, before any of the acronyms, there is one idea: **money now is worth more than money later.**

If I offer you \$100 today or \$100 a year from now, you take it today — and not only because you are impatient. The \$100 today can be put to work: lent out, invested, or spent before prices rise. So if I want to *borrow* your \$100 for a year, I have to pay you something extra to compensate you for waiting. That extra is **interest**. An **interest rate** is simply the price of using someone else's money over time, quoted as a percentage of the amount borrowed per year. If you lend \$100 and get back \$105 in a year, the interest rate was 5%.

That is the entire concept. Everything else is detail layered on top — but the details matter, so let us define the ones that recur throughout this post, each on first use.

A **basis point** (often written "bp" and pronounced "bip") is one-hundredth of one percent. So 0.01% is one basis point, and a "25 basis point hike" is a rate increase of 0.25%. The Fed almost always moves in multiples of 25 basis points, so this is the natural unit of monetary policy. When you hear "the Fed went 50 bps," that means a half-percentage-point move. Professionals use basis points precisely because saying "a quarter of a percent" gets clumsy fast, and because the difference between 4.00% and 4.25% — a mere 25 bps — is worth billions of dollars across the economy.

A **nominal interest rate** is the rate as quoted: the 5% on your savings account, the 7% on your mortgage. A **real interest rate** strips out inflation — the rate of increase in the general price level — to tell you the *purchasing power* you actually gain. The quick rule: real rate is roughly the nominal rate minus the inflation rate. If your savings pay 5% but prices are rising 3% a year, your money grows 5% in dollars but only about 2% in what those dollars can buy. That 2% is the real return, and it is what actually matters for whether you are getting richer.

A **yield** is the return an investor earns on a bond or other security, expressed as an annual percentage. For a bond, the yield is closely tied to its price: when a bond's price falls, its yield rises, and vice versa, because a fixed stream of future payments is now bought for fewer dollars. We will lean on this seesaw later, because it is how Fed decisions ripple into the trillions of dollars of bonds the world holds.

### The federal funds rate: the rate the Fed actually targets

Here is the rate at the center of everything. Banks in the United States are required to settle payments through accounts they hold at the Federal Reserve. At the end of each day, some banks find themselves with more cash in their Fed account than they need, and others find themselves short. The banks with extra lend it overnight to the banks that are short. The interest rate on those overnight, bank-to-bank loans is the **federal funds rate** — "fed funds" for short.

It is worth pausing on how narrow this rate is. It applies to a specific market: overnight loans of reserves between banks. You will never borrow at the fed funds rate. Your bank does, indirectly, and the rate you pay is built on top of it. The Federal Reserve does not set the fed funds rate by law. Instead, it announces a **target range** — a 0.25-percentage-point band, such as 4.25% to 4.50% — and then uses tools (which we will dissect) to keep the actual market rate inside that band. The rate that actually trades, the volume-weighted average of all those overnight loans, is the **effective federal funds rate** (EFFR), and it usually sits right in the middle of the band.

This is the crucial distinction the rest of the post depends on: there is a **policy rate** (the one number the Fed targets) and there are **market rates** (the thousands of rates set by supply and demand for credit across the economy — mortgages, corporate bonds, car loans, the yield on a 10-year Treasury). The Fed directly controls only the policy rate. It *influences* all the market rates, but it does not set them. Confusing those two is the single biggest source of misunderstanding about the Fed, and untangling them is half of what this post does.

### The dual mandate: what the Fed is trying to achieve

Why does the Fed move this rate at all? Because the U.S. Congress gave it a job, written into law, known as the **dual mandate**: pursue **maximum employment** and **stable prices**. In practice the Fed has translated "stable prices" into an explicit target of about **2% inflation** per year, measured over the long run. ("Why 2% and not 0%?" — a small, steady inflation gives the Fed room to cut rates in a downturn and reduces the risk of deflation, a fall in prices that can freeze an economy. We will not litigate the number here; just take 2% as the goal.)

These two goals can pull in opposite directions, and that tension is the whole drama of monetary policy. When the economy runs hot — lots of jobs, wages and prices rising fast — the Fed raises rates to cool borrowing and spending, accepting some weakness in the job market to bring inflation down. When the economy weakens — layoffs rising, spending falling — it cuts rates to make borrowing cheap and revive activity, accepting some inflation risk to protect jobs. The fed funds rate is the single lever it pulls to lean one way or the other. Every meeting is, at heart, a judgment about which side of the mandate needs more attention right now.

With those foundations in place — interest as the price of time, basis points as the unit, real versus nominal, yields, the fed funds rate as the policy rate, and the dual mandate as the goal — we can look at who actually makes the call.

## The FOMC: who decides, and what they publish

The decisions get made by a committee called the **Federal Open Market Committee** (FOMC). It is worth knowing exactly who sits on it, because the composition is a deliberate balance between centralized and regional power that shapes how policy is set.

The FOMC has **19 participants** but only **12 votes** at any meeting. The participants are:

- The **seven members of the Board of Governors**, based in Washington, D.C. They are nominated by the President and confirmed by the Senate to 14-year terms, and they vote at every meeting. One of them is the **Chair** (currently the public face of the Fed), and another is the **Vice Chair**.
- The **twelve presidents of the regional Federal Reserve Banks** (New York, Chicago, San Francisco, and so on). These banks are quasi-private institutions overseen by the Fed system. All twelve presidents attend every meeting and join the discussion, but only **five vote** at any given time: the president of the New York Fed always votes (because New York runs the market operations), and the other four votes rotate among the remaining eleven banks on a yearly schedule.

So: 7 governors who always vote + 5 reserve-bank presidents who vote on rotation = **12 voters**, drawn from a table of 19 voices. This design means Washington has a built-in majority, but the regions always have a seat and a vote, so policy is never set in a vacuum sealed off from the rest of the country.

The FOMC meets **eight times a year**, roughly every six weeks, on a schedule published far in advance. Each meeting runs two days. The committee reviews a mountain of economic data, debates, and then votes on the target range for the federal funds rate. The diagram below shows the cycle from the data that comes in to the market reaction that goes out.

![The FOMC decision cycle, eight times a year](/imgs/blogs/how-the-fed-sets-interest-rates-3.png)

What the committee *publishes* matters as much as the rate decision itself, because markets trade on the whole package, not just the number:

1. **The Statement.** A short document, released at 2:00 p.m. Eastern on the second day, announcing the new target range and a few paragraphs of carefully chosen language describing the economy and the committee's stance. Traders parse it word by word; a single changed adjective ("solid" becoming "moderate") can move billions.

2. **The Summary of Economic Projections (SEP)**, published four times a year. This is where each of the 19 participants writes down, anonymously, where they think the economy and the appropriate rate are heading. Its most famous component is the **dot plot**: a chart on which each participant places a dot showing where they expect the fed funds rate to be at the end of this year, next year, and the year after. It is not a promise or a plan — it is a snapshot of 19 individual forecasts — but markets treat the median dot as a strong signal of the committee's collective expectation.

3. **The press conference.** After every meeting, the Chair gives a live press conference, reading a statement and then taking reporter questions for around 45 minutes. This is often where the real market moves happen, because the Chair's tone and word choices reveal how the committee is *leaning* beyond what the written statement says. A "hawkish" Chair (leaning toward higher rates to fight inflation) or a "dovish" one (leaning toward lower rates to support jobs) can swing markets even when the rate itself is unchanged.

The point of all this machinery is that the FOMC is not just setting a number; it is setting *expectations* about all the future numbers. We will return to why that is half the tool. First, the part almost everyone gets wrong: how the Fed actually forces the rate to its target.

## How the Fed actually pins the rate: the floor system

Here is where most explanations wave their hands. They say the Fed "sets" the rate, as if it flips a switch. It does not. The fed funds rate is a *market* rate — banks lending to each other — so the Fed has to make that market settle where it wants. The way it does this changed completely after the 2008 financial crisis, and understanding the change is the key to understanding modern monetary policy.

### The old way: scarce reserves and open-market operations

Before 2008, the Fed kept the banking system on a tight leash. Banks held only a small amount of reserves — cash in their Fed accounts — relative to the total they wanted, so reserves were *scarce*. Because reserves were scarce, their price (the fed funds rate) was very sensitive to small changes in supply. The Fed exploited this. Every morning, the New York Fed would buy or sell government securities in the open market — **open-market operations** — adding or draining tiny amounts of reserves to nudge the fed funds rate to the target. Add reserves, the rate falls; drain reserves, the rate rises. It was like adjusting the water level in a small tank: a little in or out moved the level a lot.

The way this worked relied on reserves being kept artificially tight. It was elegant but fragile, and it broke in the crisis.

### The new way: ample reserves and administered rates

After 2008, the Fed flooded the system with reserves through large-scale asset purchases (more on that later). Reserves went from scarce to **ample** — banks now hold far more than they strictly need. In an ample-reserves world, the old trick stops working: adding or draining a few billion dollars no longer moves the rate, because the tank is now an ocean. So the Fed switched to a different method entirely. Instead of controlling the *quantity* of reserves, it now sets a few **administered rates** — rates the Fed itself offers or charges — that pen the market rate into a band from above and below. This is called a **floor system**, or a corridor.

The diagram shows the corridor: think of the fed funds rate as a fish trapped between a ceiling it cannot rise above and a floor it cannot sink below.

![The rate corridor that the Fed uses to pen the funds rate](/imgs/blogs/how-the-fed-sets-interest-rates-2.png)

Reading the corridor from top to bottom, here are the administered rates and why each one bites:

**The discount rate (the penalty ceiling, around 4.50%).** The Fed runs a lending facility called the **discount window** where banks can borrow directly from the Fed, against good collateral, at the discount rate. This rate is set slightly *above* the top of the target range, so it acts as a backstop ceiling. No bank would borrow from another bank at a rate higher than what the Fed charges at the window — why pay a peer 5% when the Fed will lend to you at 4.50%? In practice banks rarely use the window (there is a stigma to it), but its existence caps how high the fed funds rate can spike.

**Interest on reserve balances (IORB, the effective ceiling, around 4.40%).** This is the workhorse of the modern system. The Fed *pays* banks interest on the reserves they hold in their Fed accounts. Now think like a bank. If the Fed pays you 4.40% to do nothing — just leave your money parked safely at the Fed — why would you ever lend that money to another bank for *less* than 4.40%? You would not. So IORB sets a floor under what banks are willing to accept, which in the fed funds market acts as an effective ceiling on the rate, because the biggest, safest lenders simply will not go below it. This is the rate the Fed adjusts to move the whole corridor up or down.

**The ON RRP rate (the floor, around 4.25%).** Not everyone with cash to park overnight is a bank that earns IORB. Money market funds, government-sponsored enterprises, and others also have huge piles of cash. For them, the Fed runs the **overnight reverse repurchase agreement** (ON RRP) facility: these institutions can lend cash to the Fed overnight and earn the ON RRP rate, fully safe. So no one with access to ON RRP will lend in the open market for *less* than the ON RRP rate — why accept 4.10% from a private borrower when the Fed offers 4.25% risk-free? That sets a hard floor under short-term rates.

Put the three together and you have a pen. Lenders will not go below the floor (ON RRP) because the Fed itself offers more. Banks will not lend below IORB because the Fed pays them that much to do nothing. And no one borrows above the discount rate because the Fed lends there. The actual fed funds rate is forced to settle inside the band — usually a hair below IORB and above ON RRP. When the Fed wants to "raise rates," it simply raises all these administered numbers by, say, 25 basis points, and the whole corridor — and the market rate inside it — slides up with it. No frantic buying and selling of bonds required. It is, in a sense, the Fed setting prices by decree on the narrow set of rates it directly offers, and letting arbitrage do the rest.

#### Worked example: why a bank won't lend below IORB or borrow above the discount rate

Let us make the corridor concrete with a bank's actual choice. Suppose IORB is 4.40%, ON RRP is 4.25%, and the discount rate is 4.50%. Big Bank has \$1,000,000,000 (one billion dollars) of spare reserves sitting in its Fed account for one night.

- **Option A — leave it at the Fed.** It earns IORB: \$1,000,000,000 times 4.40% divided by 365 days = about \$120,548 for the night. Risk-free.
- **Option B — lend it to another bank overnight at 4.30%.** It earns \$1,000,000,000 times 4.30% divided by 365 = about \$117,808. That is \$2,740 *less* than just leaving it at the Fed, and it now carries the (small) risk that the borrower fails.

No rational bank takes Option B. Lending below 4.40% means giving up free money. So the fed funds rate cannot fall much below IORB — the lenders refuse. Now flip it: suppose Small Bank is short and needs to borrow \$1,000,000,000 for the night. A peer offers to lend at 4.55%. But the Fed's discount window will lend at 4.50%. Borrowing from the peer at 4.55% costs \$124,658 for the night versus \$123,288 at the window — \$1,370 more, for no reason. So Small Bank goes to the window, and the rate cannot rise much above the discount rate. The intuition: the Fed does not chase the market rate around; it builds a pen out of its own administered rates and the market walks itself inside.

One subtlety is worth naming, because it shows why the floor system is more robust than the old scarce-reserves regime. In the pre-2008 world, the Fed had to hit its target by fine-tuning the *quantity* of reserves every single day; if its estimate of demand was off, the rate would miss. A surprise spike in banks' demand for reserves — a tax date, a quarter-end, a panic — could send the fed funds rate jumping well above target before the Fed could react. That fragility was on full display in September 2019, when a brief shortage of reserves sent overnight rates spiking toward 10% intraday, forcing an emergency injection of cash. The floor system is sturdier precisely because it does not depend on guessing the quantity right: as long as reserves are *ample*, the administered rates do the pinning, and the exact quantity of reserves stops mattering for the rate. The Fed trades the daily precision of the old system for the structural stability of the new one — and after 2019, it has erred toward keeping reserves comfortably ample rather than risk another spike.

## Transmission: from the policy rate to your wallet

So the Fed pins one overnight rate. How does that reach a 30-year mortgage, a 5-year car loan, or a savings account? Through a chain of arbitrage and competition, each link of which is worth tracing. Look back at the very first diagram — the fan-out — as the map for this whole section.

**Step 1: policy rate to short-term market rates.** The fed funds rate is overnight. The most important benchmark it anchors is **SOFR** (the Secured Overnight Financing Rate), which is the rate at which financial institutions borrow cash overnight using Treasury securities as collateral. SOFR is now the reference rate for trillions of dollars of contracts (it replaced the old LIBOR). Because SOFR and fed funds are both overnight rates competing for the same overnight cash, they move almost in lockstep with the policy rate. When the Fed lifts the corridor, SOFR follows within hours.

**Step 2: short rates to the bank prime rate.** The **prime rate** is the rate banks charge their most creditworthy corporate customers, and by convention it is set at exactly **fed funds plus 3 percentage points**. So when the target range midpoint is around 4.50%, the prime rate is 7.50%, and the moment the Fed hikes 25 bps, every major bank moves prime up 25 bps the same day. Prime is mechanical — it is the most direct, immediate transmission of a Fed move into a consumer-facing rate.

**Step 3: prime rate to consumer loans that float.** Many consumer rates are written as "prime plus a margin." A **credit card** might charge prime + 12%, so when prime is 7.50% the card's annual rate is 19.50%. A **home equity line of credit** might be prime + 1%. An **adjustable-rate auto loan** floats similarly. These rates move almost immediately when the Fed moves, because they are contractually tied to prime. This is the fastest part of transmission — a Fed hike shows up on your credit card statement within a billing cycle or two.

**Step 4: short rates to deposit and savings rates.** Banks compete for your deposits. When the Fed raises rates, banks can earn more by parking money at the Fed (IORB) or lending it out, so they can afford to — and competition eventually forces them to — pay more on savings accounts and **certificates of deposit** (CDs, which are time deposits that lock your money up for a fixed term in exchange for a higher rate). This link is real but *sticky*: banks raise deposit rates slowly and reluctantly, which is why your savings rate often lags the Fed by months. Online banks, hungry for deposits, move faster than big brick-and-mortar banks.

**Step 5: the long end and mortgages — the link that surprises people.** Here transmission gets subtle, and it is the source of the most common confusion about the Fed. A 30-year fixed mortgage rate is *not* set off the overnight rate. It tracks the **10-year Treasury yield** plus a spread (typically 1.5 to 3 percentage points). And the 10-year yield is a *market* rate set by investors betting on average inflation and growth over the next decade — not by the Fed's overnight decision. The Fed influences it (by shaping expectations of the whole future path of short rates), but does not control it. We will give this its own section, because "the Fed controls the short end, the market controls the long end" is the single most important nuance in the entire subject.

**Step 6: short rates to corporate borrowing and equity valuations.** Companies borrow by issuing bonds or taking bank loans, both priced off the benchmark rates that move with the Fed. Higher rates raise the cost of expansion, buybacks, and acquisitions. And rates feed stock prices through valuation: a share is worth the present value of its future profits, and *discounting* those future profits back to today uses an interest rate. Higher rates mean future profits are worth less today, so all else equal, higher rates push stock valuations down — hitting high-growth companies (whose profits are far in the future) hardest.

**Step 7: rates to the dollar.** Higher U.S. rates make dollar assets pay more, attracting global capital into dollars and pushing the dollar's exchange value up. A stronger dollar makes imports cheaper for Americans and U.S. exports more expensive abroad — and, as we will see, can wreck countries that owe debt in dollars.

#### Worked example: a hike from 4.50% to 5.00% on a \$300,000 30-year mortgage

Let us run real numbers on the link people feel most. Suppose you are buying a house with a \$300,000, 30-year fixed mortgage, and your rate would have been 4.50% but rates rise so it is now 5.00% — a 50-basis-point increase.

The standard mortgage payment formula (principal and interest only) is:

```
payment = P * r / (1 - (1 + r)^(-n))

P = loan amount = 300000
r = monthly rate = annual rate / 12
n = number of payments = 30 * 12 = 360
```

At 4.50%: the monthly rate r = 0.045 / 12 = 0.00375. Running the formula gives a monthly payment of about **\$1,520**.

At 5.00%: the monthly rate r = 0.05 / 12 = 0.0041667. The formula gives a monthly payment of about **\$1,610**.

The difference is roughly **\$90 a month**. That sounds small until you annualize and compound it over the life of the loan: \$90 times 12 months times 30 years = about **\$32,400** in extra payments over the full term. A half-point move in rates — half of what the Fed often does in a *single* meeting — costs this borrower the price of a car over the loan's life. The intuition: small-sounding rate moves are enormous when applied to large balances over long horizons, which is exactly why the whole world watches the Fed.

#### Worked example: \$10,000 in savings going from 0.5% to 4.5% APY

Now the upside of higher rates — the saver's side. **APY** means annual percentage yield: the actual yearly return after compounding. Suppose you have \$10,000 in a savings account.

- **At 0.5% APY** (typical during the near-zero-rate years): after one year you earn \$10,000 times 0.005 = **\$50**. After five years, compounding, your balance is \$10,000 times (1.005)^5 = about **\$10,253** — you made \$253 in five years.
- **At 4.5% APY** (typical after the 2022–23 hikes): after one year you earn \$10,000 times 0.045 = **\$450**. After five years, your balance is \$10,000 times (1.045)^5 = about **\$12,462** — you made \$2,462.

The same \$10,000, the same five years, but **\$2,462 instead of \$253** — nearly ten times the gain — purely because the Fed moved its rate. The intuition: when the Fed hikes, it is quietly handing income to everyone who holds cash and savings, at the direct expense of everyone who borrows. A rate decision is a giant, invisible transfer between the two groups.

#### Worked example: real versus nominal — a 5% rate when inflation is 3%

People obsess over the nominal rate and forget the real one, which is what actually changes their wealth. Suppose your savings pay **5% nominal** and inflation is running at **3%**.

- In dollars, \$10,000 becomes \$10,500 after a year. You have \$500 more.
- But prices rose 3%, so the basket of goods that cost \$10,000 last year now costs \$10,300.
- Your \$10,500 buys you \$10,500 / 1.03 = about \$10,194 worth of last year's goods.

So your **real** gain is about \$194, or roughly **2%** — exactly the nominal rate (5%) minus inflation (3%). Now flip the scenario: in 2021, savings paid about 0.5% while inflation spiked to about 7%. A saver earning 0.5% nominal had a real rate of about *negative 6.5%* — their money was *losing* purchasing power fast even as it nominally grew. The intuition: a high nominal rate during high inflation can still be a losing real return, and a low nominal rate during deflation can be a great one. Always subtract inflation before deciding whether a rate is "good."

## The short end and the long end: who controls what

This is the most important nuance in the subject, so it gets its own section and its own figure. The Fed controls the **short end** of the rate spectrum — overnight to a couple of years — almost completely, because it directly sets the overnight rate and the market prices very-short maturities off the expected path of that rate. But the **long end** — 10-year and 30-year rates — is set by the bond market, and the Fed only influences it.

![The Fed owns the short end while markets own the long end](/imgs/blogs/how-the-fed-sets-interest-rates-5.png)

Why the split? A 10-year Treasury yield is, at root, the market's best guess of the *average* overnight rate over the next ten years, plus a small premium for tying up money that long. The Fed sets today's overnight rate, but it cannot dictate what investors believe the *average* rate will be across the next decade — that depends on the market's forecast of inflation, growth, and Fed behavior over a span longer than any Chair's term. So the long end is a vote, taken continuously by millions of investors, on the economy's distant future. The Fed gets one dot on a ten-year chart; the market draws the rest of the line.

This produces effects that baffle people:

- **The Fed can cut, and mortgage rates can rise.** If the Fed cuts the overnight rate but the bond market thinks the cut will reignite inflation, long-term yields can *rise* on the news — and since mortgages track the 10-year yield, mortgage rates go up even as the Fed eases. The Fed pushed the short end down; the market pushed the long end up.
- **The yield curve can invert.** Normally longer-term rates are higher than short-term ones (you demand more to lend for longer). But when the Fed jacks up short rates to fight inflation, and the market expects those high rates to cause a recession that will *force* future cuts, short rates can exceed long rates. This is an **inverted yield curve**, and it has preceded most U.S. recessions — it is the bond market saying "the Fed is tight now, but it will have to cut later."

The practical takeaway for a regular person: when you read "the Fed raised rates," do not assume your mortgage quote just went up. Mortgages had likely already moved weeks earlier, when the *bond market* repriced its expectations. The Fed's actual decision is often the least surprising part, because the market front-runs it.

## Forward guidance and the dot plot: managing expectations is half the tool

If the Fed only controls one overnight rate, but the economically important rates are long-term rates set by expectations, then the Fed's most powerful lever is not the rate move itself — it is **shaping what markets expect about all the future rate moves.** This is called **forward guidance**, and modern central banking treats it as co-equal with the rate decision.

Consider why this is so potent. The 10-year yield reflects the expected average short rate over ten years. The Fed cannot set ten years of overnight rates today, but it can *talk* about the path. If the Chair credibly signals "we expect to keep raising and hold rates high for a long time," the market revises its expected average upward, and long-term yields rise *immediately* — before the Fed has actually done anything. The Fed moved the long end with words. Conversely, in a crisis, a promise to keep rates near zero "for an extended period" can pull long-term yields down without a single additional cut.

The **dot plot** is the formalized version of this. By publishing where each participant expects rates to go, the Fed gives the market a quantified path to price against. But it is a double-edged tool. The dots are forecasts, not commitments, and when reality diverges, the Fed must walk them back — which can jolt markets. In late 2021, the dots showed only modest hikes ahead; within months, surging inflation forced the Fed into the fastest tightening in four decades, far above where the dots had pointed. The lesson the Fed itself drew was that over-precise guidance can backfire when the economy surprises.

There is a deep irony here worth sitting with: the Fed's power over the economy comes less from the mechanical effect of one overnight rate and more from its **credibility** — the market's belief that it will do what it says about inflation. A central bank that has earned that credibility can move long-term rates and inflation expectations with a paragraph. One that has lost it (think of episodes of runaway inflation) cannot, no matter how high it pushes the overnight rate. Expectations are the real channel; the rate is just the instrument the Fed uses to make its expectations believable.

## The balance sheet: the Fed's second lever

The fed funds rate is the Fed's primary tool, but it has a second, blunter one: the size and composition of its own **balance sheet** — the assets it owns (mostly Treasury bonds and mortgage-backed securities) and the reserves it has created to buy them.

When short-term rates hit zero — the **zero lower bound**, the point below which the Fed is reluctant to cut because negative rates create their own problems — the Fed cannot ease further by cutting. So it eases by *buying* long-term bonds in huge quantities, creating new reserves to pay for them. This is **quantitative easing** (QE). By buying up long-term bonds, the Fed pushes their prices up and their yields down, directly pressing on the long end it cannot reach with the overnight rate. QE is, in effect, a way to ease policy after the overnight rate is already at zero. The reverse — letting bonds mature without replacing them, shrinking the balance sheet — is **quantitative tightening** (QT), which gently lifts long-term yields and drains reserves.

The balance sheet is a big topic in its own right, and the mechanics of money creation behind it deserve their own treatment. For the full picture of how the Fed prints reserves and what that does and does not mean, see the companion posts on [quantitative easing](/blog/trading/finance/quantitative-easing-explained-printing-money) and [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier). For this post, hold onto one idea: the Fed has a short-end lever (the policy rate) and a long-end lever (the balance sheet), and it reaches for the second only when the first is exhausted at zero.

## Global spillovers: why the whole world watches

The Fed is the central bank of the United States, accountable to the U.S. Congress, with a mandate about U.S. employment and U.S. inflation. And yet a Fed meeting is front-page news in São Paulo, Istanbul, and Mumbai. Why? Because the dollar is the world's reserve currency — the currency in which a huge share of global trade is invoiced, global commodities are priced, and, critically, global debt is denominated. When the Fed moves the price of dollars, it moves the cost of money for everyone who touches dollars, which is nearly everyone.

The diagram traces the chain from a U.S. hike to a crisis halfway around the world.

![How a US rate hike becomes an emerging-market crisis](/imgs/blogs/how-the-fed-sets-interest-rates-7.png)

Walk the chain. The Fed raises U.S. rates. Now U.S. assets — Treasury bonds, dollar deposits — pay more and look like the safest high-yield game in town. Global capital that had flowed into higher-risk emerging markets chasing yield reverses course and flows *back* into dollars. Two things happen at once. First, the demand for dollars to buy those U.S. assets pushes the **dollar up** against other currencies. Second, the capital outflow drains money from emerging economies, raising their borrowing costs and weakening their currencies further.

Now the killer mechanism: many emerging-market companies and governments borrowed in *dollars* (because dollar debt was cheap and plentiful) while earning revenue in their *local currency*. When the dollar strengthens, their debt — fixed in dollars — becomes far more expensive to repay in local-currency terms, even though the dollar amount they owe has not changed. This is **currency mismatch**, and it has bankrupted countries.

#### Worked example: how a 1% US rate rise strengthens the dollar against a \$1,000 import bill

Make it concrete from the importer's side. Suppose a company in a country whose currency we will call the peso needs to pay a \$1,000 import bill, due in dollars. Initially the exchange rate is 20 pesos per dollar, so the bill costs 20,000 pesos.

Now the Fed raises U.S. rates by 1 percentage point. Capital flows into dollars, and the dollar strengthens against the peso — say the rate moves to 25 pesos per dollar (a 25% peso depreciation, the kind of move that happens fast in a stressed emerging market). The dollar bill is still \$1,000, but now it costs 25 pesos times 1,000 = **25,000 pesos**.

The importer's bill just rose by 5,000 pesos — a 25% jump in local-currency cost — without the dollar price changing at all, purely because the Fed moved its rate and the currency repriced. Multiply that across a country's entire dollar-denominated import bill and dollar debt load, and you see how a U.S. monetary decision becomes a balance-of-payments crisis abroad. The intuition: when you owe dollars but earn pesos, a stronger dollar is a tax you never voted for, levied by a central bank in another country.

This is the deep reason the world holds its breath at a Fed meeting. The Fed sets policy for America, but it sets the *price of the global reserve currency* for everyone, and there is no world central bank to appeal to.

## Common misconceptions

A handful of wrong beliefs about the Fed are so widespread that correcting them is worth a section of its own. Each is followed by why it is wrong.

**Misconception 1: "The Fed sets mortgage rates."** It does not. The Fed sets the overnight rate; mortgage rates track the 10-year Treasury yield plus a spread, and the 10-year yield is set by the bond market's expectations of inflation and growth over a decade. This is why mortgage rates can rise the day the Fed cuts, if the market reads the cut as inflationary. The Fed influences mortgages, sometimes strongly, but it does not set them — the link runs through the bond market, not by decree.

**Misconception 2: "The President controls the Fed."** The Fed is independent by design. The President nominates the governors and the Chair (and the Senate confirms them), but once confirmed, governors serve 14-year terms specifically so they cannot be fired for making unpopular rate decisions, and the Chair cannot be removed over a policy disagreement. This independence exists precisely because politicians always want lower rates before an election, and history is full of countries whose currencies collapsed when the printing press was handed to the politicians. The Fed answers to Congress (which created it and can amend its mandate) but takes no orders on individual rate decisions from any President.

**Misconception 3: "Lower rates always help everyone."** Lower rates help borrowers and hurt savers; higher rates do the reverse. A retiree living off interest income is devastated by near-zero rates that pay nothing on safe savings, while a young family with a mortgage is helped. Cheap money can also inflate asset bubbles, punish prudent savers, and stoke inflation that hits the poor hardest. There is no rate that is good for everyone — every level of rates redistributes income between groups, which is exactly why the politics of monetary policy are so fierce.

**Misconception 4: "The Fed prints the money it spends."** The Fed does not fund the government's spending; that is the job of Congress and the Treasury, financed by taxes and by selling Treasury bonds. The Fed creates *reserves* (electronic money in banks' Fed accounts) when it buys assets, but it buys those assets in the open market from banks and dealers, not from the Treasury directly, and it can drain those reserves by selling the assets back. "Money printing" is a loose metaphor for QE, but the Fed is not a checkbook the government writes spending against — the institutional separation between fiscal policy (Congress, Treasury) and monetary policy (the Fed) is deliberate and load-bearing.

**Misconception 5: "When the Fed hikes, my savings rate goes up the same day."** Deposit rates are the *stickiest* link in the chain. Banks raise loan rates immediately (especially anything tied to prime) but drag their feet raising deposit rates, because paying you more cuts their profit. After the 2022 hikes, many large banks still paid near-zero on checking accounts months later, while online banks and money-market funds offered 4–5%. The Fed sets the conditions, but competition — and your willingness to move your money — determines how much of the hike actually reaches your savings.

**Misconception 6: "A rate cut is stimulus that works right away."** Monetary policy works with **long and variable lags** — famously, six to eighteen months between a rate change and its full effect on the real economy. A cut today does not rescue a recession tomorrow; it nudges borrowing and spending over the following year-plus. This lag is why the Fed has to act on *forecasts*, not current data — by the time inflation or unemployment is obviously a problem, the policy response is already a year late. Much of the criticism the Fed takes ("too late to hike," "too slow to cut") comes from this unavoidable lag.

## How it shows up in real markets

Theory is clean; markets are not. Here are the episodes where everything above played out, sometimes painfully. Each one is a case study in transmission, expectations, or the cost of moving too fast or too slow. The timeline below maps four decades of cycles.

![Four decades of Fed rate cycles](/imgs/blogs/how-the-fed-sets-interest-rates-6.png)

### 1994: the bond massacre

Through 1993, the Fed held the funds rate at a low 3% to nurse the economy out of recession. Then, starting in February 1994, it began raising — and over the next twelve months roughly *doubled* the rate to 6%, including a startling 75-basis-point hike in a single meeting. The problem was not the destination but the surprise: markets had grown complacent, the Fed gave little warning, and bond prices crashed as yields shot up. (Recall the seesaw: rising yields mean falling bond prices.) Globally, investors holding long-term bonds lost an estimated \$1.5 trillion in market value. The episode bankrupted Orange County, California, which had bet on rates staying low, and helped trigger Mexico's peso crisis. The lesson the Fed absorbed was profound: **the surprise matters as much as the move.** A rate path the market has already priced in does little damage; a path it has not braced for is a shock. This episode is a direct ancestor of modern forward guidance — the Fed learned to telegraph its moves precisely so 1994 would not repeat.

### 2004–06: the "measured pace"

Having learned from 1994, the Fed ran the opposite playbook a decade later. From mid-2004 to mid-2006, it raised the funds rate seventeen times, each by exactly 25 basis points, from 1% to 5.25%, while repeatedly using the phrase "measured pace" to signal that more small, predictable hikes were coming. Markets were never surprised. But the episode exposed the short-end/long-end split in dramatic fashion: as the Fed raised short rates, long-term rates *barely moved* — the 10-year yield stayed stubbornly low. Then-Chair Greenspan called it a "conundrum." The market, betting on contained inflation and global demand for safe U.S. bonds, simply refused to push the long end up. The result was that mortgage rates stayed low even as the Fed tightened — feeding the housing boom that would end catastrophically. The lesson: **the Fed can lose control of the long end entirely**, and when it does, its tightening may not bite where it intends.

### 2008: the zero lower bound and the birth of the new toolkit

When the financial crisis hit, the Fed cut the funds rate from 5.25% in 2007 all the way to a range of 0–0.25% by December 2008 — and then ran out of room. You cannot cut much below zero. With the primary lever exhausted, the Fed reached for the balance sheet: successive rounds of QE, buying trillions in Treasuries and mortgage-backed securities to push long-term yields down. This is also when reserves went from scarce to ample, forcing the switch to the floor system of administered rates we walked through earlier. The 2008 episode is the hinge of modern monetary policy: it created both the ample-reserves world (and thus IORB and ON RRP) and the now-standard use of the balance sheet as a second lever. Everything about how the Fed operates today was reshaped in those months.

### 2015: liftoff

The Fed held rates near zero for *seven years*. In December 2015 it finally raised them — the first hike in over nine years — by a cautious 25 basis points, then waited a full year before the next one. This was forward guidance in its most careful form: the Fed spent years preparing markets for "liftoff," terrified of a repeat of 1994 or of the 2013 "taper tantrum" (when merely *mentioning* slowing QE spiked yields and rattled emerging markets). The slow, heavily telegraphed path showed how much the Fed had come to rely on managing expectations. It also showed the limits of caution: critics argued the Fed waited too long, leaving rates too low for too long into a recovering economy.

### 2022–23: the fastest hikes in 40 years

After the pandemic, inflation surged to levels not seen since the early 1980s — over 9% at its 2022 peak. The Fed, having initially called the inflation "transitory," was forced into the most aggressive tightening since the Volcker era: from near zero to about 5.25–5.50% in roughly sixteen months, including four consecutive 75-basis-point hikes. This is where the dot plot's limits showed: the late-2021 dots had pointed to gentle increases, and reality blew past them. The episode is a live demonstration of nearly everything in this post — the rapid transmission to mortgages (which roughly doubled toward 7%), the windfall to savers (CDs went from near-zero to 5%), the hit to growth stocks, and the global stress as the dollar surged. For the historical rhyme — a central bank crushing entrenched inflation with brutal rate increases and accepting the economic pain — see the deep dive on [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).

### Emerging-market crises: the 1997 Asian crisis and 2013 taper tantrum

The clearest demonstrations of global spillover are the crises that hit emerging markets when the Fed (or the prospect of Fed tightening) pulled capital back toward dollars. In 1997, after years of cheap dollar borrowing, a wave of capital flight hit Thailand, Indonesia, and South Korea; currencies collapsed, dollar debts exploded in local-currency terms, and the **currency mismatch** mechanism from our worked example played out across a region, requiring international bailouts. In 2013, the mere *hint* from the Fed that it might slow its bond purchases — the "taper tantrum" — sent yields up and triggered sharp capital outflows from India, Brazil, Turkey, Indonesia, and South Africa (dubbed the "Fragile Five"), forcing their central banks to hike rates to defend their currencies. The lesson recurs: **the Fed sets policy for America, but it sets the cost of the global reserve currency for everyone, and the most vulnerable are those who borrowed in dollars while earning in something else.**

### The redistribution every cycle creates

Step back from the individual episodes and notice the pattern in who is helped and hurt each time the Fed moves. The matrix below summarizes it for a hiking cycle.

![Who wins and who loses when the Fed raises rates](/imgs/blogs/how-the-fed-sets-interest-rates-4.png)

When rates rise: **savers win** (their deposits and CDs finally pay something — recall the \$10,000 going from \$253 to \$2,462 over five years). **Borrowers lose** (mortgages, cards, and auto loans all cost more — recall the extra \$90 a month on the \$300,000 mortgage). **Banks generally win**, because they can widen the spread between what they charge on loans and what they pay on sticky deposits. **The government loses**, because it must roll over its enormous debt at higher yields — at recent debt levels, the annual interest bill on U.S. federal debt has crossed roughly \$1 trillion, a number that grows with every hike. And **stocks generally lose**, because higher discount rates compress valuations, hitting future-heavy growth companies hardest. Every cut reverses all of these. The single most useful frame for any Fed decision is not "is this good or bad" but "who is this taking income *from*, and who is it giving income *to*."

## When this matters to you, and where to go next

You will never borrow at the federal funds rate, and you will probably never read an FOMC statement. But the rate the Fed pins reaches you anyway — through the rate on your mortgage and car loan, the yield on your savings and CDs, the value of your retirement portfolio, and the purchasing power of your dollars abroad. Knowing how it works changes how you read the news and how you make decisions.

A few practical takeaways:

- **When you hear "the Fed raised/cut rates," don't assume your loan or savings rate just changed by the same amount.** Floating consumer rates tied to prime move fast; deposit rates move slowly; mortgages already moved when the bond market repriced. The headline rate is the least surprising part.
- **Lock versus float is a bet on the long end, not the Fed.** If you are deciding whether to lock a mortgage rate, you are betting on where the 10-year Treasury yield (and inflation expectations) goes — not directly on the Fed's next meeting.
- **Always think in real terms.** A 5% savings rate during 7% inflation is losing you money; a 2% rate during 0% inflation is keeping you whole. Subtract inflation before judging any rate.
- **If you hold dollars and the Fed is hiking, the rest of the world is feeling it more than you are.** A strong dollar is a tailwind for U.S. importers and travelers and a headwind for anyone, anywhere, who owes dollars.

To go deeper, the natural next steps are the companion posts in this series. To understand where the Fed fits among the IMF, the other major central banks, and the architecture of global money, read [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system). To understand the plumbing of where reserves and deposits actually come from — the foundation under the floor system in this post — read [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier). To see the balance-sheet lever in full, read [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money). And for the historical case study in what it costs to break entrenched inflation with rates, read [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).

The Fed is not a wizard with a dial labeled "the economy." It is an institution that pins one overnight rate using a pen built from its own administered rates, communicates carefully about where that rate is going, and then lets arbitrage and expectations carry the signal out to every market on earth. The whole world holds its breath at a Fed meeting not because the Fed controls everything, but because that one number is the first domino — and the dominoes reach all the way to your front door, and a great deal further.

(All specific rate levels in this post — the roughly 4.25–4.50% target range, ~7% mortgages, ~4–5% savings yields, the ~\$1 trillion federal interest bill — are approximate and as-of the time of writing in mid-2026; the mechanisms are durable, but the numbers move every meeting. Nothing here is investment advice; it is an explanation of how the machinery works.)

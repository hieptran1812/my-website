---
title: "Deficits, Debt, and Bond Supply: Why Issuance Moves Yields"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Every deficit is financed by selling bonds, so a bigger deficit means more Treasury supply — and supply, all else equal, pushes yields and the term premium up. A from-zero guide to how the deficit became a first-order driver of the entire rates complex, and how to trade it."
tags: ["macro", "fiscal-policy", "deficits", "national-debt", "treasury-issuance", "bond-supply", "term-premium", "interest-expense", "debt-to-gdp", "rates"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Every dollar the government spends beyond what it taxes has to be borrowed, and it borrows by selling Treasury bonds. So a rising deficit is, mechanically, a rising supply of bonds — and more supply, with demand held fixed, means a lower bond price, which is the same thing as a higher yield.
>
> - A **deficit** is a *flow* (this year's shortfall); the **debt** is the *stock* (every past deficit piled up). The US debt is above \$37 trillion, roughly 123% of GDP, and it only grows because the government runs a deficit nearly every year.
> - Bond prices and yields move in opposite directions. When the Treasury floods the market with new bonds and demand does not grow to match, the price has to fall to find buyers — and a falling price *is* a rising yield. That extra yield investors demand to hold all that long-dated supply is the **term premium**.
> - The scariest part is a feedback loop: higher rates raise the interest the government pays on its debt, which widens the deficit, which forces more issuance, which pushes yields higher still. The interest bill nearly tripled from \$345 billion (2020) to about \$1 trillion (2025) and in 2024 passed national defense for the first time.
> - The debt is absorbed by four groups — the Fed, banks, foreigners, and the public. If a big buyer steps back (a "buyers' strike"), the price-sensitive public only takes the slack at a higher yield. That is what a supply tantrum looks like.
> - The one number to remember: the US is on track to refinance roughly **\$7 trillion** of maturing debt in a single year, and rolling it from 1% coupons into 5% coupons is a multi-hundred-billion-dollar hit to the budget all by itself.

In the late summer and autumn of 2023, the US bond market did something that genuinely scared people. The 10-year Treasury yield — the single most important interest rate in the world, the number that anchors mortgages, corporate borrowing costs, and the discount rate under every stock on earth — climbed from around 3.96% in late July toward 4.88% by mid-October. That is a move of nearly a full percentage point in under three months, in the asset that is supposed to be the safest and least volatile thing you can own.

The strange part was *why*. Inflation was falling that whole autumn, not rising. The Federal Reserve had stopped hiking and was clearly near the end of its tightening cycle. Growth was fine. By the old playbook, none of the usual reasons for yields to surge were present. And yet long yields tore higher anyway. The explanation that finally stuck — the one the Fed itself, the Treasury, and every rates desk converged on — was not about inflation or the Fed at all. It was about **supply**. The US government was running enormous deficits, the Treasury had just announced it would auction a wall of new bonds to fund them, and the market was demanding to be paid more to absorb all that paper. The yield went up because the *price* of all those bonds had to come down to clear, and a lower bond price is a higher yield. The market had repriced the term premium — the extra compensation for holding long bonds — and it had done so because of issuance.

![Chain diagram from budget deficit to new bond issuance to higher supply to higher yields and the term premium](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-1.png)

That autumn was the moment the deficit graduated from a slow-moving political worry into a first-order, day-to-day driver of the rates complex. For a long time, traders could treat fiscal policy as background noise — interesting for the ten-year view, irrelevant for this week's positioning. That era is over. With the debt above \$37 trillion and the annual interest bill now larger than the entire defense budget, the *amount* the Treasury borrows and the *form* it borrows in have become market-moving events on the level of a Fed meeting or an inflation print. Traders now watch issuance the way they watch the Fed. This post builds the whole machine from the ground up, assuming you have never thought about a government bond in your life. We will define the difference between a deficit and the debt, show exactly how a deficit forces a bond to be sold, build the supply-and-demand of Treasuries and the term premium, walk through the terrifying interest-expense feedback loop, figure out who actually buys all this debt and what happens if they stop — and then turn the whole thing into a concrete playbook you can trade.

## Foundations: deficit versus debt, debt-to-GDP, and how a deficit is financed

Almost every confused argument about government finance comes from blurring two words that mean completely different things: the *deficit* and the *debt*. Get the distinction crisp and the rest of the post falls into place, because the entire chain from fiscal policy to bond yields runs through it.

### A deficit is a flow; the debt is a stock

Think about your own finances for a second. There is a difference between *how much you overspent this month* and *how much you owe in total*. The first is a flow — a rate, measured over a period, dollars per month or dollars per year. The second is a stock — a level, measured at a single instant, just dollars. Overspend by \$500 this month and your credit-card balance grows by \$500; the \$500 is the flow, the balance is the stock.

The government works exactly the same way.

- The **budget deficit** is a *flow*. In a given year, the government collects revenue (mostly taxes) and spends money (defense, Social Security, Medicare, interest on past debt, everything). When spending exceeds revenue, the gap is the deficit. In recent years the US federal deficit has run on the order of \$1.8 trillion a year — meaning Washington spends roughly \$1.8 trillion more than it collects, every single year. (If revenue ever exceeded spending, the gap would be a *surplus*; the US last ran one around 2000.)
- The **national debt** is a *stock*. It is the accumulated total of every past deficit, minus every past surplus, still outstanding. It is the credit-card balance, not this month's overspend. As of fiscal year-end 2025 the total US federal debt was about **\$37.4 trillion**.

The relationship between them is mechanical and worth stating as a rule you can lean on: **this year's debt equals last year's debt plus this year's deficit.** The deficit is the flow that fills the bucket; the debt is the level in the bucket. The bucket only goes down if you run a surplus, which the US essentially never does anymore. So the debt grows year after year, and the only question is how fast.

![Total US federal debt rising from 22.7 trillion in 2019 to 37.4 trillion in 2025](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-2.png)

The chart above is just that bucket filling up. Notice the step-change in 2020: the debt jumped \$4.2 trillion in a single year as the government ran a \$3.1 trillion COVID deficit. That is the flow (a giant one-year deficit) becoming the stock (a permanently higher debt level). The pandemic deficits closed, but the debt they created never went away — it sits there forever, and now it generates interest.

### Why we measure debt against GDP

A raw debt number like "\$37.4 trillion" is almost meaningless on its own. Thirty-seven trillion dollars would be catastrophic for a small country and trivial for a galaxy-spanning empire — what matters is the debt *relative to the size of the economy that has to service it*. The standard yardstick is the **debt-to-GDP ratio**: the debt divided by Gross Domestic Product, the total annual output of the economy.

Why GDP? Because GDP is roughly the tax base — the pool of income the government can tax to pay interest and principal. A household earning \$50,000 a year can comfortably carry a \$50,000 car loan but would drown under a \$5 million one; the loan only makes sense relative to income. Same logic: \$37 trillion of debt against a \$30 trillion economy is a very different animal than \$37 trillion against a \$300 trillion one. The US total-debt-to-GDP ratio is about **123%** as of 2025, meaning the debt is roughly 1.23 times one year's entire economic output. (You will also hear "debt held by the public," which strips out debt the government owes itself in trust funds; that figure is closer to 99% of GDP. Both are large by historical standards — postwar US debt-to-GDP bottomed around 30% in the 1970s.)

The ratio can rise two ways: the numerator grows (more borrowing) or the denominator shrinks or stalls (slower economy). This is why a recession is doubly bad for the fiscal picture — the deficit balloons (tax revenue falls, safety-net spending rises) *and* GDP falls, so debt-to-GDP can lurch up fast, exactly as it did in 2020.

#### Worked example: debt-to-GDP, made concrete

Let us build the 123% number from the pieces so it is not just a statistic you have to take on faith.

- Total federal debt at fiscal year-end 2025: about **\$37.4 trillion** (`data.FED_DEBT_TOTAL`, last point).
- US nominal GDP in 2025: roughly **\$30.4 trillion** (the economy's annual output).
- Debt-to-GDP = \$37.4T ÷ \$30.4T ≈ **1.23**, i.e. **123%**.

Now feel why the ratio is the right lens. Suppose nominal GDP grows 5% next year (say 2.5% real growth plus 2.5% inflation), lifting it to about \$31.9 trillion. If the government runs another \$1.9 trillion deficit, the debt climbs to \$39.3 trillion. New ratio: \$39.3T ÷ \$31.9T ≈ **1.23** — flat. The debt grew in dollars but the economy grew just enough to keep pace. This is the central fiscal arithmetic: **debt-to-GDP is stable only when the economy grows at least as fast as the debt does.** When the deficit (plus interest) outruns nominal GDP growth, the ratio climbs, and the bond market starts to wonder how the story ends. The intuition that survives is that a country can "grow its way" out of debt only if growth plus inflation outpaces borrowing — and when interest costs are exploding, that gets harder every year.

### How a deficit is financed: the bond is born

Here is the step almost everyone skips, and it is the hinge of the entire post. When the government runs a deficit, *where does the money come from?* It does not, in the modern system, simply print cash to spend (we will deal with that myth head-on later). It **borrows** — and the way the US Treasury borrows is by selling **Treasury securities**: IOUs that promise to pay the holder back, with interest, on a fixed schedule.

The mechanism is a public **auction**. The Treasury announces it will sell, say, \$50 billion of a particular bond on a particular date. Investors — banks, pension funds, foreign central banks, money funds, hedge funds — submit bids saying how much they will buy and at what yield. The bonds go to the winning bidders, who pay cash. That cash flows into the Treasury's account and funds the government's spending. The IOUs that walk out the door are the national debt. Every dollar of deficit becomes, almost immediately, a dollar of new bonds sitting in some investor's portfolio.

So the chain is not subtle, it is mechanical:

**Deficit (we spent more than we taxed) → we must borrow the gap → we borrow by auctioning bonds → those bonds are new supply hitting the market.**

A \$1.9 trillion deficit is, by definition, \$1.9 trillion of net new bonds that the world's investors must collectively choose to hold. They do not have to *want* them at today's price — and that is the whole game.

![Federal budget deficit bars 2019 to 2025 with deficit as percent of GDP overlaid](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-5.png)

There is one wrinkle that trips up beginners and that we need to nail down: the Treasury issues bonds not only to fund the *new* deficit but also to **roll over** old debt that is maturing. A bond sold three years ago that comes due today must be paid off — and the Treasury pays it off by selling a brand-new bond to raise the cash. So the *gross* amount of bonds auctioned each year is far larger than the deficit. There is the **net** number (new debt added — roughly the deficit) and the **gross** number (net plus all the maturing debt being refinanced). Because the US carries a huge stock of short-term debt that matures constantly, gross issuance runs into the tens of trillions per year even though the net add is "only" \$1.8–2.0 trillion. We will come back to the rollover problem — it has its own scary feedback effect — but for now hold the simple version: **the deficit sets the net new supply of bonds, and supply is about to do something to price.**

### Treasury bills, notes, and bonds — the maturity ladder

One more piece of vocabulary, because the *kind* of bond the Treasury sells matters for who buys it and how much yield pressure it creates. All Treasury securities are IOUs, but they differ by **maturity** — how long until they pay you back:

- **Treasury bills** mature in **one year or less**. They pay no periodic interest; you buy them at a discount and get face value back (pay \$98, get \$100 in three months). Because they are so short and safe, bills behave almost like cash that pays a yield.
- **Treasury notes** mature in **2 to 10 years**. They pay a fixed semiannual **coupon** plus the face value at the end.
- **Treasury bonds** mature in **20 or 30 years**. Same structure as notes, just longer.

Notes and bonds — anything 2 years and out that pays a coupon — get lumped together as **"coupons."** The key word attached to coupons is **duration**: a measure of how sensitive a bond's price is to changes in interest rates. A 30-year bond has high duration (a small rate move swings its price a lot, because you are locked in for decades); a 3-month bill has almost none. This matters enormously, because the term premium — the extra yield demanded for holding long, duration-heavy bonds — is precisely what supply pressure tends to push around. When the Treasury floods the market with *coupons*, it is asking investors to absorb a lot of duration, and that is what makes long yields jump. (If you want the deep mechanics of how the Treasury chooses bills versus coupons and how that choice drains liquidity, that is the subject of [the Treasury issuance deep-dive](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain) — this post focuses on the supply-and-yield channel.)

### How an auction actually clears — and who is forced to catch the bonds

It is worth slowing down on the auction itself, because the *plumbing* of how bonds get sold is what makes a buyers' strike show up as a number you can trade. The Treasury does not sell bonds to the public the way a shop sells goods. It runs a sealed-bid auction, and standing between the Treasury and the world is a small club of about two dozen banks called **primary dealers** — firms like the big global investment banks that are *obligated*, as a condition of their privileged status, to bid at every auction. They are the market-makers of the Treasury market: they buy at the auction and then redistribute the bonds to everyone else (funds, foreigners, you).

This obligation is the safety valve and the stress gauge at once. Because primary dealers *must* bid, the auction always technically "succeeds" — the bonds always get sold, the government always gets funded. There is never a literal failed auction where no one shows up. But that guarantee hides the real signal: *at what price* did it clear, and *how much did the dealers get stuck holding?* When end-investors (funds and foreigners) show up hungry, they take most of the bonds and the dealers are left holding little. When end-demand is weak — a buyers' strike — the dealers are forced to absorb a big chunk onto their own balance sheets, and they will only do that at a cheaper price (higher yield). So the dealers act like a shock absorber that converts weak demand into a visible price concession. The **bid-to-cover ratio** (total bids ÷ amount offered) and the **dealer take-down** (the share the dealers were stuck with) are the two dials that tell you whether the auction was swallowed easily or jammed down a reluctant market's throat. We will use exactly these dials in the playbook.

The everyday-money version: say you *have* to sell your car this afternoon, no matter what, and your brother-in-law is contractually required to buy it if no one else will. The car always sells — but if no real buyer turns up, your brother-in-law takes it at a steep discount, and that discount is the market telling you demand was weak. Treasury auctions work the same way, at a scale of tens of billions of dollars a week.

## The supply and demand of Treasuries, and the term premium

We now have the deficit forcing the birth of new bonds. The next question is the one that decides yields: when all that new supply hits the market, what makes the price move, and by how much?

### A bond's price and its yield are the same fact, inverted

The most important relationship in all of fixed income — and the one beginners must internalize before anything else makes sense — is this: **a bond's price and its yield move in opposite directions.** They are two ways of describing the same thing.

Here is why, with no formula. A bond is a fixed stream of future dollars: a 10-year note paying a \$40-a-year coupon on \$1,000 of face value gives you \$40 every year and your \$1,000 back at the end, no matter what. Those dollar amounts never change. Now suppose you want to *sell* that bond, but meanwhile the government has started issuing brand-new 10-year notes paying \$50 a year. Why would anyone pay you \$1,000 for a bond that pays \$40 when they could buy a new one paying \$50 for the same \$1,000? They would not. To sell yours, you must drop the price — to maybe \$920 — until the \$40 coupon plus the discount works out to the same effective return as the new \$50 bonds. The bond's *price* fell; its *yield* (the return a buyer now earns) rose. Price down, yield up. Always.

Run it the other way: if new bonds start paying only \$30, your \$40 bond is suddenly the belle of the ball, and buyers bid its price *up* above \$1,000 until its effective yield falls to match the new lower market rate. Price up, yield down.

So whenever you hear "bond yields rose," you can mentally translate it to "bond prices fell," and vice versa. They are the same event. This is the lever the whole post hangs on, because supply works on the *price*, and price changes *are* yield changes.

### More supply, demand fixed: the price must fall

Now bring in the deficit. When the Treasury auctions a wall of new bonds, it is increasing the *quantity* of bonds the market must hold. Picture the ordinary supply-and-demand cross you have seen for any good: a downward-sloping demand curve (the higher the price, the fewer bonds people want) and a supply set by how much the Treasury issues. Push the supply curve out to the right — more bonds for sale — and, with the demand curve unchanged, the new intersection is at a **lower price**. And a lower price means a higher yield.

That is the entire supply mechanism in one sentence: **more bonds, demand held fixed, means a lower clearing price, which means a higher yield.** The Treasury is a price-taker on size — it has a deficit to fund and it *will* sell the bonds — so the adjustment falls entirely on price. Investors are not forced to hold the new bonds at yesterday's yield; they will hold them only at whatever yield makes the extra supply attractive, and that is a higher one.

### The term premium: the price of duration

The cleanest way traders express "supply is pushing yields up" is through the **term premium**. Define it carefully, because it is the variable that the deficit moves most directly.

A long bond's yield can be split into two parts. The first part is just the average of expected future short-term interest rates over the bond's life — if you think the Fed will hold rates around 3% for the next ten years, a 10-year bond *should* yield about 3% on that basis alone, because you could otherwise just roll short-term bills. The second part is the **term premium**: the *extra* yield investors demand on top of that, as compensation for the risks of locking their money up for ten years — mainly the risk that rates rise and crush the bond's price (duration risk), plus inflation and liquidity risk.

The term premium is where supply lives. When the Treasury issues a flood of long-dated coupons, it is asking the market to hold more duration than before. Duration is a risk investors must be *paid* to carry, and the more of it you force them to hold, the higher the price of carrying it — so the term premium rises. The expected-rates part of the yield can be perfectly stable (the Fed isn't doing anything) and long yields can *still* jump, purely because the term premium repriced to absorb the supply. **That is exactly what happened in the autumn of 2023:** the Fed was done hiking, expected future rates were flat-to-lower, and yet the 10-year yield surged — because the term premium went from slightly negative to clearly positive as the market braced for a tsunami of issuance. Supply did it, not the Fed. For how this slots into the broader shape of the curve, see [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

Here is a subtlety that confuses people and is worth resolving directly: for much of the 2010s, the term premium was actually *negative*. Investors were, in effect, accepting *less* yield on a 10-year bond than the expected path of short rates alone would justify — paying a premium to own duration rather than demanding one. How can that be? Because in that era duration was a *hedge*: when growth scares hit, long bonds rallied hard (yields fell, prices rose) exactly when stocks fell, so holding long Treasuries protected a portfolio. A hedge that pays off in a crisis is worth *paying up* for, so investors bid bond prices up and the term premium below zero. On top of that, the Fed was doing QE — actively buying duration out of the market — and foreign reserve managers and pension funds had structural, price-insensitive demand for long bonds. Supply was being soaked up by buyers who did not care about the yield, so the term premium stayed compressed for years.

What changed in 2022–2023 is that all three supports flipped at once. The Fed went from QE (buying duration) to QT (shedding it). Inflation came back, which meant long bonds *stopped* reliably hedging — in 2022 stocks and bonds fell *together*, so duration was no longer crisis insurance but an extra source of pain, and a hedge that fails when you need it is worth *less*, not more. And the deficit was ballooning, pumping out ever more duration for the market to hold. A negative term premium can persist only while price-insensitive buyers dominate; the moment the marginal buyer becomes the price-*sensitive* public — demanding compensation for duration that no longer hedges — the term premium has to climb back toward, and through, zero. That regime shift is the slow-moving backdrop behind every post-2022 supply tantrum, and it is why "the term premium is normalizing higher" has become a structural, multi-year theme rather than a one-off blip.

#### Worked example: a \$2 trillion deficit, financed entirely by issuance

Let us make the supply pressure tangible with a clean, round case. Suppose the government runs a **\$2,000,000,000,000** annual deficit — \$2 trillion — and finances every dollar of it by selling new bonds (which, mechanically, it must).

- The market starts the year willing to hold the existing stock of Treasuries at a 10-year yield of, say, 4.0%.
- Now \$2 trillion of *net new* bonds must find homes — on top of the trillions of gross issuance from rollovers. That is the supply curve shifting hard to the right.
- Demand does not magically grow by \$2 trillion at 4.0%. The marginal buyer — the price-sensitive ones who hold Treasuries for return rather than mandate — will only step up if paid more. Suppose it takes an extra 0.30 percentage points of yield (30 basis points) to coax out enough demand to absorb the supply without the price collapsing.
- The 10-year yield drifts from 4.0% to **4.3%**, with the move concentrated in the **term premium** component — expected Fed policy hasn't changed; only the compensation for holding all that supply has.

Now feel the second-order bite. A 0.30-point rise in the 10-year yield doesn't just affect new bonds — it marks down the *price* of every existing long bond in the market (price and yield move opposite). And it raises the government's *own* future borrowing cost: the next batch of bonds gets auctioned at 4.3% instead of 4.0%, so a \$2 trillion deficit financed at the higher rate costs the taxpayer roughly \$6 billion a year more in interest than it would have at 4.0% (\$2T × 0.30% × proportion issued long). The intuition to carry away is that the deficit doesn't just borrow money — it raises the price of money for everyone, including the government itself, which is the trapdoor into the next section.

## The interest-expense spiral: the scary feedback loop

Everything so far has been a one-way street: deficit → issuance → supply → higher yields. The reason the modern fiscal situation frightens serious people is that the street loops back on itself. Higher yields don't just sit there — they feed *back* into the deficit, which forces *more* issuance, which pushes yields up *again*. This is the **interest-expense spiral**, and once the debt is large enough, it can become self-reinforcing.

![Feedback loop showing higher rates raising interest then the deficit then issuance then yields again](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-4.png)

### How the loop closes

Walk the four nodes of the loop:

1. **Rates rise.** For whatever reason — the Fed hiking, supply pressure, an inflation scare — yields go up. New bonds get auctioned at 5% instead of 1%.
2. **The interest bill rises.** Interest is itself a line of government spending — a big one. As old low-rate debt matures and gets refinanced at the new higher rate, the government's annual interest payments climb. Every dollar of debt rolled from a 1% coupon to a 5% coupon quintuples the interest it generates.
3. **The deficit widens.** Interest is part of spending, so a bigger interest bill means a bigger deficit (revenue unchanged, spending up). Interest has gone from a rounding error to one of the largest single items in the budget.
4. **More issuance.** A bigger deficit means more new bonds must be auctioned to fund it — which adds to supply, which (all else equal) pushes yields up again, looping back to step 1.

The terrifying feature is that **the cure feeds the disease.** When inflation flares, the Fed raises rates to fight it — but higher rates raise the government's interest bill, widening the deficit and adding bond supply, which pushes long yields *up*. The standard tool for cooling the economy now has a side effect that pressures the very rates it is working through. With a small debt this is a non-issue; the interest bill is trivial. With \$37 trillion of debt, it is the central fiscal risk of the decade.

### The numbers behind the spiral

This is not theoretical. It is already happening, and the chart of net interest outlays is one of the most important pictures in macro right now.

![Federal net interest outlays rising from 345 billion in 2020 to about 1 trillion in 2025 passing defense in 2024](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-3.png)

#### Worked example: the interest bill nearly triples

Trace the actual path using the real figures (`data.NET_INTEREST`):

- FY2020: net interest outlays were **\$345 billion**. Rates were near zero — the average interest rate the government paid on its debt was tiny, even though the debt stock was already huge.
- FY2022: **\$475 billion**, as the Fed began hiking and new and maturing debt repriced higher.
- FY2023: **\$658 billion** — a 39% jump in one year, as more of the debt rolled into the higher-rate world.
- FY2024: **\$880 billion**. This is the milestone year: federal interest spending passed total **national defense** spending (roughly \$850 billion) for the first time in modern history. The country now spends more servicing past borrowing than it spends on its entire military.
- FY2025: about **\$1,000 billion — \$1 trillion** — by the latest estimates.

So in five years the interest bill went from \$345 billion to roughly \$1 trillion: nearly a **tripling**. Notice *why* it tripled. The debt stock grew, yes — from about \$27 trillion to \$37 trillion, roughly 39%. But the interest bill grew far faster, almost 190%. The gap is the *rate*: the average interest rate on the debt roughly doubled as near-zero-coupon bonds matured and were replaced with 4–5% bonds. The lesson that survives is that **at high debt levels, the interest bill is driven less by how much you owe and more by the rate at which you refinance it** — which is exactly the variable the supply-and-demand of bonds controls.

### Why size makes it self-reinforcing

The spiral only matters because the debt is enormous. To see the threshold, think in terms of the **interest-to-revenue ratio** — interest outlays divided by tax revenue, i.e. what fraction of every tax dollar goes straight to bondholders before funding a single other thing. In 2024, with roughly \$880 billion of interest against roughly \$4.9 trillion of federal revenue, that ratio was about **18%** — nearly one in five tax dollars. When that ratio climbs past a danger zone (analysts often flag mid-teens and up), the math gets reflexive: rising rates raise interest faster than revenue can grow, the deficit structurally widens, and the bond market starts demanding an *even higher* term premium to compensate for the rising supply and the perceived loss of fiscal control — which raises rates further. That reflexive loop, not the raw debt number, is what makes the deficit a live market driver rather than a slow-moving statistic. (The interest rate that anchors this whole spiral is itself the master variable; see [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

#### Worked example: the rollover problem

The spiral's nastiest mechanism is **rollover** — refinancing maturing debt at today's rates instead of yesterday's. Make it concrete.

Suppose the US must refinance roughly **\$7,000,000,000,000** — \$7 trillion — of maturing debt in a single year (a realistic figure, because so much of the debt is short-term bills and front-end notes that mature constantly and must be reissued). This is debt that was originally issued in the low-rate years, much of it carrying coupons near **1%**. It now has to be rolled into the current environment at roughly **5%**.

- Old annual interest on that \$7 trillion at 1%: \$7,000B × 1% = **\$70 billion** a year.
- New annual interest on the same \$7 trillion at 5%: \$7,000B × 5% = **\$350 billion** a year.
- The increase, from rolling this one slug of debt: **\$280 billion** a year — *more*, every year, with zero new borrowing. The government didn't spend an extra dime on programs; it simply refinanced existing debt at the going rate.

This is why the interest bill exploded even though the debt grew "only" modestly. A huge stock of cheap debt is maturing into an expensive world, and each rollover locks in the higher cost for years. The intuition to keep is that **a country with lots of short-term debt is effectively floating-rate** — it reprices to current yields fast, so when rates rise, the budget feels it almost immediately, and the spiral spins faster. A country that termed-out its debt at the lows (issued lots of 30-years at 2%) would feel this far more slowly. The US, having leaned heavily on short bills, feels it fast.

## Who actually buys the debt — and what happens if they step back

So far we have treated "demand" as a single curve. But Treasuries are not bought by one undifferentiated blob. They are absorbed by four distinct groups with completely different motivations, price sensitivities, and breaking points. Understanding who holds the debt is the key to understanding when a deficit becomes a *yield* problem rather than just a number — because it is the marginal buyer, the one who has to be coaxed in at the margin, who sets the price.

![Diagram of the four Treasury buyer groups and the effect of a buyers strike on auctions and yields](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-6.png)

### The four buyers

- **The Fed.** The Federal Reserve can buy Treasuries by creating reserves — this is **quantitative easing (QE)**. When the Fed is doing QE, it is a giant, *price-insensitive* buyer: it does not care about yield, it is buying to push yields down on purpose. That can soak up enormous amounts of issuance and keep the term premium suppressed. But the Fed can also go the other way — **quantitative tightening (QT)** — letting bonds mature without replacing them, which *removes* a buyer and forces the private market to absorb more. From 2020 to 2022 the Fed was a massive buyer; from 2022 onward it was shrinking its holdings, meaning the private market had to swallow both the deficit *and* the bonds the Fed was no longer holding. (For the full mechanics, see [QE explained](/blog/trading/finance/quantitative-easing-explained-printing-money).)
- **Banks and money funds.** Domestic banks, money market funds, pension funds, and insurers hold a large slice. They are partly mandate-driven (they need safe assets, regulatory liquidity, duration to match liabilities) and partly price-sensitive. Their appetite is real but capped — by reserve levels, by regulatory rules, by how much duration risk they can stomach.
- **Foreigners.** Foreign central banks, sovereign wealth funds, and overseas investors hold roughly **30%** of US debt — Japan and China the largest among them. They buy partly to manage their own currencies and reserves and partly for yield. Crucially, their demand can *shift* for reasons that have nothing to do with US fundamentals — a country defending its currency may sell Treasuries, a geopolitical rift may cool reserve appetite, a yen surge may change the hedged return. When a big foreign holder steps back, a major source of demand evaporates.
- **The public.** Everyone else — domestic bond funds, hedge funds, households, the price-sensitive money that holds Treasuries purely for return. This group is the **shock absorber of last resort**: when the other three pull back, *someone* still has to hold every bond the Treasury sells, and it falls to the price-sensitive public. They will do it — but only at a price that makes it worth their while, which means a lower price and a higher yield.

### The buyers' strike

Now the punchline. Imagine the deficit jumps (more supply) at the same moment the Fed is doing QT (one buyer removed) and foreigners pull back (another buyer removed). The same wall of bonds now has to be absorbed by a much smaller pool of willing buyers — essentially the price-sensitive public. They will take the bonds, because the market always clears, but only at a meaningfully lower price. That is a **buyers' strike**, and its fingerprint is visible right at the auctions.

When you watch a Treasury auction, two numbers tell you whether demand is healthy or buckling:

- **Bid-to-cover ratio**: total bids divided by the amount offered. A high ratio (say 2.5×) means buyers swarmed the auction; a low one (near 2.0× or below for a tenor that usually runs higher) means weak demand.
- **The tail**: the gap between the yield the auction *expected* (the pre-auction market level) and the yield it actually *cleared* at. A "tail" means the auction had to offer a higher yield than expected to sell all the bonds — a direct, real-time readout that supply overwhelmed demand. A string of tailing auctions is the market telling you the buyer base is straining.

When auctions tail repeatedly and bid-to-covers slip, yields gap higher and the term premium reprices — a **supply tantrum**, exactly the autumn-2023 pattern. This is why traders now watch auction results the way they watch CPI prints: a weak 10-year or 30-year auction can move the whole bond market in seconds, because it is hard evidence that the buyer base is choking on the supply.

#### Worked example: when a big buyer leaves

Quantify a buyers' strike so it is not just a story. Suppose the Treasury needs to sell **\$300 billion** of 10-year notes over a quarter. Historically the demand split is roughly: Fed 0% (it is in QT), foreigners 30% (\$90B), banks/funds 30% (\$90B), the public 40% (\$120B). The market clears comfortably at a 4.0% yield.

Now a large foreign holder — defending its currency — not only stops buying but trims its position, so foreign demand for this quarter's supply effectively drops to near zero. That \$90 billion of demand has to come from somewhere. The price-sensitive public is the only group that flexes, so it must absorb \$120B + \$90B = **\$210 billion**, 75% more than it planned for.

The public will hold \$210 billion of 10-years — but not at 4.0%. To pull in that much extra price-sensitive money, the yield has to rise. Say it takes 25 basis points: the auctions tail, the cleared yield moves to **4.25%**, and the term premium has just repriced by a quarter point on nothing but a shift in the buyer mix. No change in inflation, no change in the Fed — just one big buyer stepping back and the price-sensitive marginal buyer demanding to be paid more. The durable intuition: **the level of yields is set by whoever has to be coaxed in at the margin, so the deficit becomes a yield problem precisely when the price-insensitive buyers (the Fed, mandate-driven foreigners) are absent and the price-sensitive public is holding the bag.**

## Common misconceptions

Three myths cloud almost every public argument about deficits and bonds. Each is half-true, which is what makes it dangerous. Correct each with a number.

### Myth 1: "The US prints its own currency, so it can never run out of dollars — the debt doesn't matter."

The first clause is true; the conclusion does not follow. Yes, the US issues debt in dollars it controls, so it can never be *forced* to default the way a country borrowing in someone else's currency can. But "can't be forced to default" is not the same as "the debt doesn't matter." The constraint is not solvency; it is **the price the market demands and the inflation that financing the debt can create.**

Two real mechanisms make the debt matter regardless of the printing press. First, **the bond market sets the yield, not the Treasury.** The Treasury can always sell the bonds, but it cannot dictate the yield — if buyers demand 5% to absorb the supply, the government pays 5%, and that flows straight into the interest bill. The market's willingness to hold the debt at a given yield is a real, binding constraint, and it is exactly what repriced in 2023 when the 10-year ran from 3.96% to 4.88% in a few months. Second, if the government ever tried to escape the bond market by having the central bank *print* money to buy the debt instead, that is monetization, and it tends to produce inflation — which is just default by another name, repaying lenders in debased dollars. The number that grounds this: the interest bill is already roughly **\$1 trillion a year** and rising. "Can't run out of dollars" does nothing to stop that bill from crowding out every other priority in the budget. The debt matters because *servicing* it matters, even when *defaulting* on it is off the table.

### Myth 2: "Issuance always spikes yields, so every 'wall of supply' headline is a sell signal."

This is the trap that caught people in 2023 and *also* the trap that caught people who shorted bonds expecting every refunding to be a disaster. Supply pushes yields up **all else equal** — but all else is frequently not equal. Demand can grow to meet supply. In a recession, a flight to safety can send so much money rushing into Treasuries that yields *fall* even as the deficit explodes — exactly what happened in 2020, when the deficit hit \$3.1 trillion (15% of GDP, the largest since WWII) and yet the 10-year yield *dropped* to around 0.6%, because terrified investors and an aggressively QE-ing Fed wanted every safe bond they could get. Supply was enormous; demand was even more enormous. The correction: issuance is one blade of the scissors. Whether it lifts yields depends on what the *other* blade — demand — is doing. Track both. A supply scare into a strong-demand backdrop (QE on, recession bid, foreigners buying) is often a fade; a supply scare into a weak-demand backdrop (QT on, foreigners selling, term premium rising) is the real thing.

### Myth 3: "The Fed just monetizes all of it — the deficit is always financed by money-printing."

Sometimes the Fed buys a lot (QE); often it buys nothing or is actively *selling* (QT). From 2022 onward the Fed was in QT, shrinking its balance sheet from a peak of about **\$8.96 trillion** toward roughly \$6.6 trillion — meaning it was *removing* itself as a buyer and forcing the private market to absorb both the deficit and the bonds rolling off the Fed's book. Far from monetizing the deficit, the Fed was adding to the net supply the market had to digest. The correction: whether the Fed is monetizing is a *variable you check*, not a constant you assume. The single most important demand-side question for the bond market is "is the Fed in QE or QT right now?" — because that determines whether the largest, most price-insensitive buyer is present or absent. Assuming the Fed always backstops issuance is how you get blindsided by a supply tantrum. (For the full QE-versus-QT mechanics, see [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets).)

## How it shows up in real markets

Abstractions are cheap. Here are the concrete episodes where the deficit-issuance-yield chain visibly drove the tape, with real dates and numbers.

### The 2023 term-premium repricing

The cleanest case study is the one we opened with. Through the autumn of 2023, the 10-year Treasury yield climbed from **3.96%** (late July) to **4.88%** (mid-October) — nearly a full percentage point — while inflation was *falling* and the Fed was *done hiking*. Decompositions by the New York Fed and others attributed the bulk of the move to a rising **term premium**, not to higher expected policy rates. What drove the term premium? The Treasury's August refunding announcement had signaled a sharp step-up in coupon issuance to fund the widening deficit, just as the Fed's QT was removing a buyer and the debt-ceiling resolution earlier that year had unleashed a flood of catch-up issuance. The market looked at the supply pipeline, looked at the shrinking pool of price-insensitive buyers, and demanded a higher term premium to hold all that duration. Yields fell back sharply into year-end (the 10-year ended 2023 back near **3.88%**) when the next refunding signaled smaller coupon increases than feared and the demand picture stabilized — a textbook demonstration that the *supply path*, communicated at refunding, was moving the long end almost mechanically. This was the episode that taught a generation of traders to read the Quarterly Refunding Announcement.

### The interest bill passing defense

In FY2024, federal net interest outlays hit about **\$880 billion** and crossed above total national defense spending (roughly \$850 billion) for the first time in modern US history. This was not a market event with a single timestamp, but it was a profound regime marker — the moment the abstract "interest-expense spiral" became a concrete, headline fact. It reframed every subsequent fiscal debate: when interest is your single largest discretionary-sized outlay and it is *growing automatically* as debt rolls into higher rates, the deficit stops being a knob policymakers can easily turn and starts being a structural force. Bond investors took note, and the term premium has carried a persistent fiscal risk component ever since. The line to watch going forward is the interest-to-revenue ratio: at roughly **18%** in 2024 and climbing, it is the gauge that tells you whether the spiral is accelerating.

### The 2020 counter-example

It is just as important to know when the chain *doesn't* fire. In FY2020 the deficit exploded to **\$3.13 trillion** — 15% of GDP, the largest since World War II — and the debt jumped \$4.2 trillion in a year. By the naive "issuance spikes yields" rule, the 10-year should have rocketed. Instead it *collapsed* to around **0.62%** by mid-2020. Why? Because the demand blade overwhelmed the supply blade: a global flight to safety drove panicked money into Treasuries, and the Fed launched unlimited QE, buying trillions and acting as a price-insensitive backstop. Supply was historic; demand was even more historic. This is the single best reminder that **the deficit is necessary but not sufficient for higher yields** — you must always check what demand, and especially the Fed, is doing at the same time. The 2020 and 2023 episodes are the two poles every issuance trade lives between.

### The slow-motion foreign step-back

A subtler real-market story is the gradual erosion of foreign demand. For two decades, foreign official buyers — above all China and Japan — were a structural, price-insensitive bid for US debt: they accumulated dollars through trade surpluses and recycled them into Treasuries, soaking up supply without much regard for yield. That tailwind has faded. Foreign holdings have grown far more slowly than the debt itself, so foreigners' *share* of the total has drifted down from well over a third toward roughly 30% and below. When a country like China lets its Treasury holdings run off, or Japan sells Treasuries to fund currency intervention (as it did repeatedly in 2022 and 2024, with the yen sliding past 150 and then 160 to the dollar), it removes a slice of that old price-insensitive demand. The bonds those buyers used to absorb don't vanish — they get pushed onto the price-sensitive domestic public, who demand a higher yield to hold them.

The point is not a dramatic "China dumps Treasuries" crash — that is largely a myth, since a country selling its dollar reserves would crush its own holdings and currency. The realistic mechanism is quieter and more important: foreign demand simply *grows slower than supply*, so a larger and larger share of each year's issuance must be placed with price-sensitive buyers. That shift, compounded over years, raises the equilibrium term premium even with no single dramatic event — a structural reason the deficit has more yield-bite today than it did in 2010, when the foreign official bid was much stronger. For the broader story of why the world holds so many dollars in the first place, see [the dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).

## How to trade it / the playbook

You now have the full machine. Here is how to turn it into positioning. The deficit is no longer background — it is a tradeable, scheduled, signal-rich driver of the rates complex. The job is to track four gauges and act when they line up.

![Playbook pipeline tracking the deficit issuance interest-to-revenue ratio and term premium to a position](/imgs/blogs/deficits-debt-bond-supply-why-issuance-moves-yields-7.png)

### The four gauges

**1. The deficit trajectory.** Is the deficit widening or narrowing? Watch CBO projections and updates, and the **Monthly Treasury Statement**, which reports actual receipts and outlays — compare it to forecasts. A deficit running *worse* than expected (revenue softening, spending overshooting) means more net supply is coming than the market has priced. A deficit coming in *better* than feared is a demand-friendly surprise. The deficit is the slow-moving setup that determines the supply pipeline months ahead.

**2. Issuance: the Quarterly Refunding Announcement (QRA).** Four times a year (early February, May, August, November) the Treasury publishes how much it will borrow and in what mix of bills versus coupons. This is the most market-moving scheduled fiscal event there is. The key reads: Is the *total* borrowing need rising? Is the *coupon* share rising (more duration for the market to absorb = more term-premium pressure) versus the bill share? A surprise increase in long-end auction sizes is a direct supply shock to the long end. Trade the QRA the way you trade a Fed meeting — position into it and react to the surprise. (The bills-versus-coupons liquidity angle is covered in depth in [Treasury issuance](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain).)

**3. The interest-to-revenue ratio.** This is your spiral gauge. Interest outlays divided by federal revenue: at roughly 18% in 2024 and rising, it tells you whether the feedback loop is accelerating. When it is climbing and rates are high, every fiscal stress is amplified, and the market will demand a steadily fatter term premium. A ratio pushing into the high teens and beyond is a structural tailwind for higher long yields and a steeper curve.

**4. The term premium and auction tape.** This is the real-time confirmation. Watch a term-premium estimate (the New York Fed's ACM model is the standard reference) — a rising term premium *is* the market pricing in the supply. And watch the **auction results**: bid-to-cover ratios and tails. A string of weak, tailing 10-year and 30-year auctions is hard, immediate evidence the buyer base is straining — the buyers' strike showing up live.

### The position

When the four gauges align — deficit widening, QRA stepping up coupon issuance, interest-to-revenue climbing, term premium rising and auctions tailing — the trade is to **position for higher long-end yields and a steeper curve**: short long-dated Treasuries (sell the 10-year or 30-year), or put on a **steepener** (long the front end, short the long end), since supply pressure and term premium hit the long end hardest while the front end is anchored by the Fed. The deficit's signature is a *bear steepener* — long yields rising faster than short yields as the term premium fattens.

### The invalidation

This is the most important part, because the 2020 lesson is that supply trades blow up when demand surges. **Invalidate the short-bonds / steepener view when the demand blade flips.** Specifically:

- **A flight-to-safety bid.** If a recession or risk-off shock hits, money stampedes into Treasuries regardless of supply, and yields fall hard. A supply trade has no business surviving a genuine growth scare — cut it.
- **The Fed pivots to QE.** The instant the Fed signals it will resume buying (or even stops QT), the largest price-insensitive buyer is back, and the term premium can compress fast. QE on is the supply trade's kryptonite.
- **Foreign demand returns.** A stabilizing dollar or a yield level attractive enough to pull foreign buyers back in can quietly absorb the supply and stall the move.

The discipline: a supply trade is a bet on the *balance* of supply and demand, not on supply alone. You are short bonds because issuance is overwhelming the *current* buyer base — the moment that buyer base is reinforced (Fed, foreigners, or a recession bid), the thesis is dead and you exit. Issuance is the setup; demand is the trigger and the stop. Track the deficit the way you track the Fed, read the refunding like an FOMC statement, watch the auctions like inflation prints — and never, ever forget that the other blade of the scissors can move too.

## Further reading & cross-links

- [Treasury issuance: bills, coupons, and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain) — the mechanics of *how* the Treasury borrows, and why the bills-versus-coupons mix decides whether issuance drains real market liquidity.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the rate that anchors the entire interest-expense spiral, built from first principles.
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how supply pressure on the term premium shows up in the *shape* of the curve, and what a bear steepener signals.
- [Fiscal policy for traders: spending, deficits, demand](/blog/trading/macro-trading/fiscal-policy-for-traders-spending-deficits-demand) — the upstream story of *why* the deficit exists, how fiscal policy stimulates demand, and how it interacts with monetary policy.
- [Quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) and [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) — the Fed's role as the swing buyer that decides whether the deficit becomes a yield problem.

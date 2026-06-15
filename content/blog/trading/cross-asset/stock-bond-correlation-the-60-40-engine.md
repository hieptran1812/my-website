---
title: "The Stock-Bond Correlation: The Engine Inside the 60/40"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-deep guide to the single most important cross-asset relationship: whether stocks and bonds move together or apart, why the sign flips between growth and inflation regimes, why 2022 broke the 60/40 portfolio, and how to know whether bonds will protect you."
tags: ["asset-allocation", "cross-asset", "stock-bond-correlation", "sixty-forty", "diversification", "inflation", "recession-hedge", "bonds", "portfolio-construction", "regime", "treasuries", "risk-management"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The whole case for the classic 60/40 portfolio rests on stocks and bonds moving *opposite* to each other. That negative correlation is not a law of nature — it flips sign depending on whether the market's dominant fear is slow growth or high inflation. Knowing which regime you are in tells you whether your bonds will protect you.
>
> - When a **growth shock** dominates (a recession scare), the Fed cuts rates, bond prices rise, and bonds rally *while* stocks fall. Correlation is **negative**, and 60/40 works — as in 2008, when long Treasuries returned **+25.9%** against the S&P's **−37%**.
> - When an **inflation shock** dominates, the Fed hikes rates, bond prices fall, and bonds drop *alongside* stocks. Correlation goes **positive**, and diversification fails exactly when you need it. In 2022, stocks fell **−18.1%** and bonds fell **−13.0%**, dragging the 60/40 down **−16.0%** — its worst year since 1937.
> - The sign has flipped before: **positive** through the inflationary pre-2000 era, **negative** from 2000 to 2021, and **positive again** since 2022. It is a regime, not a constant.
> - The one number to remember: **the sign of the stock-bond correlation**. Negative means bonds are your hedge; positive means you need a different hedge — commodities, gold, cash, or inflation-linked bonds.

In October 2022, a 65-year-old retiree who had done everything the textbooks told them to do opened their brokerage statement and felt sick. They held the most boringly responsible portfolio in finance: 60% in a broad stock fund, 40% in a broad bond fund — the famous **60/40**. For forty years, that mix had been sold as the sensible middle path: enough stocks to grow your money, enough bonds to cushion the fall when stocks crashed. The bonds were supposed to be the airbag. And in 2022 the airbag didn't just fail to deploy — it became a second collision. Stocks fell 18%. Bonds, the safe part, fell 13%. The blended portfolio lost 16% of its value in a single year, the worst result since 1937.

What broke? Nothing about the *assets* broke. Stocks were still stocks; bonds were still bonds, paying their coupons on schedule. What broke was the **relationship** between them — the quiet assumption, baked into every retirement calculator and target-date fund on Earth, that when stocks fall, bonds rise. That assumption is true in some decades and false in others, and the difference between them is not luck. It is mechanical, it is forecastable, and it is the single most important thing a multi-asset investor can understand.

The diagram below is the mental model we will keep returning to. It is a picture of one number — the rolling correlation between stock returns and bond returns — across half a century. Notice that it is not flat. It spends long stretches *below* zero (bonds hedging stocks, the 60/40 dream) and long stretches *above* zero (bonds and stocks falling together, the 60/40 nightmare). Everything in this post is an answer to one question: what makes that line cross from one side to the other, and how do you know which side you are standing on today?

![Stock-bond correlation line through three regimes with positive and negative eras shaded](/imgs/blogs/stock-bond-correlation-the-60-40-engine-1.png)

By the end, you will be able to look at the inflation data and the Fed's behavior and say, with real confidence, "the correlation is negative right now, so my bonds are doing their job" — or "the correlation is positive right now, so I need a different airbag." That single read is worth more than any stock pick.

## Foundations: why a bond can hedge a stock at all

Before we can talk about *when* bonds hedge stocks, we have to be crystal clear about *why* they ever could. The two assets look like they have nothing in common. A stock is a slice of ownership in a company; a bond is a loan to a government or a firm. Why would owning a loan ever protect you from a falling stock? The answer is the foundation everything else rests on, so let us build it from zero.

### What a stock is, in one sentence

A **stock** (also called a *share* or *equity*) is a fractional ownership stake in a company. When you own a share, you own a tiny piece of that company's future profits. The price of the stock is, at root, the market's estimate of all the cash the company will hand its owners over the years ahead — its future *cash flows* — squeezed down into a single number for today. (We go deep on this in [equities, owning a slice of growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth).) Two things move a stock's price: how big those future cash flows are expected to be, and how much the market *discounts* them — that is, how much it shrinks a far-off dollar to express it in today's money. Hold onto both levers. They are the whole story.

### What a bond is, in one sentence

A **bond** is a loan with a fixed schedule of payments. You hand over money today; in return you get a stream of fixed interest payments (called *coupons*) and your original sum back at the end. Because the payments are fixed in dollar terms, the only thing that moves a bond's price is the *interest rate* the market demands. When prevailing interest rates rise, your old bond paying a fixed coupon looks stingy next to new bonds, so its price falls; when rates fall, your old bond looks generous, so its price rises. The shorthand every bond investor carries: **bond prices move opposite to interest rates.** (The full mechanics — coupons, yield, duration — live in [government bonds, the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).)

So a bond's price is governed by one variable: the level of interest rates. Remember that, because it is the hinge of this entire post.

### The hidden link: stocks and bonds share a discount rate

Now for the connection that most people never see, and that explains everything that follows. Stocks and bonds look like unrelated objects, but they are priced by the *same machinery*. Both are claims on future money, and the value of any future money depends on the **discount rate** — the interest rate the market uses to convert tomorrow's dollars into today's.

For a bond, this is obvious: the discount rate *is* the interest rate, and the bond's price is the present value of its fixed coupons and principal discounted at that rate. For a stock, it is one layer less obvious but exactly as true: a stock's price is the present value of all its expected future earnings, discounted at a rate that is built up *from* the prevailing interest rate plus an extra cushion for risk. Raise the interest rate, and you raise the discount rate for *both* — you shrink the present value of the bond's coupons and you shrink the present value of the stock's earnings, both at once.

This shared dependence is the secret engine of the whole correlation. When interest rates move, they tug on stocks and bonds simultaneously, through the same discount-rate channel. Whether that tug pulls the two assets *together* or *apart* depends entirely on *why* the rate is moving — and that "why" is what the rest of this post decodes. If a rate change comes packaged with a matching change in earnings (a growth story), stocks and bonds can move apart. If a rate change hits while earnings are roughly steady (an inflation story), stocks and bonds move together. Same channel, two outcomes.

#### Worked example: one interest-rate move, two assets

Take a \$1,000 bond paying a \$30 annual coupon, priced at par (a 3% yield), and a stock worth \$100 priced as \$5 of expected earnings divided by a 5% discount rate. Now suppose market interest rates jump by 2 percentage points, and nothing else changes.

- The **bond's** discount rate rises from 3% toward 5%, and its price falls toward roughly **\$900** — a loss of about \$100, or 10%, on a medium-maturity bond.
- The **stock's** discount rate rises from 5% to 7%, so its price falls from \$5 ÷ 0.05 = \$100 to \$5 ÷ 0.07 ≈ **\$71** — a loss of about \$29, or 29%.

Both fell, from one rate move, because both are discounted by the same rising rate. That is positive co-movement created purely by the rate channel — the seed of a positive correlation whenever rates are the thing on the move.

The intuition: a rising interest rate is bad news for *any* claim on future money, so when rates are the dominant force, stocks and bonds fall together.

### The recession logic that makes bonds a hedge

Now put the two together in the situation the 60/40 was built for: a **recession**. A recession is a broad economic slump — companies sell less, earnings fall, unemployment rises. Walk through what each asset does, slowly.

For **stocks**, a recession is straightforwardly bad. Companies earn less, so the future cash flows that a stock represents shrink, so stock prices fall. Nothing subtle there.

For **bonds**, a recession is good — and here is the beautiful part. When the economy weakens, the central bank (in the United States, the Federal Reserve, or "the Fed") *cuts interest rates* to try to revive borrowing and spending. We cover exactly how it does this in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates). But recall the bond hinge: **bond prices move opposite to interest rates.** If the Fed cuts rates because the economy is sick, then rates fall, and falling rates push bond prices *up*. So in the very same downturn that crushes stocks, bonds rally.

That is the hedge. It is not magic and it is not a historical accident — it is a chain of cause and effect: *recession → weaker earnings → stocks fall*, and at the same time *recession → Fed cuts rates → bond prices rise*. Two assets, one event, opposite outcomes. When that chain holds, your bonds gain exactly when your stocks are bleeding, and the gains on one side soften the losses on the other. That softening is the entire reason the 60/40 portfolio exists.

#### Worked example: the hedge doing its job

Suppose you hold a \$100,000 portfolio split 60/40 — \$60,000 in stocks, \$40,000 in bonds — and a recession hits, behaving like the 2008 crisis.

- Your stocks fall 37%. On \$60,000, that is a loss of 0.37 × \$60,000 = **−\$22,200**.
- Your bonds, with the Fed slashing rates, rise about 5%. On \$40,000, that is a gain of 0.05 × \$40,000 = **+\$2,000**.
- Net change: −\$22,200 + \$2,000 = **−\$20,200**, a portfolio loss of about **−20.2%**.

A 20% loss still hurts. But compare it to holding *only* stocks, which would have fallen the full 37%. The bond sleeve turned a 37% gut-punch into a 20% bruise. That is the hedge working: the bonds did not merely fail to lose — they actively *gained* while stocks fell, pulling the whole portfolio's loss in.

The intuition: a bond hedges a stock because a recession that shrinks earnings (bad for stocks) also forces rate cuts (good for bonds), so the two assets move in opposite directions.

### What the 60/40 portfolio is, and why it ruled for forty years

The **60/40 portfolio** is the simplest serious portfolio in existence: put 60% of your money in a broad basket of stocks and 40% in a broad basket of high-quality bonds, and rebalance back to those weights periodically. The 60 gives you growth; the 40 gives you ballast. That is it.

Its appeal was never that it maximized return — an all-stock portfolio beats it over long horizons. Its appeal was the *ride*. Because bonds historically rose when stocks fell, the 60/40's ups and downs were far gentler than stocks alone, which let ordinary people stay invested instead of panic-selling at the bottom. For the four decades from roughly 1982 to 2021, this worked almost suspiciously well. Stocks compounded at high single-to-double digits; bonds delivered steady income *and* a reliable cushion in every equity crash — 1987, 2000–2002, 2008, 2020. An investor could hold one cheap, boring blend and sleep at night.

That forty-year golden run is exactly what makes 2022 so important — and so misunderstood. The 60/40 didn't work because of some permanent truth about stocks and bonds. It worked because, for those particular forty years, the stock-bond correlation happened to be *negative*. When that condition changed, the magic stopped. So our real subject is not the 60/40 at all. It is the sign of the correlation that powers it.

## What "correlation" actually means here

Let us nail down the one technical term this whole post turns on, because the intuition is easy and the precision matters.

**Correlation** is a single number, between −1 and +1, that measures how two things move *together*. A correlation of **+1** means they move in perfect lockstep — when one goes up, the other always goes up by a proportional amount. A correlation of **−1** means they move in perfect opposition — when one goes up, the other always goes down. A correlation of **0** means there is no consistent relationship — knowing one tells you nothing about the other. (We build correlation from the ground up, including the diversification math, in [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch).)

For our purposes, only the **sign** really matters, and it has a plain-English translation:

- **Negative** stock-bond correlation → bonds tend to rise when stocks fall → **bonds are a hedge** → 60/40 diversifies.
- **Positive** stock-bond correlation → bonds tend to fall when stocks fall → **bonds are not a hedge** → 60/40 offers far less protection than you think.

Two clarifications that trip people up. First, correlation is about *direction*, not size — a negative correlation tells you bonds will lean the opposite way to stocks, but not by how much. Second, the stock-bond correlation is measured over a *window* of time, usually a rolling two-year (24-month) window of monthly returns. That is why we can say it "changes" — at any moment it summarizes the recent past, and as the regime shifts, the number drifts and eventually crosses zero. The cover figure above is exactly this rolling number, plotted across fifty years. Now we can ask the real question: what makes the sign flip?

## The driver of the sign: it is about what shocks the market

Here is the single most important idea in the post, and it is worth reading twice. **The sign of the stock-bond correlation is decided by the kind of shock that is moving markets.** Not by the assets themselves, not by some long-term average — by whichever fear is currently in the driver's seat. There are two fears, and they push the correlation in opposite directions.

The 2x2 below lays out the whole mechanism. Read it as two rows — a growth shock on top, an inflation shock on the bottom — and follow what each does to rates, to stocks, and to bonds. The last column is the verdict: whether your 60/40 is protected.

![Two by two grid of growth shock and inflation shock showing effect on rates stocks bonds and correlation](/imgs/blogs/stock-bond-correlation-the-60-40-engine-2.png)

### Growth shocks push the correlation negative

A **growth shock** is news that the economy is weakening — a recession scare, a credit crunch, a sudden collapse in demand. Trace its effects.

Stocks fall, because weaker growth means weaker future earnings, and a stock is a claim on future earnings. Meanwhile, the central bank responds to the weakening economy by *cutting* interest rates to stimulate it. Falling rates push bond prices *up*. So in a growth shock: stocks down, bonds up. They move in opposite directions, which is the definition of negative correlation. Bonds hedge. This is the world the 60/40 was designed for, and it is the world of the entire 2000–2021 era.

### Inflation shocks push the correlation positive

An **inflation shock** is news that prices are rising too fast — energy spikes, supply chains snarl, wages chase prices upward. Now trace *its* effects, which are subtler and far more dangerous for a 60/40 holder.

Stocks fall again, but for a different reason. To fight inflation, the central bank *raises* interest rates — sharply. Higher rates do two things to stocks. They raise the rate at which the market discounts future earnings, which mechanically shrinks the present value of those earnings (this is the discount-rate lever from the Foundations section). And they slow the economy, which dents the earnings themselves. So stocks fall.

But look at what is happening to bonds. The central bank is *raising* rates, and the bond hinge says **bond prices move opposite to rates.** Rising rates crush bond prices. So in an inflation shock: stocks down, *and bonds down too*. They move in the *same* direction, which is the definition of positive correlation. Bonds do not hedge — they pile on. This is the world of 2022, and the world of the entire inflationary pre-2000 era.

### The deep framing: discount rate versus cash flow

If you want the one-sentence version that a practitioner would nod at, here it is. A stock's price is its future cash flows *divided by* a discount rate. A growth shock attacks the **cash flow** (the numerator) without raising the discount rate — in fact the discount rate falls as the Fed cuts — so bonds, which live and die by that rate, rally. An inflation shock attacks the stock through the **discount rate** (the denominator) by forcing the Fed to hike — and that very same rising rate is what crushes bonds. The flow chart below traces both paths from a single starting point: the same stock-pricing equation, split into the growth path and the inflation path, ending in opposite correlation signs.

![Flow chart of growth shock and inflation shock paths leading to opposite bond outcomes and correlation signs](/imgs/blogs/stock-bond-correlation-the-60-40-engine-5.png)

So the rule, stripped to its bones: **whatever moves the discount rate decides the sign.** When stocks are falling because earnings are falling and rates are *also* falling, bonds win and the correlation is negative. When stocks are falling because rates are *rising*, bonds lose and the correlation is positive. The Fed's direction is the tell. (For the deepest version of why the *real* interest rate sits underneath every asset price, see [real versus nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)

#### Worked example: the same shock, two discount-rate stories

Take a simple stock worth \$100 today, priced as next year's expected \$5 of cash flow divided by a discount rate of 5% (\$5 ÷ 0.05 = \$100). Watch what two different shocks do.

- **Growth shock.** Expected cash flow gets cut from \$5 to \$4 (a recession bites earnings), and the Fed cuts the discount rate from 5% to 4% to fight the slump. New price: \$4 ÷ 0.04 = **\$100** — flat in this toy case, and in reality stocks fall because the cash-flow cut outweighs the rate relief. Crucially, the *rate fell*, so bonds rallied. Stock pain, bond gain.
- **Inflation shock.** Expected cash flow holds near \$5, but the Fed hikes the discount rate from 5% to 7% to fight inflation. New price: \$5 ÷ 0.07 = **\$71** — a 29% fall driven entirely by the higher discount rate. And that same jump in rates, from 5% to 7%, slammed bond prices. Stock pain, bond pain.

The intuition: a falling discount rate is what lets bonds hedge stocks, so a shock that *raises* the discount rate breaks the hedge — the denominator that prices stocks is the same rate that prices bonds.

## The history: the sign has flipped three times

The single best way to internalize that this is a regime and not a constant is to walk the actual history. Look again at the cover figure: it tells a three-act story, and each act maps cleanly onto which shock dominated.

### Act one: positive, the inflationary era before 2000

For most of the postwar period up to around 2000, the stock-bond correlation was **positive**. The dominant economic fear of that era was inflation. Through the 1970s and into the early 1980s, inflation ran hot — at times into the double digits — and the great market events were *inflation* events. When inflation surged, the Fed hiked rates, and both stocks and bonds suffered together. The era midpoints on the chart tell the tale: roughly **+0.30 in 1973**, **+0.35 in 1980**, **+0.30 in 1990**, **+0.20 in 1998**. For an investor in that world, bonds were not much of a diversifier against stocks. The 60/40 as a *crash hedge* was a weaker idea than it would later become, because the two legs leaned the same way whenever inflation flared.

### Act two: negative, the Great Moderation of 2000 to 2021

Then the sign flipped, and stayed flipped for two decades. From the dot-com bust through the pandemic, inflation was low and stable — economists call this stretch the **Great Moderation**. With inflation tamed, the dominant fear was *growth*: the 2000–2002 tech recession, the 2008 financial crisis, the 2011 and 2018 growth scares, the 2020 COVID crash. Every one of these was a growth shock, and in every one the Fed responded by cutting rates, so bonds rallied as stocks fell. The correlation went firmly negative — about **−0.20 in 2002**, **−0.40 in 2008**, **−0.35 in 2012**, **−0.30 in 2018**, **−0.25 in 2020**. This is the era that made the 60/40 legendary. A whole generation of investors and advisors learned, as if it were a law of physics, that bonds rise when stocks fall — because for their entire careers, they did.

### Act three: positive again, since 2022

And then, in 2022, the sign flipped *back* — hard. Inflation returned for the first time in forty years, the Fed hiked at the fastest pace in a generation, and the correlation rocketed to roughly **+0.55 in 2022**, holding near **+0.50 in 2023** and **+0.45 in 2024**. The Great Moderation assumption — bonds hedge stocks — quietly stopped being true, and millions of portfolios built on it took damage they were never warned about. The chart's right edge is the most important part of the whole picture: it shows the line crossing back above zero, which is the market telling you the inflation regime has the wheel again.

The lesson of the three acts is blunt: **the sign tracks the dominant shock.** Inflation-driven eras run positive; growth-driven eras run negative. If you know which fear is in charge, you know the sign, and if you know the sign, you know whether your bonds are a hedge or a liability.

## 2022: the year the engine broke

No year teaches this lesson like 2022, so let us spend real time on it. It is the cleanest natural experiment in modern markets for what happens when the correlation flips while millions of people are leaning on it being negative.

### The setup

Coming into 2022, the world looked deceptively calm. Inflation had been dormant for a decade. The Fed's policy rate sat at essentially zero — a target range of 0.00%–0.25%. Bond yields were near historic lows, which is another way of saying bond *prices* were near historic highs. The 60/40 had just delivered a strong 2021 (about **+16.6%**). Every condition that had made the 60/40 reliable seemed intact. It was not.

### The trigger: an inflation shock for the ages

Inflation, which had begun rising in 2021 on pandemic supply snarls and stimulus, broke into the open. By June 2022, the Consumer Price Index — the **CPI**, the headline measure of how fast consumer prices are rising — hit **9.06% year over year**, a 40-year high. This was the inflation shock, full-blown. And the Fed did exactly what the inflation-shock path predicts: it hiked, and hiked, and hiked. Over the course of 2022, the policy rate went from **0.25% to 4.50%** — one of the fastest tightening cycles in history. (For why a hiking cycle is the most dangerous environment a bond can face, see [government bonds, the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).)

### The result: both legs fell

Now run the mechanism. Rising rates crushed bond prices — the US Aggregate Bond Index, the benchmark for high-quality American bonds, fell **−13.0%**, its worst calendar year ever. Rising rates also crushed stocks by raising the discount rate on their future earnings, and the S&P 500 fell **−18.1%**. Both legs of the 60/40 fell at once, for the *same reason* — the rate shock — and the blended portfolio dropped **−16.0%**, its worst showing since 1937.

The bar chart below shows the 60/40's calendar-year returns across 2014–2024. Notice how unremarkable most years are — a steady drumbeat of single-to-double-digit gains, with the occasional mild dip. Then 2022 sticks down like a broken tooth. That single red bar is the regime change made visible.

![Bar chart of 60/40 portfolio calendar-year returns 2014 to 2024 with 2022 highlighted in red](/imgs/blogs/stock-bond-correlation-the-60-40-engine-3.png)

### Why diversification failed exactly when it was needed

Here is the cruel twist that makes 2022 worth studying forever. Diversification is supposed to be insurance: you accept slightly lower returns in good times in exchange for protection in bad times. But the protection is only real if the two assets actually move apart *when it matters*. In 2022, the thing that hurt stocks — surging rates — was the *exact same thing* that hurt bonds. There was no offset, because both losses came from one cause. The hedge didn't underperform; it inverted. The 40% you held specifically to cushion an equity decline instead *deepened* it.

This is the deepest point in the post: **diversification depends on a relationship, not on owning two different things.** Owning stocks and bonds is only diversified if their correlation is negative. When an inflation shock pushes that correlation positive, holding two assets that both hate rising rates is not diversification at all — it is the same bet, twice.

#### Worked example: the same \$100,000, two opposite outcomes

This is the heart of the whole post, so let us make it concrete with one portfolio in two regimes. You hold \$100,000 in a 60/40 — \$60,000 stocks, \$40,000 bonds.

**Growth-shock year (2008-style):**

- Stocks −37% on \$60,000 = **−\$22,200**.
- Bonds +5% on \$40,000 = **+\$2,000** (the Fed cut rates, so bonds rose).
- Net = −\$22,200 + \$2,000 = **−\$20,200**, or **−20.2%**. The bonds *softened* the blow by \$2,000.

**Inflation-shock year (2022 actuals):**

- Stocks −18.1% on \$60,000 = **−\$10,860**.
- Bonds −13.0% on \$40,000 = **−\$5,200** (the Fed hiked rates, so bonds fell).
- Net = −\$10,860 − \$5,200 = **−\$16,060**, or **−16.1%**. The bonds *added* \$5,200 to the loss.

Look at the bond line in each case. The *same* 40% bond sleeve handed you **+\$2,000** in the growth shock and **−\$5,200** in the inflation shock — a \$7,200 swing in how your "safe" money behaved, driven entirely by the sign of the correlation. Same portfolio, same assets, opposite bond behavior.

The intuition: your bonds are not reliably safe money — they are *conditionally* safe, and the condition is that the shock be a growth shock, not an inflation shock.

The side-by-side chart below makes that swing impossible to miss. On the left, 2008: stocks deep red, bonds green and *positive* — the hedge worked. On the right, 2022: stocks red, bonds red too — the hedge broke. Same two assets, opposite jobs, because the correlation sign flipped between the two years.

![Grouped bar chart comparing 2008 and 2022 returns for stocks and bonds showing bonds hedged then hurt](/imgs/blogs/stock-bond-correlation-the-60-40-engine-4.png)

## How to read the regime you are in

If the sign of the correlation is this important and it changes, the practical question becomes: how do you tell which regime you are standing in *right now*, without waiting for a crash to find out the hard way? There is no single magic indicator, but there is a short, reliable checklist. It comes down to one master variable — inflation — and one behavioral tell — what the Fed is reacting to.

### Signal 1: the level of inflation

The first and most powerful tell is simply *how high inflation is*. When inflation is low — say, the CPI running near the Fed's 2% target — the central bank has the freedom to fight *growth* problems. If the economy weakens, it can cut rates without worrying about reigniting prices. That freedom is what lets bonds hedge: a weak economy means rate cuts means bond gains. Low inflation is the home turf of the negative correlation.

When inflation is high — the CPI running at 4%, 6%, 9% — the central bank loses that freedom. It *must* fight inflation by hiking, even into a weakening economy, even if that deepens a slump. When the Fed is forced to hike, bonds cannot rally to hedge stocks, because hiking is precisely what sinks bonds. High inflation is the home turf of the positive correlation. The rough threshold practitioners watch is somewhere around 3–4% CPI: below it, the negative-correlation world tends to hold; sustainably above it, the positive-correlation world takes over.

### Signal 2: the volatility of inflation

Subtler but just as important: it is not only the *level* of inflation but its *volatility* — how much it jumps around. Stable inflation, even at a moderate level, lets the Fed be predictable and lets bonds behave as a hedge. Volatile inflation, lurching up and down unpredictably, keeps the Fed reactive and keeps the inflation-shock risk live. Research into the correlation's history finds that *inflation volatility* is one of the cleanest predictors of when the sign turns positive. Calm, low inflation → negative correlation. Jumpy, high inflation → positive correlation.

### Signal 3: what is the Fed reacting to?

The cleanest behavioral tell is to ask one question: **is the Fed cutting to rescue growth, or hiking to fight inflation?** This is the regime in one sentence.

- If the Fed is *cutting* rates (or clearly poised to) because it is worried about growth, you are in a growth-shock world. The correlation is negative. Your bonds are a hedge. Lean on them.
- If the Fed is *hiking* rates because it is worried about inflation, you are in an inflation-shock world. The correlation is positive. Your bonds are not a hedge. Do not lean on them.

This single question cuts through almost everything. (For the full machinery of how rate expectations ripple through every asset, see [real versus nominal: inflation, real yields, and the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).) You are not trying to forecast the economy from scratch — you are reading what the most powerful institution in markets is *already* doing, and inferring the correlation regime from it.

### Why this is harder than it sounds

Honesty requires admitting that reading the regime in real time is harder than reading it in a history book. Three traps are worth naming. First, the correlation is measured over a trailing window, so the *number* always lags the *regime* — by the time a two-year rolling correlation has clearly turned positive, the inflation shock is already well underway. The signals to watch are the leading ones — the level and direction of inflation, and what the Fed is reacting to — not the lagging correlation statistic itself. Second, regimes do not announce themselves; the transition from "low, stable inflation" to "high, volatile inflation" looks ambiguous for months, and reasonable people disagree about which side of the line they are on. Third, there is a strong pull to assume the regime you grew up with is permanent — the investors most blindsided by 2022 were precisely those whose entire careers fell inside the negative-correlation era, who had never personally seen the sign flip. The defense against all three traps is the same: anchor on the observable drivers — inflation and the Fed — rather than on the comforting habit of the recent past, and hold the portfolio that matches the regime you can see, while staying humble about exactly when it will turn.

#### Worked example: reading the regime in March 2022 versus March 2020

Put the checklist to work on two real moments two years apart.

- **March 2020 (COVID crash).** Inflation was low (CPI around 1.5%). The shock was a sudden growth collapse — the economy froze overnight. The Fed slashed rates to zero and flooded the system with support. *Regime read:* growth shock, Fed cutting, correlation negative, bonds a hedge. And indeed, as the S&P plunged about 34% from its February peak to its March trough, Treasuries rallied hard, cushioning balanced portfolios. A \$40,000 bond sleeve in a 60/40 *gained* value while stocks fell.
- **March 2022 (rate-shock begins).** Inflation was high and rising (CPI around 8% and climbing toward 9%). The shock was inflation. The Fed had just begun hiking and signaled many more to come. *Regime read:* inflation shock, Fed hiking, correlation positive, bonds *not* a hedge. And indeed, over the rest of 2022 that same \$40,000 bond sleeve *lost* about \$5,200 as rates marched higher.

The intuition: you did not need to predict the future in either case — you only had to read the inflation level and the Fed's direction, and the correlation sign followed.

### Why the magnitude matters too, not just the sign

So far we have focused on the *sign* of the correlation, which decides the direction. But how *much* damage a regime flip does to your portfolio also depends on a second number — the **duration** of your bonds. Duration is the sensitivity of a bond's price to interest rates: a bond with a duration of 7 loses roughly 7% of its value for every 1-percentage-point rise in yields, and gains roughly 7% for every 1-point fall. (We build duration carefully in [government bonds, the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration); here we only need the headline.)

Duration is what makes a regime flip especially dangerous, because it cuts both ways with the correlation. In a *negative*-correlation, growth-shock world, high duration is your friend: when the Fed cuts hard, long-duration bonds rise the most, delivering the most powerful hedge. That is why long Treasuries returned **+25.9%** in 2008 while the broad bond index returned only **+5.2%** — the extra duration multiplied the hedge. But in a *positive*-correlation, inflation-shock world, that same high duration becomes your enemy: when the Fed hikes hard, long-duration bonds fall the most, deepening the loss. The very feature that makes a bond a great hedge in one regime makes it a wrecking ball in the other.

This is why the regime read and the duration choice have to be made together. It is not enough to decide *how much* to hold in bonds; you also have to decide *what kind*. In a clear growth-shock regime you might reach for long duration to maximize the hedge. In an inflation-shock regime you want short duration or cash, to minimize the damage from rising rates. The same 40% bond allocation can behave like three completely different positions depending on the duration you choose — and the regime tells you which one you want.

#### Worked example: duration turning the dial in 2022

Take the same \$40,000 bond sleeve of a \$100,000 60/40, in 2022's roughly 2-percentage-point rise in yields, and see how three duration choices fared.

- **Short duration (≈ 2).** Loss ≈ 2 × 2% = 4% of \$40,000 = **−\$1,600**. Painful but survivable.
- **Medium duration (≈ 6, the broad index).** Loss ≈ the actual −13.0% on \$40,000 = **−\$5,200**. This is the real 2022 outcome.
- **Long duration (≈ 17).** Loss ≈ 17 × 2% = 34% of \$40,000 = **−\$13,600**. A catastrophe — long Treasury funds did indeed fall around 30% in 2022.

The same regime, the same allocation size, but the loss ranged from \$1,600 to \$13,600 depending on duration alone. In an inflation-shock regime, shortening duration is not a minor tweak — it is the difference between a scratch and a deep wound.

The intuition: the correlation's sign tells you whether bonds help or hurt, and duration tells you *how much* — so in a bad regime, cutting duration is your most powerful lever.

## Common misconceptions

This relationship is so widely misunderstood that it is worth naming the specific wrong beliefs head-on, because each one has cost real investors real money.

**Misconception 1: "Bonds always go up when stocks go down."** This is the big one, and it is false. Bonds go up when stocks go down *only in a growth shock*, when the Fed is cutting rates. In an inflation shock, bonds go *down* with stocks. The correct statement is "bonds tend to rise when stocks fall *for growth reasons*." In 2022, stocks and bonds fell together by double digits — the cleanest possible refutation of the "always" version.

**Misconception 2: "The 60/40 is dead."** After 2022, this became a popular headline, and it is an overcorrection. The 60/40 is not dead — it is *regime-dependent*. It works beautifully when inflation is low and stable (most of the last forty years) and poorly when inflation is high and volatile. Declaring it dead is just the mirror-image error of assuming it always works. The right view is conditional: know the regime, and size your bonds accordingly.

**Misconception 3: "Diversification means owning lots of different assets."** Diversification is not about *count*; it is about *correlation*. Owning ten assets that all fall together in a rate shock is not diversified. In 2022, stocks, investment-grade bonds, high-yield bonds, and many other "different" assets all fell together because they all hated rising rates. True diversification requires holding something whose correlation to your main risk is low or negative *in the regime you are actually in*.

**Misconception 4: "Bonds are the safe asset."** Bonds are *lower-volatility* than stocks, but "safe" is doing too much work. A long-dated bond can lose 20%+ in a year when rates spike. The US Aggregate's −13.0% in 2022 was its worst year on record. Bonds are safe against *one* thing — a growth-driven recession — and dangerous against another — an inflation-driven rate shock. Calling them "safe" without that qualifier is how retirees got blindsided.

**Misconception 5: "If the correlation flipped, it will flip right back."** Maybe — but regimes can last decades. The negative-correlation era ran from 2000 to 2021, twenty-plus years. The positive-correlation era before it ran for decades too. Do not assume the sign will snap back on your preferred timeline. Position for the regime you can actually observe, not the one you are nostalgic for.

**Misconception 6: "A higher bond yield makes bonds a better hedge again."** It is tempting to think that once yields have risen, bonds offer more cushion, because they now pay more income and have more room to rally if the Fed eventually cuts. There is some truth here — a bond yielding 4.5% has far more *potential* to hedge a future growth shock than a bond yielding 0.5%, simply because rates have somewhere to fall to. But this is a statement about a *future* growth-shock regime, not about the present inflation-shock one. While inflation is still high and the Fed is still hiking, a higher starting yield does not stop bonds from falling further as rates climb. The higher yield reloads the hedge for the *next* growth shock; it does not protect you in the *current* inflation shock. Confusing the two is how investors "bought the dip" in long bonds in mid-2022 and watched them fall further.

**Misconception 7: "The 60/40's 40% should always be bonds."** The "40" in 60/40 is shorthand for "the defensive sleeve," not a commandment that the defensive sleeve must be bonds in every regime. When bonds are not defensive — when the correlation is positive — the *spirit* of the 60/40 is better served by filling part of that 40% with whatever actually plays defense in the current regime: cash, commodities, gold, or inflation-linked bonds. Treating "40% bonds" as sacred rather than "40% defense" is exactly the rigidity that turned a sensible framework into a trap in 2022.

## How it shows up in real markets

Abstract rules become believable when you watch them play out in named, dated episodes. Here are the cases that matter most, each one the mechanism in action.

### 2008: the growth shock that made the 60/40 famous

The 2008 global financial crisis was a pure growth shock — a collapse in credit and demand. As the S&P 500 fell **−37.0%**, the Fed slashed rates toward zero and investors fled to the safety of government debt. The US Aggregate Bond Index returned **+5.2%**, and *long-dated* Treasuries — the most rate-sensitive bonds — returned a spectacular **+25.9%**. A 60/40 holder lost far less than a stock holder, precisely because the bond sleeve gained while stocks cratered. This is the textbook negative-correlation case, and it is the experience that taught a generation to trust the hedge.

### 2020: the growth shock at warp speed

The COVID crash of March 2020 compressed a full bear market into about five weeks. The S&P fell roughly 34% from its February peak to its March trough. Inflation was low, the shock was a growth collapse, and the Fed responded with emergency rate cuts and massive support. Treasuries rallied as stocks fell, hedging balanced portfolios through the worst of it; for the full calendar year, even as stocks recovered to finish **+18.4%**, the US Aggregate returned **+7.5%**. Another growth shock, another negative-correlation outcome, another vindication of the 60/40 — which is exactly why 2022 caught so many off guard.

### 2022: the inflation shock that broke the hedge

We have covered this in depth, but it belongs in the catalog as the defining counter-example. CPI hit **9.06%** in June. The Fed hiked from **0.25% to 4.50%** in a single year. Stocks fell **−18.1%**, bonds fell **−13.0%**, and the 60/40 fell **−16.0%** — worst since 1937. The only major asset that *rose* was commodities (the Bloomberg Commodity Index gained about **+16%**), because commodities are the thing that *goes up* in an inflation shock. That detail is the whole playbook in miniature, and we will build on it next.

### The 1970s: the original inflation regime

For the longest historical perspective, look at the stagflation of 1973–74 — "stagflation" being the ugly combination of stagnant growth and high inflation. Across those two years, US stocks fell about **−37%** cumulatively while inflation raged. Bonds, battered by rising rates, gave no meaningful protection — long Treasuries eked out only about **+4%** in nominal terms while inflation ran over 20% cumulatively, a deep *real* loss. The assets that protected wealth were the inflation hedges: gold rose enormously, and commodities surged with the oil embargo. The 1970s are the living proof that in an inflation regime, the hedge you need is not bonds — it is real assets.

### 2013: a preview that nobody heeded

There was a warning shot before 2022, and most investors missed it. In mid-2013, the Fed merely *hinted* that it might slow its bond-buying support program — an episode the press dubbed the "taper tantrum." It was not even a rate hike, just a signal that easy policy might tighten. Yet bond yields jumped sharply, and for a stretch that summer stocks and bonds *fell together* as the rate scare hit both. The US Aggregate Bond Index finished 2013 down about **−2%** — a losing year for "safe" bonds — even though stocks had a strong year overall. The episode was brief and the negative-correlation regime quickly reasserted itself, so it was filed away as a curiosity. But it was a genuine preview of the mechanism: when the *rate* itself is the shock, both legs of the 60/40 feel it at once. The 2013 tantrum was the small tremor before the 2022 earthquake, and it carried the same lesson for anyone paying attention.

### Why the order matters

Lay these episodes side by side and the pattern is undeniable. Every growth shock (2008, 2020) produced a negative correlation and a working bond hedge. Every inflation shock (the 1970s, 2022) produced a positive correlation and a failed bond hedge. The asset classes never changed their nature. What changed was the shock — and the shock is what you can actually read in advance, from inflation and the Fed.

## The allocation playbook: when bonds protect you and what to do when they don't

This is the payoff. Everything above exists to make this section actionable. The goal is not to tell you a fixed allocation — it is to teach you to *adjust* based on the regime you can observe, so that you are never again surprised by your own bonds.

The matrix below is the playbook in one picture. Three rows for three inflation environments; the columns tell you the correlation, what bonds will do, and what to hold. Use it as a decision aid: identify your row from the inflation data and the Fed, then read across.

![Regime playbook matrix mapping inflation environment to correlation bond behavior and what to hold](/imgs/blogs/stock-bond-correlation-the-60-40-engine-7.png)

### When 60/40 works: low and stable inflation

When inflation is low (CPI near 2%) and stable, and the Fed's job is to manage *growth*, the classic 60/40 is genuinely excellent. The correlation is negative, bonds hedge stocks, and the blend gives you most of equities' growth with a fraction of the heartburn. In this regime:

- Hold the classic 60/40 (or your risk-appropriate variant — 70/30 if younger, 50/50 if closer to spending the money).
- You can lean into *duration* — longer-dated bonds — because in a growth shock, long bonds deliver the biggest hedge (recall 2008's **+25.9%** for long Treasuries). Long bonds are explosive ballast when the Fed is cutting.
- This is the home base. Most of the time, for most of history, this is where you live, and the 60/40 earns its reputation here.

### When inflation is the risk: add the assets that actually hedge it

When inflation is high or rising or volatile, and the Fed is hiking to fight it, the correlation goes positive and bonds stop protecting you. This is the regime where you must *change the portfolio*, because more bonds will not help — they are part of the problem. What hedges an inflation shock are the assets that *rise* when prices and rates rise:

- **Commodities.** Raw materials — energy, metals, agriculture — are the cleanest inflation hedge, because they *are* the prices that are rising. In 2022, while stocks and bonds both fell double digits, the Bloomberg Commodity Index rose about **+16%**. A commodity sleeve is the single most direct offset to an inflation shock.
- **Gold.** A long-standing store of value that tends to hold up when faith in paper currency and bonds wobbles. Its record is streakier than commodities, but it earns its place as inflation insurance.
- **Cash and short-term instruments.** In a hiking cycle, cash is paradoxically powerful: it loses nothing to rising rates (it has near-zero *duration*, the sensitivity to rate moves), and it *earns more* as the Fed hikes. In 2022, short T-bills went from yielding almost nothing to over 4%, while bonds lost 13%. Cash is the asset that quietly wins a rate shock. (More on this underrated asset in [government bonds, the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).)
- **Inflation-linked bonds (TIPS).** *TIPS* — Treasury Inflation-Protected Securities — are government bonds whose principal rises with inflation, so they hedge the inflation component directly. They still carry duration risk, so they are not a perfect shield in a hiking cycle, but they are built for exactly this fear.

The thread connecting all four: they are the things that *go up* when the inflation-shock chain runs, precisely because that chain is what was hurting your stocks and bonds.

How big should the inflation-hedge sleeve be? There is no universal answer, but the logic is one of proportion to the risk. In a clearly low-inflation world you might hold none of it and run the classic 60/40. As inflation rises and the correlation turns positive, a sleeve in the rough range of 5% to 20% — carved out of stocks and bonds both — is enough to materially soften an inflation shock without abandoning the growth engine of equities. The point is not to bet the portfolio on commodities; it is to hold *enough* of the right hedge that, when bonds fail you, something else is rising. A small, deliberate inflation sleeve is cheap insurance in the regime where the usual insurance has stopped working.

#### Worked example: bolting an inflation hedge onto the 60/40 in 2022

Start with the unmodified 60/40 in 2022 — the **−16.1%** we computed earlier (\$60,000 stocks losing \$10,860, \$40,000 bonds losing \$5,200). Now suppose that, reading the regime, you had carved out a 10% commodity sleeve, holding 55% stocks, 35% bonds, 10% commodities on your \$100,000.

- Stocks −18.1% on \$55,000 = **−\$9,955**.
- Bonds −13.0% on \$35,000 = **−\$4,550**.
- Commodities +16.0% on \$10,000 = **+\$1,600**.
- Net = −\$9,955 − \$4,550 + \$1,600 = **−\$12,905**, or **−12.9%**.

By swapping a slice of the portfolio into the asset that hedges *this* regime's risk, you turned a −16.1% year into a −12.9% year — a 3.2-percentage-point rescue, worth about \$3,200 on \$100,000 — not by predicting the crash, but by holding the right hedge for the regime you were in.

The intuition: when bonds cannot hedge, the fix is to add the asset that *can* — in an inflation shock, that asset is whatever rises when prices and rates rise.

### What invalidates the case, and the one rule

Be honest about what would make each call wrong, because that is how you avoid overconfidence:

- The negative-correlation, lean-on-bonds case is invalidated the moment inflation climbs sustainably above ~3–4% and the Fed pivots from worrying about growth to worrying about prices. That is your signal to trim duration and add inflation hedges.
- The positive-correlation, add-hedges case is invalidated when inflation falls back toward target and the Fed's focus returns to growth. That is your signal to let the classic 60/40 resume its work and rebuild duration.

And here is the one rule that sits above all the tactics, the sentence to tape to your monitor: **never assume bonds will save you — check the regime first.** The 60/40 is not a permanent truth and it is not dead; it is a bet on a negative stock-bond correlation, and that bet pays off in a growth-shock world and loses in an inflation-shock world. Your job, every year, is to ask the simple question — *is the dominant fear growth or inflation, and is the Fed cutting or hiking?* — and to hold the hedge that matches the answer. The growth-of-\$100,000 chart below is the reward for getting this right over time: the 60/40 line rides smoothly between all-stocks and all-bonds, scarred only where 2022's rate shock caught both legs at once.

![Line chart of 100000 dollars growing in 60/40 versus all stocks versus all bonds from 2013 to 2024](/imgs/blogs/stock-bond-correlation-the-60-40-engine-6.png)

The retiree from the opening of this post was not foolish. They held a sensible portfolio built on a relationship that had held for their entire adult life. Their only mistake was treating a *regime* as a *law*. Now you know better. The stock-bond correlation is the engine inside the 60/40, and like any engine, it runs on a specific fuel: a negative correlation, which itself depends on the world fearing slow growth more than high prices. Watch the inflation data, watch the Fed, and you will always know whether the engine is running — and whether your bonds are an airbag or just more weight in the crash.

## Further reading and cross-links

- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — the deeper math of why low and negative correlations are the only thing that makes diversification work, and why count is not the same as diversification.
- [Government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — what a bond actually is, the price-yield seesaw, and why duration governs how violently a rate shock hits your bond sleeve.
- [Equities: stocks, owning a slice of growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth) — why a stock is a claim on future cash flows, and how the discount rate quietly prices it.
- [The map of asset classes: what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — where stocks, bonds, and the inflation hedges sit in the full landscape, and how the pieces fit together.
- [Real versus nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — how inflation and rate expectations ripple through every asset, why the real yield sits underneath both stocks and bonds, and how reading them tells you the regime.

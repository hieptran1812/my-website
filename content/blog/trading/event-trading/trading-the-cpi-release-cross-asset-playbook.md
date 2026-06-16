---
title: "Trading the CPI Release: A Cross-Asset Playbook"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A hands-on playbook for the 8:30 ET CPI print: build the if-then map for stocks, crypto, the dollar, gold and bonds before the release, then trade the reaction rather than the number."
tags: ["event-trading", "cpi", "inflation", "cross-asset", "crypto", "stocks", "fx", "gold", "bonds", "trading-playbook", "macro"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — On a CPI morning you do not trade the inflation number; you trade the *surprise* (actual minus what was already priced in) and the *reaction* it sets off across every market at once. Win the morning before 8:30 ET by writing the if-then map, not by predicting the print.
>
> - **What CPI is:** the monthly US Consumer Price Index, released by the Bureau of Labor Statistics at 8:30 a.m. Eastern. It is the single most-watched read on inflation, and inflation is what tells the market how much the Federal Reserve will squeeze.
> - **How the market reacts:** the *sign* of the surprise sets the *direction* of everything. A hot print (inflation above consensus) usually means stocks down, dollar up, gold down, Bitcoin down, yields up. A cool print is the mirror image. In-line means a volatility crush and a chop you usually fade.
> - **The trade:** size to the **expected move** the options market already quotes (roughly **plus or minus 1.2%** on the S&P 500 on a normal CPI day), pick the branch the data lights up, then decide fade-or-trend in the first few minutes. Reaction beats number.
> - **The one number to remember:** same report, opposite outcomes — the S&P fell **−4.32%** on the hot Sep 13, 2022 print and rose **+5.54%** on the cool Nov 10, 2022 print. The number both days was high inflation. Only the surprise sign differed.

Two CPI mornings, eight weeks apart, tell you everything about how to trade this release.

On **September 13, 2022**, the Bureau of Labor Statistics reported that consumer prices had risen 8.3% over the prior year. Economists had penciled in 8.1%. That is a tiny miss — two-tenths of a percentage point — and yet by the closing bell the S&P 500 was down **−4.32%**, the Nasdaq had cratered **−5.16%**, the US dollar index had jumped **+1.4%**, and Bitcoin had bled roughly **−9.4%**. It was the worst single CPI-day for stocks in over two years. The headline number, 8.3%, was actually *lower* than the prior month. Inflation was falling. Stocks crashed anyway.

On **November 10, 2022**, the same agency reported 7.7% inflation against a 7.9% consensus — again a two-tenths miss, this time to the downside. The S&P rocketed **+5.54%**, the Nasdaq **+7.35%**, the 10-year Treasury yield collapsed **−28 basis points**, the dollar fell **−2.1%**, and Bitcoin ripped roughly **+10%**. It was one of the best days for US stocks of the entire decade. Inflation was still 7.7% — historically enormous. Stocks soared anyway.

Same report. Same agency. Same 8:30 a.m. release time. Both prints showed inflation that any central banker would call alarming. And the two days produced almost perfectly mirror-image outcomes across every asset class on the screen. If you had spent those mornings staring at the inflation *level* and asking "is inflation high?" you would have been confused both times. The market was not asking that. It was asking a different, sharper question — and that question is the whole game.

![CPI day if-then map showing how a print routes to hot in-line or cool branches with the cross-asset move on each](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-1.png)

## Foundations: how to prepare for a CPI print

Before we trade anything, we have to build the right mental model. Almost every beginner who loses money on CPI day loses it for the same reason: they trade the *number* instead of the *surprise*. Let us take that apart, one term at a time, because once you see it you cannot unsee it.

### What CPI actually is, and who cares

The Consumer Price Index measures the average change in prices that US households pay for a fixed basket of goods and services — groceries, rent, gasoline, airline tickets, haircuts, used cars, and so on. The BLS surveys tens of thousands of prices every month and rolls them into one index. The number traders care about is usually the **year-over-year** change (this month's index versus the same month a year ago) and the **month-over-month** change (versus the prior month). When a news headline says "CPI came in at 3.2%," it almost always means year-over-year.

Why does a basket of grocery and rent prices move global markets? Because of one institution: the **Federal Reserve**, the US central bank. The Fed's job includes keeping inflation around 2% per year. When inflation runs hot, the Fed raises interest rates to cool the economy down; when inflation is tame or falling, the Fed can hold rates steady or cut them. Interest rates are the price of money, and the price of money is the gravity that pulls on every asset on Earth — stocks, bonds, the dollar, gold, and crypto included. So CPI is not really a story about grocery prices. It is the market's best monthly preview of *what the Fed will be forced to do*. (For the mechanism — how the Fed turns inflation data into a rate decision — see the macro-trading companion on [inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).)

### The single most important idea: price already contains the consensus

Markets are forward-looking. Weeks before a CPI release, professional forecasters publish their estimates, and the median of those estimates becomes the **consensus** — the number "everyone" expects. There is often a "whisper" number too, the slightly different figure that active desks are really positioned for, which can drift from the published consensus in the days before the print. Either way, the crucial fact is this: **the consensus is already baked into prices before the release.** Traders have spent weeks buying and selling in anticipation of it.

That means the consensus number itself moves nothing on release. The only thing that can move price at 8:30:00 is the gap between what actually prints and what the market already expected. We call that gap the **surprise**:

> surprise = actual − consensus

If consensus is 3.2% and the print is 3.2%, the surprise is zero, and in theory nothing should move (it is already priced). If the print is 3.5%, that is a **hot** surprise of +0.3 percentage points: inflation worse than feared, more Fed tightening implied. If the print is 2.9%, that is a **cool** surprise of −0.3 points: inflation better than feared, less Fed needed.

This is why both 2022 examples make sense. On September 13, inflation *fell* from the prior month — but it fell less than the market had priced, so the surprise was *hot*, and stocks crashed. On November 10, inflation was still 7.7% — historically huge — but it came in *below* what was priced, so the surprise was *cool*, and stocks soared. The level was high both times. The surprise had opposite signs. The surprise won.

### The expected move: how big a swing is already priced

Before you can size a trade, you need to know how big a move is *normal* for this event. You do not have to guess — the options market quotes it for you. The price of a short-dated **straddle** (buying both a call and a put at the current price, which pays off if the market moves far in either direction) is the market's own estimate of how far the underlying is likely to travel by expiry. On a typical CPI day in a calm regime, the S&P 500 straddle implies an **expected move of roughly plus or minus 1.2%** — meaning the market thinks a one-standard-deviation reaction is about 1.2% in either direction. (We have a full treatment in [the expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).)

That single number is your risk yardstick. It tells you how wide to set stops, how small to size, and — crucially — when a move is "normal" versus a genuine outlier worth chasing. A 0.6% post-CPI move is *inside* the expected move: nothing special happened, often a fade. A 4% move (like September 2022) is more than three times the expected move: a true regime-confirming shock you respect and, often, trend with.

### The hot / in-line / cool framework

Every CPI print drops the tape down one of three branches, and your job before 8:30 is to write down what you will do on each. The figure above is exactly that map. In words:

- **Hot (actual above consensus):** inflation worse than priced. In the 2022–2023 inflation regime, the market read this as "the Fed will hike more / hold higher for longer." Result: stocks down, dollar up, gold down, Bitcoin down, yields up. Risk-off.
- **Cool (actual below consensus):** inflation better than priced. The market reads "the Fed can ease off." Result: the mirror image — stocks up, dollar down, gold up, Bitcoin up, yields down. Risk-on relief rally.
- **In-line (actual equals consensus):** no new information. Often you get a fast spike in *both* directions as algos and humans jockey, then a reversion as the surprise fails to materialize. This is the classic **fade** setup, traded small.

### The headline-versus-core trap

Here is the mistake that catches even experienced traders. CPI has two headline figures and the market does not weight them equally. **Headline CPI** includes everything, food and energy. **Core CPI** strips out food and energy because those two are volatile and noisy (a hurricane or an OPEC decision swings them around without telling you much about underlying inflation). The Fed cares more about *core* — and within core, increasingly about **core services** and so-called **supercore** (services excluding housing), because that is the stickiest, most wage-driven inflation.

So the trap is this: the headline number can print cool while core prints hot, or vice versa. The algorithms fire on the headline in the first milliseconds, then human desks read the core breakdown over the next few minutes — and the **second move can completely reverse the first.** A market can spike up on a cool headline, then roll over hard when traders see core was hot. If you bought the first spike, you are now offside. Trading the reaction, not the number, means *waiting to see whether the first move survives the core read.*

A concrete way to picture the hierarchy of what the market actually watches, from least to most important to the Fed:

1. **Headline year-over-year** — the number on the newspaper front page. Fast to parse, so the algos key off it, but the Fed treats it as noisy because food and energy bounce around.
2. **Headline month-over-month** — the most recent monthly change, annualized in traders' heads. A 0.1% monthly print annualizes to roughly 1.2%; a 0.4% monthly print annualizes to nearly 5%. The monthly number often tells the trend story better than the year-over-year, which is dragged around by what is rolling off from a year ago (the so-called "base effects").
3. **Core year-over-year and month-over-month** — food and energy stripped out. This is the Fed's first real cut.
4. **Core services** — within core, the services component (rent, insurance, medical care, haircuts), which is stickier and more wage-driven than goods.
5. **Supercore (core services ex-housing)** — the Fed's single favorite gauge, because it most closely tracks the labor-cost-driven inflation that monetary policy can actually influence. When a Fed official is quoted obsessing over one number, it is usually this one.

The further down that list a surprise sits, the more *durable* the reaction tends to be, because the deeper gauges move the Fed's *medium-term* view rather than just one data point. A hot supercore print is far more dangerous for risk assets than a hot headline driven by a one-off gasoline spike — even if the headline number looks identical. This is why two prints with the same headline can produce opposite reactions: the market is reading three layers below the headline.

### Fade versus trend: the only decision that matters

Once the print lands and the dust of the first thirty seconds clears, you face one fork:

- **Fade:** the initial reaction overshot, the surprise was small or ambiguous, and price is already reverting toward where it started. You fade by trading *against* the spike with a tight stop, betting it gives back the overshoot.
- **Trend:** the surprise was large and unambiguous, it confirms the prevailing regime, and price is extending in the surprise's direction with conviction. You trade *with* the move.

Most in-line and small-surprise prints are fades. Most large, regime-confirming surprises are trends. Getting this fork right is where the discretionary edge lives, and we will return to it in detail. (The general shape of every news reaction — the spike, then the fade or the trend — is the subject of [anatomy of a news reaction: spike, fade, trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend).)

With the foundations in place, let us walk the morning minute by minute.

## Before the print: what is priced, the expected move, and your if-then map

The most important work happens before 8:30. By the time the number drops, you should already know exactly what you will do in every scenario, because in the chaos of the first thirty seconds you will not have time to think — you will only have time to execute the plan you already wrote.

### Step one: find out what is already priced

Start with the consensus. Pull the median forecast for both headline and core, month-over-month and year-over-year, from any economic calendar. Note the prior month's reading too, because the market frames the surprise relative to expectations but *narrates* it relative to the trend (is inflation accelerating or decelerating?). Then check positioning: are traders leaning hot or cool into the print? If the market has aggressively sold stocks for three days into the release, a lot of "hot" fear is already in the price, and even a hot print can rally on relief once the event risk passes. This is why the surprise is necessary but not sufficient — *what was priced going in* shapes how the same surprise lands.

There is a subtlety worth naming: the *published* consensus (the median of surveyed economists) and the **whisper number** (the figure active desks are really positioned for) can diverge in the days before a release. When the financial press runs a string of "inflation may be stickier than forecast" stories, the whisper drifts hotter than the published consensus, and the market quietly pre-sells risk. In that case, a print that merely *matches* the published consensus can still produce a *relief rally*, because it came in cooler than the whisper the market was actually braced for. You cannot read the whisper off a calendar; you infer it from price action into the print. If risk assets have been sliding hard for several sessions ahead of a CPI release, treat the effective bar as hotter than the headline consensus.

A second subtlety is **base effects**. The year-over-year number compares this month's index to the same month a year ago. If a year ago had an unusually large monthly jump, that big number "rolls off" the back of the 12-month window this month and mechanically pulls the year-over-year figure down — even if current inflation is unchanged. Sophisticated desks know the base-effect calendar months in advance, so a year-over-year "improvement" that is purely a base effect is often already fully priced and moves nothing, while the *month-over-month* surprise (which carries the fresh information) does the real work. When you read the consensus, read the monthly number hardest; it is the cleanest measure of what is actually happening right now.

The scatter below makes the core relationship concrete. Each dot is a real CPI day: the horizontal axis is the surprise in percentage points (positive is hot, negative is cool), the vertical axis is the S&P's same-day move. The pattern is unmistakable — hot surprises sit in the lower-right (stocks down), cool surprises in the upper-left (stocks up). The sign of the surprise sets the sign of the move.

![Scatter plot of CPI surprise in percentage points versus S&P 500 same-day move showing hot surprises pushing stocks down and cool surprises pushing stocks up](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-2.png)

### Step two: read the expected move and size to it

Pull the S&P 500 (or your instrument's) implied move from the front-week straddle. Say it implies plus or minus 1.2%. That is now your unit of risk. The single most common way to blow up an account on event day is to size a position as though CPI were a quiet Tuesday and then get hit by a three-sigma move. Size so that the *expected* move costs you an amount you can shrug off, and the *outlier* move — three times that — still does not threaten the account.

#### Worked example: sizing a position to the expected move

You decide your maximum acceptable loss on this CPI trade is **\$500**. The S&P expected move is plus or minus 1.2%. You set your stop just beyond the expected move — call it 1.2% away from your entry. To make a 1.2% adverse move equal a \$500 loss, you solve for position size:

> position size = max risk ÷ expected move = \$500 ÷ 0.012 = **\$41,667**

So you can carry roughly a **\$41,000** position and still cap a 1.2% adverse swing at about \$500. If you instead want the stop at *twice* the expected move (2.4%, to survive normal whipsaw), you halve the size to about \$20,800 for the same \$500 risk. The expected move is not a forecast of direction; it is the ruler you use to size so a normal reaction never hurts more than you decided in advance.

### Step three: write the if-then map

Now write — literally, on paper or in a note — what you will do on hot, in-line, and cool. For each branch, specify: the asset, the direction, the entry trigger, the stop, and the invalidation (the level that says "I was wrong, get out"). When the number prints, you do not analyze. You glance at the surprise sign, confirm which branch fired, and execute the line you already wrote. This is the entire discipline of event trading compressed into one habit: **pre-commit to the plan so the adrenaline of 8:30 cannot rewrite it.**

## The knee-jerk: algos trade the headline in milliseconds

At 8:30:00.000 a.m. Eastern, the number hits the wire. What happens next is not human. Low-latency trading systems parse the machine-readable release in microseconds and fire orders before any person has read the first digit. In the first few hundred milliseconds, the entire move is algorithmic, and it keys almost entirely off the **headline surprise** — because that is the fastest, least ambiguous field to parse. The core breakdown, the internals, the revisions — those take seconds to digest, and seconds is an eternity to a machine.

This is why the first move is called the **knee-jerk**: it is a reflex, not a considered judgment. The tape gaps, futures lurch, the dollar snaps, yields jump — all within a second or two. For a retail or discretionary trader, the lesson is blunt: **you cannot win the first second.** You will never beat a co-located algorithm to the headline. So do not try. Your edge is not speed; it is judgment about what happens *after* the knee-jerk, once the machines have done their reflex and the slower, smarter money starts to read the details.

The figure below lays out the full five-phase clock of a CPI reaction, from your pre-positioning before the bell, through the knee-jerk, the core read, the fade-or-trend fork, and finally the settle into the close. Notice where the human edge lives: not in phase 2 (the knee-jerk, which belongs to the algos) but in phase 4 (fade or trend), where reading the internals and the price action pays.

![Timeline of the five phases of a CPI reaction from pre-position through knee-jerk core read fade or trend and settle](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-3.png)

A practical consequence: in the first thirty to sixty seconds, **spreads blow out and liquidity vanishes.** Market makers widen their quotes because they do not yet know if the knee-jerk is "real." If you stab a market order into that vacuum, you pay a brutal spread. Patient traders let the first spasm pass, let liquidity return, and only then engage. The chart of any CPI minute-by-minute shows it clearly: a violent first bar, often a second bar that partly reverses it, and only by the third or fourth minute a more orderly trend or fade.

It helps to understand *why* the knee-jerk is so violent. Going into 8:30, market makers have pulled in their risk — they quote thin because they do not want to be holding a large position when the number hits. So the standing order book is unusually shallow. When the print drops and a wave of algorithmic orders all fire in the same direction within milliseconds, that thin book gets swept through several price levels at once, producing an outsized gap. Then a second wave hits — some of it the algos closing the first trade, some of it slower participants reacting — and the book is still thin, so the reversal (if there is one) is also exaggerated. Only once liquidity providers regain confidence and re-post size does the market settle into something a human can trade with normal slippage. The first thirty seconds are not a market in any normal sense; they are a liquidity vacuum. Respecting that vacuum — not trading into it — is itself an edge, because it is precisely where impatient retail traders pay the worst prices of the entire session.

There is one more reason patience pays: **the algos do not have a view.** A latency-arbitrage system that buys on a cool headline in 2 milliseconds is not making a judgment about the Fed; it is exploiting the predictable, mechanical first reaction and will often be *flat again within seconds*, having captured the spread and the initial pop. Its exit is part of what causes the first reversal. The slower money that arrives with an actual macro thesis — confirmed by the internals — is what creates the durable trend (or the durable fade). Your timeframe overlaps with theirs, not with the algos'. You are trading the considered move, which is exactly why you should not be trying to compete in the reflex.

#### Worked example: what the knee-jerk does to a \$30,000 long

Suppose you walked into both 2022 prints holding a **\$30,000** long position in an S&P 500 index fund — no event hedge, just buy-and-hold.

> Hot print, Sep 13, 2022: \$30,000 × (−4.32%) = **−\$1,296**
>
> Cool print, Nov 10, 2022: \$30,000 × (+5.54%) = **+\$1,662**

Same position, same report two months apart, a swing of **\$2,958** between the two outcomes — about 10% of the whole position — decided entirely by the sign of a two-tenths surprise. The takeaway is not "predict the surprise"; nobody reliably can. It is that an un-hedged directional position is a coin flip on CPI morning, and you should either neutralize the event risk or size so a coin flip cannot hurt you.

## The core read: the second move when desks parse the internals

Once the knee-jerk fires, the slower, more valuable phase begins: human desks open the BLS release and read the *internals*. This is where the headline-versus-core trap springs, and where the second move is born.

The questions the desks ask in those first few minutes: Was the surprise in headline or in core? Was core driven by volatile items (used cars, airfares) that will not repeat, or by sticky items (shelter, core services) that will? What did **supercore** — services excluding housing, the Fed's favorite sticky-inflation gauge — do? Were there revisions to prior months that change the trend? The answers determine whether the knee-jerk gets *confirmed* (and trends) or *faded* (and reverses).

The classic reversal looks like this: headline CPI prints cool, the algos spike everything up, and then thirty seconds later the desks see that core was actually hot — driven by sticky shelter and services — and the rally rolls over, sometimes closing red on what was nominally a "cool" headline. Or the reverse: a hot headline driven entirely by a one-off energy spike, with core soft, and the initial selloff reverses into a green close because the *sticky* inflation that the Fed actually targets came in benign.

This is the deepest reason to **trade the reaction, not the number.** The number is the headline. The reaction — especially the *second* reaction, after the internals are read — is the market's considered verdict on what the print means for the Fed. A disciplined trader waits for that verdict. (For why the *same* number can produce different reactions depending on what the market is focused on, see [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).)

#### Worked example: fading an in-line overshoot

CPI prints exactly in line — zero surprise. But in the chaos, the S&P futures spike up about 0.8% in the first ninety seconds on a burst of short-covering, well beyond what a zero-surprise print justifies. You read this as an overshoot with nothing behind it and decide to fade: you short **\$15,000** of the index future with a tight stop just above the spike high.

Over the next several minutes the spike reverts as the surprise fails to materialize and price drifts back toward the pre-print level — a give-back of about 0.8%:

> \$15,000 × 0.8% = **+\$120**

A modest \$120 gain on a small position with a tight stop. The point is not the size of the win; it is the *setup*: an in-line print plus an unjustified overshoot equals a high-probability, tightly-stopped fade. You are not predicting inflation — you are betting a reflex spike with no information behind it cannot hold.

## The cross-asset map: how each market moves per scenario

CPI does not move one market. It moves *every* market at once, because every asset prices off the same underlying variable — the expected path of Fed policy and, through it, real yields and liquidity. When the surprise revises that path, the revision ripples outward through the entire cross-asset complex in a coordinated way. Understanding the *transmission* is what lets you trade the cleanest expression of the same view. (The full mechanism — how one print propagates through stocks, the dollar, gold, crypto and bonds — is the subject of [cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).)

Here is the chain on a **hot** print, in order of who moves first:

1. **Bonds move first.** A hot surprise means more Fed tightening priced, so traders sell short-dated Treasuries and yields jump (the 2-year is the most Fed-sensitive). Bond prices fall, yields rise. This is the *source* of the whole cascade.
2. **The dollar moves next, and cleanest.** Higher US yields make dollar-denominated assets more attractive relative to other currencies, so capital flows into the dollar and the DXY (dollar index) rises. FX is often the cleanest read on a CPI surprise because the rate differential is a direct, mechanical driver.
3. **Gold falls.** Gold pays no yield, so when *real* yields (yields after subtracting expected inflation) rise, the opportunity cost of holding gold rises and it sells off. Gold is essentially a trade on real yields. (The master-signal framing is in [real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)
4. **Stocks fall**, with the highest trading volume of any reaction. Higher rates lower the present value of future earnings — hardest on long-duration, high-growth tech (which is why the Nasdaq usually falls harder than the S&P), and the discount-rate hit plus recession fear sends equities down.
5. **Bitcoin and crypto fall hardest**, because crypto trades as a high-beta liquidity asset. When the Fed is expected to squeeze liquidity, the most speculative, longest-duration risk asset gets hit the most. Bitcoin is essentially the amplified version of the equity move. (Why crypto behaves like a macro liquidity asset is covered in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset).)

A **cool** print runs the exact same chain in reverse: yields fall, the dollar sells off, gold rallies, stocks rally (tech and small caps leading), and Bitcoin rips. The figure below shows the real numbers from the hot September 2022 print — the textbook risk-off pattern, with the dollar the lone green bar.

![Bar chart of cross-asset same-day moves on the hot September 2022 CPI with stocks crypto gold down and the dollar up](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-4.png)

And here is the mirror — the cool November 2022 print, where the entire pattern flips: risk assets up, the dollar the lone red bar.

![Bar chart of cross-asset same-day moves on the cool November 2022 CPI showing the mirror image with risk assets up and the dollar down](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-5.png)

Set the two charts side by side and the lesson lands: the cross-asset reaction is *coherent*. It is not five independent coin flips. It is one macro signal — the revised Fed path — expressing itself through five instruments at once. That coherence is your friend, because it means you can pick the *cleanest* expression of the view. If you think the surprise will be hot, you might short the asset with the highest, cleanest beta (crypto or Nasdaq) or the most mechanical reaction (the dollar), rather than fighting the noisy middle.

#### Worked example: a \$10,000 Bitcoin position on the cool print

Bitcoin's high beta cuts both ways. On the cool November 2022 print, Bitcoin rose roughly +10% on the day.

> \$10,000 × (+10%) = **+\$1,000**

A 10% day on a \$10,000 position is a \$1,000 swing — versus the +5.54% (about +\$554 on the same notional) you would have made in the S&P. Crypto gave you nearly double the equity move in the *same* direction. The flip side is exactly symmetric: on the hot September print, Bitcoin's −9.4% would have cost \$10,000 × (−9.4%) = **−\$940**, against the S&P's −\$432. Higher beta means crypto is the amplifier of the macro signal — bigger reward when right, bigger pain when wrong.

## Per asset: the hot reaction, the cool reaction, and what invalidates the trade

Each asset has a fixed hot response, a mirror cool response, and — most importantly — a written **invalidation**: the level or behavior that tells you the knee-jerk was wrong and you should be out. The invalidation is what separates a plan from a gamble. The figure below is the per-asset tree; study it before you trade.

![Per-asset if-then tree for stocks crypto dollar gold and bonds showing the hot reaction the cool reaction and the invalidation for each](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-6.png)

Walking each asset:

- **Stocks (S&P 500):** the highest-volume reaction. Hot → down, with rate-sensitive tech leading lower; cool → up, often with small caps leading. **Invalidation:** a clearly hot print that nonetheless *closes green* tells you the hot news was already priced (or the internals were soft) — do not stay short into strength.
- **Crypto (Bitcoin):** the highest beta to liquidity. Hot → down hard; cool → up hard. **Invalidation:** Bitcoin rising on a clearly hot print means the macro read is being overridden by something crypto-specific — stand down on the macro trade.
- **US dollar (DXY):** moves first and cleanest because it is a direct play on the rate differential. Hot → up; cool → down. **Invalidation:** the dollar *falling* on a clear hot print is a strong signal the surprise was not as hot as the headline (check core), and the whole risk-off thesis is suspect.
- **Gold:** trades real yields. Hot (real yields up) → down; cool (real yields down) → up. **Invalidation:** gold *rising while real yields rise* means something non-CPI is driving it (geopolitics, a flight to safety) — your CPI thesis is not in control.
- **Bonds (2-year and 10-year):** the source of the entire move. Hot → yields up (the 2-year leads, most Fed-sensitive); cool → yields down (the 10-year often moves most on the growth read). **Invalidation:** yields *falling* on a hot print means the bond market disbelieves the headline — and the bond market is usually right, so respect it.

#### Worked example: the 10-year rally on a \$500,000 bond position

The cool November 10, 2022 print sent the 10-year Treasury yield down 28 basis points — a massive one-day move for bonds. Bond prices rise when yields fall. To translate yield into money you use **DV01** (the dollar change in a position's value per 1 basis point move in yield). Suppose you hold a \$500,000 10-year Treasury position with a DV01 of about \$430 per basis point (roughly consistent with a ~8.6-year duration).

> gain = DV01 × basis points = \$430 × 28 = **+\$12,040**

A single cool CPI print added about **\$12,040** to the position — roughly +2.4% of notional in one morning, just from a two-tenths downside surprise. Bonds are not the boring asset on CPI day; they are where the move *originates*, and a 28bp day is an earthquake in fixed income.

#### Worked example: small caps lead on the November 2023 cool print

On November 14, 2023, the cool print sent the small-cap Russell 2000 up +5.44% — far more than the S&P's +1.91%. Small caps carry more debt and more domestic-rate sensitivity, so they swing hardest when the rate outlook eases. On a **\$15,000** Russell position:

> \$15,000 × (+5.44%) = **+\$816**

That same \$15,000 in the S&P would have made only \$15,000 × 1.91% = **+\$287**. Nearly triple the move for picking the higher-beta, more rate-sensitive expression of the *same* cool view. When you have a clean directional read on CPI, the cross-asset map tells you not just *which way* but *which instrument* gives you the most move per dollar of risk.

## Fade versus trend, and sizing for the expected move

Everything above funnels into one real-time decision: once the knee-jerk and the core read are done, do you fade the move or trend with it? Here is the decision rule, made concrete:

**Trend with the move when:**
- The surprise is *large* (well outside what was priced) and *unambiguous* (headline and core agree).
- It *confirms* the prevailing regime (in 2022–2023, a hot print confirming the "higher for longer" Fed story).
- Price *extends* through the first few minutes rather than reverting — higher highs on a hot-driven selloff, or the cool rally building rather than fading.
- The move is *several times* the expected move (September 2022's −4.32% was over three sigma).

**Fade the move when:**
- The surprise is *small*, *in-line*, or *ambiguous* (headline and core disagree).
- The first move is an obvious overshoot inside or barely beyond the expected move, with no fresh information to sustain it.
- Price *reverts* in the first few minutes — the spike gives back, the dollar fails to hold, yields round-trip.
- Positioning into the print was extreme one way, so the event-risk release alone drives a relief move regardless of the number.

The honest caveat: this fork is genuinely hard, and you will be wrong sometimes. That is exactly why **sizing for the expected move and writing a hard invalidation are non-negotiable.** The fade/trend decision determines your win rate; the sizing and invalidation determine whether being wrong is survivable. A good event trader is wrong plenty and still grinds out a positive expectancy, because the losses are pre-sized small and the wins (the big trend days) are large.

#### Worked example: sizing the fade versus the trend differently

You run two CPI playbooks with the same \$500 max risk per trade and the S&P expected move at 1.2%.

For the **fade**, you use a *tight* stop — just beyond the overshoot, say 0.5%. Position size = \$500 ÷ 0.005 = **\$100,000**. You can carry a large position because the stop is tight and you expect a quick reversion.

For the **trend**, you use a *wide* stop to survive whipsaw — say 2× the expected move, 2.4%. Position size = \$500 ÷ 0.024 = **\$20,833**. You carry a *smaller* position because the stop is wide, but you let the winner run far past 2.4% on a genuine trend day. Same \$500 of risk, opposite sizing — because the stop distance, not the conviction, sets the size. Risk a fixed dollar amount and let the trade's structure decide the size, not your excitement.

## The regime decides the sign of the whole map

Every reaction we have described assumes one thing: the market is in an **inflation-fearing regime**, where the dominant worry is that inflation forces the Fed to tighten. In that regime, hot inflation is unambiguously bad for risk assets — "good news for the economy is bad news for stocks," because anything that keeps the Fed hawkish hurts. That was the regime of 2022 and 2023, and it is why all three of our episodes line up so cleanly.

But the sign of the entire if-then map is regime-dependent. Consider a different world:

- **A deflation or growth-scare regime.** When the dominant fear is that the economy is sliding into recession or deflation (as in parts of 2008 and 2020), a *hot* inflation print can actually be *reassuring* — it says demand is alive, deflation is not taking hold, and the recession fear is overdone. In that regime, hot inflation can *rally* stocks, the exact opposite of 2022. The number is identical; the reaction flips because the market is asking a different question.
- **A "Fed-pivot" regime.** When the market is convinced the Fed is about to start cutting rates, the focus shifts from "will inflation force more hikes" to "is inflation falling fast enough to justify cuts." Here a cool print is rocket fuel (it accelerates the cut timeline) and a hot print is a sharp disappointment (it delays the cuts) — the magnitudes can be larger than the surprise alone would justify, because the print moves the *timing of the first cut*, not just one inflation reading.
- **A fiscal-dominance or currency-crisis regime.** When the worry is government debt or a collapsing currency rather than consumer inflation, CPI can take a back seat entirely, and the dollar and bonds may react to fiscal headlines more than to the print.

The practical rule: before you write your if-then map, write one sentence at the top of it — *"the regime is X, so the market cares about CPI because Y."* If you cannot finish that sentence, you do not yet understand which way the map points. The discipline of writing the map down never changes; the *signs in the boxes* do. (For the deeper treatment of how the market's focus rewires the same data into different reactions, see [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).)

## CPI is a US release, but it trades the whole world

US CPI is published by a US agency about US consumer prices, yet it moves Tokyo, London, and Ho Chi Minh City. The reason is the dollar's role as the world's reserve currency and US Treasuries' role as the global risk-free benchmark. When a US CPI surprise revises the expected Fed path, it revises the price of dollar liquidity *everywhere*, and that ripples through every market that funds, borrows, or invests in dollars.

Three transmission channels carry US CPI overseas:

1. **The dollar channel.** A hot US print strengthens the dollar, which pressures every other currency and tightens financial conditions abroad. Emerging-market central banks — including Vietnam's State Bank (SBV) — often have to respond to a strong dollar to defend their currencies, even though their own domestic inflation has not changed. (The SBV's toolkit and how it manages the dong is covered in [Vietnam monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling).)
2. **The rates channel.** US Treasury yields are the anchor for global bond pricing. A 28-basis-point move in the US 10-year, like the one on November 10, 2022, drags global yields with it, repricing mortgages, corporate borrowing, and the discount rate on every equity market on Earth.
3. **The risk-on/risk-off channel.** When a US CPI print triggers a global risk-off, foreign investors pull money out of riskier markets and back into dollar safety. For a market like Vietnam's, that shows up as **foreign net selling** of stocks, which pressures the VN-Index regardless of Vietnamese fundamentals. The 2022–2023 dollar surge, partly fed by hot US inflation prints, coincided with heavy foreign outflows from Asian equity markets. (How foreign flows move the index is covered in [foreign flows, ETFs, and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).)

So even if you trade only Vietnamese stocks or only your local market, a US CPI morning is your morning too. The cleanest expression of a US-CPI view is usually in US instruments (the S&P, the dollar, Treasuries) where the surprise hits first and most mechanically — but the *consequence* washes through to every market within hours. (For the full cross-market mechanism, see [cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).)

## How it reacted: real episodes

Theory is cheap. Here are three real, dated CPI days with cross-asset numbers — the same report producing wildly different outcomes because the surprise sign and the regime decided everything.

### September 13, 2022 — the hot print and the rout

The August CPI report landed at 8.3% year-over-year against an 8.1% consensus — a two-tenths *hot* surprise. It was a small miss in absolute terms, but it landed in the middle of the most aggressive Fed tightening cycle in forty years, when the market's deepest fear was "the Fed is not done." The reaction was a textbook risk-off cascade:

- **S&P 500: −4.32%** — the worst CPI day for stocks in over two years.
- **Nasdaq: −5.16%** — rate-sensitive tech led the decline, as long-duration growth always does when rates are expected to rise.
- **Bitcoin: ≈ −9.4%** — the high-beta liquidity asset got hit roughly twice as hard as the S&P.
- **US dollar index: +1.4%** — the lone winner, on the widened rate differential.
- **Gold: −0.4%** — pressured by rising real yields.

Everything moved in the coordinated direction the cross-asset map predicts for a hot print. The dollar up, everything risky down, with beta determining magnitude: dollar least, then gold, then S&P, then Nasdaq, then crypto most. What made this print so instructive is that the trade and the regime aligned perfectly. Going in, the market had been *hoping* inflation had peaked and had rallied for several days on that hope. The hot surprise did not just fail to confirm the hope; it shattered it. So the move was not merely the mechanical reaction to a two-tenths miss — it was the unwinding of a multi-day "peak inflation" rally that had gotten ahead of itself. That is why a small surprise produced a three-sigma day: it repriced a *narrative*, not just a data point. The knee-jerk down and the trend down were one and the same, so a trader who shorted the open-of-day reaction and held into the close was paid the full move.

### November 10, 2022 — the cool print and the rip

Eight weeks later, the October report printed 7.7% against a 7.9% consensus — a two-tenths *cool* surprise. Inflation was still enormous by any historical standard, but it came in below what was priced, and the market read it as the first real evidence that inflation had peaked and the Fed could slow down. The reaction was the mirror image:

- **S&P 500: +5.54%** — one of the best days of the decade.
- **Nasdaq: +7.35%** — tech, the asset that fell hardest in September, rallied hardest now.
- **10-year Treasury yield: −28 basis points** — a colossal one-day collapse in yields, the source of the whole rally.
- **US dollar index: −2.1%** — the dollar sold off hard as the rate-differential trade unwound.
- **Bitcoin: ≈ +10%** — the high-beta amplifier, up roughly double the S&P.

Same report, same agency, two months apart, and a near-perfect mirror — because the surprise sign flipped and the regime (good-news-is-good-for-risk on a cool print) stayed the same.

### November 14, 2023 — the cool print and the small-cap surge

A year later, the October 2023 report printed cool again, and the rate outlook eased on hopes the Fed's hiking cycle was finished. The reaction confirmed the cross-asset pattern with a small-cap twist:

- **S&P 500: +1.91%** — a solid risk-on day.
- **Russell 2000: +5.44%** — small caps, the most rate- and debt-sensitive equities, led by nearly triple the S&P, because an easing rate outlook helps the most leveraged, domestic companies most.
- **10-year Treasury yield: −19 basis points** — yields fell again, the now-familiar source of the equity rally.

The chart below puts the S&P's same-day move across all three days on one axis. One red bar (the hot September 2022 print) and two green bars (the cool 2022 and 2023 prints) — same report, opposite outcomes, the surprise sign deciding the color of the bar every time.

![Bar chart of S&P 500 same-day move across three CPI days with one red hot day and two green cool days](/imgs/blogs/trading-the-cpi-release-cross-asset-playbook-7.png)

One more piece of context worth holding: these reactions were all in the 2022–2023 *inflation regime*, where the market's overriding fear was the Fed, so "hot inflation = bad for risk" held cleanly. In a different regime — say, a deflation scare or a growth scare — a hot inflation print could *rally* stocks (because it eases recession fear). Always ask which regime you are in and why the market cares about *this* number *now*. The if-then map is regime-dependent; the discipline of writing it down is not.

### What the three episodes teach together

Lay the three days side by side and a few durable lessons fall out. First, **the surprise sign was a perfect predictor of direction in all three cases** — hot meant down, cool meant up, every time, with no exceptions. Second, **magnitude scaled with how much the print confirmed or broke the prevailing narrative**, not with the raw size of the surprise; all three surprises were roughly two-tenths, yet the S&P moves ranged from +1.91% to +5.54% to −4.32%. Third, **beta ordering held every time**: crypto moved most, then the Nasdaq or small caps, then the broad S&P, then gold, then the dollar in the opposite direction. That ordering is the single most reliable feature of a CPI reaction, and it is what lets you choose your instrument deliberately rather than just trading the index out of habit.

The November 2022 and November 2023 episodes also show the small-cap and crypto "amplifier" effect from two angles: in 2022 the amplifier was crypto (+10% versus the S&P's +5.54%); in 2023 it was the Russell 2000 (+5.44% versus the S&P's +1.91%). Whenever the rate outlook eases, the most rate-sensitive, most leveraged corners of the market — small caps, long-duration tech, crypto — give you the most move per dollar of view. The same logic runs in reverse on a hot print, which is exactly why those same corners are the most dangerous to hold un-hedged into a release you expect could surprise hot.

## Common misconceptions

**"High inflation means stocks go down."** No — *hot surprises* in an inflation-fearing regime move stocks down. The *level* of inflation barely matters on the day. November 10, 2022 had 7.7% inflation — historically enormous — and stocks rose **+5.54%**, because the surprise was cool. The September 13, 2022 print showed inflation *falling* from the prior month and stocks fell **−4.32%**, because the surprise was hot. Trade the surprise, not the level.

**"I should buy or sell the second the number drops."** No — the first second belongs to co-located algorithms parsing the machine-readable feed in microseconds. You cannot beat them to the headline, spreads blow out, and liquidity vanishes. On the September 2022 print the knee-jerk and the trend happened to align, but on many prints the first move reverses within a minute when the core read disagrees with the headline. Your edge is the second move, not the first.

**"Headline CPI is the number that matters."** No — the Fed targets *core*, and increasingly *supercore* (services ex-housing). A cool headline with a hot core can spike the market up and then reverse it red within minutes. The algos fire on the headline; the smart money trades the internals. If you only watch the headline, you will get caught on the wrong side of the second move.

**"CPI only matters for bonds and the dollar."** No — it moves every asset, and often crypto moves *most*. On the cool November 2022 print Bitcoin rose roughly **+10%** versus the S&P's +5.54%, and on the hot September print it fell roughly **−9.4%** versus the S&P's −4.32%. Crypto is the highest-beta expression of the CPI macro signal, not an asset that ignores it.

**"A small surprise means a small move."** Not necessarily — a two-tenths surprise produced a **−4.32%** S&P day in September 2022 and a **+5.54%** day in November 2022, both far beyond the roughly ±1.2% expected move. When a surprise *confirms* a fragile regime narrative ("the Fed is not done" or "inflation has peaked"), the reaction can be several times the surprise's mechanical size, because it shifts the whole probability distribution of the rate path, not just one data point.

## The playbook: how to trade the CPI release

Here is the full if-then map, the thing you should have written before 8:30. The numbers are the typical direction and magnitude in an inflation-fearing regime; the invalidations are what tell you the knee-jerk was wrong.

| Scenario | Stocks (S&P) | Crypto (BTC) | US dollar (DXY) | Gold | Bonds (yields) | The trade | Invalidation |
|---|---|---|---|---|---|---|---|
| **Hot** (actual > consensus) | Down (tech leads, e.g. −4.32% / Nasdaq −5.16%) | Down hard (≈ −9.4%) | Up (≈ +1.4%) | Down (≈ −0.4%) | Up (2y leads, +18bp typical) | Short risk / long USD; trend if surprise is large and confirms regime | Hot print closes green; USD falls; yields fall — the headline was already priced or core was soft |
| **In-line** (actual ≈ consensus) | Spike both ways then revert | Spike then revert | Spike then revert | Spike then revert | Spike then revert | Fade the overshoot, small size, tight stop; expect a vol crush | The "spike" keeps extending with conviction — it was not an overshoot, stand aside |
| **Cool** (actual < consensus) | Up (small caps lead, e.g. +5.54% / Russell +5.44%) | Up hard (≈ +10%) | Down (≈ −2.1%) | Up | Down (10y leads, −19 to −28bp) | Long risk / short USD; trend if surprise is large and confirms peak-inflation narrative | Cool print closes red; USD rallies; yields rise — core was hot under a cool headline |

The sizing and risk discipline that wraps the table:

1. **Size to the expected move.** Pull the straddle-implied move (≈ ±1.2% on the S&P in a normal regime), pick a fixed dollar risk (e.g. \$500), and set position = risk ÷ stop distance. Fades get tight stops and larger size; trends get wide stops and smaller size, then you let the winner run.
2. **Never trade the first second.** Let the knee-jerk and the liquidity vacuum pass. Engage on the second move, once the core read is in and spreads have normalized.
3. **Watch core, not just headline.** The reversal trade lives in the gap between a cool headline and a hot core (or vice versa). Supercore is the Fed's tell.
4. **Pick the cleanest expression.** The dollar moves first and cleanest; crypto and the Nasdaq give the most beta; bonds are the source. Choose the instrument that gives the most move per dollar of risk for your specific view.
5. **Write the invalidation before you enter.** The level that flips your thesis (a hot print closing green, the dollar failing to rally, yields disbelieving the headline) is your exit. Pre-commit to it so the 8:30 adrenaline cannot talk you out of it.
6. **Respect the regime.** The entire table assumes an inflation-fearing regime where hot is bad for risk. In a growth-scare or deflation regime, the signs can flip. Re-derive the map for the regime you are actually in.

The whole playbook reduces to one sentence you already know: **trade the reaction, not the number.** The number is just the trigger that tells you which branch of the map you pre-wrote actually fired.

## Further reading & cross-links

- [CPI: the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) — the deeper background on what CPI measures, how it is constructed, and why it dominates the macro calendar.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — the full mechanism of how a single data point propagates through bonds, the dollar, gold, stocks and crypto.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — why the same CPI print can be bad for stocks in one regime and good in another.
- [Anatomy of a news reaction: spike, fade, trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) — the universal shape of every event reaction and how to trade each phase.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — how the options market quotes the move you should size to.
- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — where CPI sits among the other market-moving releases and how to plan a month of event risk.
- [Inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — the policy layer: how the Fed turns an inflation surprise into a rate decision.

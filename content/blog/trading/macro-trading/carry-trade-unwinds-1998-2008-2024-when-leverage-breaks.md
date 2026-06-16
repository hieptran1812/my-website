---
title: "Carry-Trade Unwinds: 1998, 2008, 2024, and How Leverage Breaks the System"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Carry-trade unwinds are the recurring way leverage detonates the financial system: traders borrow a low-yield currency, lever up, earn the steady carry for years, then a shock forces a violent all-at-once exit that spikes volatility and cascades across markets."
tags: ["macro", "carry-trade", "leverage", "deleveraging", "volatility", "yen", "ltcm", "financial-crisis", "positioning", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A carry trade borrows a low-yield currency, levers up, and earns the interest-rate spread for years — until a shock forces a violent, all-at-once unwind that spikes volatility and cascades across every market. 1998, 2008, and 2024 are the same movie: quiet income that turns into catastrophe in days.
>
> - **The setup always rhymes.** A crowded, levered carry book pays a steady spread in calm markets; the calm itself lures in more size and more leverage, which is exactly what makes the eventual exit so violent.
> - **The unwind is a self-reinforcing loop.** Shock to margin call to forced selling to a volatility spike to *more* margin calls. Nobody decides to crash the market; the leverage decides for them.
> - **Leverage is the multiplier.** A 5% currency move is a scratch on cash and a 50% wipeout on a 10x-levered book. The move didn't get bigger — the leverage did.
> - **The one number to remember:** on 5 August 2024 the VIX spiked to **65.7** intraday as the yen-carry trade unwound — the same volatility signature as LTCM's 45.7 in 1998. Carry pays pennies for years and takes them all back in a single session.

On the morning of Monday, 5 August 2024, the Japanese stock market fell 12.4% in one session — its worst day since the 1987 crash. The selling jumped time zones: European futures gapped lower, US stocks opened deep in the red, and the VIX — Wall Street's "fear gauge," the index that measures how much volatility traders expect — briefly spiked above 65. To put that in scale, the VIX sits near 13 to 17 in calm markets and only reaches the 60s during full-blown crises. Yet nothing had actually *broken*. No bank had failed. No country had defaulted. No war had started.

The trigger was almost comically small. A few days earlier, the Bank of Japan had raised its policy interest rate by a quarter of a percentage point — a tiny move — and a US jobs report had come in soft. Two minor pieces of news about *interest rates*. And the global market convulsed.

Why? Because for three years, one of the most crowded trades on the planet had been to borrow cheap Japanese yen and use the proceeds to buy higher-yielding assets everywhere else — US dollars at over 5%, Mexican pesos, US tech stocks, you name it. That trade is the **carry trade**, and it had quietly printed money for years. When the rate gap that powered it started to close — even slightly — every leveraged trader tried to exit the same position through the same door at the same moment. The exit was too small for the crowd. The yen ripped higher, the assets bought with borrowed yen were dumped, volatility exploded, and the whole thing fed on itself.

![Carry crowded shock deleveraging vol spike cascade flow](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-1.png)

The figure above is the entire story on one page, and the rest of this post earns the right to read it. A crowded, levered carry book earns its quiet spread on the far left. A shock arrives. Margin calls fire. Forced selling begins. Volatility spikes. And lower prices trigger *more* margin calls — the loop closes on itself and cascades across markets. Hold this shape in your head, because we are going to watch it run three times: in **1998** (LTCM and the Russian ruble), in **2008** (the yen-carry unwind inside the global financial crisis), and in **2024** (the August yen-carry unwind). Different cast, different trigger, identical machinery.

This is the finale of the *Macro for Traders* series, and it is the master case study of how leverage breaks the system. Everything we have built — what money is, how liquidity moves, how policy works, how regimes turn, how policy transmits to each asset class — comes due here, in the crisis where a quiet income trade detonates the whole market.

## Foundations: the carry trade, leverage, the crowded trade, and the unwind cascade

Before we can watch leverage break anything, we have to build four ideas from absolute zero: the carry trade itself, leverage, what makes a trade "crowded," and the mechanical loop of an unwind. If you internalize these four, every crisis in this post will look like the same gear turning.

### The carry trade: borrow cheap, hold expensive, pocket the difference

Start with the everyday version. Imagine your bank lets you borrow money at 1% a year and, across the street, another bank pays you 5% a year on a savings account. You could borrow \$100,000 at 1%, deposit it at 5%, and pocket the 4% difference — \$4,000 a year — without putting up any of your own money for the spread. You did nothing but stand in the middle of two interest rates and collect the gap.

That is the carry trade. In its purest form, **carry is the income you earn for holding a position, separate from any price change.** The classic macro version uses currencies: you borrow a currency with a low interest rate (the *funding currency*), convert the proceeds into a currency with a high interest rate (the *target currency*), and earn the difference between the two rates for as long as you hold.

For most of the last 30 years, the world's favorite funding currency has been the Japanese yen, because Japan held interest rates near zero for decades. The favorite target currencies have been whatever paid the most at the time — Australian and New Zealand dollars in the 2000s, US dollars in the 2020s. "The yen carry trade" is shorthand for: **borrow yen for almost nothing, hold higher-yielding assets, earn the spread.**

Two terms you will need, defined once:

- **Funding currency** — the low-yield currency you borrow (the yen, near 0.1%). You owe interest on it.
- **Target currency / asset** — the higher-yield thing you buy with the borrowed money (the US dollar at 5.3%, or any higher-returning asset). It pays you interest.

The carry is the target rate minus the funding rate. Borrow yen at 0.1%, hold dollars at 5.3%, and the carry is about 5.2% a year. We will put exact dollars on that in a moment.

There is a deep question lurking here that you should hold onto, because it explains why the carry trade exists at all. In theory, the carry trade *shouldn't* make money. Economic theory says that if the dollar pays 5.2% more than the yen, then over time the dollar should *weaken* against the yen by exactly 5.2% — so the extra interest you earn is exactly cancelled by the currency loss, and you net zero. This idea is called **uncovered interest parity**, and it is the textbook prediction. The thing is, it reliably *fails* over the horizons traders care about. The high-yielding currency tends to stay strong or even strengthen for long stretches, so the carry trader earns the rate spread *and* often a currency gain on top. That persistent failure of the textbook is not a footnote — it *is* the carry trade. You are being paid a premium for bearing a risk the textbook ignores: the risk that, occasionally and unpredictably, parity reasserts itself all at once in a violent unwind. The carry premium and the crash risk are the same coin. You earn the premium in the calm years; you pay it back, with interest, in the crash.

### Leverage: the same money, controlling a much bigger position

The second idea is **leverage** — using borrowed money so that a small amount of your own capital controls a much larger position. If you put up \$100,000 of your own money and borrow another \$900,000, you control a \$1,000,000 position with 10x leverage: your \$100,000 of equity is one-tenth of the position you control.

Leverage is the amplifier at the heart of every story in this post. It does two things, symmetrically:

- It **multiplies the carry.** If you earn 5.2% carry on a \$1,000,000 position that only took \$100,000 of your own money, your return on *your money* is 52%, not 5.2%. Leverage turns a modest spread into a spectacular yield on equity.
- It **multiplies the loss, identically.** A 5% adverse currency move on that \$1,000,000 position is a \$50,000 loss — half your \$100,000 equity gone. The position moved 5%; your capital moved 50%. Leverage is exactly as generous on the way up as it is merciless on the way down.

This symmetry is the single most important fact in this article. The carry traders are not stupid or reckless; they are doing arithmetic. The arithmetic just happens to be a coin that pays small on most flips and occasionally takes everything.

### The crowded trade: when everyone is on the same side

The third idea is what makes a quiet income trade dangerous: **crowding.** A trade is crowded when a large fraction of market participants hold the same position. The carry trade is structurally prone to crowding because it works — for years it pays a steady, attractive return with low day-to-day volatility, which is precisely the profile that attracts more and more capital. Pension funds, hedge funds, retail traders in Japan (the famous "Mrs. Watanabe"), and bank trading desks all pile into the same direction: short the yen, long higher-yielding assets.

Here is the trap. While everyone is *holding*, crowding looks like nothing — the trade is calm and profitable. Crowding only matters at the *exit.* When a shock makes everyone want out at once, the position that took years to build has to be unwound in days, and there are no buyers on the other side because everyone who would have bought is already long the same trade and trying to sell. A crowded trade has a narrow exit. The more crowded, the narrower.

We will measure crowding later with positioning data — the **Commitments of Traders (COT)** report, which shows how speculators are net-positioned in futures — but the mechanism is simple: when a trade becomes consensus, it becomes fragile, because the marginal buyer is exhausted and only sellers remain.

### The unwind cascade: the self-reinforcing deleveraging loop

The fourth and final idea is the mechanism — the loop that turns a small shock into a market-wide cascade. It runs in five steps, and the crucial feature is that the output of each step feeds the input of the next:

1. **A shock arrives.** The rate gap starts to close (the funding currency's central bank hikes, or the target currency's central bank cuts), or a default lands, or any event that makes the carry trade look less safe.
2. **The position moves against the crowd.** The funding currency strengthens — the yen rallies — turning the carry book red.
3. **Margin calls fire.** Because the positions are levered, lenders demand more cash or collateral as losses mount. Traders who cannot post more are *forced* to sell.
4. **Forced selling spikes volatility.** Everyone sells the same assets at once. Prices gap, liquidity vanishes, bid-ask spreads widen, and the VIX explodes.
5. **The spike triggers more selling.** Higher volatility raises the margin requirements on *everyone's* positions (risk models demand more collateral when volatility rises), and lower prices mean more losses, which means more margin calls — which sends the loop back to step 3.

That feedback from step 5 to step 3 is the entire reason carry unwinds are violent. It is not a one-time sell-off; it is a *self-reinforcing* deleveraging spiral. Nobody at any desk decides to crash the market. The leverage decides for them: once you are a forced seller, you sell into a market full of other forced sellers, and the price you get makes the next seller's margin call worse.

This loop has a name in the literature — a **liquidity spiral** or a **deleveraging cascade** — but you do not need the jargon. You need the shape: *quiet income, then a shock, then a loop that eats itself.* Now let us watch it run.

## The anatomy of an unwind: why it goes from quiet to catastrophe in days

Let us slow down the loop and look at the timing, because the timing is the whole point. A carry trade has a deeply asymmetric profile in time as well as in money.

On the way up, time is your friend. The carry accrues daily — a little bit of interest-rate spread every single day — and the funding currency tends to drift weaker slowly, which *adds* to your return (you borrowed yen, the yen is falling, so your debt shrinks in your home currency). Years of this build a track record, a Sharpe ratio (return per unit of risk) that looks fantastic, and a position that has grown comfortable. The longer it works, the safer it feels, and the more leverage and size the market piles on.

On the way down, time is your enemy, and the clock runs in days, not years. When the shock hits, the funding currency does not drift — it *gaps*. The yen can move 5% in a day when a carry unwind starts, because the same flows that pushed it down for years all reverse at once. And here is the asymmetry that defines the trade: a 5% gap in the funding currency, on a levered book, can erase the carry you accumulated over *years* in a single session.

This is the famous description of the carry trade: **picking up pennies in front of a steamroller.** You collect a steady stream of small gains (the pennies — the daily carry), and the risk is not a series of small losses but one enormous loss that arrives all at once (the steamroller — the unwind). The return distribution is not a bell curve. It is a long string of small positive days punctuated, rarely and unpredictably, by a catastrophic negative one. Standard risk math — which assumes returns are roughly normally distributed — badly understates the danger, because the danger lives entirely in the rare event the average doesn't see.

Two structural features make the unwind so fast once it starts:

- **Leverage compresses the timeline.** An unlevered investor who is down 5% can wait it out. A 10x-levered trader who is down 5% has lost half their equity and is getting a margin call *today.* Leverage converts a slow drawdown into an immediate, forced, all-at-once exit.
- **Crowding removes the buyers.** In a normal sell-off, lower prices attract bargain hunters who stabilize the market. In a crowded-trade unwind, the natural buyers are the very people being forced to sell — they are already maxed long and now liquidating. The order book is one-sided. Prices fall until they reach someone who was *not* in the trade, and that price can be a long way down.

Put leverage and crowding together and you get the signature of every episode in this post: a quiet, profitable trade that, when it breaks, breaks completely and almost instantly. Now the three case studies.

One more piece of the anatomy deserves a name, because it explains why these unwinds are so hard to stop once they start: the **margin spiral.** When you hold a levered position, your broker requires you to post collateral — *margin* — sized to the risk of the position. The amount of margin required is not fixed; it rises with volatility, because the broker's own risk model says a more volatile position can lose more, so it demands more cushion. Now watch the loop tighten. The unwind begins, prices fall, volatility rises. Higher volatility means the broker raises the margin requirement on your position — so even if you had posted exactly enough collateral yesterday, today you are short, and you get a margin call demanding *more* cash. To raise that cash, you sell. Your selling pushes prices lower and volatility higher, which raises the margin requirement again, on you and on everyone else. The requirement to hold the position grows at the exact moment the position is losing money — the worst possible time. This is why a carry unwind, once triggered, does not gently fade: the very act of deleveraging raises the price of staying levered, forcing more deleveraging. The system demands the most collateral precisely when collateral is hardest to come by, which is the financial equivalent of a fire that burns hotter the more water you throw on it.

## 1998: LTCM and the ruble — genius plus leverage plus a shock

The first time the modern world watched leverage detonate the system, the firm at the center was run by literal geniuses. **Long-Term Capital Management (LTCM)** was a hedge fund founded in 1994 by John Meriwether, the legendary bond trader from Salomon Brothers, with two Nobel laureates on its board — Myron Scholes and Robert Merton, who had won the prize for the option-pricing math that underpins modern finance. If anyone could engineer a safe, high-return strategy, it was this group.

LTCM's core strategy was not the currency carry trade specifically, but it was the same *idea* wearing different clothes: **find tiny, reliable spreads and lever them enormously.** LTCM specialized in *convergence trades* — buying a slightly cheap bond and short-selling a nearly identical slightly expensive one, betting that the small price gap between them would close. The gap on any single trade was tiny, a few basis points (hundredths of a percent), so to make those tiny spreads into real money, LTCM applied massive leverage. At its peak the fund had roughly \$5 billion of equity controlling positions worth over \$100 billion — leverage around 25x — and far more than that in derivative exposure on top.

This is the carry-trade structure exactly: collect a small, steady spread (the pennies), levered up enormously, on a position that quietly works for years. The genius was real; so was the leverage; so was the assumption that the small spreads would behave the way history said they would.

Then the shock. In August 1998, Russia did something the models treated as nearly impossible: it **defaulted on its domestic ruble-denominated debt and devalued the ruble.** A sovereign government defaulting on debt in its own currency — which it can, in principle, always print to repay — broke the assumption baked into countless trades that "safe" spreads stay correlated and orderly. Panic spread. Investors everywhere fled to the safest, most liquid assets (a *flight to quality*), and the very spreads LTCM bet would *converge* instead **diverged** violently — the cheap bonds got cheaper, the expensive ones got more expensive, exactly the wrong way.

Now run the cascade. LTCM's levered positions moved against it. Losses on 25x leverage were catastrophic — the fund lost roughly half its capital in weeks and was down about 90% by the time it was rescued. Margin calls forced selling. The selling moved prices further against the same positions held by *other* levered players, who then faced their own margin calls. The VIX, which had sat in the teens, spiked to **45.7** — the original modern panic reading.

![VIX crisis peaks bar chart LTCM GFC COVID yen carry](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-4.png)

That 45.7 is the leftmost red bar in the chart above, and it is the smallest of the four crisis peaks — yet it was alarming enough that the Federal Reserve Bank of New York convened fourteen major banks to organize a roughly \$3.6 billion rescue of a single hedge fund, because LTCM's positions were so large and so entangled with the rest of Wall Street that a disorderly liquidation threatened the whole system. One fund, picking up pennies with enormous leverage, nearly broke the market when the steamroller finally arrived. (For the full LTCM story, see the cross-linked deep dive at the end.)

The detail that makes LTCM the perfect opening case is *why* the geniuses got it so wrong, because the error is the same error every carry trader makes. LTCM's models were built on historical data, and in the historical data, the spreads they traded were stable and mean-reverting — they always converged eventually. The models therefore concluded the positions were low-risk and could safely be levered 25x. What the models could not capture was the *correlation* that appears in a crisis: in normal times, LTCM's dozens of trades were diversified and independent, so a loss on one was offset by a gain on another, and the whole book looked safe. But in the August 1998 panic, every position moved against them *at the same time* — the diversification evaporated exactly when it was needed, because in a flight to quality, all risky spreads widen together. The book that looked diversified at 25x leverage was, in the crisis, a single concentrated bet on "calm continues," levered 25x. This is misconception two, made flesh, three years before anyone named it: diversification across positions that share a hidden common factor is not diversification. LTCM did not have a hundred independent bets; it had one bet — that markets stay orderly — wearing a hundred costumes, and when the music stopped, all hundred costumes came off at once.

The other detail worth carrying forward: LTCM was *right*, eventually. Most of its convergence trades would have converged if the fund could have held them. The spreads that blew out in 1998 did narrow again in 1999. But being right eventually is worthless when you are levered 25x, because leverage does not let you wait. A 25x-levered position that moves 4% against you wipes out your equity, and once your equity is gone, you are liquidated at the worst possible price — you never get to collect on being right. This is the cruelest feature of leverage: it converts a temporary, survivable drawdown into a permanent, fatal one. Solvency and timing become inseparable. The market can stay irrational longer than a levered fund can stay solvent, and "longer" can be as little as a few weeks.

#### Worked example: the carry income that lured the leverage in

Before the steamroller, there were the pennies — and the pennies were genuinely attractive, which is the whole reason anyone takes the trade. Let us price the income on the modern yen-dollar version, using real rates, so you feel why a trader piles in.

You run the classic 2023-2024 yen carry. You borrow yen at Japan's policy rate of about **0.1%** and convert the proceeds into US dollars earning the Fed funds upper bound of **5.3%** (the peak policy rate in the 2023 cycle from the Fed funds data). You put on a **\$1,000,000** position.

The annual carry is the rate spread times the position:

```
carry spread  = 5.3%  - 0.1%  = 5.2% per year
annual carry  = 5.2% x $1,000,000 = $52,000
```

So just for holding the position — before any currency move — you collect **\$52,000 a year**, about **\$4,333 a month**, or roughly **\$142 every single day** the position is open. That income lands like clockwork. After a year of this, you have \$52,000 of realized carry and a chart that looks like a steady upward staircase with almost no volatility. *That* is the seduction: a 5.2% yield that arrives daily and barely wiggles is exactly the profile that pulls in more capital and more leverage. The steady \$142-a-day drip is the bait; the steamroller is the part you don't see on the calm chart.

## 2008: the yen-carry unwind inside the global financial crisis

By the mid-2000s, the yen carry trade had become a defining feature of global markets. Japan's interest rates were pinned near zero while the US, Australia, New Zealand, and emerging markets paid much more. So the world borrowed yen — cheaply, in enormous size — and bought everything that yielded more. Estimates of the total yen-carry position ran into the hundreds of billions to over a trillion dollars. It was, by any measure, one of the most crowded trades on Earth, and it had worked for years: the yen drifted weaker, the targets paid their spread, and the carry compounded.

Then 2008 arrived. The trigger here was not a single sovereign default but the unraveling of the US subprime mortgage market and the seizure of global funding markets — the worst financial crisis since the Great Depression. As banks failed and credit froze, every leveraged position in the world came under pressure at once. And the yen carry was a leveraged position, sitting right in the blast radius.

Run the cascade one more time. As risk assets fell, carry traders faced losses on their targets *and* margin calls on their leverage. To raise cash and cut risk, they had to unwind the carry — which meant *buying back yen* to repay the yen they had borrowed. With everyone buying yen at once, the yen did not drift; it **rocketed.** USD/JPY (yen per dollar; a falling number means a strengthening yen) collapsed from around 110 in mid-2007 toward the mid-80s and eventually below 80 — one of the most violent currency moves in a major pair in modern history. The strengthening yen turned every carry book deeply red, which forced more unwinding, which drove the yen higher still. The loop again.

This is the part that surprises people: the yen, the funding currency of the world's biggest carry trade, *strengthened dramatically during the worst risk-off event in living memory.* It seems backwards — Japan was not the cause of the crisis — but it is the carry mechanism showing its face. In a crisis, the funding currency strengthens because the crisis *is* the unwind: everyone scrambling to buy back what they borrowed. The yen behaves like a safe haven not because traders suddenly love Japan but because the world is short yen and the short is being covered.

The VIX in 2008 reached **80.9** (the second red-context bar in the chart above) — the highest reading until COVID. The carry unwind was not the whole crisis; the GFC had many engines. But the yen-carry unwind was one of them, and it transmitted the panic from the funding markets into the currency market and back, amplifying the deleveraging spiral. (For how the same crisis broke the *liquidity* plumbing and forced the policy response, see the cross-linked liquidity and policy posts.)

It is worth dwelling on the *direction* of the yen in 2008, because it is the single most counterintuitive fact in this whole subject and getting it right is what separates a trader who understands the carry from one who memorizes it. A beginner's instinct says a currency strengthens when its economy is strong and weakens when its economy is weak. By that logic the yen should have *fallen* in 2008 — Japan was a slow-growth, near-zero-rate economy heading into a global recession. Instead the yen had one of its strongest years on record, *strengthening* roughly 20% against the dollar during the worst of the crisis. The reason is purely mechanical and has nothing to do with Japan's economy: the world was *short* yen, in enormous size, because the world had borrowed yen to fund the carry. When the crisis forced everyone to close their positions, "closing" a yen-funded position means *buying yen back* to repay the loan. A short being covered en masse is a buy order, and a buy order of that scale, with everyone hitting it at once, sends the price vertical. The yen strengthened in 2008 *because* it was the funding currency of a collapsing trade, not despite the recession. Once you see this, the yen's behavior in every risk-off event clicks into place: the funding currency of the carry is, by construction, a crisis safe haven, because a crisis *is* the unwind, and the unwind *is* a wall of buying in the funding currency. The yen is not a safe haven because traders trust Japan; it is a safe haven because the market is structurally short it and the short gets covered in a panic.

#### Worked example: the unwind loss that erases years of carry

This is the steamroller, in dollars. Take the 2024 version of the levered yen carry and run the August unwind through it, so you can see one currency move erase years of accumulated income.

You hold a **\$1,000,000** yen carry, financed by borrowing yen — meaning you are effectively *short the yen / long the dollar* in that amount. The yen-per-dollar rate stood at its July 2024 intraday peak of **161.9** (the real high). Over the August unwind, USD/JPY fell from about **161.9 toward 145** — the yen strengthened, the number dropped — a move of roughly:

```
fx move  = (161.9 - 145) / 161.9 = 16.9 / 161.9 = 10.4%
```

Because you were short the yen on \$1,000,000, a 10.4% adverse move is a loss of:

```
unwind loss = 10.4% x $1,000,000 = $104,000
```

Now compare that to the carry. From the previous example, the position earned **\$52,000 a year** in carry. So this single currency move — over a matter of days — wiped out **two full years** of carry income (\$104,000 / \$52,000 = 2.0 years), and that is *before* any leverage. You collected pennies daily for two years, and the steamroller took all of it back in about a week. The carry trade's entire risk lives in this one number: the FX move on the unwind, which dwarfs the income that lured you in.

![USD/JPY year close 2019 to 2025 with July 2024 peak](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-2.png)

The chart traces exactly this arc. USD/JPY bottomed near 103 in 2020 (a strong yen), then the number climbed for years as the US-Japan rate gap blew out — that long upward staircase is the carry *building*, the yen weakening as the world borrowed it and sold it. The red star marks the July 2024 intraday peak at 161.9: the top of the trade, the point of maximum crowding and maximum leverage. And the amber arrow marks the August unwind back toward 145 — the violent reversal that priced in the two years of carry loss we just computed. The build took years; the break took days. That shape is the carry trade.

## 2024: the August yen-carry unwind — a BOJ hike, a Fed pivot, and a 65 VIX

Now the most recent run of the movie, and the one with the cleanest causal chain. By the summer of 2024, the yen carry had rebuilt to an extreme. The US-Japan interest-rate gap was enormous: the Fed had hiked to a peak of 5.50% in 2023 to fight inflation, while the BOJ had held its rate near zero. That gap — visible as the wide amber band in the figure below — was the fattest carry in a generation, and the world had piled in. USD/JPY had marched from 103 in 2020 to a 161.9 intraday peak in July 2024, a ~57% move, almost entirely on the back of that rate differential and the crowded trade exploiting it.

![Fed funds upper bound versus BOJ near zero rate gap](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-6.png)

The chart shows the fuel. The blue line steps up hard as the Fed hikes through 2022-2023 to a 5.50% peak, while the lavender line — the BOJ at roughly 0.1% — barely moves off the floor. The amber-shaded gap between them *is* the carry: borrow yen near 0.1%, hold dollars at 5.3%, earn ~5.2% a year, as the annotation says. The wider that band grew and the longer it stayed wide, the more capital and leverage the trade attracted — which is precisely what set up the violence of the exit.

Then two small things happened in late July and early August 2024, and both pushed the rate gap the wrong way:

- **The BOJ hiked.** On 31 July 2024, the Bank of Japan raised its policy rate by 0.25 percentage points — a tiny move in absolute terms, but a move in the *direction* that shrinks the carry. The funding currency's rate was now rising.
- **The Fed pivoted dovish.** A soft US jobs report on 2 August stoked fears the Fed had held rates too high for too long and would have to cut soon. The target currency's rate was now expected to *fall.* Both blades of the scissors were closing the gap at once.

The market did the arithmetic instantly. If the rate gap is shrinking, the carry is shrinking, and the years of yen weakness that funded everyone's gains could reverse. Every leveraged carry trader reached for the exit simultaneously — and the exit, as always, was too narrow for the crowd.

Run the cascade for the third time. The yen ripped higher (USD/JPY collapsed from 161.9 toward 145 in days). Carry books went red. Margin calls fired. Forced selling hit the assets bought with borrowed yen — including US tech stocks and the Japanese equity market itself. The Nikkei fell 12.4% on 5 August, its worst day since 1987. Volatility exploded: the VIX spiked to **65.7** intraday (the rightmost red bar in the VIX chart) — a crisis-level reading produced not by a bank failure or a war but purely by the mechanical unwind of a crowded, levered trade.

And then — the part that distinguishes 2024 from 1998 and 2008 — it was largely over within a week or two. Because nothing fundamental had actually broken (no insolvent banks, no frozen funding markets), once the forced sellers were flushed out, the market stabilized and risk assets recovered much of the loss within weeks. The 2024 episode was a *pure positioning* unwind: a crowded, levered trade detonating on its own crowding, with no underlying solvency crisis underneath. That makes it the cleanest specimen of the species — leverage breaking the system all by itself, with the real economy as a bystander.

That quick recovery is itself an important lesson, and a double-edged one. On one hand, it tells you that a *pure positioning* unwind — one with no solvency or liquidity crisis underneath — tends to be sharp but short, because once the forced sellers are gone, the assets are simply cheaper than they were and real buyers step back in. On the other hand, the speed of the recovery is exactly what makes the next carry trade so tempting: traders look at the August 2024 chart, see that everything bounced back within weeks, and conclude that carry unwinds are "buying opportunities" you can ride through. That conclusion is fatal *on leverage.* The unlevered investor who held through the 2024 unwind was fine; the 10x-levered trader who held was margin-called and liquidated at the bottom, days before the recovery, and never participated in the bounce. The recovery is real, but you only collect it if you survive the drawdown — and leverage is precisely the thing that determines whether you survive. A V-shaped recovery rewards the patient and unlevered and annihilates the impatient and levered, on the identical price path. Same chart, opposite outcomes, decided entirely by leverage.

It is worth pausing on *who* was in this trade, because the crowding was unusually broad. The 2024 yen carry was not only run by hedge funds and bank desks. It reached deep into retail: Japanese households — the archetypal "Mrs. Watanabe" — had for years borrowed cheap yen to buy higher-yielding foreign assets, and a wave of newer global retail traders had piled into the same idea through margin accounts and leveraged products, often without naming it "the carry trade" at all. When a trade gets that crowded — institutions and retail, professionals and tourists, all on the same side — the exit is not just narrow, it is mobbed, because the least sophisticated participants are also the quickest to panic and the first to be margin-called. Breadth of crowding is depth of fragility. The more *kinds* of people are in a trade, the more violently it unwinds, because the weakest hands fold first and start the cascade that forces the strong hands out too.

#### Worked example: the leverage magnifier that wiped the equity

The 2024 move was painful at 1x. At the leverage real traders used, it was an extinction event. Let us put the magnifier on it explicitly.

Two traders each commit **\$100,000** of their own money to the yen carry.

Trader A is *unlevered.* Her \$100,000 controls a \$100,000 position. When the yen rallies ~5% in the early days of the unwind, her loss is:

```
trader A loss = 5% x $100,000 = $5,000
equity left   = $100,000 - $5,000 = $95,000   (down 5%)
```

She is down 5% — a bad week, a scratch. She can wait it out, and indeed if she holds, much of it comes back within weeks.

Trader B runs **10x leverage.** His \$100,000 of equity controls a \$1,000,000 position (he borrowed \$900,000). The *same* 5% yen rally produces:

```
trader B loss = 5% x $1,000,000 = $50,000
equity left   = $100,000 - $50,000 = $50,000   (down 50%)
```

The identical market move that cost Trader A 5% costs Trader B **50% of his capital** — and triggers an immediate margin call demanding he post more cash or be liquidated. He cannot wait it out; he is a forced seller *today.* And if the yen rallies the full ~10% of the actual unwind, his loss is \$100,000 — his entire equity, wiped out. The market moved the same for both traders. Only the leverage was different, and the leverage is the entire difference between a scratch and a wipeout.

![Unlevered versus 10x levered position equity comparison](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-3.png)

The figure makes the magnifier visceral. On the left, the unlevered book: \$100,000 of your own money, a \$5,000 loss, \$95,000 of equity surviving — green, you keep trading. On the right, the 10x book: the same \$100,000 of equity now controls \$1,000,000, the same 5% move is a \$50,000 loss, and half the equity is gone in red — with a 10% move erasing it entirely. Same currency move, same starting capital, opposite outcome. Leverage didn't make the move bigger; it made *your share of it* fifty percentage points bigger.

## What the three episodes share: the same movie, three times

Lay 1998, 2008, and 2024 side by side and the structure is unmistakable. The cast changes — a hedge fund of geniuses, the global banking system, the entire macro-tourist crowd — and the trigger changes — a sovereign default, a mortgage collapse, a quarter-point central-bank hike. But the *skeleton* is identical, every time.

![Matrix comparing 1998 2008 2024 unwinds across five rows](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-5.png)

The matrix above reads the three episodes against the five-part anatomy, row by row:

- **The crowded levered bet.** 1998: LTCM's convergence trades at ~25x leverage. 2008: borrow ~0% yen and buy higher-yielding assets worldwide. 2024: borrow ~0.1% yen and lever risk across the globe. Same structure — a small, steady spread, levered enormously, held by a crowd.
- **The shock.** 1998: Russia defaults and devalues the ruble. 2008: subprime cracks and funding markets seize. 2024: the BOJ hikes, US jobs come in soft, the Fed pivot looms. Different events, identical *function* — each makes the "safe" trade suddenly unsafe.
- **The forced unwind.** 1998: spreads blow out and no one can exit. 2008 and 2024: the yen rips and everyone sells the carry at once. In all three, the exit is too narrow for the crowd, and selling begets selling.
- **The vol signature.** VIX to 45.7, then 80.9, then 65.7. The volatility spike is the *fingerprint* of a leverage unwind — it is what forced deleveraging looks like on a screen.
- **The damage.** LTCM down ~90% and rescued; the global system nearly failing and QE born; the Nikkei down 12.4% and years of carry erased. The scale differs, but every time, the losses dwarf the carry that lured the capital in.

Why do these rhyme so precisely? Because the carry trade is not a strategy that *sometimes* blows up — it is a structure that *must* eventually blow up, because the very thing that makes it attractive (steady, low-volatility income) is the thing that makes it crowd, and crowding is what makes the exit lethal. Calm breeds leverage; leverage breeds crowding; crowding guarantees that the next shock — and there is always a next shock — finds an over-levered, one-sided market with no buyers. The trade contains the seed of its own unwind. That is why it rhymes.

There is one more shared feature worth naming explicitly, because it is the part that links the carry unwind to *everything else* in the market: the volatility spike is not a side effect of the unwind — it is the *transmission mechanism* that turns a single crowded trade into a system-wide event. When the VIX jumps, it does not just measure fear; it *manufactures* forced selling, because risk models everywhere key their position limits off volatility. A trader running an unrelated strategy — long stocks, short bonds, anything — finds that when volatility doubles, the risk model that governs their book demands they cut position size to keep risk constant. So a vol spike born in the yen carry forces deleveraging in books that never touched the yen. That is how a quarter-point BOJ hike reaches a tech-stock portfolio in California: through the VIX, which is the wire that carries the panic from the originating trade to every levered book on Earth. The volatility spike is both the symptom *and* the contagion.

#### Worked example: the vol-spike signature across LTCM, 2008, and 2024

Volatility is the fingerprint of a leverage unwind, so let us put the three episodes on one scale and read what the numbers say. The VIX measures expected annualized volatility of the S&P 500 in index points; calm markets sit near 13 to 19, and only forced, system-wide deleveraging pushes it into the 40s, 60s, and 80s.

Take the long-run average VIX of about **19.5** as the baseline "normal." Now the three carry-style unwinds:

```
LTCM 1998:      VIX 45.7   = 2.3x normal   (the original modern panic)
GFC 2008:       VIX 80.9   = 4.1x normal   (yen carry unwind inside the crisis)
Yen carry 2024: VIX 65.7   = 3.4x normal   (a pure positioning unwind, no solvency crisis)
```

Read the 2024 number against the others, because it is the most telling. A VIX of **65.7** — *3.4 times* the normal level, deep crisis territory — was produced in 2024 with no bank failure, no default, no recession, no frozen funding market. Just the mechanical unwind of a crowded, levered trade. Compare that to 2008's 80.9, which accompanied the near-collapse of the entire global banking system. The 2024 spike reached roughly four-fifths of the GFC's intensity *on the strength of positioning alone.* The lesson in the arithmetic: a crowded levered carry can generate a near-GFC volatility event by itself, without any underlying economic catastrophe. The leverage is sufficient; you do not need a real crisis underneath to get a crisis-level vol print.

## Common misconceptions

A few beliefs about the carry trade are not just wrong but *dangerous*, because they are exactly the beliefs that get traders maximally exposed right before the steamroller. Let us correct three, each with a number.

### Misconception 1: "Carry is free money"

The most seductive error. The carry trade pays a steady, attractive return — \$52,000 a year on a \$1,000,000 dollar-yen book, ~5.2%, arriving daily with almost no volatility — which *looks* like free money for months or years on end. It is not free; it is *insurance you are selling.* You are being paid the carry to bear the risk of the unwind, exactly the way an insurance company collects premiums to bear the risk of a disaster. Most years, no disaster comes and the premium is pure profit. The year the disaster comes, a single ~10% currency move erases two years of carry (the \$104,000 unwind loss versus \$52,000 annual carry from the worked examples) — and on leverage, it erases your capital. The carry is not free; it is the premium on a policy that pays out against *you*.

### Misconception 2: "Diversification protects a levered carry book"

Carry traders often hold many target currencies and assets — long dollars, pesos, real, tech stocks — and tell themselves they are diversified, because in *normal* times these positions are only loosely correlated. The number that kills this belief: in a crisis, correlations go to **1.** In the August 2024 unwind, the yen rallied, US tech stocks fell, the Nikkei fell 12.4%, emerging-market currencies fell — *all at once*, because they were all expressions of the same underlying bet (funded by borrowed yen, dependent on calm and a wide rate gap). When the funding leg reverses, every position funded by it gets liquidated together. Diversification across positions that share a single funding source and a single risk factor is not diversification — it is the same trade wearing different hats, and they all come off at once.

### Misconception 3: "The unwind is predictable — I'll get out in time"

Every carry trader believes they will sense the turn and exit before the crowd. The problem is structural: the trade is crowded precisely *because* it is consensus, which means most participants are confident and complacent at exactly the moment of maximum danger. And the unwind, when it comes, is faster than human reaction — the VIX went from the teens to **65.7** in 2024 essentially overnight, on news (a 0.25% BOJ hike) that no one treated as a crisis trigger in advance. You cannot reliably time the exact day, because the trigger is often trivial and the move is a gap, not a glide. What you *can* read is the *conditions* — crowding, leverage, a compressing rate gap, suppressed volatility — and size accordingly. Trying to time the exact top of a crowded levered trade is how you end up as the forced seller, not the one who front-ran the exit.

## How it shows up for traders: the crowded-leverage warning signs

You will never get a memo that says "the carry unwind begins Tuesday." What you get instead is a *configuration* — a set of conditions that, together, mean the market is fragile and a small shock could trigger a large unwind. Learn to read the configuration. Four signals, in order of importance.

**1. A wide, stretched carry (the rate gap).** The fuel for a carry trade is the interest-rate differential. Track it directly: the Fed funds rate versus the BOJ rate, or whatever funding-versus-target pair is in vogue. A wide gap that has been wide for a long time means the trade has had years to crowd and lever. The danger is not the wide gap itself — it is the wide gap that *starts to close.* When the funding central bank signals hikes or the target central bank signals cuts, the carry's fuel is being cut off, and that is the classic unwind trigger. The 2024 episode fired the instant the gap began closing from both ends at once.

**2. Extreme positioning (crowding).** Use the data. The **Commitments of Traders (COT)** report, published weekly by the CFTC, shows how speculators are net-positioned in currency futures. When speculators are at a multi-year extreme short the yen (long the carry), the trade is crowded and the exit is narrow. Positioning extremes do not tell you *when* the unwind comes, but they tell you the *fuel load* is high — that if a shock arrives, the resulting fire will be large. (We covered reading positioning in the cross-linked COT post.)

**3. Suppressed volatility (the calm before).** Low, compressed volatility — a VIX in the low teens, an FX volatility index near multi-year lows — is not the all-clear; it is often the *setup.* Calm is what lets leverage and size build, because low realized volatility means low margin requirements, which lets traders hold bigger positions on the same capital. The most dangerous market is a quiet, low-volatility market with a crowded, levered consensus trade — because the quiet is precisely what inflated the position that will have to unwind. Volatility being cheap and low is a reason to be *more* alert, not less.

**4. The trade "everyone" is in.** The qualitative tell. When a carry trade has worked for years, it stops being a trade and becomes a *truth* — "the yen only goes down," "borrowing yen is free money," "Mrs. Watanabe can't lose." When the consensus hardens into something nobody questions, the marginal buyer is exhausted and only sellers remain. The moment a profitable trade becomes conventional wisdom is the moment its risk is highest, because conviction and crowding peak together.

None of these four is a timing signal on its own. Together, they are a *fragility reading.* Wide-and-closing carry + extreme positioning + suppressed vol + consensus conviction = a market primed for a violent unwind, waiting only for a trigger. You cannot know the trigger or the date. You can absolutely know the fragility.

## How to trade it: the playbook and the lessons

You will not consistently time the exact top of a crowded carry trade — almost nobody does, including the geniuses at LTCM. But you do not need to. The edge is not in calling the day; it is in *respecting the asymmetry* and reading the fragility. Here is the playbook the three episodes teach.

![Carry unwind playbook watch crowding leverage vol size small](/imgs/blogs/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks-7.png)

The playbook figure runs left to right through the discipline: watch crowding, watch leverage, watch the rate gap, watch volatility, then size small and front-run the exit. Walk through it concretely.

**Respect the asymmetry above all.** The carry trade pays pennies for years and takes them all back in days. Internalize the math from the worked examples: ~\$52,000 of annual carry against a single ~\$104,000 unwind loss — two years of income gone in a week, *before* leverage. Any position whose best case is a steady drip and whose worst case is a sudden multi-year loss must be sized for the worst case, not the best. If a 10% move in your funding currency would do unacceptable damage, your position is too big — full stop. The asymmetry is the whole trade; everything else is detail.

**Size for the steamroller, not the pennies.** The single most important defense is leverage discipline. From the magnifier example: at 1x, a 10% unwind is a 10% loss you survive; at 10x, the same move is a wipeout. The carry trade is survivable at modest leverage and lethal at high leverage, and the *only* variable you fully control is how much leverage you use. Use little. The traders who survive carry unwinds are not the ones who time them — they are the ones who were never levered enough to be forced sellers.

**Read the fragility, then act on the trigger.** Monitor the four warning signs — the rate gap, positioning extremes, suppressed volatility, consensus conviction. When all four light up, the market is primed. You still need a trigger, but the triggers rhyme: a funding-currency central bank turning hawkish (the BOJ hiking), a target-currency central bank turning dovish (the Fed pivoting), or any event that closes the rate gap. When the gap starts to turn against a maximally crowded, levered, low-volatility consensus, *that* is your signal to cut carry exposure before the crowd — or to flip and position for the unwind itself.

**Consider owning the other side.** The sophisticated move is not just to avoid the carry but to be positioned for its unwind when fragility is extreme. Long volatility (owning options or VIX exposure) is cheap precisely when the market is calmest — which is exactly when a crowded levered trade is most fragile. Long the funding currency (long yen) is the direct expression: when the carry unwinds, the funding currency rips higher. These positions bleed a little while you wait (the mirror image of the carry's drip) but pay enormously in the unwind. You are buying the insurance the carry traders are selling. The trick is to buy it when it is cheap — in the calm — not when everyone wants it in the panic.

**Never confuse a track record with safety.** LTCM had two Nobel laureates and years of stellar returns. The 2024 yen carry had worked for the better part of a decade. A long, smooth track record on a levered carry trade is not evidence of safety — it is evidence of *crowding*, which is the opposite of safety. The longer it has worked, the more leverage and consensus have built, and the more violent the eventual unwind. Treat a too-good, too-smooth return stream on a levered carry as a warning, not a comfort.

And here, at the end of the series, is the thread that ties it all together. We started with **money** — what it actually is, base money and broad money. We followed money into **liquidity** — the tide of reserves, repo, and net liquidity that funds every position. We watched central banks set the **policy** that prices the cost of money through rates, QE, and forward guidance. We learned to read the **regimes** — growth and inflation, the business cycle, risk-on and risk-off — that determine which assets win. We traced how policy **transmits** to every asset class, from bonds to stocks to currencies to crypto. And here, in the finale, all of it comes due in a single mechanism. A carry-trade unwind is the whole macro machine running in reverse at high speed: cheap money funds a crowded, levered bet; liquidity and a wide rate gap let it grow; a policy shift closes the gap; the regime flips from calm to crisis; and the deleveraging cascades across every asset class at once, transmitting the break system-wide. Leverage is where money, liquidity, policy, regimes, and transmission all meet — and when leverage breaks, it breaks all of them together, in days. Read the fragility, respect the asymmetry, and size for the steamroller. That is the discipline the whole series was building toward.

## Further reading & cross-links

- [What Actually Moves a Currency: Rates, Flows, and the Carry Trade](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — the foundations of the carry trade and interest-rate parity that this post levers up into a crisis.
- [How Monetary Policy Moves Currencies: Rate Differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — why the rate gap between two central banks is the fuel that builds and breaks every carry trade.
- [LTCM 1998: When Genius Failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — the full story of the first modern leverage detonation, in detail.
- [Following the Flows: Positioning, COT, and Dealer Hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) — how to read the crowding that turns a profitable carry into a fragile, one-sided trade.

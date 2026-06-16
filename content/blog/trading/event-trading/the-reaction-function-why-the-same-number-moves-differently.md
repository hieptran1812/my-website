---
title: "The Reaction Function: Why the Same Number Moves Markets Differently"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The surprise tells you the direction of the data; the market's reaction function decides the sign and size of the price move. Learn to read what the market cares about right now."
tags: ["event-trading", "macro", "reaction-function", "regime", "inflation", "recession", "nfp", "fomc", "risk-on-risk-off", "cross-asset"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The surprise (actual minus consensus) tells you which way the data came in; the market's *reaction function* decides whether that surprise pushes prices up or down, and by how much.
>
> - A reaction function is the rule that translates a data surprise into a price move. It is set by **what the market is most afraid of right now** — inflation, recession, a liquidity freeze, or a credit blow-up.
> - The same hot jobs print can crash stocks in an inflation regime (strong data = more Fed = higher rates = lower stocks) or rally them in a growth-scare regime (strong data = no recession). Bonds, by contrast, almost always hate strong data.
> - The trade is not "is this number good or bad?" It is "what does the market care about today, and which way does *that* lens bend this number?" Get the regime wrong and you fade the wrong direction.
> - The one number to remember: on **Feb 3 2023** a blowout **+517k** payrolls print sent the S&P **−1.04%** and the 2-year yield **+18bp** — good news, red screen, because the market only cared about the Fed.

On the morning of Friday, February 3rd, 2023, the U.S. economy printed one of the strongest jobs reports in living memory. Nonfarm payrolls came in at **+517,000** against a consensus of roughly **+187,000** — nearly three times what economists expected. The unemployment rate fell to **3.4%**, a 53-year low. By any plain-English reading, this was spectacular news: more people working, more paychecks, a labor market firing on every cylinder.

And the stock market fell. The S&P 500 closed the day **−1.04%**. The 2-year Treasury yield — the part of the bond market most sensitive to what the Federal Reserve will do next — jumped **+18 basis points**, one of its larger single-day moves of the year. The dollar rallied **+1.2%**. A report that screamed "the economy is healthy" was met with selling in stocks and a sharp repricing of interest rates higher.

If you traded that morning on the instinct that "good economic news is good for stocks," you lost money. The number was good. The reaction was bad. And the gap between those two sentences is the single most important idea in event trading: the surprise tells you the *direction of the data*, but a separate, invisible rule — the market's **reaction function** — decides the *sign and size of the price move*. This post is about how to find that rule before you put a trade on.

![Two-branch diagram showing a strong data surprise producing opposite stock moves in an inflation regime versus a growth-scare regime](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-1.png)

## Foundations: what a reaction function is

Let me build this from zero, because the whole post rests on three ideas that most beginners blur together: the *number*, the *surprise*, and the *reaction*.

**The number** is the raw data point — payrolls of +517k, CPI of 8.3% year-over-year, GDP of 3.1%. On its own, a number is just a fact about the economy. It is not a trade.

**The surprise** is the number minus what the market already expected. Markets are forward-looking auction machines: today's price already contains the *consensus forecast* of every release that hasn't happened yet. If economists expect +187k jobs and the market has priced that in, then a print of exactly +187k changes nothing — it was already in the price. Only the part the market did *not* expect — the surprise, here +330k above consensus — has the power to move price on release. This is why a "great" number can do nothing (everyone saw it coming) and a merely "okay" number can detonate the tape (it landed far from consensus). If you want the full mechanics of how consensus gets baked into price and why only the deviation matters, the companion piece [The Macro Calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) walks through each release schedule in detail.

**The reaction function** is the rule that converts a given surprise into a price move. Write it as a near-equation in your head:

```
price move  =  reaction_function( surprise , current_regime )
```

The surprise is the input. The regime is the dial. The reaction function is the machine in the middle. Change the dial — change *what the market is most afraid of* — and the *same* surprise comes out the other side with the opposite sign. That is the entire thesis: a strong jobs print is one fixed fact, but it routes through whatever the market currently cares about, and the routing decides whether stocks rally or crash.

### Good-news-is-good vs good-news-is-bad

There are two basic settings of the dial, and naming them is half the battle.

In a **good-news-is-good** regime, strong data is bullish for risk assets. The market's dominant worry is *growth*: will the economy slow, will earnings fall, are we heading into recession? When the worry is growth, a strong number is reassurance. More jobs means more spending means higher corporate earnings means stocks should be worth more. The number and the reaction point the same way. This is the world most beginners assume they live in, because it is the intuitive one — and for long stretches of history it is correct.

In a **good-news-is-bad** regime, strong data is *bearish* for stocks. The market's dominant worry is *inflation* and, by extension, *the central bank*. When prices are running too hot and the Fed is hiking to cool the economy, a strong number is bad news: it means the economy is *not* slowing the way the Fed needs it to, which means the Fed will hike *more*, or hold rates high *longer*. Higher rates lower the present value of future earnings (you discount them more steeply), drain liquidity out of risky assets, and make safe cash and bonds relatively more attractive. So a strong jobs print becomes a forecast of *more tightening* — and stocks fall even though the underlying news was "good."

The mechanism is worth slowing down on, because it is the hinge of the whole post.

### The Fed-path channel: why strong data can be bearish

A stock is a claim on future cash flows. Its value today is those future cash flows discounted back to the present at some interest rate. Two things move that value: the *cash flows* (earnings) and the *discount rate* (driven by interest rates).

When the market's worry is growth, the binding variable is earnings. Strong data lifts the earnings outlook, so stocks rise — the earnings channel dominates.

When the market's worry is inflation, the binding variable flips to the discount rate. Strong data does not change near-term earnings much, but it dramatically changes the *expected path of interest rates*: a hot economy forces the central bank to keep policy tight. Higher expected rates raise the discount rate, which lowers the present value of every future dollar of earnings. The discount-rate channel dominates, and it points *down*. Strong data, lower stocks. For a deeper treatment of how the Fed actually decides — and how the market builds the expected rate path it trades against — see [Inflation and the Fed Reaction Function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot), which covers the dot plot and the policy mechanism this post sits on top of.

### Risk-on, risk-off, and the dominant worry

You will hear the shorthand **risk-on** (money flowing into stocks, crypto, high-yield credit, away from safe havens) and **risk-off** (the reverse: money fleeing to Treasuries, the dollar, gold, cash). The reaction function is what decides whether a given surprise is read as risk-on or risk-off. And the dial that sets the reaction function is the **dominant macro worry** — the single thing the market is most scared of in the current window. Identify the dominant worry and you have, in one stroke, identified the reaction function. The rest of this post is a toolkit for naming that worry quickly and reading it off recent price action when you are not sure.

### Why the surprise, not the level, is what moves price

It is worth lingering on *why* price reacts to the surprise rather than the absolute level of the data, because beginners constantly trip on this. A stock or bond price is, in theory, the discounted value of every future cash flow, and those forecasts already incorporate the market's best guess of every upcoming release. If the entire forecasting industry expects +187k jobs and that expectation is embedded in today's price, then the price *already reflects* a +187k economy. A print of exactly +187k confirms what was assumed and changes nothing — the price does not move because no new information arrived. Only the *unexpected* component — the deviation from the embedded forecast — is genuine news, and only news moves price.

This has a sharp practical consequence: a print can be objectively *excellent* and produce *zero* reaction if it merely matched a high bar, while a print that is objectively *mediocre* can detonate the tape if it lands far from a low bar. The number's quality in absolute terms is almost irrelevant to the trade. What matters is the *gap to consensus* (the surprise) routed through the *regime* (the reaction function). Two inputs, and neither of them is "was the economy good or bad in absolute terms."

### The knee-jerk, the fade, and the trend

One more piece of vocabulary, because the reaction function plays out over time, not in an instant. When a number hits the tape, three things can happen to the price:

- The **knee-jerk** is the first, automatic move in the seconds after release, as algorithms and fast traders price the headline surprise. It is often the largest move of the session and it frequently *overshoots*.
- The **fade** is the partial reversal of the knee-jerk over the following minutes to hours, as slower money digests the internals (the components beneath the headline) and the overshoot corrects. Many event moves are knee-jerk-then-fade: the headline shocked, but the details were less dramatic, so the price drifts partway back.
- The **trend** is when the knee-jerk does *not* fade but extends, because the print genuinely shifted the regime's central variable — most often the expected path of the Fed. A print that resets the rate path tends to trend all session and into the next, because it changed the thing the whole market is pricing against.

The reaction function tells you which to expect. In an inflation regime, a print that meaningfully moves the Fed-path expectation tends to *trend* (the discount rate has genuinely repriced); a print that beats the headline but leaves the rate path unchanged tends to knee-jerk and *fade*. Knowing the regime is what lets you guess, before the dust settles, whether the first move is the move to chase or the move to fade.

#### Worked example: the strong-NFP good-is-bad day

Put real money on the Feb-3-2023 reaction so the mechanism becomes concrete. Say you walked into that Friday long the S&P 500 with a **\$25,000** position, betting that a strong jobs number would lift stocks the way it "should."

- The S&P closed **−1.04%**. Your loss: **\$25,000 × −0.0104 = −\$260**.
- Now suppose a colleague read the regime correctly. She was short Treasuries (betting yields would rise) into the print, because she knew strong data meant more Fed. The 2-year yield jumped **+18bp**, and her short profited as bond prices fell.
- The number was identical for both of you: **+517k** jobs. The reaction function split your P&L: **−\$260** for the person trading "good is good," a gain for the person trading "good is bad."

The lesson in one line: the same \$25,000 and the same number produced a loss or a gain purely on which reaction function you traded.

![Two-panel chart of the Feb 3 2023 reaction: S&P 500 down 1.04 percent and the 2-year Treasury yield up 18 basis points](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-2.png)

## The two regimes and the mechanism for each

Let me make the two regimes airtight, because almost every event-trading mistake is a regime error in disguise.

### Regime A — good-news-is-good (the growth-worry world)

**When it holds:** the market's overriding fear is that growth is too weak — a recession looming, earnings rolling over, demand evaporating. Inflation is either low or simply not the headline. The central bank is on hold or easing, so the rate path is not the swing factor.

**The mechanism:** strong data eases the recession fear. The chain runs: strong print → "the economy is not falling apart" → earnings outlook holds or improves → risk premium compresses → stocks up, credit spreads tighten, the dollar's safe-haven bid fades. Crucially, because the Fed is not in inflation-fighting mode, a strong number does *not* threaten a wave of new hikes, so the discount-rate channel stays quiet. The earnings channel runs the show and it points up.

**Examples in history:** much of the 2010s expansion, where for years a strong jobs Friday was greeted with a green open. The 2021 reopening, where every data point confirming the recovery was bullish. And — critically — the **growth-scare regime of late 2024**, which we'll dissect below, where the worry flipped *back* to recession and good-news-is-good returned.

### Regime B — good-news-is-bad (the inflation-worry world)

**When it holds:** the market's overriding fear is inflation, and therefore the central bank. Prices are running well above target, the Fed is actively hiking or threatening to, and every release is scored against one question: *does this make the Fed do more?*

**The mechanism:** strong data threatens *more tightening*. The chain runs: strong print → "the economy is still too hot for the Fed's comfort" → market reprices the rate path higher → discount rate up → present value of equities down → stocks fall, the front end of the bond curve sells off (2-year yield up), the dollar rallies on the higher-rate bid. The earnings channel is overwhelmed by the discount-rate channel. Good news, red screen.

**Examples in history:** essentially all of 2022 through early 2023. The marquee case is the **August 2022 CPI print** released September 13th 2022: inflation came in *hot* at 8.3% year-over-year versus 8.1% expected — a tiny 0.2 percentage-point surprise — and the S&P 500 fell **−4.32%**, its worst day since June 2020, with the Nasdaq down **−5.16%**, Bitcoin off roughly **−9.4%**, and the dollar up **+1.4%**. A two-tenths-of-a-point inflation miss erased over four percent of the entire U.S. equity market in a session, because in that regime a hot print meant the Fed was nowhere near done.

The flip side proves the rule. When inflation came in *cool* on the October 2022 CPI (released November 10th 2022) — 7.7% versus 7.9% expected, another small surprise but in the dovish direction — the S&P *rose* **+5.54%**, the Nasdaq **+7.35%**, the 10-year yield *fell* **−28bp**, and the dollar dropped **−2.1%**. In an inflation regime, the reaction function is symmetric and steep: hot inflation crushes stocks, cool inflation rockets them, because both are really bets on the Fed's path. If you want the cross-asset rotation mechanics behind these whippy sessions, [Risk-On, Risk-Off: How Money Rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) lays out exactly which assets catch the flow in each direction.

#### Worked example: a 2-year yield move sized in dollars

The bond reaction in an inflation regime is where a lot of event-driven money is actually made, and it is easy to mis-size if you only think in basis points. Say you hold **\$1,000,000** face of the 2-year Treasury note going into a strong jobs print, long (you own it, so you profit if yields fall and lose if yields rise).

- A 2-year note has a DV01 — the dollar value of a one-basis-point yield change — of roughly **\$190 per basis point** on a \$1,000,000 position (a ~2-year duration times \$1mm times 0.0001).
- The print lands strong, and the 2-year yield jumps **+18bp**, exactly as it did on Feb 3 2023.
- Your loss: **18bp × \$190/bp = −\$3,420**. Yields up, bond price down, and you were long.
- Had you instead been *short* that \$1,000,000 of 2-year notes — the correct inflation-regime trade into a strong number — you would have *gained* **+\$3,420** on the same move.

The takeaway: an 18-basis-point flicker that looks trivial on a screen is **\$3,420** of real money on a million-dollar position, and the reaction function told you in advance which side to be on.

## What the market cares about right now

The reaction function is set by the dominant worry, so the practical skill is naming that worry fast. There are four it cycles through. The decision tree below is the whole diagnostic on one page; the prose after it explains each branch.

![Decision tree mapping the market's dominant worry to its reaction-function rule for inflation, growth, liquidity, and credit](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-3.png)

### Worry 1 — inflation (2022–2023)

When inflation is the dominant worry, the reaction function is **good-news-is-bad** and it is steep and symmetric. You know you are here when: CPI prints are the highest-volatility events on the calendar; the financial press leads with "will the Fed hike again?"; the front end of the yield curve (the 2-year) moves violently on data; and "hot" and "cool" become the only adjectives anyone uses. In 2022–2023 the entire market was a leveraged bet on the path of the fed funds rate, which the Fed dragged from 0.25% to over 5% in eighteen months. Every release was a referendum on that path.

The tell that distinguishes an inflation regime from everything else: **strong growth data and stocks move in opposite directions, and the dollar rises with rate expectations.** When you see a strong number meet a falling stock market and a rising 2-year yield, you are unambiguously in good-news-is-bad.

### Worry 2 — growth / recession (2024 softening)

When the worry rotates to growth, the reaction function flips to **good-news-is-good**. Strong data soothes recession fear and lifts risk; weak data stokes it and hurts risk. You know you are here when: jobs reports and ISM/PMI surveys become the highest-volatility events (not CPI); the press leads with "is the Fed too late?" rather than "will the Fed hike?"; the unemployment rate and jobless claims get obsessed over; and the conversation is about *cuts*, not hikes.

The tell: **weak data now hurts stocks instead of helping them.** That is the precise inversion of the inflation regime, and catching the moment it happens is the most valuable single read in event trading — because the entire market's reflexes are still wired for the old regime. We'll devote a whole section to that handoff below. To place the growth worry inside the broader rhythm of expansions and recessions, [The Business Cycle: Four Phases for Traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) maps which worry tends to dominate in each phase.

### Worry 3 — liquidity (2020)

Sometimes the worry is neither inflation nor growth but *liquidity itself* — the plumbing of the financial system seizing up, where institutions cannot fund themselves and are forced to sell whatever they can. March 2020 was the archetype: as COVID lockdowns hit, even safe assets like Treasuries were dumped at one point because everyone needed cash *now*. In a liquidity regime, the reaction function collapses to a single mode: **flight to quality / dash for cash.** Data barely matters; what matters is who can get funding. Stocks, credit, even gold can fall together as positions are liquidated, while the dollar (the ultimate funding currency) spikes. The cure is usually the central bank flooding the system with liquidity, after which risk violently reverses. You know you are here when correlations across everything go to one and bid-ask spreads blow out — the topic of [When Correlations Go to One in a Crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) on the cross-asset side.

### Worry 4 — credit / solvency (2008)

The deepest worry is solvency — not "can this institution fund itself today?" but "is this institution *bankrupt*?" 2008 was the canonical credit regime: the fear was that banks and counterparties would actually fail, taking your money with them. In a credit regime the reaction function is again flight-to-quality, but with a vicious twist: the safe assets are *Treasuries and gold specifically* (claims on the government, not on a private balance sheet), and anything with credit risk — corporate bonds, bank stocks, mortgage paper — gets repriced toward default. Economic data is almost irrelevant; the only number that matters is the perceived probability that the next domino falls. You know you are here when credit spreads (the extra yield demanded to lend to companies over the government) are blowing out and the conversation is about *which firm is next*.

The two deeper worries — liquidity and credit — share a feature that makes them dangerous for event traders: in those regimes, *the calendar stops mattering*. A scheduled CPI or jobs print is almost irrelevant when the market's question is "can I get funded today?" or "is my counterparty solvent?" The reaction function collapses to flight-to-quality and the *triggers* become unscheduled — a bank failure, a fund blowing up, a funding market seizing. This is why the worst event-trading losses come from running an inflation- or growth-regime playbook *into* a liquidity or credit event: you are positioned for a data-driven reaction function while the market has switched to a panic reaction function that ignores data entirely. The tell that the dial has jumped to liquidity/credit is that *correlations go to one* — assets that normally move independently all crash together — and *volatility explodes*. When the VIX (the market's "fear gauge," the implied volatility of S&P options) leaps from the low 20s toward 40 or 60, the reaction function is no longer about the next number; it is about survival, and the only trade is reducing risk. We'll see exactly this in the August-2024 episode below, where the VIX spiked intraday to **65.73**.

The cross-asset reaction grid below sets these worries side by side, because the same strong print produces a *different sign in different assets* — bonds almost always hate it, but stocks flip with the regime.

![Matrix of how a strong data surprise moves bonds, stocks, the dollar, and gold across inflation and growth-scare regimes](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-7.png)

#### Worked example: mispricing the regime

This is where the money leaks out of beginner accounts, so price it explicitly. Suppose it is autumn 2022 — peak inflation regime — and a strong retail-sales number prints. You have internalized "good news is good for stocks" and you put on a **\$25,000** long S&P position expecting a pop on the strong consumer.

- Instead, the market reads strong consumer spending as "the Fed has more work to do," reprices the rate path higher, and the S&P sells off **−2.0%** on the day.
- Your loss: **\$25,000 × −0.020 = −\$500**.
- A trader who correctly named the regime did the opposite — faded the strength, or simply stayed flat into a print that could only hurt longs — and kept their **\$500**.
- The data was *exactly what you wanted to see*. The regime turned it into a **\$500** loss because you trained your reflexes on the wrong reaction function.

The intuition: in an inflation regime, being long risk into a strong print is not a bullish bet on the economy — it is a bet against the Fed, and the Fed has an unlimited balance sheet and your stop-loss does not.

## How a regime flips: the inflation-to-growth handoff

Regimes are not permanent. The reaction function flips, sometimes over weeks, sometimes in a single session, and the trader still running the *previous* regime's playbook gets run over at the handoff. The most important flip of the recent cycle is the **inflation-to-growth handoff**, and understanding its anatomy is worth more than memorizing any single reaction.

Here is the mechanism. In an inflation regime, the market hates strong data because strong data means more Fed. But the Fed's hiking *works* — eventually. Tight policy slows the economy, inflation rolls over, and at some point the dominant worry quietly shifts from "the Fed will hike too much" to "the Fed has hiked too much and broken something." The instant that shift completes, the reaction function inverts: now *weak* data is the scary thing (it confirms the recession) and *strong* data is reassuring (no recession yet). Good-news-is-bad becomes good-news-is-good.

The handoff is treacherous because it is not announced. For a transition period, the market is *ambiguous* — half of it still trades the inflation reaction function, half has switched to growth. You get whippy, two-sided reactions where a number rallies stocks for an hour and then reverses. The skill is to watch *which surprises are getting the bigger reaction*. When cool inflation prints stop producing huge stock rallies (the inflation trade is played out) and weak jobs prints start producing stock *selloffs* (the growth fear has taken over), the handoff is complete.

There is a characteristic *intermediate* phase worth naming explicitly: the **soft-landing window**, where both worries are partly live and the reaction function is genuinely two-sided. In early 2024 the market wanted inflation to keep falling (so the Fed could cut) *and* growth to stay solid (so earnings held up) — a "Goldilocks" setup where moderately cool data was the best of both worlds. In that window, a weak print is *ambiguous*: it is good for the rate path (more cuts) but bad for growth (recession risk), and the two pull stocks in opposite directions. The reaction function in a soft-landing window is unstable and prints fade more often than they trend, because the market itself has not settled on which worry dominates. The correct response to an unstable reaction function is to *size down* — when you cannot confidently name the dominant worry, the regime read is low-confidence and your position should be too. The whippiest, hardest-to-trade tape is not a strong inflation regime or a clean growth scare; it is the ambiguous handoff between them, and recognizing that you are *in* a handoff is itself a tradeable read: trade smaller, expect fades, and wait for the dial to settle.

The timeline below traces the actual sequence the market walked from 2021 to 2024 — reopening growth, to inflation, to soft landing, to growth scare — so you can see the dial physically turn.

![Timeline of the regime flip from 2021 reopening growth to 2022-23 inflation to 2024 soft landing and growth scare](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-5.png)

Trace the path. In **2021** the worry was "is the reopening recovery real?" — pure growth worry, good-news-is-good, strong data lifted stocks. Through **2022 and into early 2023** inflation hit a 40-year high of 9.06% (June 2022) and the worry became the Fed; good-news-is-bad ruled, and a 0.2pp hot CPI cost the S&P 4.32% in a day. Into **early 2024**, as inflation cooled toward target and the Fed signaled it was nearly done, the regime turned *ambiguous* — a soft-landing window where data could be read either way. And by **August 2024** the worry had fully rotated back to growth, with the unemployment rate climbing to 4.3% and triggering recession alarms. Good-news-is-good returned — which is why a *weak* jobs print in August 2024 hammered stocks, exactly as a *strong* one had hammered them eighteen months earlier. Same red close, opposite cause.

#### Worked example: the weak-NFP recession scare

Put dollars on the *other* end of the regime, so you feel the symmetry. It is August 2nd 2024. The jobs report comes in *weak*: **+114k** payrolls versus **+175k** expected, and the unemployment rate ticks up to **4.3%**, tripping a recession-warning rule of thumb. You are long the S&P with a **\$20,000** position.

- The S&P closed **−1.84%** on the recession scare. Your loss: **\$20,000 × −0.0184 = −\$368**.
- Note what just happened: *weak* data hurt your *long*. Eighteen months earlier, *strong* data hurt longs. The number's sign flipped, the P&L sign did not.
- That is the regime flip in one trade. In the inflation regime you lose being long into strength; in the growth-scare regime you lose being long into weakness. The reaction function, not the number, owns your P&L.
- The bond side confirms the flip: the 2-year yield *fell* **−28bp** that day (the market priced *cuts*), the mirror image of the **+18bp** rise on the strong Feb-2023 print. On a \$1,000,000 long 2-year position at ~\$190 DV01, that −28bp rally would have *gained* **\$5,320** — bonds love weak data in every regime.

The intuition: bonds are consistent (weak data is always good for them, because it means lower rates), but stocks are conditional, and the condition is the regime.

## Reading the reaction function from recent prints

You will often face a release without a textbook label for the regime. The market does not ring a bell. So here is the practical diagnostic: **read the reaction function off how the market responded to the last few prints.** Price action is the market telling you its own reaction function, in its own words. You do not have to guess the dominant worry in the abstract — you can observe it.

Run this three-question checklist before any major release:

1. **Which release moved the market most over the last month?** If CPI days are producing the biggest swings, inflation is the worry (good-news-is-bad). If jobs and growth surveys are producing the biggest swings, growth is the worry (good-news-is-good). The market votes with its volatility on what it cares about.

2. **On the last strong data point, which way did stocks go?** This is the single cleanest read. Strong data + stocks *down* = inflation regime. Strong data + stocks *up* = growth regime. You are reverse-engineering the reaction function from a known input-output pair.

3. **What is the front of the bond curve doing on data?** When the 2-year yield is whipping around on every release and the press is counting hikes, the Fed path is the swing variable — inflation regime. When the conversation has moved to cuts and the 2-year is grinding lower on weak data, the worry has rotated to growth.

The two jobs days below are the clinic for question 2. February 2023 was a *strong* print and stocks fell; August 2024 was a *weak* print and stocks *also* fell. From the inputs and outputs alone — without any macro theory — you can read off the regime each time: red-on-strong is good-is-bad, red-on-weak is good-is-good. The market told you its reaction function both times.

![Bar chart comparing two jobs days where the S&P 500 fell on both a strong and a weak payrolls surprise](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-4.png)

A further subtlety: the reaction function is also visible in the *cross-asset coherence* of a move. If a strong print sends stocks down, yields up, and the dollar up *together*, that internal consistency confirms an inflation-regime read — every asset is pricing the same "more Fed" story. If the assets disagree (stocks down but yields also down), the market is ambiguous, mid-handoff, and you should size smaller because the reaction function is not stable. Crowded positioning amplifies whichever reaction function is in force: when everyone is leaning the same way, a surprise that wrong-foots the crowd produces an outsized move as positions are unwound.

### Headline versus internals: which number the regime is reading

There is a deeper layer to "which number the market cares about." Most releases have a *headline* figure and a stack of *internals* (the components beneath it), and the regime determines *which line item* the market actually trades. This is where two traders looking at the same report can disagree about whether it was hot or cool.

Take the jobs report. Its headline is the payrolls count, but it also carries the unemployment rate, average hourly earnings (wage growth), labor-force participation, and revisions to prior months. In an *inflation* regime, the line the market obsesses over is often **average hourly earnings**, because wage growth is what feeds back into sticky services inflation — a strong payrolls number with *cooling* wages can actually be read as benign, while a weak payrolls number with *hot* wages can spook the rate market. In a *growth-scare* regime, the line that matters most is the **unemployment rate**, because a rising jobless rate is the cleanest signal that the labor market is cracking — which is exactly why the jump to 4.3% in August 2024 mattered more than the headline +114k itself, since it tripped the Sahm-rule recession warning.

The same is true of inflation reports. In the heat of 2022, the market traded the **core** CPI (excluding volatile food and energy) and, within it, **core services ex-housing** — the so-called "supercore" the Fed singled out as the stickiest component. A headline CPI that looked cool could still tank stocks if the supercore ran hot, because the supercore was the number the *Fed* was watching, and the market was trading the Fed. The practical rule: to read the reaction function, you must know not just the dominant worry but the *specific internal line item* that worry makes load-bearing — and that line item changes with the regime. The headline is the bait; the internals are the trade.

### Why the macro backdrop sets the dial

The deepest source of the reaction function is the macro backdrop — specifically, where inflation is relative to target and where the central bank is in its hiking or cutting cycle. When inflation is far above target and the Fed is chasing it, the Fed path *is* the market, and every data point gets scored as "more Fed or less Fed." The chart below overlays inflation and the fed funds rate across 2020–2025: the surge of CPI to a 40-year high and the Fed slamming rates from near-zero to over 5% is precisely the backdrop that *forced* the good-news-is-bad reaction function of 2022–2023.

![Dual-axis chart of CPI inflation and the fed funds rate from 2020 to 2025 showing the inflation regime backdrop](/imgs/blogs/the-reaction-function-why-the-same-number-moves-differently-6.png)

The reading: when the red inflation line is far above the 2% target and the blue policy-rate line is racing to catch it, you are structurally in an inflation regime, and your default reaction-function assumption should be good-news-is-bad until the data tells you otherwise. As inflation falls back toward target and the policy rate plateaus and then turns down (the right side of the chart, 2024 onward), the structural pressure for good-news-is-bad fades and the worry is free to rotate back to growth.

## The reaction function is asset-specific

A crucial refinement, because beginners over-simplify the regime into a single risk-on/risk-off switch: the reaction function is *different for different assets*. There is no one "the market" reaction; there is a bond reaction, a stock reaction, a dollar reaction, a gold reaction, and they do not always agree.

**Bonds almost always love weak data and hate strong data — in every regime.** A Treasury's price is mechanically tied to the expected path of interest rates. Strong data raises expected rates, which lowers bond prices (yields up); weak data lowers expected rates, which lifts bond prices (yields down). This is the one reaction function that *does not flip* with the regime, which makes bonds the cleanest expression of "the data was strong/weak" regardless of how stocks interpret it. That is why the 2-year yield is such a reliable regime instrument: it tells you what the market thinks about the Fed path without the editorializing that stocks add.

**Stocks flip with the regime,** as we've belabored — down on strong data in an inflation regime, up on strong data in a growth regime — because the discount-rate channel and the earnings channel trade dominance depending on the worry.

**The dollar usually rises on strong U.S. data,** because strong data lifts expected U.S. rates, which makes dollar-denominated assets more attractive relative to other currencies (the rate-differential channel). The dollar's reaction is more stable than stocks' — it mostly tracks the relative rate path — though in a pure liquidity panic the dollar also catches a flight-to-cash bid that has nothing to do with data.

**Gold is the contrarian:** it pays no coupon, so its reaction function is dominated by *real* (inflation-adjusted) yields and by fear. Strong data that lifts real yields hurts gold (the opportunity cost of holding a zero-yield asset rises); but data that stokes a liquidity or credit panic *helps* gold (flight to a hard asset). Gold's reaction depends less on the headline number and more on real rates and tail risk — the topic of [Real vs Nominal: Real Yields, the Master Signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) is the deeper read here, though we'll note simply that gold often moves *against* the dollar.

The practical upshot: do not trade "risk-on/risk-off" as a monolith. Trade each asset off *its own* reaction function. On a strong inflation-regime print you can be short stocks, short bonds, and long the dollar simultaneously — three different reaction functions, all consistent with the same "more Fed" surprise.

### Crypto: the high-beta amplifier of the regime

Bitcoin and the broader crypto complex deserve a special note, because their reaction function is the most volatile of all and the most regime-dependent. Crypto pays no coupon and has no earnings, so — like gold — it has no internal anchor; but unlike gold, it trades as a *high-beta risk asset* and a *liquidity sponge*. When the regime is good-news-is-bad (inflation, tightening), strong U.S. data drains liquidity expectations and crypto sells off hard: on the hot September-13-2022 CPI print, Bitcoin fell roughly **−9.4%** the same day the S&P fell −4.32% — crypto amplified the equity reaction by more than double. When a liquidity or growth panic hits, crypto is among the first things sold, as we saw in the August-2024 cascade where Bitcoin dropped roughly **−15%** in the carry unwind. The crypto reaction function is essentially "the stock reaction function, leveraged up" — it reads the same regime but with a bigger multiplier, because it is the purest expression of marginal liquidity and risk appetite. The deeper case for why crypto trades as a macro-liquidity asset rather than a hedge is its own subject, but for the reaction-function trader the rule is simple: in any regime, expect crypto to do what stocks do, only more so.

#### Worked example: crypto as the leveraged regime trade

Size the crypto amplification so the multiplier is concrete. Suppose on the hot Sep-13-2022 CPI you held a **\$10,000** Bitcoin position alongside a **\$10,000** S&P position, both long, into an inflation-regime print you misread as "inflation can't surprise hot again."

- The S&P fell **−4.32%**: your equity leg lost **\$10,000 × −0.0432 = −\$432**.
- Bitcoin fell roughly **−9.4%**: your crypto leg lost **\$10,000 × −0.094 = −\$940**.
- Same regime, same surprise, same dollar size — the crypto leg lost **more than double** the equity leg, because crypto is the high-beta expression of the identical reaction function.
- Combined, a single misread inflation print cost the two-leg book **−\$432 − \$940 = −\$1,372** on \$20,000 deployed, a **−6.9%** hit driven entirely by reading the reaction function wrong on the most leveraged asset.

The intuition: crypto does not have a *separate* reaction function from stocks; it has the *same* one with the volume turned up, so a regime error there is the most expensive error you can make.

### Vietnam: the same dial, with a local lag

The reaction-function logic is not U.S.-only, but emerging markets like Vietnam add two wrinkles: a *transmission lag* and a *currency constraint*. When the Fed is in an aggressive inflation-regime hiking cycle, the pressure flows offshore through the dollar. A surging dollar — the DXY peaked near 114.8 in late September 2022, up sharply on the back of every strong U.S. print — forces emerging-market central banks to defend their currencies. The State Bank of Vietnam raised its refinancing rate from **4.0%** to **6.0%** across autumn 2022 specifically to defend the dong against dollar strength, and the VN-Index — Vietnam's main stock benchmark — fell from a January-2022 peak near **1,528** to a trough of **911** on November 15th 2022, a roughly **−39%** drawdown. The reaction function abroad was the *same* good-news-is-bad logic (strong U.S. data → more Fed → stronger dollar → pressure on the dong → tighter SBV policy → lower VN-Index), but transmitted with a lag through the currency rather than instantaneously through the discount rate. When the U.S. regime turned and the SBV could *cut* (three cuts from 6.0% back to 4.5% across April–June 2023), the VN-Index recovered toward 1,130 by year-end 2023 and on toward 1,267 by end-2024. For the local-market trader, the lesson is that the *U.S.* reaction function sets the weather, and Vietnam's foreign-flow and policy reaction transmits it home a step later — which means watching the dollar and the U.S. regime is part of reading the VN-Index.

## How it reacted: real episodes

Theory is cheap. Here are two dated episodes — one strong print, one weak — where the *regime*, not the number, owned the price action. They are the bookends of the regime flip, and side by side they are the most efficient way to internalize the whole post.

### Episode 1 — Feb 3 2023: strong jobs, stocks down

The setup: early 2023, peak inflation regime. The Fed had hiked from 0.25% to 4.50% over the prior year and the market was obsessed with one question — when will the Fed stop? Every data point was scored against the rate path.

The print: January 2023 payrolls (released February 3rd) came in at **+517k** versus roughly **+187k** expected — a colossal beat — with unemployment at a 53-year low of 3.4%.

The reaction: the S&P 500 fell **−1.04%**, the 2-year Treasury yield jumped **+18bp**, and the dollar rallied **+1.2%**. A spectacular labor market was read, correctly for the regime, as "the economy is too hot for the Fed to stop hiking." Good news, red screen. The cross-asset coherence — stocks down, short yields up, dollar up — was the textbook signature of an inflation-regime good-news-is-bad reaction.

The lesson: in that regime, a strong number was not a reason to buy stocks; it was a reason to sell them and short the front of the bond curve. The trader who read the regime made money on both legs.

### Episode 2 — Aug 2 2024: weak jobs, stocks down too

The setup: by mid-2024 inflation had cooled toward target and the Fed had stopped hiking. The worry had rotated. The market was no longer scared of inflation; it was newly scared that the Fed had held rates too high for too long and the economy was cracking. Good-news-is-good had returned, which means its mirror image — *bad-news-is-bad* — was also in force.

The print: July 2024 payrolls (released August 2nd) came in *weak* at **+114k** versus **+175k** expected, and the unemployment rate jumped to **4.3%**, tripping the "Sahm rule" recession warning.

The reaction: the S&P 500 fell **−1.84%**, the 2-year yield *fell* **−28bp** (the market rushed to price rate *cuts*), and the dollar slipped **−0.4%**. This was a *growth-scare* reaction: weak data confirmed the recession fear, so stocks fell *and* bonds rallied — the opposite cross-asset signature from February 2023.

That weak print was the leading edge of a violent few sessions. Combined with a Bank of Japan rate hike that same week, the weak U.S. jobs number helped trigger a global carry-trade unwind on **August 5th 2024**: the Nikkei fell **−12.4%** (its worst day since the 1987 crash), the S&P dropped **−3.0%**, the VIX spiked to an intraday **65.73**, and Bitcoin fell roughly **−15%** — before the Nikkei reversed **+10.2%** the very next session. The full anatomy of that cascade lives in [Carry Trade Unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks), but the seed was a single weak jobs print landing in a regime where weak data was, for the first time in two years, the thing the market feared.

The lesson across both episodes: **stocks fell on a strong print and on a weak print.** If you only watched the number, you would conclude the market was irrational. Once you watch the regime, both reactions are perfectly logical — and perfectly opposite in cause.

#### Worked example: trading both episodes with the right reaction function

Tie the two episodes together with a single hypothetical book to feel how much the regime read is worth. Suppose you ran a **\$50,000** event-trading account and you sized a **\$10,000** position into each jobs print, trading the *correct* reaction function each time.

- **Feb 3 2023 (inflation regime):** you read good-news-is-bad and went *short* \$10,000 of S&P exposure into the strong print. The index fell **−1.04%**, so a short gained **\$10,000 × 0.0104 = +\$104**.
- **Aug 2 2024 (growth-scare regime):** you read good-news-is-good (and its mirror, bad-news-is-bad) and again went *short* \$10,000 into a print you expected to disappoint, or simply stayed long bonds. The S&P fell **−1.84%**, so a short gained **\$10,000 × 0.0184 = +\$184**.
- Combined: **+\$104 + \$184 = +\$288** across two prints, by reading two *opposite* reaction functions correctly.
- The naive trader who was long stocks "because the jobs number was good/bad for the economy" lost on both — roughly **−\$104** and **−\$184**, a **−\$288** round trip.

The intuition: the same checklist — name the worry, read the sign off recent prints — flipped your position 180 degrees between the two events and was right both times. That is the entire edge.

## Common misconceptions

**"Good economic data is always good for stocks."** This is the costliest myth in event trading, and the whole post is its refutation. On Feb 3 2023 the strongest jobs print in years sent the S&P **−1.04%**. On Sep 13 2022 a hot inflation number cost the index **−4.32%** in a session. Whether good economic data helps or hurts stocks depends entirely on the reaction function, which depends on the dominant worry. In an inflation regime, good economic data is *bad* for stocks because it means more Fed. Memorize the exception and you'll stop fading the wrong way.

**"The bigger the surprise, the bigger the move."** Mostly true, but the *sign* and *multiplier* are set by the regime, not the surprise size. A 0.2pp inflation miss moved the S&P **−4.32%** in September 2022 because the regime made CPI the most important number on earth. The identical 0.2pp miss in a low-inflation, growth-worry regime might move stocks a fraction of a percent — or the *opposite* way. Surprise magnitude sets the raw energy; the reaction function decides how that energy gets translated into price, and in which direction.

**"Risk-on and risk-off are a single switch for all assets."** No — the reaction function is asset-specific. Bonds love weak data in every regime; stocks flip; the dollar tracks the rate differential; gold tracks real yields and fear. On a strong inflation-regime print you can correctly be short stocks, short bonds, and long the dollar at once. Treating the market as one risk dial throws away most of the available edge and gets the gold and bond legs wrong.

**"If the data is strong and stocks fall, the market is being irrational."** The market is not irrational; it is trading a different variable than you are. You are looking at the economy; the market is looking at the Fed. In an inflation regime, strong data is a forecast of higher rates, and higher rates lower equity values through the discount-rate channel. The reaction is the rational consequence of the regime — it only looks irrational if you are scoring the number on the wrong scale.

**"Once you know the regime, you're set for months."** Regimes flip, sometimes fast, and the flip is where the most expensive mistakes happen. The inflation-to-growth handoff of 2023–2024 inverted the entire reaction function: weak data went from being a stock *tailwind* to a stock *headwind*. The trader who kept running the 2022 playbook into the 2024 regime got the sign wrong on every print. The reaction function must be re-read continuously, not set once.

## The playbook: trade the reaction function, not the number

Here is the operational sequence to run before, during, and after any major macro release. The discipline is to decide the regime *first*, then map the surprise into a position — never the reverse.

**Step 1 — Name the dominant worry (set the dial).** Before the release, answer the three diagnostic questions: which release has been moving the market most, which way did stocks go on the last strong print, and what is the 2-year yield doing on data? Write down "inflation regime, good-news-is-bad" or "growth regime, good-news-is-good" *before* you see the number. If you can't name the regime, you are not ready to trade the print — size to zero or to a tiny exploratory clip.

**Step 2 — Get the consensus and define the surprise scenarios.** Find the consensus forecast (it is published for every major release). Sketch three branches: a hot surprise (well above consensus), an in-line print (no surprise, expect a fade of any knee-jerk move), and a cool surprise (well below). The in-line case is the trap most beginners forget — a "good" number that merely matches expectations usually does *nothing*, because it was already in the price.

**Step 3 — Map each surprise to a position, per asset, using the regime's reaction function.** In an inflation regime: hot surprise → short stocks, short the front of the curve (yields up), long dollar; cool surprise → the reverse. In a growth regime: strong surprise → long stocks, long dollar, short bonds; weak surprise → short stocks, *long* bonds (the recession-scare trade). Build the matrix in advance so you are reading from a script, not improvising while the tape is moving.

**Step 4 — Trade the reaction, not the number.** The first move on release is the knee-jerk; it can overshoot and fade, or it can extend into a trend if the print genuinely changed the regime's central variable (e.g., a print that resets the Fed path). Wait for the cross-asset coherence to confirm: if stocks, yields, and the dollar all line up with your regime read, the reaction is "clean" and likely to trend; if they disagree, the market is ambiguous and the move is more likely to fade. The deeper mechanics of spike-then-fade-or-trend deserve their own treatment, but the regime read is what tells you whether to expect a fade or a trend in the first place.

**Step 5 — Set the invalidation: what would prove the regime wrong.** Your single biggest risk is that the regime is *flipping* underneath you. Define, before the trade, the price action that would tell you the reaction function has changed: a strong print that suddenly *rallies* stocks in what you thought was an inflation regime; a weak print that suddenly *sells off* stocks in what you thought was a growth regime. The moment you see the reaction function invert, your regime read is invalidated — exit, reassess, and re-name the worry. Do not marry a regime.

**Step 6 — Size for the regime's volatility.** Inflation-regime CPI days can move the S&P 4–5% in a session; a quiet growth-regime jobs Friday might move it half a percent. Size the *same dollar risk* across events by scaling position size *down* when the regime makes the event high-volatility. A \$25,000 position that risks an acceptable amount on a calm day risks five times as much on a CPI day in an inflation regime — adjust accordingly so one print can't blow the account.

#### Worked example: sizing the same dollar risk across regimes

Make the position-sizing math explicit, because constant *dollar* risk requires *variable* position size as the regime changes the event's volatility. Say your rule is to risk no more than **\$500** on any single release.

- **Calm growth-regime jobs Friday**, where the S&P typically moves about **0.5%** on the print: to cap risk at \$500, you can carry a position of **\$500 / 0.005 = \$100,000**, because a 0.5% adverse move on \$100,000 is **\$500**.
- **Inflation-regime CPI day**, where the S&P can move **4.0%** (as on Sep-13-2022): to cap the same **\$500**, your position must shrink to **\$500 / 0.040 = \$12,500** — one-eighth the size — because a 4.0% adverse move on \$12,500 is also **\$500**.
- Carry the calm-day \$100,000 size *into* the CPI day and a −4.32% print would cost **\$100,000 × −0.0432 = −\$4,320**, more than eight times your risk budget on a single number.

The intuition: the reaction function does not just set the *direction* of the move, it sets the *size* — so the same dollar discipline forces you to trade a far smaller position when the regime makes a release explosive.

The one-sentence version of the whole playbook: **the surprise is the input, the regime is the dial, and the reaction function is the trade — so spend your preparation naming the regime, and the position falls out of it automatically.**

## Further reading and cross-links

- [Inflation and the Fed Reaction Function: the Dot Plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — the policy mechanism beneath the good-news-is-bad regime: how the Fed decides and how the market builds the expected rate path it trades.
- [Risk-On, Risk-Off: How Money Rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the cross-asset rotation that the reaction function drives, asset by asset.
- [The Business Cycle: Four Phases for Traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — where each dominant worry tends to sit in the expansion-to-recession cycle, so you can anticipate regime flips.
- [Risk-On, Risk-Off: the Cross-Asset Rotation](/blog/trading/cross-asset/risk-on-risk-off-the-cross-asset-rotation) — the allocator's-lens view of how the same regime read maps to portfolio positioning across asset classes.

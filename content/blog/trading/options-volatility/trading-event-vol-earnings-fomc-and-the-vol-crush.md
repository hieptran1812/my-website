---
title: "Trading Event Vol: Earnings, FOMC, and the Vol Crush"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to read the expected move off the straddle, why implied vol ramps into a known catalyst and crushes the instant it lands, and how to structure long-vol, short-vol, and calendar trades around the event so you collect the crush instead of donating to it."
tags: ["options", "volatility", "vol-crush", "earnings", "fomc", "expected-move", "straddle", "iron-condor", "calendar-spread", "event-trading"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A known catalyst (earnings, an FOMC decision, a CPI print) is a scheduled spike of uncertainty. The option market prices that uncertainty *in advance* by inflating implied volatility into the event, and then *releases* it the instant the news lands — the **vol crush**. Trading event vol means betting on whether the actual move will beat the move the market already paid for, while surviving the crush that hits everyone holding the event expiry.
>
> - The **expected move** is readable straight off the at-the-money straddle: its price is roughly the one-standard-deviation move the chain is pricing for the event. Compute it, then compare it to how big the stock *actually* tends to move on its prints.
> - **Long vol** (buy the straddle/strangle) only wins if realized beats the priced expected move *and* the move outruns the crush. **Short vol** (sell the straddle, the iron condor, or a calendar that sells the event expiry) harvests the crush if the move stays inside the expected move — and carries a fat tail when it doesn't.
> - The crush is mechanical: the event variance is priced as extra implied vol, and once the variance is realized there is nothing left to price, so front-expiry IV collapses — often from 75% to 35% on a single name overnight.
> - The one number to remember: **the breakeven is the expected move, not zero.** A 7% move on a stock whose straddle priced an 8.3% breakeven is a *losing* trade for the straddle buyer and a *winning* trade for the seller — direction was right, magnitude wasn't.

It was an earnings night, and two traders had opposite bets on the same \$100 stock. The first, a straddle buyer, had paid \$8.28 for the at-the-money call plus put expiring in seven days. Her thesis was simple: the company was about to report, the stock always jumped on earnings, and she wanted to own the move without guessing direction. The second trader had sold an iron condor — a defined-risk short-volatility structure — and collected a \$1.69 credit, betting the stock would stay inside a range. They were mirror images: she was long the event's volatility, he was short it.

The print came out after the close. It was a *good* print — the stock gapped up 7% overnight, opening at \$107. A 7% overnight move is a real move; it makes the morning headlines. The straddle buyer woke up, saw the gap, and assumed she'd won. Then she looked at her account. Her \$8.28 straddle was worth about \$7.32 — she was *down* \$0.96 a share, an 11.5% loss, on a night the stock moved 7% in her favor. The condor seller, meanwhile, watched the same 7% gap land *inside* his short strikes; with the post-event volatility collapsing, he bought his condor back for \$1.37 and kept \$0.32 of his \$1.69 credit. One trader was right about direction and lost. The other never had a directional view and won.

What separated them was not luck and not direction. It was that the market had *already priced* a roughly 8.3% move into that straddle, and when only 7% materialized — and the moment the uncertainty resolved, the implied volatility that had pumped the premiums collapsed from about 75% to 35% — the buyer's options deflated faster than the move could refill them. This is the **vol crush**, and structuring trades around it is one of the most reliable, most misunderstood corners of options trading. This post is the options-structuring playbook for event volatility: how to read the expected move, why IV ramps and crushes, and how to be the condor seller instead of the straddle buyer — or, when the setup is right, the straddle buyer who actually wins.

![Front-expiry implied volatility ramping up into a scheduled event then collapsing the next day, with the vol crush marked](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-1.png)

The chart above is the whole phenomenon in one shape. The blue line is the front-expiry at-the-money implied volatility of a single stock in the trading days around its earnings report. For weeks it drifts at a baseline. As the report approaches, it *ramps* — climbing from the mid-30s toward 75% — not because anyone is panicking, but because the option that expires *after* the report now has to cover a known burst of uncertainty in a shrinking number of days. The instant the report is out, the uncertainty is resolved: there is no more event variance to price, and IV *crushes* back toward its baseline overnight. The amber band marks the crush — the cliff every event-expiry option holder rides down. The entire art of trading event vol is positioning *for* that cliff, not getting run over by it. We will build the whole picture from first principles, price every leg with the Black-Scholes model, and end with a decision framework for which side to take.

For the macro-event *mechanics* — how a stock, a currency, or the index reacts to a CPI surprise or an FOMC dot-plot shift — this post links out to the event-trading series: [the expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) and [event volatility: implied vs realized and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush). Here, we own the *options-structuring* angle: which spread to put on, what Greeks it carries, and where it blows up.

## Foundations: what "event vol" actually is

Let me build this from zero, because every term in the hook is load-bearing and the whole trade collapses if you skip one.

An **option** is a contract that gives you the right — not the obligation — to buy (a *call*) or sell (a *put*) a stock at a fixed price, the *strike*, before a fixed date, the *expiry*. You pay a *premium* up front for that right. The premium has two parts: **intrinsic value** (how far in-the-money the option already is) and **time value** (everything else — what you pay for the *possibility* that the stock moves in your favor before expiry). For an at-the-money option, the price is essentially all time value, and time value is mostly a function of one input: **volatility**. If you want the full pricing derivation, it lives in [Black-Scholes](/blog/trading/quantitative-finance/black-scholes); our job here is to *trade* the model's output, not re-derive it.

**Volatility** is the annualized standard deviation of a stock's returns. A stock with 20% volatility has a one-standard-deviation move of about 20% *over a year*. There are two flavors, and the gap between them is the entire game of options trading. **Realized volatility** is how much the stock *actually* moved, measured after the fact. **Implied volatility** (IV) is the volatility number that, plugged into the pricing model, reproduces the option's market price — it is the market's *forecast* of future volatility, backed out of what people are paying. When you buy an option you are paying for implied vol and hoping realized comes in higher; when you sell one you are collecting implied and hoping realized comes in lower. That trade lives at the heart of every options position, and we treat it in full in [implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options).

Now add the word **event**. An **event** (or *catalyst*) is a scheduled release of information that the market knows is coming and knows roughly when: a company's quarterly **earnings** report, a Federal Reserve **FOMC** rate decision, a monthly **CPI** inflation print, a jobs report. The defining feature is that it is *known in advance*. Everyone can see it on the calendar. And because everyone can see it, the option market prices it in advance — which is exactly why event vol behaves differently from ordinary vol.

Here is the core mechanism, stated plainly. The variance (volatility squared) an option must cover over its life is the sum of the variance from *ordinary* day-to-day trading plus the variance from any *event* that falls inside its window. Variance is additive over time — that is why volatility scales with the square root of time. So an option expiring *after* the earnings date has to cover ordinary daily noise *plus* the one-day burst of the earnings move, while an option expiring *before* the report covers only the noise. The event burst is a chunk of extra variance crammed into a single day. To price it, the model has to lift the *implied* volatility of the post-event option well above the stock's normal level — because IV is the single knob that absorbs all that extra variance. That elevated IV is the **event premium**.

> [!note]
> **Why the crush is mechanical, not emotional.** People describe the vol crush as fear "wearing off," but it is pure arithmetic. Before the report, the option's price has to cover the event's variance, so its implied vol is pumped. The instant the report is released, that variance is *realized* — it has happened, it is now part of the stock's price. There is no more event variance left inside the option's remaining window, so the only volatility left to price is the ordinary daily kind. The model's implied vol drops to the ordinary level. The premium deflates accordingly. The crush is the option's IV returning to baseline because the thing it was pricing has occurred.

#### Worked example: how much of the straddle is pure event premium

Take a \$100 single-name stock that reports earnings tomorrow. The at-the-money seven-day options trade at an implied volatility of **75%** — a level no stock sustains for long, but normal *into* a print. Price the at-the-money straddle (buy the \$100 call and the \$100 put) with the Black-Scholes model, `T = 7/365`, `r = 4%`, no dividend:

- \$100 call: **\$4.18** per share
- \$100 put: **\$4.10** per share
- **Straddle premium: \$8.28** per share

Now ask what that straddle *would* cost if the stock's IV were its ordinary, non-event level of, say, 35%. Repricing the same seven-day straddle at 35% gives roughly **\$3.87**. The difference — about **\$4.41** per share — is the **event premium**: the portion of the straddle's price that exists *only* because an earnings report sits inside the seven days. More than half the straddle's cost is paying for one day of news. **The intuition: when you buy a straddle into earnings, more than half of what you pay is rent on a single event, and that rent evaporates the moment the event happens.**

### The Greek signature of an event position

Before structures, internalize the Greeks, because they explain *why* the crush hits the way it does. Every option position is described by the signs of four sensitivities. **Delta** is sensitivity to the stock price (direction). **Gamma** is how fast delta changes — convexity, the thing that makes a long-options position accelerate in your favor as the stock moves. **Vega** is sensitivity to implied volatility — how much you gain or lose per vol point of IV change. **Theta** is sensitivity to the passage of time — the daily bleed of time value.

A long-vol event position (a bought straddle) is **long vega, long gamma, short theta, and roughly flat delta**. That single line is the trade and the trap. Long vega means a *falling* IV — the crush — hurts you directly: you own the thing that deflates. Short theta means each passing day, especially the most expensive day right before expiry, costs you. Long gamma is the *only* thing working for you: if the stock moves far enough fast enough, convexity can outrun the vega and theta losses. Into an event, you are betting your one long Greek (gamma, fed by the realized move) beats your two short ones (the vega loss from the crush and the theta loss from the burned day). That is a hard bet, because the crush is large and certain while the move that would feed your gamma is uncertain and already priced.

A short-vol event position (a sold condor or calendar) flips every sign: **short vega, short gamma, long theta**. The crush — falling IV — *pays* you, because you are short the thing that deflates. Time passing *pays* you. Your one short Greek is gamma: a large, fast move feeds the buyer's convexity at your expense, which is precisely the tail. So the symmetry is exact: the long-vol trader prays for gamma to beat the crush; the short-vol trader collects the crush and prays gamma stays asleep. The deeper treatment of vega and the *vol of vol* — the volatility of implied volatility, which spikes hardest around exactly these scheduled events — is in [vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol).

## Reading the expected move off the straddle

The single most useful number in event trading is the **expected move**: how far the market is pricing the stock to travel on the event. You do not have to model it — the option chain hands it to you. The at-the-money straddle's price is, to a close approximation, the one-standard-deviation move the chain is pricing through expiry. The reason is the same arithmetic as before: the straddle's value is dominated by the expected size of the move, and for a roughly symmetric distribution the ATM straddle prices at about 0.8 standard deviations of that move. Practitioners use the straddle price directly as the expected move because it is the breakeven of the purest long-vol bet, and it lines up closely with the true one-sigma figure.

![The expected move read off the ATM straddle, with breakevens and a distribution of historical earnings moves](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-2.png)

The chart above makes it concrete. The blue marker is the straddle price (\$8.28) plotted as a symmetric band around the \$100 spot: the chain is pricing a move of roughly **±\$8.28**, putting the breakevens at **\$91.72 and \$108.28**. Overlaid in gray is a distribution of the stock's *actual* historical earnings moves — the absolute one-day percentage change on each of its past prints. The teaching point is the relationship between the priced move and the realized distribution. If most historical prints clustered *below* the priced ±8.28% band, the straddle is *rich* — the market is overpaying for this stock's typical reaction, and the seller has the edge. If the realized moves routinely *exceeded* the band, the straddle is *cheap* and the buyer has the edge. The straddle price tells you the bar; the history tells you how often the stock clears it.

There is a subtlety worth internalizing. The straddle price (\$8.28) sits a little *below* the model's exact one-standard-deviation move. The one-sigma move implied by 75% vol over seven days is `\$100 × 0.75 × √(7/365) ≈ \$10.39`, or about 10.4%. The straddle's \$8.28 is about 0.80 of that, exactly the ATM-straddle-to-sigma ratio. So the *breakevens* of a long straddle (±\$8.28) sit *inside* the one-sigma band (±\$10.39): the straddle starts paying off a touch before a full one-standard-deviation move. When traders say "the chain is pricing an 8% move," they usually mean the straddle price; when a risk model says "one-sigma is 10%," it means the standard deviation. Both are correct; they differ by that 0.8 factor. For the rest of this post, "the expected move" means the straddle price — because that is the number the buyer must beat and the seller wants the stock to stay inside.

#### Worked example: computing the expected move and comparing it to history

Our \$100 stock, seven-day straddle at 75% IV:

- Straddle price = **\$8.28**, so the expected (priced) move is **±8.28%**.
- Breakevens: **\$91.72** and **\$108.28**.
- The exact one-sigma move = `\$100 × 0.75 × √(7/365)` = **\$10.39 (10.4%)**.

Now suppose this stock's last eight earnings moves were, in absolute terms, 5%, 9%, 4%, 6%, 11%, 3%, 7%, and 5% — an average absolute move of about **6.3%**. The market is pricing an 8.28% breakeven, but the stock has *historically* moved 6.3% on average, clearing the 8.28% bar only twice in eight prints (the 9% and the 11%). That is a straddle priced *above* the stock's typical reaction: the seller's edge. **The intuition: the expected move is the bar the buyer must clear; if the stock historically clears that bar less than half the time, you want to be the one selling the straddle, not buying it.**

The expected move is also the bridge to the event-trading series, where the same number is computed for the index ahead of an FOMC or CPI release — see [the expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options). The arithmetic is identical; only the underlying changes.

## The IV ramp and the crush, quantified

We have the two halves of the story: IV *ramps* into a known catalyst and *crushes* the instant it lands. Now let's quantify both, because the magnitudes are what determine whether a trade survives.

**The ramp.** In the days before a report, traders who want exposure to the event buy the post-event options, and market makers, knowing the event variance must be covered, mark implied vol up. The front-expiry ATM IV climbs steadily — a stock that trades at 35% IV in a quiet month might see its earnings-week IV at 60%, 75%, even 100%+ for a volatile name. The ramp is gradual and predictable; it is not a surprise to anyone. Crucially, *the closer to the event, the more concentrated the event premium becomes*, because the same fixed chunk of event variance is being spread over fewer and fewer days. That is why the ramp accelerates into the final day or two.

**The crush.** The moment the report is released — typically after the close for a stock, or at 2:00 PM Eastern for an FOMC decision — the event variance is realized. The next morning, the same front-expiry option that priced 75% IV reprices at the stock's ordinary level, perhaps 35%. The IV does not drift down; it *gaps* down overnight. For the option holder, two things happen simultaneously and in the same direction: vega loss (the IV dropped, and you were long vega) and theta loss (a day passed, and the most expensive day of time value just burned off). Both drain the premium.

The size of the crush is the whole risk. A drop from 75% to 35% is a **40-vol-point** crush — and on a seven-day ATM option, vega is large, so 40 points of IV is an enormous amount of premium. This is why the long straddle can lose on a correct-sized move: the realized move adds intrinsic value, but the crush subtracts time value, and near the money the crush usually wins.

#### Worked example: decomposing the crush into vega and theta

Our \$100 straddle, paid \$8.28 at 75% IV with seven days left. The next morning, one day has elapsed (six days left) and IV has crushed to 35%. Suppose the stock is *unchanged* at \$100 (the cleanest case — no intrinsic value added, pure premium decay). Reprice:

- Straddle at 75% IV, 7 days = **\$8.28** (what you paid)
- Straddle at 35% IV, 6 days = **\$3.58** (what it's now worth)
- Loss = **\$4.70** per share, or **57%** of the premium, *with no stock move at all*

Of that \$4.70, the bulk is the vega loss (40 points of IV × the straddle's vega) plus one day of the steepest theta on the calendar. **The intuition: on an unchanged stock, the vol crush alone vaporizes more than half a seven-day straddle overnight — which is why "the stock didn't move so I'm fine" is exactly backwards for a long-vol event trade.** And it is exactly *right* for the short-vol trade: the seller who collected \$8.28 and bought it back at \$3.58 just kept \$4.70.

## Structuring around the event: the three families

Once you understand the ramp and crush, every event trade reduces to a choice among three families. The first decision is always the same — is the implied volatility (the priced expected move) *cheap* or *rich* relative to the move you actually expect? — and the answer routes you to long vol, short vol, or a calendar.

### Long vol: buy the straddle or strangle

You buy the at-the-money straddle (or a cheaper, wider out-of-the-money **strangle**) when you believe the stock will move *more* than the priced expected move. This is the trade we cover in full in [straddles and strangles: the long-volatility bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet). Into an event, the long straddle has a brutal double hurdle: the realized move must beat the breakeven (the expected move) **and** the move must outrun the post-event crush. Because you are paying *peak* IV, you are buying the crush — you own the thing that is about to deflate. The long-vol event trade wins only when the move is genuinely *bigger* than what was priced, which, given that the market priced it with full knowledge of the calendar, is a high bar.

![Long straddle into earnings, P&L at a seven percent move with the vol crush versus the no-crush counterfactual](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-3.png)

The chart above is the straddle buyer's tragedy from the hook. The red curve is the straddle's value the morning after, *with* the crush (IV 75% → 35%, six days left); the amber dashed curve is the counterfactual where IV had *stayed* at 75%. The blue horizontal line is the \$8.28 premium paid. Follow the +7% move (the stock at \$107, marked): on the no-crush curve the straddle is worth \$9.86 — a tidy \$1.58 profit, the trade the buyer thought she was making. But on the real, crushed curve it is worth only \$7.32 — a **\$0.96 loss**. The 7% move was real and in her favor; the crush simply outran it. The vertical reference lines mark the breakevens (±8.28%): the stock had to clear *those*, not zero, and 7% fell short.

#### Worked example: the long straddle that loses on a 7% move

Buy the \$100 straddle for **\$8.28** (75% IV, 7 days). Earnings hit; the stock gaps to **\$107** (+7%). The next morning, six days left, IV crushed to 35%:

- Straddle value at \$107, 35% IV, 6 days = **\$7.32**
- P&L = `\$7.32 − \$8.28` = **−\$0.96 per share (−11.5%)**

How big a move *would* have won? Repricing at the crushed 35% IV, the straddle returns to \$8.28 (breakeven) at about a **+8.3%** move (the stock at \$108.30). You needed the stock to clear roughly the *priced expected move* of 8.28%, not merely to "move a lot." A +9% move makes \$0.88; a +12% move makes \$3.80. **The intuition: the breakeven for an event straddle is the expected move the chain already priced — beating zero is not enough, you have to beat the number the seller charged you for, after the crush takes its cut.**

### Short vol: sell the straddle, or define risk with an iron condor

You sell volatility when the priced expected move looks *rich* — when the market is overpaying for this stock's typical reaction. The purest version is selling the straddle naked, which collects the full premium but carries *unbounded* risk if the stock blows through a breakeven. The disciplined version caps that risk: the **iron condor**, which we treat in [iron condors and credit spreads: selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range). A condor sells an out-of-the-money put spread and an out-of-the-money call spread simultaneously — you collect a credit for betting the stock stays *between* the short strikes, and the long wings cap your loss if it doesn't.

![Short iron condor harvesting the crush, P&L when the move stays inside the expected move versus a tail move that breaks the wing](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-4.png)

The chart above shows the condor seller's P&L at expiry as a function of where the stock lands. The green plateau in the middle is the max profit — the full credit kept when the stock finishes between the short strikes (\$92.5 and \$107.5). The payoff slopes down through the short strikes and flattens into the red max-loss shelves once the long wings (\$87.5 and \$112.5) engage. The amber band marks the **expected move** (±8.28%): notice that the short strikes sit just *outside* the breakevens of the straddle the seller is implicitly fading. As long as the stock stays inside the expected move — which, by construction, it does more than half the time when the straddle is rich — the condor seller harvests the crush.

#### Worked example: the iron condor that harvests the crush

Pre-event, with everything at 75% IV, sell the \$92.5/\$107.5 short strikes and buy the \$87.5/\$112.5 wings (5-point-wide spreads):

- Sell \$92.5 put: **\$1.29** + Sell \$107.5 call: **\$1.57** = \$2.86 collected
- Buy \$87.5 put: **\$0.45** + Buy \$112.5 call: **\$0.72** = \$1.17 paid
- **Net credit: \$1.69** per share. Max loss = `5.00 − 1.69` = **\$3.31**.

Now the stock gaps +7% to \$107 and IV crushes to 35%. The \$107 close is *inside* the \$107.5 short call, and the crush has deflated every leg. Buy the whole condor back:

- Cost to close after the crush = **\$1.37**
- P&L = `\$1.69 − \$1.37` = **+\$0.32 per share** — the seller keeps part of the credit *even though the stock moved 7% toward his short strike*, because the crush deflated the options faster than the move inflated them.

But the tail is real. Suppose instead the stock gaps **+12% to \$112** and stays there to expiry: the short \$107.5 call is \$4.50 in-the-money, the long \$112.5 call is worthless, and the position loses `\$1.69 − \$4.50 = −\$2.81` — near the **−\$3.31** max loss. **The intuition: the condor seller wins small and often by harvesting the crush, but a move that breaks a short strike turns a \$1.69 winner into a \$3.31 loser — the structure trades a high win rate for a fat, defined tail.**

### Calendar: sell the event expiry, own a later one

The third family is the cleverest, and it is purpose-built for the crush: the **calendar spread** (also called a time spread). You *sell* the front-month option that expires right after the event and *buy* a same-strike option that expires a month or two later. We cover the full mechanics in [calendars and diagonals: trading time and term structure](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure). The calendar's logic for events is precise: the *front* option carries almost all of the event premium (the event sits inside its short window, so its IV is maximally pumped), while the *back* option carries the same event variance spread over a far longer window, so its IV is much less inflated per day. You sell the maximally-pumped front and own the lightly-pumped back. After the event, the front crushes hard and the back crushes far less — you keep the difference.

This is the trade that *isolates the crush* with the least directional and vega risk, as long as the stock stays near the strike. It is short the front IV, long the back IV — a short-event-vol bet that explicitly harvests the term-structure hump we describe next.

#### Worked example: the earnings calendar that sells the event-expiry IV

ATM at \$100. Sell the front 7-day call (at the pumped 75% IV) and buy the 35-day call (at a much lower 45% IV, because the event is one of many days inside its window):

- Sell front 7-day call @ 75% = **\$4.18** collected
- Buy back 35-day call @ 45% = **\$5.74** paid
- **Net debit: \$1.56** per share — the cost of the spread.

After earnings, with the stock unchanged at \$100, the front IV crushes to 35% and the back to about 40%:

- Front 6-day call @ 35% = **\$1.82** (you're short this — it collapsed)
- Back 34-day call @ 40% = **\$5.05** (you're long this — it held up far better)
- Position value = `\$5.05 − \$1.82` = **\$3.22**
- P&L = `\$3.22 − \$1.56` = **+\$1.66 per share**, more than doubling the debit.

The whole gain came from the front crushing harder than the back. **The intuition: a calendar is a bet on the *shape* of the crush — you sell the expiry where the event premium is most concentrated and own the one where it is diluted, so you profit from the front collapsing toward the back.** The risk: a large move away from the strike hurts the calendar (the back loses delta value and the front you're short can't fully offset), so calendars want the stock to *pin* near the strike — exactly when the move stays inside the expected move.

## Earnings-specific structure: the term-structure hump

Earnings introduce a feature you can *see* on the term structure of implied volatility — the curve of IV across expiries. Normally this curve is smooth: nearer expiries trade a bit higher or lower than far ones depending on the regime (see [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve)). But when a dated event sits between two expiries, the expiry that *contains* the event bumps up above its neighbors — a visible **hump** or **kink** in the curve.

![Implied volatility by expiry showing a hump on the earnings expiry above its neighbors, marked sell it](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-5.png)

The figure above shows the mechanism. The weekly expiry *before* earnings trades at the stock's ordinary IV; the weekly expiry that *includes* earnings spikes up — the hump; the monthly expiry *after* sits lower again because the event variance is diluted across more days. That hump *is* the event premium, made visible by the term structure. And the hump is the trade signal: it is the expiry you want to *sell* (it carries the concentrated event premium) and the neighbor you want to *own* (cheaper per day). Selling the hump and owning a neighbor is, structurally, the earnings calendar from the previous section. The term-structure hump is *why* calendars work into earnings: you are arbitraging the kink the event carves into the curve.

This also explains a common confusion. Traders see the front-week IV at 75% and the monthly at 45% and conclude "the monthly is cheap." It is not cheap — it simply contains the same one-day event spread across thirty-five days instead of seven, so the event's contribution to its annualized IV is much smaller. Both expiries price the *same* dollar event move; they just annualize it differently. The hump is an artifact of annualization, not a mispricing — but it is a tradable artifact, because the front crushes to its floor while the back barely moves.

### Single-name vs index event vol, and the drift caveat

**Single-name** event vol (one stock's earnings) is large and idiosyncratic: a single company can move 10%, 20%, even 30% on a print, and its event premium is correspondingly huge — IVs of 75% to 150% into the report are routine. **Index** event vol (the S&P 500 into an FOMC or CPI print) is far smaller, because the index is a diversified average: a macro surprise moves it 1% to 2%, not 10%, so the event premium is a few vol points, not forty. The structures are the same; the magnitudes differ by an order of magnitude. A single-name calendar can double; an index calendar harvests a few tenths.

The **post-earnings-announcement drift** (PEAD) is the one caveat that complicates the clean "sell the crush" story. Empirically, stocks that beat expectations tend to keep drifting *up* for days or weeks after the print, and missers keep drifting down — the market underreacts to the surprise. For the short-vol seller, drift is mostly harmless if the initial move stays inside the strikes, but it means the stock may keep trending *through* a short strike in the days after, turning a first-morning winner into a loser if you hold too long. The practical rule: event-vol structures are *event* trades, not *trend* trades — close them once the crush has been harvested (often the next morning), and do not let a drifting stock walk into your short strike.

## FOMC and macro events: the whole surface moves

Earnings move *one* stock's vol. A macro event — an FOMC rate decision, a CPI print, a nonfarm payrolls report — moves the *entire* index volatility surface at once, because it re-prices the discount rate and growth outlook for every name simultaneously. The mechanics of the FOMC meeting itself (the statement, the dot plot, the press conference, the two-stage reaction) are covered in [the FOMC meeting: full anatomy](/blog/trading/event-trading/the-fomc-meeting-full-anatomy); here we take the options-structuring view.

Into an FOMC decision, the index's front-expiry IV ramps just as a single name's does — the weekly SPX options that expire after the 2:00 PM decision carry an event premium — but the ramp is measured in a few vol points, not forty, because the index moves less. The crush is correspondingly gentler. The expected move read off the index straddle is small in percentage terms but large in points, and it is the number every desk watches into the print.

#### Worked example: the expected move on the index into an FOMC decision

The S&P 500 index sits at 5000, with the FOMC decision in two days. The ATM two-day straddle trades at 18% IV (a mild ramp from a ~14% baseline). Price it:

- 5000-strike call + put, `T = 2/365`, 18% IV = **≈ 53 index points**
- So the chain prices a move of **±53 points (±1.06%)**, breakevens at **4947 and 5053**.
- One-sigma = `5000 × 0.18 × √(2/365)` = **66.6 points (1.33%)**.

If you believe the decision is a foregone conclusion — the dots are well telegraphed and the market is positioned — selling that 53-point straddle (or, with defined risk, an index iron condor or a 2-day/9-day calendar) collects the event premium that crushes from 18% back toward 14% once the statement is out. If you believe the decision could genuinely surprise — a hawkish dot-plot shift, an unexpected hold — buying the straddle bets the move beats 53 points. **The intuition: on the index, the expected move is small in percent but the whole surface ramps and crushes together — so the structuring choice is identical to earnings, just at one-tenth the magnitude.** The full reaction patterns across crypto, FX, gold, and equities are catalogued in [earnings season as a macro event](/blog/trading/event-trading/earnings-season-as-a-macro-event) and the event-trading series.

Positioning *into* the print matters more for macro events than for earnings, because the whole market is leaning the same way. When everyone is short vol into a "telegraphed" FOMC, a surprise produces an outsized move as shorts cover — the crush you were counting on becomes a *spike*. This is the short-vol seller's nightmare and the reason event-vol sizing must be small: see [vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) for how the *vol of vol* — the volatility of implied volatility itself — spikes precisely around these events.

### Timing the entry: where in the ramp to put the trade on

A subtlety that separates practitioners from spectators is *when* in the IV ramp to enter. The ramp is gradual, so the implied vol you sell (or buy) depends on how early you act. Sell too early and the ramp is still climbing — your short vega can lose money on a *rising* IV even before the event, as the event premium keeps building into the print. Sell at the last moment and you capture the peak premium, but the spread you cross is widest and the book is most crowded. The same logic, inverted, applies to the long-vol buyer: buy too early and you pay theta for days while the ramp does you no good; buy at the last moment and you pay the richest IV of the cycle.

The practical convention for short vol is to sell *into* the final ramp — the day or two before the event — when the event premium is most concentrated and the time left for an adverse pre-event drift is shortest. You are selling the maximally-pumped front IV with the least exposure to the ramp still climbing. For long vol, the opposite: if you genuinely believe the move will beat the priced expectation, you want to own the structure *before* the last leg of the ramp inflates your entry price — but this trades a cheaper entry for more days of theta bleed and more IV-ramp vega gain. There is no free lunch; the ramp is priced, like everything else.

The crush itself is *fast*. For a stock reporting after the close, the entire crush is delivered in the overnight gap: you go to bed with 75% IV and wake to 35%. There is no intraday window to react — the IV gaps at the open. For an FOMC decision at 2:00 PM, the crush plays out over the minutes after the statement and through the press conference, so there is a (frantic) intraday window. Either way, the edge is *delivered* almost entirely at the moment the news hits, which is why holding past the crush converts a vol trade into a directional one. The trade is the crush; once it is harvested, the position is just residual risk you no longer want.

## The win/lose threshold: it's always the expected move

Step back and the whole subject collapses to one chart. Every event-vol structure has a P&L that depends on the *realized* move, and there is always a threshold — the expected (priced) move — that separates the long-vol winner from the short-vol winner. The long position needs the realized move to *beat* the expected move; the short position needs it to *stay inside*.

![Structure P&L versus realized move size, with the priced expected move marked as the threshold between long and short winners](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-6.png)

The chart above plots the post-event P&L of the long straddle (red) and the short iron condor (green) against the size of the realized move, with the crush applied. The vertical amber line is the **expected move** (8.28%) — the breakeven the chain priced. To its *left* (small moves), the condor seller wins and the straddle buyer loses: the move stayed inside what was priced, the crush dominated, and the premium the seller collected is the buyer's loss. To its *right* (large moves), the lines cross: the straddle buyer's profit grows without bound as the move gets larger, while the condor seller hits a defined max loss once the move breaks a wing. The expected move is the fulcrum. Everything in event-vol trading is a bet on which side of that fulcrum the stock lands — and the market, having priced the straddle with full knowledge of the calendar, has done its best to put the fulcrum exactly where the odds are even.

This is the most important figure in the post, so read its asymmetries carefully. The condor seller (short vol) has a *high probability* of a *small* win and a *low probability* of a *capped* loss — a tall, narrow green plateau that falls off the cliff at the wings. The straddle buyer (long vol) has a *low probability* of a *large* win and a *high probability* of a *small* loss — a red valley that only climbs out on the extremes. Neither is "better"; they are mirror payoffs of the same expected-move bet. Which one is correct depends entirely on whether the priced move is rich or cheap relative to the move you actually expect — and *that* is the only judgment that matters.

#### Worked example: the threshold in dollars

Our \$100 stock, 7-day straddle at \$8.28, crush to 35%. The threshold:

- **Long straddle** breaks even (after the crush) at about a **±8.3%** move. Below it, the buyer loses (e.g., −\$0.96 at +7%); above it, the buyer profits (+\$0.88 at +9%, +\$3.80 at +12%).
- **Short condor** keeps the \$1.69 credit fully if the stock stays inside ±7.5% (the short strikes); it begins losing past the strikes and hits the −\$3.31 max loss past the ±12.5% wings.

So at a **+7% realized move**, the buyer loses \$0.96 and the seller keeps \$0.32 — both confirm the move stayed *inside* the priced expectation. At a **+12% move**, the buyer makes \$3.80 and the seller loses \$2.81 — the move *beat* the priced expectation. **The intuition: the entire game is realized-versus-priced; the priced expected move is the line, and you only have an edge if you have a genuine reason to think the stock lands on a particular side of it — not a directional view, a *magnitude* view.**

## Common misconceptions

**"Buy a straddle before earnings to profit from the move."** This is the single most expensive folk belief in retail options, and the worked examples above demolish it with numbers. Into a known print, you pay *peak* IV, which means you pay for the expected move *in advance*. The breakeven is not zero — it is the expected move (±8.28% in our example, or ±8.3% after the crush). A 7% move, which feels enormous, *loses* \$0.96 because it fell short of the priced 8.3% and the crush deflated the premium. To win, the stock must move *more* than the market — which already knew about the earnings date — chose to price. You are not betting the stock moves; you are betting it moves *more than the people who priced it think it will*. That is a real bet sometimes, but "buy the straddle for the move" treats a high bar as if it were zero.

**"The vol crush only matters if the stock doesn't move."** Backwards. The crush matters *most* when the stock moves a normal amount — inside the expected move — because that is precisely the region where the vega-and-theta loss from the crush exceeds the intrinsic value the move adds. On an unchanged stock the crush costs the long straddle its whole \$4.70 of premium; on a +7% move it still costs \$0.96 net. The crush only *stops* mattering once the move is so large that intrinsic value swamps it — past the expected move. So "the stock moved, so the crush is irrelevant" is exactly the trap: a 7% move is large enough to feel like a win and small enough that the crush still wins.

**"Selling vol into events is free money because the crush is guaranteed."** The crush *is* nearly guaranteed — IV does collapse after the event resolves. What is not guaranteed is that the move stays inside your strikes. Short vol has a high win rate and a fat tail: our condor wins \$0.32 on a 7% move and loses \$2.81 on a 12% move — almost ten times the win. Sell event vol enough times without sizing for the tail and one gap through a short strike erases a year of crush harvests. The crush is the *edge*; the tail is the *risk*; "free money" ignores the second half and is how short-vol books blow up. Size small, define the tail with wings, and never sell naked event vol in size.

**"A bigger expected move means the stock is more likely to make a big move, so buy it."** A bigger expected move means the option is *more expensive*, which raises the breakeven by exactly the same amount. A stock pricing a 15% expected move is not "more likely to deliver a big move you can profit from" — it has simply set a 15% bar you must clear. The expected move scales the *cost* and the *breakeven* together, so a hot, high-IV name is not a better long-vol buy than a quiet one. The only thing that makes a long-vol trade good is realized *exceeding* the priced move, and rich names are, if anything, *harder* to beat because the market has already paid up for their reputation for moving.

**"Calendars are a directional trade."** A calendar's profit comes from the front expiry crushing harder than the back, which happens when the stock *pins* near the strike — it is short event vol and long time, not a directional bet. A large move *away* from the strike actually *hurts* the calendar, because the back option loses value as it goes out-of-the-money faster than the front you're short can offset. The calendar wants the same thing the condor wants: the move to stay inside the expected move. Treating it as "I think the stock goes up so I'll buy a call calendar" misunderstands the structure — you put it on at the strike you expect the stock to *sit at*, harvest the crush, and close it.

## How it shows up in real markets

The clearest live laboratory for event vol is the four-times-a-year cadence of mega-cap tech earnings. Take the canonical pattern that repeats across names like Netflix, Tesla, Meta, and Nvidia: in the two weeks before the report, front-week IV ramps from the 35-45% baseline toward 60-90%, the term-structure hump on the earnings expiry becomes visible, and the straddle prices an expected move in the high single digits or low double digits. Desks compute that expected move daily and compare it to the company's historical print-day moves. When the priced move sits well above the historical average — the straddle is rich — the variance-risk-premium harvesters (covered in [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt)) line up to sell condors and calendars. When a name has a history of *gapping harder than priced*, the long-vol crowd buys the straddle.

The single most instructive episode is the recurring pattern where a stock reports a *good* number, gaps up a respectable amount, and the straddle buyers *still* lose — the exact scenario in our hook. It happens every earnings season, on multiple names, because the move, however real, lands inside the priced expectation and the crush does the rest. Traders who learned options on directional payoff diagrams find it almost impossible to believe the first time: the company beat, the stock rose, and the call-plus-put position lost money. The crush is the missing variable, and once you have priced one with the model, the pattern is obvious forever after.

There is a structural reason the short-vol side has the long-run edge here, and it is the same edge that pays option sellers everywhere: the *variance risk premium*. Across thousands of events, the implied volatility priced into options runs a few points above the realized volatility that follows — the market systematically overpays for the insurance that options provide, because most participants are net buyers of protection. Event premium is a concentrated, dated form of that same overpricing: the straddle into earnings prices a move that, on average across all names and all prints, comes in a little smaller than priced. That is why the disciplined condor-and-calendar sellers grind out the crush quarter after quarter — they are harvesting the variance risk premium at its most concentrated point. It is also why they must respect the tail: the premium is positive *on average*, but the realized distribution has a fat right tail (the surprise gap), and a seller who ignores the tail will eventually meet the print that pays for all the others. The whole structure of that edge-and-its-limit is the subject of [the variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt).

The macro side shows up around the FOMC calendar and the monthly CPI print. The whole SPX surface ramps a few points into the 2:00 PM decision and crushes after the press conference, with the well-known "two-stage" reaction: an initial move on the statement, then a second move on the chair's tone. The single most dangerous episode for short-vol sellers is the *telegraphed* meeting that surprises — when the entire market is short vol into a "nothing" FOMC and the dot plot shifts hawkish, the crush everyone bet on becomes a spike, shorts scramble to cover, and the move blows through the strikes that looked safe. The yen-carry-unwind episode of August 2024, when the VIX closed at 38.57, is the recent reminder that the short-vol tail is not theoretical: a positioning-driven event can detonate a crowded short-vol book in a day. Event-vol structuring is not about predicting the news; it is about pricing the move the chain implies and taking the side where the realized distribution gives you an edge — while sizing for the day the tail arrives.

## The playbook

Here is how to actually trade event vol, start to finish. The figure below is the decision in one frame — the single question that routes you to long vol, short vol, or a calendar.

![Event-vol decision tree routing rich versus cheap implied vol to short vol, long vol, or a calendar](/imgs/blogs/trading-event-vol-earnings-fomc-and-the-vol-crush-7.png)

**1. Read the expected move off the straddle first.** Before any view, price the front-expiry ATM straddle. Its price is the expected move (the breakevens are `strike ± straddle`); the exact one-sigma is the straddle divided by 0.8. This is the bar. Write it down: "the chain is pricing a ±8.3% move."

**2. Compare the priced move to the realized distribution.** Pull the stock's last 8-12 earnings moves (or the index's last several FOMC/CPI moves). Is the priced expected move *above* or *below* the typical realized move? Above → the straddle is rich → lean short vol. Below → cheap → lean long vol. No clear edge → stand aside; the market has priced it fairly and you are just paying the spread and the crush.

**3. Choose the structure to match the view and cap the tail.**
- **Rich move, expect a quiet print → sell vol.** Use an **iron condor** ([selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range)) with short strikes near the expected-move breakevens and defined wings, *not* a naked straddle. You harvest the crush; the wings cap the tail.
- **Rich move, expect the stock to pin near a level → sell the event expiry.** Use a **calendar** ([trading time and term structure](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure)) at that strike: sell the pumped front, own the lightly-pumped back, profit from the front crushing toward the back.
- **Cheap move, genuine reason to expect a bigger jump than priced → buy vol.** Use a **straddle or strangle** ([the long-volatility bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet)), and accept you must beat the expected move *after* the crush. Only do this when you have a real edge on magnitude, not a directional hunch.

**4. Know your Greek profile going in.** Long vol = long vega, long gamma, short theta — you own the crush and you bleed if the stock sits. Short vol = short vega, short gamma, long theta — you collect the crush and you are exposed to the tail. The single dominant risk into an event is vega: a 40-point IV crush on a single-name straddle is the whole P&L. Track your net vega; if a surprise *spike* would blow up your book, you are too big.

**5. Size for the tail, not the base case.** Short-vol event structures win small and often and lose large and rarely. Size each trade so the *defined max loss* (the wing-to-wing distance minus the credit) is a small fraction of capital — the kind of position sizing covered in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). Never let one telegraphed-event-that-surprised erase a year of crush harvests.

**6. Harvest the crush, then get out.** Event-vol trades are *event* trades. The edge is the overnight crush. Once it has been harvested — usually the morning after — close the position. Do not hold a short condor into the post-earnings drift, and do not let a long straddle's residual vega ride hoping for more. The reason you put the trade on (the event premium) is gone after the event; staying in turns a vol trade into a directional one you never wanted.

**7. The one rule.** *The breakeven is the expected move, not zero.* Whether you are long or short, every event-vol trade is a bet on whether the realized move beats the move the chain already priced. Long vol needs realized to beat it; short vol needs it to stay inside. The crush is the gravity that pulls every long-vol holder down and every short-vol holder up — price it with the model before you ever click the trade, and you will be the condor seller keeping the credit, not the straddle buyer who needed 9% and got 7%.

## Further reading & cross-links

Within this series:

- [Straddles and strangles: the long-volatility bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet) — the long-vol structure and why it loses into events.
- [Iron condors and credit spreads: selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — the defined-risk short-vol structure for harvesting the crush.
- [Calendars and diagonals: trading time and term structure](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure) — selling the pumped event expiry against a later one.
- [Vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — the Greek that the crush acts on, and why vol-of-vol spikes around events.
- [Implied vs realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the gap every event-vol trade is built on.
- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the curve the earnings hump sits on.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the structural edge of the crush harvester.

For the macro-event mechanics (this post is the options-structuring angle):

- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options)
- [Event volatility: implied vs realized and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush)
- [The FOMC meeting: full anatomy](/blog/trading/event-trading/the-fomc-meeting-full-anatomy)
- [Earnings season as a macro event](/blog/trading/event-trading/earnings-season-as-a-macro-event)

For the pricing theory underneath: [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) and [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) for sizing the tail.

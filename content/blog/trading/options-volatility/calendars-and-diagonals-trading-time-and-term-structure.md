---
title: "Calendars and Diagonals: Trading Time and Term Structure"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How a calendar spread sells fast front-month decay against a slow long-dated option, why its P&L is a tent that peaks at the strike at front expiry, why it is long vega and positive theta at the same time, and how a diagonal adds a directional tilt."
tags: ["options", "volatility", "calendar-spread", "diagonal-spread", "theta", "vega", "term-structure", "time-decay", "implied-volatility", "options-greeks", "black-scholes"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A calendar spread sells a near-dated option and buys a longer-dated option at the *same strike*. You collect the front month's fast time decay while owning the back month's slow decay, and because the back month carries more volatility exposure, the structure is **long vega and positive theta at the same time** — a combination a single option can never give you. Its profit is not a hockey stick; it is a *tent* that peaks at the strike and must be drawn at the front expiry date, not the back.
>
> - **The trade is differential decay.** A 30-day at-the-money call bleeds about **−\$4.36 per contract per day**; the 60-day call at the same strike bleeds only about **−\$3.24**. Sell the first, own the second, and you net roughly **+\$1.12 a day** in theta while the stock sits still.
> - **The payoff is a tent.** Build the 30/60-day \$100 calendar for a **\$1.11 net debit** (\$111 per contract). At front expiry the position is worth the most if the stock pins the strike — about **+\$134 per contract** at \$100 — and the most you can lose is the \$111 you paid.
> - **It is long vega, so it lives and dies on implied vol.** A calendar *gains* if IV rises and *loses* if IV falls — the opposite of a condor. This is why the textbook calendar is an *event* trade: sell the inflated event-expiry option, own a cheaper post-event one, and harvest the front-month crush.
> - **The one rule to remember:** a calendar is a bet that the front decays faster than the back *and* that back-month implied vol holds up. If you only watch theta and ignore vega, a volatility crush on your long leg can turn a "positive-theta" position into a loss.

## The trader who sold the earnings straddle against a cheaper back month

A volatility trader I'll call Priya ran a small earnings book at a prop shop. Her edge was not predicting where stocks went on earnings — she was honest enough to admit she had no idea. Her edge was reading the *term structure* of implied volatility around a known event and trading the part of it the market reliably overpaid for.

Here is the pattern she traded, over and over. A large-cap tech name was due to report earnings on a Thursday after the close. In the days before the print, the options expiring that Friday — the ones that would *contain* the earnings move — repriced sharply higher. Their implied volatility ran up to 60%, 70%, sometimes 90% annualized, because everyone knew a big one-day move was coming and the option that straddled the event was the only instrument that paid off on it. Meanwhile the options expiring a month later, which also contained the same earnings event but diluted it across many more calm days, traded at a far tamer implied vol — maybe 35%. The near-dated option was *rich*; the far-dated one was *cheap*, on a per-day-of-variance basis.

So the week of the print, Priya put on a calendar. She *sold* the Friday at-the-money straddle — the front month, fat with earnings premium — and *bought* the same-strike straddle one expiry out, the cheaper back month. She paid a small net debit. Then she waited for the event.

The stock reported, gapped 5%, and by Friday's open the front-month straddle she had sold collapsed. Not because she was right about direction — she wasn't, the stock had moved more than she'd have guessed — but because the *uncertainty* the front-month option was pricing had been *resolved*. Once the number was out, the Friday option that had cost 70% implied vol crashed to almost nothing of time value; it was now just intrinsic, melting toward expiry. The earnings premium she'd sold evaporated overnight. Meanwhile her long back-month straddle, which still had a month of life and still priced ordinary post-earnings volatility, held most of its value. She bought back the cheap front-month remains, kept the back month or sold it into the still-elevated post-event vol, and booked the difference.

She wasn't betting on the move. She was betting on the *shape of the curve collapsing* — that the grotesque front-month earnings IV would crush back toward normal faster than the back month would. That bet, sized small and repeated across dozens of earnings events a quarter, was a business.

![Calendar spread profit and loss versus stock price at front expiry, a tent peaking near the strike](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-1.png)

This post is about the structure Priya traded and its cousins. The calendar spread — and its strike-shifted relative, the diagonal — is the cleanest way an options trader expresses a view on *time* and on the *term structure of volatility* rather than on direction. It is also the structure most often misunderstood, because its payoff diagram cannot be drawn the way you draw a vertical spread, and its Greeks point in a combination that sounds impossible until you see why. Before we get to the trade, we have to build the two ideas it rests on: that options at different expiries decay at different speeds, and that implied volatility has a *term structure* you can trade.

If you want the underlying machinery — what theta is, what vega is, what the term structure of volatility *means* — this post leans on three earlier ones and assumes you've at least skimmed them: [theta, the price of being long options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options), [vega, your exposure to implied volatility](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol), and [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve). Here we put them together into a position.

## Foundations: two options, two clocks, one strike

Let's build the calendar spread from absolute zero, defining every term as we go.

An **option** is a contract giving its owner the right, but not the obligation, to buy (a *call*) or sell (a *put*) a stock at a fixed price — the **strike** — on or before a fixed date — the **expiry**. The price you pay for that right is the **premium**. (If any of those words are new, start with [calls, puts, and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options); we won't re-derive the basics here.)

A **calendar spread** — also called a *time spread* or *horizontal spread* — is built from exactly two options that are identical in every way *except* their expiry:

- **Sell one near-dated option** (the *front month*). You receive its premium.
- **Buy one longer-dated option** (the *back month*). You pay its premium.
- **Same strike, same type** (both calls, or both puts). Same underlying.

Because the back month always costs more than the front month — more time means more optionality, hence more premium — you pay a **net debit** to put a calendar on. That debit is the most you can lose. We'll prove that shortly.

![Calendar structure showing a sold front-month option and a bought back-month option at the same strike](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-3.png)

The figure lays out the two legs and what each contributes. The front leg is the *income* side — you sold it, it decays fast, it pays you rent every day. The back leg is the *asset* side — you own it, it decays slowly, and it carries the volatility exposure. Between them sits the single shared strike and the small net debit you pay. Hold that structure in mind; everything that follows is just reading its consequences.

The word "horizontal" comes from how options chains are laid out: strikes run down the rows (vertical), expiries run across the columns (horizontal). A vertical spread trades two strikes at one expiry; a calendar trades two expiries at one strike. The calendar moves *across* the chain, hence horizontal. A **diagonal**, which we'll build later, moves both across expiries and down strikes — it travels the chain diagonally, which is where its name comes from, and that diagonal travel is exactly what gives it a directional lean the plain horizontal calendar lacks.

There is one more framing worth installing before the math, because it is the framing professional vol traders actually use. A single option bundles together three different bets — a bet on direction (delta), a bet on time (theta), and a bet on volatility (vega) — and you cannot easily separate them. A calendar *unbundles* them. By holding the strike and the type fixed across both legs, the calendar nets out most of the directional bet (the deltas nearly cancel) and isolates the two bets that remain: the bet on differential time decay and the bet on the back month's volatility. That is why the calendar is the structure of choice when your view is "I have an opinion about time and vol, but not about direction." It is a *purifying* trade: it strips the direction out and leaves you holding the time-and-vol view cleanly.

### Why two expiries decay at different speeds

The entire profit engine of a calendar is one fact: **a near-dated option loses its time value faster than a far-dated one.** This is the heart of theta, and it is worth re-deriving the intuition.

An option's premium is made of two parts. **Intrinsic value** is what you'd collect if it expired right now — `max(stock − strike, 0)` for a call. **Extrinsic value**, also called *time value*, is everything above intrinsic — the premium you pay for the chance that the stock moves favorably before expiry. Time decay, **theta**, only eats the extrinsic part; intrinsic value doesn't decay.

Now here's the asymmetry. An at-the-money option's extrinsic value scales roughly with the *square root* of the time remaining. A 60-day option does not have twice the time value of a 30-day option — it has only about √2 ≈ 1.41 times as much. That square-root law is the engine. It means the *per-day* rate of decay is small when there's lots of time left and large when there's little: each day removed from a 30-day option is a bigger fraction of its remaining √time than each day removed from a 60-day option. The front-month melt rate is steeper because it is closer to the cliff.

Make that concrete with the square-root rule itself. The at-the-money time value is roughly proportional to `S × sigma × √T`. Differentiate with respect to `T` and the per-unit-time decay scales as `1/√T` — it *blows up* as `T → 0`. With 60 days left (`√60 ≈ 7.75`) the decay rate is gentle; with 5 days left (`√5 ≈ 2.24`) it is more than three times faster; in the final day it goes near-vertical. This is why time decay is sometimes drawn as a curve that's almost flat for months and then plunges off a cliff in the last weeks. A calendar deliberately positions you on the *steep* part of one option's decay curve (the short front) while you own the *flat* part of another's (the long back). You are short the cliff and long the plateau. Every calm day, the cliff gives up more than the plateau, and the gap is yours.

A second way to see the same thing: the front and back start with different amounts of time value, but the *front gives up its share faster*. At inception the front has \$2.45 of pure time value (it's at-the-money, so all extrinsic) and the back has \$3.56. Thirty days later, if the stock hasn't moved, the front has *zero* time value left (it just expired) while the back still has \$2.45 — it has aged into being a fresh 30-day option. The front surrendered all \$2.45 of its time value over the month; the back surrendered only \$1.11 (from \$3.56 down to \$2.45). You were short the leg that lost \$2.45 and long the leg that lost only \$1.11, so the net time value moved \$1.34 in your favor over the month — which, not coincidentally, is exactly the +\$134-per-contract peak we'll compute for the tent. The differential decay and the tent's peak are the same number seen two ways.

#### Worked example: how much faster does the front decay?

Take a stock at \$100, risk-free rate 4%, implied volatility 20% for both expiries. Price two at-the-money \$100 calls through this series' Black-Scholes pricer — one with 30 days to expiry, one with 60.

- The **front (30-day) call** is worth **\$2.45 per share**, \$245 per contract. Its model theta is −\$15.90 per share per year, so its per-calendar-day decay is −15.90 ÷ 365 = **−\$0.0436 per share**, or **−\$4.36 per contract per day**.
- The **back (60-day) call** is worth **\$3.56 per share**, \$356 per contract. Its theta is −\$11.82 per year, or **−\$3.24 per contract per day**.

Notice the back month costs about 45% more (\$356 vs \$245) yet decays *slower* per day (−\$3.24 vs −\$4.36). That is the square-root law made concrete: 1.41× the value, but a gentler daily melt.

Now form the calendar: **short the front, long the back.** Your *net* theta is the front's decay working *for* you (you're short it, so you collect its melt) minus the back's decay working against you:

```
short front theta:  +$4.36 per day   (you collect the front's melt)
long  back  theta:  -$3.24 per day   (you pay the back's melt)
net position theta: +$1.12 per day   (you collect the difference)
```

The intuition: you are renting out a fast-melting ice cube and renting in a slow-melting one, and pocketing the difference in melt rates every day the stock sits still.

![Theta per day of a 30-day option versus a 60-day option as time elapses, the front bleeding faster](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-2.png)

The chart above plots each option's per-day theta as the calendar advances toward the front's expiry at day 30. Both curves are negative — both options are losing time value — but the front's curve bends *down* far harder as it approaches its expiry, while the back's barely budges (it still has another 30 days of life when the front dies). The shaded gap between them, widening toward the right, is exactly the differential decay the calendar harvests. The structure exists to own that gap.

### Why implied volatility has a term structure you can trade

The second pillar is **the term structure of volatility**: implied volatility is not one number, it is a *curve* across expiries. The market quotes a different implied vol for next week's options than for options three months out. Plot implied vol on the y-axis against time-to-expiry on the x-axis and you get the **term structure** — and it slopes.

In calm markets the curve usually slopes *up*: near-dated implied vol sits below far-dated. This is **contango**. The intuition is that over a longer horizon, more can go wrong — a recession, an election, a shock — so the market charges more annualized vol for distant expiries. In a panic the curve *inverts* — near-dated implied vol spikes above far-dated, because the fear is *right now* — and that's **backwardation**. (The full mechanics live in [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve); we use it here, we don't re-derive it.)

![Calm contango volatility term structure with the front tenor sold and a richer back tenor owned](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-6.png)

This matters for the calendar because **you are short the front month's implied vol and long the back month's.** When you build a calendar, you are not just trading time — you are trading the *shape of the IV curve* between two points on it. If the front-month IV is *rich relative to the back* (as it is around a known event, when the event-expiry option is jammed with one-day premium), selling the front and owning the back is selling expensive vol and buying cheap vol. If the curve later normalizes — the front crushes back toward the back — you profit on the shape change, independent of where the stock goes.

So a calendar is really two trades bundled into one structure: a *theta* trade (own the differential decay) and a *term-structure / vega* trade (own the back-month vol, sell the front-month vol). Most of the post is about keeping those two threads straight, because they can reinforce each other or fight each other depending on what implied vol does.

It's worth pausing on *when the two threads reinforce versus fight*, because that distinction is the difference between a good calendar and a bad one. The two threads **reinforce** when the front month is rich relative to the back — a steeply inverted, event-driven term structure. There, selling the front harvests both fast theta *and* expensive vol, and when the event resolves the front crushes while the back holds: theta and the vol move both pay you. The two threads **fight** when the front is *cheap* relative to the back — a steeply upward-sloping contango curve. There you're still collecting differential theta, but you've bought the expensive back-month vol and sold the cheap front-month vol, so if the whole curve shifts down (a general vol crush) your long back-month vega loses more than your theta collects. Same structure, opposite outcome, entirely because of *where on the term structure you entered*. Keep asking, before every calendar: is the front rich or cheap relative to the back? That one question tells you whether your theta and vega are rowing together or against each other.

## The payoff is a tent, not a hockey stick — and it must be drawn at front expiry

Here is the single most common confusion about calendars, and the reason their payoff diagram looks nothing like a vertical spread's.

For a vertical spread or a single option, you draw the payoff *at expiry* — the date everything settles into intrinsic value, the kinked hockey-stick line. That works because all the legs expire on the same day. A calendar's legs *do not expire on the same day.* The front dies first. At that moment, the back is still alive — it still has weeks of time value left, and a price that depends on the stock, on implied vol, and on the model. **You cannot draw a calendar's payoff as a simple intrinsic-value hockey stick, because at the only natural "expiry" date — the front's — one of your two legs is still a live, model-priced option.**

So the calendar's P&L picture must be drawn **at the front expiry date**, and the back leg's value at that date must be *computed from a pricing model*, not read off an intrinsic-value kink. When you do that, you get the **tent**.

### Why the shape is a tent

Walk through what the position is worth at front expiry as the stock lands at different prices. At front expiry, the short front option settles to its intrinsic value (you pay out `max(S − K, 0)` on a short call); the long back option is worth a model price with its remaining time still on the clock.

- **Stock far below the strike (say \$80).** The front call expires worthless — good, you keep its premium. But the back call, now deep out-of-the-money with only 30 days left, is also worth almost nothing. Both legs are near zero, so you've lost roughly the whole debit.
- **Stock far above the strike (say \$120).** The front call you're short is \$20 in-the-money — you pay out \$20 of intrinsic. The back call is worth about \$20 of intrinsic plus a little time value. The two nearly cancel; you're left with a small loss (you spent the debit, and the back's leftover time value doesn't quite cover it). Both deep-ITM legs move almost dollar-for-dollar, so the spread between them goes nearly flat.
- **Stock right at the strike (\$100).** The front call expires *exactly* worthless — you keep its entire premium. And the back call is now a fresh 30-day at-the-money option with maximum time value. This is the sweet spot: you've harvested all of the front's premium and the back is worth the most it can be relative to its strike. Maximum profit.

Connect those points and you get a tent: low on both wings, peaking sharply at the strike. The structure *wants the stock to sit still at the strike* through the front expiry, then it wants you to hold or sell the back month into whatever vol is left.

#### Worked example: building the tent and finding its peak

Build the calendar from the same inputs: stock \$100, strike \$100, 20% IV both months, r = 4%. Front is the 30-day call (\$2.45), back is the 60-day call (\$3.56).

```
net debit = back premium - front premium
          = $3.56 - $2.45
          = $1.11 per share  ($111 per contract)
```

That \$111 is the most you can lose: if both legs go worthless (stock collapses far below the strike), you're out the debit and no more. Now compute the position's value **at front expiry** (front = 0 days, back = 30 days remaining) across spot prices. The spread is worth `back_call_value(30d) − front_intrinsic`, and the P&L is that minus the \$1.11 debit:

| Stock at front expiry | Back call (30d) value | Front intrinsic owed | Spread value | P&L per contract |
|---|---|---|---|---|
| \$80 | \$0.00 | \$0.00 | \$0.00 | **−\$111** |
| \$95 | \$0.63 | \$0.00 | \$0.63 | **−\$48** |
| \$100 | \$2.45 | \$0.00 | \$2.45 | **+\$134** |
| \$105 | \$5.91 | \$5.00 | \$0.91 | **−\$20** |
| \$110 | \$10.43 | \$10.00 | \$0.43 | **−\$68** |
| \$120 | \$20.33 | \$20.00 | \$0.33 | **−\$78** |

The peak is **+\$134 per contract** right at the \$100 strike — you turned a \$111 debit into a position worth \$245 (the back month's fresh 30-day ATM value), a clean +\$134. The wings sag toward losses on both sides. The intuition: a calendar pays the most when the stock pins the strike at front expiry, and its worst case is simply the debit you paid.

That table *is* the cover figure. Look at it again: the green region around the strike is where the tent pokes above zero, and the red wings on both sides are where the stock drifted too far in either direction. There is a breakeven band — roughly \$96 to \$104 in this example — inside which the calendar prints. That band, not a single breakeven point, is what you're trading.

### The pin and the gamma sign

Because the calendar's peak sits at the strike at front expiry, the position has a strong *preference for the stock pinning the strike* — and that tells you its gamma sign. **Near the strike at front expiry, a calendar is short gamma.** Move away from the strike in either direction and the position loses value (you slide down the tent's slope); the curvature works against you. This is the price you pay for collecting that differential theta — and it should sound familiar, because [it is the same theta-gamma trade-off](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) that governs every premium-selling position. You cannot collect net theta without being net short gamma somewhere; the calendar collects it near the strike and pays for it with negative gamma near the strike.

There's a subtlety worth flagging: the short-gamma is concentrated *near the strike as the front expiry approaches*, because the front option's gamma explodes as it nears expiry while the back's stays tame. Early in the trade's life the net gamma is mild. In the final days before front expiry, if the stock is camped right on the strike, the short front gamma gets large and the position becomes twitchy — a pin can turn into a whipsaw. We'll return to this when we talk about managing the roll.

## The net Greeks: long vega, positive theta, short gamma — all at once

Now we assemble the position's full risk profile. This is where the calendar reveals why it's special: it is the structure that lets you be **long vega and positive theta simultaneously**, which a single option flatly cannot do (a long option is long vega but pays theta; a short option collects theta but is short vega). The calendar splits the difference by being short the small-vega front and long the big-vega back.

Let's compute each Greek for the 30/60-day \$100 calendar at entry (stock \$100, 20% IV, r = 4%), per contract pair.

#### Worked example: the four net Greeks at entry

Run both legs through the pricer and net them (short front contributes the negative of its Greek, long back contributes the positive):

```
DELTA  (shares of directional exposure)
  short front call delta:  -0.5343
  long  back  call delta:  +0.5484
  net delta:               +0.0141  ->  about +1.4 shares per contract pair (nearly flat)

GAMMA  (per $1 move, x100 shares)
  short front gamma:       -0.0693
  long  back  gamma:       +0.0488
  net gamma:               -0.0205  ->  -2.05 per contract pair (short gamma)

VEGA  (per 1 vol-point change, x100 shares)
  short front vega:        -11.40 per vol point per share unit... netted:
  long  back  vega:        +16.06
  net vega:                +4.66 per vol point  (long vega)

THETA  (per calendar day, x100 shares)
  short front theta:       +$4.36 per day  (collected)
  long  back  theta:       -$3.24 per day  (paid)
  net theta:               +$1.12 per day  (positive theta)
```

Read that block carefully, because it is the whole thesis of the structure:

- **Delta ≈ +1.4 shares — essentially flat.** At the strike, the two call deltas nearly cancel. The calendar is a *non-directional* trade at inception. (It does pick up delta as the stock moves away from the strike — more on that under diagonals.)
- **Gamma = −2.05 — short gamma.** As established, you're short curvature near the strike. Big moves hurt.
- **Vega = +4.66 per vol point — long vega.** The back month, with twice the life, has far more vega than the front, so the net is positive. **If implied vol rises, the calendar gains.** This is the defining feature.
- **Theta = +\$1.12 per day — positive theta.** The front's faster decay wins the daily melt race. **If nothing moves and vol holds, you collect.**

![Net Greeks of an at-the-money calendar shown as signed bars, positive theta and vega, negative gamma](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-5.png)

The intuition: a calendar is the rare position that pays you to wait (positive theta) *and* gains if fear rises (long vega) — but it makes you pay for both with negative gamma, so a large, fast move is its enemy.

That combination — collect theta *and* be long vega — is genuinely unusual. Most positive-theta positions (short straddles, condors, credit spreads) are *short* vega: they want vol to fall. The calendar wants vol to *rise* (or at least hold) while still collecting decay. That is why it is sometimes called a "long-vol-but-positive-carry" structure, and why it behaves so differently from the income strategies it's often lumped in with.

## The vol risk: a calendar is the opposite of a condor

The long-vega feature is also the calendar's signature *risk*, and it is the thing that catches new traders. Because you are net long vega — concentrated in the back month — **the calendar loses money if implied volatility falls, even if the stock behaves perfectly.**

This is exactly backwards from an [iron condor](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard), which is short vega and *wants* IV to fall. A condor profits from a vol crush; a calendar gets hurt by one. If you've internalized "income trades love a vol crush," the calendar will surprise you, because it's the one income-flavored structure that does not.

#### Worked example: the long-vega P&L when IV shifts

Hold the stock pinned at the strike and hit the calendar with an instant implied-vol shock that moves *both* months. The entry debit was struck at 20% IV. Mark the position immediately after the shock and compute the change:

| Implied vol after shock | Spread mark | P&L per contract |
|---|---|---|
| 16% | \$0.92 | **−\$19** |
| 20% (entry) | \$1.11 | \$0 (breakeven) |
| 24% | \$1.30 | **+\$19** |

A 4-vol-point *drop* costs you about \$19 per contract; a 4-vol-point *rise* gains you about \$19. That's the +\$4.66 vega doing its work (4 points × \$4.66 ≈ \$19). The P&L is nearly symmetric around the entry vol — a clean long-vega exposure.

![Calendar mark-to-market profit and loss versus implied volatility after a vol shock, rising with IV](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-4.png)

The chart drives it home: the line slopes up and to the right, crossing zero at the 20% entry vol. The red region to the left — where IV has fallen — is the vol-crush risk. The intuition: hold the stock still and the calendar is a pure long-vega bet; a falling IV bleeds it, a rising IV feeds it.

This is why **where in the term structure you put the calendar matters enormously.** If you build a calendar by selling an already-elevated front month and owning a calm back month, then a normalization of the curve — front IV crushing while back IV holds — is a *double* win: you collect the differential theta *and* the relative vol move goes your way. But if you build a calendar in a calm market and then a general vol crush hits both months equally, your long back-month vega bleeds and the trade loses even though the stock sat still. The calendar's vega is not abstract; it is a directional bet on the level (and shape) of implied vol over the back month's life.

### Event calendars: selling the crush

This is precisely Priya's trade, and it deserves its own treatment because it's the calendar's signature use case. An **event calendar** is built around a known volatility event — an earnings report, an FDA decision, an FOMC meeting, a jobs number — by **selling the expiry that contains the event and owning an expiry beyond it.**

The logic: the event-expiry option is jammed with one-day "expected move" premium, so its implied vol is grotesquely high. The option a month out also contains the event, but spread across many more calm days, so its implied vol is far tamer. You sell the rich one and own the cheap one. When the event resolves, the front-month earnings premium *crushes* — the uncertainty is gone, the time value collapses — while the back month, still pricing ordinary post-event vol, holds up. You harvest the difference. (We'll go much deeper into the event-vol crush mechanics in the forthcoming [trading event vol: earnings, FOMC, and the vol crush](/blog/trading/options-volatility/trading-event-vol-earnings-fomc-and-the-vol-crush); the calendar is one of the cleanest vehicles for that trade.)

#### Worked example: a term-structure normalization trade

Suppose the front month is rich because of an event — say its IV is 26% — while the back month is calm at 18%. You sell the front (priced at 26%) and buy the back (priced at 18%):

```
front 30d 100-call @ 26% IV  =  $3.14  (you receive)
back  60d 100-call @ 18% IV  =  $3.24  (you pay)
net debit                    =  $0.11  ($11 per contract)
```

The debit is tiny — only \$11 — because you sold a *rich* front month that nearly paid for the cheap back month. Now the event passes and the term structure normalizes: the front-month IV crushes to 18% and the back month firms to 20%. Mark the position immediately:

```
front 30d 100-call @ 18% IV  =  $2.22  (you buy it back)
back  60d 100-call @ 20% IV  =  $3.56  (you sell it / it's worth this)
spread mark                  =  $1.34
P&L = $1.34 - $0.11 debit    =  $1.23 per share  ($123 per contract)
```

You turned an \$11 debit into \$123 — a 10x return on the debit — purely on the curve normalizing, without the stock having to do anything. The intuition: when the front is artificially rich, the calendar's debit is cheap and the payoff from the crush is enormous relative to it.

Compare that to the calm-market calendar, where both months sat at 20% and the debit was \$1.11. The event calendar's edge is that you *entered when the front was overpriced*, so the structure's natural long-vega bias was offset by selling expensive front-month vol. That is the difference between a good calendar and a bad one: not the structure, but *where on the term structure you traded it.*

## Diagonals: a calendar with a directional tilt

A **diagonal spread** is a calendar where the two legs have *different strikes* as well as different expiries. (The name, again, comes from the chain: it moves both across expiries *and* down strikes, hence diagonally.) Everything about the calendar carries over — sell the front, own the back, harvest differential decay, long vega — but the strike offset **adds a directional delta** that a same-strike calendar doesn't have.

The standard bullish call diagonal: **buy the back-month at-the-money (or in-the-money) call, sell the front-month out-of-the-money call.** Because your long leg is closer to the money than your short leg, the position is net long delta — it leans bullish. The tent's peak slides *up* toward the short strike, so the trade now wants the stock to drift higher (toward the short strike) by front expiry, not just to pin where it started.

![Side-by-side comparison of a same-strike calendar and a strike-offset diagonal, the diagonal tent sliding upward](/imgs/blogs/calendars-and-diagonals-trading-time-and-term-structure-7.png)

#### Worked example: a bullish call diagonal vs the plain calendar

Same stock at \$100, 20% IV, r = 4%. Buy the 60-day \$100 call (\$3.56), sell the 30-day \$105 call (\$0.71):

```
buy  60d 100-call  =  $3.56  (long back, at-the-money)
sell 30d 105-call  =  $0.71  (short front, out-of-the-money)
net debit          =  $2.85  ($285 per contract)
```

The diagonal costs *more* than the plain calendar (\$285 vs \$111) because you sold a cheaper, further-out-of-the-money front and bought the same rich at-the-money back. Now the Greeks:

```
DELTA: long back 100-call (+0.55) + short front 105-call (-0.22) = +0.33  (bullish tilt)
       vs plain calendar net delta = +0.01  (flat)
```

The diagonal carries **+0.33 delta per contract pair** — about +33 shares of bullish exposure — versus the calendar's near-zero. And its tent peaks higher: run the value at front expiry across spot and the maximum lands near the \$105 short strike, not at \$100:

| Stock at front expiry | Back 100-call (30d) | Front 105 owed | P&L per contract |
|---|---|---|---|
| \$95 | \$0.63 | \$0.00 | **−\$222** |
| \$100 | \$2.45 | \$0.00 | **−\$40** |
| \$103 | \$4.35 | \$0.00 | **+\$150** |
| \$105 | \$5.91 | \$0.00 | **+\$306** |
| \$110 | \$10.43 | \$5.00 | **+\$258** |

The diagonal's tent peaks around \$105 — exactly the spot you want if you're bullish — and it actually *makes money* across a wide upper band, including spots above the short strike, because the back-month call keeps gaining as the stock rises. The intuition: a diagonal is a calendar with a directional opinion baked in — you still harvest differential decay and long vega, but now you also want the stock to drift toward your short strike.

The diagonal generalizes far beyond this one configuration. Shift the strikes the other way (sell a lower front, buy a lower back, on puts) for a bearish tilt. Widen the offset for more delta and a flatter tent. The famous **"poor man's covered call"** is just a deep-in-the-money long-dated call (the back leg, standing in for stock) against a short out-of-the-money front call — a deeply directional diagonal that mimics a covered call for a fraction of the capital. The structure is a dial: same-strike for pure time/vol, offset for direction, with the offset size setting how much directional tilt you take on.

The way to *think* about a diagonal is as a calendar with an extra lever. A plain calendar gives you a tent centered on the strike; the diagonal lets you (a) slide the tent's peak left or right by choosing where to place the short front strike, and (b) tilt the whole structure's delta by choosing how far apart the two strikes sit. Those two levers interact. Selling a further-out-of-the-money front collects less premium (a smaller theta engine) but slides the peak further toward that strike and adds more bullish delta. Selling a front close to the back's strike collects more premium (more theta) but keeps the peak and the delta near the calendar's neutral profile. There is no free lunch in the dial: every step toward more directional tilt trades away some of the calendar's defining delta-neutrality, and you start to take on the directional risk you were trying to avoid in the first place. The art is matching the offset to how much directional conviction you actually have — a little conviction, a small offset; a lot, a wide one; none, stay at the same strike.

One practical warning about diagonals that the Greeks above understate: because the long back leg and the short front leg sit at *different* strikes, they no longer cancel cleanly when the stock moves far. A same-strike calendar's deep-ITM legs track each other almost dollar-for-dollar (both are essentially stock), so the wings of its tent flatten out. A diagonal's mismatched strikes don't track as tightly, so a diagonal can develop a larger directional P&L — and a larger directional *loss* — than a calendar if the stock runs hard against the tilt. The "poor man's covered call" is the cautionary case: when the underlying gaps down, the deep-ITM long call loses delta cushion fast, and the small front-call premium you collected does almost nothing to offset a 15% gap in the stock. A diagonal is a calendar plus a directional bet, and you can lose on the directional bet the same way any leveraged long or short can.

## Strike and expiry selection: where you put the legs

The structure is simple; the parameter choices are where the skill lives. A few principles.

**Strike: build the tent where you expect the stock to be at front expiry.** A same-strike calendar peaks at the strike, so put the strike where you think the stock will pin. At-the-money calendars are the most common (maximum theta, maximum vega, symmetric). Out-of-the-money calendars (strike above spot for calls) are a bet the stock drifts *to* that strike — they're cheaper and carry a small directional lean even before you offset the back leg.

**Front expiry: the closer the front, the faster the theta but the steeper the gamma risk.** A 7-day front against a 37-day back has enormous differential decay (the front bleeds ferociously in its final week) but a vicious short-gamma profile near expiry — a small move blows you off the peak fast. A 30-day front against a 60-day back is gentler on both. Weekly-front calendars are a high-octane theta play; monthly-front calendars are the steady version.

**Back expiry: how much vega and slow decay you want to own.** A 30/90-day calendar owns far more back-month vega than a 30/45-day one — more long-vol exposure, more sensitivity to the back month's IV. If your thesis is "front-month IV is rich and will crush," you want enough back-month duration to *hold* its vol while the front collapses, so go further out. If you just want differential theta with minimal vega, keep the back close to the front.

**Double calendars: tent over a wider band.** A **double calendar** puts on two calendars at once — one at a strike below spot, one above (typically a put calendar below and a call calendar above). The two tents overlap into a broad plateau, so the position profits across a wider range of where the stock lands, at the cost of a larger debit. It's the calendar's answer to the iron condor: a non-directional, range-bound bet, but long vega instead of short. Use it when you expect the stock to stay in a band *and* you want IV to hold or rise — for example, ahead of an event where you expect the post-event range to stay contained but vol to stay bid.

## Managing the roll and the pin at front expiry

The calendar has a built-in decision point that the vertical spread doesn't: **what to do at front expiry, when one leg dies and the other is still alive.** This is the *roll*, and how you handle it is half the trade's P&L.

When the front expires, you have three live choices:

1. **Close the whole position.** Buy back any residual front value (usually near zero if it's out-of-the-money), sell the back month at its current mark, and book the result. Clean, simple, takes the tent's value as drawn.
2. **Roll the front.** Sell a *new* front-month option (the next expiry out) against your still-living back month, re-establishing the calendar and collecting fresh front-month premium. This turns a single calendar into a *repeated* theta harvest against one long back-month leg — the engine of strategies like the "poor man's covered call," where you keep selling fronts against a long-dated back for months.
3. **Keep only the back.** Let the front expire, keep the long back month as a now-outright long-vol, long-delta position. You'd do this if your thesis flipped from "harvest decay" to "I now want to be long this option outright."

**The pin risk.** As front expiry nears with the stock camped on the strike, the short front option's gamma explodes and the position gets jumpy. If the stock is sitting *exactly* at the short strike at expiry, you face **assignment uncertainty** — the front might or might not be exercised, and you can wake up with an unexpected stock position if it's a call you're short and it finishes a penny in-the-money. The professional habit is to *not let a short option sit at-the-money into the final hours of expiry* unless you're prepared to manage the assignment. Roll or close before the pin gets dangerous. (The mechanics of assignment and settlement live in [the options chain and contract mechanics](/blog/trading/options-volatility/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement); the calendar just makes pin risk especially live, because its whole edge is in pinning the strike.)

## Common misconceptions

Each of these is a half-truth that costs money when believed whole. The corrections are numbers, not opinions.

**Myth 1: "Calendars are theta plays."** They are *also* vega plays — and the vega is usually the bigger risk. Yes, the calendar collects roughly **+\$1.12 a day** in theta on our example. But it also carries **+\$4.66 of vega per vol point**. A modest 5-vol-point crush in the back month costs about 5 × \$4.66 ≈ **\$23 per contract** — more than three days of theta wiped out by a single repricing of implied vol. Traders who size a calendar purely off its daily theta, ignoring the vega, are mis-measuring the position the way Priya's colleague did when he put on calendars in a calm market and got run over by a general vol crush that left the stock untouched. A calendar is a theta-*and*-vega play; quote both or you're flying blind.

**Myth 2: "A calendar is non-directional, so the stock doesn't matter."** It's *delta*-neutral at inception, but it is very much *gamma*-negative — it cares enormously about *how far* the stock moves, just not which way. The tent peaks at the strike and sags to a loss on both wings. In our example a move to \$110 by front expiry — only 10% — turns the +\$134 peak into a **−\$68 loss**. The stock matters; it just matters through magnitude, not direction. "Non-directional" means "I have no view on up-vs-down," not "the stock can do anything."

**Myth 3: "The longer the back month, the better — more time value to keep."** More back-month duration means more *vega*, which cuts both ways. A 30/120-day calendar owns roughly twice the back-month vega of a 30/60-day one, so it gains more if IV rises *and loses more if IV falls.* If your edge is the differential theta and you have no strong view that back-month IV will rise, piling on back-month duration just loads up vega risk you didn't intend to take. The right back-month length is set by your *vol* thesis, not by "more time = more better."

**Myth 4: "You draw the payoff at the back-month expiry."** You draw it at the **front** expiry, with the back leg priced by a model as a live option. Drawing a calendar at the back expiry would show both legs as expired intrinsic-value hockey sticks that exactly cancel (same strike, same type) — a flat line at the debit, which would tell you the calendar can only lose. That's nonsense; it ignores that the front dies first and you *keep its premium*. The tent only appears when you mark the position at front expiry. Drawing it at the wrong date is the single most common way people convince themselves calendars don't work.

**Myth 5: "An event calendar profits from the earnings move."** It profits from the *vol crush*, and a big move can actually hurt it. The event calendar's edge is the front-month IV collapsing after the event resolves, *not* the stock jumping. Because the position is short gamma near the strike, a large gap *away* from the strike can slide you down the tent — the very move that "proves you right on volatility" can lose you money if it's big enough. In our \$11-debit normalization example, the win came from the *curve normalizing* (front IV 26% → 18%), not from any stock move; if the stock had gapped to \$115 the front intrinsic you owed would have eaten much of the gain. Sell the crush, but respect the gamma.

## How it shows up in real markets

**Earnings season, every quarter.** The cleanest real-world calendar is the earnings calendar, exactly as Priya traded it. Before a high-profile earnings report — think a mega-cap tech name reporting after the close — the weekly options that expire that Friday routinely price implied vols of 60–90%, while the monthly options a few weeks out price 30–40%. The ratio of front-to-back IV is a tradeable signal: when the front is more than roughly 1.5× the back, the term structure is steeply inverted around the event (backwardation driven by the single-day "expected move"), and a calendar that sells the front and owns the back is selling that inversion. After the print, the front-month IV mean-reverts violently — the "earnings crush" — and the calendar harvests it. Whole desks run nothing but pre-earnings calendars across the season.

**The VIX term structure as a regime gauge.** Zoom out from single stocks to the index, and the VIX term structure tells you which way the calendar wants to lean. In calm regimes the VIX futures and the SPX implied-vol term structure are in *contango* — near-dated below far-dated — and a long-dated-vs-short-dated index calendar is, in effect, long the back of an upward-sloping curve. In a panic the curve *inverts* into backwardation, near-dated spiking above far-dated, and the same calendar's relationship flips. Index-level vol traders read the term structure's slope (and its history) the way an equity trader reads a chart; the calendar is one way to express "I think the front is too rich relative to the back," which is a recurring, structural opportunity around scheduled macro events (FOMC, CPI, jobs). For the products and the mechanics of the curve, see [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve).

**The "Volmageddon" lesson for long-vol structures.** February 5, 2018 — the day the VIX closed at 37.32 after the inverse-VIX products imploded — is usually told as a story about *short*-vol blowing up. But it carries a quieter lesson for calendar traders: when a vol spike hits, it does *not* hit all tenors equally. The front month spikes far more than the back (the curve slams into backwardation), which is *bad* for a normal contango calendar (you're short the front, which just exploded) but *good* for an inverted-curve calendar. The general point: a calendar is a bet on the *relative* move of two tenors, and during a vol event the tenors decouple violently. A trader who modeled their calendar as "long vega, so a vol spike helps me" — without noticing they were short the very front month that spikes hardest — would have been badly surprised. The position's vega is in the *back* month; the *front* month is where the event-driven spike concentrates.

**The poor man's covered call, in retail portfolios.** Walk into any retail options community and you'll find the "PMCC" — a deep-in-the-money LEAPS call (a long-dated back leg, often a year or more out) against a steady program of selling 30-to-45-day out-of-the-money calls (the rolling front leg). It's a diagonal, dressed up as an income strategy. The long back leg stands in for owning 100 shares at a fraction of the capital, the rolling short fronts harvest theta, and the whole thing is a directional, long-vega diagonal that the trader rolls month after month. It's enormously popular precisely because it captures the calendar's positive-carry-plus-upside profile — and it blows up the same way every leveraged long does when the underlying gaps down and the deep-ITM back leg loses its delta cushion.

## The playbook

Here is how a desk actually puts on, monitors, and takes off a calendar or diagonal.

**The view.** A calendar is the right structure when you believe *all three* of: (1) the stock will stay near a strike through the near expiry (low realized move), (2) the front month's time decay will outrun the back's, and (3) back-month implied vol will hold or rise. The purest setup adds a fourth: the front month is *rich relative to the back* — an event-driven inversion — so you sell expensive vol and own cheap vol. If you only have conviction on time (not vol), keep the back month short to minimize vega; if your real edge is the vol-term-structure crush, that's an event calendar and you size it as a vol trade.

**The structure.** For a pure time/vol bet, same-strike at-the-money calendar: sell the front, buy the back, same strike. For a directional tilt, offset the strikes into a diagonal — long back closer to the money, short front further out, in the direction you lean. For a range bet, double calendar (a put calendar below, a call calendar above) for a wide plateau. Pick the front-expiry length by how much gamma risk you'll tolerate (closer front = more theta, more gamma danger near expiry); pick the back-expiry length by how much vega you want to own.

**The Greek profile to expect.** Long vega (concentrated in the back), positive theta, short gamma near the strike, roughly delta-neutral at the money (delta-tilted if it's a diagonal). Write these on your [risk dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) and watch all four — especially vega, which is the risk most likely to surprise you. For our base case: +\$1.12/day theta, +\$4.66/vol-point vega, −2.05 gamma, ~flat delta.

**Entry.** Pay the debit; that debit is your defined max loss. Prefer entering when the front month's IV is *elevated relative to the back* (positive term-structure carry working for you) rather than in a flat or backwardated curve where you'd be long vega into a possible crush. For event calendars, enter a few days before the event, once the front-month earnings premium has fully inflated but before the print.

**Monitoring.** Watch three things daily. (1) *Is the stock staying near the strike?* If it's drifting toward a tent wing, your gamma is bleeding you. (2) *What is back-month IV doing?* A falling back-month IV is the silent killer — your long vega is losing even if the stock is perfect. (3) *How close is front expiry, and is the stock pinning the strike?* As the pin approaches, short-front gamma spikes; be ready to roll or close before assignment risk gets live.

**Exit and the roll.** At front expiry, close the whole thing (take the tent value), roll the front to the next expiry (re-harvest against the same back), or keep only the back (convert to an outright long). Take profits when the tent has built most of its value — you don't need to ride to the last day, and the final days are where short-front gamma is most dangerous. For event calendars, the natural exit is the morning after the event: buy back the crushed front, sell or hold the back into the still-elevated post-event vol, and book the normalization.

**Sizing and invalidation.** Size by the debit (your defined max loss per contract) *and* by the vega, because a vol crush can lose you a multiple of a day's theta. Your view is invalidated when: the stock makes a large, fast move away from the strike (short gamma hurts, and the realized move exceeds what you sold); back-month IV crushes (your long vega loses); or the term-structure carry you entered on flips against you (the front gets *cheaper* relative to the back, the opposite of the normalization you wanted). If any of those fire, the structure's edge is gone — take the loss at the debit you defined and move on. The whole appeal of the calendar is that its worst case is the debit; honor that by not turning a defined-risk trade into an open-ended one by stubbornly rolling a losing structure into a falling-vol regime.

The calendar and the diagonal are, in the end, the options trader's tools for separating *time* and *term structure* from *direction*. A vertical spread trades direction at one expiry; a calendar trades the relationship between two expiries with direction held flat (or, in a diagonal, dialed in deliberately). They are how you express "the front month is too rich," "back-month vol is cheap," and "this stock will go nowhere fast" — three views that no single option can hold at once, but that one calendar holds naturally. Master the tent, respect the vega, and you have the precision instrument for trading the clock and the curve.

## Further reading & cross-links

Within this series:

- [Theta: Trading the Clock and the Price of Being Long Options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options) — the differential decay the calendar harvests, in full.
- [Vega: Your Exposure to Implied Volatility and the Vol of Vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — why the back month dominates the calendar's vol risk.
- [The Term Structure of Volatility: Contango, Backwardation, and the VIX Curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the curve the calendar trades between two points of.
- [Reading the Vol Surface Like a Trader: The 3D Map of Fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — the surface a calendar slices across in the time dimension.
- [The Net Greeks of a Position: Building Your Risk Dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — how to monitor a calendar's long-vega, positive-theta, short-gamma profile.
- [Gamma: The Greek That Bites — Curvature, Convexity, and the Toxic Short](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the short-gamma cost you pay near the strike.

Coming in this series:

- [Trading Event Vol: Earnings, FOMC, and the Vol Crush](/blog/trading/options-volatility/trading-event-vol-earnings-fomc-and-the-vol-crush) — the event-calendar trade in depth.
- [Butterflies, Ratio Spreads, and Broken Wings: The Precision Tools](/blog/trading/options-volatility/butterflies-ratio-spreads-and-broken-wings-the-precision-tools) — more strike-shaped structures for pinning the move.

For the pricing theory behind every number here:

- [The Volatility Surface as a No-Arbitrage Object](/blog/trading/quantitative-finance/volatility-surface) — why the term structure can't be arbitrary.
- [Options Theory: Pricing Fundamentals](/blog/trading/quantitative-finance/options-theory) — the Black-Scholes machinery that prices both legs.

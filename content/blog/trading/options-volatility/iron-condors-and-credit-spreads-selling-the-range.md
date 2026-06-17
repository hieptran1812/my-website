---
title: "Iron Condors and Credit Spreads: Selling the Range"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How the iron condor harvests the variance risk premium with defined risk, why its small-credit-large-loss math demands management, and the exit rules that keep a tail move from erasing months of gains."
tags: ["options", "volatility", "iron-condor", "credit-spreads", "premium-selling", "short-vega", "short-gamma", "theta", "variance-risk-premium", "risk-management", "probability-of-profit"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An iron condor is a bull put spread plus a bear call spread: four out-of-the-money legs that pay you a credit up front and let you keep it if the stock stays inside a range. It is the canonical defined-risk way to sell the variance risk premium — short gamma, short vega, long theta — and the entire game is management.
>
> - The structure is symmetric: sell an out-of-the-money put and an out-of-the-money call (the inner strikes, which bring the premium), buy a further-out put and call (the outer wings, which cap the loss). You profit if the underlying expires between the two short strikes.
> - The math is deliberately lopsided. Our worked example collects a **\$93.86** credit per lot but risks **\$406.14** to make it — a 4.33-to-1 risk/reward — paid for by a high probability of profit (about **73.7%**). Small wins, large rare losses: the same insurance-company shape as every short-vol trade.
> - The Greeks are the premium-selling fingerprint: roughly flat delta at the center, **short gamma** (a fast move hurts), **short vega** (a vol spike hurts even at a flat price), and **long theta** (time passing pays you, about **\$2.32 per day** per lot).
> - The one rule to remember: **manage the exit, never let a winner run to a max loss.** Take profit near half the credit, roll the untested side in when one wing is threatened, and close a breached spread for a partial loss before short gamma turns it into the full capped loss.

A trader had been running the same income strategy for the better part of two years. Once a month, with the S&P 500 sitting quietly in the middle of its recent range, the position went on: sell an out-of-the-money put spread, sell an out-of-the-money call spread, collect a few hundred dollars per contract, and wait. Most months the index drifted, the options decayed, and the credit was booked. The brokerage statement showed a tidy, upward-sloping equity curve — the kind that makes the strategy feel like it has been *solved*. Wins came in at roughly nine out of every ten months. The losses, when they came, were small and quickly recovered by the next two or three wins.

Then, on one of those quiet weeks, an inflation print landed hot, a central-bank speaker turned hawkish, and the index gapped down 4% over two sessions while the VIX — the market's 30-day implied-volatility index — jumped from the mid-teens to the low thirties. The short put spread, comfortably out of the money on Monday, was deep in the money by Wednesday. The position that had been quietly paying a couple of hundred dollars a month registered its full capped loss: more than four times the monthly credit, in a single move. Eight months of patient gains, gone in three days.

This is the defining tension of the iron condor, and the subject of this post. The structure is genuinely elegant — it lets you sell volatility with a *known* worst case, unlike a naked short option that can lose without limit. It harvests one of the most robustly documented edges in finance, the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt). But that edge is shaped like an insurance business: a steady drip of small premiums, punctuated by the occasional large claim. Understanding the iron condor means understanding both halves — how the four legs combine into a flat profit zone, and exactly why management is the difference between a durable income strategy and a slow-motion account blow-up.

![Iron condor profit and loss at expiry showing a flat profit zone between the short strikes and capped losses on both wings](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-1.png)

The chart above is the whole strategy in one picture. The blue line is the profit and loss of the position at expiration, plotted against where the stock finishes. In the middle — between the two short strikes at 94 and 108 — the line is flat at the maximum profit of **\+\$93.86** per contract: this is the range you are selling, and you keep the full credit anywhere inside it. Outside the breakevens at \$93.06 and \$108.94, the position loses money, but only down to a *capped* floor of **−\$406.14** on each wing, set by the long options you bought for protection. The shape is a plateau with two cliffs that level off. You win a little if the stock behaves, you lose a bounded amount if it does not, and the bet is that "behaves" is more likely than the market's option prices imply.

## Foundations: from a single credit spread to the four-legged condor

Before we can build a condor, we need the building block it is made of: the **credit spread**. And before that, we need to be precise about what it means to *sell* an option, because everything that follows hangs on it. If you have read [calls, puts, and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) this will be familiar territory viewed from the seller's chair; if not, we build it from zero here.

**An option is a contract, and you can be on either side of it.** A call option gives its *buyer* the right (not the obligation) to buy the underlying at a fixed strike price before expiry; a put gives its buyer the right to sell at the strike. The buyer pays a premium for that right. The *seller* — also called the writer — receives that premium and takes on the matching obligation: if the buyer of a call exercises, the seller must deliver the stock at the strike; if the buyer of a put exercises, the seller must buy the stock at the strike. The buyer has a right; the seller has an obligation and a fee for carrying it. When you *sell* an option you are, in effect, the insurance company: you collect a premium today in exchange for a promise to pay out if the world moves against you.

**Out-of-the-money (OTM)** means the option has no intrinsic value yet — the strike is on the "wrong" side of the current price for the holder. An OTM call has a strike *above* the spot price (the stock would have to rise to make it worth exercising); an OTM put has a strike *below* spot (the stock would have to fall). For a deeper treatment of where a strike sits relative to spot, see [moneyness and the strike](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying). The condor lives entirely in OTM territory: all four of its legs have strikes away from the current price, which is precisely why it pays a modest credit rather than a large one.

### The vertical credit spread

A **vertical spread** is two options of the same type (both calls or both puts) and the same expiry, at two different strikes. You are long one and short the other. When the option you *sell* is more expensive than the one you *buy*, you receive net cash to put the trade on — a **credit spread**. The two flavors:

- A **bull put spread**: sell a put at a higher strike, buy a put at a lower strike. You collect a credit and profit if the stock stays *above* your short put. It is a bet that the stock will *not* fall below your short strike. (We cover the full mechanics in [vertical spreads: defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk).)
- A **bear call spread**: sell a call at a lower strike, buy a call at a higher strike. You collect a credit and profit if the stock stays *below* your short call. It is a bet that the stock will *not* rise above your short strike.

In both cases the option you buy is further out of the money and therefore cheaper than the option you sell — so you net a credit. That further-out option is your **insurance on the insurance**: it caps how much you can lose. Without it, a sold put or call has effectively unlimited downside. With it, your maximum loss is the distance between the strikes (the "width") minus the credit you took in. This is the entire reason a defined-risk seller buys the wing: it converts an unbounded liability into a known, sizeable, but survivable one.

### Adding the two together

Here is the key idea. A bull put spread profits if the stock stays *above* its short put. A bear call spread profits if the stock stays *below* its short call. If you put both on at once — selling an OTM put spread *below* the current price and an OTM call spread *above* it — you have built a position that profits if the stock stays *between* the two short strikes. That combination is the **iron condor**.

![The iron condor built from a bull put spread plus a bear call spread, with the total shown as the sum of the two component curves](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-2.png)

The chart makes the addition literal. The green dashed line is the bull put spread on its own: flat-positive on the right (where the stock is well above the short put and both puts expire worthless, so you keep that side's credit), sloping down into a capped loss on the far left. The lavender dashed line is the bear call spread: flat-positive on the left, capped loss on the far right. The solid blue line is their sum — the iron condor. Adding two credit spreads that profit on opposite sides gives a structure with a *flat shelf in the middle*: the range where both spreads expire worthless and you keep both credits. The condor is not a new instrument; it is two credit spreads stapled together, and every number in it is just the sum of the corresponding numbers in its two halves.

Notice why the name fits. A condor is a bird with a wide wingspan; the payoff diagram has a flat "body" in the middle and two "wings" that drop and then level off. The "iron" prefix is options jargon meaning the structure uses *both* puts and calls (a plain condor uses only one type, four calls or four puts; the iron version uses puts on the downside and calls on the upside, which is more capital-efficient because the two short strikes are never both in the money at once).

It helps to see the four legs laid out on the strike ladder, because the *placement* of each leg is what makes the structure work.

![Four legs of an iron condor laid out on the strike ladder with the short strikes inside and the long protective wings outside](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-4.png)

The ladder reads from the spot price outward. The two **inner** strikes — the \$94 short put and the \$108 short call — are the ones you *sell*; they are closest to the money, so they carry the most premium, and they are the source of essentially all the credit. The two **outer** strikes — the \$89 long put and the \$113 long call — are the ones you *buy*; they sit further from the money, cost very little, and serve only as the cap on each side's loss. The geometry is symmetric by design: a put pair straddling the downside, a call pair straddling the upside, with the current price sitting in the gap between the two short strikes. Every dollar of the \$93.86 net credit comes from selling the inner pair; every dollar of loss protection comes from owning the outer pair. The whole structure is just "sell the near, buy the far, on both sides."

#### Worked example: building the condor leg by leg

Let us build the exact position in the figures, pricing every leg from the Black-Scholes model. (We treat the pricing model as given here; for the derivation see [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) and [options theory](/blog/trading/quantitative-finance/options-theory).) The setup: a stock trading at \$100, options 45 days to expiry, implied volatility 20%, risk-free rate 4%, no dividends. We pick our short strikes at roughly the **0.16 delta** — a common choice we will justify shortly — which lands the short put at the \$94 strike and the short call at the \$108 strike. We make each wing **5 points wide**, so we buy the \$89 put and the \$113 call. Pricing the four legs:

- Sell the \$94 put for **\$0.62** per share.
- Buy the \$89 put for **\$0.11** per share.
- Sell the \$108 call for **\$0.58** per share.
- Buy the \$113 call for **\$0.15** per share.

The bull put spread brings in `\$0.62 − \$0.11 = \$0.51` per share. The bear call spread brings in `\$0.58 − \$0.15 = \$0.43` per share. The total **net credit** is `\$0.51 + \$0.43 = \$0.94` per share, which on the standard 100-share contract multiplier is **\$93.86 per lot**. That cash hits your account the moment the trade fills. The intuition: you have sold two pieces of out-of-the-money insurance and bought two cheaper pieces further out to cap your liability, and the difference is what you keep.

### Max profit, max loss, and the two breakevens

The four numbers that define any condor follow directly from the credit and the width.

**Maximum profit** is the net credit, full stop. It is realized if the stock finishes anywhere between the two short strikes at expiry, because then all four options expire worthless and you simply keep what you collected. In our example, max profit = **\+\$93.86** per lot, earned whenever the stock lands between \$94 and \$108.

**Maximum loss** is the width of one wing minus the net credit, times the multiplier. Only *one* wing can be breached at expiry — the stock cannot be both below \$89 and above \$113 at the same time — so the loss is capped at one width, not two. Each wing is 5 points wide, so:

```
max loss = (width − net credit) × 100
         = (5.00 − 0.9386) × 100
         = $406.14 per lot
```

This is the worst case, reached if the stock blows past \$89 on the downside or \$113 on the upside. Below \$89, your short \$94 put and long \$89 put are both deep in the money and their values move together, locking the loss at exactly the 5-point difference minus the credit. The long wing has done its job: it stopped the bleeding at \$406, not infinity.

**The breakevens** are where the payoff line crosses zero. On the downside, you lose the credit one-for-one as the stock falls below the short put, so the lower breakeven is the short put strike minus the credit: `94 − 0.94 = $93.06`. On the upside, the upper breakeven is the short call strike plus the credit: `108 + 0.94 = $108.94`. Between these two prices you make money at expiry; outside them you lose. The profit zone is therefore `108.94 − 93.06 = 15.88` points wide — almost a 16% range the stock can wander in and still leave you whole.

#### Worked example: the risk/reward asymmetry made explicit

Stare at those numbers together, because they are the crux of the whole strategy. You collected **\$93.86** and you are risking **\$406.14** to keep it. The ratio is `406.14 / 93.86 ≈ 4.33`, meaning you are putting up \$4.33 of risk for every \$1 of reward. If you only ever looked at the risk/reward ratio, you would never do this trade — it looks like picking up dimes in front of a bus.

What makes it sane is the *probability* attached to each outcome. You keep the full \$93.86 whenever the stock stays inside the breakevens, and we will compute shortly that this happens about **73.7%** of the time. You lose the maximum only when it travels outside them, which is the remaining tail. The expected-value calculus is not "risk \$406 to make \$94"; it is "win \$94 about three times for every one time you give back something up to \$406." A simple sketch: in four trades you might win \$93.86 three times (`+\$281.58`) and take a loss the fourth. As long as that loss is smaller than \$281.58 *on average* — which, with management, it usually is — the strategy carries positive expectancy. The asymmetry in the payoff is *deliberately* offset by an asymmetry in the probabilities. The whole craft is keeping that balance from tipping over, which is why management is not optional.

## The Greeks: why a condor is the canonical premium-selling structure

A payoff diagram tells you what happens *at expiry*. But you live with the position day to day, *before* expiry, and what you feel during that time is the Greeks — the sensitivities of the position's mark-to-market value to the things that move: price, the rate of price change, volatility, and time. The iron condor has a Greek profile so characteristic that it functions as a fingerprint of the entire short-volatility family. Let us read it.

![Net Greeks of the iron condor shown as signed bars: near-zero delta, short gamma, short vega, and long theta](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-3.png)

The bar chart shows the four net Greeks of our condor at the center of its range (stock at \$100), computed by summing each leg's contribution with the right sign — plus for the long options, minus for the short ones. Read each bar:

**Delta ≈ 0 (flat).** Delta is the position's sensitivity to a \$1 move in the stock. A symmetric condor centered on the spot has its put-side delta (positive, because short puts are bullish) almost exactly cancel its call-side delta (negative, because short calls are bearish). Our net delta is about **\+1.3 shares-equivalent** per lot — essentially flat. The condor is *not* a directional bet. You do not care, at the center, whether the stock ticks up or down a little; you care whether it stays in the range. This is why a condor is a *non-directional* income trade: you are selling the *amount* of movement, not its sign. (For the full treatment of delta as the hedge ratio, see [delta: direction exposure](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio).)

**Gamma < 0 (short gamma).** Gamma is the rate at which delta *changes* as the stock moves — the curvature of the position. Our net gamma is about **−4.3** per lot, and the negative sign is the dangerous one. Being **short gamma** means that as the stock moves *against* you, your delta moves against you *too*, accelerating the loss. Sell a put spread and watch the stock fall toward it: your once-flat delta turns increasingly negative (short, i.e. positioned for further falls), so each additional point down hurts more than the last. Short gamma is the mathematical signature of "the trade gets worse the faster it goes wrong." It is also why short gamma is at its most vicious near expiry, when gamma spikes — a point we return to. (The full anatomy of short gamma and why it bites is in [gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short).)

**Vega < 0 (short vega).** Vega is the position's sensitivity to a change in *implied* volatility — the market's forward-looking estimate of movement, baked into option prices. Our net dollar-vega is about **−\$10.56 per +1 volatility point** per lot. The negative sign means that if implied volatility *rises*, the position *loses* — even if the stock has not moved at all. You sold options; options get more expensive when implied vol rises; the cost to buy them back goes up; you are underwater on the mark. This is the cruelest surprise for new condor sellers: you can be perfectly right about direction (the stock is sitting still, right in your range) and still show a paper loss because fear spiked and lifted every implied vol on the board. (For the mechanics of vega and the "vol of vol," see [vega: your exposure to implied volatility](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol).)

The short-vega bite is worth seeing as a picture, because it is the one risk that does not require the stock to move at all.

![Iron condor open profit and loss at two implied volatility levels, showing the whole curve drops when implied volatility jumps even at a flat stock price](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-6.png)

The chart plots the same condor's *open* (mark-to-market) profit and loss against the stock price, with 30 days still to run, at two implied-vol levels: the entry vol of 20% in green, and a shocked 35% in red. The vertical gap between the two lines is pure vega. Hold the stock perfectly still at \$100 and let implied vol jump 15 points: the position swings from a comfortable **\+\$38** open gain (the green line, where time decay had been working for you) to a **−\$82** open loss (the red line) — a roughly \$120-per-lot hit *with the stock price unchanged*. The whole P&L sheet sinks. That is what it means to be short vega: a spike in the market's fear gauge re-prices your short options upward, and the cost to close them rises, even when the underlying you were betting on stays exactly where you wanted it. New sellers who only watch the stock price are blindsided by this; the experienced seller watches implied vol as closely as price.

#### Worked example: the vol-spike loss at a flat price

Make the short-vega bite concrete with the numbers behind the chart. Our condor was sold at 20% implied vol. Suppose, two weeks later, the stock is still pinned at \$100 (you were right about direction) but a scare hits and implied vol jumps to 35%. Re-pricing all four legs at the higher vol and 30 days remaining, the net cost to buy the condor back has risen by about \$1.20 per share. So instead of the gentle theta gain you should have banked for two weeks of a quiet stock — about \+\$38 per lot of open profit at the unchanged 20% vol — your position now marks at roughly **−\$82** per lot. The vol spike alone moved you about **\$120 per lot** in the wrong direction, *despite the stock being exactly where you wanted it*. The intuition: a condor is a bet that the world will be calm, so the world *becoming visibly less calm* — even before any actual movement — is already a loss on your books.

**Theta > 0 (long theta).** Theta is the position's sensitivity to the passage of time — how much its value changes as one calendar day ticks by, all else equal. Our net theta is about **\+\$2.32 per day** per lot, and the *positive* sign is the engine of the whole trade. You sold options; options lose value as they approach expiry (an out-of-the-money option with less time left has less chance of finishing in the money); so every day that passes with the stock inside your range transfers a little value from the option buyers to you. Time is your ally. The condor *bleeds in your favor*. (For why options decay and how theta accelerates, see [theta: trading the clock](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options).)

Put the four together and you have the canonical premium-selling fingerprint: **flat delta, short gamma, short vega, long theta.** That combination *is* what it means to be a short-volatility income trade. You are paid by the clock (theta), you are hurt by realized movement (short gamma) and by rising fear (short vega), and you have deliberately neutralized your exposure to direction (flat delta) so the bet is purely on *how much the stock moves, not which way*. Compare this to the net-Greek dashboard we built for a whole book in [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — the condor is the cleanest single-position expression of that short-vol profile.

### You are selling the variance risk premium, with defined risk

Why does this trade have positive expectancy at all? Because of the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt). Across decades of S&P 500 data, *implied* volatility (what option sellers charge) runs persistently *above* the *realized* volatility (what actually shows up) — on the order of **+3.7 volatility points** on average, because buyers of protection are structurally willing to overpay for it. When you sell options, you are charging that inflated implied vol; if the stock then realizes *less* movement than implied — which it does most of the time — you keep the difference. The condor is one of the most natural ways to harvest this premium, because it is short vega (you profit when realized vol comes in below the implied vol you sold) while being defined-risk (the long wings cap your tail). It is the same insurance business as a short straddle, but with a seatbelt.

The relationship to [implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) is exact. The condor profits when realized volatility comes in below the implied volatility embedded in the options you sold. If the stock chops around inside your range, realized vol is low and your short options decay; if it makes a large move, realized vol is high and your short options pay off against you. The condor is, at its core, a bet that *the market's option prices are charging more for movement than the stock will deliver.* That is the variance risk premium, expressed as a position.

## Strike selection: delta, width, and days to expiry

A condor is not a single trade but a family of trades, parameterized by three choices: how far out you place your short strikes, how wide you make your wings, and how many days to expiry you sell. Each choice trades one good thing against another.

### Where to put the short strikes: the delta-wing method

The most common professional approach is to select short strikes by their **delta**, not by their dollar distance from spot. Delta, for an out-of-the-money option, doubles as a rough approximation of the *probability the option finishes in the money* by expiry. A 0.16-delta short put has roughly a 16% chance of being in the money at expiry, which means roughly an 84% chance of expiring worthless and letting you keep that side's premium. The popular **0.16 delta** target is not arbitrary: in a lognormal model it corresponds to about one standard deviation of the expected move, so selling the 0.16-delta strikes on each side is, loosely, "selling the one-standard-deviation range." That is why our short strikes landed at \$94 and \$108 — those are the ~0.16-delta strikes for a 20%-vol, 45-day option on a \$100 stock.

Using delta to pick strikes is the disciplined way to standardize a strategy across underlyings of different prices and volatilities. A \$50 stock and a \$500 stock have wildly different dollar moves, but their 0.16-delta strikes encode the same *probabilistic* distance. Selling by delta means you are always selling roughly the same odds, regardless of the ticker.

### Probability of profit versus credit: the fundamental trade-off

Here is the central design tension. The further out you place your short strikes (the lower their delta), the *wider* your profit zone and the *higher* your probability of profit — but the *less* credit you collect, because options far out of the money are cheap. Push the strikes in closer (higher delta) and you collect more, but your profit zone narrows and your odds of keeping it all drop. You cannot have a high probability of profit *and* a large credit; the market prices them against each other.

![Probability of profit versus credit collected, showing higher probability comes with smaller credit as short strikes move farther out](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-5.png)

The curve quantifies the trade-off exactly. Each point is a 5-wide condor with its short strikes set by a delta target, from a far-out 0.10 delta down to an aggressive 0.40 delta. At the 0.10-delta end (top left), the profit zone is wide and the probability of profit is about **82%**, but the credit is only about **\$55** per lot. At the 0.40-delta end (bottom right), you collect about **\$266** per lot, but the probability of profit has fallen to roughly **49%** — a coin flip. The amber dot is our worked example at 0.16 delta: about **\$94** credit for a **73.7%** probability of profit. There is no free lunch on this curve; you are choosing a point on a frontier where moving toward more income necessarily means accepting worse odds.

#### Worked example: computing the probability of profit

Probability of profit (POP) for a condor is the chance the stock finishes *between the two breakevens* at expiry. We can estimate it from the same lognormal model that prices the options. Under the model, the log of the stock price at expiry is normally distributed, with a mean reflecting the drift and a standard deviation of `σ × √T`. With `σ = 20%` and `T = 45/365 ≈ 0.123` years, the standard deviation of the log-return is `0.20 × √0.123 ≈ 0.0702`, or about 7% over the 45 days.

The breakevens are \$93.06 and \$108.94. Converting to log-distances from the \$100 spot: the lower breakeven is `ln(93.06/100) ≈ −0.0719` and the upper is `ln(108.94/100) ≈ +0.0856`. Measured in standard deviations (and adjusting for the small risk-neutral drift), the lower breakeven sits about 1.0 standard deviation below and the upper about 1.2 above. The probability of landing between them — the area of the normal distribution between those two z-scores — works out to about **73.7%**. So in the model, you keep at least part of your credit (finish inside the breakevens) about three times out of four, and you take some loss the remaining one time in four. That number, paired with the 4.33-to-1 risk/reward, is the entire statistical character of the trade. The intuition: a condor is a high-win-rate, low-payout bet whose survival depends on the rare losing month being managed down, not allowed to run.

### Wing width

The wing width — the distance from the short strike to the long strike — sets your maximum loss and your capital requirement. A wider wing collects slightly more credit (the long option is cheaper the further out it is) and gives a marginally better credit-to-risk ratio, but it commits more capital because your max loss is larger. A 5-wide wing on our example risks \$406; a 10-wide wing would risk closer to \$900 but collect only a little more credit. Most retail-scale condors use wings narrow enough to keep the max loss to a tolerable fraction of the account, accepting the slightly worse credit ratio in exchange for a smaller, more sizeable worst case. The wing is the seatbelt; how tight you cinch it is a capital-and-comfort decision.

### Days to expiry

Days to expiry (DTE) controls how fast theta works and how violent gamma becomes. Selling further-dated options (say 45 days, as in our example) gives slower, steadier decay and gentler gamma — the position is more forgiving of a move because there is still time for the stock to come back. Selling very short-dated options (a week or less) maximizes the *rate* of theta but supercharges gamma: near expiry, a small move in the stock swings the position's value violently because the options are right at the knife's edge of in-the-money or worthless. The common professional compromise is to *sell* around 30–45 DTE and *close* around 21 DTE or at a profit target, deliberately stepping out before the gamma near expiry turns toxic. You harvest the meaty part of the decay curve and leave the dangerous tail to someone else.

## Managing the trade: the part that actually matters

Everything to this point is setup. The reason iron condors can be a durable income strategy *or* a slow account-killer comes down entirely to management. A condor put on and forgotten until expiry is a fundamentally different — and worse — bet than the same condor actively managed. Here is the playbook for handling it once it is live.

![Management decision flow for an iron condor: take profit at half the credit, roll the untested side when one wing is tested, or close for a partial loss when a wing is breached](/imgs/blogs/iron-condors-and-credit-spreads-selling-the-range-7.png)

The decision tree has one entry point and three branches, depending on what the stock does after you are in. The whole thing is organized around a single discipline, stated in the amber box at the bottom: *do not let a winner turn into a max loss.*

### Branch one: take profit at ~50% of the credit

The most important and most counterintuitive rule: **do not hold a condor to expiry to capture the last dollar of credit.** Instead, close the position once you can buy it back for about *half* what you sold it for — locking in roughly 50% of the maximum profit. Why give up the other half? Because the last half of the credit is the *slowest and riskiest* to earn. By the time a condor has decayed to half its value, you have collected most of the easy theta and you are now holding a position with rising gamma into expiry, exposed to a tail move for an ever-shrinking remaining reward. Closing at 50% frees your capital to redeploy into a fresh, full-credit condor and resets your gamma risk to a gentler level. Over many trades, the higher *turnover* of taking quick partial profits beats the slower grind of squeezing every condor to expiry, *and* it sidesteps the worst of the gamma risk.

#### Worked example: the 50%-profit-take math

Take our \$93.86-credit condor. The 50% target is to close it once we can buy it back for roughly half the credit — about \$0.47 per share, locking in **\$46.93** per lot. How long does that take if the stock cooperates and sits at \$100? Re-pricing the same four legs as time passes, with the stock flat:

- With 45 days left (entry): position worth the full credit, 0% captured.
- With 35 days left: the position has decayed to capture about **\$24.80** (26% of max).
- With 25 days left: about **\$52.31** captured (56% of max) — *past* the 50% target.

So in this benign path, the 50% profit target is hit around 25 days to expiry, roughly three weeks into a 45-day trade. You close, book \$46.93, and you are out — *before* the final stretch where gamma turns sharp. Compare that to holding to expiry: you would earn the remaining \$46.93, but you would carry the position through the most dangerous two weeks of its life to do it. The intuition: the second half of a condor's credit is rented from the gamma gods at a steep premium, and the disciplined seller declines to pay it.

### Branch two: roll the untested side in

Suppose the stock does not sit still — it drifts toward one of your short strikes. Say it rallies from \$100 toward \$106, pressing on your call side. The call spread is now "tested" (the stock is approaching the short \$108 call), while the put side is now far away and nearly worthless. The standard adjustment: **roll the untested (put) side up, closer to the new price, to collect more credit and recenter the position.** You buy back the now-worthless original put spread for almost nothing and sell a new, higher put spread that brings in fresh premium. The extra credit does two things: it widens your overall breakevens and it *reduces your maximum loss on the tested side*, because every dollar of new credit comes straight off the capped loss.

Crucially, you roll the *untested* side, never the tested side. Rolling the tested side "out" to a worse strike to collect more credit is the classic mistake — it adds risk exactly where the danger is, doubling down on a losing leg. The disciplined move is to harvest premium from the *safe* side, which is now cheap, to subsidize the threatened side.

#### Worked example: rolling the untested put side

The stock has rallied from \$100 to \$106 with 25 days left. The call side is being tested, but the original \$94/\$89 put spread is now worth almost nothing — you could buy it back for about \$0.02 per share. Meanwhile, you can sell a new, higher put spread, the \$100/\$95, which at the \$106 spot brings in about **\$0.29** per share. The roll: pay \$0.02 to close the old put spread, collect \$0.29 on the new one, for a net *additional* credit of about \$0.28 per share — **\$27.78** more per lot.

That \$27.78 is not just income; it directly shrinks your worst case. Your total credit is now `\$93.86 + \$27.78 = \$121.64`, so if the call side is ultimately breached, your max loss falls to `(5.00 − 1.2164) × 100 = \$378.36` per lot, down from the original \$406.14. You have traded a little bit of newly-collected put-side risk (the new put spread is closer to the money than the old one) for a meaningfully smaller call-side max loss and a wider downside breakeven. The intuition: when one wing is under pressure, you mine the calm wing for premium and use it to buy down the loss on the wing that is actually at risk.

### Branch three: just take the loss

Sometimes neither patience nor a roll is enough — the stock blows clean through a short strike and keeps going, often on a gap or a vol spike. This is the moment that defines whether the strategy survives. **Close the breached spread for a partial loss before short gamma drags it to the full capped loss.** If your short put spread is \$94/\$89 and the stock is at \$91 and falling, you might close it for a \$200 loss now rather than ride it to the \$406 max loss at expiry. Taking a \$200 loss feels bad; taking a \$406 loss feels much worse, and short gamma near expiry means the position can travel from one to the other fast.

This is the literal meaning of "do not let a winner turn into a max loss." A condor that was up \$50 a week ago and is now down \$200 is still salvageable — you exit, take the bounded hit, and live to put on next month's trade. The seller who *hopes* it comes back, watching short gamma compound the loss into expiry, is the one who gives back eight months of credits in three days. The defined-risk structure caps your loss in the *worst* case; active management caps it well below that in the *common* case.

### The gamma risk into expiry

One more reason management trumps holding to expiry: gamma *explodes* near the strikes as time runs out. Recall that being short gamma means your delta turns against you as the stock moves. Near expiry, with the stock sitting close to a short strike, that effect goes parabolic — the position can swing from near-max-profit to near-max-loss on a single day's move, because the short option is teetering right at the boundary between worthless and in-the-money. This is the "pin risk" zone, and it is precisely why the 50%-profit and 21-DTE-exit rules exist: they get you out *before* gamma becomes unmanageable. Holding a condor into the final week, with the stock near a short strike, is volunteering for the most violent part of the entire trade for the smallest remaining reward. (The mechanics of why short gamma is at its most toxic near expiry are covered in depth in [gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short).)

## The iron butterfly: the condor's aggressive cousin

A close relative deserves a brief mention. An **iron butterfly** is structurally the same idea — a put spread plus a call spread — but with the two short strikes placed *at the money* (at the current price) rather than out of the money. You sell an at-the-money straddle (the \$100 put *and* the \$100 call) and buy protective wings further out. Because at-the-money options are far more expensive than out-of-the-money ones, the butterfly collects a much larger credit. But because the short strikes are right at the current price, the profit zone is narrow and centered exactly on spot: the stock has to stay *very* close to where it started for you to keep most of the credit.

#### Worked example: the butterfly versus the condor

On the same \$100 stock, 45 days, 20% vol, an iron butterfly that sells the \$100 straddle and buys the \$90 put and \$110 call (10-wide wings) collects a credit of about **\$5.09 per share — \$509 per lot**, more than five times the condor's \$94. But its breakevens are only `100 ± 5.09 = $94.91 and $105.09`, a profit zone barely 10 points wide, and its probability of profit is just about **53%** — roughly a coin flip versus the condor's 73.7%. The butterfly is the condor turned aggressive: far more credit, far narrower range, far lower odds. You use it when you have a strong conviction the stock will *pin* near a specific price (often the current price into an expiry); the condor is the more relaxed bet that the stock will merely stay *somewhere in a range*. Same machinery, very different risk appetite. The intuition: the butterfly sells the *peak* of the option-price tent for a big premium and bets the stock sits on the pin; the condor sells the gentler slopes for less and bets only on the range.

### Condor versus strangle: defined versus undefined risk

It is worth contrasting the condor with its undefined-risk sibling, the **short strangle** — selling an out-of-the-money put and an out-of-the-money call *without* buying the protective wings. The short strangle collects a *larger* credit than the condor (you are not spending anything on the long wings) and has a wider profit zone. But its loss is *unbounded*: there is no long option to cap the downside, so a large move can produce a loss many times the credit, in principle without limit on the upside and limited only by the stock going to zero on the downside. The iron condor is, precisely, a short strangle with the tail amputated: you give up some credit to buy the two wings, and in exchange you convert an open-ended liability into a known, capped one. For anyone without a large balance sheet and an iron stomach, the defined-risk condor is the responsible way to express the same view. We dig into the undefined-risk long side of this — buying the strangle — in the forward-looking post on [straddles, strangles, and the long volatility bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet).

## Common misconceptions

**"A 73.7% probability of profit means this is a high-probability winning strategy, so it's safe."** This is the most dangerous misread, because it confuses *win rate* with *expectancy*. Yes, you win about 74% of the time — but you win \$93.86 and you can lose up to \$406.14. A simple expectancy check, unmanaged and held to expiry: if you win the full credit 74% of the time and lose the full max loss 26% of the time, the expected value is `0.737 × \$93.86 − 0.263 × \$406.14 ≈ \$69.18 − \$106.82 = −\$37.64` per trade. *Negative.* The raw, unmanaged condor at these strikes actually has negative expectancy in the lognormal model — the variance risk premium is what tilts the *real-world* odds in your favor (realized vol comes in below implied), and *management* is what turns the gross edge into a net one by cutting the losing trades short of their max. A high win rate with a worse-than-proportional payout is not automatically profitable; the arithmetic, not the win rate, decides.

**"Defined risk means I can't get badly hurt."** Defined risk caps a *single* condor's loss, but it says nothing about *how many* condors you hold or *how correlated* they are. A trader running ten index condors at once, all short the same market, has a defined loss *per position* and a thoroughly *undefined* loss as a portfolio when the index gaps and all ten breach together. The \$406 cap is real for one lot; ten lots that all go to max loss in the same crash is \$4,060, and the correlation is exactly 1.0 in the scenario that matters. Defined risk is a property of the structure, not of your book. Sizing — covered in [position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — is what bounds the portfolio.

**"If the stock stays in my range, I make my max profit — so I just need to be right about the range."** Not before expiry, you don't. The condor's *expiry* payoff is flat in the range, but its *mark-to-market* value swings with implied volatility the whole time you hold it. If fear spikes and implied vol jumps, you can be dead center in your range and still show a loss on the screen, because the cost to buy back your short options has risen. You only realize the full credit if you hold to expiry *and* the stock finishes in the range — and holding to expiry is exactly what the management rules tell you not to do. Being right about the range is necessary but not sufficient; you also have to survive the vol-driven drawdowns along the way.

**"The wings are a waste of money — I'd collect more by selling naked."** This is true right up until the day it bankrupts you. The wings cost you a few cents of credit per leg, and in exchange they convert an unbounded loss into a \$406 one. In our example, dropping the wings would lift the credit from \$93.86 to maybe \$118, a ~26% boost — in return for unlimited downside. The wings are the cheapest insurance you will ever buy, priced at a fraction of the tail they protect against. The traders who "save" the wing cost are the ones who become cautionary tales when a 5-sigma move arrives, which, in markets with fat tails, it does far more often than a normal distribution predicts.

**"Iron condors are a passive, set-and-forget income strategy."** The set-and-forget version is the *worst* version. As we computed, the unmanaged condor held to expiry can carry negative expectancy. The edge comes from active management: taking profits at 50%, rolling the untested side, and cutting breached spreads before they max out. A condor is not a bond that pays a coupon; it is an insurance contract you must *manage* like an underwriter manages a book of policies — adjusting exposure, taking gains, and paying small claims promptly to avoid large ones.

## How it shows up in real markets

**The monthly index condor and the gap-down month.** The most common live iron condor is the monthly S&P 500 (SPX) or related-index condor, sold around 30–45 days out, short strikes near the 0.16 delta. In a typical year, this trade wins most months: the index grinds, implied vol stays elevated relative to what realizes, and the condors decay to their profit targets. The equity curve looks smooth and the Sharpe ratio looks enviable. Then a month arrives like our opening hook — a hot inflation print, a hawkish central-bank pivot, a credit scare — and the index gaps several percent while the VIX doubles. The short put spread, comfortably OTM days earlier, is breached, *and* the surviving call spread loses value too because the vol spike lifted its mark. The unmanaged condor takes its full \$406 (per lot), and the smooth equity curve takes a cliff. This is the variance-risk-premium tail in the flesh: months of carry, one large claim. The traders who survive it are the ones who closed breached spreads on the way down instead of hoping for a bounce.

**February 2018, "Volmageddon."** On February 5, 2018, the VIX closed at **37.32**, having roughly doubled in a single session from the high teens. Any short-volatility position — condors, strangles, short-vol exchange-traded products — that was sitting in what *had been* a quiet range was suddenly underwater on a brutal combination of a sharp index drop (short gamma) and an implied-vol explosion (short vega). Condor sellers who were short the affected expiries saw their positions swing toward max loss as both Greeks worked against them at once. The lesson was not that the structure failed — the defined-risk wings did exactly their job, capping the loss — but that *several* condors held simultaneously, all short the same market, all breached together, can sum to a portfolio loss far larger than any single position's cap suggested. The structure was sound; the *sizing* and the *correlation* were the problem.

**The earnings-week condor and the vol crush.** A different and friendlier use: selling a short-dated condor on a single stock *into* an earnings announcement, to harvest the [implied-vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush). Before earnings, implied vol on a stock's options is jacked up to price the expected jump; the moment the report drops and the uncertainty resolves, implied vol collapses. A condor sold the afternoon before earnings, with short strikes outside the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options), profits from that vol crush — *provided* the stock's actual move stays inside the range. The risk is exactly the asymmetry we have been discussing in concentrated form: if the stock jumps more than the expected move, the condor's wing is breached on a single overnight gap, with no chance to manage in between. The earnings condor is the variance risk premium at its purest and its most dangerous: a high-probability vol-crush win with an ungappable, unmanageable tail.

**The calm-regime grind and the regime change.** Iron condor selling shines in calm, range-bound regimes — exactly the conditions where the variance risk premium is fattest and realized vol stays low. The danger is that *calm regimes end*, often abruptly, and the seller who has been lulled into larger size by a long winning streak is most exposed precisely when the regime breaks. The VIX term structure flipping from contango to backwardation (covered in [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve)) is one of the earliest tells that a calm regime is cracking — and a cue for the condor seller to reduce size, not add it. Owning a little long volatility as a hedge, as discussed in [volatility as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear), is the institutional way to carry a short-vol condor book through a regime change without ruin.

## The playbook: how to trade the iron condor

Pull it together into a concrete, tradable process.

**The view.** You believe the underlying will stay in a range — that the market's option prices are charging more for movement than the stock will deliver. This is a bet on low *realized* volatility relative to *implied*, i.e. harvesting the variance risk premium, with no directional opinion. Best deployed when implied vol is *elevated* (so you are selling rich premium) but you expect realized movement to be modest.

**The structure.** A bull put spread plus a bear call spread, all four legs out of the money. Sell the short strikes near the **0.16 delta** on each side (about a one-standard-deviation range); make the wings wide enough to keep the max loss to a tolerable fraction of your account but narrow enough that the worst case is sizeable. Sell around **30–45 DTE** for steady theta and manageable gamma.

**The Greek profile to expect.** Flat delta at the center (non-directional), **short gamma** (a fast move hurts and accelerates), **short vega** (a vol spike hurts even at a flat price), **long theta** (time decay pays you, on the order of a few dollars per lot per day). You are short volatility with defined risk.

**Entry.** Put it on when implied vol is at or above its recent norm (rich premium to sell) and the underlying is mid-range with no obvious imminent catalyst. Confirm the credit, max loss, breakevens, and POP *before* you click — know all four numbers. For our worked example: \$93.86 credit, \$406.14 max loss, breakevens \$93.06 / \$108.94, POP ~73.7%.

**Management (this is where the money is made).**

- **Take profit at ~50%** of the credit — close once you can buy it back for half what you sold it for. Do not hold to expiry to squeeze the last dollar; it is the slowest, riskiest dollar to earn.
- **Roll the untested side in** when one wing is tested: buy back the now-cheap far spread and sell a new closer one for extra credit, which widens your breakeven and trims the max loss on the threatened side. Never roll the tested side out for more risk.
- **Close a breached spread for a partial loss** before short gamma drags it to the full capped loss. The defined-risk wing caps the *worst* case; your discipline caps the *common* case well below it.
- **Exit by ~21 DTE** regardless, to step out before gamma near expiry turns toxic.

**Sizing.** Treat the per-lot max loss as the unit of risk and size so that a *simultaneous* max loss across your whole condor book — which is what a market gap delivers — is a survivable dent, not a death. Defined risk is per-position; portfolio risk is your job. See [position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) and the broader treatment of [managing a trade: rolling, adjusting, and when to just take the loss](/blog/trading/options-volatility/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss).

**The invalidation.** Get out — or refuse to put it on — when the regime is changing: implied vol spiking and the VIX term structure flipping to backwardation, a major catalyst looming inside your expiry, or your short strike breached with momentum. The condor is a trade for calm, range-bound markets; the moment the market stops being calm, the structure stops being your friend, and the discipline is to shrink, not to "average down" into a falling market.

**The one rule above all others:** *do not let a winner turn into a max loss.* The variance risk premium is real, the structure is sound, and the win rate is high — but the entire edge is contingent on cutting the rare losing trade short of its capped maximum. The condor seller who manages every trade keeps the steady credits *and* survives the tail. The one who sets and forgets collects pennies for months and then donates them all back, with interest, on the day the range finally breaks.

## Further reading & cross-links

Within this series:

- [Vertical spreads: defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — the single credit spread that is the condor's building block.
- [Gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — why short gamma is at its most toxic near expiry.
- [Theta: trading the clock](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options) — the time decay that powers the condor.
- [The net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — reading a multi-leg position's risk on one screen.
- [The variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the structural edge the condor harvests, and the tail that comes with it.
- [Implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the trade the condor is fundamentally making.
- [Straddles, strangles, and the long volatility bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet) — the other side of the trade, buying movement.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — how to size a short-vol book so the inevitable bad month is survivable.
- [Managing a trade: rolling, adjusting, and when to just take the loss](/blog/trading/options-volatility/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss) — the management discipline in full.

For the theory underneath:

- [Options theory](/blog/trading/quantitative-finance/options-theory) — option pricing fundamentals.
- [Volatility as an asset: owning fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — the allocator's view of long volatility as a hedge against short-vol books.

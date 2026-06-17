---
title: "Butterflies, Ratio Spreads, and Broken Wings: The Precision Tools"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn the three precision option structures: build a long butterfly for a cheap high-payout bet on a pin, see exactly why a front-ratio spread's naked tail is never free money, and use a broken-wing fly to take in a credit with risk only on the side you choose."
tags: ["options", "volatility", "butterfly", "ratio-spread", "broken-wing", "options-strategies", "defined-risk", "convexity", "theta", "pin-risk", "gamma", "income-options"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A butterfly, a ratio spread, and a broken-wing fly are the same object — longs and shorts on a strike ladder — tuned to three different jobs: a precise bet on where the stock lands, a cheap directional view, and a one-sided credit. The whole skill is reading which one you're holding from its leg counts, and never confusing a credit for a free lunch.
>
> - **The butterfly** (long 1 / short 2 / long 1 across three equal-spaced strikes) is a defined-risk tent: max loss is the tiny debit, max profit is the strike width minus that debit, and it pays out only if the stock **pins** near the body at expiry. A far-out-of-the-money body can pay **8:1, 20:1, even 60:1** for cents — pure cheap convexity.
> - **The front-ratio spread** (buy 1 / sell 2) finances itself into a credit by leaving one short option **naked**. It looks free and behaves beautifully right up to the moment a big move blows through the extra short strike — then the loss is *uncapped*. The credit is the bait, not the edge.
> - **The broken-wing butterfly** shifts one wing further out to turn the debit into a credit and erase the risk on *one* side, in exchange for a larger, but still **defined**, loss on the other. It's a directional/income hybrid: you pick which side you're willing to lose on.
> - **The one rule to remember:** defined-risk structures (flies, broken-wings) are sized by their max loss; the front-ratio is the *only* one of the three whose size must be set by the worst-case tail, never by the credit it pays.

## Two trades, one lesson about "free"

A trader I'll call Devin put on what his screener told him was a free trade. The stock sat at \$100, he was mildly bullish, and a "1-by-2 call ratio" — buy one \$100 call, sell two \$105 calls — printed a small *credit* of about \$30 per spread. He read the marketing and not the risk graph: "I get paid \$30, I make up to \$530 if it grinds to \$105, and the only way I lose is if it rockets, which it won't." He sold forty of them. \$1,200 in his pocket on open, a payoff diagram that looked like a friendly hill, and a thesis ("it won't rocket") doing all the load-bearing work.

It rocketed. A surprise acquisition headline gapped the stock to \$122 overnight. Above \$110 each spread bleeds a dollar a share for every dollar the stock climbs — and Devin was net short *forty* extra calls with no upper wing to catch them. At \$122 the position was down about \$970 per spread; on forty spreads that's roughly **\$38,000** — on a trade his platform had cheerfully labelled a credit. He didn't lose because he was wrong about direction. He was *right* about direction. He lost because he never priced the naked tail he'd sold, and he sized the trade by the \$30 he collected instead of by the catastrophe he'd underwritten.

The same week, a different trader put on the opposite kind of bet. She thought a sleepy index name pinned near \$110 into a known event window, and she bought a \$105/\$110/\$115 call butterfly for a **\$55 debit**. If she was wrong, she lost the \$55. If she was right and it pinned at \$110, the fly was worth its full \$5 width — about **\$445**. That's an **8:1 payout on a defined, tiny risk**. It pinned. She made the \$445. Two traders, two "cheap" structures, opposite outcomes — and the difference between them is the entire subject of this article.

![A long call butterfly payoff diagram at expiry showing a tent peaking at the body strike with labelled max profit, max loss, and breakevens](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-1.png)

That tent above is the butterfly: a small defined loss everywhere except a narrow window around the body strike, where the payoff spikes. It is the most *surgical* structure in options — a bet not on direction, not even on a range, but on a **point**. The front-ratio Devin sold and the broken-wing we'll build later are its close relatives, assembled from the same parts but with one leg moved or removed. By the end you'll be able to glance at any of the three, read off its leg counts, and know instantly whether you're holding defined risk or a hidden bomb.

## Foundations: the strike ladder and the three structures

Everything in this article is built from one primitive — a single option contract — arranged on a **strike ladder**: a row of equally spaced strike prices (say \$95, \$100, \$105, \$110, \$115), with the stock currently somewhere on it. A *structure* is just a recipe of how many contracts you're long (you bought, you own the right) or short (you sold, you owe the obligation) at each rung. That's the whole game. Let me build the vocabulary from zero, then assemble the three tools.

### A quick refresher on the single leg

A **call** is the right to buy the stock at a fixed **strike** by **expiry**; a **put** is the right to sell at the strike. You pay a **premium** to be long an option; you collect a premium to be short one. At expiry, an option is worth only its **intrinsic value** — how far in-the-money it is — and nothing else: a \$100 call is worth `max(0, stock − 100)`, a \$100 put is worth `max(0, 100 − stock)`. The full pricing machinery (the Black-Scholes model, why time and volatility have value before expiry) is derived in the [options-theory primer](/blog/trading/quantitative-finance/options-theory) and the [Black-Scholes derivation](/blog/trading/quantitative-finance/black-scholes); this series owns the *practice*. Throughout this post I'll price every leg with the same Black-Scholes model — \$100 stock, 20% implied volatility, 4% interest rate, and a quarter-year (90-day) expiry unless I say otherwise — so all the dollar figures are computed, not invented.

The single most important habit for reading any multi-leg structure: **you build the whole thing by summing the legs.** The payoff of a position at any stock price is the sum of each leg's payoff (intrinsic value, with a plus sign if you're long and a minus sign if you're short), minus what you paid net (or plus what you collected net). There's no interaction term, no blending. Three legs? Add three numbers. This additivity is the same property that lets you sum the Greeks of a book, covered in [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard). It's a gift, and it's how we'll construct every figure here.

### The butterfly: long 1 / short 2 / long 1

A **long butterfly** (a "fly") uses three equally spaced strikes. You buy one option at the lower strike (the lower **wing**), sell two at the middle strike (the **body**), and buy one at the upper strike (the upper wing). All calls, or all puts — a long *call* fly buys the wings and sells two body calls; a long *put* fly does the same with puts. The two structures have nearly identical payoffs by put-call parity (the [parity proof](/blog/trading/quantitative-finance/options-theory) lives in the theory posts), so traders pick whichever side has better liquidity or fits their skew view.

Why this shape? The body is sold *twice*, so as the stock climbs past the body strike, those two short options start losing value to you faster than your single long lower-wing call gains — the payoff turns over and heads down. But then the *upper* long wing kicks in and catches the fall, capping your loss. Below the lower wing, all four legs expire worthless and you've lost only your small net debit. The result is a **tent**: flat-and-low on both sides, with a sharp peak exactly at the body strike. You are betting the stock **pins** at the body.

The defining numbers of a long fly are clean and worth memorizing:

- **Max loss = the net debit you paid.** It happens anywhere outside the wings (the stock either far above or far below).
- **Max profit = the strike width − the net debit.** It happens at expiry *only* if the stock sits exactly at the body. The "width" is the distance between adjacent strikes (e.g., \$5 in a \$95/\$100/\$105 fly).
- **Breakevens = body's two sides:** lower breakeven = lower wing + debit; upper breakeven = upper wing − debit. (Equivalently, body ± (width − debit). They're symmetric around the body.)

### The front-ratio spread: buy 1 / sell 2

Strip the upper wing off a fly and you've got a **front-ratio spread**: long one option at a lower strike, short *two* at a higher strike, and **nothing** above to cap the second short. ("Front" ratio means more shorts than longs, sold closer to the money; a "back" ratio is the reverse — more longs than shorts — and is a long-volatility, long-tail bet, the opposite animal.) Because you sell two options and buy only one, the structure is often **self-financing** or even pays a **credit**. That missing upper wing is exactly what makes it look free and exactly what makes it dangerous: one of those two short options is **naked** — uncovered, with theoretically unlimited loss on a call ratio (or large, strike-bounded loss on a put ratio).

The front-ratio has a **profit hill** that peaks at the short strike (like the right half of a fly), but past the upside breakeven it doesn't flatten — it keeps falling, because above the short strike you're net short one extra option and lose a dollar a share for every dollar the stock moves. The credit is real; the tail is also real; and the second is far bigger than the first.

### The broken-wing butterfly: shift one wing out

Now the clever middle ground. Start with a fly, but move *one* wing further out — say, from \$110 to \$120 — instead of removing it. This is a **broken-wing butterfly** (also "skip-strike fly," because you skip a strike when placing the far wing). The asymmetry does two useful things at once. First, the far wing is cheaper (it's more out-of-the-money), so the whole structure costs less — often it flips to a **credit**. Second, the unequal widths erase the risk on *one* side entirely: a call broken-wing fly built with a wide upper wing carries **no downside risk** (you keep the credit if the stock falls) and a **defined** loss only on the upside. You've taken a symmetric, two-sided, defined-risk fly and turned it into a one-sided, defined-risk, credit-collecting hybrid. You choose which side you're willing to be wrong on.

![A long call butterfly built from its three legs, the long lower wing, the short two body calls, and the long upper wing, summed into the tent](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-2.png)

The model above shows the fly assembled from its parts. The green dashed line is the long \$95 call (it gains as the stock rises, after paying its premium). The red dashed line is the two short \$100 calls (they bleed you as the stock rises past \$100 — twice as fast as one). The amber dashed line is the long \$105 call (it gains in the far upside and catches the falling payoff). Add the three together — the blue line — and you get the tent: the body shorts pull the middle down, the upper wing lifts the right side back up, and the whole thing peaks where the body sits. Reading a structure as a *sum of legs* is the skill that makes all three tools obvious instead of memorized.

## The butterfly in depth: a bet on a point

Let's go deep on the fly, because once you understand it precisely, the ratio and the broken-wing are one edit away.

### What you're really betting on

A long butterfly is the purest expression of "I think the stock will be **right here** at expiry, and I don't think it'll move much getting there." It's three bets rolled into one structure:

1. **A pin bet.** Your maximum payoff requires the stock to land *at the body strike*. The closer it lands, the more you make; the further, the less. The payoff is a tent, not a plateau — there's exactly one peak.
2. **A low-volatility bet.** A fly profits when *realized* volatility comes in low — when the stock chops quietly into expiry rather than trending or gapping. Big moves in either direction push you toward your wings, where you lose the debit. This is why a fly is a **mean-reversion / low-vol** structure: it wants nothing to happen.
3. **A short-vega bet at the body.** When the stock sits near the body, the position is *short* implied volatility — a rise in implied vol *hurts* it, because higher implied vol fattens the body options you're short more than the wings you're long. We'll quantify this below.

The reason traders love the fly is the **asymmetry of the bet**: you risk a tiny, fixed debit to win a multiple of it. That's the same convexity you'd get from buying a lottery ticket — except, unlike a lottery ticket, the odds are knowable and you control the strikes. The tradeoff is the *probability*: pinning a point is hard, so most flies expire worthless and you lose the (small) debit. The fly pays you in *payoff ratio* for accepting a *low hit rate*.

#### Worked example: building a \$95/\$100/\$105 call butterfly

Stock at \$100, 20% implied vol, 90 days to expiry, 4% rate. Price each leg with the Black-Scholes model:

- Buy one **\$95 call** at **\$7.5459**
- Sell two **\$100 calls** at **\$4.4852** each → collect **\$8.9704**
- Buy one **\$105 call** at **\$2.3909**

Net cost: `7.5459 − 8.9704 + 2.3909 = \$0.9663` per share. Times the 100-share multiplier, that's a **net debit of \$96.63** to put on one fly. Now the defining numbers:

- **Max loss = the debit = \$96.63.** If the stock finishes below \$95 or above \$105, every leg's intrinsic relationships net to zero and you simply lose what you paid.
- **Max profit = width − debit = \$5.00 − \$0.9663 = \$4.0337 per share = \$403.37 per fly.** This happens only if the stock pins exactly at \$100: the lower \$95 call is worth \$5 intrinsic, the two \$100 calls expire worthless (you keep their premium), and the \$105 call expires worthless. Five dollars of value, minus the debit.
- **Lower breakeven = 95 + 0.9663 = \$95.97.** **Upper breakeven = 105 − 0.9663 = \$104.03.** Between these two prices, the fly makes money; outside them (but you only lose the full debit beyond the wings), it loses.

The intuition: you've paid \$96.63 for the right to make up to \$403.37 if the stock sits still at \$100 — a roughly **4.2:1 payout** on a defined risk, with a profit window from \$95.97 to \$104.03. You're not betting on direction; you're betting on *stillness*, centered on \$100.

### The Greeks of a fly: short the center, defined at the edges

A fly's Greek profile *changes sign* depending on where the stock sits relative to the body — that's what makes it interesting and what trips people up. The key is to evaluate the Greeks **at the body**, because that's where you want the stock to be.

#### Worked example: the net Greeks of the fly at the body

Take the same \$95/\$100/\$105 call fly with the stock right at the body (\$100). Summing each leg's Greek (long wings positive, short body doubled and negative), the Black-Scholes model gives:

- **Net gamma ≈ −0.0092** (short gamma at the center). The two short body calls have the *highest* gamma of the three strikes — they're at-the-money — so their doubled negative gamma overwhelms the wings' positive gamma. You are **short gamma** when the stock is near the body, exactly where you want it.
- **Net vega ≈ −0.046 per vol point** (short vega at the center). Same logic: the at-the-money body has the most vega, doubled and short. A *rise* in implied volatility **hurts** a fly sitting at its body — the body options you're short fatten faster than the wings you own.
- **Net theta ≈ +\$0.0052 per day** (collecting time at the center). The flip side of short gamma is positive theta: at the body, time decay works *for* you, because the body options you're short are decaying faster than the wings.
- **Net delta ≈ −0.005** (roughly flat). A symmetric fly at its body is nearly delta-neutral — it's a bet on location and stillness, not direction.

The intuition: **a long fly, when the stock sits at its body, behaves like a tiny short-straddle — short gamma, short vega, long theta — but with the catastrophic tails bought back by the wings.** It's "short volatility with a seatbelt." This connects directly to [gamma, the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short): being short gamma means you're hurt by movement and helped by stillness, but the wings cap the bite, so unlike a naked short straddle, your worst case is the known debit, not a margin call.

Move the stock *away* from the body, though, and the signs flip: out in a wing the position becomes long the surviving wing's gamma. This is why the fly is genuinely a *local* structure — its character depends on where price is. Near the body: short vol, collect theta, want stillness. Out in the wings: you've already lost the debit and the Greeks barely matter. The whole position lives or dies on whether the stock comes to rest near the body by expiry.

### Pin risk and the expiry knife-edge

A fly's headline max profit comes with a sharp practical catch that deserves its own treatment, because it's where novice fly traders lose money they thought they'd already made. The full \$403.37 in our worked example materializes *only at the instant of expiry, and only if the stock closes exactly at the body.* Up to that instant, the tent is rounded and squashed by the time value still left in the body options you're short. With a month to go, a fly sitting right at its body might be marked at only 40% to 60% of its theoretical max, because to *close* the position you'd have to buy back two near-the-money short calls that still carry meaningful premium. The profit you see on screen mid-trade is real but partial; the peak is a limit, not a current value.

That gap creates a genuine dilemma in the final day. Hold to the close and you're exposed to **pin risk**: if the stock settles right at the body strike, your short body options are *at-the-money at expiry*, and whether they get assigned is a coin-flip you don't control. You can wake up the next morning unexpectedly long or short 200 shares of stock (two contracts × 100) from the assignment of one or both body shorts, with the long wings having expired worthless against them — a delta you never wanted and a margin call you didn't budget for. The professional habit is to **close the fly before the final hours** rather than chase the last sliver of theoretical profit into the assignment lottery. The few dollars of edge you give up by closing early is cheap insurance against waking up with a surprise stock position. The deeper mechanics of expiration-day gamma and dealer pinning live in [gamma, the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short); the practical takeaway here is simply that a fly's profit is *harvested early*, not held to the knife-edge.

### The iron butterfly: the same tent, sold as a credit

There's a popular variant worth knowing because it's the fly's mirror image. An **iron butterfly** is built from *both* calls and puts: you **sell** an at-the-money straddle (short the \$100 call and short the \$100 put) and **buy** protective wings (long a \$95 put and a \$105 call). Same three strikes, same tent-shaped payoff — but instead of paying a debit you *collect a credit*, because you're net short the expensive at-the-money options.

#### Worked example: the iron butterfly is the fly turned inside-out

Same parameters (\$100 stock, 20% vol, 90 days, 4% rate). Price the legs:

- Sell one **\$100 call** at **\$4.4852** and one **\$100 put** at **\$3.4902** → collect **\$7.9754**
- Buy one **\$95 put** at **\$1.6006** and one **\$105 call** at **\$2.3909** → pay **\$3.9915**

Net **credit = 7.9754 − 3.9915 = \$3.9840 per share = \$398.40** per iron fly.

- **Max profit = the credit = \$398.40**, realized if the stock pins at \$100 (all four legs' intrinsic relationships collapse to keeping the credit).
- **Max loss = width − credit = \$5.00 − \$3.9840 = \$1.0160 per share = \$101.60.**
- **Breakevens = 100 ± credit = \$96.02 and \$103.98.**

Notice the symmetry with the call fly: the iron fly's **max profit (\$398.40)** is almost exactly the call fly's **max profit (\$403.37)**, and the iron fly's **max loss (\$101.60)** is almost exactly the call fly's **debit (\$96.63)** — the tiny differences come from interest and the put-call skew. They're the *same economic position*, one quoted as a debit you pay and one as a credit you collect. The intuition: whether your broker shows a fly as "pay \$96.63 to make up to \$403.37" or "collect \$398.40 risking \$101.60," it's the identical tent — don't let the credit-versus-debit framing fool you into thinking they're different trades. (This is the same fly-vs-iron-condor relationship explored in [iron condors and credit spreads](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range); a condor just widens the body into a range.)

## The ratio spread: where "free" goes to die

Now we remove the upper wing and meet the structure that ran Devin over. The front-ratio spread is seductive precisely because its *initial* numbers are so attractive — and its *tail* numbers are usually invisible on the order ticket.

### Anatomy of a front-ratio

Buy one call at a lower strike, sell two at a higher strike. With the stock at \$100, a classic is **buy one \$100 call, sell two \$105 calls.** Below the lower strike, everything expires worthless and you keep whatever credit you took. Between the strikes, your long call is in-the-money and gaining while the shorts are still worthless — pure profit, climbing. At the short strike you hit **maximum profit**. Above it, the *two* short calls start losing value against your *one* long: net, you're short one extra call, and you lose a dollar a share for every dollar up — forever, with no wing to stop it. The payoff is a hill that becomes a cliff.

The reason this gets sold so often is that on equity indices the [volatility skew](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) means out-of-the-money options carry their own implied vol, and selling two of them frequently brings in more premium than the one you buy costs — so the ticket shows a credit. The trader sees "credit + a profit hill + I only lose on a big rally" and stops reading. What they've actually done is **sell a naked call** and use a long call to dress it up.

![A front-ratio call spread payoff showing the profit hill peaking at the short strike and the naked uncapped loss tail beyond the upside breakeven](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-3.png)

The chart above is Devin's trade. The friendly green hill peaks at \$105; the credit (\$30) is the little sliver kept if the stock stays below \$100. But look right: past the \$110.30 breakeven the blue line dives into the red-shaded danger zone and *keeps going*. There is no floor. That uncapped tail is the entire risk of the structure, and it's the part the word "credit" hides.

#### Worked example: a front-ratio's credit and its naked-tail loss

Stock at \$100, 20% vol, 90 days, 4% rate. Buy one \$100 call, sell two \$105 calls:

- Buy one **\$100 call** at **\$4.4852**
- Sell two **\$105 calls** at **\$2.3909** each → collect **\$4.7818**

Net: `4.4852 − 4.7818 = −\$0.2965` per share → you **collect a credit of \$29.65** per spread. Now walk the expiry payoff at several prices (per share, then ×100):

- **Stock ≤ \$100:** every call expires worthless, you keep the credit → **+\$29.65.**
- **Stock = \$105 (the short strike):** long \$100 call worth \$5, shorts worth 0, plus credit → `5 + 0.2965 = \$5.2965` per share → **max profit +\$529.65.**
- **Upside breakeven:** above \$105 you lose a dollar a share per dollar up, starting from the \$5.2965 peak, so you cross zero at `105 + 5.2965 = \$110.30.**
- **Stock = \$120:** long \$100 call worth \$20, two short \$105 calls worth \$15 each = \$30 owed, plus credit: `20 − 30 + 0.2965 = −\$9.7035` per share → **−\$970.35.**
- **Stock = \$130:** `30 − 50 + 0.2965 = −\$19.7035` → **−\$1,970.35**, and climbing without limit.

The intuition: you were paid **\$29.65** to accept a loss that is **\$970 at \$120 and \$1,970 at \$130 and unbounded above that.** The credit is one-thirtieth of the loss you take on a routine 20% rally. A front-ratio is not free money; it's a small premium for underwriting a naked tail, and the only honest way to hold it is to size it by that tail — never by the credit.

### Quantifying the tail: the number that should scare you

The discipline that separates a survivor from a Devin is computing the **tail loss at a stress price before you put the trade on.** For any front-ratio with `n` extra short calls (here `n = 1`, since we're short two and long one), the loss above the upper region grows linearly: `loss ≈ n × (stress price − short strike) × multiplier − credit`. With one extra short \$105 call and a stress price of \$130, that's `1 × (130 − 105) × 100 − 29.65 = \$2,470.35` of *additional* exposure beyond the peak — and there's no upper bound, so "stress price" is a choice, not a cap.

Compare that to the fly: the *only* difference between Devin's ratio and a fly is the missing \$110 upper wing. Buying that one wing back — a \$110 call for about \$1.14 — would have converted his uncapped tail into a defined \$5-wide structure, turning a possible \$38,000 loss into a few hundred dollars at most. The wing costs almost nothing relative to the catastrophe it prevents. **The ratio's "extra" credit is the price you're paid for *not* buying that cheap insurance — and it is almost never worth it on an index that can gap.** This is the same lesson as the [variance risk premium](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear): selling tails pays a steady trickle until the one day it doesn't, and the one day erases years of trickle.

### Defending a ratio — and why the best defense is structure, not heroics

When a front-ratio starts going wrong, traders reach for "adjustments" — and most of them make it worse. The instinct to "roll the shorts up and out" (buy back the two short calls and sell two further-out, further-dated ones) usually just *resets the same naked tail at a higher strike and a later date*, deferring the problem while adding more time for the move to extend. The instinct to "add a long call to cap it" is correct — but at that point you're paying a panicked, in-the-money price for the upper wing you should have bought at the start, when it cost \$1.14. The cheap insurance you skipped to harvest \$30 of credit now costs many multiples more, precisely because it's needed.

The honest defense is built in *before* the trade, not improvised after: either buy the upper wing up front (which converts the ratio into a broken-wing or a fly — defined risk) or set a mechanical stop at the short strike and take the loss without negotiation. A front-ratio defended by "rolling and hoping" is how a \$970 loss at \$120 becomes a \$1,970 loss at \$130 and worse — the position is *already* short the tail, and every roll keeps you short it. The structural fix (a wing) is cheap and permanent; the behavioral fix (discipline at the stop) is free but hard. Both beat the third option, which is to treat an uncapped short as something you can manage with cleverness once the move is underway. You cannot. The only winning move against a naked tail is to not have one, or to size it so small that the catastrophe is survivable.

## The broken-wing butterfly: choosing your wrong side

The broken-wing fly is the structure that resolves the tension. It keeps the defined-risk discipline of the fly but borrows the credit-collecting appeal of the ratio — by being honest about which side you're willing to lose on.

### How the asymmetry works

Take a standard call fly — long \$100, short two \$105, long \$110 — and **shift the upper wing out** from \$110 to \$120. Two things change. First, the \$120 call is much cheaper than the \$110 call (it's further out-of-the-money), so the structure costs less and typically flips to a **credit**. Second, the two wings are now *unequal width*: \$5 on the lower side (\$100→\$105) and \$15 on the upper side (\$105→\$120). That unequal geometry is what creates the one-sided risk. Below the lower strike, all legs expire worthless and you keep the credit — **no downside risk at all.** Above the body, the payoff falls (the two shorts hurt you) but the far \$120 wing eventually catches it, so the upside loss is **defined**, just larger than a symmetric fly's would be because the wing is further away.

The mental model: a broken-wing call fly says *"I want a credit, I'm fine if the stock falls or stays put, and I'm willing to take a known loss only if it rallies hard through my body."* You've chosen the upside as your "wrong side." A broken-wing *put* fly does the mirror — it's the income trader's favorite, collecting a credit with no *upside* risk and a defined loss only on a sharp drop. You break the wing on the side you fear least.

![A broken-wing butterfly payoff with the upper wing shifted out, showing a credit and no downside risk versus a symmetric fly with one-sided upside loss](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-4.png)

The chart contrasts the two. The gray dashed line is the symmetric fly (\$100/\$105/\$110) — a clean tent that cost an \$84 debit, with losses on *both* sides. The blue line is the broken-wing (\$100/\$105/\$120): the same \$511 peak at \$105, but the left side now sits *above zero* (the green credit floor — you keep \$11 no matter how far the stock falls), while the right side carries the full one-sided loss out to \$120. You've traded a symmetric two-sided debit structure for an asymmetric one-sided credit structure. Same peak, completely different risk shape.

#### Worked example: a broken-wing fly's credit and one-sided risk

Stock at \$100, 20% vol, 90 days, 4% rate. Long one \$100 call, short two \$105 calls, long one \$120 call:

- Buy one **\$100 call** at **\$4.4852**
- Sell two **\$105 calls** at **\$2.3909** each → collect **\$4.7818**
- Buy one **\$120 call** at **\$0.1882**

Net: `4.4852 − 4.7818 + 0.1882 = −\$0.1084` per share → **credit of \$10.84** per fly. Walk the payoff:

- **Stock ≤ \$100:** all calls expire worthless → you keep **+\$10.84.** This is the *entire* downside: **no loss if the stock falls.**
- **Stock = \$105 (body):** long \$100 worth \$5, shorts worth 0, far wing worth 0, plus credit → `5 + 0.1084 = \$5.1084` → **max profit +\$510.84.**
- **Upside breakeven = 105 + 5.1084 = \$110.11.** Above it you start losing.
- **Stock = \$115:** `15 − 2×10 + 0 + 0.1084 = −\$4.8916` → **−\$489.16.**
- **Stock = \$120 and above:** the long \$120 wing engages and *caps* the loss: `20 − 2×15 + 0 + 0.1084 = −\$9.8916` → **−\$989.16**, flat from there up. The loss is **defined** at \$989.16, no matter how high the stock goes.

The intuition: you were **paid \$10.84** to take on a structure with **zero downside risk and a capped \$989.16 upside risk**, peaking at \$510.84 if the stock drifts to \$105. Unlike Devin's ratio — same body, same shorts, but no upper wing — your worst case is a known \$989, not an unbounded catastrophe. The \$120 wing, which cost you \$18.82, is the difference between "defined risk" and "naked tail." That single cheap wing is the whole point of the broken-wing.

Notice the family resemblance across all three worked examples: the fly (\$96.63 debit, two-sided defined risk), the ratio (\$29.65 credit, one naked tail), and the broken-wing (\$10.84 credit, one *defined* tail) are the **same three short body calls** with the upper wing placed at \$105 (closed, a fly), nowhere (open, a ratio), or far out (\$120, broken). Where you put — or don't put — that one wing decides everything.

### The put broken-wing: the income trader's favorite

In practice the *put* broken-wing fly is the more common income structure, because equity skew makes downside puts richer and a put broken-wing collects a cleaner credit. The construction mirrors the call version, reflected: long one upper put, short two body puts (below the upper), and a long far-out lower put with the wing skipped wider. The result is **no upside risk** (you keep the credit if the stock rises or stays flat) and a defined loss only if the stock drops hard through the body. It's the natural expression of "mildly bullish, willing to be wrong only on a sharp decline, happy to be paid to wait."

The trade-off to respect is the skew itself: because the lower put you buy as protection is cheaper than the body puts you sell, the credit is appealing — but the defined loss is concentrated exactly where crashes happen, and equity crashes move *fast*. A put broken-wing run as a disciplined income program needs a hard stop on the downside body, not a "hope it bounces" reflex, because the same gap risk that makes the front-ratio dangerous applies (in *defined* form) here too. The structure caps your loss, but it doesn't slow down the move that triggers it.

![A strike-ladder table comparing long and short contract counts at each strike for a symmetric butterfly, a front-ratio spread, and a broken-wing fly with the resulting net cash](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-5.png)

The table above is the Rosetta Stone for the three structures. Read across each row: the symmetric fly is `+1, −2, +1, 0` — balanced, tails closed, a small debit. The front-ratio is `+1, −2, 0, 0` — that missing third long leaves one short **naked**, and the credit comes *from* that nakedness. The broken-wing is `+1, −2, skip, +1` — it keeps the closing long but parks it far out at \$120, skipping \$110, which is what produces both the credit and the one-sided risk. **The leg counts tell you the risk before any pricing model does:** equal longs and shorts with both tails covered means defined two-sided risk; more shorts than longs means a naked tail; a skipped strike means a one-sided credit. Learn to read the ladder and you can never again be surprised by what you're holding.

## Cheap convexity: why a far-out fly pays 60:1

We've seen a near-the-money fly pay about 4:1. The hook trader made 8:1. Where does that ratio come from, and how high can it go? This is the fly's secret weapon — **cheap convexity** — and it's worth its own section because it's the reason flies are a precision tool rather than a curiosity.

### The payout/cost ratio and where it explodes

The fly's max payout is *always* the strike width (here \$5, or \$500 per fly) minus the debit. But the *debit* shrinks dramatically as you push the body further from the current price, because all three options get cheaper and the structure's net cost collapses toward zero. A fly with its body far out-of-the-money costs pennies but still pays the full \$500 width if the stock happens to pin there. So the **payout-to-cost ratio rises steeply** the further out you place the body — you're buying a lottery ticket on a specific point, and far-out points are cheap tickets.

#### Worked example: the payout/cost ratio across body distance

Same \$100 stock, 20% vol, 90 days, 4% rate, always a \$5-wide call fly, just moving the body strike out:

- **Body at \$100** (\$95/\$100/\$105): debit **\$96.63**, max profit **\$403.37** → **4.2:1**
- **Body at \$105** (\$100/\$105/\$110): debit **\$84.39**, max profit **\$415.61** → **4.9:1**
- **Body at \$110** (\$105/\$110/\$115): debit **\$59.79**, max profit **\$440.21** → **7.4:1**
- **Body at \$115** (\$110/\$115/\$120): debit **\$35.29**, max profit **\$464.71** → **13.2:1**
- **Body at \$120** (\$115/\$120/\$125): debit **\$17.74**, max profit **\$482.26** → **27.2:1**
- **Body at \$125** (\$120/\$125/\$130): debit **\$7.75**, max profit **\$492.25** → **63.5:1**

And it gets even more extreme near expiry: a \$5-wide fly with its body at \$110 with only 45 days left costs about \$55.41 for a \$444.59 max profit — the **8:1** the hook trader collected. The intuition: the payout is capped at the width, but the cost goes to zero as the body moves out or time runs down, so the *ratio* explodes — you can buy a 60:1 shot on a precise pin for a few dollars. That's pure convexity: a tiny defined risk for an outsized defined reward, the exact opposite of the front-ratio's tiny reward for an outsized naked risk.

![A chart showing a five-wide butterfly's payout-to-cost ratio rising sharply as the body strike moves further from the spot price, with the debit shrinking](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-6.png)

The figure makes the convexity vivid: the amber bars (the debit) shrink toward nothing as the body moves out, while the green ratio line rockets from 4:1 to 60:1. The max payout never changes — it's always the \$500 width — so cheapening the ticket is the *only* thing happening, and it's everything. This is why flies are the natural tool for cheap convexity **into expiry** and especially for the [0DTE pin trades](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure) we'll cover later: with hours to expiry, a far-out fly costs almost nothing and pays the full width if the stock pins. The catch, always, is *probability* — a 60:1 fly hits roughly one time in sixty if priced fairly. The convexity is real; so is the low hit rate. You're being paid in payoff multiple for accepting that you'll be wrong most of the time.

## Common misconceptions

These are the specific wrong beliefs that cost real money on these structures. Each is corrected with the numbers we've already computed.

### Misconception 1: "Ratio spreads are free money because they pay a credit"

This is Devin's mistake, and it's the most expensive one in this article. The \$29.65 credit on the \$100/\$105 call ratio is not your edge — it's your *bait*. The same position loses **\$970 at \$120** and **\$1,970 at \$130**, with no upper bound. The credit is one-thirtieth of a routine-rally loss. A credit on your ticket tells you which direction cash moved *today*; it tells you nothing about the risk you've underwritten. **Always price the tail at a stress level before the trade, and size by that — never by the credit.** A credit-paying structure with a naked leg is a short option in a costume.

### Misconception 2: "A butterfly is a high-probability income trade"

Flies are sold as "income" because of the high payout ratio, but the ratio and the probability are *inversely* related. A 4:1 near-the-money fly needs the stock to land in a roughly \$8-wide window (\$95.97 to \$104.03 in our example) — plausible but far from certain. A 60:1 far-out fly needs the stock to pin a *specific* far-away point — it hits maybe one time in sixty. **A fly is a low-hit-rate, high-payoff convexity bet, not a steady income stream.** If you trade flies for income, you'll have many small losses (the debit) punctuated by occasional large wins, and your expectancy depends entirely on whether the body is priced better than the market's implied distribution. Treating it like a high-probability trade leads to over-sizing and a string of "small" \$96 losses that add up faster than the occasional \$400 win replaces them.

### Misconception 3: "The max profit on a fly is easy to capture"

The fly's max profit requires the stock to sit *exactly* at the body **at expiry** — not during the trade, at expiry. Before expiry, the tent is rounded and squashed: with 30 days left, a fly trading at its theoretical max-profit body might show only 40–60% of that peak, because the body shorts still carry time value you'd have to buy back to close. The full \$403.37 only materializes if you *hold to expiry* and the stock cooperates — which exposes you to **pin risk** (assignment chaos if the stock closes right at the body) and to the gamma whipsaw of the last day. Most fly profits are taken *early* at a fraction of theoretical max, precisely to avoid the expiry knife-edge. **The headline max-profit number is a limit you rarely touch, not a target you usually hit.**

### Misconception 4: "A broken-wing butterfly is risk-free because it's a credit with no downside"

The broken-wing call fly genuinely has *no downside* risk — but "one side is free" is not "the trade is free." Our broken-wing collected \$10.84 and risked **\$989.16** on the upside. That's a 91:1 risk-to-credit ratio on the *defined* loss. The structure is excellent when your view is "won't rally hard," but if the stock *does* rip through the body, you lose nearly a thousand dollars to have collected eleven. **The broken-wing converts two-sided risk into one-sided risk; it does not eliminate risk.** Size it by the defined max loss (the \$989), exactly as you'd size a defined-risk vertical, not by the credit.

### Misconception 5: "Wider strikes always make a fly better because the payout is bigger"

A wider fly does have a larger *potential* payout (the width is bigger), but it also costs more and has a *lower* payout-to-cost ratio when centered the same way, and — crucially — a wider profit window doesn't mean a higher *expected* value. The market prices the wider window into a proportionally higher debit. A \$10-wide fly isn't "twice as good" as a \$5-wide fly; it's a different bet with roughly the same edge (or lack of it), just larger in absolute dollars. **Width is a position-sizing and target-precision choice, not a free upgrade.** Choose width by how precisely you can predict the pin, then size the *number* of flies by your risk budget — covered in [position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading).

## How it shows up in real markets

These structures aren't textbook curiosities — they're the daily bread of index option desks, earnings traders, and the exploding 0DTE crowd. Here's where you actually see them.

### Pinning into expiration and 0DTE flies

Heavily-traded names with enormous open interest at round strikes often **pin** to those strikes on expiration day, as dealers hedging their gamma trade the stock back toward the strike where their exposure is smallest (the mechanics live in [gamma, the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short)). A trader who believes a name will pin \$450 by the close can buy a \$445/\$450/\$455 fly for cents that morning and collect the full \$5 width if the pin holds. With the rise of **zero-days-to-expiry** options on the index, this has become a daily structure: 0DTE flies are the cheapest, most convex way to bet on an intraday pin, and we'll cover the market-structure consequences in [the rise of short-dated options](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure). The appeal is exactly the cheap-convexity math above: with hours to expiry, the debit is tiny and the payout ratio is enormous.

### Earnings: the fly as a "stays-in-the-range" bet

Before an earnings report, implied volatility is jacked up because the market is pricing a big move (the "expected move"). A trader who thinks the *actual* move will be smaller than the implied — that the stock will land near its current price or a specific target — can buy a fly centered on that target. If the stock obeys, the fly pays multiples; if it gaps far, the loss is the small defined debit. This is a defined-risk way to be **short the earnings vol-crush** without the unbounded risk of selling a naked straddle. Many desks prefer the iron-fly variant here to collect the inflated credit, accepting the defined wing loss if the move is large.

### Front-ratios as "skew harvesting" — and the blow-up risk

On equity indices, the persistent put skew means out-of-the-money puts are richly priced. Some income traders sell **put front-ratios** (buy one closer put, sell two further-out puts) to "harvest" that skew for a credit, betting the index won't crash through the lower short strike. This works for months — the variance risk premium pays — and then a crash like February 2018's "Volmageddon" or the March 2020 COVID plunge sends the index straight through the naked short strike and the uncapped tail detonates. **Every front-ratio is implicitly a bet that the fat tail won't show up before you close.** The structure is fine in small, sized-by-the-tail doses with a hard stop; it is a portfolio-killer when sized by the credit, which is exactly how it tends to be sized because the credit is what's visible.

### Broken-wings as a directional income workhorse

The broken-wing put fly is a staple of systematic income programs: collect a small credit, no risk if the underlying rises or stays flat, defined loss only on a sharp drop. Run as a disciplined, sized program with a defined-loss stop, it's a reasonable way to express "mildly bullish, willing to be wrong only on a crash, want to get paid to wait." It pairs naturally with the [calendar and diagonal](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure) structures that trade the time dimension, and with the credit-spread logic in [iron condors and credit spreads](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range). The discipline that makes it work — and the discipline Devin lacked — is sizing by the *defined max loss*, treating the credit as a rebate, not a reason.

## The playbook

Here's how to actually deploy the three precision tools — the view each expresses, the Greek profile, entry and exit, sizing, and what invalidates the trade.

### When to reach for each tool

- **The butterfly — for a pin or a precise target.** Use it when your view is specific: "the stock will be near X at expiry, and won't move much getting there." It's the right tool for pinning into expiration, for a defined-risk earnings "stays-in-range" bet, and for cheap convexity into [0DTE](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure). Greek profile at the body: short gamma, short vega, long theta, near-flat delta — "short vol with a seatbelt." Place the body where you think the stock lands; place it further out for a cheaper ticket and a higher payout ratio at lower odds.
- **The front-ratio — for a cheap directional view, accepting the tail.** Use it *only* when you have a directional view (toward the short strike), genuinely believe a violent move past the shorts is unlikely, **and** you've sized by the stress-level tail loss with a hard stop. The credit and the profit hill are the appeal; the naked tail is the price. If you can't price and stomach the loss at a 20–30% adverse move, you cannot hold this structure.
- **The broken-wing — for a one-sided credit.** Use it when you want to *get paid* to hold a directional/income view with risk only on the side you fear least: a broken-wing call fly for "won't rally hard," a broken-wing put fly for "won't crash." Greek profile: a credit, no risk on the chosen side, defined loss on the other. It's the disciplined trader's answer to the ratio — same income appeal, but the risk is *capped*, not naked.

![A decision flowchart for choosing a butterfly for a pin, a front-ratio for cheap directional with a tail, or a broken-wing for a one-sided credit](/imgs/blogs/butterflies-ratio-spreads-and-broken-wings-the-precision-tools-7.png)

The decision chart above routes a view to a tool. Start with what you believe: a precise pin → fly; cheap directional with low tail odds → ratio (sized by the tail); a credit with one-sided protection → broken-wing. The blue node is the question; the three colored nodes are the tools. The single most important thing the chart encodes: the fly and the broken-wing are **defined-risk** (size by max loss), while the front-ratio is the only one with a **naked tail** (size by the worst case, never the credit).

### Entry, exit, and sizing

**Entry.** For flies, enter when implied vol is *high* (you want to be short vega at the body) and your target is clear; the body is your forecast, the width is your precision. For broken-wings, enter for a credit when you want one-sided income; choose the safe side to match what you fear. For ratios, enter only with a directional edge, a low-tail-probability conviction, and a pre-computed stop.

**Exit.** Flies are usually closed *early* — take 25–50% of max profit rather than holding for the knife-edge expiry pin (Misconception 3). Don't ride a fly to expiry unless you specifically want the pin lottery and accept the pin/assignment risk. For broken-wings and ratios run as income, set a profit-take (e.g., close at half the max profit) and an absolute-dollar stop on the risky side.

**Sizing — the rule that matters most.** Size *defined-risk* structures (flies, broken-wings) by their **max loss**: decide how many dollars of the book you'll risk on the idea, divide by the per-fly max loss, that's your contract count. Size the *front-ratio* by its **stress-level tail loss**, not its credit and not even its breakeven — pick an adverse price you could plausibly see (a 20–30% gap on a single name, more on a meme stock), compute the loss there, and size so that loss is survivable. The full framework is in [position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading). The one-line version: **defined risk is sized by the cap; naked risk is sized by the catastrophe.**

### What invalidates the trade

- **A fly is invalidated** when the stock leaves your wing range with time still on the clock and the move looks like a trend rather than noise — your low-vol/pin thesis is wrong, and the right move is to take the small defined loss, not to "adjust" by chasing.
- **A front-ratio is invalidated** the instant the stock approaches the short strike with momentum — that's your stop, and you take it *mechanically*, because the whole risk of the structure lives just beyond that point. Hesitating here is exactly how the \$30 credit becomes a \$38,000 loss.
- **A broken-wing is invalidated** when the underlying threatens the body from the risky side; close before the defined loss fully develops if your directional thesis has broken.

The thread tying all three together is the lesson from the two traders in the hook: **a credit is not an edge, and a defined risk is not the same as no risk.** The butterfly pays you in convexity for accepting a low hit rate. The broken-wing pays you a credit for accepting a defined one-sided loss. The front-ratio pays you a credit for selling a naked tail — and the only way it doesn't eventually run you over is to size it by that tail and stop out at the strike, every single time, without exception. Read the strike ladder, count the legs, find the naked one if it's there, and you'll always know which of these three precision tools you're actually holding.

## Further reading & cross-links

- [Vertical spreads: debit and credit, defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — the two-leg building blocks; a fly is a debit spread plus a credit spread sharing a strike.
- [Iron condors and credit spreads: selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — widen a fly's body into a range and you've got a condor; the credit-spread sizing logic carries straight over.
- [Gamma, the Greek that bites: curvature, convexity, and the toxic short](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — why a fly is short gamma at the body and why dealer gamma drives the pins flies bet on.
- [The net Greeks of a position: building your risk dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — how to sum the legs of any structure into one risk readout.
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — the skew that makes front-ratios pay a credit, and why that credit is the bait.
- [Calendars and diagonals: trading time and term structure](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure) — the other family of precision structures, trading the time axis instead of the strike axis.
- [0DTE and the rise of short-dated options](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure) — where cheap-convexity flies live today.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — the sizing discipline that separates a fly book from a ratio blow-up.
- [Options theory: pricing fundamentals](/blog/trading/quantitative-finance/options-theory) and [exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives) — the pricing machinery underneath every leg, and where structured payoffs go next.

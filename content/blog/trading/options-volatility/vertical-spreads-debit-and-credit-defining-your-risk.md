---
title: "Vertical Spreads: Debit, Credit, and Defining Your Risk"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to turn a directional view into a defined-risk trade with the four verticals — bull call, bear put, bull put, bear call — and choose debit or credit by reading implied versus realized volatility."
tags: ["options", "volatility", "vertical-spreads", "debit-spread", "credit-spread", "defined-risk", "delta", "skew", "theta", "trading"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A vertical spread is one long option plus one short option of the same type and expiry at different strikes; it caps your profit and your loss, which is exactly why it is the workhorse directional trade. You are no longer betting on price alone — you are buying or selling a defined slice of volatility and time.
>
> - **The four verticals**: bull call (debit), bear put (debit), bull put (credit), bear call (credit). Each one trims the open-ended tail off a single option to pay for the position or to collect cash up front.
> - **Defined risk both ways**: max loss and max profit are both fixed before you enter. A debit spread's max loss is the cash you pay; a credit spread's max loss is the strike width minus the credit you collect.
> - **The trade-off you cannot dodge**: a narrow out-of-the-money credit spread is high win-rate / low payout; a wide in-the-money debit spread is lower win-rate / higher payout. The short strike's delta is your win-rate dial.
> - **The one rule to remember**: pick the spread *type* from your direction, then pick *debit or credit* from whether implied vol is cheap or rich versus realized. Buy vol when it is cheap; sell it when it is rich.

A trader I will call the impatient bull was right about the stock and still lost money. In late spring he was convinced a \$100 name would grind higher into an event four weeks out. Implied volatility was elevated — the options were pricing a 30% annualized move — but he did not think about that. He bought the obvious thing: a single at-the-money call, the 100-strike, for \$3.59 a share, \$359 for one contract. A clean, leveraged bet on his view. If the stock ran, he would make multiples.

Two weeks later the stock had done exactly nothing. It sat at \$100. He had been *right* about direction — the thing had not fallen — and yet his call was worth \$1.93. He had lost \$166 per contract, 46% of his money, while the underlying did not move a penny. Two forces had eaten him alive: the clock (two weeks of time decay) and the vol crush (implied volatility drifted from 30% down to 22% as the event got closer and the market calmed). He was long an asset that melts, and he had paid a fat premium for volatility that then deflated. When the stock finally ticked up to \$106 at expiry, he scraped back to a \$241 profit — but he had spent two weeks underwater on a correct call, and a single bad print would have wiped him out.

Now run the tape again with the same view expressed as a **defined-risk debit spread**. Instead of buying the 100-call outright, he buys the 100-call and *sells* a 110-call against it for \$0.66, cutting his cost from \$3.59 to \$2.94. He has sold off the part of the upside he did not really expect (a move above \$110) to subsidize the part he did. Two weeks later, flat stock, vol crushed: the spread is down only \$105 instead of \$166, because the short call he is also holding *gained* value as time and vol bled out — the short leg hedges the long leg's decay. At expiry with the stock at \$106, the spread pays \$306, *more* than the naked call's \$241, because he was not overpaying for tail he never used. Same view. Smaller bleed, capped risk, and in the realistic outcome, a better result. That is the entire argument for verticals, and the rest of this post is the mechanics behind it.

![Four payoff diagrams in a grid showing bull call, bear put, bull put, and bear call spreads at expiry with max profit, max loss, and breakeven labelled](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-1.png)

## Foundations: what a vertical spread actually is

Strip away the jargon and a **vertical spread** is the simplest possible two-legged options position. You buy one option and sell another that is identical in every way except the strike price:

- **Same underlying** (both on the same stock or index).
- **Same type** (both calls, or both puts — never mixed).
- **Same expiration** (both expire on the same day).
- **Different strike** (one is higher, one is lower).

The word "vertical" comes from how an options chain is laid out: strikes run down the page in a vertical column for a single expiry, and you are picking two rows from that one column. (Contrast a *horizontal* or *calendar* spread, which uses the same strike across two different expiry columns — a different animal we will not cover here.) Everything in this post is one expiry, two strikes.

Why would anyone deliberately sell away part of their own position? Because a single option has an open-ended tail, and that tail is expensive. A naked long call keeps paying off as the stock rises toward infinity — and you pay for every dollar of that infinite upside in the premium, even though you will almost never collect it. A vertical spread is a deliberate trade: you give up the part of the distribution you do not believe in, and in exchange you either pay less (a debit spread) or get paid (a credit spread). You are trimming the position down to the slice of outcomes you actually have a view on.

A homely comparison makes the point. Suppose you want fire insurance on a \$500,000 house but the full policy is pricey. You could buy a policy that pays out *up to \$500,000* — expensive, because it covers the rare total loss. Or you could buy a policy that pays the first \$50,000 of damage and not a dollar more — far cheaper, because you have sold away the catastrophic tail you think is unlikely. A debit spread is the second policy: you buy protection (or upside) for a defined band of outcomes and refuse to pay for the extreme you do not expect. A credit spread is being the *insurer* of that defined band: you collect a premium to cover a slice of someone else's risk, and you buy a cheaper backstop policy further out so that a true catastrophe does not bankrupt you. Either way, the long-and-short pairing exists to fence off a defined range of the distribution — not the whole infinite tail.

The reason this matters so much in options specifically is that the infinite tail of a single option is not free to hold — it bleeds. Every day you own a naked long option, time decay nibbles the premium, and every tick down in implied volatility deflates it (the two forces that gutted our impatient bull). By selling the tail, a spread also sells most of that bleed. So the trim is not only about cost at entry; it is about *how the position behaves while you wait*, which is where most directional option buyers actually lose.

### The four verticals

There are exactly four. Two are built from calls, two from puts; two cost you cash to put on (debit), two pay you cash (credit). The naming follows a simple rule: **bull** spreads profit when the stock rises (or holds), **bear** spreads profit when it falls (or holds).

| Spread | Legs | Cash flow | Profits when | Type |
|---|---|---|---|---|
| **Bull call** | Long lower call, short higher call | You pay (debit) | Stock rises | Debit |
| **Bear put** | Long higher put, short lower put | You pay (debit) | Stock falls | Debit |
| **Bull put** | Short higher put, long lower put | You receive (credit) | Stock rises or holds | Credit |
| **Bear call** | Short lower call, long higher call | You receive (credit) | Stock falls or holds | Credit |

A useful pattern hides in there. The two **debit** spreads buy the strike closer to the money and sell the strike farther away — you are net long the option that has more value, so cash flows *out*. The two **credit** spreads do the reverse: they sell the strike closer to the money and buy the cheaper, farther one as protection — you are net short the more valuable option, so cash flows *in*. Whether you call it a debit or a credit is just the sign of that net premium.

### A worked anatomy of the bull call spread

Let me make this concrete with real prices. Throughout the post I price every option from the Black-Scholes model with the stock at \$100, 45 calendar days (0.123 years) to expiry, 25% implied volatility, and a 4% risk-free rate. (I am not going to re-derive Black-Scholes — for the pricing engine and where these numbers come from, see [what sets an option's price](/blog/trading/quantitative-finance/black-scholes) and the [options pricing fundamentals](/blog/trading/quantitative-finance/options-theory).)

#### Worked example: pricing a bull call spread

You are bullish on a \$100 stock over the next six weeks. You build a **5-wide bull call spread**: buy the 100-strike call, sell the 105-strike call.

- **Long 100 call** costs **\$3.74** per share.
- **Short 105 call** brings in **\$1.78** per share.
- **Net debit** = 3.74 − 1.78 = **\$1.97** per share. For one contract (100 shares) that is **\$197** out of your account.

Now the three numbers that define the trade, all settled the moment you enter:

- **Max loss** = the debit = **\$1.97/share (\$197)**. If the stock is below \$100 at expiry, both calls expire worthless and you lose exactly what you paid. Not a penny more.
- **Max profit** = strike width − debit = 5.00 − 1.97 = **\$3.03/share (\$303)**. If the stock is at or above \$105 at expiry, your long 100-call is worth \$5 and your short 105-call is worth \$0, the spread is worth its full \$5 width, and you keep \$5 − \$1.97.
- **Breakeven** = lower strike + debit = 100 + 1.97 = **\$101.97**. The stock has to clear your strike *plus* the premium you paid before you make a cent.

Your risk/reward is 3.03 to 1.97, about **1.5 to 1** — risk roughly two to make three. The intuition: you paid \$197 for the right to capture the stock's move from \$100 to \$105, and nothing above \$105. You sold the tail and bought a defined window.

![Three payoff lines showing a long call rising, a short call capping it, and the bull call spread that results, with max profit, max loss, and breakeven labelled](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-2.png)

The chart above is the single most important thing to internalize about spreads. The dashed green line is the long 100-call on its own — it rises forever once the stock clears the strike. The dashed red line is the short 105-call — flat until \$105, then it *drops* a dollar for every dollar the stock rises above \$105, because you owe that money to whoever you sold it to. Add the two together (the thick blue line) and the open-ended tail of the long call gets sawn off flat at \$105. You have converted an infinite-upside, expensive position into a finite-upside, cheaper one. That flat top *is* your max profit, and the premium you saved by selling the 105-call *is* why your max loss shrank.

### The other three, by the same logic

Once you see the bull call, the rest are reflections of it.

**Bear put spread (debit).** You think the \$100 stock falls. Buy the 100-put, sell the 95-put. You are long the more valuable (higher-strike) put and short the cheaper one, so you pay a debit. It is the mirror image of the bull call, pointed downward.

#### Worked example: pricing a bear put spread

Same \$100 stock, same 45 days, same 25% vol, now bearish. You build a **5-wide bear put spread**: buy the 100-put, sell the 95-put.

- **Long 100 put** costs **\$3.25** per share.
- **Short 95 put** brings in **\$1.35** per share.
- **Net debit** = 3.25 − 1.35 = **\$1.91** per share (**\$191** for one contract).
- **Max loss** = the debit = **\$1.91/share**. If the stock is above \$100 at expiry, both puts expire worthless.
- **Max profit** = width − debit = 5.00 − 1.91 = **\$3.09/share**. Reached if the stock is at or below \$95 at expiry.
- **Breakeven** = upper strike − debit = 100 − 1.91 = **\$98.09**.

The risk/reward is 3.09 to 1.91, about **1.6 to 1** — essentially the same shape as the bull call, just pointed down. The intuition: you paid \$191 for the right to capture the stock's fall from \$100 to \$95, and nothing below \$95, where your short put caps the gain.

**Bull put spread (credit).** You think the stock rises or at least holds. Instead of buying calls, you *sell* a put spread below the market: sell the 95-put, buy the 90-put as a backstop. Cash comes in.

#### Worked example: pricing a bull put credit spread

Same \$100 stock, same 45 days, same 25% vol. You sell a **5-wide bull put spread** below the market:

- **Short 95 put** brings in **\$1.35** per share.
- **Long 90 put** costs **\$0.41** per share (this is your insurance against a crash).
- **Net credit** = 1.35 − 0.41 = **\$0.93** per share, **\$93** into your account on entry.

The defined-risk numbers:

- **Max profit** = the credit = **\$0.93/share (\$93)**. If the stock is at or above \$95 at expiry, both puts expire worthless and you keep every cent of the credit.
- **Max loss** = width − credit = 5.00 − 0.93 = **\$4.07/share (\$407)**. If the stock collapses below \$90, you owe the full \$5 width on the short put but your long 90-put caps the damage, so you lose \$5 − \$0.93.
- **Breakeven** = short strike − credit = 95 − 0.93 = **\$94.07**. As long as the stock holds above \$94.07, you make money.

Notice the cash flow flipped. You got paid \$93 to take the trade, and your job is now to *keep* it. You make your maximum profit by doing nothing — if the stock just stays above \$95, the options decay to zero and the \$93 is yours. The intuition: a credit spread is a bet that the stock will *not* fall below your short strike, and you are paid up front for taking that bet.

**Bear call spread (credit).** The fourth one. You think the stock falls or holds, so you sell a call spread above the market: sell the 100-call, buy the 105-call as protection. Cash comes in, and you profit as long as the stock stays below your short strike plus the credit. It is the bull put's mirror, pointed downward.

Four spreads, two pairs of mirror images, one shared skeleton: long one strike, short another, both legs the same type and expiry. The rest of this post is about *which* one to use and *how wide* to make it.

## Defined risk: why both ends are capped, and why that matters

The defining feature of every vertical is that **both your maximum profit and your maximum loss are known and fixed the moment you enter**. This sounds modest. It is actually the whole reason spreads are the bread-and-butter directional trade for serious options traders rather than naked single options.

Consider what you are *not* exposed to:

- **A naked long option** has capped loss (the premium) but pays a steep price in time decay and vol sensitivity for an open-ended upside you rarely capture. The impatient bull from the hook lived this.
- **A naked short option** has the opposite problem — you collect a premium, but a short call has *theoretically unlimited* loss and a short put can lose almost the entire strike if the stock craters. One bad gap and you are explaining a margin call to your spouse.

A vertical spread takes the short option's premium income and *neutralizes its catastrophic tail* by buying a cheaper option further out as a backstop. The long 90-put in our credit spread is not there to make money — it is there to convert "I could lose \$95 a share if this stock goes to zero" into "I can lose exactly \$4.07 a share, full stop." That cap is what lets you size the position rationally. You know your worst case to the penny, so you can risk a fixed slice of your account and never be surprised.

This is why the cliché holds: **verticals are the workhorse.** They express a directional view, they are cheaper than the equivalent naked long, they are far safer than the equivalent naked short, and their risk is knowable in advance. Almost every defined-risk options strategy — iron condors, butterflies, broken-wing structures — is built by stacking verticals. Learn the vertical and you have learned the atom from which the rest are assembled. (We will assemble two credit verticals into an [iron condor in a later post](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range), and slice verticals finer into [butterflies and ratio spreads](/blog/trading/options-volatility/butterflies-ratio-spreads-and-broken-wings-the-precision-tools).)

### Debit vs credit: the same payoff, a different bank statement

Here is a fact that surprises people the first time: **a bull call debit spread and a bull put credit spread can produce nearly identical payoff diagrams.** Both are bullish. Both have a defined max loss and max profit. If you set the strikes right, the shape of the P&L line at expiry is the same. So why are there two of them?

Because the *cash flow* and the *margin* are opposite, even when the payoff is the same.

![Side by side comparison of a bull call debit spread and a bull put credit spread showing same payoff but opposite cash flow and margin](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-4.png)

Walk the two columns of the chart above. On the left, the debit version: you pay \$1.97 up front, that \$1.97 *is* your max loss, your max profit is \$3.03, and the broker ties up exactly the \$1.97 you paid as buying power. On the right, the credit version: you collect \$0.93 up front, your max loss is the \$4.07 width-minus-credit, your max profit is the \$0.93 you collected, and the broker ties up the \$4.07 of risk as margin (you do not get to spend the credit — it sits as a buffer against the position).

The practical differences that follow from this:

- **Capital efficiency.** The debit spread ties up \$197 to make up to \$303. The credit spread ties up \$407 to make up to \$93. The debit version is using less buying power for more potential profit *in this particular pairing* — but that is because it sits at a different point on the probability curve (more on that below). Compare apples to apples by matching the win-rate, not the strikes.
- **What "winning" looks like.** With a debit spread you need the stock to *move* in your favor past breakeven. With a credit spread you can win by the stock doing *nothing* — time decay alone carries you. That difference in temperament matters: credit sellers are happy with a quiet market; debit buyers need movement.
- **Assignment exposure.** The credit spread's short leg sits closer to the money, so it carries more early-assignment risk (we will get to assignment in the playbook). The debit spread's short leg is further out, so it is calmer.

The honest framing: **debit and credit are not different strategies, they are different ways to finance the same view.** Pay for it, or get paid for it. Which one you pick comes down to one question we will answer in the playbook — is implied volatility cheap or rich? — but first you have to understand the trade-off that strikes themselves impose.

#### Worked example: a debit and a credit at the same strikes are the same position

This is the deepest version of "same payoff, different cash flow," and it falls straight out of put-call parity (proved in the [put-call parity](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews) post). Take a **bull call spread at the 100/105 strikes** and a **bull put spread at the *same* 100/105 strikes**, and watch the numbers.

- **Bull call 100/105 (debit)**: long 100-call \$3.74 − short 105-call \$1.78 = **\$1.97 debit**.
- **Bull put 105/100 (credit)**: short 105-put \$5.55... actually, short the 105-put and long the 100-put gives a **\$3.01 credit**.

Now add them: 1.97 (paid) + 3.01 (received) = **\$4.98**. And the present value of the \$5 strike width, discounted at 4% for 45 days, is \$5 × e^(−0.04 × 0.123) = **\$4.98**. They are equal — not approximately, exactly, up to rounding. That is no coincidence: a bull call spread *plus* the opposing bull put spread at the same strikes is a synthetic position that always pays the full width at expiry, so today it must cost the present value of that width. The intuition: **at the same strikes, the debit version and the credit version are literally the same payoff** — the only thing that differs is whether you front the cash (debit) or post the width as margin and collect the cash (credit). The choice between them is a financing and margin decision, never a payoff decision.

When you *do* see a difference in the payoff, it is because you chose *different* strikes — a bull call debit centered at the money versus a bull put credit sitting below the market are different trades sitting at different points on the probability curve. Same strikes, same trade; different strikes, different point on the curve.

## The probability-vs-payout trade-off

Every vertical forces a choice that no amount of cleverness can escape: **you can have a high probability of winning, or a high payout when you win, but not both.** This is not a market inefficiency you can arbitrage away — it is a direct consequence of the math. Options are (roughly) fairly priced, so a structure that wins more often must pay less when it does, and vice versa.

The cleanest way to *quantify* your win probability is through the **delta of your short strike**. Delta, which we cover in depth in [delta: direction exposure and the hedge ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio), is the option's sensitivity to a \$1 move in the stock — but it has a second, enormously useful interpretation: **delta is approximately the risk-neutral probability the option finishes in the money.** A short put with a delta of −0.25 has roughly a 25% chance of expiring in the money. So your **probability of profit (POP)** on a credit spread — the chance the short strike stays out of the money and you keep the credit — is approximately **1 minus the short strike's delta**.

#### Worked example: POP from the short-strike delta

Back to the bull put credit spread — short the 95-put, long the 90-put. The 95-put's delta is **−0.25**.

- **Probability the short put expires in the money** ≈ |delta| = **25%**.
- **Probability of profit (POP)** ≈ 1 − 0.25 = **75%**.

So that credit spread wins roughly **three times out of four** — but look at what it pays. Your max profit is the \$0.93 credit and your max loss is \$4.07, so when you lose, you lose about four times what you make when you win. The expected value is (0.75 × \$0.93) − (0.25 × \$4.07) ≈ \$0.70 − \$1.02 ≈ **−\$0.32 per share** in this fair-value world. The high win-rate is *real*, but it is paid for with a brutal loss ratio. The intuition: a 75%-win-rate credit spread is not free money — the market charges you, in the size of the losses, exactly for the comfort of winning often.

(That negative expected value is the fair-value baseline. The reason vol-sellers run these *anyway* is the variance risk premium — implied vol prints systematically above the realized vol that follows, so the real-world edge is a little better than the fair-value math suggests. That is a whole post on its own; see [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt). For this post, the point stands: high win-rate is bought with a bad loss ratio.)

Now slide the short strike around and watch the trade-off move:

![Trade-off curve showing win-rate rising as max payout falls across a range of short strikes for a bull put spread](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-3.png)

The chart sweeps the same 5-wide bull put spread across short strikes from just out of the money down to deep out of the money:

- **Short strike near the money** (95-ish to 99, delta around 0.40): win-rate around 58–60%, but the credit is fat — the payout can be **half the width**. You win less often, but each win is big relative to the loss.
- **Short strike at the classic delta-25** (the 95-put): about **75% win-rate**, payout around **23%** of max loss. The popular middle ground.
- **Short strike deep out of the money** (90 or below, delta under 0.10): **90%+ win-rate**, but a microscopic payout — you risk \$4.67 to make \$0.33. One loss erases fourteen wins.

There is no free lunch on this curve. Pushing your strikes further out buys you a higher win-rate at a worse-and-worse loss ratio. Pulling them in does the reverse. **The wide in-the-money debit spread and the narrow out-of-the-money credit spread are just two ends of this one curve.** A trader who brags about an 85% win rate is telling you about *one axis only* — ask what they lose on the 15%, and the curve will tell you it is probably enough to wipe out the wins.

This is the single most important number to have in your head before you place a vertical: **what is the delta of my short strike, and therefore roughly how often does this win and how much does it pay?** Everything else is detail.

## How width, strikes, IV, and skew set the payoff

Three knobs shape every vertical: how far apart the strikes are (**width**), where you center them (**strike selection**), and the **volatility** you trade them at. Let me take each.

### Width sets the leverage

Widening a spread — moving the short strike further from the long strike — does three things at once, all of which the model computes for you.

![Grouped bars showing max profit and max loss rising and breakeven shifting as a bull call spread is widened from 2.5 to 20 points](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-6.png)

#### Worked example: widening a bull call spread

Hold the long 100-call fixed and push the short strike out:

| Width | Debit (max loss) | Max profit | Breakeven | R:R |
|---|---|---|---|---|
| 2.5-wide (short 102.5c) | \$1.12 | \$1.38 | \$101.12 | 1.24 |
| 5-wide (short 105c) | \$1.97 | \$3.03 | \$101.97 | 1.54 |
| 10-wide (short 110c) | \$3.02 | \$6.98 | \$103.02 | 2.31 |
| 20-wide (short 120c) | \$3.67 | \$16.33 | \$103.67 | 4.45 |

As you widen, the **debit grows** (you are buying back less of the tail), so your **max loss grows** — but your **max profit grows faster**, so the risk/reward improves from 1.24 to 4.45. The catch is in the breakeven column: a wider spread needs a *bigger* move to pay off (\$101.12 versus \$103.67), and its potential loss is larger in absolute dollars. The intuition: a wide spread is closer to a naked long — more upside, more cost, needs more movement; a narrow spread is a tight, cheap, high-probability scalp of a small move.

Notice, too, that the debit does *not* keep growing one-for-one with width. Going from a 15-wide to a 20-wide spread adds only ~\$0.18 of cost (\$3.49 to \$3.67) for \$5 more of potential profit, because the far-out short call you are giving up is nearly worthless. There is a point of diminishing protection: very wide spreads barely differ from the naked long, so you stop getting the cost reduction that was the whole point.

**Strike placement is a separate dial from width.** You can build a 5-wide debit spread *in the money* (long 95-call, short 100-call), *at the money* (long 100, short 105), or *out of the money* (long 105, short 110), and each is a different bet. The in-the-money debit spread costs the most and behaves the most like owning stock — high probability the long leg pays, but you front a big debit and your profit is mostly already "baked in" as intrinsic value, so it is low-risk, low-reward, and barely sensitive to vol. The out-of-the-money debit spread costs the least, has the lowest probability of paying, but offers the fattest risk/reward and the most leverage to a move — it is the lottery-ticket end. The at-the-money version sits in between and carries the most *vega and gamma*, because at-the-money options are where those Greeks peak. The general rule: **the deeper in the money you place a debit spread, the more it behaves like stock (high probability, low payout, low vol-sensitivity); the further out of the money, the more it behaves like a long option (low probability, high payout, high vol-sensitivity).** That is the same probability-vs-payout curve from before, now expressed through *where* you center the spread rather than how wide you make it. Width sets the size of the bet; placement sets where on the distribution it lives.

### IV and skew set the *price* of the spread

Volatility enters twice. First, the **level** of implied vol sets how expensive every option is. High IV inflates both legs — but it inflates the leg nearer the money *more* (it has more vega). So in a high-IV environment, **debit spreads cost more** (the long leg you are buying is pricier) and **credit spreads pay more** (the short leg you are selling is pricier). This is the first half of the debit-or-credit decision: if vol is *expensive*, you would rather be the seller (credit); if vol is *cheap*, you would rather be the buyer (debit). We will formalize that in the playbook against the [implied-vs-realized framework](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options).

Second, the **skew** — the fact that different strikes trade at different implied vols. In equity indices, out-of-the-money puts trade at *higher* implied vol than at-the-money or out-of-the-money calls, a downward-sloping "smirk" driven by crash fear. This is covered in full in [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more). For verticals, skew is not a curiosity — it directly changes the credit you collect.

#### Worked example: skew's effect on a put credit spread

Take the bull put spread again — short the 95-put, long the 90-put — but now price it with realistic *skew* instead of a flat 25% across all strikes. With at-the-money vol at 17%, the 95-put (slightly OTM) trades at about 19% and the 90-put (further OTM) at about 22%, because the deeper-OTM puts carry the richer crash premium.

- **In a no-skew world** (everything at the 17% ATM vol): short 95-put = \$0.54, long 90-put = \$0.07, credit = **\$0.47**.
- **With skew**: short 95-put = \$0.72 (sold at the inflated 19%), long 90-put = \$0.25 (bought at the even-more-inflated 22%), credit = **\$0.47**.

The headline credit barely moves — but look at *where* the money is. The strike you **sell** is fattened by skew (\$0.54 → \$0.72, a \$0.18 boost you pocket), and the strike you **buy** as protection is fattened even more (\$0.07 → \$0.25). The skew gives with one hand (richer short strike) and takes with the other (richer long wing). The intuition: **a credit spread is "short skew" on the leg you sell and "long skew" on the leg you buy** — you benefit from the steep near-strike premium but pay for the steeper far-wing premium, so the net edge from skew depends on exactly how steep the curve is between your two strikes. The flatter the skew between them, the more of the near-strike premium you keep.

The practical takeaway: **put credit spreads in equity indices are selling the part of the skew that is bid for crash protection** — which is precisely why they tend to carry a structural edge, and precisely why that edge evaporates in a crash, when the skew you sold reprices violently against you.

### The net Greeks of each vertical

A vertical's behavior between now and expiry is summarized by its **net Greeks** — the sum of the two legs' Greeks. This is the live risk dashboard we build out fully in [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard); here is how it specializes to verticals.

![Signed bar chart of the net delta, gamma, vega, and theta of a bull put credit vertical](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-5.png)

#### Worked example: the net Greeks of a credit vertical

For the bull put credit spread (short 95-put, long 90-put), sum each Greek across the two legs:

- **Net delta = +0.15.** Mildly bullish — you want the stock up or flat. (The short put is bullish-positive delta; the long put claws a little back.) Far less directional than a naked long, which is the point.
- **Net gamma = −0.016.** *Short* gamma. Your delta works against you on a fast move — exactly the curvature risk we dissect in [gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short). A credit spread does not like the stock moving fast.
- **Net vega = −0.05 per vol point.** *Short* vega. Rising implied vol hurts you, falling vol helps. You are short volatility — consistent with collecting a credit.
- **Net theta = +0.012 per day.** *Long* theta. Time decay works *for* you; every day the stock sits still, you make a little money.

The signature is the seller's fingerprint: **short gamma, short vega, long theta.** A debit vertical flips the signs — long gamma, long vega, short theta — the buyer's fingerprint. The intuition: a credit spread is a small, defined-risk way to *sell volatility and collect the clock*, and a debit spread is a small, defined-risk way to *buy volatility and a directional move*.

### How spreads dampen vega and theta versus a single option

Here is the under-appreciated benefit of the two-leg structure: **the short leg cancels most of the long leg's Greeks, leaving you with a cleaner directional bet.** A naked long call has a big positive vega and a big negative theta — it bleeds hard on time and gets clobbered by vol crush (the impatient bull's exact problem). A debit *spread* nets the short call's negative vega against the long call's positive vega, so the position's net vega and net theta are a *fraction* of the single option's.

#### Worked example: how a spread dampens vega and theta

Compare the naked 100-call to the bull call 100/105 spread, leg by leg, at our base case (S=100, 45 days, 25% vol):

| Greek | Naked 100-call | Bull call 100/105 spread | Reduction |
|---|---|---|---|
| Net delta | +0.54 | +0.22 | ~60% less directional |
| Net vega (per vol point) | +\$0.139 | +\$0.013 | ~91% less vol-sensitive |
| Net theta (per day) | −\$0.044 | −\$0.006 | ~87% less time decay |

The numbers are dramatic. The naked call loses about **\$0.044 a share every day** to time decay and swings about **\$0.139 a share for every vol point** of implied volatility. The *spread* loses only **\$0.006 a day** and moves only **\$0.013 per vol point** — roughly a *tenth* of the exposure on each. The short 105-call you sold has its own positive vega and negative theta that nearly cancel the long leg's. The intuition: a debit spread keeps most of the directional bet (it is still net positive delta) while shedding almost all of the vega and theta bleed — it is a *cleaner* expression of "I think it goes up" that does not also force you to be a hostage to implied volatility and the calendar.

That is precisely why, in the hook, the spread lost only \$105 to the vol crush while the naked call lost \$166 — the short leg absorbed most of the IV deflation. **A spread is a more vol-neutral, more theta-neutral way to bet on direction.** You give up some upside; in exchange, you stop bleeding to the two forces that kill most naked-option buyers.

## Common misconceptions

**"A credit spread is safer because you start with cash in hand."** No — and this one bankrupts people. You collected \$0.93, but your max loss is \$4.07. The cash you received is not profit; it is the *most* you can make, and you are risking more than four times that to earn it. The "free money on entry" feeling is exactly backwards: a credit spread is a position where you can lose four-plus times your max gain, and it loses precisely when markets gap — when everything else you own is also losing. The credit is compensation for that tail risk, not a gift.

**"Buying a debit spread is just cheap leverage on my view."** Cheaper than a naked long, yes — \$197 versus \$374, a 47% saving — but the saving is not free. You sold the entire tail above your short strike. If the stock rips to \$120, the naked call makes \$16.26 a share and your spread is capped at \$3.03. You traded away the home-run outcome to lower your cost. That is often the right trade, but call it what it is: a *narrowing of your distribution*, not a discount.

**"Higher win-rate means a better strategy."** A 90%-win-rate vertical and a 58%-win-rate vertical can have *identical* expected value — they are just two points on the same probability-vs-payout curve. The 90% version loses fourteen times its win on the rare loss; the 58% version loses about as much as it makes. Win-rate in isolation is a vanity metric. The only thing that matters is **win-rate times average win versus loss-rate times average loss** — the full expectancy. A trader selling deep-OTM credit spreads can win 47 trades in a row and give it all back on the 48th.

**"My max loss is the credit I'd lose if I'm wrong."** A common beginner error on credit spreads. If you sell the 95/90 put spread for \$0.93 and the stock falls to \$88, you do *not* simply lose the \$0.93 — you lose the full width minus the credit, **\$4.07 per share**. The credit caps your *profit*, not your *loss*. Your loss is capped by the *long leg* (the 90-put), at width minus credit. Always quote your risk as width minus credit, never as the credit.

**"Verticals don't have assignment risk because they're defined-risk."** The *position* is defined-risk, but the *short leg can be assigned early* — especially if it goes in the money near expiry or around a dividend. Walk a concrete case: you sold the 95/90 put spread, the stock drops to \$92, and the holder of your short 95-put exercises early. Overnight you are *assigned* 100 shares of stock at \$95 — a \$9,500 long stock position you did not ask for — while your long 90-put sits there as your only remaining hedge. Your spread has temporarily morphed into "long 100 shares + long one 90-put," which is a completely different risk profile (and may trigger a margin call if you do not have the cash). Your *eventual* max loss is still the \$4.07 width-minus-credit, but only *if you act*: you must sell the assigned shares and the orphaned long put, or exercise the long put to flatten, the next morning. Leave it unmanaged over a weekend and the "defined risk" is defined only on paper. Defined risk assumes you do not fall asleep at the wheel — the standard defense is simply to close any in-the-money short leg before expiration week and to avoid holding short calls through an ex-dividend date.

## How it shows up in real markets

**The earnings vol-crush trap (the hook, generalized).** The single most common way retail traders lose money on a correct view is buying naked options into an event when implied vol is elevated, then watching the post-event vol crush evaporate their premium even when direction is right. Our impatient bull lost 46% on a flat stock. The defined-risk fix is to express the same view as a debit spread — which cuts the vega exposure by roughly two-thirds and the cash at risk by nearly half — or, if you think the event is *over-priced*, to *sell* a credit spread and harvest the crush. Around earnings, traders quantify the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) the options are pricing and then sell credit spreads with short strikes *outside* that move, betting the realized move stays inside the implied one — the [vol-crush trade](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) in spread form.

**The "0.16-delta short strike" convention.** A huge amount of systematic credit-spread selling — and the entire premium-selling cottage industry — centers on short strikes around delta 0.16 (one standard deviation out) to delta 0.30. That convention is not arbitrary: it sits at the high-win-rate, moderate-payout part of the trade-off curve, and it is where the skew premium on index puts is fattest relative to the risk. The discipline that separates survivors from blow-ups is not the entry — it is the loss management when the 16% case shows up, because on this part of the curve a single uncapped loss undoes a long winning streak.

**The 2018 "Volmageddon" lesson, in vertical terms.** On February 5, 2018, the VIX spiked from the mid-teens to over 37 in a session, and a generation of short-vol traders learned what "short gamma, short vega" means when it all hits at once. Anyone who was *naked* short vol was destroyed; anyone who was short vol through *defined-risk verticals* lost their max loss and lived. That is the entire case for the long protective leg: it converts an account-ending tail into a survivable, sized loss. The credit spread sellers who survived 2018 did so because they had bought the wing, even though that wing had felt like a waste of money for years. Defined risk is insurance you resent paying for until the one day it saves you.

**Index put credit spreads as a yield strategy.** Because index skew is persistently bid (crash insurance is structurally over-demanded), selling out-of-the-money put credit spreads on the S&P has behaved, over long stretches, like collecting a steady premium — a few good months for every quiet stretch, punctuated by a sharp drawdown when the market drops. It is, in spread form, a bet that implied vol keeps printing above realized vol. The defined-risk version (a spread, not a naked put) trades away some of that premium for a known worst case — and that trade is exactly why it is the version that is still standing after each crash. We pull this thread all the way through in [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt).

**Earnings, expressed two ways.** Suppose a stock trades at \$100 the day before earnings, with the options pricing a 30% implied vol — an elevated number that will collapse the morning after the report. Two traders share the same bullish lean but read the vol differently. The first thinks the move will be *bigger* than priced, so she buys a bull *call debit* spread: she wants the directional move and accepts that she is paying for vol, but the short leg cuts her vega so the post-earnings crush does not gut her if she is right on direction but the move is modest. The second thinks the move will be *smaller* than priced — the classic "earnings are over-priced" view — so he sells a bull *put credit* spread with the short strike below the implied move: he is harvesting the vol crush directly, winning if the stock simply holds above his short strike. Same stock, same lean, opposite vol read, opposite cash flow. The defined-risk structure lets *both* of them size the trade to a known max loss across the event gap, which a naked option around earnings does not. This is the cheap-vs-rich decision from the playbook, made live on the one night a year when the vol crush is most violent.

**Why "free" credit spreads are a trap retail keeps falling for.** Brokerage marketing and social media lean hard on the high-win-rate face of credit spreads — "win 80% of the time!" — because it sells. What it omits is the loss ratio on the other 20%, and the correlation of those losses: credit spreads on equity indices tend to lose *together*, on the same down-day, because they are all short the same crash. A book of "diversified" index put credit spreads is not diversified at all when the market gaps 5% — every one of them is tested at once. The traders who run these strategies for a living size each spread small precisely because they know the losses cluster, and they treat the defined-risk wing as non-negotiable. The recurring retail blow-up is someone who saw the 80% win-rate, sized too big, skipped the protective wing (or sold naked), and met the 20% on a day when it arrived all at once.

## The playbook: how to trade verticals

Here is the decision and execution flow, start to finish.

![Decision tree choosing a vertical from directional view then cheap or rich implied volatility](/imgs/blogs/vertical-spreads-debit-and-credit-defining-your-risk-7.png)

**Step 1 — Pick the spread type from your direction.** Bullish or stable-to-up → bull call (debit) or bull put (credit). Bearish or stable-to-down → bear put (debit) or bear call (credit). The direction picks the *family*; it does not yet pick debit or credit.

**Step 2 — Pick debit or credit from implied versus realized vol.** This is the volatility-first heart of the decision, and it ties straight back to [implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options):

- **If implied vol is *cheap* relative to expected realized vol** — options are under-pricing the move you expect — **buy a debit spread.** You are buying cheap volatility plus your directional view. You need the move to happen, but you are not overpaying for it.
- **If implied vol is *rich* relative to expected realized** — options are over-pricing the move — **sell a credit spread.** You are selling expensive volatility and getting paid the clock. You win if the move stays *smaller* than what is priced. This is the more common regime for index options, thanks to the variance risk premium.

The mnemonic: **buy spreads when vol is cheap, sell spreads when vol is rich.** Direction tells you up or down; vol tells you buy or sell.

**Step 3 — Pick the short strike (your win-rate dial).** Set the short strike by delta. Delta 0.30–0.40 for a more aggressive, higher-payout / lower-win-rate trade; delta 0.16–0.25 for the high-win-rate / lower-payout standard; below 0.10 only if you genuinely accept fourteen wins to fund one loss. Know your POP (≈ 1 − short delta) before you click.

**Step 4 — Pick the width (your leverage dial).** Wider = more max profit and better risk/reward, but more dollars at risk and a further breakeven. Narrower = cheaper, tighter, higher-probability scalp. Match the width to the size of the move you expect and to the dollar risk you are willing to take.

**Step 5 — Size to the max loss.** This is the gift of defined risk: your worst case is known to the penny, so size it directly. Risk a fixed fraction of the account per trade — many traders cap any single defined-risk vertical at 1–2% of equity — using *max loss* as the risk number. (For the sizing math, see [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion).) Never size off the credit you collect; size off what you can lose.

**Step 6 — Take profit early, around 50%.** The standard credit-spread discipline is to close the trade once you have captured roughly **half the maximum profit**, rather than holding to expiry for the last few cents. The reasoning is risk-adjusted: by the time you have made 50% of the credit, most of the easy theta is harvested and you are now holding gamma and pin risk near expiry for a shrinking reward. Closing at 50% frees capital, reduces the time you are exposed to a tail event, and improves your risk-adjusted return even though it lowers your raw win-rate-to-expiry. For debit spreads, the mirror discipline is to take profits when the move you expected has largely played out rather than waiting for the full max profit, which only arrives if the stock pins above your short strike.

**Step 7 — Manage the tested side: the roll.** When the stock moves against you and threatens your short strike (the "tested" side), the standard adjustment is the **roll**: close the threatened spread and open a new one further out in time (and sometimes further out in strike), ideally for a net credit so you are paid to extend your duration. A roll is not a magic fix — it adds time and can add risk — but it buys the position room and resets the clock. The discipline is to roll *for a credit* and to know your adjusted max loss; rolling for a debit just throws good money after bad.

Make that concrete. You sold the 95/90 put spread for \$0.93, and the stock has slid to \$96 with two weeks left — your short 95-put is now being tested and the spread is showing a loss. You have three honest choices. **One**, take the loss: buy the spread back for, say, \$1.80 and book a \$0.87 loss, well inside your \$4.07 max. **Two**, roll out: close this spread and sell the *next month's* 95/90 spread, collecting another credit (say \$1.20) so your combined credit now exceeds the buy-back cost — you are paid to give the trade more time, and your breakeven improves, at the cost of staying exposed longer. **Three**, roll down-and-out: move the short strike to 92 in the next expiry, lowering your probability of being tested again but collecting less credit. The wrong move — the one that turns a defined-risk trade into a blow-up — is to roll for a *debit* into a bigger size, "doubling down" to lower your average. That converts a known, sized loss into an open-ended one and defeats the entire purpose of trading verticals. The roll is a tool for buying time on a still-valid thesis, not a way to refuse to be wrong.

**Step 8 — Watch assignment on the short leg.** Near expiry, if your short option is in the money, expect possible early assignment — especially short calls before an ex-dividend date and short puts that go deep in the money. If assigned, act immediately: close the resulting stock position and the now-orphaned long leg so your risk stays defined. The contract mechanics (multiplier, settlement, exercise style) are covered in [the options chain and contract mechanics](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options); the rule of thumb is to close or roll any in-the-money short leg before expiration week rather than gamble on assignment.

**The invalidation.** A vertical is invalidated when (a) the stock breaks decisively through your short strike with time still on the clock — your thesis is wrong and the position is at or near max loss — or (b) the volatility regime flips against you (you sold a credit spread and implied vol is now exploding higher; or you bought a debit spread into cheap vol and the move you expected has clearly failed to materialize). In either case the defined-risk structure has done its job: it told you the worst case in advance, and your only decision is whether to take the max loss, close early to salvage value, or roll. The structure removed the one thing that destroys options traders — the unbounded surprise.

That is the whole craft of the vertical. Pick the direction, read the volatility, dial the win-rate with the short strike's delta, dial the leverage with the width, size off the known max loss, and manage the tested side. You will not hit home runs — you sold the tail to make the trade defined and cheap — but you will know, every single time, exactly what you can lose. In options, that knowledge is most of the edge.

## Further reading & cross-links

Within this series:

- [Calls, puts, and the payoff diagram: the language of options](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) — the single-leg payoffs that verticals are built from.
- [Delta: direction exposure and the hedge ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) — why the short strike's delta is your win-rate dial.
- [The net Greeks of a position: building your risk dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — how to sum the two legs into a live risk read.
- [Gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the short-gamma risk of a credit vertical.
- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — the skew you sell in a put credit spread.
- [Implied vs realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the cheap-vs-rich decision behind debit vs credit.
- [Long calls and puts: the pure directional bet and why it usually loses](/blog/trading/options-volatility/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses) — the naked long the vertical improves on.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the structural edge of credit spreads.

Coming next in the series:

- [Iron condors and credit spreads: selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — two credit verticals combined into a range-bound seller.
- [Butterflies, ratio spreads, and broken wings: the precision tools](/blog/trading/options-volatility/butterflies-ratio-spreads-and-broken-wings-the-precision-tools) — slicing verticals finer for a precise view.

The theory underneath:

- [What sets an option's price: the Black-Scholes model](/blog/trading/quantitative-finance/black-scholes) — the pricing engine behind every number here.
- [Options pricing fundamentals](/blog/trading/quantitative-finance/options-theory) — the foundations of option valuation.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — how to size a defined-risk vertical off its max loss.

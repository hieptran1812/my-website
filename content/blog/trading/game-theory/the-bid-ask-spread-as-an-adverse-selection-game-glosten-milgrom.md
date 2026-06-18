---
title: "The Bid-Ask Spread as an Adverse-Selection Game: Glosten-Milgrom"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "The gap between the buy price and the sell price is not a fee for paperwork, it is the exact amount a competitive market maker must charge to break even against the traders who know more than he does."
tags: ["game-theory", "trading", "glosten-milgrom", "bid-ask-spread", "market-microstructure", "adverse-selection", "market-making", "bayesian-updating", "price-discovery", "order-flow"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The bid-ask spread is the price of being on the other side of a trade you cannot read; a competitive market maker sets it so that the money he loses to informed traders is exactly paid back by the money he makes off everyone else.
>
> - **Every order is news.** A buy is mild evidence the asset is worth more; a sell is mild evidence it is worth less. The market maker cannot tell an informed trader from a noise trader, so he treats *all* of them as partly informed.
> - **The spread is the lemons discount, made exact.** The market maker quotes ask = E[value | someone buys] and bid = E[value | someone sells]. With a \$110-or-\$90 asset, a coin-flip prior, and 30% informed flow, that is ask \$103, bid \$97 — a \$6 spread that exists purely to cover adverse selection, not order-processing cost.
> - **It widens with information and uncertainty.** More informed traders, or a wider range of possible values, both pull the ask and bid apart. At 100% informed the spread blows out to the full \$20; at 0% informed it collapses to zero.
> - **Each trade moves the price.** The market maker is a Bayesian: he updates his belief after every fill, so the mid-price walks toward the true value as informed flow leaks in. That walk *is* price discovery.
> - **The one rule:** when you cross the spread, you are paying the market maker an insurance premium against the chance that you are the one who knows something. If you genuinely don't, that premium is the cost of admission, and your edge has to clear it.

In the early 1980s, two economists, Lawrence Glosten and Paul Milgrom, set out to answer a question that sounds almost too simple to be interesting: why does the price to *buy* a stock differ from the price to *sell* it at the very same instant? You can pull up any quote screen and see it — a stock might show a price of \$100.02 to buy and \$99.98 to sell. That four-cent gap is the *bid-ask spread*, and for most of finance's history the standard explanation was a shrug: it covers the dealer's costs. Rent, clerks, the bother of holding inventory. A toll for using the road.

Glosten and Milgrom did not believe the toll story, and they were right not to. The toll story cannot explain why the spread on a sleepy utility stock is a penny while the spread on a biotech the night before a drug-trial readout is fifty cents — the rent is the same. It cannot explain why spreads gap wider in the seconds before an economic data release and snap back the moment the number prints. It cannot explain why a market maker, the person quoting both prices, *wants* the spread there even when his desk, his screens, and his clerks are already paid for. The toll story treats the spread as a cost of doing business. The truth is that the spread is a **defense against the people on the other side of the trade** — specifically, the ones who know something you don't.

This post builds their 1985 model from absolutely nothing. We will define a market maker, define what it means for a trader to be "informed," and then watch a competitive, break-even dealer reason his way to a spread using nothing but conditional probability — the same Bayesian arithmetic you'd use to update your guess about the weather after seeing someone carry an umbrella. The payoff is one of the most important ideas in all of market microstructure: **the spread is the lemons discount made precise.** It is what asymmetric information costs, denominated in cents, and it is the clearest case in this whole series of a price that exists only because of *who is on the other side*.

![A buy nudges the market maker belief up and a sell nudges it down, so the ask and bid straddle the prior mean](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-1.png)

The figure above is the entire model in one picture. The market maker starts with a *prior* — a best guess of the asset's value, here \$100, sitting in the middle. Then an order arrives. If it is a buy, he nudges his belief up, because buys come disproportionately from people who think the thing is cheap; that nudged-up number is his **ask**, \$103. If it is a sell, he nudges his belief down to his **bid**, \$97. He does not know whether this particular trader is informed or just flipping a coin — but he knows the *mix*, and the mix is enough. The gap between the two nudges, \$6, is the spread. Everything else in this post is the arithmetic behind those two nudges and what happens when you make the mix more dangerous.

## Foundations: the market maker, the informed, and a spread built from belief

Before any math, we need four characters and one rule. Let us meet them one at a time, defining every term the first time it appears, because the whole model is just these pieces interacting.

The cleanest everyday analogy is a used-car lot — the same one Akerlof used for the lemons problem. You run the lot. People come to you all day, some wanting to *sell* you their car, some wanting to *buy* one off your lot. You can't perfectly inspect any car. Here's the trap: the people most eager to *sell* you their car at your price are disproportionately the ones who know it's a lemon — they have a reason to dump it. And the people most eager to *buy* a specific car off your lot are disproportionately the ones who spotted it's secretly a gem. So you can't quote one fair price; you have to quote a *low* buy-from-you price (to protect against the lemons being dumped on you) and a *high* sell-to-you price (to protect against the gems being plucked from you). The gap between those two prices is your spread, and it exists entirely because some of your counterparties know more about the car than you do. Glosten-Milgrom is that lot, made mathematical, with shares instead of cars and probabilities instead of hunches.

**The asset.** There is one thing being traded — call it a share. Right now nobody knows exactly what it is worth, but everyone agrees it will turn out to be one of two values: a *high* value (we'll use \$110) or a *low* value (\$90). Maybe a court ruling lands tomorrow; maybe an earnings number drops at the close. The truth already exists; it just hasn't been revealed. Before the reveal, the market's shared belief is a *prior probability* — say a 50/50 coin flip between high and low. The **prior mean**, the value-weighted average of the two outcomes, is then `0.5 × 110 + 0.5 × 90 = 100` dollars. That \$100 is what the asset is "worth" to someone with no special knowledge.

**The market maker.** The *market maker* (MM), also called a *dealer*, is the person — these days, the algorithm — who stands ready to trade at all times. He posts two prices: a **bid**, the price at which he will *buy* from you, and an **ask** (or *offer*), the price at which he will *sell* to you. You, the incoming trader, hit one or the other. The MM does not get to choose his counterparty or wait for a better one; his job is to always be there. He is the house in the sense that everyone trades against him — but unlike a casino, he does not know the odds are in his favor. He has to *make* them favorable by setting his two prices correctly.

**The two crucial assumptions about the MM.** First, he is *competitive*: there are many dealers fighting for the order flow, so he cannot post a fat spread and pocket the difference — a rival would undercut him. Competition drives him to a **zero-profit** condition: in expectation, he makes nothing. That sounds strange — why would anyone make markets for zero profit? — but it is the right idealization, because it isolates the one force we care about. If the spread isn't there to *enrich* the dealer, then whatever spread survives competition must be there for some other reason. We are about to find that reason. Second, he is *risk-neutral*: he cares only about expected value, not about the variance of his outcomes. This keeps the arithmetic about information and nothing else.

**The traders.** A stream of traders arrives, one at a time, each placing a single order — a buy or a sell — against the MM's quotes. Here is the heart of the model: the traders are not all alike. A fraction of them, which we'll call `frac_informed` (write it as the symbol $\alpha$, "alpha"), are **informed**. They have already seen the truth. An informed trader who knows the value is \$110 will *buy* at any ask below \$110, because he is getting a bargain; an informed trader who knows it is \$90 will *sell* at any bid above \$90. The rest — a fraction $1 - \alpha$ — are **uninformed**, also called *noise traders* or *liquidity traders*. They trade for reasons that have nothing to do with the value: they need cash, they're rebalancing a pension, they're a tourist. We model them as buying or selling by a coin flip, 50/50, regardless of the truth.

The MM cannot tell which is which. When a buy order lands on his ask, he sees an anonymous "buy." It might be an informed trader who knows the asset is worth \$110 and is happily lifting his \$103 offer. It might be a noise trader who'd have bought no matter what. The MM's entire problem is that **he must quote one price for both**, and that price has to survive a population he cannot sort.

**The rule that ties it together: adverse selection.** Because informed buyers only ever buy (when the truth is high) and informed sellers only ever sell (when the truth is low), the informed traders are *not* split 50/50 across the MM's two prices. They pile onto whichever side is profitable for them — which is, by construction, the side that's *bad* for the MM. So a buy order is contaminated: it is more likely than a coin flip to be coming from someone who knows the value is high. This tilt — the systematic way that the people most eager to take a given side of your trade are disproportionately the ones who know it's the right side — is **adverse selection**, the same force that drives the lemons problem in used-car markets and the winner's curse in auctions. (We unpack the general idea in [the lemons post](/blog/trading/game-theory/asymmetric-information-the-lemons-problem-in-markets) and [the fast-fill / winner's-curse post](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news); here we turn it into a number.)

**The solution: quote the conditional expectation.** Since competition forces zero profit, the MM cannot charge more than the asset is worth *given what the order tells him*. So his rule is exactly this:

$$\text{ask} = \mathbb{E}[\,V \mid \text{a buy arrives}\,], \qquad \text{bid} = \mathbb{E}[\,V \mid \text{a sell arrives}\,].$$

Here $V$ is the asset's true value, $\mathbb{E}[\cdot]$ is the expected value (the probability-weighted average), and the bar "$\mid$" means "conditional on" — "given that." Read it plainly: the ask is the average value of the asset *across all the worlds in which the next order is a buy*, and the bid is the average value *across all the worlds in which the next order is a sell*. Because buys carry good news and sells carry bad news, the ask sits above the prior mean and the bid sits below it. The spread between them is forced into existence by the asymmetry. It is not chosen; it is computed.

Everything from here is filling in those two expectations with actual probabilities. We will do it twice by hand, then let the model's `glosten_milgrom` function do the bookkeeping so we can sweep the inputs and watch the spread breathe.

### Why the spread is not a fee — a quick gut-check

It is worth pausing on what we have *not* assumed. We did not give the MM any costs. No rent, no salaries, no exchange fees, no cost of capital. In this model the dealer's overhead is literally zero. And yet the spread will come out strictly positive whenever there are any informed traders at all. That single fact demolishes the toll story. A spread can exist in a world with no costs whatsoever, purely because the dealer is trading against people who know more than he does and he has to break even against them. Real spreads do also contain a little order-processing cost and a little inventory risk — we'll add those back in the real-markets section — but the *irreducible core* of the spread, the part that won't go away no matter how cheap the technology gets, is adverse selection.

## The Bayesian arithmetic, step by step

Let us build the ask price from scratch, with the base-case numbers, slowly enough that nothing is a leap. We have: high value \$110, low value \$90, prior $P(\text{high}) = 0.5$, and a fraction $\alpha = 0.30$ informed. So 30% of traders know the truth and act on it; the other 70% are noise, flipping a coin.

**Step 1 — How likely is a buy, given the truth is high?** Split the population. The 30% informed traders, *in the high world*, all buy — they know it's cheap. The 70% noise traders buy with probability one-half. So:

$$P(\text{buy} \mid \text{high}) = \underbrace{0.30 \times 1}_{\text{informed all buy}} + \underbrace{0.70 \times 0.5}_{\text{noise coin-flip}} = 0.30 + 0.35 = 0.65.$$

**Step 2 — How likely is a buy, given the truth is low?** In the low world, the informed *sell*, so none of them buy. Only the noise traders buy, and only half of them do:

$$P(\text{buy} \mid \text{low}) = \underbrace{0.30 \times 0}_{\text{informed never buy when low}} + \underbrace{0.70 \times 0.5}_{\text{noise}} = 0 + 0.35 = 0.35.$$

Already you can see the asymmetry: a buy is almost twice as likely to come from the high world (0.65) as from the low world (0.35). That ratio is the whole engine.

**Step 3 — How likely is a buy overall?** Weight each world by its prior:

$$P(\text{buy}) = P(\text{high}) \cdot P(\text{buy} \mid \text{high}) + P(\text{low}) \cdot P(\text{buy} \mid \text{low}) = 0.5 \times 0.65 + 0.5 \times 0.35 = 0.5.$$

A buy and a sell are each 50% likely, which makes sense by symmetry — the prior is a coin flip and the setup is mirror-image.

**Step 4 — Flip it around with Bayes' rule.** We don't actually care about $P(\text{buy} \mid \text{high})$; we care about the reverse, $P(\text{high} \mid \text{buy})$ — given that a buy *just happened*, how likely is the high world now? Bayes' rule is the machine that flips a conditional probability:

$$P(\text{high} \mid \text{buy}) = \frac{P(\text{high}) \cdot P(\text{buy} \mid \text{high})}{P(\text{buy})} = \frac{0.5 \times 0.65}{0.5} = 0.65.$$

The prior was 0.50; after seeing one buy it jumps to 0.65. The order *moved the belief*. That is Bayesian updating, and it is the same operation a doctor does when a positive test raises the probability of a disease.

**Step 5 — Turn the updated belief into a price.** The ask is the expected value given a buy:

$$\text{ask} = \mathbb{E}[V \mid \text{buy}] = P(\text{high} \mid \text{buy}) \times \$110 + P(\text{low} \mid \text{buy}) \times \$90 = 0.65 \times \$110 + 0.35 \times \$90 = \$71.50 + \$31.50 = \$103.$$

There it is. The ask is \$103, three dollars above the \$100 prior mean. Repeat the mirror calculation for a sell and you get $P(\text{low} \mid \text{sell}) = 0.65$ and a bid of \$97. The **spread is \$103 − \$97 = \$6.** No costs were ever assumed. The \$6 is pure adverse-selection insurance.

![Bayesian tree showing how an informed-or-noise split turns a buy into a 0.65 posterior that the value is high](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-2.png)

The tree above lays out that branching explicitly: the first split is high-versus-low (the prior), the second split is informed-versus-noise, and the leaves are buy/sell orders. Reading the tree backward from a "buy" leaf to its roots is exactly Bayes' rule — you collect all the paths that end in a buy and ask what fraction of them started in the high world. The answer, 0.65, is the posterior that prices the ask.

There is a second way to see the same structure that some readers find cleaner: instead of starting from the value and branching to the order, start from the order and ask who could have sent it. Every buy that lands on the dealer's ask came from one of two sources — an informed trader who knows the value is high, or a noise trader who happened to flip "buy." The informed buyer *only* exists in the high world; the noise buyer exists in both. So the pool of buyers is contaminated: the informed are all on the high side, the noise is split evenly, and the net tilt of the pool is upward. The mirror holds for sells. The spread is just the dealer pricing the contamination of each pool.

![Order-probability tree splitting traders into informed and uninformed and routing each to a buy or sell](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-7.png)

The order-probability tree above traces the population the other direction: from the top, traders split into the informed fraction (0.30) and the uninformed fraction (0.70); the informed are routed entirely by the truth (buy if high, sell if low), while the uninformed split 50/50 no matter what. Sum the paths that end in "buy" and you recover the 0.65/0.35 contamination that prices the ask. The two trees are the same machine read from opposite ends; whichever you find more natural, the punchline is identical — a buy is contaminated upward, a sell downward, and the dealer prices the contamination.

#### Worked example: the market maker breaks even, order by order

Let's prove the zero-profit claim with money, because it is the part people don't believe. The MM quotes ask \$103, bid \$97. Consider the *buy side* only. A buy arrives; the MM sells one share at \$103 and must later mark it to whatever the value turns out to be.

With probability $P(\text{high} \mid \text{buy}) = 0.65$, the value is \$110, and the MM — who just sold at \$103 — *loses* \$110 − \$103 = \$7. With probability $P(\text{low} \mid \text{buy}) = 0.35$, the value is \$90, and the MM sold at \$103, so he *gains* \$103 − \$90 = \$13. His expected profit on the trade:

$$0.65 \times (-\$7) + 0.35 \times (+\$13) = -\$4.55 + \$4.55 = \$0.$$

Exactly zero. He bleeds \$4.55 in expectation to the informed (the 0.65-weighted loss) and recovers exactly \$4.55 from the noise traders (the 0.35-weighted gain). The ask is set at the precise point where those two flows cancel. Set the ask any lower and the informed bleed him dry; set it any higher and a competitor undercuts him. The intuition: the spread is the dealer's way of taxing the harmless majority to refund the damage done by the dangerous minority.

#### Worked example: re-deriving the same spread with the model

Hand arithmetic is convincing once; for the sweeps we let the series' `glosten_milgrom` model carry the bookkeeping. It takes the high value, the low value, the prior on high, and the informed fraction, and returns the ask, bid, spread, and the order probabilities. Feeding it our base case:

```
import data_gametheory as gt

r = gt.glosten_milgrom(v_high=110, v_low=90, p_high=0.5, frac_informed=0.30)
print(r["ask"], r["bid"], r["spread"], r["mid"])
//  ->  103.0   97.0   6.0   100.0
print(r["p_buy"], r["p_sell"])
//  ->  0.5     0.5
```

The model reproduces our hand numbers to the cent: ask \$103, bid \$97, spread \$6, mid \$100, each order 50% likely. Now change one input. Crank the informed fraction to 50%:

```
r = gt.glosten_milgrom(110, 90, 0.5, 0.50)
print(r["ask"], r["bid"], r["spread"])
//  ->  105.0   95.0   10.0
```

The spread jumps from \$6 to \$10. Half the room knowing the truth instead of just under a third forces the dealer to widen by four dollars. The lesson the model makes vivid: the spread is a dial wired directly to how dangerous the order flow is.

### What happens when the prior is not a coin flip

Our base case used a symmetric 50/50 prior, which keeps the mid pinned to \$100 and the spread symmetric around it. Real markets rarely start from a coin flip — the dealer usually leans one way already. Let's see what the model does when the prior is skewed. Suppose the dealer thinks the high value is *more* likely: $P(\text{high}) = 0.70$, still with \$110/\$90 values and 30% informed.

```
skew = gt.glosten_milgrom(110, 90, 0.70, 0.30)
print(round(skew["ask"],2), round(skew["bid"],2), round(skew["spread"],2))
//  ->  106.25   101.14   5.11
print(round(skew["prior_mean"],2), round(skew["mid"],2))
//  ->  104.0    103.69
```

Three things change at once, and each one teaches something. First, the **prior mean jumps to \$104** (`0.7 × 110 + 0.3 × 90`), so the whole market re-centers higher — the dealer's no-trade estimate of value moved up. Second, the **spread narrows to \$5.11** from \$6. That is subtle: when the dealer is already fairly confident the value is high, a buy is *less* surprising — it mostly confirms what he believed — so it moves his belief less, and the ask and bid sit closer together. The most informative orders arrive when the dealer is maximally uncertain (the 50/50 prior); confidence shrinks the spread. Third, the **bid and ask are no longer symmetric around the prior mean**: the ask is \$2.25 above \$104 but the bid is only \$2.86 below it, because a *sell* against a high-leaning prior is the surprising, information-rich event, so it moves the belief more than a buy does. The spread tilts toward the surprising side. This is why, in a strong trend, the dealer's quotes can look lopsided — he's pricing the fact that a trade *against* the trend carries more information than a trade *with* it.

#### Worked example: the lopsided spread under a skewed belief

Let's read those numbers as a story. The dealer is 70% sure a stock is worth \$110, so his fair value is \$104 and he quotes \$101.14 bid / \$106.25 ask. A buy arrives. It nudges his belief from 0.70 up toward 0.78 — a small move, because he half-expected a buy — and he barely re-centers. But suppose a *sell* had arrived instead: that's the dog that wasn't supposed to bark. It pulls his belief from 0.70 down toward 0.56, a much larger move, because a sell against a high-leaning book is exactly the kind of order an informed trader who knows the value is actually \$90 would place. So the dealer protects the downside more: the bid sits \$2.86 below fair value while the ask sits only \$2.25 above it. The intuition: the dealer charges the most for the trade that would surprise him most, because surprise is where the information — and the danger — lives.

### Why the mid still equals the prior — and why that matters

Notice the **mid-price** — the midpoint of bid and ask — stayed at \$100 in the base case, exactly the prior mean. That is not a coincidence; it is a deep property. *Before* an order arrives, the MM's best estimate of the value is the prior mean, and the bid and ask straddle it symmetrically (when the prior is symmetric). The spread is the dealer's hedge against the order's *content*; the mid is his estimate *absent* any order. So the quoted mid is an unbiased read on value, and the spread is the uncertainty band around it. A trader who wants the market's honest opinion of the price reads the mid; a trader who wants to actually transact pays the half-spread to cross it. We'll see in a moment that once an order *does* arrive, the mid moves — that movement is price discovery.

## How the spread responds to the two things that drive it

A model earns its keep when you can turn its knobs and the answers make sense. The Glosten-Milgrom spread has two knobs: **how many traders are informed** ($\alpha$), and **how uncertain the value is** (the gap between high and low). Both push the spread the same direction — wider — and the model lets us draw the exact curves.

### Knob one: the fraction of informed traders

Hold the values at \$110/\$90 and the prior at 50/50, and sweep $\alpha$ from 0 to 1. At $\alpha = 0$ — nobody is informed, all flow is noise — the spread is exactly zero. There is no adverse selection to defend against, so the MM quotes the prior mean to everyone and the bid equals the ask. At the other extreme, $\alpha = 1$ — *everybody* who trades knows the truth — the spread blows out to the full \$20, the entire distance between high and low. Why the full \$20? Because if every buyer is informed, then a buy *certainly* means the value is \$110, so the only break-even ask is \$110 itself; every seller is informed, so a sell certainly means \$90, and the only break-even bid is \$90. The market maker is fully exposed and quotes the two outcomes as the two prices. In between, the spread climbs roughly in proportion to $\alpha$.

![Spread rising from zero to the full value range as the informed fraction climbs from zero to one](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-3.png)

The curve above is computed straight from the model at each informed fraction. Read it as a danger gauge: the x-axis is "what fraction of the people I'm trading against can see my cards," and the y-axis is "how wide I have to quote to survive." The dot at 30% marks our base case, \$6. The relationship is the reason a dealer's first question about any name is never "what does it cost me to make this market?" but "who's trading it, and what do they know?"

#### Worked example: a quiet stock versus an event stock

Picture two stocks, identical in every way except who trades them. Stock A is a boring utility: almost all the flow is index funds and retail rebalancing — noise. Say $\alpha = 0.10$, ten percent informed. Stock B is a biotech the day before a phase-3 trial readout: the flow is thick with people who've talked to doctors, modeled the endpoint, or are just better at this — say $\alpha = 0.50$. Same \$110/\$90 value range, same coin-flip prior. The model:

```
quiet = gt.glosten_milgrom(110, 90, 0.5, 0.10)
event = gt.glosten_milgrom(110, 90, 0.5, 0.50)
print(quiet["spread"], event["spread"])
//  ->  2.0   10.0
```

The utility shows a \$2 spread; the biotech shows \$10 — five times wider — with zero difference in the dealer's costs. If you trade the biotech as a noise trader, you pay \$5 to cross (half of \$10) versus \$1 on the utility. The intuition: you are charged for the company you keep, and on the event stock your company is dangerous.

### Knob two: how much the value can move

Now hold $\alpha$ fixed at 30% and widen the *range* of possible values, keeping the prior centered at \$100. With a tight \$105/\$95 range (the asset is worth one of two close numbers) the spread is \$3. Widen to \$120/\$80 and it doubles to \$12. Widen to \$140/\$60 and it doubles again to \$24. The spread scales linearly with how far apart the two possible truths are — which is the model's version of *volatility*. More uncertainty about the value means each order moves the belief across a bigger gap, so the ask and bid land further apart.

![Spread doubling as the gap between the high and low possible values doubles, at a fixed informed fraction](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-4.png)

The bars above are computed at four value ranges with the informed fraction pinned at 30%. This is the model's explanation for why spreads gap wider before scheduled news. Before a Fed decision, a jobs report, or an earnings release, the *range* of plausible post-news values fans out — the high-low gap widens — and even with the same mix of informed traders, the dealer must widen to match. The instant the number prints and the uncertainty collapses, the range snaps shut and so does the spread. You can watch it happen on any quote screen at 8:30 a.m. on a payroll Friday.

#### Worked example: the spread as a volatility tax

Suppose you are a noise trader who needs to get in and out of a \$100 stock once, round-trip, and you're choosing between a calm week and an event week. In the calm week the value range is \$105/\$95 ($\alpha = 0.30$), so the spread is \$3 and your round-trip cost of crossing it twice is \$3 per share — half the spread to buy, half to sell, \$1.50 each way... actually the full spread round-trip, \$3. In the event week the range is \$120/\$80, spread \$12, round-trip \$12. On a 1,000-share position that's \$3,000 versus \$12,000 in pure spread cost — a \$9,000 difference driven by nothing but the width of the value distribution. The intuition: half the spread you pay is a volatility tax, and the time to pay it is when the asset is sleepy, not when the whole world is waiting for a number.

### Adding back the real-world pieces: the three-part spread

The pure model gives a spread of zero cost and pure adverse selection. Real spreads are a little wider than that, and microstructure researchers decompose the observed spread into three pieces, only one of which is the Glosten-Milgrom term. It is worth naming all three so you know what fraction of a real spread the model actually explains.

The first piece is **adverse selection** — the Glosten-Milgrom core, the part that pays for trading against the informed. On liquid stocks this is typically the *largest* component of the spread, often more than half. The second piece is **order-processing cost** — the exchange fees, the clearing and settlement cost, the tiny per-trade overhead. This is the old "toll" story, and it is real but small; on a liquid name it might be a fraction of a cent. The third piece is **inventory-holding cost** — the risk the dealer takes by ending up long or short a position he didn't want. A dealer who buys from a seller now holds inventory he must offload, and he bears the price risk in between; he widens his spread a touch to be paid for that risk, and he *skews* his quotes (lowering both bid and ask when he's too long) to encourage flow that flattens his book.

The reason the model focuses on the first piece is that it's the one that won't go away. Order-processing cost shrinks as technology gets cheaper; inventory cost shrinks as dealers get faster at hedging and laying off risk. But adverse selection is irreducible — it's a property of *who* is trading, not of how the trading is plumbed — so it sets the floor that no amount of efficiency can push through. When you read that a stock's spread is "one cent," you are looking at a name where the adverse-selection component has been competed down to almost nothing because the flow is overwhelmingly noise; when you see a fifty-cent spread, the adverse-selection term is doing the work.

#### Worked example: how much of a real spread is information?

Take a mid-cap stock with a quoted spread of 8 cents. Suppose careful estimation (the standard approach decomposes price changes following buys versus sells) attributes 5 cents to adverse selection, 1 cent to order processing, and 2 cents to inventory. The Glosten-Milgrom term is 5/8 of the spread — the majority. Now imagine the same stock the day before an FDA decision. The value range fans out, the informed fraction climbs, and the adverse-selection term alone might triple to 15 cents while the processing and inventory pieces barely move, giving a quoted spread near 18 cents. The 10-cent widening is *almost entirely* the information term. The intuition: when a spread moves, it's the adverse-selection component that's moving, because the other two are roughly constant — so a widening spread is the market telling you the flow just got more informed.

## Price discovery: how each trade walks the price toward the truth

So far we've looked at a single order in isolation. The real magic of the model appears when orders arrive *one after another*, because the MM is a Bayesian who never forgets. After he updates his belief from a buy, that updated belief becomes his new prior for the *next* order. Over a sequence of trades, the informed traders — who keep pushing on the same side, because they all know the same truth — leak their information into the price one order at a time. The mid-price walks toward the true value. This walk is **price discovery**: the market figuring out what something is worth, not by anyone announcing it, but by the slow Bayesian digestion of order flow.

Watch it happen. Start at the prior mean, \$100, with $\alpha = 0.30$. Suppose the truth is \$110 (high), so the informed traders are net buyers. The first buy updates $P(\text{high})$ from 0.50 to 0.65 and the mid moves to \$103 — which, note, is exactly the first ask. (That is no accident: the post-buy mid *is* the conditional expectation given the buy, which is the ask.) The second buy updates 0.65 to 0.78 and the mid to \$105.5. A third pushes it to \$107.3. Each step is a smaller jump than the last, because as the belief approaches certainty there's less surprise left in each order — but the direction is relentless as long as the informed keep buying.

![Mid-price stepping upward toward the true value of 110 as a sequence of buys and sells arrives](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-5.png)

The path above is computed by iterating the Bayesian update over a realistic order sequence — mostly buys, because the truth is high and informed flow is net long, with a couple of noise sells that briefly knock the mid back down. The mid climbs from \$100 toward \$110 and never quite arrives, because there is always residual doubt that the run of buys was a noise fluke. The two dips are noise traders selling; the dealer dutifully marks the price down, then the informed flow resumes and pulls it back up. The whole staircase is the market learning the value with no one ever stating it.

There is a beautiful and counterintuitive corollary hiding here: **the noise traders are not a nuisance, they are essential.** Without them, the market would not work at all. Recall the $\alpha = 1$ corner: if *every* trader were informed, the dealer would face a spread equal to the full value range, and at those prices nobody could ever profit from trading, so no informed trader would bother and no information would enter the price. It is precisely because the noise traders are there — flipping coins, providing cover — that the informed can trade at all, and it is precisely because the informed are there that the noise traders' orders get a fair mid-price to transact against. The two populations need each other. The noise traders pay the spread; in exchange they get a price that has been honestly discovered by the informed flow. This is the deep reason markets full of "dumb money" can still produce accurate prices: the dumb money is the lubricant that lets the smart money's information leak in. A market that somehow purged all its noise traders would seize up, spreads to infinity, no trades at all.

This is also where the informed trader's incentive to *hide* comes from. The more an informed trader can disguise his orders as noise — splitting them up, randomizing timing, trading across venues — the slower the dealer learns, the less the price moves against him, and the more of his decaying edge he keeps. The Glosten-Milgrom dealer assumes orders arrive one anonymous unit at a time and can't be strategically disguised; relaxing that single assumption, and letting the informed trader optimize how he feeds his order into the noise, is exactly the leap from this model to Kyle's.

#### Worked example: one buy, and the mid becomes the old ask

Let's nail the mechanism with the base numbers. Prior $P(\text{high}) = 0.50$, mid \$100, ask \$103. A buy arrives. The posterior, from Step 4 above, is $P(\text{high} \mid \text{buy}) = 0.65$. The new mid-price — the MM's fresh estimate of value after digesting the buy — is:

$$\text{new mid} = 0.65 \times \$110 + 0.35 \times \$90 = \$103.$$

The new mid is \$103, *exactly the ask the dealer was quoting before the trade*. So when you lift a market maker's offer, you don't just pay the ask — you *teach* him the value is probably higher, and he immediately re-centers his entire market around your old purchase price. His next quote might be a \$103 mid with a \$100 bid and \$106 ask. The intuition: crossing the spread is not a private transaction; it is a public signal that permanently moves where the market thinks the price is.

#### Worked example: the informed trader's leaking profit

The informed trader is the one with the edge, but the model shows his edge *decays* as he trades, because every order he places tips his hand. Suppose he knows the value is \$110 and starts buying at the \$103 ask. His first share earns him \$110 − \$103 = \$7 of value-versus-price. But his buy moves the mid to \$103 and the next ask to roughly \$106, so his *second* share earns only \$110 − \$106 = \$4. The third ask climbs again, his edge thins to maybe \$2, and so on. If he tried to buy his entire position at once, he'd walk the ask all the way up to \$110 and capture nothing on the marginal share. His total profit is finite and shrinking — the market maker recoups the early losses through the later, higher asks. The intuition: information is worth the most on the first trade and decays with every print, which is exactly why informed traders work so hard to *hide* — the subject of [Kyle's model](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise), the next post, where the informed trader strategically slices his order into the noise so the price moves slower and he keeps more of his edge.

### The market maker's full inference loop

Step back and see the whole machine as a loop the dealer runs on every single order. He starts with a prior. He posts a bid and an ask straddling it. An order arrives. He updates his prior with Bayes' rule, conditioning on the side of the order. The updated belief is his new mid, and he posts fresh quotes around it. Then the next order arrives and the loop turns again. The spread is set by the *danger* of the flow (the informed fraction and the value uncertainty); the mid is moved by the *content* of the flow (which side keeps getting hit).

![The market maker inference loop from prior to quotes to order to Bayesian update and back](/imgs/blogs/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom-6.png)

The loop above is the algorithm a modern market-making engine runs millions of times a second, with continuous values instead of two, and a richer model of who's informed — but the skeleton is exactly Glosten-Milgrom. Prior in, quotes out, observe the fill, update, repeat. The spread protects; the mid learns. The reason this loop is so robust is that it never needs to *identify* the informed trader. It treats every order as a weighted average of informed and noise, and lets the math do the sorting.

## Common misconceptions

**"The spread is the dealer's profit."** In the competitive model it is exactly zero profit. The spread is not margin; it is the break-even charge that lets the dealer survive trading against people who know more. Real dealers do earn a little — they pad the spread slightly for inventory risk and processing cost, and they earn the spread on the noise traders net of what the informed take — but the *idea* that a wider spread means a fatter dealer profit is wrong. A wider spread usually means a more *dangerous* stock, not a greedier dealer. If anything, the dealer would love the spread to be zero, because that means no one can pick him off.

**"If I'm a small retail trader, the spread isn't about me."** It is *entirely* about people like you — and that's good news. You are the noise trader, the one the model assumes flips a coin. The spread is the premium the informed minority forces the dealer to charge *everyone*, and you pay it even though you have no edge. The flip side is that you are not the one being defended against; you are the one subsidizing the defense. The practical takeaway is to trade where the flow is least toxic (liquid, low-information names) so the premium you're forced to pay is as small as possible.

**"A trade just moves price because of supply and demand pushing on a curve."** In the mechanical "more buyers than sellers" story, price moves because order flow physically consumes resting liquidity. Glosten-Milgrom gives a deeper reason that holds even with a single dealer and infinite depth: price moves because the trade *reveals information*. A buy doesn't move the price by using up shares; it moves the price by *updating the dealer's belief* about value. This is why a single 100-share buy in a thin, information-rich stock can move the quote more than a 100,000-share buy in a deep, information-poor one. Impact tracks information, not just size. ([Kyle's model](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise) makes the size-versus-information tradeoff precise with a linear price-impact coefficient.)

**"Wider spreads mean the market is broken or being gouged."** Sometimes a wide spread is a perfectly rational dealer protecting himself against flow he's afraid of. Spreads gapping wide before a Fed announcement or in the first seconds after a shock are not a malfunction; they are the model working — the value range fanned out and the dealer widened to match. A market that *didn't* widen its spreads into uncertainty would be one whose dealers were about to get run over, and that market would soon have no dealers at all. The wide spread is the price of keeping someone willing to quote.

**"Adverse selection only matters in stocks."** The exact same arithmetic prices the spread in foreign exchange, in bonds, in crypto, and in the bid-ask of a sports betting exchange. Anywhere a dealer quotes two-sided prices to a population that includes some informed traders, the spread is an adverse-selection premium. The numbers differ, but the machine is identical: ask = E[value | buy], bid = E[value | sell]. The used-car dealer who quotes you a low trade-in and a high lot price is running the same model on cars.

## How it shows up in real markets

**The decimalization and the shrinking spread (2001).** When U.S. stock markets switched from quoting in sixteenths of a dollar (6.25 cents) to pennies in April 2001, average spreads collapsed — for many liquid large-caps, the quoted spread fell from roughly 6 cents toward 1–2 cents. The Glosten-Milgrom reading: the minimum tick had been forcing spreads *wider* than adverse selection alone required, so dealers were earning a bit of rent on top of the break-even spread. Decimalization let competition push spreads down toward the true adverse-selection floor, and on the most heavily traded, least-information-asymmetric names, that floor is a penny or two. On harder-to-value small-caps, spreads stayed much wider — exactly where the informed-fraction and uncertainty knobs are turned up.

**Spreads gapping into scheduled news.** On any U.S. payroll Friday, watch the quotes on rate-sensitive instruments at 8:30 a.m. Eastern, when the jobs number prints. In the seconds before the release, spreads on Treasury futures and rate-sensitive ETFs visibly widen and depth thins; within a second or two after the print, they snap back. This is the value-uncertainty knob in real time: the *range* of plausible post-news prices fans out before the number, so dealers widen to avoid being picked off by anyone with a faster read; once the number is public, the uncertainty collapses and so does the spread. Academic event studies of FOMC and CPI releases find the same pattern — a measurable pre-announcement widening that is hard to explain with any cost story but falls right out of adverse selection.

**The "flash crash," May 6, 2010.** In the span of minutes that afternoon, the Dow fell about 1,000 points and several blue-chip stocks briefly traded at absurd prices (Accenture printed at a penny; Sotheby's at \$100,000). The microstructure post-mortems converged on a Glosten-Milgrom-flavored story: as a large automated sell program hit the market, dealers and market-making algorithms could not tell informed flow from a fat-finger panic, so — facing what looked like dangerously toxic, one-sided order flow — they widened spreads dramatically or pulled their quotes entirely. When the informed-fraction knob spikes toward 1 (everything looks informed), the model says the spread blows out toward the full value range; in the real, continuous market that meant quotes vanishing and "stub" prices a penny wide on the bid. The crash is what the $\alpha \to 1$ corner of the model looks like when it happens all at once.

**FX dealing and "last look."** In foreign-exchange markets, electronic dealers quote tight two-way prices but reserve a brief "last look" — a fraction of a second to reject a trade after you've hit it. The controversy around last look is, at bottom, an adverse-selection fight: dealers argue they need it to avoid being systematically picked off by faster, better-informed flow (exactly the informed traders of the model); critics argue it lets dealers cherry-pick. Both sides are arguing about the same thing Glosten and Milgrom formalized — how a quoter defends a two-sided price against a population he can't fully sort. The tight headline spread plus a last-look option is just another way of pricing the adverse-selection risk that a flat, firm spread would have to bake in directly.

**Crypto and the toxic-flow premium.** On crypto exchanges, market makers quote wider spreads on tokens with thin information and concentrated holders — where a single wallet may know about an unlock, a listing, or a hack before the dealer does — and tight spreads on the most liquid pairs like BTC/USD where the flow is dominated by noise. The same model prices both: BTC/USD has a low informed fraction and (relative to its price) a modest near-term uncertainty, so the spread is a few basis points; a freshly listed micro-cap has a high informed fraction and enormous uncertainty, so its spread is enormous. Liquidity providers in automated market makers face the identical problem under the name *loss-versus-rebalancing* — the systematic loss to informed arbitrageurs is adverse selection by another name, and the fee tier is the spread.

**Earnings and the post-announcement re-quote.** When a company reports earnings after the close, the next morning's opening spread is wide and then narrows through the day. The model's price-discovery loop is running: overnight, the value range is huge (nobody has fully digested the report), so the open is wide; as informed traders trade and the mid walks toward the new fair value, uncertainty resolves and the spread tightens. By mid-morning the price has discovered its new level — the staircase from the convergence figure, played out over a few hours instead of a few trades.

## The playbook: how to trade against the spread

The Glosten-Milgrom model is not just an explanation; it is a checklist for whoever is about to cross a spread or quote one. Here is how to use it from each side of the trade.

**Know which player you are.** The single most important question: *are you the informed trader or the noise trader on this name?* If you have genuine edge — a model the market hasn't priced, a faster data feed, a structural insight — you are the informed minority, and the spread is the toll you pay to monetize information that decays the moment you trade. If you have no edge and you're trading for liquidity, rebalancing, or a hunch, you are the noise trader, and the spread is a pure cost with no offsetting information rent. Most people overestimate which one they are. The honest default is: assume you're the noise trader unless you can name your specific, durable edge.

**If you're the noise trader, minimize the premium you pay.** The spread is a tax wired to the informed fraction and the value uncertainty, so pay it where both are lowest. That means: trade the most liquid names, where noise dominates and the adverse-selection premium is thinnest; avoid trading into scheduled news, when the uncertainty knob spikes and the spread fans out; and prefer to *provide* liquidity (post a limit order at or inside the spread) rather than *take* it (cross the spread with a market order) when you're not in a hurry — though beware that a resting order can itself be picked off by the informed, which is the [fast-fill problem](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news). Your edge, if any, has to clear the round-trip spread before it's real; a strategy with a \$5 expected gain per share on a stock with a \$6 spread is a money-loser disguised as a winner.

**If you're quoting (the market maker's seat), price the danger, not the cost.** The model says your spread should track adverse selection: widen on names with concentrated, informed flow; widen into events; widen the instant the flow turns one-sided and starts walking your mid (that's the informed leaking in). Your inventory and processing costs are real but second-order — the first-order driver is *who is hitting you and what they know.* And update relentlessly: every fill is data. If your offer keeps getting lifted, the market is telling you the value is higher than your mid; move it up before you get run over, exactly as the inference loop prescribes.

**The invalidation.** The clean model assumes one-shot, single-unit trades and a competitive, risk-neutral dealer. It breaks when: the dealer has market power (a monopoly specialist can charge a spread above the adverse-selection floor — the rent decimalization squeezed out); when inventory risk dominates information risk (a dealer who's already very long will skew his quotes to lay off risk, not just to defend against information); when traders are *strategic about size and timing* rather than placing one anonymous unit (that's Kyle's model, where the informed trader hides in the noise and the simple buy-equals-good-news logic gets diluted on purpose); and when the "informed" aren't actually informed but are reflexively *creating* the value they trade on (a coordination or bubble dynamic, covered elsewhere in this series). Use Glosten-Milgrom as the floor and the intuition, then layer the wrinkles.

**The sizing-and-exit discipline.** As a taker, size so that the spread you pay is a small fraction of your expected edge, and never let the spread itself be your strategy's largest cost line — if it is, you're trading too often or in names that are too toxic. As a maker, size your quotes so that being adversely selected on any single fill can't ruin you, because you *will* be picked off regularly; the business model is to lose small to the informed and win small off the noise, many times. Either way, treat the spread as the model teaches it: not a fee, but the running price of not knowing what the person on the other side knows.

## Further reading & cross-links

- [Asymmetric information: the lemons problem in markets](/blog/trading/game-theory/asymmetric-information-the-lemons-problem-in-markets) — the general theory of what hidden information does to a market. The Glosten-Milgrom spread is the lemons discount made exact and denominated in cents.
- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — the same adverse-selection force from the *taker's* side: why an instant fill or a won auction is a warning, and why a resting order can be picked off by the informed.
- [Kyle's model: how an informed trader hides in the noise](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise) — the next post. Where Glosten-Milgrom has the informed trader place one anonymous order, Kyle lets him strategically slice it into the noise so the price moves slower and he keeps more of his decaying edge.
- [How an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the real-world dealer's view, where the same adverse-selection spread shows up around the Greeks and the dealer hedges what he can't avoid quoting.
- [Who is on the other side of your trade?](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade) — the series' founding question. Glosten-Milgrom is the first fully formal answer: the other side is a mix of the informed and the noise, and the spread is the price of not knowing which one just hit you.

*This is educational material about market mechanics, not financial advice.*

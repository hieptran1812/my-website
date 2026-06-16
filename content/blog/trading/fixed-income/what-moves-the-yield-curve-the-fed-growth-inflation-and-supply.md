---
title: "What moves the yield curve: the Fed, growth, inflation, and supply"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-practitioner guide to the four forces that move the yield curve — why the front end follows the Fed, why the long end answers to growth, inflation, the term premium, and supply, and why the same news hits each maturity differently."
tags: ["fixed-income", "bonds", "yield-curve", "term-premium", "interest-rates", "inflation", "treasury-supply", "monetary-policy"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **The big idea:** the yield curve is not moved by one force but by several, and each one owns a different part of the curve — so the short end and the long end can move in opposite directions on the very same day.
> - The **front end** (3-month to 2-year) is anchored almost entirely by the **Fed**: a short maturity is basically the average policy rate you expect to be paid over its life, so it tracks the path of the fed funds rate.
> - The **long end** (10-year, 30-year) is driven by four slower forces: **growth expectations**, **inflation expectations**, the **term premium** (the extra yield demanded for tying money up), and **supply and demand** (Treasury issuance, the Fed's QE/QT, and foreign and safe-asset demand).
> - The **belly** (2-to-7-year) is a blend of the two — part Fed path, part growth story.
> - The same headline hits maturities differently: a hot inflation print can spike the 2-year on rate-hike fears while the 30-year's reaction depends on whether the market still trusts the Fed to bring inflation back down.
> - A move can be a **shift** (the whole curve up or down — a level change) or a **twist** (the slope changes — a steepening or flattening), and *which* it is tells you *what* moved it.
> - Our running example traces one day: a +0.5% inflation surprise and a \$300 billion increase in Treasury issuance, and how they push the 2-year and the 10-year by different amounts.

Here is a puzzle that trips up almost everyone the first time they watch bond markets closely. The government releases an inflation report that comes in hot — prices rising faster than anyone expected. You would think every interest rate in the country jumps in lockstep. Instead, you watch the screen and see the 2-year Treasury yield leap 25 basis points while the 30-year barely flinches, or sometimes even *falls*. Same news, same instant, same government's debt — and the front of the curve and the back of the curve go their separate ways. What on earth is going on?

The answer is the subject of this entire post, and it is the single most useful thing you can understand about the bond market: **the yield curve is not a single object moved by a single force. It is a row of maturities, and different forces tug on different parts of it.** Once you internalize which force owns which maturity, the daily chaos resolves into a small number of clean, recurring stories. A flattening curve is the Fed hiking into a long end that doesn't believe the hikes will last. A bear steepening is a growth scare in reverse — the economy running hotter than thought, or the Treasury flooding the market with bonds. None of it is random. It is four forces, each with its own home on the curve.

![An annotated yield curve with the front end labeled as driven by the Fed and policy-rate expectations, the belly as a blend, and the long end driven by growth expectations, inflation expectations, the term premium, and supply and demand](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-1.png)

The diagram above is the mental model for the whole post. Picture the yield curve — yield on the vertical axis, maturity (3 months out to 30 years) on the horizontal. The leftmost slice, the front end, has one master: the Federal Reserve. As you move right into the belly and then the long end, the Fed's grip loosens and three other forces take over — the market's view of future growth, its view of future inflation, the extra yield it demands for locking money up that long, and the raw push-and-pull of how many bonds are being sold versus how many people want to buy them. Every figure in this post is really just a close-up of one of those forces in action.

This is post #20 in *The Bond Market, From the Ground Up*, and it is the capstone of the yield-curve track. We have already learned [what the yield curve is and how to read it](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance), [why it usually slopes up](/blog/trading/fixed-income/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations), [what its inversions warn about](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works), and [how traders bet on its shape](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies). This post answers the question those posts kept circling: *what actually makes it move?* For the policy-transmission side of the story — exactly how the Fed's decisions ripple through the economy — we will link out to the [macro-trading take on interest rates as the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) rather than re-derive it here. Our job is the curve itself: which lever moves which maturity, and by how much.

## Foundations: the curve, its parts, and the units we measure moves in

Before we can talk about *what moves* the curve we need to be crystal-clear about *what the curve is* and the vocabulary of how it moves. None of this requires a finance background; we will build every term from zero.

**The yield curve** is a single picture: a plot of the interest rate (the *yield*) you earn on government debt against how long until that debt matures. On the left are the shortest maturities — a 3-month Treasury bill, a 1-year bill. On the right are the longest — the 10-year note, the 30-year bond. Connect the dots and you get a curve. Most of the time it slopes gently upward: longer loans pay more, because you are tying up your money longer and bearing more uncertainty. A *yield* itself is just the annual return you earn if you buy the bond and hold it to maturity, expressed as a percentage. When someone says "the 10-year is at 4.5%," they mean a 10-year Treasury note is priced so that holding it to maturity earns you about 4.5% a year.

**A basis point** (bp) is one hundredth of a percentage point — 0.01%. So 25 basis points is a quarter of a percent, and 100 basis points is a full percentage point. Bond people quote everything in basis points because the moves that matter are small: a yield going from 4.50% to 4.75% is "a 25-basis-point move," and that small-sounding change can move a bond's price by several percent. We will live in basis points for the rest of this post.

**The parts of the curve** have nicknames, and you need them:

- The **front end** (sometimes "the short end") is the shortest maturities — overnight out to about 2 years. The 3-month bill and the 2-year note live here.
- The **belly** is the middle — roughly the 2-year through the 7-year. The 5-year note is the belly's center.
- The **long end** (or "the back end") is the longest maturities — the 10-year, 20-year, and 30-year.

These are not arbitrary bins. As we will see, they correspond to which *force* dominates that maturity, which is exactly why the front and the long end can disagree.

**Two ways the curve can move.** Any change in the curve can be decomposed into two basic motions, and naming them is half the battle:

- A **shift** (also called a *level* move) is when the whole curve moves up or down together — every maturity's yield rises by roughly the same amount, or falls by roughly the same amount. The curve keeps its shape but slides vertically. A level shift up means *all* borrowing got more expensive.
- A **twist** is when the *slope* of the curve changes — the front and the back move by *different* amounts, so the curve gets steeper or flatter. A **steepening** means the gap between long and short yields widens (the long end rose more, or the short end fell more). A **flattening** means that gap narrows. An **inversion** is the extreme: the front end ends up *higher* than the long end, so the curve slopes downward.

Why does this distinction matter so much? Because *the kind of move tells you the cause*. A pure level shift usually means a force hit every maturity at once — a broad repricing of the whole interest-rate world. A twist means a force hit one end harder than the other — and since we know which forces own which end, a twist is a clue to *which* force just moved. This is the core analytical skill of the post: see the motion, infer the driver.

**Real versus nominal, the one decomposition that underlies everything.** A nominal yield — the headline number you see quoted — can be split into two pieces: a *real yield* (the return after stripping out inflation) and *expected inflation* (the compensation for the purchasing power you lose while you hold the bond). In a rough but powerful identity:

$$\text{nominal yield} \approx \text{real yield} + \text{expected inflation} + \text{(small risk terms)}$$

where the real yield reflects the economy's underlying growth and the Fed's policy stance, and expected inflation reflects what the market thinks prices will do. This split is why inflation news and growth news move the curve through *different* channels — they hit different terms in that sum. We lean on the [cross-asset treatment of real yields](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) for the asset-pricing implications; here we just need the decomposition itself.

### Our running example: a Treasury benchmark and one eventful morning

To keep everything concrete, we will trace a single hypothetical morning across the whole post. Picture the US Treasury curve at the open, with a few benchmark yields:

- 3-month bill: **4.30%**
- 2-year note: **4.30%**
- 5-year note: **4.20%**
- 10-year note: **4.45%**
- 30-year bond: **4.65%**

Then two things happen before lunch. First, the monthly inflation report comes out **0.5 percentage points hotter** than expected — the kind of upside surprise that makes the market fear the Fed will have to keep rates higher for longer. Second, the Treasury announces it will **increase its quarterly issuance by \$300 billion** — more bonds for the market to absorb. We will ask, force by force, how each of these moves the 2-year versus the 10-year, and assemble the answer at the end. By the time we are done you will be able to look at a real day's move and reason your way to its cause.

#### Worked example: a level shift versus a twist, in dollars

Let's make "shift" and "twist" tangible. Suppose you own two bonds: a 2-year note and a 10-year note, each with a \$100,000 face value. A 2-year has a *duration* — its price sensitivity to rates, which we built up in the [duration post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) — of about 1.9 years; a 10-year, about 8.5 years. Duration tells you that a 1% (100 bp) rise in yield costs you roughly *duration* percent of the bond's price.

Now run two scenarios.

*Scenario A — a level shift:* every yield rises 25 bps. Your 2-year falls about 1.9 × 0.25% = 0.48%, or \$475. Your 10-year falls about 8.5 × 0.25% = 2.13%, or \$2,125. Total loss: about \$2,600. Notice that even though *the yield change was identical*, the long bond lost over four times as much — because duration is longer at the back.

*Scenario B — a twist (a flattening):* the 2-year rises 50 bps but the 10-year is unchanged. Your 2-year falls about 1.9 × 0.50% = 0.95%, or \$950. Your 10-year loses nothing. Total loss: \$950.

*The intuition: a "shift" and a "twist" are not just chart shapes — they hit your portfolio through completely different maturities, so knowing which one is happening tells you exactly where your money is at risk.*

## Force #1: the Fed owns the front end

Start where the curve is simplest. The shortest maturities — the 3-month bill, the 6-month, the 1-year, the 2-year — are governed, to a first approximation, by exactly one thing: **what the market expects the Federal Reserve's policy rate to be over the life of that bond.**

Here is the logic, and it is airtight. The Fed directly controls one rate: the *federal funds rate*, the overnight rate at which banks lend reserves to each other, which the Fed pins to a target range through its operations. (For the full plumbing of how it does this, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).) Now ask: what is a 2-year Treasury, *really*? It is a way to be invested in essentially risk-free overnight money, locked in for two years. But you could also just roll overnight money yourself — invest overnight, get it back tomorrow, reinvest, and keep rolling for two years at whatever the Fed's rate happens to be each night. By the no-arbitrage logic we built in the [forward-rates post](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be), those two strategies must earn roughly the same. So the 2-year yield is approximately **the average expected fed funds rate over the next two years**, plus a tiny premium.

That single sentence explains why the front end follows the Fed. If the market expects the Fed to hold rates at 4.30% for two years, the 2-year sits near 4.30%. If the market expects the Fed to hike to 5.30% over the next year and hold, the 2-year jumps to reflect that *average* — something like 4.80%. The 2-year is not a forecast of the Fed's *current* rate; it is a forecast of the Fed's *average rate over the bond's life*. This is why the 2-year often moves *before* the Fed does: the moment the market becomes convinced hikes are coming, it bids the 2-year up, weeks or months ahead of the actual meeting.

![A Fed rate hike lifting the front of the curve sharply while the long end barely moves, showing the curve flattening as the 2-year jumps far more than the 10-year](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-2.png)

The figure shows what a hike does to the curve's shape. Suppose the Fed surprises the market with a 50 bp hike and signals more to come. The front end — which lives and dies by the expected policy path — leaps. In the figure the 2-year jumps about 45 bps. But the 10-year, which is governed by *other* forces (growth, inflation, term premium), barely moves — it rises maybe 5 bps. The result is a **flattening**: the gap between the 10-year and the 2-year shrinks. Push hard enough and the curve inverts, with the 2-year above the 10-year. This is exactly the mechanism behind the famous recession signal — the Fed hiking the front end up into a long end that is pricing slower growth ahead. The [macro-trading post on reading the curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) takes the recession-forecasting angle; for us, the point is mechanical: hikes hit the front, so hikes flatten.

#### Worked example: pricing a hike into the 2-year

The Fed's policy rate is 4.30% and the 2-year sits at 4.30% — the market expects no change. Then the hot inflation report from our running example lands. The market now believes the Fed will hike by 25 bps at each of the next two meetings — call it +50 bps total — reaching 4.80% within six months, and then hold there.

What should the 2-year do? Roughly, the 2-year is the *average* expected overnight rate over the next two years. For the first six months the rate averages around 4.55% (climbing from 4.30% to 4.80%); for the remaining eighteen months it sits at 4.80%. The two-year average is approximately:

$$\frac{(0.5 \times 4.55\%) + (1.5 \times 4.80\%)}{2} \approx 4.74\%$$

So the 2-year should jump from 4.30% to roughly 4.74% — a **+44 bp move** — purely on the changed Fed expectation. In our running example we will round this to a +30 bp move from inflation alone (the market doesn't fully price two hikes off one report), but the mechanism is exactly this.

*The intuition: the front end is a running average of the Fed's expected path, so it moves the instant the market changes its mind about the Fed — not when the Fed actually acts.*

### Why the front end is the "cleanest" part of the curve

There is a reason analysts treat the front end as the most readable part of the curve: it is the part with the *fewest* competing forces. Growth and inflation matter at the front too, but only through one channel — how they change the Fed's expected path. The term premium (the extra yield for long maturities) is negligible at 3 months or 2 years because there is almost no "long" to be premium for. And supply effects are muted because the Fed's grip is so dominant. So at the front, you really can read the price as "the market's forecast of the Fed," with little noise. This is why the 2-year is the single best market gauge of where monetary policy is headed — it is the Fed's expected path, distilled into one number.

## The influence figure: the 2-year is the Fed's shadow

If you only believe one claim in this post, make it this one, because it is the most empirically robust relationship in all of fixed income: **the 2-year Treasury yield tracks the federal funds rate so tightly that the two lines are almost the same picture.** This is the "front end follows the Fed" thesis made visible.

![The fed funds rate and the 2-year Treasury yield plotted together over five years, the two lines hugging each other closely with the 2-year leading the fed funds rate slightly at each turn](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-3.png)

The figure plots two lines over a multi-year stretch: the fed funds rate (the Fed's actual policy rate, which moves in discrete steps at meetings) and the 2-year Treasury yield (which moves continuously, every second the market is open). Look at how they hug each other. When the Fed is on hold, the 2-year sits flat near the funds rate. When a hiking cycle begins, the 2-year doesn't wait for the meetings — it climbs *ahead* of the funds rate, pricing the hikes the market sees coming. At the top of the cycle, when the market starts to anticipate cuts, the 2-year *peels away below* the funds rate, falling while the Fed is still on hold, because it is already pricing the cuts. The 2-year is the funds rate's shadow, cast a few months into the future.

This leading behavior is not a quirk; it is the whole point. The 2-year *is* the market's forecast of the average funds rate over two years, so by construction it moves before the Fed acts. When you read that "the market is pricing three cuts next year," that information is *literally encoded in the gap between the 2-year and the current funds rate*. The front end is a forecast you can read off a screen.

#### Worked example: reading the cut expectation out of the gap

Suppose the fed funds rate is 5.30% but the 2-year Treasury is trading at 4.55% — a full 75 bps *below* the policy rate. What is the market saying?

The 2-year is the average expected funds rate over two years. For that average to be 4.55% when the rate *starts* at 5.30%, the market must expect the funds rate to fall substantially over the period. Roughly, if the rate stays near 5.30% for the first half-year and then is cut steadily, ending around 3.80%, the two-year average lands near 4.55%. That implies the market is pricing about **150 bps of cuts** over the next year and a half — six quarter-point cuts. You extracted a detailed forecast of Fed policy from a single number, with arithmetic you can do on a napkin.

*The intuition: the gap between the 2-year and today's policy rate is the market's forecast of where the Fed is headed — a negative gap means cuts are priced, a positive gap means hikes are priced.*

## Force #2: growth expectations drive the long end

Now travel to the other end of the curve, where the Fed's grip fades and a slower, deeper force takes over: **the market's expectation of long-run economic growth.**

Why should growth move long yields? Two linked reasons. First, the real economy's underlying growth rate sets the *natural* level of interest rates — economists call it r-star (r\*), the neutral real rate that neither stimulates nor restrains the economy. A faster-growing economy can sustain higher real interest rates, because capital earns more when the economy is productive and investment opportunities are plentiful. When the market revises its view of trend growth *upward*, it revises r\* upward, and long real yields rise to match. Second, stronger growth eventually pulls the Fed's policy rate up too, but the long end front-runs that by pricing a higher *average* rate over the whole 10 or 30 years — a horizon long enough that the current hiking cycle is a small part of the story and the *trend* dominates.

So the long end is, at its core, a bet on the economy's long-run trajectory. A 30-year yield embeds the market's guess at average growth and inflation over an entire generation. That is why long yields are so much more stable than short ones in normal times — a single data point barely moves your view of the next thirty years — and why, when they *do* move, it signals a genuine shift in the market's view of the economy's destiny.

#### Worked example: a growth upgrade lifting the 10-year through r\*

Suppose the market has believed for years that the US economy's trend real growth is about 1.8%, implying a neutral real rate (r\*) of roughly 0.5%. With expected long-run inflation at 2.0%, that puts the "fair" long-run nominal rate around 0.5% + 2.0% = 2.5%, and the 10-year trades near there in a calm regime.

Now a run of strong productivity data convinces the market that trend growth has stepped up to 2.5%, lifting r\* to about 1.5%. Holding inflation expectations at 2.0%, the new fair long-run nominal rate is 1.5% + 2.0% = 3.5% — a full **100 bps higher**. The 10-year drifts up toward 3.5% not because of any single Fed meeting, but because the market repriced the economy's *potential*. The front end barely budges, because the next two years of Fed policy haven't changed much; the move is concentrated at the back.

*The intuition: the long end is a vote on the economy's trend growth, so a durable growth upgrade lifts long yields through a higher neutral rate, even if the Fed hasn't moved.*

### Growth and the belly: where the two forces meet

The belly of the curve — the 5-year is its center of gravity — is where the Fed channel and the growth channel overlap, and it is worth dwelling on, because the belly is where most "macro" trades live. A 5-year yield is the average expected short rate over five years. Five years is long enough that the *current* hiking or cutting cycle is only part of the story — the rest is the market's view of where the economy, and therefore the neutral rate, settles after the cycle ends. So the belly responds to growth news through two doors at once: a strong growth print raises the *near-term* Fed path (more hikes sooner, the front-end door) *and* raises the *terminal* rate the economy can sustain (a higher r\*, the long-end door). That double exposure is exactly why the belly is the most volatile part of the curve on a growth surprise — it gets hit from both sides.

This is also why curve traders obsess over the belly's position relative to the wings. When the belly is "cheap" (its yield high relative to a smooth line drawn from the 2-year to the 10-year), the market is pricing an aggressive but short-lived hiking cycle; when the belly is "rich" (yield low relative to the wings), the market expects the Fed to cut and the economy to slow. The shape of the belly is a compressed forecast of the *entire* policy cycle, not just its next step. The [post on trading the curve](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies) turns this into the butterfly trade; here the point is conceptual: the belly is where growth expectations and the Fed path are hardest to disentangle, because both forces own a piece of it.

## Force #3: inflation expectations also drive the long end

Growth's twin at the long end is **inflation expectations**. Recall the decomposition: a nominal yield is a real yield plus expected inflation. The long end is sensitive to both terms, but the inflation term is where the drama usually lives, because inflation expectations can move fast and far when the market loses confidence.

The mechanism is straightforward compensation. If you lend the government money for 30 years and you expect prices to rise 2% a year, you need at least 2% a year of yield just to break even on purchasing power, plus a real return on top. If your inflation expectation jumps to 3%, you now demand an extra percentage point of yield, or you simply won't buy the bond — you'd be guaranteeing yourself a loss in real terms. So a rise in *long-run expected inflation* pushes long nominal yields up, basis point for basis point, through the inflation-compensation term.

![Rising long-run inflation expectations lifting the long end of the curve far more than the front end, with the 30-year rising 50 basis points while the 2-year stays anchored by the Fed, producing a bear steepening](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-4.png)

The figure shows the effect. When long-run inflation expectations rise — say the market's *breakeven inflation rate*, the inflation rate implied by the gap between regular Treasuries and inflation-protected ones, climbs 0.5% — the long end lifts to demand the extra compensation. The 30-year jumps maybe 50 bps. But the front end stays roughly anchored, because the Fed is expected to *fight* the inflation, capping how high short rates need the bond market's help to go. The result is a **bear steepening**: yields rise (a "bear" market for bonds, since higher yields mean lower prices) and the curve steepens, because the long end rose more than the front.

But here is the subtle, important part — and it is where the *same news hits maturities differently*. The long end's reaction to an inflation surprise depends entirely on **the Fed's credibility.** If the market believes the Fed will respond forcefully and bring inflation back to target, then a one-month inflation spike *raises the front end* (rate hikes coming) but *barely moves the long end*, because long-run inflation expectations stay anchored — the market trusts the Fed to win. That's a flattening. But if the market *loses faith* — if it thinks the Fed will let inflation run — then long-run inflation expectations un-anchor, the long end blows out, and you get a steepening that signals a genuine inflation problem. So one inflation report can flatten the curve (credibility intact) or steepen it (credibility lost), and *which way the long end goes is the market's verdict on the central bank.*

#### Worked example: the credible Fed versus the doubted Fed

Our running example's hot inflation print is +0.5% above expectations. Trace the long end's reaction under two regimes.

*Regime A — the Fed is credible.* The market thinks: "Inflation is hot, so the Fed will hike harder — front end up. But the Fed will succeed in bringing inflation back to 2%, so long-run inflation expectations are unchanged." The 2-year jumps +30 bps (hike fears); the 10-year rises just +15 bps, mostly sympathy and a touch of term premium. The curve **flattens**. The bond market is saying: *painful, but contained.*

*Regime B — the Fed is doubted.* The market thinks: "Inflation keeps surprising high and the Fed is behind the curve. We no longer trust 2% to hold." Long-run breakeven inflation jumps 0.4%. The 2-year still rises +30 bps, but now the 10-year rises +45 bps and the 30-year +55 bps, driven by inflation compensation. The curve **steepens**. The bond market is saying: *the inflation is getting away.*

Same +0.5% print, opposite curve moves — because the long end is pricing not the inflation itself but the *credibility of the response*.

*The intuition: a hot inflation print always lifts the front (hike fears), but whether the long end rises with it depends on whether the market still believes the Fed will win — the long end is a credibility meter.*

## Force #4: the term premium — the price of uncertainty

There is a fourth force that lives almost entirely at the long end and that beginners almost always miss, because it isn't tied to any single piece of news: the **term premium.** It is the extra yield investors demand, *over and above* the expected average of future short rates, simply for the risk of locking their money up for a long time.

Here is why it exists. Even if you knew the exact average path of the Fed's rate over the next ten years, a 10-year bond would still be *risky* to hold, because its price swings a lot when rates move (long duration), and because the longer your horizon, the more can go wrong — an inflation shock, a fiscal crisis, a liquidity squeeze. Investors are not indifferent to that risk; they demand to be paid for bearing it. That payment is the term premium. It is the reason the curve slopes up even when the Fed is expected to hold rates flat: the long end carries a premium the front end does not. We built the expectations-versus-term-premium decomposition in the [post on why the curve slopes up](/blog/trading/fixed-income/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations); here we treat the term premium as the fourth mover.

The term premium is invisible — you cannot observe it directly; it is what's left in the long yield after you subtract the expected average short rate. But it is real money, and it *moves*. It rises when uncertainty about the future is high — when inflation is volatile, when the fiscal outlook is shaky, when the supply of bonds is ballooning. It falls when bonds are scarce and prized as a safe haven, or when a big, price-insensitive buyer (a central bank doing QE, a foreign reserve manager) is hoovering up duration. A *rising* term premium lifts the long end and steepens the curve; a *falling* term premium compresses the long end and flattens it. Much of the great bond rally of the 2010s — when 10-year yields fell toward 1.5% — was a *collapse in the term premium*, driven by central-bank buying and a global glut of savings, not by expectations of ever-lower Fed rates.

#### Worked example: decomposing a 10-year yield into its pieces

Suppose the 10-year Treasury yields 4.45%. Where does that number come from? Decompose it:

- **Expected average real short rate** over ten years: about 1.0% (the market's view of average r\* over the decade).
- **Expected average inflation** over ten years: about 2.3% (long-run inflation expectations).
- **Term premium**: about 1.15% (the compensation for locking up money for a decade amid uncertainty).

Wait — 1.0% + 2.3% + 1.15% = 4.45%. The yield is the sum of three independent pieces. Now you can see how the *same* 4.45% could be reached different ways: a world with low growth (real rate 0.5%), tame inflation (2.0%), and a fat term premium (1.95%) gives the same headline. The headline yield hides the story; the *decomposition* is the story. And it shows why a move in *any one* of the three — a growth upgrade, an inflation scare, a supply-driven premium widening — can move the long end without the others changing at all.

*The intuition: a long yield is the sum of an expected-rate piece and a term-premium piece, so two yields that look identical can carry completely different forecasts about growth, inflation, and risk.*

### What actually makes the term premium move

Because the term premium is the residual — the part of the long yield left over after subtracting the expected rate path — it can feel like a fudge factor. It is not. It moves for identifiable reasons, and recognizing them is how you tell a "real" long-end move from noise:

- **Inflation uncertainty.** The term premium is partly compensation for *not knowing* future inflation. When inflation is stable and predictable (the 2010s), the premium shrinks; when inflation is volatile and hard to forecast (the 1970s, 2022), it balloons, because lending long becomes a genuine gamble on purchasing power. A rise in inflation *uncertainty* lifts the long end even if expected inflation itself is unchanged.
- **Supply and the fiscal outlook.** More bonds to hold, and more doubt about whether the government will keep its debt sustainable, raise the premium investors demand to own long duration. This is the channel through which deficits reach the curve, covered in Force #5.
- **The duration-supply balance.** When a giant price-insensitive buyer (the Fed via QE, or foreign reserve managers) removes long bonds from the market, the remaining investors hold less duration risk and accept a lower premium. When that buyer reverses (QT), the premium re-widens. The long end's level over the past two decades has tracked this balance more than it has tracked the expected Fed path.
- **The safe-haven bid.** In a flight to safety — a stock crash, a banking scare — investors will *pay up* for the safety of Treasuries, compressing or even pushing the term premium negative. This is why long yields can fall during a crisis even as the world looks riskier: the premium for owning the safest asset collapses as everyone reaches for it at once.

The practical upshot: when the long end moves and you cannot pin it on a change in the expected Fed path or in inflation expectations, the term premium is your suspect — and the four bullets above are the line-up.

## Force #5: supply and demand — issuance, QE/QT, and who's buying

The forces so far — Fed, growth, inflation, term premium — are about *expectations*. The last force is brute *plumbing*: **how many bonds are being sold, and how many people want to buy them.** Bonds are not magic; they are a product, and like any product their price falls when supply outstrips demand. This force lives mostly at the long end (where the term premium absorbs it) and has become enormously more important as government deficits have ballooned.

There are three supply-and-demand channels worth knowing:

**Treasury issuance (supply).** When the government runs a bigger deficit, it must sell more bonds to fund it. More bonds for the market to absorb means, all else equal, the price has to fall — the yield has to rise — to entice enough buyers. A surprise *increase* in the size of Treasury auctions, especially of long-dated bonds, pushes long yields up by widening the term premium. The [macro-trading post on deficits and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) digs into the politics and the magnitudes; the mechanism is just supply-and-demand.

**Quantitative easing and tightening (the Fed as a buyer or seller).** In *QE*, the Fed creates money and buys bonds — usually long-dated ones — to push long yields down. It is a giant, price-insensitive buyer removing duration from the market, which compresses the term premium and flattens the curve. In *QT* (quantitative tightening), the Fed does the reverse: it lets its bonds mature without replacing them, *adding* supply back to the market, which lifts long yields. So the Fed moves the long end too — not through the policy rate, but through its *balance sheet*. The [central-bank toolkit post](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) covers QE/QT in depth.

**Foreign and safe-asset demand.** US Treasuries are the world's premier safe asset and reserve currency instrument. Foreign central banks, sovereign wealth funds, pension funds, and insurers buy them by the trillion — not because they're forecasting the Fed, but because they *need* safe duration to match liabilities or park reserves. This price-insensitive demand pushes long yields *down* and compresses the term premium. When that demand wavers — a foreign central bank diversifying away from dollars, or a flight *out* of Treasuries in a crisis — long yields can spike on the supply-demand channel alone, with no change in growth or inflation expectations at all.

![A pipeline showing a large fiscal deficit forcing more Treasury issuance, the Fed shrinking its balance sheet through QT, more bonds for private buyers to absorb, the price falling to clear the supply, the long yield rising as the term premium widens, and the effect spilling outward to mortgages and corporate bonds](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-5.png)

The figure traces the supply channel end to end. A big deficit forces the Treasury to issue more bonds. At the same time, QT means the Fed has stopped being a buyer and is in fact letting its holdings run off — *adding* to the net supply private investors must absorb. More bonds chasing the same pool of buyers means the price must fall (yield must rise) to clear the market. The long yield rises as the term premium widens, and because the 10-year is the benchmark off which mortgages and corporate bonds are priced, the higher long yield spills out into the broader economy. This is purely a quantity story — no one revised their growth forecast or their inflation forecast; there were simply too many bonds.

#### Worked example: how much does \$300 billion of extra issuance move the 10-year?

Our running example includes a \$300 billion increase in quarterly issuance. How much should that lift the 10-year? There is no exact formula — supply effects are noisy and depend on how the market is positioned — but we can sanity-check with a rule of thumb from the research literature: a sustained increase in the supply of long-duration Treasuries equal to about 1% of GDP tends to lift the term premium by roughly 5 to 10 bps.

US GDP is about \$28 trillion, so \$300 billion is about 1.1% of GDP. If the increase is concentrated in longer maturities and the market sees it as persistent, that points to roughly a **+10 to +20 bp** rise in the 10-year term premium. In our running example we will use **+20 bps** for the 10-year from supply, fading to near zero at the 2-year (the front end barely cares about long-bond supply). Note how *uneven* the effect is across the curve: supply is a long-end story.

*The intuition: bonds are a product, and flooding the market with extra long-dated supply forces the price down and the long yield up through the term premium — a quantity effect that barely touches the Fed-anchored front end.*

## Putting it together: the same news, different maturities

We now have all five forces. The payoff is being able to take a single piece of news and predict how it moves *each part* of the curve — because each force has a home, the same headline lands differently at the front and the back. Let's assemble our running example.

Recall the morning: a +0.5% inflation surprise and a \$300 billion issuance increase. Walk the curve maturity by maturity.

**The 2-year.** Two forces touch it. The inflation print raises rate-hike fears, lifting the front end through the Fed channel — call it **+30 bps**. The issuance increase barely touches the 2-year, because long-bond supply isn't the 2-year's concern — **+2 bps**. Net: the 2-year rises about **+32 bps**, almost entirely Fed-driven.

**The 5-year (the belly).** It's a blend. Some Fed-path effect from the inflation print (**+25 bps**) and some supply pressure as the belly absorbs part of the issuance (**+8 bps**). Net: about **+33 bps**.

**The 10-year.** The inflation print's effect here depends on credibility — assume the Fed is mostly credible, so the long-end inflation reaction is muted at **+15 bps**. But the issuance increase hits the 10-year squarely through the term premium — **+20 bps**. Net: about **+35 bps**.

![A growth surprise causing a bear steepening, with the long end rising more than the front end as stronger expected growth lifts the neutral rate and the curve sells off and steepens at once](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-6.png)

The figure above shows the *shape* of a related move — a bear steepening — for contrast: when the dominant force is at the long end (growth or supply), the long end rises *more* than the front, and the curve steepens as it sells off. Our running example is closer to a parallel shift with a slight twist, because the inflation print pushes the front and the supply pushes the back by similar amounts — but you can see how, if the supply effect or a growth upgrade dominated, the same machinery would produce a clean steepener.

![A decomposition matrix breaking one curve move into its drivers, with rows for the inflation print, the supply increase, and the net move, and columns for the 2-year, 5-year, and 10-year, showing how each driver hits each maturity by a different amount](/imgs/blogs/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply-7.png)

The matrix lays the whole decomposition out at once. Read across each row to see how one driver — the inflation print, then the supply increase — hits the 2-year, 5-year, and 10-year differently. Read down each column to see how the two drivers stack up at a given maturity. The bottom row is the net: roughly +32 bps at the 2-year, +33 at the 5-year, +35 at the 10-year. This is the analytical superpower the whole post was building toward: **given a move, attribute it to forces; given the forces, predict the move.** A bond strategist's daily job is essentially filling in this matrix.

#### Worked example: attributing a real-looking day's move

Suppose you arrive at your desk and see: the 2-year up 32 bps, the 5-year up 33 bps, the 10-year up 35 bps, and the 2s10s slope (the 10-year minus the 2-year) up 3 bps — a tiny steepening. What happened?

Reason backwards through the forces. The big, broad rise across all maturities (a level shift of ~32 bps) screams a *Fed repricing* — the market suddenly expects materially higher rates, which only a meaningful macro surprise (an inflation print, a blowout jobs number) would cause. The fact that the long end rose *slightly more* than the front (the 3 bp steepening) tells you a *long-end-specific* force was also at work — extra supply, or a small loss of inflation credibility — adding a touch on top of the Fed story. If instead the front had risen *more* (a flattening), you'd conclude the move was purely a Fed-path repricing with the long end skeptical that the hikes would last. The *level* told you the Fed; the *twist* told you the second force.

*The intuition: every curve move decomposes into a level piece (the Fed and broad repricing) and a slope piece (the long-end forces), and reading those two numbers tells you which forces were in the room.*

## Common misconceptions

**"The Fed sets all interest rates."** The Fed directly sets *one* rate — the overnight federal funds rate — and through it, dominates the front end. It has powerful *indirect* influence on the long end through QE/QT and forward guidance. But it does *not* set the 10-year or 30-year yield. Those are set by the market's collective view of growth, inflation, the term premium, and supply. This is why the Fed can hike the funds rate aggressively and watch the 10-year *fall* (as happened in several "conundrum" episodes) — the long end answers to forces the Fed only partly controls.

**"When rates go up, the whole curve goes up by the same amount."** Almost never. A pure parallel shift is the textbook simplification, not the norm. Real moves are mostly *twists* — the front and the back moving by different amounts because different forces hit them. The entire skill of reading the curve is in the *differences* between maturities, which a "rates went up" mental model erases.

**"A hot inflation print is always bad for long bonds."** Counterintuitively, no. If the Fed is credible, a hot inflation print can *help* the long end relatively: it makes the market expect *aggressive hikes now*, which lifts the front but reassures the market that inflation will be crushed, keeping long-run inflation expectations anchored. The long end can even rally (yields fall) on a hot print if the market concludes the Fed will over-tighten and cause a recession. The long end prices the *endgame*, not the data point.

**"Treasury supply doesn't matter — bonds always find buyers."** They do find buyers, but *at a price*. The clearing price of an extra few hundred billion in long-dated issuance is a higher yield, often 10–20 bps of term premium for a 1%-of-GDP supply shock. Investors who insist supply is irrelevant are usually the ones surprised when a heavy auction calendar steepens the curve for no "macro" reason at all.

**"The term premium is just an academic abstraction."** It is unobservable, but it is the single biggest swing factor in long yields over multi-year horizons. The 2010s bond rally and the 2022–2023 selloff were both, in large part, term-premium stories — a collapse and then a re-widening — not changes in the expected Fed path. Ignoring the term premium means mis-attributing the biggest moves in the bond market.

**"The 2-year is a forecast of the Fed's current rate."** It is a forecast of the Fed's *average* rate over two years. That's why it can sit far below the current policy rate (pricing cuts) or far above it (pricing hikes). The *gap* between the 2-year and today's funds rate is the market's forecast of the Fed's direction — read it, and you read the market's mind on policy.

## How it shows up in real markets

**The 2004–2006 "conundrum."** When Alan Greenspan's Fed raised the funds rate from 1% to 5.25% across seventeen consecutive meetings, the textbook said long yields should follow. Instead the 10-year barely moved, hovering near 4.5% the whole time — Greenspan himself called it a "conundrum." The explanation was the forces in this post: the front end dutifully followed the Fed up (the funds rate channel), but the long end was held down by a collapsing term premium — a global savings glut and heavy foreign central-bank buying of Treasuries were soaking up duration faster than the Fed could push it up. The curve flattened and then inverted. It is the cleanest historical demonstration that the Fed owns the front, not the back.

**The 2013 "taper tantrum."** In May 2013, Fed Chair Ben Bernanke merely *hinted* that the Fed might slow ("taper") its QE bond-buying. The Fed had not changed the funds rate and would not for years. Yet the 10-year yield rocketed from about 1.6% to 3.0% over the following months. This was a pure *supply-and-term-premium* shock: the market suddenly realized the giant price-insensitive buyer was going to step back, so the term premium it had compressed came roaring back. The front end barely moved; the long end blew out. A textbook case of the balance-sheet channel moving the long end while the policy rate sat still.

**The 2022 inflation shock and bear steepening, then flattening.** As US inflation surged to 9% in 2022, the bond market repriced violently. Early on, before the Fed acted, long yields rose on a loss of inflation credibility — a bear steepening, the long end pricing the inflation getting away. Then, as the Fed launched the fastest hiking cycle in decades (the funds rate went from near zero to over 5%), the front end exploded upward and the curve *inverted* — the 2-year rose far above the 10-year as the market bet the aggressive hikes would tame inflation but cause a recession. The 2022 episode ran the full playbook: an inflation-driven steepening (credibility wobble) giving way to a Fed-driven flattening and inversion (credibility restored). The [cross-asset case study of 2022](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell) covers the cross-asset fallout.

**The 2023 regional-bank crisis and the front end's violence.** In March 2023, when Silicon Valley Bank failed, the 2-year Treasury yield fell about 60 bps in a *single day* — one of the largest one-day moves in its history. Why the 2-year and not the 30-year? Because the front end is the Fed forecast, and in one day the market went from pricing more hikes to pricing emergency *cuts*. The long end moved far less, because the crisis didn't change the thirty-year growth-and-inflation outlook nearly as much as it changed the next-few-months Fed path. It is the mirror image of the conundrum: a Fed-expectations shock that lives almost entirely at the front. The [post on SVB and the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) tells the institutional story.

**The 2023 long-end selloff on supply.** In the autumn of 2023, the US 10-year yield pushed toward 5% even as inflation was *falling* and the Fed was on hold. Much of the move was attributed to a *supply and term-premium* shock: the Treasury announced heavier-than-expected long-dated issuance to fund a widening deficit, fiscal-sustainability worries grew, and a ratings agency downgrade rattled confidence. Growth and inflation expectations weren't deteriorating — there were simply too many bonds and a re-widening term premium. When issuance plans were later trimmed and the Treasury skewed toward shorter maturities, the long end rallied hard. A live demonstration that supply moves the long end on its own.

**The Volcker shock, 1979–1982.** When Paul Volcker's Fed drove the funds rate above 19% to break double-digit inflation, the front end went to the moon. But the *long* end told a more interesting story: 30-year yields stayed extremely high for years *after* short rates began falling, because the market did not yet believe inflation was permanently beaten — long-run inflation expectations (and the term premium) stayed elevated until credibility was painstakingly rebuilt. Only as the market came to trust that inflation was dead did the long end grind down, kicking off the great bond bull market of the 1980s–2010s. The episode is the ultimate illustration that the long end is a *credibility meter*, not a Fed thermometer. See [Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).

## When this matters to you, and further reading

This is not an abstraction that lives only on a trading desk. The 10-year Treasury yield — the long end we have spent this post dissecting — is the benchmark off which your 30-year mortgage rate, your car loan, and the discount rate on every stock in your retirement account are set. When you understand that the long end moves on growth, inflation, the term premium, and supply rather than on the Fed's last meeting, you understand why mortgage rates can keep climbing even after the Fed pauses, and why they can fall before the first cut. The front end, meanwhile, is where your savings-account and money-market yields live — and those *do* follow the Fed, almost one for one, which is why they move the instant policy expectations shift.

The practical takeaway is a habit of mind: when you see a big yield move in the news, don't ask "did rates go up?" Ask *which maturities moved, and by how much relative to each other?* A flattening is a Fed story; a bear steepening is a growth-or-supply story; a front-end collapse is a crisis-and-cuts story. The shape of the move is the message.

To go deeper: for the policy machinery behind the front end, read the macro-trading posts on [interest rates as the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and the [central-bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance). For the supply story, see [deficits, debt, and why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields). For the real-versus-nominal decomposition that underlies the inflation channel, see [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) and the macro take on [real versus nominal rates](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). And for the heavy mathematics of modeling the curve and decomposing the term premium formally, the quantitative-finance posts on [yield-curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling) and [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) are where this qualitative picture becomes equations. Within this series, the natural next step is the credit track — [credit risk, the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) — where we leave the risk-free curve behind and ask what happens when the borrower might not pay.

*This post is educational, not investment advice. It explains the mechanisms that move the yield curve; it does not recommend buying or selling any security.*

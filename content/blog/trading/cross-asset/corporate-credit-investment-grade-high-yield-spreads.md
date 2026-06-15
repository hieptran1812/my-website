---
title: "Corporate Credit: Investment Grade, High Yield, and the Spread"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A corporate bond is a Treasury plus a credit spread for taking default risk. This is what the spread pays you for, why high yield behaves like equity, and when in the cycle to actually own credit."
tags: ["asset-allocation", "cross-asset", "corporate-bonds", "credit-spread", "high-yield", "investment-grade", "default-risk", "fixed-income"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A corporate bond is a government bond plus an extra slice of yield, the *credit spread*, that pays you for taking the risk the company doesn't pay you back. Credit is a hybrid: part bond, part stock.
>
> - The spread is the price of fear. It grinds tighter in booms and blows out in recessions: high-yield spreads ran from ~300 bps in calm 2021 to ~2,000 bps at the 2008 peak.
> - Investment grade (IG) is mostly a rate bet (long, rate-sensitive); high yield (HY, "junk") is mostly a default bet and behaves like a quieter cousin of stocks, with a stock correlation around 0.75.
> - That correlation is the trap: HY fails as a diversifier exactly in a crash. In 2008 high yield lost ~26% while Treasuries *gained*.
> - The one number to remember: a ~450 bps HY spread, against a 4% default rate and 40% recovery, pays you ~240 bps for expected losses and leaves only ~210 bps as your actual risk premium. That thin cushion is why credit is "picking up pennies in front of a steamroller."

In March 2020, as the pandemic froze the world, something strange happened in the bond market. The bonds of perfectly solvent companies — airlines, hotels, but also software firms and grocery chains — fell as if half of corporate America were about to file for bankruptcy. The extra yield that high-yield bonds paid over US Treasuries, a number traders call *the spread*, exploded from around 350 basis points (a *basis point* is one hundredth of a percent, so 0.01%; 350 bps is 3.5%) to roughly 1,100 bps in a matter of weeks. For a few days, you could buy a basket of junk-rated corporate bonds yielding double digits.

Then the Federal Reserve announced it would, for the first time in history, buy corporate bonds directly. The spread collapsed almost as fast as it had widened. Anyone who bought credit in that three-week window — when the spread screamed "catastrophe" — earned equity-like returns over the following year, with most of the threatened defaults never arriving.

That whole episode is corporate credit in miniature: an asset whose entire character lives inside one number, the spread, which is really just the market's price tag on fear. The diagram below is the mental model we'll build the whole post around: a corporate bond is a risk-free Treasury yield with an extra layer stacked on top, and that extra layer is everything.

![Yield stack showing Treasury yield plus a credit spread equals corporate yield](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-1.png)

This post is the credit chapter of the cross-asset playbook. By the end you'll understand what a corporate bond actually *is*, what the spread decomposes into, why high yield behaves like a hybrid of bonds and stocks, why it betrays you as a diversifier in a crash, and — the payoff — when in the cycle you want to own it and when you very much do not. None of this is investment advice; it's the mechanism, the history, and the decision framework.

## Foundations: what a corporate bond actually is

Let's build from zero. You already understand the risk-free anchor of the whole system from [government bonds and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration): a US Treasury is a loan to the government that, in dollar terms, will be paid back for sure. The government can always print the dollars it owes, so there's no *default risk* — the risk of not being repaid. A corporate bond takes that same structure and removes the safety net.

### A bond is a loan with a fixed schedule

When you buy a corporate bond, you are lending money to a company. In return the company promises two things: a stream of *coupons* (fixed interest payments, usually twice a year) and the return of your *principal* (the face value of the loan) when the bond *matures* (reaches its end date). That's it. A bond is a loan sliced into a tradable security.

#### Worked example: the cash flows of one bond

You buy one corporate bond with a face value of \$1,000, a 5% annual coupon, and 3 years to maturity. The promised cash flows are:

- End of year 1: \$50 coupon (5% of \$1,000)
- End of year 2: \$50 coupon
- End of year 3: \$50 coupon **plus** \$1,000 principal back = \$1,050

You paid \$1,000 today and you're promised \$1,150 in total over three years (\$150 of coupons + \$1,000 back). If the company pays every cent, your return is exactly the 5% coupon. The catch is in that word *if*: a Treasury makes the same promise and the "if" is essentially certain; a corporation can fail, and then the schedule above is just a wish list.

The one-sentence intuition: a bond is a fixed schedule of payments, and the only question that matters for a *corporate* bond is whether the company actually makes them.

### Default risk is the whole difference

If the company runs out of money, it *defaults* — it stops paying. But "default" rarely means you lose everything. When a firm goes bankrupt, its assets get sold off and the proceeds are paid out to lenders in a strict order. Bondholders, especially *senior* ones (first in line), typically recover a meaningful fraction of their money. That fraction is the *recovery rate*.

Historically, senior unsecured corporate bonds recover roughly **40 cents on the dollar** in default. The flip side, what you lose, is the *loss given default* (LGD): if recovery is 40%, then LGD is 60%. Keep that 40%/60% pair in your head; it's load-bearing for everything that follows.

So the real economics of a corporate bond are: *you collect a slightly higher yield than a Treasury in exchange for accepting a small chance that, in a bad year, you get back only ~40 cents of every dollar you lent.* That higher yield is the credit spread, and the whole game is figuring out whether it's enough to cover the risk.

#### Worked example: what a default actually costs you

You hold ten identical junk bonds, \$1,000 face value each, so \$10,000 lent. Over the year, nine companies pay you in full and one defaults, recovering 40 cents on the dollar.

- Nine bonds pay back \$1,000 each plus, say, a \$70 coupon (7%) = \$1,070 each = \$9,630.
- The defaulted bond pays you a partial-year coupon plus a recovery of ~\$400 of its \$1,000 face ≈ \$400.
- Total back: \$9,630 + \$400 = \$10,030 on \$10,000 lent — a thin **+0.3%** for the year.

Now compare to the no-default world, where all ten paid \$1,070: you'd have \$10,700, a +7% year. That single default dragged your 7% promised yield down to a 0.3% realized return — it cost you ~6.7 percentage points. The math: one default in ten names is a 10% default rate; at 60% LGD that's 10% × 60% = 6% of capital lost, roughly the gap we found. This is why a single recession year, where the default rate jumps from 1% to 10%+, devastates a credit portfolio that looked perfectly healthy the year before.

The one-sentence intuition: a default doesn't cost you the whole bond, it costs you the loss-given-default — but even a handful of them in a bad year can swallow your entire spread.

### Prices move opposite to yields — and spreads

One more mechanical fact you need before the spread discussion makes sense: a bond's *price* moves opposite to its yield. When a bond's yield rises, its price falls, because the fixed coupons it pays are now worth less relative to the higher yields available elsewhere. For a corporate bond, the yield has two moving parts — the Treasury rate and the spread — so its price can fall for either reason. If Treasury rates rise, the price falls (a *duration* loss). If the spread widens because the company looks shakier, the price *also* falls (a *credit* loss). When traders say "spreads widened 50 bps today," they are simultaneously saying "the prices of those bonds fell," and roughly how much depends on the bond's duration: a 50 bps spread widening on a duration-5 bond knocks about 5 × 0.50% = 2.5% off the price. This is why the spread is not an abstraction — every basis point of spread movement is a real, immediate gain or loss on the position.

### Ratings: the credit ladder

Because no individual can analyze the balance sheet of every borrower, three rating agencies (Moody's, S&P, Fitch) grade issuers on a letter scale. The scale runs from AAA (safest) down through AA, A, BBB, BB, B, CCC, and finally D (in default). The single most important line on that ladder sits between **BBB-** and **BB+**:

- **BBB- and above is *investment grade* (IG)** — low default risk, the bonds pension funds and insurers are allowed to hold in size.
- **BB+ and below is *high yield* (HY)**, also bluntly called *junk* — higher default risk, higher yield to compensate.

That fault line is not cosmetic. Default rates climb steeply as you descend it. The figure below is the ladder, with the IG band on top and the HY band below, and a rough annual default rate on each rung.

![Credit ratings ladder from AAA to D split into investment grade and high yield](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-2.png)

Notice the asymmetry of the rungs. From AAA down to BBB, default rates barely move — they're all under half a percent a year. Then below the BBB- line they accelerate fast: ~1% at BB, ~3-4% at B, and into the double digits for CCC. A bond that slips from BBB- to BB+ — a so-called *fallen angel* — is the same company the day before and the day after, but its label, its buyer base, and its price can change sharply, because a wall of institutional money is contractually forbidden from owning junk.

### Yield = risk-free yield + credit spread

Now we can state the core equation of the whole post. The yield on a corporate bond is:

$$ y_{\text{corp}} = y_{\text{risk-free}} + s $$

where $y_{\text{corp}}$ is the corporate bond's yield, $y_{\text{risk-free}}$ is the yield on a Treasury of the same maturity, and $s$ is the *credit spread* — the extra yield you demand for taking default (and liquidity) risk. The spread is what you, the lender, charge the company for the privilege of borrowing from you instead of the government borrowing risk-free. Everything interesting about credit is in that one term $s$.

## The spread: the one number credit traders watch

Walk onto a credit desk and you will not hear people quote bond *prices* much. You'll hear them quote *spreads* — "IG is at 95 over," "CCCs blew out 80 today." The spread is the credit market's native unit because it strips out the part of a bond's yield that has nothing to do with credit (the level of interest rates) and isolates the part that does (default and liquidity risk).

### Why not just look at the yield?

Because a bond's yield mixes two completely different bets. Suppose a corporate bond yields 6%. Is that good? You can't say, because 6% might be 5% risk-free + 1% spread (a rock-solid IG bond in a high-rate world) or 2% risk-free + 4% spread (a shaky junk bond in a low-rate world). The yield alone confuses the *price of money* (rates, covered in [interest rates, the master variable](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders)) with the *price of credit risk*. The spread separates them. When you quote the spread, you've already subtracted out the Treasury, so two bonds with different maturities and different rate environments become directly comparable.

### OAS: the version traders actually use

The clean version of the spread is the *option-adjusted spread* (OAS). Most corporate bonds have embedded options — the company can often *call* (redeem early) the bond if rates fall, which is bad for you. OAS is the spread after mathematically stripping out the value of those options, so it measures pure compensation for credit and liquidity risk. When you see "ICE BofA US High Yield OAS," that's the OAS averaged across the whole high-yield market — the single thermometer for credit-market fear.

### Decomposing the spread

Here is the mental model that makes you dangerous on the topic. The spread is not one thing; it's a sum of three:

$$ s \approx \underbrace{p \times \text{LGD}}_{\text{expected loss}} + \underbrace{\pi_{\text{risk}}}_{\text{risk premium}} + \underbrace{\ell}_{\text{liquidity premium}} $$

where $p$ is the annual probability of default, $\text{LGD}$ is the loss given default (≈ 60%), $\pi_{\text{risk}}$ is the extra you demand for bearing *uncertain* losses (defaults cluster in bad times, exactly when you can least afford them), and $\ell$ is compensation for the fact that corporate bonds are harder to sell quickly than Treasuries.

The first term is the part you *expect* to lose on average. The second and third are your actual profit for taking risk — the part that, if you're right that the world isn't ending, you get to keep.

#### Worked example: decomposing a 450 bps high-yield spread

The long-run median high-yield spread is about 450 bps. Let's split it using realistic, long-run-average inputs: an annual default rate of 4% and a recovery of 40% (so LGD = 60%).

- **Expected credit loss** = default rate × LGD = 4% × 60% = **2.4%**, or 240 bps. This is the slice of the spread that, on average, just compensates you for bonds that go bad. You don't get to keep it; it's a refund of expected losses.
- **Risk + liquidity premium** = 450 bps − 240 bps = **210 bps**. *This* is your real pay for owning high yield: a little over 2% a year for bearing the uncertainty that defaults could spike and the friction that junk bonds are illiquid.

So when high yield is quoted at "450 over," you are being paid roughly 2.1% a year, net of expected defaults, to hold a basket of risky corporate loans. Whether that's a good deal depends entirely on whether 4% is the right default forecast — and in a recession it very much is not.

The one-sentence intuition: the headline spread looks generous, but after subtracting what you expect to lose to defaults, your true risk premium is often surprisingly thin.

![Spread decomposition into expected credit loss and risk plus liquidity premium across calm median and crisis regimes](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-5.png)

The figure above runs that same decomposition across three regimes. The crucial insight is what happens at the crisis end. When the 2008 spread hit ~2,000 bps, the *market* was implicitly pricing in a default rate around 13% (the green "risk premium" slice was huge). Defaults did spike — the 2009 high-yield default rate peaked near 13% — but for the bonds that *survived*, the realized loss was a fraction of what the spread priced. That gap, between the catastrophe priced and the catastrophe realized, is the entire reason buying credit in a panic can be a fat pitch.

### What actually moves spreads

If the spread is the thermometer, what's the weather it's reading? Four forces dominate, and a good credit allocator watches all of them:

- **The default outlook.** This is the fundamental driver. When earnings weaken, leverage rises, or a recession looms, expected defaults climb and spreads widen to compensate. The credit cycle — the slow build-up of corporate debt in good times and its painful unwind in bad — is the backbone of spread behavior.
- **Risk appetite.** Spreads are part of the broader risk-on/risk-off machine. When investors are confident, they reach for yield, bidding credit up and spreads down, often well past what fundamentals justify; when fear strikes, they dump risk indiscriminately. This is why high-yield spreads correlate so tightly with the stock market and the VIX — they're all reading the same mood.
- **Liquidity and funding.** Corporate bonds are far harder to trade than Treasuries. When market-wide liquidity dries up — dealers pull back, funding gets expensive — the liquidity slice of the spread balloons even if default risk hasn't changed. A chunk of every crisis blowout is pure liquidity premium, which is exactly why a central-bank liquidity backstop can collapse spreads so fast.
- **Supply and technicals.** A flood of new bond issuance, or a wave of fallen-angel downgrades dumping bonds into the HY index, can widen spreads mechanically through sheer supply, independent of fundamentals.

#### Worked example: separating a rate move from a spread move

You hold a corporate bond yielding 6%: a 4% Treasury rate plus a 2% spread. Over a month, two things could move your bond's price, and they mean opposite things.

- **Scenario A — rates fall to 3.5%, spread unchanged at 2%.** Your yield drops to 5.5%. On a duration-7 bond, the 0.5% yield fall lifts the price ~3.5% (7 × 0.5%). You made money, but for a *macro* reason — the economy or the Fed, not the company. This is good news that says nothing about credit quality.
- **Scenario B — rates unchanged at 4%, spread widens to 3% on bad company news.** Your yield rises to 7%. The 1% spread widening costs you ~7% of the price (7 × 1%) on the same duration. You lost money for a *credit* reason — the market now thinks default is likelier. This is a warning about the issuer.

Same 50-to-100 bps of yield movement, opposite stories. The whole reason credit desks quote spreads instead of yields is to make Scenario B visible without it being masked by Scenario A.

The one-sentence intuition: a bond's price moving tells you nothing until you split it into the rate part (macro) and the spread part (credit) — and only the spread part is news about the borrower.

## IG vs HY: two different bets wearing the same costume

Investment grade and high yield are both "corporate bonds," but as portfolio building blocks they are almost opposites. The difference comes down to which of two risks dominates: *duration* (sensitivity to interest rates) or *credit* (sensitivity to default).

### Investment grade is mostly a rate bet

IG bonds default so rarely (under 0.3% a year even at the BBB rung) that their spread is small and stable — typically 100 to 120 bps in calm times. Because the spread is a thin slice of the total yield, an IG bond's price is driven mostly by the *Treasury* part of its yield. And IG bonds tend to have long maturities (companies lock in cheap funding for 10, 20, 30 years), which means high *duration* — a measure of how much a bond's price moves when rates move. As a rule of thumb, a bond with a duration of 8 loses about 8% of its value if its yield rises 1%.

So owning IG is, to first approximation, owning a slightly-juiced Treasury. Your big risk is not that the companies default; it's that interest rates rise and your long-duration bond falls in price. That's exactly what 2022 taught everyone the hard way.

### High yield is mostly a credit bet

High yield flips this. HY bonds carry fat spreads (300 bps in calm, 800-2,000 in stress), so the *spread* dominates the yield, not the Treasury. They also tend to have *shorter* maturities and higher coupons, which gives them lower duration. The practical consequence: a junk bond barely cares whether the 10-year Treasury yield moves 50 bps. It cares enormously whether the issuer's business is deteriorating — whether it's heading toward default.

This is why high yield behaves like a quieter version of the stock of the same company. If a firm's prospects collapse, its stock falls (equity holders are wiped first) and its junk bonds fall too (because default probability just jumped). If the firm thrives, the stock soars while the junk bond just keeps paying its coupon and grinds toward par. High yield gets equity-like *downside* with bond-like *upside capped at par* — an asymmetry we'll return to.

### The comparison, side by side

| Attribute | Investment grade (IG) | High yield (HY / junk) |
|---|---|---|
| Typical rating | AAA to BBB- | BB+ to D |
| Calm spread | ~100-120 bps | ~300-450 bps |
| Dominant risk | Duration (rates) | Credit (default) |
| Annual default rate | < 0.3% | ~3.5-4% long-run |
| Behaves like | A juiced Treasury | A quieter stock |
| Worst single year (recent) | -13.0% (2022, rate shock) | -26.2% (2008, credit shock) |
| Who's forced to hold it | Pensions, insurers, IG-only funds | Specialist HY funds, risk-takers |

#### Worked example: same rate move, two very different reactions

Say the 10-year Treasury yield jumps 1% (100 bps) in a year, but the economy is fine and defaults don't rise.

- An **IG bond** with duration 8 and a stable 1% spread: its price falls roughly 8% from the rate move (8 × 1%), partly offset by its ~4-5% coupon income, for a total return around **-3% to -4%**. The rate move dominated.
- A **HY bond** with duration 4 and a 4% coupon-plus-spread cushion: its price falls roughly 4% from the rate move, but its fat ~7% yield income more than offsets it, for a total return around **+2% to +3%**. The rate move barely registered against the carry.

This is exactly what happened in 2022: the US Aggregate (mostly IG) fell **13.0%** as rates surged, while high yield, with its shorter duration and fatter income, fell a milder **11.2%** — and most of *that* loss was the recession scare, not the rate move.

The one-sentence intuition: when you buy IG you're mostly betting on rates; when you buy HY you're mostly betting on the economy.

### Fallen angels and rising stars: the BBB-/BB+ boundary in motion

The line between IG and HY is not just a label — it's a *flow*, and the flow across it is one of the most reliable sources of opportunity and pain in credit. A company downgraded from BBB- (lowest IG) to BB+ (highest HY) is a *fallen angel*. The moment it crosses, a large set of forced sellers appears: index funds tracking IG benchmarks, insurers with regulatory limits, and mandates that simply cannot hold junk. They must sell regardless of price, which often pushes the bond *below* where its fundamentals justify — and the high-yield buyers who scoop it up have, historically, earned strong returns precisely because the selling was forced rather than informed.

The reverse, a *rising star*, is a company upgraded from BB+ back into IG. Now a wave of forced *buyers* appears, and the spread tightens as the bond joins the IG-eligible universe. Watching the BBB-rated tier of the market matters enormously for this reason: BBB is the largest slice of IG, and in a downturn a cascade of BBB downgrades into junk can flood the high-yield market with supply, widening spreads mechanically even before defaults rise. The 2020 COVID shock briefly threatened exactly this "fallen-angel avalanche" until the Fed's backstop stopped it.

#### Worked example: buying a forced-sale fallen angel

A solid company's bond trades at \$98 (per \$100 face) while rated BBB-, yielding a 1.4% spread over Treasuries. It gets downgraded to BB+. Index-tracking IG funds must dump it, and with no immediate buyer, the price gaps to \$90 — its spread blows out to ~3.5% even though the company's actual default risk barely changed.

You buy at \$90. If the company simply survives and the bond drifts back toward \$98 over the next year while paying its ~5% coupon, your return is roughly the \$8 price recovery (≈ +8.9% on \$90) *plus* the ~5.5% coupon yield ≈ **+14%** — an equity-like return from a bond, earned by buying what others were forced to sell. The risk, of course, is that the downgrade was the first of many and the company keeps deteriorating; forced-sale bargains and falling knives can look identical at the moment of purchase.

The one-sentence intuition: the IG/HY boundary creates forced sellers whose selling overshoots, so the best credit bargains often appear at the exact moment a bond is kicked out of investment grade.

### Carry and roll-down: where the steady return comes from

When credit is *not* in crisis — which is most of the time — its return comes from two quiet engines: *carry* and *roll-down*. Carry is simply the income: the coupon and spread you earn just for holding the bond as time passes, with nothing happening. Roll-down is subtler: as a bond ages, it "rolls down" the yield curve toward shorter maturities, which usually carry lower yields, so its price drifts *up* even if rates and spreads don't move at all. Together, carry and roll-down are why a calm year in credit can hand you a 5-8% return with no drama — and why investors get lulled. The danger is that these returns are *steady and visible*, while the offsetting default risk is *lumpy and invisible until it arrives*. The carry feels like free money right up until the year it isn't.

## How credit actually behaves: the asymmetry

Now the part that separates people who understand credit from people who just own it. Credit returns are profoundly *asymmetric*, and that asymmetry is the source of both its appeal and its danger.

### The shape of a credit return

Think about the best and worst a single bond can do. The best case: the company pays every coupon and your principal back. You earn the yield — and not a penny more. There is no upside surprise in a bond; par is the ceiling. The worst case: the company defaults and you recover 40 cents. You can lose 60% of your money on one name.

So a credit portfolio collects small, steady "carry" (the income from coupons and spread) in the vast majority of months, punctuated by occasional sharp losses when defaults cluster. The return distribution is *negatively skewed*: lots of small wins, rare large losses. Statisticians call this fat-tailed-on-the-downside; traders call it **"picking up pennies in front of a steamroller."**

### The capped upside is the whole problem

It's worth dwelling on *why* the asymmetry is so unforgiving, because it's the deepest structural fact about credit. A stock's upside is unbounded — a great company can 10x. A bond's upside is *bounded by par plus the remaining coupons*. The most a healthy bond can do is pay you exactly what was promised. So when you buy credit, you are selling something like an insurance policy: you collect a small premium (the spread) in every calm period, and in exchange you've agreed to absorb a large loss in the rare disaster. That's the same payoff shape as a short option position, and it carries the same trap — the strategy *looks* brilliant for years (steady premium income, low volatility) and then surrenders much of the accumulated gain in a single event.

This is also why the gain from spread *tightening* is limited but the loss from spread *widening* is not. If spreads are at 300 bps, they can tighten maybe 100-150 bps in the best case (a modest price gain) but can widen 700, 1,000, even 1,700 bps in a crisis (a brutal price loss). The risk-reward is geometrically lopsided at tight spreads — small potential gain, enormous potential loss — and only becomes attractive after spreads have already blown out, when the asymmetry finally tilts in your favor.

### Spreads grind tight, then gap wide

Because of that asymmetry, spreads behave in a very particular rhythm over the cycle. In good times — growth steady, defaults low, money easy (see [risk-on, risk-off, how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates)) — investors reach for yield, bid up credit, and spreads *grind* slowly tighter, month after month, sometimes for years. Then a shock hits: a recession scare, an energy crash, a pandemic. Spreads don't drift back; they *gap* — they blow out violently in days or weeks. The grind up is slow; the fall is a cliff.

![High yield credit spread over the cycle with the 2008 and 2020 spikes highlighted](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-3.png)

The figure tells the whole story of the last twenty years of credit fear. Look at the shape: long stretches near or below the ~450 bps median (the dashed line), punctuated by three violent spikes — the 2008 financial crisis to ~2,000 bps, the 2016 energy/commodity stress to ~840 bps, and the 2020 COVID shock to ~1,100 bps. Each spike round-tripped within months. The spread spends most of its life being boring and a few weeks per decade being terrifying.

### Defaults cluster — they don't arrive smoothly

The reason spreads gap rather than drift is that defaults don't happen at a constant rate. They *cluster*. In a normal year, 1-2% of high-yield issuers default. In a recession, that figure can triple or quadruple in a single year, because the same macro shock that hurts one over-levered company hurts all of them at once. Credit risk is correlated risk.

![High yield default rate by year with the 2009 and 2020 recession spikes](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-7.png)

The default-rate chart makes the clustering vivid. The long-run average is ~3.5-4% (the dashed line), but the realized number swings from ~1% in benign years to a ~13% peak in 2009 and ~6.7% in 2020. This is why the "expected loss" calculation we did earlier is so treacherous: 4% is the *average*, but you never actually experience the average — you experience long stretches of 1-2% lulling you into complacency, then a single year of 13% that wrecks several years of carry.

#### Worked example: how one bad year eats years of carry

Suppose you hold a high-yield portfolio yielding 7% (spread + Treasury) and clipping that as income each year. In four calm years, defaults run at 1.5%; with 60% LGD, you lose 1.5% × 60% = 0.9% per year to defaults, netting ~6.1% a year. Over four years you've banked roughly 4 × 6.1% = **+24.4%**.

Then year five is a recession. Defaults spike to 13%; you lose 13% × 60% = 7.8% to defaults, *and* spreads blow out so your bonds' mark-to-market prices fall another ~15%. Your year-five total return is roughly +7% income − 7.8% default loss − 15% price drop ≈ **-15.8%**.

Four good years (+24.4%) minus one bad year (-15.8%) leaves you with about +8.6% over five years — a meager ~1.7% a year, and you white-knuckled the whole way. That's the steamroller catching up with the penny-picker.

The one-sentence intuition: credit's steady carry is real, but a single clustered-default year can erase several years of it, so the average return overstates how it actually feels to hold.

## Correlation: why credit fails you exactly when you need it

This is the most important section for an allocator, because it's where credit's promise breaks. The pitch for corporate bonds is seductive: "more income than Treasuries, and it's still a *bond*, so it'll cushion my stocks." The second half of that sentence is the dangerous lie.

### High yield correlates with stocks at ~0.75

*Correlation* measures how two assets move together, on a scale from +1 (move in lockstep) to -1 (move exactly opposite); 0 means no relationship. Treasuries, in the modern era, often had a *negative* correlation with stocks — they rallied when stocks fell, which is what makes them a great hedge ([see government bonds](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration)).

High yield does the opposite. Its correlation with stocks runs around **0.75** — almost as high as one stock index to another. The reason is mechanical: junk bonds and stocks are both claims on the *same risky companies*. When equity investors get scared about corporate earnings, credit investors get scared about corporate solvency, and both sell at once. IG sits in between — closer to bonds than HY, because its low default risk lets the Treasury component dominate — but even IG correlation with stocks rises sharply in a crisis.

![Scatter of high yield versus S&P 500 annual returns showing positive correlation](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-6.png)

The scatter plots eleven years of annual high-yield returns against S&P 500 returns. The cloud slopes clearly upward — when stocks were up, HY was up; when stocks fell, HY fell. The fitted line through these eleven points has a positive slope, and the full-period correlation between the two is about 0.75. Crucially, look at the bottom-left red zone: **the years where both fell are not random — they're the same years.** 2018, 2022 — when stocks hurt, high yield hurt with them.

### Why diversification evaporates in a crash

The cruelest property of correlation is that it isn't constant. In calm markets, high yield's correlation with stocks might be a moderate 0.5-0.6 — enough to feel like you're diversified. In a crash, correlations *converge toward 1*: everything risky falls together as investors flee to cash and Treasuries. So the diversification you thought you owned was there in the good times and vanished in the one moment you needed it.

This is the structural reason credit "fails as a diversifier in a crash." A true diversifier is something that *zigs when your stocks zag*. High yield zags when your stocks zag. Owning stocks and high yield together isn't a diversified portfolio; it's a concentrated bet on corporate health wearing two different labels.

Even investment grade, the "safe" end of credit, betrays you partially here. In calm markets IG's stock correlation is low — the Treasury component dominates and it behaves like a bond. But in a genuine crisis, IG spreads widen too (they hit ~370 bps in March 2020 and ~600 bps in 2008), so the credit component of IG suddenly reasserts itself and IG falls alongside risk assets at the worst moment. IG won't fall as hard as high yield, but it won't give you the clean negative correlation that a Treasury does. The further down the credit ladder you go, the more your "bond" turns into a stock under stress — and the stress is exactly when that conversion hurts.

#### Worked example: the 2008 diversification test

Imagine two investors entering 2008, both wanting to soften an equity portfolio.

- **Investor A** pairs stocks with **Treasuries**. In 2008, the S&P 500 fell ~37%, but long Treasuries *gained* ~25.9% and the Aggregate gained ~5.2%. The bonds did their job — they rose as stocks fell. A 60/40 stock/Treasury mix lost far less than stocks alone.
- **Investor B** pairs stocks with **high yield**, reasoning that "it's still a bond and it pays more." In 2008, high yield fell ~**26.2%** — nearly as bad as the stock market. Investor B's "diversifier" cratered alongside the thing it was supposed to diversify.

Same crisis, opposite outcomes. Investor A owned a hedge; Investor B owned a second helping of the same risk. The extra ~3% of yield high yield paid in the good years was no consolation for the ~31% gap in 2008.

The one-sentence intuition: high yield gives you income that *looks* like a bond's, but in a crash it behaves like a stock — so it can't be the thing that saves a stock portfolio.

## Common misconceptions

**"Bonds are safe, so corporate bonds are safe."** Treasuries are safe in dollar terms; corporate bonds are not. High yield lost ~26.2% in 2008 and ~11.2% in 2022 — those are equity-sized losses. The word "bond" describes the *structure* (fixed coupons + principal), not the *risk*. A junk bond is closer in risk to a stock than to a Treasury.

**"A higher yield means a better deal."** A higher yield means a higher *risk*, and often a worse risk-adjusted deal. When a bond yields 12%, the market is not being generous; it's pricing a high probability of default. The relevant question is never "what's the yield?" but "what's the yield *after subtracting expected default losses*?" — which, as our decomposition showed, can be a thin 200 bps even on a fat 450 bps spread.

**"Credit diversifies my stock portfolio because it's a different asset class."** It's a different label, not a different risk. High yield correlates ~0.75 with stocks, and that correlation rises toward 1 in exactly the crashes where you wanted protection. If you want to diversify equity risk, the tool is Treasuries or cash, not credit.

**"Defaults are rare, so the spread is mostly free money."** Defaults are rare *on average* but they *cluster*. You collect the spread smoothly for years, then surrender several years of it in one recession when defaults spike to double digits and prices gap down. The average default rate (4%) is a number you never actually live through — you live through 1.5% then 13%.

**"Tight spreads mean the market is calm and safe."** Tight spreads mean credit is *expensive* and your cushion is thin. The single worst time to *buy* high yield is when spreads are tightest and the cycle is late, because you're being paid the least for risk that's quietly building. The asymmetry is brutal: at 300 bps you have almost nothing to gain and a long way to fall.

**"I'll just hold credit to maturity and ignore the price swings."** This works only if every issuer survives. Buy-and-hold protects you from *mark-to-market* swings (the daily price noise) but not from *defaults*, which are permanent losses of capital, not temporary dips. A Treasury held to maturity is genuinely safe in dollar terms; a portfolio of junk bonds held to maturity will still lose the ~4% a year that defaults claim on average — and far more in a recession. "Hold to maturity" is a comfort for rate risk, not for credit risk.

## How it shows up in real markets

**The 2008 financial crisis — credit and equity fall as one.** As Lehman collapsed, high-yield spreads tore from ~600 bps to ~2,000 bps and IG spreads hit ~600 bps. High yield returned ~-26.2% for the year while Treasuries gained. The market priced a ~13% default rate — a depression-level wave. The 2009 realized default rate did peak near 13%, validating *some* of the fear, but spreads of 2,000 bps over-discounted the damage for survivors. Investors who bought the index in late 2008 earned among the best multi-year credit returns in history. The lesson: panic prices in catastrophe, and catastrophe rarely fully arrives for the survivors.

**The 2015-16 energy crash — a sector-specific blowout.** Oil collapsed from over \$100 to the \$20s, and a huge slug of the US high-yield market was energy companies (shale drillers had funded the boom with junk bonds). Spreads gapped to ~840 bps even though the broad economy was fine. High yield returned ~-4.5% in 2015, then rebounded ~+17.1% in 2016 as oil stabilized. The lesson: high-yield stress can be a sector story, not a macro one — and those are often the better buying opportunities, because the contagion is contained.

**March 2020 COVID — the fastest blowout and the fastest reversal.** Spreads exploded to ~1,100 bps in three weeks as the economy shut down. Then the Fed announced it would buy corporate bonds and ETFs directly — an unprecedented backstop, an extension of the [quantitative-easing toolkit](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders). Spreads collapsed almost immediately. High yield finished 2020 at ~+7.1% — a *positive* year despite the March crater. The 2020 default rate rose to ~6.7%, well below the priced fear. The lesson: a credible liquidity backstop can short-circuit a credit panic before defaults validate it, which is why central-bank intervention is now part of every credit trader's risk model.

**2021 — the grind to complacency.** With rates pinned near zero and the recovery roaring, investors reached for yield and high-yield spreads ground down to ~300 bps, near the tightest in the data. Credit returned a modest ~+5.3% — almost pure carry, almost no spread tightening left to harvest. This is the textbook "late, tight, expensive" setup: you're being paid the least for the most cyclical risk. The lesson: tight spreads are not safety; they're a small cushion and a warning.

**2022 — the rate shock that spared no bond.** Inflation forced the Fed to hike aggressively. This was a *duration* event, not primarily a *credit* event: the IG-heavy Aggregate fell ~13.0%, even more than high yield's ~11.2%, because IG's long duration made it more exposed to the rate move. High yield's shorter duration and fatter carry actually *cushioned* it relative to IG. The lesson: not all bond losses are credit losses — sometimes the long, "safe" IG bond is the riskier one, and the junk bond's short duration is a feature.

**2023-24 — carry comes back.** With recession fears fading and spreads moderate, high yield delivered ~+13.4% in 2023 and ~+8.2% in 2024 — the income engine working as designed in a benign macro. By late 2024 spreads had ground back to ~300 bps, setting up the same late-cycle question all over again. The lesson: credit's best years are the boring recoveries, when defaults stay low and you simply collect the coupon.

![High yield versus investment grade annual total returns 2014 to 2024](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-4.png)

The grouped-bar chart pulls a decade of this together: high yield (amber) and the IG-heavy Aggregate (blue), side by side, 2014-2024. Three things jump out. First, in most years high yield out-earns IG — that's the carry. Second, the divergences tell the regime: 2016 was a high-yield rebound year (energy recovering) while IG was flat; 2022 was a rate shock that hit IG (-13.0%) even harder than HY (-11.2%). Third, the bad years for both are the *same* years — there's no year where one zigs while the other zags, which is the correlation problem in one picture.

## The allocation playbook: when to own credit

Everything above lands here. Credit is not a buy-and-hold-forever asset like broad equities; it's a *cyclical* asset whose attractiveness swings enormously with the spread and the cycle. The job is to be paid well for the risk, which means buying when the spread is fat and the worst is passing, and shrinking when the spread is thin and the cycle is late.

### The two variables that decide everything

You only need to track two things: **the spread level** (are you being paid a lot or a little?) and **where you are in the default cycle** (is recession risk rising or fading?). The matrix below crosses them into four cells, and your action in each is different.

![Credit allocation playbook matrix of spread level versus cycle stage](/imgs/blogs/corporate-credit-investment-grade-high-yield-spreads-8.png)

- **Wide spread + recession risk fading (top-left): the fat pitch.** This is the rare, valuable setup — spreads have already blown out to price in disaster, but the default wave is cresting and the economy is stabilizing. Late 2008, mid-2020. You're paid a fortune and the worst is behind. This is when to *add* high yield aggressively, because the catastrophe is priced but won't fully arrive for survivors.
- **Wide spread + recession risk rising (top-right): the value trap.** Spreads are wide, but for a *reason* — defaults are still climbing. Buying here means catching a falling knife. If you must own credit, favor BB (the safest junk) over CCC (the distressed end), and size small.
- **Tight spread + recession risk fading (bottom-left): carry, gently.** Spreads are thin so there's little cushion, but the macro is benign, so you clip the coupon and accept that most of the return is income, not capital gain. Stay up-in-quality (BB and IG).
- **Tight spread + recession risk rising (bottom-right): avoid or trim.** The worst cell. You're paid the least for risk that's quietly building — pennies in front of the steamroller. This is when to *reduce* high yield and rotate into IG, Treasuries, or cash.

### IG vs HY by regime

The IG-or-HY choice maps onto the same cycle:

| Regime | Favor | Why |
|---|---|---|
| Early cycle, spreads wide, defaults cresting | High yield | Maximum spread, defaults peaking, equity-like upside as spreads tighten |
| Mid cycle, growth steady, spreads moderate | High yield (BB) + some IG | Carry is good, defaults low, still room to tighten |
| Late cycle, spreads tight, rates the main risk | IG (short-to-intermediate) | Up-in-quality; but watch IG's duration if rates can rise |
| Recession feared, spreads about to gap | Treasuries / cash | Credit correlation to stocks spikes; you want the real hedge |

### Sizing and what invalidates the case

Because credit's downside is equity-like, **size it like a partial equity position, not like a bond**. A common allocator framing: treat high yield as roughly "half a stock and half a bond" for risk-budgeting, so a 10% HY sleeve carries something like the risk of a 5-6% additional equity allocation. Don't let a "bond" label fool you into oversizing.

The case for owning credit **invalidates** when: spreads grind to historic tights (you're not paid for the risk); the cycle turns late and leading indicators of recession rise (defaults are coming); or the stock-bond correlation regime means you specifically need a *hedge*, which credit cannot provide. In any of those, the move is up-in-quality — toward IG, Treasuries, and cash — even though it means giving up income. As the [map of asset classes](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) frames it, every asset earns its place by what it does *with* the others; credit earns its place as an income engine in benign-to-recovering regimes, and loses it as a diversifier in crashes.

### Quality up, quality down: the dial within credit

Even once you've decided to own credit, there's a second dial: *where on the quality ladder*. This is the credit allocator's most-used lever, because it lets you stay invested while changing your risk. "Up-in-quality" means rotating from CCC toward BB toward IG — giving up spread and income in exchange for far lower default risk. "Down-in-quality" (reaching for yield) means the reverse. The rule of thumb: move *down* in quality early in the cycle when spreads are wide and defaults are cresting (the riskiest bonds rebound most), and move *up* in quality late in the cycle when spreads are tight and the cushion is thin (you want to be in the bonds that survive). The difference is enormous: in a recovery, CCC bonds can return 30-40% as their spreads collapse from distressed levels; in a downturn, those same CCC bonds default at 10-25% a year and can lose half their value. The quality dial is how you express a cycle view without going fully to cash.

### How you actually own credit — and what it costs

In practice, almost nobody buys individual corporate bonds; the market is illiquid and lot sizes are large. Most investors access credit through funds and ETFs: an IG corporate fund, a high-yield fund, or a total-bond fund that blends Treasuries and IG. This is convenient but it adds a wrinkle worth understanding. A high-yield ETF holds illiquid bonds but trades like a liquid stock, so in a panic — exactly when the underlying bonds gap down and stop trading — the ETF can swing violently and trade at a discount to the value of its holdings. The "liquidity transformation" works smoothly until it doesn't. There's also a cost layer: high-yield funds charge higher fees (often 0.4-0.5% a year versus near-zero for a Treasury index) because credit analysis is labor-intensive, and that fee comes straight out of your thin risk premium. When your true compensation for risk is ~210 bps, a 50 bps fee is taking nearly a quarter of it before you've absorbed a single default.

#### Worked example: the fee bite on a thin spread

You buy a high-yield fund when the spread is 450 bps. We computed earlier that ~240 bps of that is expected default loss, leaving ~210 bps of real risk premium. Now subtract a 0.45% (45 bps) fund fee:

- Gross risk premium: 210 bps.
- After fee: 210 − 45 = **165 bps**.

The fee just took ~21% of your actual pay for risk. Compare a calm-market 300 bps spread, where the risk premium might be only ~120 bps after expected losses: the same 45 bps fee now eats nearly 38% of your compensation. This is why credit is least attractive precisely when spreads are tight — there's so little premium left that fixed costs like fees and bid-ask spreads consume a punishing share of it.

The one-sentence intuition: because credit's true risk premium is thin, the cost of access — fees and trading frictions — matters far more than it does for cheaper, more liquid assets, and it bites hardest exactly when spreads are tight.

#### Worked example: sizing high yield as part-equity

You run a simple portfolio and you're tempted to put 20% into high yield because it yields 7% versus 4% on Treasuries. Before you do, risk-budget it honestly.

High yield's stock correlation is ~0.75 and its crisis drawdowns (~-26% in 2008) are roughly two-thirds the size of equity's (~-37%). So a 20% HY sleeve contributes something close to the *equity-like* risk of a ~12-13% additional stock position, plus only a thin diversification benefit. If your target was a balanced book and you already hold, say, 50% stocks, that 20% HY quietly pushes your effective equity-risk exposure toward ~62-63% — far more aggressive than the labels suggest.

The fix is not necessarily to avoid high yield, but to *fund it from your equity sleeve, not your bond sleeve*: hold (for example) 45% stocks + 10% high yield + 45% Treasuries rather than 50% stocks + 20% high yield + 30% Treasuries. Same income ambition, but you've kept a real Treasury hedge intact for the crash.

The one-sentence intuition: count high yield against your *stock* budget, because in the moment that matters it behaves like stock — and keep enough true Treasuries to actually hedge.

### What to watch — the credit dashboard

You don't need a Bloomberg terminal to track credit's regime; a handful of free signals do most of the work. Watch the *level* of the high-yield OAS against its own history — below ~350 bps is rich and late-cycle, above ~700 bps is stress, above ~1,000 bps is panic and often opportunity. Watch the *direction* — spreads grinding tighter for months is the benign carry regime; spreads gapping wider over days is the warning to reduce. Watch the *default rate trend* in the rating-agency reports — rising defaults validate widening spreads, falling defaults can mean a blowout is overdone. And watch *whether IG and HY are moving together with equities* — when all three sell off in lockstep, the diversification you thought you had is gone and the only true hedges left are Treasuries and cash. None of these is a precise timing tool, but together they tell you which of the four playbook cells you're standing in, which is most of the decision.

## When this matters to you

If you own a "bond fund" in a retirement account, check what's in it. A total-bond or Aggregate fund is mostly IG and Treasuries — a genuine cushion for your stocks, but rate-sensitive (that's why it fell ~13% in 2022). A "high-yield" or "income" fund is something else entirely: an equity-like risk wearing a bond label, paying you more in calm years and falling ~26% alongside your stocks in a crash. Neither is wrong to own, but they play opposite roles, and confusing them is how people discover in a crisis that their "safe" bonds weren't.

The deeper lesson is the one the whole cross-asset playbook keeps returning to: an asset is defined not by its name but by *what drives it and how it behaves alongside everything else you own*. Corporate credit is the cleanest example — the same "bond" structure gives you a Treasury-like instrument (IG) or a stock-like one (HY) depending entirely on where it sits on the credit ladder, and the spread is the live readout of how the market is pricing that risk right now.

If you want to keep pulling the threads from here: the spread sits on top of the [risk-free curve and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) that determines IG's rate risk; high yield's equity-like behavior connects directly to [owning a slice of growth through equities](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth); and the way credit spreads gap and reverse with the flight to safety is the credit-market face of [risk-on, risk-off rotation](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) and the [liquidity tides](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) that decide whether a panic becomes a depression or a buying opportunity.

*This is educational, not financial advice. Spreads, default rates, and recovery figures cited are illustrative levels and long-run averages as of 2024-2025 and will be different by the time you read this — always check current data before drawing conclusions.*

## Further reading & cross-links

- [Government Bonds: The Risk-Free Anchor and Duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — the Treasury yield that sits underneath every corporate bond, and the duration risk that drives IG.
- [The Map of Asset Classes: What You Can Own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — where credit fits in the full menu, and why each asset earns its place by what it does with the others.
- [Equities: Stocks, Owning a Slice of Growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth) — the asset high yield secretly resembles, and why the ~0.75 correlation exists.
- [Risk-On, Risk-Off: How Money Rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the rotation that drives spreads tight in booms and wide in panics.
- [What Liquidity Means: Market, Funding, and the Global Traders](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) — why a liquidity backstop (the Fed in 2020) can reverse a credit panic before defaults validate it.

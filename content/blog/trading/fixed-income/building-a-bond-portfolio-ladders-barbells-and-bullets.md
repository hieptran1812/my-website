---
title: "Building a bond portfolio: ladders, barbells, and bullets"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the three classic ways to assemble a bond portfolio — the ladder, the bullet, and the barbell — what each one bets on, the reinvestment-versus-price-risk trade-off behind them, and how a $100,000 portfolio behaves when rates rise and the curve steepens."
tags: ["fixed-income", "bonds", "bond-ladder", "barbell", "bullet", "portfolio-construction", "duration", "reinvestment-risk", "convexity", "yield-curve"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — there are three classic shapes for a bond portfolio, and choosing between them is really a choice about *which risk you want to carry* — the danger that rates fall and you have to reinvest your cash at a worse yield, or the danger that rates rise and your bonds lose price.
> - A **ladder** spreads equal amounts across every maturity (say 1 through 10 years) and rolls each maturing bond out to the long end; it makes no bet on rates, throws off steady income, and hands you liquidity every single year.
> - A **bullet** concentrates the whole portfolio near one maturity; you use it to match a single future bill (a tuition payment, a balloon mortgage) or to express one clean view about one point on the curve.
> - A **barbell** holds short bonds and long bonds and skips the middle; for the same overall *duration* it carries more **convexity**, so it loses a little less when rates rise and gains a little more when they fall — and it carries a built-in view on the *shape* of the curve.
> - The deepest trade-off underneath all three is **reinvestment risk versus price risk**: short bonds protect your price but force you to keep reinvesting at unknown future rates; long bonds lock in today's yield but swing hard in price.
> - The number that ties it all together is **portfolio duration** — the weighted-average rate sensitivity of everything you own — and for a *parallel* move in rates, duration, not shape, is what sets your gain or loss.

Here is a question that sounds like personal-finance trivia and turns out to be the central craft of fixed income. You have \$100,000 to put into bonds. You could buy one ten-year bond and be done. You could buy ten bonds, one maturing each year for the next decade. You could buy a pile of two-year bonds and a pile of thirty-year bonds and nothing in between. Each of those is a real, defensible portfolio holding the *same* \$100,000. They will pay you different amounts of income, hand you cash back on wildly different schedules, and react very differently the next time the Federal Reserve moves or the yield curve changes shape. So which one is "right"?

The honest answer is that none of them is right in the abstract — each is right *for a particular goal and a particular view*, and the entire job of building a bond portfolio is matching the shape of what you own to the shape of what you need. A retiree who wants predictable spending money wants something different from a parent saving for a tuition bill due in exactly five years, who wants something different again from an investor who thinks long rates are about to fall and wants to profit from it. The three structures in this post — the **ladder**, the **bullet**, and the **barbell** — are the three canonical answers, and once you understand what each one bets on, you can read almost any real bond portfolio as some blend of them.

![A grid comparing how a ladder a bullet and a barbell spread one hundred thousand dollars across short middle and long maturities with the ladder even the bullet concentrated and the barbell at the two ends](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-1.png)

The diagram above is the mental model for the whole post: take the maturity line — short bonds on the left, long bonds on the right — and ask *where do my dollars sit?* A ladder spreads them evenly. A bullet piles them at one point. A barbell loads the two ends and leaves the middle empty. Three shapes, same money, three different bets. (Everything here is educational, not investment advice — the goal is to understand the mechanism, not to tell you what to buy.)

This is the opening post of the portfolio track, and it leans on everything the series has built so far. We will use [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) constantly as the measure of rate sensitivity, lean on [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story) to explain why the barbell has an edge, draw on [reinvestment risk versus price risk](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield) as the trade-off underneath all three shapes, and use the [yield curve](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) as the thing each structure quietly takes a position on. If any of those words are unfamiliar, the Foundations section rebuilds the pieces you need.

## Foundations: maturity, the curve, and the two risks

Let's build every term from zero, because the three structures are just three ways to arrange a handful of simple ideas.

A **bond** is a tradable loan. You hand over money today (the **price**), and in return you receive a fixed stream of interest payments (the **coupons**) plus your original loan amount back at the end (the **par value** or **face value**, conventionally \$1,000 per bond). The date you get your par back is the **maturity** — a two-year bond returns your principal in two years, a thirty-year bond in thirty. If the word "bond" itself is new, the [very first post in this series](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction) builds it from scratch.

The **yield** of a bond is the single annualized return you earn if you buy it at today's price and hold it to maturity, collecting and reinvesting every coupon. It is the bond's "interest rate" as the market sees it right now. When people say "the ten-year yield is 4%," they mean a ten-year Treasury bought today pays the equivalent of 4% a year. Crucially, **price and yield move in opposite directions**: when rates in the economy rise, the fixed coupons on bonds you already own look stingy, so their prices fall to make their yields competitive again. That seesaw — covered in depth in [price and yield, the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) — is the engine behind everything in this post.

The **yield curve** is the picture you get when you plot yield against maturity — short bonds on the left, long bonds on the right. Usually it slopes up: a thirty-year bond pays more than a two-year bond, because lenders demand extra compensation for tying their money up longer (the **term premium**). The curve can *steepen* (long yields rise relative to short yields, or the gap widens), *flatten* (the gap shrinks), or even *invert* (short yields exceed long ones). Where your bonds sit on the curve determines how a change in its *shape* — not just its overall level — hits you. This is the whole reason a barbell is different from a bullet.

Now the two risks that drive every structural choice.

**Price risk** (also called interest-rate risk) is the danger that rates rise and your bonds lose market value before you can sell them. The longer the maturity, the bigger the price swing for a given rate move — a thirty-year bond can drop 20% on a 1% rate rise, while a two-year bond barely flinches. We measure this sensitivity with **duration**: a bond's duration is, roughly, the percentage its price falls for each 1% rise in yield. A bond with a duration of 7 falls about 7% when yields rise 1%. (Duration is also, more precisely, the weighted-average time to receive the bond's cash flows — its "center of gravity" in time — which is why it ties so neatly to maturity. The [duration post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) develops both meanings.)

**Reinvestment risk** is the mirror image: the danger that rates *fall* and, when your bonds mature or pay coupons, you are forced to put that cash back to work at a *lower* yield than you were earning. A retiree living on bond income who watches a 5% bond mature into a world of 2% rates feels reinvestment risk acutely. The shorter your bonds — the sooner and more often they mature — the more reinvestment risk you carry.

Here is the keystone: **these two risks pull in opposite directions, and you cannot escape both at once.** Buy long bonds and you crush reinvestment risk (you've locked in today's yield for decades) but take on huge price risk. Buy short bonds and you crush price risk but take on huge reinvestment risk (you'll be rolling over constantly at unknown rates). The three structures are three different *settlements* of this tug-of-war — and that is the real meaning of choosing one. The post on [the two faces of yield](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield) is the deep treatment; here we put it to work.

One last building block: **portfolio duration**. A portfolio's duration is just the dollar-weighted average of its bonds' durations. If you hold \$50,000 of a 2-year bond (duration ~2) and \$50,000 of a 30-year bond (duration ~20), your portfolio duration is roughly the weighted average, `(0.5 × 2) + (0.5 × 20) = 11`. Portfolio duration is the number that tells you how the *whole* book reacts to a parallel shift in rates — and, as we'll see, it's the number that makes a fair comparison between a bullet and a barbell possible at all. This concept ties directly to [duration as the risk-free anchor](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) on the cross-asset side.

### Why "average maturity" lies and "average duration" doesn't

It is tempting — and wrong — to summarize a portfolio by its average *maturity*. The trap is that a bond's price sensitivity does not grow in a straight line with maturity for a given dollar amount; the long end is wildly more sensitive than the short. A 30-year bond isn't "fifteen times riskier" than a 2-year by some gentle measure — its duration (~20) versus the 2-year's (~1.9) means it moves about *ten times as much* in price for the same rate change. So when you mix a 2-year and a 30-year fifty-fifty, the "average maturity" of 16 years badly understates how the portfolio behaves, while the *duration-weighted* average of ~11 tells the truth. Throughout this post, whenever we compare two structures, we will silently insist that they share a duration first — because only then is the comparison about *shape* rather than about who took the bigger rate bet. Keep that rule in your pocket; it is the single most useful habit in fixed-income construction.

Here is a compact table of the bond durations we'll use throughout, so the arithmetic in the worked examples is reproducible. These are round, illustrative figures for plain coupon bonds at roughly par; real durations shift with coupon and yield.

| Maturity | Approx. duration | Price change for +1% yield |
|---|---|---|
| 1-year | ~1.0 | ~ −1% |
| 2-year | ~1.9 | ~ −1.9% |
| 5-year | ~4.6 | ~ −4.6% |
| 10-year | ~8.5 | ~ −8.5% |
| 30-year | ~20 | ~ −20% (cushioned by convexity) |

With those pieces in hand, let's build the three portfolios.

## The ladder: rate-agnostic income and yearly liquidity

A **bond ladder** is the most intuitive structure and, not coincidentally, the most popular with ordinary investors. You divide your money into equal chunks and buy bonds maturing at evenly spaced intervals — the classic version is one bond maturing each year for ten years. Each maturity is a "rung." When the nearest rung matures and hands you your principal back, you reinvest that cash into a *new* bond at the far end of the ladder, keeping the ladder the same length forever.

The genius of the ladder is what it refuses to do: **it makes no bet on the direction of rates.** Because you always hold bonds across the whole maturity spectrum, you are never caught all-short (drowning in reinvestment risk) or all-long (exposed to price risk). Some of your money is locked in at long yields; some of it comes due soon and gets reinvested at whatever the new rate is. If rates rise, your maturing rungs roll into higher yields — you benefit. If rates fall, your existing long rungs keep paying the old, higher yields — you're protected. You give up the chance to win big from a correct rate forecast in exchange for never losing big from a wrong one.

![A pipeline showing the front rung of a ladder maturing returning ten thousand dollars of principal and being reinvested as a fresh ten-year rung so the ladder renews itself each year](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-2.png)

The figure traces the "roll." Every year the shortest bond matures, your \$10,000 of principal comes back as cash, you buy a fresh 10-year bond with it, and the ladder is whole again — now with a brand-new 10-year rung and what used to be the 2-year as the new front rung. The ladder is a self-renewing machine: set it up once, and it quietly maintains its own shape.

#### Worked example: building a \$100,000 ten-year ladder

You have \$100,000 and you build a 1-through-10-year ladder, \$10,000 per rung. Suppose the yield curve is gently upward-sloping, and the bonds you buy yield, by maturity: 1-year at 3.5%, 2-year 3.7%, 3-year 3.9%, on up to the 10-year at 4.5%. Your average coupon income across the ten rungs lands around 4.0%, so the ladder throws off roughly `\$100,000 × 4.0% = \$4,000` a year in coupons.

Now run a year forward. The 1-year rung matures and returns its \$10,000. Say rates have risen across the board by 1% in the meantime, so a fresh 10-year now yields 5.5%. You reinvest the \$10,000 into that new 10-year rung at 5.5%. Your ladder's average yield ticks *up*, because you swapped a maturing low-coupon bond for a new high-coupon one. Had rates instead *fallen* 1%, the new 10-year would yield only 3.5% — your reinvested rung earns less, but the other nine rungs are still paying their old, higher coupons, so the blended income barely moves.

*The intuition: a ladder never has to guess where rates are going, because every year it reinvests a little at the new rate while the rest of the book stays locked at old rates — the two effects cushion each other.*

The ladder's second gift is **liquidity on a schedule.** Because a rung matures every year, you have a predictable stream of cash arriving without ever having to *sell* a bond into the market (where you'd pay a bid-ask spread and might be selling at a loss). If your life changes — you need the money, you want to redeploy elsewhere — you simply stop reinvesting the next maturing rung and take the cash. That makes a ladder unusually forgiving of an uncertain future.

![A timeline of a bond ladder paying about four thousand dollars of coupons every year plus ten thousand dollars of returning principal from one maturing rung that gets reinvested at the long end](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-5.png)

The income picture above is what makes the ladder beloved by retirees and endowments: a flat, dependable stream. Every year you collect coupons from all ten rungs *and* receive \$10,000 of principal from the rung that matured. That \$10,000 is the part you decide what to do with — reinvest it to keep the ladder running, or spend it. The structure turns a lump sum into a paycheck.

There's a quieter mechanism inside a ladder worth naming, because it's the source of much of its long-run return: **roll-down**. When the curve slopes upward, a bond's yield falls as it ages toward maturity — a bond bought as a 5-year becomes a 4-year a year later, and if the curve hasn't moved, a 4-year yields less than a 5-year, which means its *price has risen*. So even with no change in rates, each rung in your ladder slowly "rolls down" the curve and gains a little price as it ages. A ladder built across an upward-sloping curve harvests this roll-down on every rung continuously. It's a subtle tailwind, and it's part of why a ladder's realized return often beats its naive starting yield in a stable-rate world. (The flip side: if the curve is *inverted*, roll-down works against you, and the short rungs you keep reinvesting may carry the highest yields — one of several ways an inverted curve scrambles the usual intuitions.)

What does the ladder cost you? Two things. First, **you forgo the upside of a correct bet.** If you were certain rates would fall, an all-long portfolio would have crushed the ladder — but certainty about rates is exactly what almost nobody has. Second, **a ladder's blended yield is usually a touch below what you'd earn reaching all the way out the curve**, because part of your money sits in low-yielding short rungs. You're paying a small yield give-up for the insurance of never being wrong about direction. For most people, that's a bargain.

How long should the ladder be? That's the one real design choice, and it's a direct expression of the price-risk-versus-reinvestment-risk trade-off. A short ladder (1–5 years) keeps duration low — gentle price swings, but you're reinvesting a fifth of the book every year, so your income tracks rates closely (good when rates rise, painful when they fall). A long ladder (1–20 years) reaches further out the curve for more yield and roll-down, locks in today's rates for longer, but carries a higher duration, so a rate spike marks it down more. The 1–10 ladder in our running example is the popular middle: a duration around 5, an income stream that's reasonably stable, and a rung maturing every year. Lengthening or shortening the ladder is how you dial duration up or down *without* abandoning the rate-agnostic, liquidity-on-a-schedule design — the same dial a manager turns when they extend or shorten the whole book.

## The bullet: one point, one purpose

A **bullet** portfolio concentrates your money near a single maturity. Instead of spreading \$100,000 across ten years, you put most or all of it into bonds maturing around, say, year five. The name comes from the shape: plotted across the maturity line, your holdings look like a single spike — a bullet — rather than a spread.

Why would you deliberately give up the ladder's diversification across maturities? Because a bullet is the right tool when you have **one specific job to do at one specific time.** The textbook case is a known future liability: you owe \$130,000 for a child's college in five years, or a balloon payment on a property, or a planned large purchase. You don't want a stream of cash dribbling in over a decade; you want a known sum available on a known date. So you buy bonds that mature exactly when you need the money. This is the personal-finance version of what pensions and insurers do at scale — the discipline of [matching assets to liabilities](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge).

The second reason to bullet is to **express a view about one point on the yield curve.** A professional who believes the five-year part of the curve is unusually cheap — that its yield is too high relative to neighbors and will fall (lifting its price) — can concentrate there to profit if they're right. A bullet is a focused bet; it has no hedge against being wrong about that point.

#### Worked example: bulleting a known liability

You owe \$130,000 in exactly five years for tuition. Today, 5-year bonds yield 4.0%. How much do you need to invest now, and how do you structure it?

To grow into \$130,000 in five years at 4.0%, you need roughly `\$130,000 ÷ (1.04)^5 = \$130,000 ÷ 1.2167 = \$106,850` today. You buy \$106,850 of 5-year bonds — coupons plus the final principal are sized so that, reinvesting the coupons along the way at roughly 4%, you arrive at about \$130,000 right when the bill is due. (In practice you'd often use a 5-year zero-coupon bond or a STRIPS to make the match exact, since a zero pays everything at the end with no coupons to reinvest — eliminating reinvestment risk entirely for that horizon.)

Now suppose rates jump to 6% the day after you buy. Your 5-year bonds drop in price — but you don't care, because you're not selling them; you're holding them to maturity, exactly when the bill comes due. The price wobble in between is noise. Your job was to have \$130,000 on a date, and a maturity-matched bullet delivers that with the least fuss.

*The intuition: when you have one bill due on one date, the cleanest portfolio is one that matures on that date — a bullet turns "I need a specific sum at a specific time" into a bond you can just hold and forget.*

The danger of the bullet is the flip side of its focus. **All your reinvestment risk lands at once.** When the bullet matures, your entire principal comes back as a single lump, and you must redeploy it all into whatever the rate environment happens to be on that day. A ladder spreads that decision across ten years; a bullet bunches it into one. If you bullet a 5-year and rates have collapsed by year five, your whole portfolio reinvests at the new low rate simultaneously. And a bullet gives you **no liquidity until maturity** — nothing matures in between, so if you need cash early you must sell into the market and eat whatever price the market offers.

There is one subtle structural feature worth flagging. A bullet, by sitting all at one maturity, has *less convexity* than a barbell of the same duration. That sounds like jargon, but it's the entire reason the next structure exists — so let's build to it.

## The barbell: convexity and a view on the curve's shape

A **barbell** holds short bonds and long bonds and deliberately skips the middle. The classic version pairs something very short (2-year) with something very long (30-year) — two heavy weights at the ends of a bar, nothing in between. At first glance this looks strange: why own the extremes and avoid the middle, where the bullet sits?

The barbell exists for two reasons, and both are subtle enough that beginners usually miss them.

**Reason one: convexity.** Recall that duration measures a bond's price sensitivity to rates as a *straight line* — "down 5% for a 1% rise." But the true price-yield relationship is *curved*: bonds gain a little more when rates fall than they lose when rates rise by the same amount. That curvature is **convexity**, and it is a genuinely good thing to own — it means rate moves help you slightly more than they hurt you. Crucially, **a barbell has more convexity than a bullet of the same duration.** The long leg's huge convexity, blended with the short leg, beats the modest convexity of a single mid-maturity bond. So if you build a barbell and a bullet to the *same* duration — meaning they react identically to a *small* parallel rate move — the barbell will outperform on any *large* move in either direction.

![An XY chart with portfolio value on the vertical axis and change in yield on the horizontal axis showing the barbell curve bowing above the straight bullet line so the barbell loses less when rates rise and gains more when they fall](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-6.png)

The figure makes the convexity edge visual. Both portfolios have the same value and the same slope (duration) at today's yield — that's the point they're tuned to match. But the barbell's value follows a *bowed* curve while the bullet's follows a nearly straight line. Move right (rates up) and the barbell's curve sits above the bullet's line: it falls less. Move left (rates down) and the barbell's curve again sits above: it rises more. The gap between curve and line is the convexity advantage, and it widens the bigger the rate move.

**Reason two: a view on the shape of the curve.** A barbell isn't just a convexity play; it's also a bet on how the curve will *change shape*. Because the barbell's value is dominated by its long leg (a 30-year bond is enormously more rate-sensitive than a 2-year), the barbell does well when **long rates fall relative to short rates** — a *flattening* of the curve at the long end. It does poorly when long rates rise relative to short — a *steepening*. A bullet, sitting in the middle, is far more neutral to these shape changes. So choosing a barbell over a bullet is implicitly choosing to bet that the long end will outperform. This is the bridge to [curve trading — steepeners, flatteners, and butterflies](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies), where a barbell-versus-bullet trade is literally called a "butterfly."

#### Worked example: barbell versus bullet, same duration

You have \$100,000. You build two portfolios, both tuned to a duration of about 5.

- **The bullet:** \$100,000 in a 5-year bond. A 5-year bond has a duration of roughly 4.6 — close enough; call the duration ~5 after rounding for coupons.
- **The barbell:** to hit a 5-year duration with a 2-year (duration ~1.9) and a 30-year (duration ~20), you solve for the weight `w` in the long bond such that `w × 20 + (1 − w) × 1.9 = 5`. That gives `w ≈ 0.17` — about \$17,000 in the 30-year and \$83,000 in the 2-year. (Real desks would use slightly different weights to match duration precisely; the running \$50k/\$50k "2y/30y barbell" in our headline example is a rounder, more aggressive version with a *longer* duration — we'll use the duration-matched weights here to make the convexity comparison fair.)

Now shock rates up 1% in a parallel move. Both portfolios, having a duration of ~5, fall about 5% — to roughly \$95,000. Almost identical, because duration governs the first-order move. But shock rates up *3%* instead. The bullet falls about `5 × 3% = 15%`, minus a little convexity, landing near \$85,800. The barbell falls about 15% *minus more convexity* — landing near \$86,500. The barbell saved you several hundred dollars on a big move, purely from its extra curvature. On a *down* 3% move, the same convexity makes the barbell gain a few hundred more than the bullet.

*The intuition: tune a barbell and a bullet to the same duration and they tie on small moves, but the barbell's extra convexity makes it the better book on large moves in either direction — convexity is the prize you collect for owning the extremes.*

The barbell's costs are real, though. First, **it's a concentrated bet on the long end's behavior** — if the curve steepens (long rates rise faster than short), the barbell's heavy long leg gets hammered and the bullet wins. Second, the barbell often **gives up some yield**: when the curve is upward-sloping, the missing middle (5–10 year) maturities sometimes offer the best yield-per-unit-of-duration, and skipping them costs you. Third, it requires **active maintenance** — as the long bond ages and the short bond matures, the barbell's duration drifts, and you have to rebalance to keep its shape. The barbell is the most "professional" of the three structures, and the one most often run by managers who explicitly have a curve view.

## The influence figure: how each structure reacts to a curve move

So far we've leaned on a *parallel* rate move — all yields rising or falling together. But the curve rarely moves in parallel; it changes shape. The single most important thing to understand about these three structures is how differently they react to a **steepening** versus a **flattening**, because that difference *is* the curve view each one embeds.

![A bar chart comparing the value change of a ladder a bullet and a barbell when the curve steepens versus when it flattens showing the barbell swinging the most and the ladder barely moving](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-3.png)

The figure shows the same scenario hitting all three: long rates move by 1% while short rates stay roughly put — a steepening on the left, a flattening on the right. Read it as a sensitivity test.

- **The ladder barely moves.** Because it holds bonds across the whole curve, a shape change at the long end is partly offset by stability at the short end. Its long rungs take a hit on a steepener, but they're only a fraction of the book. The ladder's value change is small in either scenario — it's the rate-agnostic structure, and that shows up as a small bar in both directions.
- **The bullet moves moderately.** Sitting in the middle (around 5 years), the bullet is exposed to the *level* of mid-curve rates but not strongly to the long end specifically. A steepening that concentrates at the very long end touches the bullet less than the barbell; the bullet's reaction tracks roughly its own duration at its own point on the curve.
- **The barbell swings the most.** Its long leg dominates its value, so when long rates rise (steepener), the barbell takes the biggest loss of the three; when long rates fall (flattener), it takes the biggest gain. The barbell is the *most curve-sensitive* structure — which is exactly why it's the tool of choice for someone with a strong view on the long end.

#### Worked example: the steepener that punishes the barbell

You hold the \$50k/\$50k 2y/30y barbell from our headline example. The 30-year leg has a duration of about 20; the 2-year leg about 1.9. Now the curve steepens: the 30-year yield rises 1% while the 2-year yield is unchanged.

The long leg's loss: `\$50,000 × 20 × 1% = \$10,000` of price decline (duration times the rate move times the dollar amount). The short leg: `\$50,000 × 1.9 × 0% = \$0`, because its yield didn't move. So the barbell loses about \$10,000, or 10% of the \$100,000 — a brutal hit from a move that left the short end untouched. Now flip it: the 30-year yield *falls* 1% while the 2-year is unchanged. The long leg *gains* about \$10,000 (plus a little extra from convexity), and the barbell is up ~10%.

Compare a 5-year bullet through the same steepener. If the steepening concentrates entirely at the long end and the 5-year yield is roughly unchanged, the bullet barely moves at all. The barbell ate a 10% swing while the bullet shrugged — *because the barbell took a position on the long end and the bullet didn't.*

*The intuition: a barbell isn't just a convexity machine, it's a long-end bet in disguise — own one only if you actually have a view that long rates will fall (or you're willing to be wrong if they rise).*

This is the heart of why "which structure" is never a neutral question. The ladder says "I have no view." The bullet says "I have a view about one point, or one date." The barbell says "I have a view about the shape of the curve, and I want convexity while I wait." Each shape *is* a statement.

## The trade-off underneath everything: reinvestment risk versus price risk

We've now met all three structures and seen them react to rate moves. Step back and notice that everything reduces to a single trade-off — the one introduced in Foundations — and that each structure settles it differently.

![A matrix comparing reinvestment risk price risk yearly liquidity and what each structure bets on across the ladder the bullet and the barbell](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-4.png)

The matrix lays the three structures against the four dimensions that actually distinguish them. Read down each column to see a structure's personality.

- **Reinvestment risk.** The ladder *spreads* it — one rung matures each year, so you reinvest small amounts continuously and never bet your whole book on one day's rates. The bullet *concentrates* it — your entire principal comes back at once and reinvests at one moment's rate. The barbell loads its short leg, so its reinvestment risk clusters at the short end, where bonds roll frequently.
- **Price risk if rates rise.** The ladder is moderate (mid-ish duration, ~5 for a 1–10 ladder). The bullet is moderate too, set by its single maturity. The barbell's long leg makes its price risk *uneven and large on the long end* — that's the convexity prize and the steepener danger in one.
- **Liquidity each year.** The ladder has it built in — cash arrives annually without selling. The bullet has none until maturity. The barbell gets liquidity from its short leg rolling off frequently.
- **What it bets on.** Ladder: nothing — rate-agnostic income. Bullet: one date or one point. Barbell: convexity plus a flattening curve.

Laid out as a table, the personalities are even clearer:

| Dimension | Ladder | Bullet | Barbell |
|---|---|---|---|
| Rate view it embeds | none (rate-agnostic) | one point / one date | flattening + convexity |
| Reinvestment risk | spread across years | concentrated at maturity | clustered at the short leg |
| Price risk on a parallel move | set by duration (~5 for 1–10) | set by its single maturity | set by duration, driven by long leg |
| Convexity (edge on big moves) | moderate | lowest of the three | highest of the three |
| Liquidity before maturity | built in (a rung yearly) | none until maturity | from the short leg rolling off |
| Maintenance | low (just reinvest the rung) | none until maturity | high (rebalance as legs drift) |
| Best for | steady income, flexibility | a known future bill | a long-end view with convexity |

The deep lesson is that **you are always carrying one of these risks** — there is no free lunch where you escape both reinvestment and price risk. Locking in long yields (low reinvestment risk) means accepting big price swings; staying short (low price risk) means accepting that you'll keep rolling over at unknown rates. The structures are three different *allocations* of an unavoidable burden, chosen to fit what you can tolerate and what you're trying to achieve. Notice, too, that the burden interacts with your *horizon*: there is a single holding period — the **duration of your portfolio** — at which price risk and reinvestment risk exactly offset, so that you end up with the same wealth regardless of which way rates jump right after you buy. Matching that crossover point to the date you actually need the money is the formal idea behind [immunization](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge), and it's why a bullet maturing on your spending date is so clean: it sets the crossover exactly where you need it.

#### Worked example: the retiree versus the saver

Two people each have \$100,000.

A **retiree** needs steady spending money and cannot stomach a big drawdown — they'll be spending this, not waiting decades. Their enemy is *both* a price crash (they might need to sell) and a reinvestment cliff (their income shouldn't lurch). A **ladder** fits: ~\$4,000/year of coupons, \$10,000/year of returning principal to spend or reinvest, modest duration so a rate spike doesn't gut the portfolio, and no forced selling. The ladder's "no view" stance is exactly right for someone who can't afford to be wrong.

A **saver** with a \$130,000 tuition bill due in exactly five years has a different enemy: not income volatility, but *missing the target sum on the target date*. A **bullet** (or a 5-year zero) fits — match the maturity to the bill and the in-between price wobble is irrelevant. Their reinvestment risk is concentrated at year five, but by then they're spending the money, not reinvesting it, so the concentration is harmless.

*The intuition: there's no universally best structure — the retiree and the saver hold the same \$100,000 and rationally choose opposite shapes, because they're optimizing for different risks.*

## Putting it together: the \$100,000 portfolio under a +1% move

Let's make the comparison brutally concrete with the running example, and confront the single most counter-intuitive result in portfolio construction.

![A matrix showing one hundred thousand dollars built as a ladder a bullet and a barbell each with about five-year duration all marked down roughly five percent by a parallel one percent rate rise with the barbell helped slightly by convexity](/imgs/blogs/building-a-bond-portfolio-ladders-barbells-and-bullets-7.png)

Here are three portfolios, each holding \$100,000, each tuned to a duration of about 5: a 1–10-year ladder, a 5-year bullet, and a duration-matched 2y/30y barbell. Now apply a *parallel* +1% rise in all yields and watch what happens.

All three lose **about the same amount — roughly 5%, down to about \$95,000.** The ladder loses ~\$5,000 minus a touch. The bullet loses ~\$5,000 on the nose. The barbell loses ~\$4,900 (convexity shaves a little off). The differences are tiny.

This is the result that surprises everyone: **for a parallel move, the *shape* of the portfolio barely matters — *duration* sets the loss.** Three portfolios that look completely different on the maturity line — even spread, single spike, split ends — take almost the identical hit, because they were tuned to the same duration. The shape only starts to matter when (a) the move is *large* (convexity separates them) or (b) the move is *non-parallel* — a curve shape change, where the barbell and bullet diverge sharply as we saw in the influence figure.

#### Worked example: why duration, not shape, drives the parallel loss

Take the bullet: \$100,000, duration 5. A 1% parallel rise costs `\$100,000 × 5 × 1% = \$5,000`. Done. Take the barbell: \$50,000 in a 2-year (duration ~1.9) and \$50,000 in a 30-year (duration ~20) — wait, that's a *portfolio* duration of `(0.5 × 1.9) + (0.5 × 20) ≈ 11`, not 5! That aggressive \$50k/\$50k barbell would lose `\$100,000 × 11 × 1% = \$11,000`, more than double the bullet.

This is the crucial subtlety: **a fair comparison requires equal duration.** The headline \$50k/\$50k barbell is a *longer-duration, more aggressive* portfolio than a 5-year bullet — comparing them directly is comparing a bigger bet to a smaller one. To compare *shape* cleanly, you must first equalize duration (the \$83k/\$17k weights from earlier give a 2y/30y barbell a duration of 5). Once duration is equal, the parallel losses converge — and only then does the barbell's convexity edge and curve exposure show as the *real* difference between the shapes.

*The intuition: never compare two bond portfolios without first checking their durations match — duration is the apples-to-apples adjustment, and once it's equal, structure is a bet on convexity and curve shape, not on the level of rates.*

This also explains how the headline example behaves. The aggressive \$50k/\$50k 2y/30y barbell, with its ~11 duration, is a *bigger* rate bet than the ladder or 5-year bullet. Under a +1% parallel rise it loses about \$11,000, far more than the others — not because barbells are "riskier" by nature, but because *that particular* barbell is much longer in duration. Shape and duration are two separate dials, and confusing them is the most common mistake in this entire subject.

## The dimensions the three shapes ignore: credit, taxes, and costs

Ladder, bullet, and barbell are all defined along the *maturity* axis — they're answers to "where on the curve do my dollars sit?" But a real portfolio lives in more dimensions than maturity, and a beginner who masters the three shapes still has three more decisions to make. None of them changes the shape logic; they layer on top of it.

**Credit quality.** Everything above implicitly assumed Treasuries — bonds with no meaningful chance of default, so the only risk was rates. The moment you add corporate bonds, you add **credit risk** (the chance the issuer doesn't pay you back) and **credit spread** (the extra yield you demand for it). You can ladder, bullet, or barbell *within* a credit tier, or you can use the shapes to manage credit too: a common move is a "credit barbell" — pairing safe short Treasuries with riskier long corporates — so the safe leg anchors the book while the risky leg reaches for yield. The shape vocabulary still applies; you're now spreading or concentrating along *two* axes (maturity and credit) instead of one. The series treats credit in depth starting with [credit risk, the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back).

**Taxes.** For a US investor, *where* a bond lives matters as much as *what* it is. Treasury interest is exempt from state tax; municipal-bond interest is often exempt from federal tax. A laddered muni portfolio is one of the most popular real structures precisely because it stacks tax-free income on top of the ladder's steady-cash design. Taxes also affect the bullet-versus-ladder choice: a ladder's annual maturities can trigger reinvestment at inconvenient tax moments, while a single zero-coupon bullet defers (and in a taxable account, accrues "phantom" taxable income each year even though no cash arrives). The shape is chosen first; the tax wrapper is chosen alongside it.

**Transaction costs and minimum sizes.** Bonds trade over-the-counter with a bid-ask spread, and odd lots (small quantities) trade at worse prices than round lots. A ten-rung ladder means ten separate purchases, each crossing a spread; a bullet means one. For small portfolios, the *frictions* of building and maintaining a granular ladder can quietly eat the yield advantage, which is one reason many investors implement these shapes through bond *funds* and ETFs rather than individual bonds — a topic the [next post in the track](/blog/trading/fixed-income/bond-funds-and-etfs-vs-owning-individual-bonds) takes up. The structure you can *afford to maintain* is part of choosing the structure.

#### Worked example: a credit barbell's yield versus its hidden risk

You build a credit barbell with \$100,000: \$70,000 in 2-year Treasuries yielding 4.0%, and \$30,000 in 10-year high-yield ("junk") corporate bonds yielding 9.0%. The blended yield is `(0.7 × 4.0%) + (0.3 × 9.0%) = 2.8% + 2.7% = 5.5%` — a full 1.5% above an all-Treasury 2-year, which looks like free money.

It isn't. The \$30,000 junk leg carries real default risk; if high-yield default rates run ~4% a year in a downturn with ~40% recovery, the *expected* annual credit loss on that leg is roughly `4% × (1 − 40%) × \$30,000 = \$720` — wiping out a chunk of the extra \$1,500 of income in a bad year, and far more if defaults spike. And the long junk leg is sensitive to *both* rates (duration ~7.5) and spreads, so it can fall sharply when credit conditions sour, often exactly when you'd want it to be stable. The barbell shape gave you a clean way to *separate* the safe anchor from the risky reach — but it didn't make the reach safe.

*The intuition: the three shapes organize maturity risk, but stapling credit onto a leg adds a whole second risk the shape can't hedge — read the yield pickup as compensation for that risk, not as a free lunch.*

**Rebalancing and drift.** A ladder is famously self-maintaining, but a barbell is not. As time passes, every bond's remaining maturity shrinks: a 2-year becomes a 1-year, a 30-year becomes a 28-year, and the *duration* of the whole barbell drifts down. To keep a barbell at a target duration — say, to keep it duration-matched against a benchmark or a bullet — you have to periodically sell the now-shorter short leg, reinvest into a fresh long bond, and re-strike the weights. A bullet drifts too (a 5-year becomes a 4-year), which is fine if you're holding to a fixed date but a problem if you're targeting a constant duration. This maintenance burden is a real cost, and it's why the barbell is the structure most associated with active managers who are watching the book anyway, and the ladder is the structure most associated with set-and-forget retail investors.

## Blends, tilts, and what real portfolios actually look like

In practice, almost no one runs a pure version of any single structure. Real portfolios are blends, and reading them as blends is the skill.

A retiree might run a **ladder with a long-end tilt** — a 1–15 year ladder rather than 1–10 — to capture more yield while keeping the annual-liquidity machine. A pension might run a **bullet-of-bullets**: several maturity-matched buckets, each a bullet against a known wave of future benefit payments, which together look like a lumpy ladder. A bond fund manager with a flattening view might run a **barbell tilt** on top of a benchmark — overweighting the 2-year and 30-year, underweighting the 5–10 — without going to a pure barbell, so they keep some convexity and curve exposure while staying near their benchmark's duration.

The point of learning the three pure structures is that they're the *basis vectors* for reading any portfolio. Show me a bond book and I can decompose it: how laddered (spread across maturities), how bulleted (concentrated), how barbelled (loaded at the ends). And once decomposed, I can read its bets: its duration (level view), its curve shape exposure (steepener/flattener view), its convexity (how it does on big moves), and its liquidity profile (how often cash comes back). That decomposition is the whole craft.

#### Worked example: reading a mystery portfolio

A friend shows you their bond holdings: \$40,000 in a 2-year, \$10,000 in a 5-year, \$10,000 in a 10-year, and \$40,000 in a 30-year. What is this, and what does it bet on?

Decompose it. The big weights at 2 years and 30 years with thin middle holdings make it **mostly a barbell** with a small laddered core. Compute duration: `(0.4 × 1.9) + (0.1 × 4.6) + (0.1 × 8.5) + (0.4 × 20) = 0.76 + 0.46 + 0.85 + 8.0 = 10.1`. A duration of ~10 means this book loses about 10% on a 1% parallel rise — a *long, aggressive* portfolio. The barbell shape says they want convexity and have a flattening (long-end-rallies) view; the long duration says they're also betting rates fall overall. This is a confident, two-part bet: rates down *and* curve flatter. If they're wrong on either — rates rise, or the curve steepens — they'll feel it sharply.

*The intuition: any bond portfolio decomposes into ladder-ness, bullet-ness, and barbell-ness plus a duration number, and reading those four things tells you exactly what the owner is betting on.*

## Common misconceptions

**"A bond ladder protects you from rising rates."** Half true, and the half that's false matters. A ladder *softens* a rate rise — maturing rungs reinvest at the new higher yields, lifting future income — but the bonds you already hold still *fall in price* when rates rise, just like any bonds. If you marked your ladder to market the day after a 1% rate spike, you'd see a loss of roughly its duration (~5% for a 1–10 ladder). What the ladder protects is your *income trajectory* and your *need to sell*, not your instantaneous market value. The protection is real but it's about timing and liquidity, not immunity to price moves.

**"A barbell is always better than a bullet because it has more convexity."** Only when they're duration-matched *and* the move is large or favorable to the long end. Convexity is a second-order effect — it only meaningfully helps on *big* rate moves. On small moves, a duration-matched barbell and bullet are nearly identical. And the barbell carries a *cost*: it's a concentrated long-end bet, so if the curve steepens (long rates rise faster than short), the barbell underperforms badly. Free convexity isn't free; you pay for it with curve risk and often with a bit of forgone yield in the skipped middle.

**"The 2y/30y barbell and the 5-year bullet are the same risk because they 'average' to 5 years."** No — averaging *maturities* is not the same as matching *duration*, and it's the most common error in the whole subject. A \$50k/\$50k 2y/30y barbell has a duration near 11, not 5, because the 30-year leg's sensitivity is enormous and dominates the average. The "average maturity is 16 years" or "the two legs average out" intuition is wrong; you must duration-weight, not maturity-average. Always compute portfolio duration before comparing.

**"A bullet has no reinvestment risk because you just hold it to maturity."** It has *no price risk* if held to maturity for your horizon, but it has *concentrated* reinvestment risk in two places: the coupons you collect along the way (which you must reinvest at unknown rates) and, more importantly, the entire principal that comes back on the maturity date and must be redeployed all at once. A true zero-coupon bond held to your exact horizon does eliminate reinvestment risk for that horizon — but a coupon bullet does not, and a bullet whose maturity is *past* your spending date leaves you reinvesting the whole lump at once.

**"Laddering means buying and holding forever, so it's the lazy, low-return option."** Laddering is low-*maintenance*, not low-return. Its return is close to the average yield across the curve, which over long periods is competitive with reaching for the long end — and it achieves that return with far less price volatility and far more flexibility. The "lazy" framing confuses *making no rate bet* with *earning no return*. A ladder earns the term premium across its rungs; it just declines to gamble on the direction of rates on top of that.

**"Building a bond portfolio is just picking the highest-yielding bonds."** Chasing yield ignores both duration and shape. The highest-yielding bonds are usually the longest (most price risk) or the lowest-quality (most credit risk). A portfolio built by yield-chasing is typically a long-duration, low-credit-quality bet that looks great until rates rise or a default hits. Construction is about matching *shape and duration* to your *goal and risk tolerance* — yield is one input, not the objective.

## How it shows up in real markets

**Retail bond ladders and the 2022–2023 yield surge.** When the Fed hiked rates from near zero to over 5% across 2022–2023, brokerages reported a surge in retail investors building Treasury ladders — buying 1-, 2-, and 3-year T-bills and notes in rungs to lock in yields not seen in fifteen years. The ladder was the natural structure for the moment: it let savers capture the high front-end yields while keeping rungs maturing so they could keep reinvesting if rates stayed high, or stop if rates fell. It's the textbook use of a ladder's rate-agnostic, liquidity-on-a-schedule design during a period when nobody knew if rates had peaked.

**PIMCO and the barbell trade.** Large active bond managers routinely express curve views through barbell and bullet tilts rather than outright duration bets. A manager who believes the long end will rally (a flattening view) overweights the 2-year and the 30-year and underweights the belly — a barbell tilt that picks up convexity while betting on the long end. The reverse — a "bullet" overweight to the 5–10 year belly — expresses the opposite, that the belly is cheap or the curve will steepen. This is the everyday machinery of active fixed-income management; the post on [PIMCO and the bond market](/blog/trading/finance/pimco-and-the-bond-market) covers how the largest of these books are run.

**Defined-benefit pensions and cash-flow matching.** Mature pension funds with predictable benefit payments often run something between a bullet-of-bullets and a ladder: they buy bonds whose maturities and coupons line up with the schedule of benefits they owe each year — "cash-flow matching" or "dedication." Each year's benefit obligation is effectively a small bullet, and the whole book is a lumpy ladder dedicated to the liability stream. This is the institutional, large-scale version of the saver matching a bullet to a tuition bill, and it's the practical face of [immunization and liability-driven investing](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge).

**The 2022 bond rout and the duration lesson.** In 2022, long-duration bond portfolios suffered historic losses — the 30-year Treasury fell roughly a third in price as yields rose about 2.5%, because a ~20 duration times a ~2.5% rate move is a ~50% first-order loss (cushioned by convexity to roughly a third). Investors who had reached for yield by piling into long bonds — effectively running an extreme long-end bullet or a long-tilted barbell — learned the duration lesson the hard way, while laddered and short-duration holders took comparatively mild hits. The episode, detailed from the allocation side in [the 2022 case study where stocks and bonds both fell](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell), was a live demonstration that for a parallel rate move, duration is destiny — exactly the lesson of our \$100,000 comparison.

**Barbell versus bullet as a "butterfly" trade.** On professional rates desks, simultaneously going long the wings (2-year and 30-year) and short the belly (10-year), all duration-neutral, is a named trade: the **butterfly**. It isolates a pure bet on the *curvature* of the yield curve — profiting if the belly cheapens relative to the wings, regardless of the overall level of rates. It's the barbell-versus-bullet comparison from this post turned into a market-neutral position, and it's covered in depth in [trading the curve — steepeners, flatteners, and butterflies](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies).

## When this matters to you / further reading

If you ever hold more than one bond — in a brokerage account, a retirement plan, even indirectly through a target-date fund — you already own one of these shapes whether you chose it or not. Knowing the three structures lets you ask the right question: *what is my bond portfolio actually betting on?* Is it laddered (no view, steady income, flexible)? Bulleted (a date or a point)? Barbelled (convexity and a long-end view)? And critically — what's its *duration*, the number that sets your loss on a parallel rate move?

The practical takeaways are simple. If you have a known future bill, a maturity-matched bullet is the cleanest tool. If you want dependable income and flexibility without betting on rates, a ladder is hard to beat. If you have a genuine view on the long end and want convexity, a barbell expresses it — but never compare it to a bullet without first matching durations. And whatever you build, compute the portfolio duration first; it's the one number that tells you how the whole book breathes when rates move.

To go deeper: the [duration post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) and [convexity post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story) are the analytical backbone of everything here; [reinvestment risk and the two faces of yield](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield) is the trade-off underneath all three shapes; [trading the curve](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies) turns the barbell-versus-bullet comparison into live curve trades; and on the allocation side, [government bonds as the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) places these structures inside a whole-portfolio view. For the heavy math of pricing the bonds that fill these structures, see [bond pricing](/blog/trading/quantitative-finance/bond-pricing) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics).

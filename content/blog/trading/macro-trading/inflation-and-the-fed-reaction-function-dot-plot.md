---
title: "Inflation and the Fed Reaction Function: From CPI to the Dot Plot"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The Fed does not move randomly. It follows a reaction function — tighten when inflation runs above 2%, ease when jobs weaken. Model that function from the data and you can anticipate policy before it is announced."
tags: ["macro", "monetary-policy", "inflation", "federal-reserve", "dual-mandate", "core-pce", "taylor-rule", "dot-plot", "fomc", "interest-rates", "cpi", "data-dependence"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The Federal Reserve does not set interest rates on a whim; it follows a **reaction function** — broadly, it raises rates when inflation runs above its 2% target and eases when unemployment rises. If you can model that function from the inflation and jobs data, you can anticipate policy before it is announced, which is the entire macro edge.
>
> - **The dual mandate is the input.** Congress gave the Fed two jobs — stable prices and maximum employment. Every rate decision is the Fed weighing an **inflation gap** (how far inflation is from 2%) against an **employment gap** (how far the labor market is from full employment).
> - **The Fed targets core PCE, not the CPI you see in headlines.** Its 2% goal is on core PCE inflation, which peaked at **5.6% in February 2022** — and that 3.6-point gap above target is what drove the fastest hiking cycle in 40 years.
> - **The Taylor rule and the dot plot encode the reaction function.** The Taylor rule turns the two gaps into an implied rate; the quarterly dot plot shows where each Fed official thinks rates are headed. The median dot is the signal.
> - **The one number to remember:** the **inflation gap** — core PCE minus 2%. When it is large and positive, the Fed hikes and holds. When it closes, cuts come into view. Watch the gap, not the headlines.

In the second half of 2021, the most powerful central bank on Earth told the world that the inflation rolling through the economy was **"transitory."** Prices were rising — used cars, shipping containers, lumber, eventually rent and restaurant meals — but the official line from the Federal Reserve was that these were pandemic-reopening distortions that would fade on their own. Rates stayed pinned near zero. The Fed was still *buying* bonds, pumping money into the system, as inflation climbed. The word "transitory" was repeated so many times it became a meme.

It was one of the largest forecasting errors in the modern history of monetary policy. By the time the Fed admitted it — Chair Jerome Powell formally "retired" the word transitory in late November 2021 — consumer prices were rising at the fastest pace since the early 1980s, and they were still accelerating. What followed was a violent correction: the Fed went from zero to a 5.25%–5.50% policy rate in seventeen months, the steepest tightening cycle in four decades. Markets that had partied on free money for years got repriced, brutally, in real time.

Here is the thing a trader needs to extract from that episode. The Fed's catastrophic delay was not random, and neither was its furious catch-up. Both were the output of the same machine: a **reaction function** that maps the state of inflation and the labor market onto a policy rate. The Fed misread the *inputs* in 2021 — it thought the inflation gap was temporary — and then, once the data proved otherwise, the function did exactly what the function does: it slammed rates higher until inflation broke. If you had been modeling the reaction function yourself, reading the same CPI and jobs prints, you would have seen the catch-up coming long before most of the market did. That is the whole game, and this post is about how to play it.

![Two input gaps feeding a reaction function that sets the Fed policy rate with hike and ease branches](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-1.png)

## Foundations: the dual mandate, the 2% target, and how inflation is measured

Before we can model what the Fed does, we have to be precise about what the Fed is *for*. Almost everyone has a vague sense that the Fed "controls interest rates," but very few people can state, cleanly, the rules it operates under. Those rules are the foundation of the reaction function, so we build them from zero.

### The dual mandate: stable prices and maximum employment

The Federal Reserve is not free to pursue whatever goals it likes. In 1977, the U.S. Congress amended the Federal Reserve Act to give the central bank a specific, legally defined job, and that job has two parts. The Fed is instructed to conduct monetary policy so as to promote **maximum employment** and **stable prices** (a third, often-forgotten clause adds "moderate long-term interest rates," which in practice follows from the first two). This pairing is what everyone means by the **dual mandate**.

Read those two goals again and notice that they pull in opposite directions much of the time. To support employment, you generally want *easy* money — low rates, cheap credit, more spending, more hiring. To keep prices stable, you sometimes need *tight* money — high rates, expensive credit, less spending, cooler demand. The Fed lives permanently in the tension between these two, and almost every interesting decision it makes is a judgment call about which side of the mandate is more at risk *right now*. That tension is the seesaw we will return to throughout this post.

Most other major central banks, by contrast, have a **single mandate**: price stability, full stop. The European Central Bank's primary objective is inflation near 2%; employment is, at most, a secondary consideration. This is why the Fed sometimes moves differently from its peers — it has explicit legal cover to tolerate a bit more inflation if the labor market is weak, or to push unemployment up a little if inflation is the bigger threat. The dual mandate is not a slogan; it is the constitution of the reaction function.

### The 2% target: a number that did not always exist

"Stable prices" is a goal, not a number. For decades the Fed pursued it without ever saying precisely what it meant. That changed in **January 2012**, when the Fed formally announced, for the first time, that it interprets stable prices as **2% inflation per year** over the longer run. Not zero — *two percent*. This surprises people. Why would a central bank deliberately aim for prices to rise every year rather than stay flat?

There are two good reasons. First, a small positive inflation rate is a buffer against **deflation** — falling prices — which is far more dangerous than mild inflation because it makes people delay purchases ("it'll be cheaper next month"), crushes spending, and is brutally hard to escape (Japan spent the better part of three decades fighting it). Second, 2% gives the Fed **room to cut**. If the normal level of interest rates already sits a couple of points above zero because of that built-in inflation, the Fed has more space to slash rates in a recession before it hits the dreaded zero lower bound. A 2% target is, in effect, insurance against the Fed running out of ammunition.

The practical upshot for a trader is enormous: **2% is the line.** Every inflation print is implicitly being compared to it. When inflation is at 2%, the Fed is at peace and rates can drift toward neutral. When inflation is meaningfully above 2%, the inflation side of the mandate dominates and the bias is to tighten. The distance between actual inflation and that 2% line — the **inflation gap** — is the single most important number in the entire reaction function.

### The framework shift that caused the 2021 error

There is a piece of recent history embedded in the 2% target that directly explains the "transitory" disaster, and understanding it tells you something about how the reaction function can be *deliberately* re-weighted. In August 2020, the Fed announced a new framework called **flexible average inflation targeting**, or FAIT. The old framework treated 2% as a ceiling-ish target the Fed wanted to hit symmetrically. FAIT changed the rule: after periods when inflation had run *below* 2% — which had been the case for most of the 2010s — the Fed would now deliberately aim to run inflation *moderately above* 2% for a while, so that inflation *averaged* 2% over time. The logic was that a decade of undershooting deserved a period of overshooting to make up for it.

FAIT also rewired the employment side. The Fed said it would now respond only to *shortfalls* from maximum employment, not to a hot labor market per se. In the old framework, an unemployment rate falling to very low levels was itself a reason to pre-emptively tighten, because tight labor markets were assumed to generate inflation (the Phillips curve logic). Under FAIT, the Fed announced it would *not* tighten just because unemployment was low; it would wait for actual inflation to show up.

Now you can see the 2021 trap mechanically. FAIT told the Fed two things that, combined, were a recipe for being late: (1) tolerate above-target inflation for a while because we undershot for years, and (2) do not pre-emptively tighten just because the labor market is hot. So when inflation surged in 2021, the framework itself disposed the Fed to *wait* — to treat the overshoot as the planned "make-up" overshoot rather than a genuine inflation problem, and to ignore the screaming-hot labor market because the new framework said low unemployment was not, by itself, a tightening trigger. The reaction function had been deliberately re-weighted to be more patient, and that patience became a costly delay when the inflation turned out to be real and persistent. In 2025 the Fed formally walked back much of FAIT, returning to a more symmetric 2% target and dropping the "shortfalls only" language — an admission that the re-weighting had backfired. The lesson for a trader is that the *weights* in the reaction function are not fixed forever; the Fed can and does change them, and a framework change is itself a tradable shift in how policy will respond to the same data.

### CPI vs core PCE, and why the Fed prefers core PCE

Now the subtle part, and the one that trips up most beginners. There is not one inflation number; there are several, and the Fed does not target the one you see in the news.

**CPI**, the Consumer Price Index, is published monthly by the Bureau of Labor Statistics. It tracks a fixed-ish basket of goods and services a typical urban household buys. When a headline screams "inflation came in at 4.0%," it almost always means CPI year-over-year. CPI is the number markets react to in the moment because it comes out first and is the most widely watched.

**PCE**, the Personal Consumption Expenditures price index, is published by the Bureau of Economic Analysis. It covers a broader, *shifting* basket and explicitly accounts for substitution — when beef gets expensive, people buy more chicken, and PCE captures that switch. PCE typically runs a few tenths *below* CPI because of this substitution effect and different weightings (notably, housing carries a smaller weight in PCE).

Both come in two flavors. **Headline** inflation includes everything. **Core** inflation strips out food and energy, the two most volatile components, because they are jumpy, driven by global supply shocks (an oil embargo, a drought) rather than by domestic monetary conditions, and they tend to mean-revert. Core is the cleaner read on the *underlying trend* — the inflation that monetary policy can actually influence and that tends to stick around.

Put it together and you get the Fed's preferred gauge: **core PCE**. When the Fed says it targets 2% inflation, it means 2% on core PCE, measured year-over-year. This is not trivia — it changes how you read every release. A hot CPI print driven by a gasoline spike might barely move core PCE and need not force a policy response; a hot *core* print, especially in sticky services like rent and wages, is exactly what makes the Fed tighten and hold. When you decompose an inflation surprise, the real question is always: *does this move the gauge the Fed actually steers by?*

![Core PCE inflation line from 2021 to 2026 with a 2 percent target line and a 5.6 percent peak marked](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-2.png)

The chart above is the variable that runs the reaction function. Core PCE sat near or below 2% before the pandemic, then tore upward through 2021, and peaked at **5.6% in February 2022** — the highest in four decades. Everything shaded above the green 2% line is the inflation gap, and that gap is what the Fed was reacting to when it hiked seventeen times. Notice the right side of the chart, too: core PCE settled near 2.8% through 2024 but has been drifting back up toward 3% in 2025–26. Inflation is not a problem you "solve" once; it is a regime that can return, which is precisely why traders treat the inflation gap as a live variable rather than a settled fact.

It is worth seeing the two main gauges side by side, because the gap between them is itself a tradable insight.

![Headline CPI and core PCE plotted together from 2020 to 2026 with the CPI peak of 9.06 percent marked](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-3.png)

Here is the contrast in one picture. Headline CPI (amber) is the loud line — it rocketed to **9.06% in June 2022**, then fell almost as fast as it rose, swinging wildly with energy prices. Core PCE (blue) is the quiet line — it topped out lower, at 5.6%, and came down far more gradually because services inflation is sticky. The lesson is visual and permanent: **the headline is the noise, core is the signal.** Markets trade the noise on release day; the Fed steers by the signal over quarters. A trader who understands the reaction function watches both — the headline for the short-term market reaction, core PCE for the actual policy trajectory.

For a deeper treatment of *why* the inflation-adjusted version of any rate is what ultimately prices assets, see the companion piece on [real vs nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). Here, we stay focused on the policy side: how these inflation readings become a rate decision.

## The reaction function in words: two gaps, one rate

Now we can state the reaction function plainly, in language before any math. The Fed is constantly asking two questions:

1. **The inflation question.** How far is inflation (core PCE) from the 2% target? This is the **inflation gap**. If inflation is at 5.6% and the target is 2%, the gap is +3.6 percentage points — far too hot. A large positive gap says: *tighten.*
2. **The employment question.** How far is the labor market from "maximum employment"? This is the **employment gap** (sometimes framed via the **output gap** — how far GDP is from its sustainable potential). If unemployment is very low and the economy is running hot, there is no slack to support — and a tight labor market can itself stoke wage-driven inflation. If unemployment is rising and the economy is cooling, the gap says: *ease.*

The reaction function combines these two answers into one number: the **policy rate** (the federal funds target). When both questions point the same way — inflation too high *and* the economy overheating, as in 2022 — the answer is unambiguous: hike hard. When they conflict — say, inflation slightly above target but unemployment climbing — the Fed must weigh which risk is larger, and that judgment is where the art lives.

The seesaw is the right mental model. Picture a balance beam with the Fed as the fulcrum in the middle. On one side sits the inflation risk; on the other, the unemployment risk. Whichever side is "heavier" tilts the beam, and the Fed adjusts the policy rate to push back and re-level it. In 2022, the inflation side was crushingly heavy, so the Fed leaned all its weight onto rates to lift it. In 2024, as inflation cooled and the jobs side started to weaken, the beam began to tilt the other way, and the Fed pivoted toward cuts.

### The neutral rate: the anchor the seesaw balances around

There is a hidden third number in the reaction function that you cannot see directly but that governs everything: the **neutral rate**, often written **r-star** (r\*). It is the level of the policy rate that neither stimulates nor restrains the economy when inflation is at target and employment is full — the "resting" rate the beam settles to when both sides are balanced. If the Fed sets rates *above* neutral, policy is restrictive (slowing the economy, fighting inflation). If it sets rates *below* neutral, policy is accommodative (stimulating, supporting jobs). Neutral is the dividing line between "the Fed is pressing the brake" and "the Fed is pressing the gas."

The catch is that nobody can observe r-star; it can only be estimated, and the estimates are wide and frequently revised. For most of the 2010s, the Fed's longer-run dot — its proxy for the neutral *nominal* rate — drifted down toward roughly 2.5% (about 0.5% real plus the 2% inflation target), reflecting a belief that structural forces like aging demographics and slow productivity had lowered the resting rate of the economy. That low neutral estimate is part of why the Fed was comfortable holding rates near zero for so long: if neutral is only 2.5%, then zero is deeply stimulative, which seemed appropriate after the 2008 crisis.

Here is why r-star matters for trading the reaction function. Two questions are always in play: *how far is the rate from neutral right now* (the stance of policy), and *is the market's estimate of neutral itself shifting* (which re-anchors the whole curve). When you hear debates about whether neutral has risen post-pandemic — whether r-star is now 3% or 3.5% rather than 2.5% because of deficits, deglobalization, and the energy transition — that is not academic. A higher neutral rate means the Fed can hold rates higher *without* being as restrictive as the market assumed, which is profoundly hawkish for the long end of the curve and supports the "higher for longer" regime. Watching the longer-run dot creep up in successive SEPs is one of the most important slow-moving signals in macro, because it tells you the entire seesaw is being re-anchored to a higher resting point.

![A balance beam with inflation and unemployment weights pressing down on the Fed as the central fulcrum](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-4.png)

This is the diagram to keep in your head whenever you read a Fed decision. Every statement, every press conference, every dot plot is the Fed telling you which side of this seesaw it judges to be heavier and how hard it intends to push back. Your job as a trader is to model the same beam — read the same inflation and jobs data — and form your own view of where it tilts, so you can see the policy move *before* the Fed announces it.

This connects directly to the mechanics of how the Fed actually sets the rate it targets — the open market operations, the floor and ceiling of the corridor — which is covered in detail in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates). Here we are one level up: not *how* the rate is set, but *why* it lands where it does.

#### Worked example: the inflation gap that drove 2022

Let us put numbers on the inflation side of the seesaw at its most extreme. In June 2022, headline CPI printed **9.06%** year-over-year — a 40-year high — and core PCE, the Fed's actual gauge, was running near **5.6%** at its February peak and still around 5% mid-year. The target is **2%**.

The inflation gap on the Fed's preferred measure was therefore roughly:

- Core PCE gap = 5.6% − 2.0% = **+3.6 percentage points** at the peak.

Think about what a 3.6-point gap means. Inflation was running at nearly *three times* the target. On the headline number it was worse — a gap of 9.06% − 2.0% = **+7.06 points**, more than four times target. A reaction function does not shrug at numbers like these. A gap of this size, sustained, demands an aggressive and prolonged tightening response, and that is precisely what the data delivered: the Fed raised the funds rate from effectively zero to 5.25%–5.50% and then *held* there for over a year. **A 3.6-point core inflation gap is not a "watch it" situation; it is a "tighten until it breaks" situation — which is exactly how the Fed behaved.**

## The Taylor rule made simple

The reaction function in words is intuitive, but it can be made quantitative, and the most famous attempt is the **Taylor rule**, proposed by economist John Taylor in 1993. It is a simple formula that takes the two gaps and spits out an implied policy rate. You do not need to memorize it, but you should understand its shape, because it makes the reaction function concrete.

In plain language, the Taylor rule says the policy rate should equal:

- a **neutral real rate** (the rate that neither stimulates nor restrains the economy, often assumed around 0.5% in real terms), plus
- **current inflation** (so the rate keeps pace with rising prices), plus
- a weight times the **inflation gap** (inflation minus target), plus
- a weight times the **output/employment gap** (how far the economy is above or below potential).

The classic version uses a weight of **0.5** on each gap. Written as a formula:

```
target rate = neutral_real + inflation
              + 0.5 * (inflation - target)
              + 0.5 * (output_gap)
```

The single most important property of this rule is hidden in the inflation term. Notice that when inflation rises by one point, the rate rises by *more* than one point: once for the "current inflation" term and again, half a point, for the "inflation gap" term. This is the **Taylor principle**, and it is the most important idea in modern monetary policy. To actually fight inflation, the central bank must raise *nominal* rates by more than the rise in inflation, so that the **real** (inflation-adjusted) rate goes up. If you only matched inflation point-for-point, real rates would stay flat and you would not be restraining anything. The Taylor principle is the mathematical statement of "you must get ahead of inflation, not chase it" — and it is precisely the principle the Fed violated in 2021 by holding rates at zero while inflation climbed.

There is one more feature of how the Fed actually behaves that the simple rule misses: **interest-rate smoothing**, also called policy inertia. Central banks do not jump to the rule-implied rate in one move; they move in steps and adjust gradually, partly to avoid disrupting markets and partly because the data is noisy and they would rather not whipsaw. A more realistic reaction function therefore includes a term that pulls the new rate toward *last period's* rate — the Fed moves a fraction of the way to the target each meeting. This is why hiking and cutting cycles play out as *sequences* of 0.25-point (or, in 2022, 0.75-point) moves rather than a single jump, and it is why "how fast" is often a separate trade from "how far." The smoothing term is also why the Fed telegraphs so heavily: it wants the market to anticipate the gradual path so that each individual step is barely a surprise.

Different versions of the rule put different weights on the gaps, and the spread between them is informative. The original 1993 Taylor rule uses 0.5 on each gap. A popular "balanced-approach" variant doubles the weight on the output/employment gap to 1.0, making it more responsive to a weakening labor market — closer to how a dual-mandate central bank with a soft jobs market actually behaves. The Fed itself publishes a menu of these rule prescriptions in its semi-annual Monetary Policy Report, precisely so that observers can see the range. When the different rule variants *disagree* sharply — say, the inflation-focused version says hike but the balanced-approach version says hold because jobs are soft — that disagreement is the quantitative signature of a near-balanced seesaw, the judgment regime where the Fed's call is hardest to predict and the market's pricing swings most.

A crucial caveat: the Fed does **not** mechanically follow any Taylor rule. It is a benchmark, a sanity check, not an autopilot. The neutral rate is unobservable and debated; the output gap is estimated and revised; and the Fed reserves the right to use judgment. But the rule is invaluable precisely because it tells you what a "by-the-book" reaction function would prescribe — and the *gap* between the Taylor rule and where the Fed actually sits is itself a signal. When the Fed is far below what the rule implies, it is running easy relative to the rules-based benchmark, and you should expect either hikes to come or a deliberate, explainable reason for the dovishness. When it is *above* what the rule implies — holding rates high even as the rule-implied rate falls — the Fed is signaling extra caution about re-igniting inflation, a "higher for longer" stance that reprices the whole front end.

#### Worked example: a simplified Taylor-rule estimate for early 2022

Let us run the rule at the moment the Fed was most behind the curve. In early 2022, core PCE inflation was at its peak of **5.6%**, the target was **2%**, unemployment had fallen to **3.4%** — well below most estimates of the "full employment" rate of roughly 4.4%, implying a positive output gap of perhaps +2 points (the economy running hot). Take the neutral real rate as **0.5%**.

Plug in the classic weights of 0.5 on each gap:

- Neutral real rate: **0.5%**
- Plus current inflation: + 5.6% → running subtotal **6.1%**
- Plus 0.5 × inflation gap = 0.5 × (5.6% − 2.0%) = 0.5 × 3.6 = **+1.8%** → subtotal **7.9%**
- Plus 0.5 × output gap = 0.5 × (+2.0%) = **+1.0%** → **implied rate ≈ 8.9%**

The Taylor rule, fed the early-2022 data, implied a policy rate near **8.9%**. Where was the Fed actually sitting in early 2022? At **0% to 0.25%** — it had not even started hiking yet, and was *still buying bonds*. The rule said roughly 9%; reality said zero. **A gap of nearly nine percentage points between the rules-based benchmark and the actual policy rate is the clearest possible quantification of "the Fed was catastrophically behind the curve" — and it foretold the violent catch-up that followed.** Even using gentler assumptions, the rule screamed that rates needed to be dramatically higher, fast.

## The dot plot and how to read it

So far we have the reaction function (in words and via the Taylor rule). But the Fed also tells you, four times a year, where it *expects* to take rates. This forecast is published in the **Summary of Economic Projections (SEP)**, and its most famous component is the **dot plot**.

Here is what it is. At four of the eight annual FOMC meetings (March, June, September, December), each of the nineteen Fed officials — the seven Board governors and the twelve regional Fed bank presidents — anonymously plots a single dot showing where they think the appropriate federal funds rate will be at the end of the current year, each of the next two or three years, and over the "longer run." Stack all the dots and you get a scatter: a column of dots for each horizon, showing the spread of opinion inside the committee.

The dots are anonymous, so you cannot tell *whose* dot is whose — but you do not need to. **The number that matters is the median dot** in each column: line the dots up from low to high and take the middle one. The median is the best single read on where the committee, as a body, expects rates to go. It is, in effect, the Fed publishing its own forecast of its own reaction function.

![A dot plot scatter with FOMC member rate projections per horizon and the median dot highlighted](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-6.png)

Read the figure left to right. Each column is a horizon — end of this year, next year, the year after, and the "longer run" (the Fed's estimate of the neutral rate once the economy is balanced). Within a column, every blue dot is one official's projection; the amber dot is the median. The median path here steps **down** over time — say 5.00% this year, 4.00% next, 3.00% the year after, converging toward a longer-run estimate near 2.50%. That descending median path *is* the Fed's expected glide back toward neutral as it forecasts inflation falling toward 2%.

Three things to extract every time a dot plot drops:

1. **The median path** — the headline. Did it move up or down versus the last SEP? An upward shift in the dots is hawkish (higher rates for longer); a downward shift is dovish.
2. **The longer-run dot** — the committee's estimate of "neutral." If this drifts up, the Fed is signaling that the whole rate regime may settle higher than markets assumed, which reprices everything long-duration.
3. **The dispersion** — how spread out the dots are. Tight clustering means consensus and confidence; a wide spread means genuine disagreement inside the committee, which makes future decisions less predictable and the median dot less reliable as a forecast.

The companion post on [trading the FOMC statement, presser, and dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) goes deep on the mechanics of trading these releases minute by minute. Here, the point is conceptual: the dot plot is the reaction function's own forecast of itself, and your edge comes from comparing it to two things — your model, and what the market is pricing.

One more refinement makes the dot plot far more useful than most people realize: read it *together with* the rest of the SEP. The dots never come alone. The same Summary of Economic Projections also publishes the committee's median forecasts for GDP growth, the unemployment rate, and inflation (both headline and core PCE) over the same horizons. That means the dots are not floating numbers — they are *conditional on* a specific economic forecast. If the SEP projects core PCE falling to 2.4% next year and unemployment at 4.2%, then the median dot is the rate the committee thinks is appropriate *given that path*. The instant the incoming data diverges from the SEP's forecast, you know the dots will move. This is the secret to using the dot plot well: do not just read the dots; read the inflation and unemployment forecasts beside them, then watch whether reality is tracking above or below those forecasts. If inflation is coming in *below* the SEP's projected path, the dots will drift down at the next meeting — and you can position for that drift before it prints, because you are watching the same data the committee is.

#### Worked example: reading the median dot path against market pricing

Suppose the September dot plot shows a median path of **5.00%** for end of this year, **4.00%** for next year, and **3.00%** the year after. Now look at what the interest-rate market is pricing through SOFR futures (the contracts that bet on the future path of the funds rate). Suppose the market is pricing the funds rate at only **3.25%** by the end of next year — well below the Fed's 4.00% median dot.

That is a **0.75-point divergence**, and it is a tradable disagreement. The market is effectively saying, "we think the Fed will cut faster than its own dots, because we expect inflation to fall faster (or the economy to weaken faster) than the Fed does." One of the two will be wrong:

- If the **incoming data** (core PCE, jobs) cools faster than the Fed expects, the dots will move *down* toward the market — the market was right, and front-end rates fall further.
- If inflation proves **sticky**, the Fed holds, the cuts the market priced do not arrive, and front-end rates have to reprice *up* toward the dots — the market was wrong, and you would have wanted to be positioned for higher-for-longer.

The trade is to take a view on the gap. **When the market is priced far more dovishly than the dots and the inflation data is *not* cooperating, fading the market's dovishness — positioning for the Fed to deliver fewer cuts than priced — has historically been one of the cleanest macro setups, because the reaction function says the Fed cannot cut while the inflation gap stays open.** This exact divergence played out repeatedly in 2023–24, when markets kept pricing imminent cuts that the data kept postponing.

#### Worked example: how wrong the dots can be

To drive home that the dots are forecasts and not promises, take the most embarrassing miss in their history. At the **December 2021** meeting, the median dot for the end of **2022** implied a federal funds rate of roughly **0.9%** — the committee, in its own published projection, thought it would raise rates only about three-quarters of a point over the coming year.

What actually happened? By the end of 2022, the funds rate was **4.50%** (upper bound). The dots had implied roughly 0.9%; reality delivered 4.50%:

- Miss = 4.50% − 0.9% ≈ **+3.6 percentage points** in twelve months.

The committee underestimated its own year-ahead rate by more than three and a half points — an enormous error for a one-year forecast from the people who *set* the rate. Why? Because the dots were conditional on the December 2021 inflation forecast, and that forecast was the "transitory" view: the SEP projected inflation falling on its own. When inflation did the opposite, the dots had to be torn up and the rate raised five times faster than projected. **The dots tell you the committee's current conditional plan, not its commitment — and when the data underlying that plan is wrong, the dots can miss by 3.6 points in a single year, so you trade them as a starting hypothesis to be revised on every print.**

## Data-dependence: why every CPI print is a market event

There is one more pillar of the modern reaction function, and it is the reason your economic calendar matters more than any pundit. The Fed describes itself as **data-dependent**: rather than committing to a fixed path of hikes or cuts, it adjusts meeting by meeting based on the incoming data, especially inflation and jobs. Powell says some version of "we will be guided by the totality of the data" in essentially every press conference.

Data-dependence has a direct, mechanical consequence for trading: **each major data release is a chance for the market's expectation of the reaction function to be revised.** If the reaction function maps data onto rates, then a surprise in the data instantly changes the expected rate path — and bonds, the dollar, and equities all reprice in seconds. This is why the monthly CPI release and the monthly jobs report (non-farm payrolls) are the two most violent scheduled events on the macro calendar. They are not just numbers; they are *inputs to the function the whole market is trying to model.*

The key idea is that markets trade the **surprise**, not the level. Before each CPI print, there is a consensus forecast (say, +0.3% month-over-month). The market has already priced in that expectation. What moves prices is the *deviation* from it. A CPI print that comes in hotter than expected pushes the market's estimate of the inflation gap up, which pushes the expected policy rate up, which sends short-term bond yields up and (usually) stocks down — all within milliseconds. A cooler-than-expected print does the reverse. The level barely matters; the gap between actual and expected is everything.

This is also why the *composition* of a print matters as much as the headline. A CPI surprise driven by gasoline (volatile, transitory, not core) gets faded quickly because the market knows it does not move the Fed's gauge. A surprise driven by **core services** — rent, wages, the sticky stuff — moves markets far more, because it shifts the read on the underlying inflation the Fed actually fights. Learning to decompose a print in real time, to ask "did this move *core*, and did it move the *sticky* part of core?", is what separates a trader who understands the reaction function from one who just reacts to the headline number.

The Fed has even publicized which slices of inflation it watches most closely. During the 2022–23 fight, Powell repeatedly pointed to **core services excluding housing** — sometimes called "supercore" inflation — as the cleanest read on whether wage-driven, demand-side inflation was genuinely cooling, because it strips out both the volatile goods (which were already deflating as supply chains healed) and the lagging, formula-driven housing component. So a single CPI release actually contains a hierarchy of signals: the headline (what the public sees), core (food and energy stripped out), core services (goods stripped out), and supercore (housing stripped out too). The deeper you go down that hierarchy, the closer you get to the part of inflation the reaction function is genuinely targeting. A trader who, on release day, can quickly judge "the headline was hot but it was all energy and used cars — supercore actually decelerated" will frequently take the *opposite* side of the knee-jerk move, because the print that scared the headline-readers was reassuring on the measure the Fed steers by.

There is a second, subtler dimension to data-dependence: the **jobs side** has its own hierarchy of signals, and a reaction-function trader weights them too. The monthly payrolls number gets the headlines, but the unemployment rate, the participation rate, average hourly earnings (wage growth feeds directly into services inflation), and the quits rate (a gauge of labor-market tightness) all feed the employment-gap read. In 2024, the gradual rise in the unemployment rate from 3.4% toward 4.2% was the single data trend that flipped the Fed from "hold" to "cut," precisely because a *rising* unemployment rate — even from a low base — is the labor-market signal the reaction function weights most heavily. Watching that one series climb told you the easing cycle was coming long before the first cut.

The transmission from a shifted rate expectation to actual asset prices — how a change in the expected path of the funds rate ripples through the yield curve, credit, the dollar, and equities — is its own deep topic, covered in [monetary policy transmission](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets). For our purposes, the takeaway is that data-dependence turns the calendar itself into the trade: you position around your model of how each print will move the reaction function.

## Common misconceptions

A handful of beliefs about the Fed are widespread, repeated by commentators, and wrong. Each one, corrected with a number, sharpens the reaction-function model.

**Misconception 1: "The Fed targets CPI."** No. The Fed targets **core PCE**, and the difference is material. Core PCE typically runs a few tenths to a full point below headline CPI. In June 2022, headline CPI was **9.06%** while core PCE was near **5%** — a gap of about four points. A trader who only watched CPI would have systematically *overestimated* how far above target the Fed thought inflation was. When you model the inflation gap, model it on core PCE versus 2%, not CPI versus 2%. The market reacts to CPI on release day, but the Fed steers by core PCE over quarters.

**Misconception 2: "The dots are promises."** They are not. The dot plot is a snapshot of each official's *current best guess*, and it changes every quarter as the data changes. The dots are notorious for being wrong: at the end of 2021, the median dot for end-2022 implied a funds rate under 1%; the Fed actually ended 2022 at 4.50% — off by roughly **3.5 percentage points** in a single year. The dots tell you the reaction function's *current* forecast, conditional on the Fed's *current* view of the economy. When the data changes, the dots change. Trade them as a conditional forecast, never as a commitment.

**Misconception 3: "The Fed reacts to the stock market."** Mostly false, and dangerous to trade on. The Fed reacts to its mandate variables — inflation and employment — not to the S&P 500 level. The "Fed put" (the belief that the Fed will cut to rescue falling stocks) is real only insofar as a market crash *threatens the real economy and the mandate*. Through all of 2022, stocks fell roughly **25%** and the Fed kept hiking, because inflation, not equity prices, was the binding constraint. The Fed will let stocks fall a long way if the inflation gap is open. It cares about financial conditions as a transmission channel, not about the index for its own sake.

**Misconception 4: "If inflation is at 2%, the Fed will cut."** Not necessarily, and the confusion costs money. The 2% target is on inflation, but the *level* of rates the Fed holds depends on the **neutral rate** and the employment side too. With unemployment low and inflation at target, the Fed can comfortably sit at a "neutral" rate — neither cutting nor hiking. Reaching 2% inflation removes the pressure to *tighten*; it does not automatically trigger *easing*. Easing requires either inflation falling decisively below target or the labor market weakening. Hitting target is necessary but not sufficient for cuts.

**Misconception 5: "Rate decisions are unpredictable."** The opposite is true, and it is the thesis of this entire post. The Fed telegraphs relentlessly — through the dot plot, the statement, speeches, and its own framework. The 2022 hiking cycle was *visible months ahead* to anyone running the reaction function on the inflation data. What is unpredictable is the *data*; the Fed's *response* to the data is highly modelable. Your edge is not predicting the Fed's whims — there are none — but modeling its function and anticipating its response to the prints you can see coming.

## How it shows up in real markets

The reaction function is not an academic toy. It explains, cleanly, the three defining macro episodes of this decade. Let us walk through each with real dates and numbers, because seeing the function operate in the wild is what makes it usable.

### 2021: the transitory error

In 2021, the Fed misjudged the *inputs* to its own function. Core PCE climbed from 1.6% in January 2021 to **4.9% by December 2021** — it had nearly *tripled* and blown well past target — yet the funds rate stayed at 0%–0.25% and the Fed kept expanding its balance sheet through quantitative easing until March 2022. The reaction function was not broken; the Fed's *reading of the inflation gap* was. It believed the gap was transitory — driven by reopening, supply chains, and base effects that would fade — so it treated a large gap as if it were small.

The lesson for a trader is precise and uncomfortable: the function is only as good as the inputs. The error was not that the Fed abandoned its reaction function but that it mis-classified the inflation it was seeing. A trader watching *core* PCE march from 1.6% to 4.9% — not the comforting "transitory" narrative — had every reason to expect a violent catch-up, and to position short the front end of the bond market well before the Fed admitted it was wrong. The data was screaming; the Fed's framing was wishful. Trust the data over the framing.

### 2022: the catch-up

Once the Fed accepted that the inflation gap was real and persistent, the reaction function did exactly what the function does — and it did it at record speed. From March 2022 to July 2023, the Fed raised the funds rate from 0.25% to **5.25%–5.50%**, a move of more than five full points in seventeen months, including four consecutive **0.75-point** hikes — the most aggressive tightening since Paul Volcker's campaign in the early 1980s. (For the historical precedent of a central bank crushing entrenched inflation with brutal rate shocks, see [Paul Volcker and the 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).)

This is the function in its purest form: a 3.6-point inflation gap, an overheating labor market with unemployment at 3.4%, both sides of the seesaw pointing the same way — tighten. There was no conflict to weigh, so the Fed moved with conviction. The trade was to be positioned for higher-for-longer in the front end and to respect that the Fed would *not* blink for stocks while the inflation gap stayed open.

![Core PCE inflation, the unemployment rate, and the fed funds upper bound on a shared timeline from 2021 to 2026](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-5.png)

This chart is the reaction function in one frame. Watch the sequence. Core PCE (red) peaks in early 2022. The fed funds rate (blue, stepping) chases it upward with a lag — the function responding to the inflation gap. And then, with a further lag, unemployment (green) begins to tick up as the higher rates bite and the economy cools. **Inflation leads, the policy rate follows, and the labor market responds last** — that ordering is the whole transmission of the reaction function, and you can read it directly off this single chart.

### 2024: the slow glide to cuts

By 2024, the seesaw began to tilt back. Core PCE had fallen from 5.6% to roughly **2.6%–2.8%**, narrowing the inflation gap to under a point, while the labor market softened — unemployment drifted up from 3.4% toward **4.1%–4.2%**. Now the two sides of the mandate were no longer pointing the same way: inflation was still a touch above target (argues for holding tight), but the jobs side was weakening (argues for easing). This is the *judgment* regime, and the Fed moved cautiously, cutting the funds rate from 5.50% to **4.50%** across late 2024 — a deliberate, data-dependent glide rather than a rush.

The 2024 episode is the reaction function operating in its hardest mode, where the two mandate goals conflict and the Fed must weigh them. For a trader, this is where reading the *dots* and the *dispersion* pays off most: when the committee itself is split (wide dot dispersion) because the seesaw is near balance, the market's pricing swings hard on each data print, and the gap between market pricing and the median dot becomes the richest trade.

#### Worked example: the dual-mandate tradeoff in 2024

Let us quantify the judgment call the Fed faced in mid-to-late 2024 using the data. Core PCE had come down to about **2.7%**, so the inflation gap was:

- Inflation gap = 2.7% − 2.0% = **+0.7 points** — still above target, but a fraction of the +3.6 it had been.

Meanwhile, unemployment had risen from its **3.4%** low to around **4.1%–4.2%**, a move of roughly **+0.7 to +0.8 points** off the bottom. A rising unemployment rate, even from a low level, is historically a warning sign (the labor market rarely cools gently — it tends to break). So the seesaw in late 2024 had a *small* inflation weight (+0.7 gap, falling) against a *growing* employment weight (unemployment up ~0.8 and rising).

Which side wins? With the inflation gap small and shrinking but the jobs side deteriorating, the reaction function tilts toward **easing** — but cautiously, because inflation is not yet at target. That is exactly what the Fed delivered: 1.00 point of cuts (5.50% → 4.50%) spread over three meetings, not a panicked slashing. **When the inflation gap shrinks to under a point while unemployment starts rising, the reaction function rotates from "hold tight" to "begin easing" — and the Fed's measured 2024 cuts were the textbook output of a near-balanced seesaw.**

#### Worked example: the reaction function in dollars

Translate the gap into money, because the policy rate *is* the price of money. Suppose the reaction function says policy should sit roughly 2 percentage points higher than where it currently is, and the Fed gets there. A company carrying \$50,000,000 of floating-rate debt watches its annual interest bill climb by about \$1,000,000. A household refinancing a \$400,000 mortgage from a 4% rate to a 6% rate pays roughly \$480 more every month — about \$5,760 a year. The reaction function is not an academic curve; it is the dollar cost of the Fed's response to the inflation and jobs data, billed straight to every borrower. The lesson: once you can estimate the inflation and employment gaps, you can estimate the dollar squeeze headed for leveraged borrowers before the Fed says a word.

## How to trade it: the playbook

Everything above converges on a single, repeatable process. The edge is not knowing something secret; it is *modeling the reaction function yourself, off public data, and trading the gap between your model and what the market or the Fed is signaling.* Here is the playbook, step by step.

![A five-step playbook pipeline from tracking the gaps to taking the trade with an invalidation step](/imgs/blogs/inflation-and-the-fed-reaction-function-dot-plot-7.png)

**Step 1 — Track the two gaps, updated on every print.** Maintain two running numbers: the **inflation gap** (core PCE minus 2%) and the **employment gap** (unemployment versus the ~4.4% you treat as full employment, plus the *direction* it is moving). Update them the instant CPI, PCE, and the jobs report land. These two numbers are the state of the system. In 2022 the inflation gap was +3.6 and dominant; in late 2024 it had shrunk to +0.7 while the employment gap turned, flipping the regime.

**Step 2 — Run the rule to get an implied path.** Plug the gaps into a simplified Taylor rule to get a rough implied policy rate. You are not trying to nail the exact number; you are building *your model of the Fed* — what a by-the-book reaction function would prescribe given today's data. When your implied rate sits far above where the Fed actually is, expect tightening pressure; far below, expect easing pressure.

**Step 3 — Read the dots as the Fed's own forecast.** Each quarter, overlay the SEP median dot path onto your model. Note the direction of the shift versus last quarter, the longer-run dot (the Fed's neutral estimate), and especially the *dispersion*. The dots are the reaction function's published forecast of itself — a conditional one that will move with the data.

**Step 4 — Find the gap between the dots and market pricing.** Pull what SOFR futures imply for the funds-rate path and compare it to the median dots. The divergence is the heart of the trade. Markets pricing many more cuts than the dots, with inflation *not* cooperating, is a classic setup to fade the dovishness. Dots far above market pricing while data softens is the reverse.

**Step 5 — Take the position, with the right instruments.** The cleanest expression of a reaction-function view is in the **front end of the rate curve** — 2-year Treasury yields and SOFR futures, which are mechanically tied to the expected funds-rate path. A "Fed will cut less than priced" view is short front-end bonds (higher yields), usually dollar-positive and a headwind for gold and long-duration equities. A "Fed will cut more than priced" view is the mirror image. The front end is where the reaction function lives most directly; that is where to express the trade.

Why the front end specifically? Because of how the yield curve decomposes. A 2-year Treasury yield is, roughly, the market's average expected overnight rate over the next two years plus a small term premium. That makes it almost a *pure* bet on the path of the funds rate — which is exactly the output of the reaction function. The 10-year yield, by contrast, mixes in long-run growth expectations, the neutral rate, term premium, and inflation expectations, so it is a noisier expression of a near-term policy view. If your edge is "the Fed will react differently to the data than the market expects over the next year," the 2-year and SOFR futures are the instruments that isolate that edge. The shape of the curve itself — the gap between the 2-year and the 10-year — then tells you how the market is splitting "near-term policy" from "long-run neutral," which is its own rich signal but a separate trade.

**Invalidation — know what kills the view.** The reaction-function model breaks when the *inputs* shift regime: inflation **re-accelerates** (the gap reopens and the Fed is forced back to tightening, as the 2025–26 uptick in core PCE threatens), or the labor market **cracks hard** (a jobs report that breaks rather than cools, flipping the Fed to emergency easing). Either event resets the gaps abruptly, and your position must be sized and stopped so that a single regime-shifting print does not ruin you. Data-dependence cuts both ways: the same prints that prove your thesis can destroy it.

The deepest point is the one to leave with. The Fed is the most powerful price-setter in global markets, and it is *not* a black box. It is a function — a transparent, legally mandated, relentlessly telegraphed function that maps inflation and jobs onto a rate. Most market participants react to Fed decisions after the fact, treating each move as a surprise. The trader who has internalized the reaction function does the opposite: reads the same CPI and payrolls prints the Fed reads, runs the same gaps through the same logic, and forms a view on the policy path *before it is announced*. That anticipation — being positioned for the move the data has already made inevitable — is the entire edge. Model the function, watch the gaps, and trade the space between your model and the crowd's.

## Further reading & cross-links

- [Real vs Nominal: Inflation, Real Yields, and the Number That Moves Everything](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the inflation-adjusted version of any rate is what ultimately prices every asset.
- [Trading the FOMC: Statement, Presser, and Dot Plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) — the minute-by-minute mechanics of trading the releases this post describes.
- [Monetary Policy Transmission: How Rate Changes Reach Markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) — how a shifted rate expectation ripples through the curve, credit, the dollar, and equities.
- [How the Fed Sets Interest Rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the operational plumbing behind the policy rate the reaction function produces.
- [Paul Volcker and the 1980 Rate Shock: Killing Inflation](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the historical precedent for a central bank crushing entrenched inflation with brutal rate hikes.

---
title: "Inflation Expectations: Breakevens, Surveys, and Why They Move Markets"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What people expect inflation to be can matter more than what it actually is — how to read market-based breakevens and the 5y5y forward, survey gauges like UMich and the NY Fed, what anchored versus unanchored means, and why a survey spike can move the rate path more than a CPI print."
tags: ["event-trading", "macro", "inflation-expectations", "breakevens", "tips", "5y5y-forward", "umich-survey", "ny-fed-sce", "anchoring", "bonds", "fed"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — What people *expect* inflation to be can move markets more than what inflation actually *is*, because expectations drive wage demands, pricing decisions, and the entire bond market. The Fed's whole credibility rests on keeping those expectations "anchored" near 2%.
>
> - There are two families of expectation gauges: **market-based** (TIPS breakevens, the 5y5y forward — the inflation rate priced into bond yields) and **survey-based** (University of Michigan, NY Fed SCE, Conference Board — what households actually say).
> - The reaction map: an *unexpected* jump in a breakeven or a survey forces the market to reprice the Fed's path — and that can move 2y and 10y yields, rate-sensitive stocks, gold and crypto more than the underlying CPI print itself, because the print may already be priced.
> - The trade: watch the **expectation surprise**, not the level. A UMich reading that comes in far above what was expected is the kind of thing that has reportedly pushed the Fed into a larger hike — a survey, not a hard print, moving the rate path.
> - The one number to remember: in June 2022 the University of Michigan's preliminary 1-year inflation expectation printed near **5.3%**, and that single survey is widely reported to have tipped the Fed from a 50bp hike to a **75bp** hike days later.

In the second week of June 2022, the most important number in the world was not a CPI print. It was a survey. On Friday, June 10, the Bureau of Labor Statistics had already reported that May headline inflation hit 8.6% year-over-year — a fresh four-decade high, and itself a shock. But the number that reportedly changed the path of US monetary policy landed the *same morning*, buried in a release most people had never heard of: the University of Michigan's preliminary Survey of Consumers. Its measure of where ordinary households expected inflation to be in one year had jumped, and its longer-run 5-to-10-year reading ticked up to a preliminary 3.3% — the highest in over a decade.

That long-run number is the one the Federal Reserve watches like a hawk, because it is supposed to *never move*. If consumers expect inflation to average 2% over the next decade no matter what gas prices do this month, the Fed's credibility is intact and a temporary price spike will fade on its own. If that long-run expectation starts drifting up, it means people no longer believe the Fed will get inflation back to target — and once people stop believing, they start *acting* on the higher number: asking for bigger raises, accepting higher prices, building inflation into contracts. That is how a temporary shock becomes a permanent problem. Over that weekend, according to widely-reported accounts, Fed officials looked at that survey, decided anchoring itself was now at risk, and abandoned their guidance for a 50bp hike. On June 15, 2022, they raised rates by 75 basis points — the largest single hike since 1994.

A survey moved the rate path more than the CPI print did. That is the entire thesis of this post, and it is one of the least-intuitive truths in macro: markets do not just trade what inflation *is*. They trade what everyone *expects* it to be — because expectations are the thing that makes inflation self-fulfilling, and breaking that self-fulfilling loop is the Fed's only real job.

![Diagram of the inflation expectations loop showing expected inflation driving wage demands and pricing, which create actual inflation, which feeds back into expectations, with the Fed anchoring at the center](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-1.png)

The loop in the figure is the whole story in one picture. Expected inflation drives what workers demand in wages and what firms set as prices; those decisions create actual inflation; actual inflation teaches people to expect even more next time; and around it goes. The Fed sits in the middle with one lever — credibility — trying to anchor expectations so the loop never spins up. Everything else in this post is detail on how to *measure* where expectations sit, how to tell anchored from unanchored, and how a trader reads the surprise when one of those gauges jumps.

## Foundations: what inflation expectations are and how we measure them

Let me build this from zero, because the term "inflation expectations" gets used loosely and the precision matters enormously when you trade it.

**Inflation** is the rate at which the general price level rises — if a basket of goods cost \$100 last year and \$103 this year, that is 3% inflation. The Consumer Price Index (CPI) is the most-watched measure of *realized* inflation: it tells you what already happened to prices last month. (If you want the full mechanics of the CPI release and how markets react to it tick-by-tick, that is the subject of a separate post in this series; here we are after the forward-looking cousin.)

**Inflation expectations** are something different and stranger: they are a forecast — a number representing what some group of people *believe* inflation will be over some future window. There is no single "the" expectation; there are many, produced by different people through different mechanisms, and they disagree. The two big families are:

1. **Market-based measures**, derived from the prices of inflation-sensitive securities. The market is constantly betting real money on future inflation, and you can back out the implied forecast from bond prices. The headline market gauge is the **breakeven inflation rate**, and the most-watched refinement is the **5-year, 5-year forward** (the "5y5y").

2. **Survey-based measures**, where you literally ask people. The University of Michigan asks consumers; the New York Fed runs its Survey of Consumer Expectations (SCE); the Conference Board has its own; and there are surveys of professional forecasters (the Philadelphia Fed's SPF) and of businesses.

These two families measure different things and have different biases, which is exactly why a trader needs both. Let me define each term carefully.

### What a breakeven is

The US Treasury sells two kinds of bonds. **Nominal Treasuries** pay a fixed dollar coupon — a 10-year note yielding 4% pays you 4% in dollars regardless of what inflation does, so if inflation runs hot, your *real* (inflation-adjusted) return gets eaten away. **TIPS** — Treasury Inflation-Protected Securities — are different: their principal is adjusted upward with the CPI, so they pay you a fixed *real* yield plus whatever inflation turns out to be. A 10-year TIPS yielding 1.5% real gives you 1.5% *above* inflation, whatever inflation is.

Now line them up. If a 10-year nominal note yields 4.0% and a 10-year TIPS yields 1.5% real, the gap is 2.5 percentage points. That gap is the **breakeven inflation rate**: it is the average annual inflation rate over the next 10 years that would make an investor *indifferent* between the two bonds. If inflation averages exactly 2.5%, you "break even" — both bonds return the same. If inflation comes in higher than 2.5%, the TIPS wins; if lower, the nominal wins.

So the breakeven is the market's collective bet on average inflation over the bond's life, expressed as a number you can read off a screen every second. As I write the convention loosely: `breakeven = nominal yield − real (TIPS) yield`. A 2.5% 10-year breakeven means the bond market, with billions of dollars at stake, expects roughly 2.5% average inflation for a decade.

There is one honest complication you need to carry with you whenever you read a breakeven: it is *not* a pure inflation forecast. It is what the market will *pay* to insure against inflation, and that price embeds two extra premia. First, the **inflation risk premium** — investors will pay a bit extra for the protection a TIPS gives, because they fear inflation being higher than expected more than they hope for it being lower, which nudges the breakeven slightly above the true expectation. Second, and pulling the other way, a **liquidity premium**: TIPS trade in a much smaller, less liquid market than nominal Treasuries, so in a panic (March 2020 is the textbook case) investors dump TIPS and demand a higher real yield to hold them, which mechanically *depresses* the measured breakeven even though nobody's inflation view changed. The two premia partly offset, but the net effect is that the breakeven can sit perhaps 20–40bp away from the "true" expectation, and that gap widens exactly when markets are stressed. This is why the breakeven is a superb signal for *changes* (a 20bp jump on a release is real information) but an imperfect read on the *level* (a breakeven of 2.3% does not mean expectations are precisely 2.3%). A good trader watches the move, not the decimal.

### The 5-year, 5-year forward (5y5y)

The plain 10-year breakeven has a problem for the Fed: it blends near-term inflation (which bounces around with gas and food) with long-term inflation (which is what anchoring is about). If gas prices spike, the *front end* of the breakeven curve jumps even though nobody's long-run view changed. The fix is the **5y5y forward breakeven**: the inflation rate the market expects for the 5-year period that *starts 5 years from now*. It strips out the next five years of noise entirely and isolates the long-run anchor.

You build it from the breakeven curve. If the 10-year breakeven is 2.5% and the 5-year breakeven is 2.4%, then the inflation priced for years 6 through 10 must be roughly `(10 × 2.5% − 5 × 2.4%) / 5 = 2.6%`. That 2.6% is the 5y5y. The Fed and the ECB both treat the 5y5y as their single best market read on whether long-run expectations are anchored, because by construction it ignores the temporary stuff.

Why does the forward construction matter so much? Because a central banker does not care whether inflation runs hot for the next year — they expect to be able to handle a one-year overshoot. What they cannot tolerate is the market pricing inflation *above target a decade out*, because that says the market no longer believes the central bank will ever get back to 2% — it is a vote of no confidence in the institution. The 10-year breakeven blends those two horizons and so muddies the signal; the 5y5y cleanly isolates the "do you still believe us?" question. When Mario Draghi, then ECB president, gave his famous 2014 Jackson Hole speech warning that euro-area inflation expectations were slipping, the single number he pointed to was the 5y5y, which had fallen below 2% — that is how central a role this one construction plays in real policy. For a trader, the takeaway is hierarchy: a wobble in the 1-year breakeven is weather; a move in the 5y5y is climate, and climate is what reprices the entire long end of the curve.

### Survey-based measures: just ask people

Markets are smart but they are a narrow slice of the economy — bond traders, not the median household deciding whether to demand a raise. Surveys reach the people whose behavior actually creates inflation.

- **University of Michigan Surveys of Consumers (UMich):** a monthly telephone survey of US households, with a closely-watched preliminary reading mid-month and a final reading at month-end. It reports a 1-year-ahead inflation expectation and a 5-to-10-year-ahead one. It is the oldest and most market-moving consumer survey, partly because it comes out frequently and partly because the Fed has explicitly cited it.
- **New York Fed Survey of Consumer Expectations (SCE):** a monthly internet panel survey, methodologically more modern (it tracks the same people over time, which reduces noise). It reports 1-year and 3-year expectations and is generally considered the more statistically robust consumer survey.
- **Conference Board** and others add color but move markets less.
- **Survey of Professional Forecasters (SPF):** quarterly, surveys economists rather than households — useful as a sanity check on the "expert" view.

Surveys have well-known quirks. Consumers anchor heavily on the prices they see most often — gasoline above all — so the UMich 1-year number tracks gas prices almost mechanically and overshoots actual inflation most of the time. Survey expectations also show political tilt: respondents who dislike the sitting administration report higher expected inflation. None of that makes the surveys useless; it means you read them as a *behavioral* signal (are households starting to act inflationary?) rather than as a precise forecast.

### Anchored vs unanchored — the concept everything hinges on

Expectations are **anchored** when they barely respond to incoming inflation news — people expect ~2% over the long run no matter what this month's gas price did, because they trust the central bank to bring inflation back. Expectations are **unanchored** when they start chasing realized inflation — a price spike makes people expect more inflation, which makes them demand raises and accept higher prices, which creates more inflation.

Anchored expectations are the Fed's most valuable asset, because they make inflation *self-correcting*: a shock fades because nobody changes their behavior. Unanchored expectations are catastrophic because inflation becomes *self-sustaining*: it feeds on itself through the wage-price loop in the cover figure, and the only way to break it is a deep, deliberate recession — the 1980 Volcker shock, which took the policy rate near 20% and produced double-digit unemployment. That is why a single survey reading that suggests anchoring is slipping can terrify the Fed far more than a high CPI print that everyone expected.

### The self-fulfilling / wage-price channel

Why do expectations *cause* inflation rather than merely forecast it? Because of how people behave when they believe prices will rise. A worker who expects 5% inflation next year demands at least a 5% raise just to stand still — and if employers grant it, labor costs rise 5%, which firms pass into prices, producing 5% inflation. A landlord who expects 5% inflation raises the rent 5%. A firm that expects its inputs to cost 5% more pre-emptively marks up its own prices. Expectations are an instruction set the economy executes. Realized inflation is downstream of them. That is the deepest reason the Fed cares more about a drift in long-run expectations than about any one hot month: the month is history, but the expectation is a forecast the economy will *make come true*.

## Section 1 — Market-based gauges: breakevens and the 5y5y, and how to read them

The single most useful market-based chart for a macro trader is the path of the 10-year breakeven, because it tells you, in real time, whether the bond market thinks inflation is a permanent problem or a passing one.

![Line chart of 10-year breakeven inflation from 2020 to 2025 with the April 2022 peak near 2.95 percent and the 2 percent Fed target line](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-2.png)

Look at what the breakeven did through the worst inflation in 40 years. In March 2020, at the COVID crash, it collapsed to roughly 0.55% — the market briefly priced *deflation*. As stimulus and reopening hit, it climbed back through the 2% Fed target and kept rising, peaking around **2.95% in April 2022**. That peak is the headline: at the absolute height of the inflation scare, with realized CPI on its way to 9%, the *market's* 10-year inflation forecast never even broke 3%. Then it fell back, settling in a tight 2.1%–2.35% band from 2023 through 2025. The market, in other words, believed the Fed would win — it priced a temporary spike, not a regime change.

How do you read this in practice? A few rules:

- **Levels matter relative to target.** A 10-year breakeven near 2.0%–2.3% is "anchored at target." Above ~2.6% the bond market is starting to price persistent overshoot; below ~1.6% it is pricing a deflation/recession scare. The 0.55% reading in 2020 and the 2.95% reading in 2022 are the two extremes of the modern era.
- **The *change* on an event is the signal.** A breakeven that jumps 15–20bp on a CPI release is the bond market revising its long-run inflation view — a much bigger deal than the same-day equity wiggle.
- **Decompose nominal yields.** Any nominal Treasury yield splits into a real yield plus a breakeven: `nominal = real + breakeven`. When the 10-year yield rises, you *must* ask which piece moved. If the breakeven rose, the market is pricing more inflation (bad for risk, the Fed has to fight). If the real yield rose, the market is pricing tighter policy or stronger growth — a different animal entirely. The companion macro post on real versus nominal yields treats this decomposition as the master signal, and it is exactly right to.

Let me put real money on a breakeven move, because percentages are abstract and dollars are not.

#### Worked example: a breakeven jump repricing a 10-year bond position

You are long \$500,000 face value of the 10-year Treasury. A hot inflation surprise pushes the 10-year breakeven up by 20bp, and because the move is an inflation-expectation move (not a real-yield move), the nominal 10-year yield rises by the full 20bp. Yields up means bond prices down.

The sensitivity of a bond's price to a 1bp yield change is its **DV01** (dollar value of an 01). For a 10-year note, DV01 is roughly \$0.086 per \$100 face, so on \$500,000 face:

- DV01 ≈ \$500,000 × 0.086 / 100 ≈ **\$430 per basis point**.
- A +20bp yield move costs ≈ \$430 × 20 = **−\$8,600** on the position.

So that one breakeven repricing — driven by an inflation expectation, not by realized CPI — just cost a long bondholder **\$8,600**. The lesson: a breakeven is not an abstract forecast; it is the thing that turns into your bond P&L the instant expectations move.

#### Worked example: decomposing a nominal yield into real and breakeven

The 10-year nominal Treasury yields 4.0% and the 10-year breakeven is 2.5%. The real (TIPS) yield is therefore `4.0% − 2.5% = 1.5%`. On a \$100,000 position held for a year, the pieces are:

- Nominal return ≈ \$100,000 × 4.0% = **\$4,000** in dollars.
- Expected inflation eats ≈ \$100,000 × 2.5% = **\$2,500** of purchasing power.
- Expected *real* return ≈ \$100,000 × 1.5% = **\$1,500** — the actual gain in what you can buy.

If a survey or a hot CPI then drives the breakeven from 2.5% to 3.0% while the real yield is unchanged, the nominal yield rises to 4.5% — and a trader who watched only the headline 4.5% would miss that the entire move was the inflation-expectation component repricing, which is the signal that matters.

There is a mirror-image lesson on the downside, and it is one most people forget because we have spent recent years fearing high inflation. In March 2020, as COVID shut the economy, the 10-year breakeven collapsed to roughly **0.55%** — the bond market was pricing barely any inflation at all, and the front end of the curve briefly priced outright deflation (falling prices). That collapse was partly real expectation (a demand shock crushes prices) and partly the liquidity-premium effect described above (a fire-sale in TIPS dragged the measured breakeven down). Either way, it was a screaming signal of the *opposite* problem: when expected inflation falls far below target, the Fed's nightmare is not a wage-price spiral but a deflationary trap, where consumers delay purchases because they expect things to be cheaper later, demand falls further, and prices fall again — the loop in the cover figure spinning in reverse. The Fed's response was the largest stimulus in history precisely to re-anchor expectations *up* toward 2%. So the breakeven is a two-sided thermometer: too far above target warns of a wage-price spiral, too far below warns of a deflation trap, and the target zone of roughly 2.0%–2.5% is "anchoring intact." A trader who only ever fears the high side is reading half the instrument.

One more reading nuance: the *TIPS market itself* is something you can trade directly, and it expresses a pure inflation-expectation view. If you think realized inflation will beat the breakeven, you buy TIPS and short an equal-duration nominal Treasury — a "breakeven trade" that profits if inflation comes in above the priced-in rate, with the level of real rates hedged out. That trade *is* a bet on inflation expectations, isolated from everything else, and its existence is the reason the breakeven is such a clean signal: real money is constantly arbitraging it toward the market's best inflation guess.

#### Worked example: a pure breakeven (TIPS vs nominal) trade

You believe inflation will beat the priced-in 2.5% 10-year breakeven, so you put on the pure expression: buy \$1,000,000 of 10-year TIPS and short \$1,000,000 of duration-matched 10-year nominal Treasuries. The real-rate exposure cancels; you are left long the breakeven. Suppose a UMich shock and a hot CPI together push the 10-year breakeven up 15bp, from 2.50% to 2.65%, with real yields flat.

- The position's breakeven DV01 is roughly \$860 per basis point on \$1,000,000 of 10-year exposure, so a +15bp move is ≈ \$860 × 15 = **+\$12,900**.
- Because real rates were flat, the nominal short and the TIPS long offset on the real-yield leg — the entire **+\$12,900** is the inflation-expectation repricing you bet on.
- Had the breakeven instead *fallen* 15bp on a disinflation surprise, the same position would lose **−\$12,900**.

The intuition: you isolated the inflation expectation and turned a 15bp shift in what the market *expects* into nearly \$13,000 of P&L — with the level of real interest rates hedged completely out.

## Section 2 — Survey-based gauges: UMich and the NY Fed, and their quirks

Now the other family. The market-based gauges are forward-priced and disciplined by real money, but they live in the bond market. The survey gauges reach the households whose wage and spending decisions actually drive the wage-price loop — so when the Fed worries about anchoring *behavior*, it looks at surveys.

![Line chart of University of Michigan one-year inflation expectations from 2021 to 2025 with the 2022 peak near 5.4 percent and a 2025 tariff-driven spike annotated](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-3.png)

The UMich 1-year expectation tells a far more dramatic story than the breakeven did. It ran near 3% in early 2021, climbed to a **5.4%** peak in early 2022, and the June-2022 reading sat near 5.3% — the print that reportedly tipped the Fed to 75bp. It then drifted back toward 3% through 2023–24… and then *spiked again* to **6.5% in April 2025**, this time driven by tariff fears, before settling back near 5% in June. Notice how much more volatile this consumer series is than the bond-market breakeven: consumers' fear lurches; the market's priced inflation barely budges.

That contrast is the most important thing in this post, so I will make it explicit. The reasons the two diverge:

- **Surveys overweight what you buy weekly.** Gasoline and groceries dominate a consumer's perception, so the UMich 1-year number tracks gas prices closely and routinely overshoots realized inflation. In April 2025 the spike was tariff-anxiety, not a measured price change.
- **Surveys carry behavioral and political noise.** The same survey can swing several points across an election as partisan sentiment flips. Treat the *trend* and the *surprise*, not the absolute level.
- **The 5y5y survey number is the anchoring tell.** The UMich 1-year reading bounces with gas; the 5-to-10-year reading is supposed to be flat. The June-2022 episode mattered because the *long-run* survey reading ticked up to a preliminary 3.3%, which is what scared the Fed. (It was later revised down — a separate cautionary tale about reacting to preliminary survey data.)

So how do you use the two families together? Read the figure below as your cheat sheet.

![Comparison matrix of four inflation expectation gauges - 10y breakeven, 5y5y forward, UMich survey, NY Fed SCE - showing what each measures, its source, and its known bias](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-4.png)

The matrix lays out the four gauges you actually trade off. The two market gauges (the 10-year breakeven and the 5y5y) are forward-priced by real money but carry a **liquidity and risk premium** — the breakeven is not a *pure* expectation, because TIPS are less liquid than nominal Treasuries and demand a premium, which distorts the implied number by perhaps 20–40bp. The 5y5y is the Fed's favorite because it strips out near-term noise and gets closest to the pure long-run anchor. The two survey gauges (UMich, NY Fed SCE) are raw household sentiment: they run high, they jump, and the NY Fed's panel design makes it steadier than UMich's — but both sit above the market gauges most months. A trader who knows each gauge's bias does not panic when UMich prints 5% while breakevens sit at 2.4%; that gap is *normal*. The panic signal is when the gauges that are *supposed* to be stable — the 5y5y and the survey long-run readings — start to move together.

It is worth knowing the methodological differences that produce these biases, because they tell you which gauge to trust in a given situation. The **UMich** survey is a repeated *cross-section*: each month it phones a fresh sample of households, so the month-to-month reading is noisy — part of any move is just a different set of people answering. It also asks an open-ended question ("what do you expect prices to do?") and respondents anchor hard on the most salient price they see, which is almost always gasoline. That is why the UMich 1-year number is the jumpiest gauge and the one most likely to fake you out. The **NY Fed SCE**, by contrast, is a *rotating panel*: it follows the same households for twelve months before rotating them out, so it can measure how a *given person* changes their view, which strips out a lot of the sampling noise. Its design also uses a probabilistic question format (asking respondents to assign probabilities to inflation ranges) that produces cleaner aggregate statistics. The cost is timeliness — the SCE comes out later in the month than UMich's closely-watched preliminary — so UMich moves markets *first* even though SCE is the better number. The **Conference Board** survey is watched mainly for consumer confidence rather than inflation, and the **Survey of Professional Forecasters** (quarterly, economists rather than households) is the slow, expert benchmark you check to see whether the pros agree with what households fear. The practical hierarchy for a trader: UMich for the market-moving *surprise*, NY Fed SCE for the *truth*, SPF and the market gauges for the *anchoring sanity check*.

One more quirk that catches people out: the **preliminary versus final** distinction. UMich releases a preliminary reading mid-month and a final reading near month-end, and the two can differ — sometimes materially. The June-2022 episode turns on exactly this: the Fed reacted to the *preliminary* long-run reading of 3.3%, which was later revised down to 3.1% in the final. So a violent market reaction to a UMich preliminary carries built-in revision risk; part of the discipline of trading these is sizing for the chance that the number you reacted to gets walked back two weeks later.

#### Worked example: a UMich spike repricing the Fed path into an equity book

You run a \$25,000 book of rate-sensitive growth equities. A UMich preliminary print comes in far above expectations — the kind of surprise that pushes the market to price an extra 25bp of Fed tightening over the next two meetings. Rate-sensitive equities fall on a higher discount rate; say the repricing knocks 2% off your book on the day.

- Day-one mark: \$25,000 × −2% = **−\$500**.
- The trigger was a *survey*, not a CPI print — the hard inflation data did not change that morning.
- If the surprise were larger and priced 50bp of extra tightening, a −4% day would be **−\$1,000** on the same book.

The intuition: a consumer survey, by changing the *expected* Fed path, reached into an equity book and took \$500 — without a single hard inflation number being released.

## Section 3 — Anchored vs unanchored: the Fed's nightmare

Everything so far has been measurement. This section is the *why it matters*, and it is the concept the Fed loses sleep over.

![Two-column diagram contrasting anchored and unanchored expectations after the same oil shock - anchored expectations fade the shock while unanchored expectations spiral into a wage-price loop](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-6.png)

The figure runs the same shock — an oil-price spike that pushes headline CPI up — through two worlds. In the **anchored** world (left), expectations stay put because people trust the 2% target; there is no rush for big raises, firms hold their list prices, and the shock fades within about a year as oil normalizes. Inflation returns to 2% on its own, and the Fed barely has to act. In the **unanchored** world (right), the same shock makes people expect 5%+ next year; they demand and receive big raises, those raises feed costs, costs feed more raises — the wage-price spiral — and inflation gets *embedded* in the structure of the economy. Now the only cure is a Volcker-scale recession: deliberately crushing demand until people stop expecting inflation.

This is why "anchoring" is not central-bank jargon for jargon's sake. The difference between the two columns is the difference between a Fed that cuts a few times and moves on, and a Fed that has to engineer mass unemployment. The entire institution's credibility — its ability to keep the left column from becoming the right — is its only tool, and that credibility *is* anchored expectations. Lose it and no amount of rate hikes works cheaply.

Two historical anchors (pun intended) make this concrete. In the 1970s, US expectations *unanchored*: a sequence of oil shocks — the 1973 OPEC embargo and the 1979 Iranian-revolution shock — taught Americans to expect persistent inflation. Once that belief set in, it became structural: labor contracts contained automatic cost-of-living adjustments (COLAs) that hard-wired expected inflation into wages; firms raised prices pre-emptively; and each new shock ratcheted expectations higher rather than fading. Inflation that started as an energy-price spike metastasized into a decade-long, self-sustaining problem peaking near 14% in 1980. Breaking it required Paul Volcker to push the funds rate toward 20%, which triggered the deepest recession since the Depression — unemployment near 11% — explicitly *in order to* convince people that the Fed would do whatever it took, and thereby re-anchor expectations. The recession was not a side effect; re-anchoring was the *point*. (The dedicated post on the Volcker shock walks through that episode in full.)

In 2021–23, by contrast, despite the worst *realized* inflation since the 1970s, long-run expectations stayed *anchored* — the 5y5y never broke far above target, the 10-year breakeven peaked under 3% — and the Fed brought inflation down with hikes that were painful but far short of Volcker's, and crucially without a deep recession. Why the different outcome from similar-sized shocks? Forty years of credibility. The post-Volcker Fed had spent decades demonstrating it would defend 2%, so when COVID-era inflation hit, households and markets largely assumed it was temporary — and that assumption made it temporary, because nobody locked in 1970s-style behavior. There were no widespread COLA contracts; the wage-price spiral never caught. The 2020s were not the 1970s precisely because expectations held, and the reason they held is that the Volcker generation had paid the price to anchor them. That is the Fed's win, and you can see it in the data.

![Two-line chart of 10-year breakevens versus actual headline CPI from 2020 to 2025 - CPI spikes to 9 percent while breakevens stay near 2.5 percent](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-5.png)

This is the single most important chart in the post. The amber line is *realized* inflation — headline CPI — which screamed to 9.1% in mid-2022. The blue line is *expected* inflation — the 10-year breakeven — which, through that entire spike, stayed glued near 2.5%. Realized inflation went to the moon; expected inflation shrugged. That gap *is* anchoring. The market looked at 9% CPI and said, in effect, "temporary — the Fed will handle it," and priced the next decade at roughly 2.5%. Had the blue line chased the amber line up toward 5% or 6%, the Fed would have been in the 1970s, and the policy response would have been catastrophic rather than merely painful.

#### Worked example: what unanchoring would do to a long-dated bond position

Suppose expectations *had* unanchored — the 10-year breakeven jumping from an anchored 2.5% to a 1970s-style 5.0%, a +2.5 percentage point (250bp) shift in priced inflation. You hold \$1,000,000 face of a 30-year Treasury, where the long maturity makes price brutally sensitive to yield. A 30-year bond's duration is roughly 18–20 years, so a 1% (100bp) rise in yield costs roughly 18–20% of the position's value.

- A +250bp inflation-expectation repricing, passed straight into the nominal yield, is ≈ 2.5 × 19% ≈ a **47% price decline**.
- On \$1,000,000 of 30-year exposure that is roughly **−\$470,000** — nearly half the position wiped out.

Compare that to the \$8,600 hit from the 20bp breakeven move on the 10-year earlier: unanchoring is not a bigger version of a normal move, it is a different magnitude of disaster, concentrated in the longest-dated assets. That asymmetry is exactly why the Fed treats a drift in *long-run* expectations as an emergency.

## Section 4 — Why expectations move markets: the bond and policy channel

We have the gauges and the concept. Now the mechanism that connects a survey or breakeven jump to a same-day market move — because this is where the event-trading edge lives.

The chain has two links. First, an expectations gauge surprises to the upside. Second, the market reasons: "if expectations are drifting, the Fed *must* respond more aggressively to defend anchoring" — and it reprices the expected Fed path, pulling forward hikes (or pushing back cuts). That repriced path is what moves every asset:

- **Bonds (the direct channel):** a higher expected Fed path lifts short-term yields immediately (the 2-year, which tracks the expected path of the funds rate over the next two years, is the most sensitive). Longer yields rise too, partly on the path and partly on a higher inflation premium. Yields up means bond prices down.
- **Equities (the discount-rate channel):** stocks are the present value of future earnings discounted at a rate tied to bond yields. Higher expected rates raise the discount rate and compress valuations — and the longest-duration equities (unprofitable growth, tech) fall hardest. This is the channel that produces "good economic news is bad for stocks" in an inflation regime, which the companion post on the reaction function unpacks fully.
- **The dollar (the rate-differential channel):** higher expected US rates pull capital toward dollar assets, so the dollar tends to firm on a hawkish expectations surprise.
- **Gold and crypto (the real-rate channel):** gold and Bitcoin pay no yield, so they compete with real (inflation-adjusted) returns on bonds. A hawkish surprise that raises *real* yields is a headwind; but if the surprise is purely an *inflation*-expectation jump that the market thinks the Fed will be slow to counter, gold can actually rally as an inflation hedge. The sign depends entirely on whether real yields or breakevens are doing the moving — which is why decomposing the move is everything.

Crucially, the size of the reaction scales with the *surprise*, not the level. If a breakeven jump or a survey print merely confirms what was already priced, markets barely move. If it forces a genuine repricing of the Fed path, the move can dwarf the reaction to a CPI print that came in exactly on consensus. A consensus CPI print is a non-event; an off-consensus expectations gauge is the trade.

The microstructure of the move is worth slowing down on, because it tells you *when* to trade. The reaction unfolds in a sequence. First, the **2-year yield** moves almost instantly, because it is the most direct embodiment of the expected Fed path over the relevant horizon — algorithms reprice it within milliseconds of the headline. Then the move propagates: the 10-year follows (path plus inflation premium), equity index futures reprice off the higher discount rate, the dollar firms, and gold and crypto sort themselves out depending on whether the move was real-yield-led or breakeven-led. The whole cascade can complete in seconds for the liquid futures and minutes for the cash equity open. A trader who understands the sequence does not chase the 2-year — it has already moved — but instead trades the slower-adjusting legs (rate-sensitive single stocks, the equity open, the FX cross that lags) where the repricing is still flowing through.

This sequencing also explains a recurring pattern: the **knee-jerk and the fade**. The first move is the algorithmic reaction to the surprise. But survey readings especially are noisy and get revised, so once human discretion enters — once desks ask "is this a real anchoring signal or just a gas-price artifact?" — the move often partly reverses. The cleanest expression of a high-conviction trade is therefore not the knee-jerk but the *confirmation*: position after the dust settles if the anchoring gauges (the 5y5y, the survey long-run reading) actually moved, not just the gas-driven 1-year number.

A note on scope, because this series spans more than the US. The mechanism is global. The European Central Bank watches its own 5y5y inflation swap obsessively (Draghi's 2014 speech is the canonical example). Japan spent two decades fighting *unanchored-to-the-downside* expectations — households so conditioned to expect flat or falling prices that the Bank of Japan struggled for years to lift inflation toward 2%, the deflation-trap version of the problem. And in Vietnam, where the State Bank manages the dong and a credit ceiling rather than a Western-style inflation target, expectations transmit through a different but related channel: when households expect the dong to weaken or prices to rise, they shift savings into gold and US dollars, which pressures the currency and forces the State Bank to defend it — exactly what happened in autumn 2022 when the SBV hiked its refinancing rate from 4.0% to 6.0% to defend the dong. The instrument differs; the logic — expectations driving behavior driving the policy response — is universal.

## Section 5 — How a trader uses them: the expectation surprise

Here is the operating manual: trade the *surprise* in the expectations gauge, then trade the *repricing of the Fed path* that follows.

![Flow diagram of a trader reading an expectations surprise - a survey or breakeven jump reprices the expected Fed path and moves bonds and risk assets more than a CPI print](/imgs/blogs/inflation-expectations-breakevens-surveys-and-why-they-matter-7.png)

The figure is the playbook in one picture. An expectations surprise — UMich jumping from 4.4% toward 6.5%, or the 10-year breakeven popping +20bp versus what was priced — hits the tape. The Fed reads it as anchoring at risk (credibility is the whole ballgame), so the market reprices the Fed path toward more or faster hikes, pricing in an extra 25–50bp. That repricing sells bonds (2-year and 10-year yields jump — DV01 turns straight into P&L) and drags risk assets down (higher discount rate hits rate-sensitive equities and crypto). And the contrast box at the bottom-left is the punchline: a CPI print that comes in *in line* with consensus moves almost nothing, because it was already in the price. The expectations surprise is the trade precisely because it is the part that was *not* priced.

The practical workflow:

1. **Know what's priced before the release.** Read the consensus for the survey (Bloomberg/Reuters carry a median), and watch the 2-year yield and Fed-funds futures for the priced Fed path. The companion post on consensus and "priced in" is the deep dive here.
2. **On the release, compute the surprise.** Actual minus expected for the survey; the change in the breakeven for the market gauge. A large *long-run* surprise (the 5y5y or the survey 5-10y reading) is far more dangerous than a 1-year-number surprise that's just gas prices.
3. **Trade the path repricing, not the gauge.** The cleanest expression is in rates — short the front end (sell 2-year, or pay in swaps) on a hawkish expectations surprise — and in rate-sensitive equities. Fade the gauge level; trade the path change.
4. **Respect the fade.** Survey preliminary readings get revised (June 2022's long-run number was later cut). If the move was driven by a preliminary print that smells like a gas-price artifact, the knee-jerk often fades. The anatomy-of-a-reaction post in this series covers the spike-fade-trend pattern in detail.

#### Worked example: sizing the front-end trade off an expectations surprise

A UMich long-run reading surprises hawkish, and the market prices an extra 25bp of tightening into the next year — the 2-year yield jumps 18bp. You are short \$1,000,000 face of the 2-year Treasury (a bet that yields rise / prices fall), which has a DV01 of roughly \$190 per basis point.

- Gain on the short ≈ \$190 × 18bp = **+\$3,420** as the 2-year sells off.
- A long-only bond holder of the same \$1,000,000 2-year would lose that **\$3,420**.
- Had you instead been long \$500,000 of the 10-year (DV01 ≈ \$430/bp) and the 10-year rose 12bp on the same surprise, you'd be down \$430 × 12 = **−\$5,160** — the long-duration position bleeds more from the same expectations shift.

The intuition: a survey reading translated, within minutes, into thousands of dollars of bond P&L through nothing but a repricing of the *expected* Fed path — the print itself never moved.

## How it reacted: real episodes

Time to put dates and numbers on it. Three episodes show expectations driving markets — and one of them is the clearest case on record of a survey, not a hard print, moving the rate path.

### April 2022 — the breakeven peak that *didn't* break

By April 2022, realized CPI was over 8% and climbing toward its 9.1% June peak. The 10-year breakeven topped out around **2.95%** — its highest of the cycle, and a genuine warning that the market was starting to price persistent overshoot. But notice the framing: even at the worst of the worst inflation in four decades, the market's 10-year inflation forecast *never crossed 3%*. The breakeven peak was a stress signal, not an unanchoring. The bond market was nervous but still believed the Fed would win — and it was right. Within months the breakeven was back below 2.4%, where it has stayed. A trader watching the breakeven in April 2022 learned the most useful lesson available: realized inflation can be terrifying while expected inflation stays under control, and it is the second number that tells you whether you are in 2022 (painful, recoverable) or 1979 (catastrophic).

### June 2022 — the survey that pushed the Fed to 75bp

This is the marquee episode and the reason this post exists. Friday, June 10, 2022: May CPI prints 8.6% — a hot, four-decade-high number that itself rattled markets. But the same morning, the University of Michigan's preliminary June survey shows the 1-year inflation expectation near 5.3% and, critically, the 5-to-10-year reading ticking up to a preliminary 3.3% — the highest since 2008. The Fed had guided markets toward a 50bp hike at the June 15 meeting. Over that weekend, according to widely-reported accounts (the *Wall Street Journal* report that signaled the shift is itself a famous moment), policymakers looked at that long-run survey number, judged that anchoring itself was now at risk, and changed course. On June 15 they hiked **75bp** — the biggest move since 1994 — and Powell explicitly cited the deterioration in survey-based expectations as a reason. The S&P 500 had already fallen sharply that week (the June 13 session was brutal, down over 3.8% as the market sniffed out the bigger hike). The signal that flipped the policy path was a *survey*. Not the CPI print, which was expected to be hot — the survey, which surprised. (The long-run reading was later revised down to 3.1%, which is its own lesson: the Fed acted on a preliminary number that turned out to be partly noise.)

### 2025 — the tariff-driven survey spike

Fast-forward to spring 2025. New tariff announcements sent the UMich 1-year inflation expectation rocketing to **6.5% in April** — higher than even the 2022 peak — before easing to around 5% in June. This time the divergence from market gauges was stark: while the survey screamed 6.5%, the 10-year breakeven stayed near 2.3%, barely above target. The market read the tariff shock as a one-off price-level adjustment, not a regime change; consumers, who feel tariff-driven price hikes directly at the register, panicked. A trader who only watched the UMich number would have braced for a 1970s rerun; a trader who watched *both* gauges saw a classic survey-overshoot driven by a visible price shock, with anchoring intact in the bond market. The episode is the cleanest recent illustration of why you never trade one gauge in isolation.

The 2025 case also teaches the *political-tilt* lesson in real time. Tariffs are a tax households can see directly — a more expensive imported good at the register is the most salient price signal there is — so the consumer survey reacted violently while the cooler, money-weighted bond market shrugged. The right read was to treat the survey spike as a one-off level adjustment (a tariff raises the price *level* once; it is not ongoing inflation unless it feeds expectations), and to watch the 5y5y and the breakeven for any sign the level shock was contaminating long-run expectations. It was not — the breakeven held near 2.3% — so the trade was to fade the survey-driven panic rather than position for an anchoring crisis. The lesson generalizes: a visible, one-time price shock (a tariff, a gas spike, a supply disruption) will always spook the surveys; only if it bleeds into the *long-run* anchoring gauges is it a trade.

## Common misconceptions

**"Only the actual CPI matters — expectations are soft data."** This is the big one, and it is backwards. Realized CPI is *history* — it tells you what already happened. Expectations are a *forecast the economy will try to make come true* through the wage-price loop. The June-2022 episode is the proof: a hard CPI print that was expected to be hot moved the Fed less than a survey of expectations that surprised. The Fed's mandate is, in its own framing, fundamentally about keeping expectations anchored; the realized print is a symptom, the expectation is the disease. A trader who ignores expectations is reading yesterday's news.

**"Breakevens and surveys should agree."** They almost never do, and the gap is information, not error. Surveys run structurally higher than market breakevens — the UMich 1-year number can sit at 5% while the 10-year breakeven sits at 2.4%, and that is *normal*, because surveys overweight gas prices and carry behavioral bias while breakevens are disciplined by real money (and carry their own liquidity premium). The signal is not the level of either; it is when the gauges that are *supposed* to be stable — the 5y5y and the survey long-run readings — start moving together.

**"A high inflation expectation always crashes stocks."** Only the *surprise* relative to what's priced moves markets, and the sign depends on the regime. In the 2022 inflation regime, a hawkish expectations surprise crashed stocks via the discount-rate channel. But a confirmed-in-line expectations number is a non-event, and in a different regime — say one where the market fears deflation — a *rise* in expectations toward target could be read as good news. Always ask: what was priced, and what does the market care about *now*? The reaction function determines the sign.

**"The 5y5y and the 1-year UMich are the same kind of number."** Dangerously not. The 1-year UMich is volatile, gas-driven, and overshoots — it can spike to 6.5% on a tariff headline and tell you almost nothing about anchoring. The 5y5y forward and the survey 5-to-10-year readings are the anchoring gauges; they are *supposed* to be flat, so a move in *them* is the real alarm. Reacting to a 1-year survey spike as if it were an anchoring crisis is how you get faked out of position.

**"If expectations unanchor, the Fed can just hike a bit more."** This underrates the asymmetry. Re-anchoring expectations once they've slipped is enormously more expensive than keeping them anchored — the 1970s needed Volcker's near-20% rates and a deep recession, while the 2020s, with anchoring intact, needed hikes that were painful but a fraction as severe. The long-dated-bond worked example showed why: unanchoring is a different *magnitude* of disaster, not a bigger version of a normal move.

**"The breakeven is the market's exact inflation forecast."** No — it is the inflation rate that makes a TIPS and a nominal bond break even, and it bakes in an inflation risk premium (nudges it up) and a liquidity premium (drags it down, especially in a panic). The net distortion can be 20–40bp, and it *widens* exactly when markets are stressed and you most want a clean read. That is why every rule in this post is framed around the *change* in the breakeven on an event, not its absolute decimal. A breakeven that jumps 20bp is unambiguous information; a breakeven sitting at 2.31% versus 2.28% is within the premium noise.

**"A consumer survey can't possibly move the Fed."** June 2022 is the standing refutation: a preliminary UMich survey is exactly what is widely reported to have tipped the FOMC from a 50bp hike to a 75bp hike. The Fed reacts to surveys because surveys measure the *behavioral* risk — whether households are starting to act inflationary — that the realized CPI cannot see until it is too late. Dismissing the surveys as soft data is the most expensive mistake a macro trader can make in an inflation regime.

## The playbook: how to trade it

Here is the if-then map for trading inflation-expectation events.

**Before the release:**
- Pull the consensus for the survey (UMich preliminary mid-month; NY Fed SCE monthly) and note where the priced Fed path sits — read it off the 2-year yield and Fed-funds futures.
- Separate the gauges by importance: a surprise in a *long-run* anchoring gauge (5y5y, survey 5-10y) is a major event; a surprise in a 1-year survey number is usually a gas-price artifact and fades.
- Set your scenarios: in-line (non-event), hawkish surprise (expectations jump), dovish surprise (expectations fall back faster than priced).

**On a hawkish expectations surprise (the gauge jumps, anchoring looks at risk):**
- *Rates:* the cleanest trade. Short the front end — sell the 2-year, or pay fixed in swaps — because the market reprices the Fed path higher. DV01 turns the yield move into P&L (≈ \$190/bp on \$1,000,000 of 2-year).
- *Equities:* fade rate-sensitive growth and small caps (the longest-duration equities fall hardest on a higher discount rate).
- *FX:* the dollar tends to firm on higher expected US rates.
- *Gold/crypto:* sign depends on real-yield decomposition — headwind if real yields rise, possible tailwind if it's a pure inflation-premium move the Fed seems slow to counter.
- *Invalidation:* if the breakeven *didn't* move alongside the survey (as in April 2025), the bond market is calling it a one-off — don't position for an anchoring crisis off a gas-driven survey spike.

**On a dovish expectations surprise (expectations fall faster than priced):**
- Reverse it: rally in bonds (buy the front end), relief in rate-sensitive equities, softer dollar. This is the "disinflation is winning" trade.

**On an in-line print:** stand aside or fade the knee-jerk. There is no edge in trading a number that was already priced; the move, if any, is noise that reverts.

**Sizing and risk around the event:**
- Size for the *surprise distribution*, not the central case — expectations gauges can gap, especially survey preliminaries.
- Respect the revision risk: survey preliminary readings get revised (June 2022's long-run number was cut from 3.3% to 3.1%), so a violent reaction to a preliminary print often fades. Have a plan for the fade as well as the trend.
- Cross-check the two families: never size a big position off one gauge in isolation. The trade you can hold is the one where the *anchoring* gauges (5y5y, survey long-run) and the *market* gauges agree.

The meta-rule for the whole post: trade the expectation surprise, not the realized print — because what people expect inflation to be is the thing that makes inflation real, and the bond market and the Fed both price the expectation first.

## Further reading & cross-links

- [Real vs nominal: inflation, real yields, and the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the decomposition of nominal yields into real yields plus breakevens, treated as the single most important macro signal.
- [Inflation and the Fed reaction function: the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — how the Fed translates inflation and expectations into the policy path you trade against.
- [Reading the yield curve: slope, inversion, and recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how breakevens and real yields sit inside the broader curve, and what the slope is telling you.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the institutional mechanics behind the rate path that expectations reprice.

Within this series, the companion posts on consensus and "priced in," on the reaction function, and on the anatomy of a news reaction (spike-fade-trend) build directly on the surprise-trading framework used here.

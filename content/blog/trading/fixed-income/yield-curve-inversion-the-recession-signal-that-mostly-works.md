---
title: "Yield curve inversion: the recession signal that mostly works"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What it means when short-term yields rise above long-term yields, why an inverted yield curve has preceded almost every US recession, the lead time involved, and why it is a probability rather than a promise."
tags: ["fixed-income", "bonds", "yield-curve", "yield-curve-inversion", "recession", "2s10s", "treasuries", "monetary-policy", "term-premium"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — a yield curve "inverts" when short-term interest rates rise above long-term rates, and that upside-down shape has preceded almost every US recession of the last sixty years, which is why it is the single most-watched recession signal in finance.
> - The curve normally slopes **up**: you earn more for lending money longer. An **inversion** flips that, so a 2-year Treasury yields *more* than a 10-year one.
> - It happens because the **front end** of the curve tracks what the Fed is doing now (hiking), while the **long end** prices what the market expects later (cuts, as growth slows). When those two forces pull hard enough in opposite directions, short yields end up above long yields.
> - The two measures everyone watches are the **2s10s** (10-year minus 2-year) and the **3m10y** (10-year minus 3-month). The Fed's own research favors the **3m10y**; traders quote the 2s10s.
> - The track record is genuinely strong — it has led most US recessions — but the **lag is long and variable** (roughly 6 to 22 months), and there has been at least one false alarm. The 2022–2023 inversion was the deepest in four decades and, for a long time, did *not* deliver a recession on schedule.
> - Treat it as a **probability gauge**, not a countdown clock: an inverted curve says recession risk is elevated, not that one is scheduled.

Imagine a bank offers you two ways to lock up \$10,000. Lend it for two years and earn 4.8% a year. Lend it for ten years and earn 4.3% a year. That should feel wrong. You are being asked to tie your money up *five times longer* — through five more years of unknown inflation, unknown crises, unknown everything — and you get paid *less* for it. Why would anyone do that?

When the bond market does exactly this — when it pays you less to lend long than to lend short — it is sending one of the loudest signals in all of finance. The shape it makes is called an **inverted yield curve**, and over the last six decades it has shown up, like a recurring character, in the months before almost every US recession. It inverted before 1980, before 1990, before 2001, before 2008, before 2020. Each time, a recession followed. That track record is why economists, central bankers, fund managers, and financial journalists watch this one number — the gap between long and short yields — more closely than almost anything else.

![A normal yield curve sloping upward beside an inverted yield curve sloping downward, with yield on the vertical axis and maturity on the horizontal axis](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-1.png)

The diagram above is the mental model to carry through the whole post. On the left is the *normal* curve: short maturities on the lower left, long maturities on the upper right, yields climbing as you move out in time — the everyday shape of a healthy economy. On the right is the *inverted* curve: the same axes, but the line slopes *down*, because the short end now sits above the long end. That downward slope is the warning. The rest of this post is about why that shape forms, what it has actually predicted, how long the warning runs before the storm, and — crucially — when it has been wrong. Because the honest version of this story is not "the curve inverts, therefore recession." It is "the curve inverts, therefore the *odds* of a recession have gone up a lot, on a timeline nobody can pin down." That nuance is the whole game, and it is what separates someone who understands the signal from someone who just repeats the headline.

## Foundations: yields, maturities, and the curve

Before any of this makes sense, we need to be precise about a handful of terms. None are hard, but the signal lives in the relationship between them, so let's define each from zero.

A **bond** is a tradable loan. You hand over money today, and in return you receive a fixed stream of payments plus the return of your principal at the end. A **US Treasury** is a bond issued by the US government; because the government can always print dollars to pay dollar debts, Treasuries are treated as the closest thing to a risk-free loan in the world. That matters here, because by sticking to Treasuries we strip out credit risk — the chance of *not getting paid* — and isolate the thing the yield curve is really about: the price of time.

A **yield** is the annual return you earn on a bond if you buy it at today's price and hold it. When a 2-year Treasury "yields 4.8%", it means the market price is set so that holding it to maturity earns you about 4.8% a year. Yields and prices move in opposite directions — when a bond's price falls, its yield rises, and vice versa — but for this post you can mostly think in yields directly. A **basis point** (written *bp* or *bps*) is one hundredth of a percent — 0.01% — so a move from 4.8% to 4.3% is a 50 bp drop. Bond people quote everything in basis points.

The **maturity** is how long until the loan is repaid. A 3-month bill matures in three months; a 30-year bond matures in thirty years. The US Treasury issues debt across a whole ladder of maturities — 1-month, 3-month, 6-month, 1-year, 2-year, 5-year, 10-year, 30-year, and more — and each one trades with its own yield.

Now put those together. The **yield curve** is just a chart: maturity on the horizontal axis, yield on the vertical axis, one dot for each Treasury, connected into a line. It answers a single question — *what does it cost to borrow dollars for different lengths of time, right now?* The curve is a snapshot of the price of money across the entire spectrum of time, all at once. When people say "the curve steepened" or "the curve flattened" or "the curve inverted," they are describing how the *shape* of that line changed.

Most of the time the curve **slopes upward**: longer loans pay higher yields. There are two clean reasons. First, the longer you lend, the more uncertainty you take on — more chances for inflation to erode your dollars, more chances for the issuer's situation to change — so you demand extra compensation, called the **term premium**: the bonus yield a long bond pays over a short one purely for the risk of waiting. Second, the curve embeds the market's expectation of where short-term rates are headed. If the economy is growing and the central bank is expected to keep rates around current levels or higher, long yields naturally sit above short ones. An upward-sloping curve is the resting state of a normal economy.

An **inversion** is when that ordinary shape turns upside down: a short-maturity yield rises *above* a long-maturity yield. The slope of the curve, measured as long-minus-short, goes negative. That is a strange, unnatural state — you are being paid less to take more time risk — and markets do not arrive there by accident. Something specific has to be happening, and understanding *what* is the key to understanding why the signal works.

#### Worked example: measuring the slope of the curve

Let's make "slope" concrete with numbers. Suppose today the 2-year Treasury yields **4.80%** and the 10-year Treasury yields **4.30%**. The most-quoted measure of the curve's slope is the **2s10s spread** — shorthand for "the 10-year yield minus the 2-year yield":

$$
\text{2s10s} = y_{10} - y_{2} = 4.30\% - 4.80\% = -0.50\%
$$

Here $y_{10}$ is the 10-year yield and $y_{2}$ is the 2-year yield. A positive number means a normal, upward-sloping curve; a negative number means an inverted one. At −0.50% — equivalently −50 basis points — the curve is inverted by half a percentage point. If instead the 10-year yielded 4.30% and the 2-year only 3.00%, the spread would be +1.30%, a healthy upward slope.

*The slope is just one subtraction: long yield minus short yield. When that subtraction comes out negative, the curve is inverted — and the alarm starts ringing.*

### The expectations hypothesis and the term premium

There is one more pair of ideas to install before the signal becomes fully legible, because they are the engine underneath everything: the **expectations hypothesis** and the **term premium**. Together they answer the question, "what determines a long yield in the first place?"

The expectations hypothesis says that a long-term yield is, at its core, the market's *average expectation of short-term rates over the life of the bond*. The logic is an arbitrage argument. You have two ways to invest for ten years: buy a single 10-year bond, or buy a 1-year bond and roll it over ten times, reinvesting each year at whatever the short rate then is. If those two strategies are expected to pay very different amounts, investors pile into the better one until the yields adjust and the gap closes. In equilibrium, the 10-year yield should roughly equal the *average of the expected one-year rates* over the next decade. This is why the long end of the curve is best understood as a forecast: it is the market's distilled guess about the entire future path of the central bank's policy rate, compressed into a single number.

That is the pure theory. Reality adds the **term premium** — the extra yield investors demand for holding a long bond instead of rolling short ones, as compensation for the *risk* of being locked in. Long bonds swing far more in price than short bonds when yields move (this is **duration**, the subject of [duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)), so a long bond is genuinely riskier to hold, and that risk normally earns a premium. The full picture is:

$$
y_{10} \approx \underbrace{\text{average expected short rate over 10 years}}_{\text{expectations}} + \underbrace{\text{term premium}}_{\text{compensation for duration risk}}
$$

When the curve is upward-sloping in a normal economy, both pieces push the same way: short rates are expected to hold steady or rise, *and* the term premium is positive, so the long yield sits comfortably above the short one. An inversion requires the *expectations* piece to turn sharply negative — the market expecting big future cuts — by enough to overwhelm a positive term premium. And here is the subtle, modern wrinkle that will matter later: if the **term premium itself shrinks toward zero or goes negative** (because, say, central banks are buying up long bonds or foreigners are desperate for safe US assets), then the curve can invert on a *much smaller* expected-cuts story than usual. A depressed term premium makes inversions easier and less meaningful — a point we return to when we dissect 2022–2023.

#### Worked example: backing out the term premium

Suppose the market expects the average short rate over the next ten years to be **3.5%**, and the 10-year Treasury actually yields **4.0%**. Then the implied term premium is simply the difference:

$$
\text{term premium} = y_{10} - \text{expected avg short rate} = 4.0\% - 3.5\% = 0.50\%
$$

Investors are earning an extra half-point for accepting ten years of duration risk. Now imagine quantitative easing and heavy foreign demand crush that premium to **−0.20%** (negative — investors will *pay up* for the safety and liquidity of the 10-year). With the same 3.5% rate expectation, the 10-year now yields $3.5\% + (-0.20\%) = 3.30\%$. If the 2-year is at 3.6%, the curve is inverted by 30 bp — *not* because the market is forecasting a deep slump, but because the term premium has gone negative. Same inversion, very different meaning.

*A long yield is the expected path of short rates plus a premium for duration risk; an inversion can come from either a grim rate forecast or a vanished term premium, and telling the two apart is the hard part.*

## What "inversion" actually means, and why it feels so wrong

Step back and feel the strangeness again, because the intuition is the whole point. In a normal world, time costs money. If I borrow your \$10,000 for ten years, I should pay you more than if I borrow it for two, because over ten years more can go wrong: inflation can chew through the real value of your dollars, the world can change, and you give up flexibility for far longer. The upward slope of the curve is the market saying, sensibly, "longer loans, higher rates."

An inversion says the opposite: *short money is more expensive than long money.* The market is willing to lock in a *lower* yield for ten years than for two. The only way that makes sense is if the market expects short-term rates to be **much lower in the future** than they are today. Investors are reaching out to the 10-year and accepting 4.30% — locking in today's relatively high long yield — precisely because they believe that two, three, five years from now, short-term rates will have fallen well below 4.30%. They would rather grab 4.30% for a decade now than keep rolling over short-term money that they expect will soon pay far less.

Why would short rates fall a lot in the future? Because the central bank cuts them — and the central bank cuts them when the economy weakens. So an inverted curve is, at its core, a collective bet that **the economy is going to slow enough that the Fed will have to cut rates substantially**. That is the mechanism in one sentence. The curve inverts because the bond market, in aggregate, is pricing in a future of lower rates, and lower rates mean a weaker economy ahead.

There is a second, more mechanical channel that reinforces this and is worth holding onto: the **bank-lending channel**. Banks make money by borrowing short (your deposits, money-market funding) and lending long (mortgages, business loans). They profit from the *gap* between the long rate they earn and the short rate they pay — which is, essentially, the slope of the yield curve. When the curve inverts, that gap vanishes or goes negative, and lending becomes structurally less profitable. Banks respond by tightening credit: fewer loans, stricter terms, higher hurdles. Less credit creation means less spending, less investment, less hiring — which is exactly how an inversion can help *cause* the slowdown it is busy predicting. The signal is partly a forecast and partly a self-fulfilling prophecy, and the bank channel is why.

#### Worked example: why the 10-year accepts a lower yield

Suppose you manage a bond portfolio and you believe the Fed, currently holding short rates at 5.0%, will be forced to cut to 2.5% over the next two years as the economy cools. You have \$1,000,000 to put to work. You can:

- **Roll 2-year notes:** earn ~4.8% now, but in two years you reinvest at whatever the short rate is then — and you think that will be far lower, maybe 2.5%. Over ten years your blended return drifts down toward that lower future rate.
- **Buy the 10-year at 4.3%:** lock in 4.3% a year for the full decade, immune to the cuts you expect.

If you are right that short rates collapse, the 10-year at 4.3% is the *better* deal even though its headline yield is lower than the 2-year's, because it protects you from reinvesting at 2.5%. So you — and thousands of investors thinking the same thing — buy the 10-year, pushing its price up and its yield down, even while the Fed pins the 2-year up near 4.8%. The result is mechanical: $y_2 = 4.8\% > y_{10} = 4.3\%$, an inverted curve, produced by exactly the expectation that drives it.

*An inversion is the market voting with real money that future short rates — and therefore the economy — are headed down.*

## The record: the spread really does go negative before recessions

Here is the figure that makes the case, and it is the centerpiece of this whole post. It plots the 2s10s spread — the same long-minus-short number from our worked example — across roughly five decades, with US recessions marked.

![The 2s10s yield curve spread plotted over five decades with dashed lines marking US recessions, showing the spread dipping below zero before each one](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-2.png)

The values in that chart are illustrative — drawn to show the *pattern* faithfully rather than to be read off to the basis point — but the pattern itself is real and well documented. Each dashed vertical line marks a US recession. And the same thing happens before each one: the spread, which spends most of its life comfortably above the zero line (a normal, upward-sloping curve), dips *below* zero in the months leading up to the downturn. It inverted before the early-1980s double-dip recession, before the 1990–91 recession, before the 2001 dot-com recession, before the 2007–09 Great Recession, and before the brief 2020 pandemic recession. Then, at the far right, comes the deepest inversion in forty years — the 2022–2023 episode — which we will return to, because its story is more complicated.

The strength of this record is genuinely striking. Going back to the late 1960s, an inverted 2s10s or 3m10y curve has preceded *every* US recession. That is why the signal earned its reputation. It is not a vague correlation dredged out of a hundred indicators; it is a single, simple, economically grounded measure that has fired before each of the last several recessions with very few misses. The New York Fed publishes a recession-probability model built largely on the 3m10y spread, and academic work — most famously by economist Campbell Harvey, who first documented the relationship in his 1986 doctoral dissertation — established the curve's predictive power decades ago.

What makes the signal special among the dozens of indicators economists track is the combination of three properties that rarely coexist. First, it is **simple**: one subtraction of two numbers anyone can look up, with no model, no revisions, and no judgment calls. Most economic data — GDP, employment, manufacturing surveys — is revised repeatedly after the fact and arrives with a lag, but a yield spread is a live market price, known to the basis point in real time. Second, it is **forward-looking by construction**: it is not a measurement of what the economy did last quarter but an aggregation of what millions of investors, betting real money, expect it to do. Most indicators are rear-view mirrors; the curve is a windshield. Third, it is **economically grounded**: we have a clean mechanism — the expectations of future cuts plus the bank-lending channel — that explains *why* it should work, which is what separates it from a data-mined coincidence that happens to have lined up a few times. An indicator that is simple, forward-looking, and mechanistically explicable is a rare and valuable thing, and that is why the inversion, for all its caveats, sits at the top of the recession-watcher's toolkit.

It is also worth being precise about what "preceded every recession" does and does not claim. It is a statement about **sensitivity** — when a recession came, the curve had usually inverted first. It is *not* a statement about **specificity** — that every inversion is followed by a recession. Those are different things, and conflating them is the root of most bad takes about the curve. A signal can catch every recession (high sensitivity) while still occasionally crying wolf (imperfect specificity). The 1966 near-miss and the drawn-out 2022–2023 episode are specificity failures, not sensitivity failures. Keeping the two straight is the difference between "the curve has a great track record of leading recessions" (true) and "every time the curve inverts, a recession is coming" (not quite true).

But notice two things even in this clean picture. First, the *lead time* between the dip below zero and the recession is not constant — some dips precede the recession by a year, some by closer to two, one by only about six months. Second, the spread does not just invert and stay inverted until the recession hits; it wiggles, sometimes crossing zero more than once, and it typically *re-steepens* (climbs back above zero) right around the time the recession actually begins. That re-steepening, often driven by the Fed cutting rates in a hurry, is itself part of the pattern. Keep both of those in mind — they are where the "mostly works" in the title comes from.

#### Worked example: from a +150 bp curve to a −50 bp inversion

Let's walk a single curve through a full cycle, the way it really happens, using round numbers. Start at the beginning of a tightening cycle:

- **Year 0 (easy policy, normal curve):** the Fed funds rate is 0.25%. The 2-year yields **0.50%** (a hair above the funds rate, pricing a few small hikes). The 10-year yields **2.00%**. The 2s10s spread is $2.00\% - 0.50\% = +1.50\%$, or +150 bp. A nicely upward-sloping, healthy curve.

- **Year 1 (mid-hiking cycle):** inflation has surged, and the Fed is hiking fast — funds rate now 3.0% and rising. The 2-year, which tracks expected policy over the next two years, jumps to **4.0%**. But the 10-year barely moves to **3.5%**: investors think these high rates will *break* the economy and force cuts later, so they are happy to lock in 3.5% for a decade. The spread is now $3.5\% - 4.0\% = -0.50\%$, or −50 bp. The curve has **inverted**.

The whole −200 bp swing — from +150 to −50 — came from the front end rising 350 bp (0.50% → 4.0%) while the long end rose only 150 bp (2.00% → 3.5%). The Fed lifted the short end; the market's expectation of future cuts capped the long end. That divergence *is* the inversion.

*An inversion is manufactured by the front end and the long end moving by different amounts in the same tightening cycle — policy pushes one up hard, expectations hold the other down.*

## The mechanism, step by step

We have the intuition; now let's lay out the machinery cleanly, because the *why* is what makes the signal trustworthy rather than superstitious.

![A pipeline showing how Fed hikes lift the front end while expected future cuts pull the long end down until short yields exceed long yields](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-3.png)

The chain runs like this. First, the **Fed hikes** its policy rate — say from 0.25% toward 5.25% — to cool an overheating economy and bring down inflation. Second, the **front end of the curve follows**: the 2-year yield is, to a first approximation, the market's average expectation of the overnight rate over the next two years, so as the Fed hikes and signals more hikes, the 2-year is dragged up to roughly 4.9%. The front end is, in effect, *anchored to current and near-term policy*.

Third, the **market starts expecting cuts**. High rates bite: borrowing slows, hiring cools, inflation begins to fall. Investors look ahead and conclude that the Fed will eventually have to *reverse* — to cut rates back down to support a slowing economy. Fourth, the **long end prices that in**. The 10-year yield is roughly the market's average expectation of the short rate over the *next ten years*, plus a term premium. If short rates are 5% now but expected to average, say, 4% over the coming decade as cuts arrive, the 10-year settles around 4.3% — *below* the 2-year. Fifth and finally, $y_2 > y_{10}$: short yields sit above long yields, the spread goes negative, and **the curve inverts**.

The deep point is that an inversion requires *both* forces. A flat or inverted curve is not the Fed acting alone, and it is not the market acting alone — it is the collision of present policy (pushing the short end up) and future expectations (holding the long end down). That is also why the signal carries information: it can only happen when the market genuinely believes rates — and growth — are heading lower. A curve cannot invert because of optimism. It inverts because of a widely held belief that tight policy is going to slow the economy enough to be undone.

This is the right place to connect the dots to the wider world, which is the spine of this whole series: the yield curve is not an abstract academic plot. It is the price of money across time, and that price sets other prices. The long end of the curve is the benchmark for **30-year mortgage rates**, for corporate borrowing costs, for how companies discount future profits when deciding whether to invest. When the curve inverts, it is telling you the bond market expects the cost of money to fall — which is another way of saying it expects the economy to need cheaper money. For the deeper machinery of how the central bank moves the front end, the sibling post [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) is the companion piece; here we stay focused on the *shape* and what it forecasts.

#### Worked example: decomposing the 10-year into expected short rates

The "long end is an average of expected short rates" idea deserves one explicit calculation, because it is the load-bearing concept. Ignore the term premium for a moment and use the pure expectations view. Suppose the market expects the average overnight rate to be:

- 5.0% in year 1, 5.0% in year 2 (the Fed holds high),
- then 4.0%, 3.0%, 2.5% in years 3, 4, 5 (cuts begin),
- and 2.5% in each of years 6 through 10 (settled at a lower level).

The 10-year yield, under pure expectations, is roughly the average of those ten numbers:

$$
y_{10} \approx \frac{5.0 + 5.0 + 4.0 + 3.0 + 2.5 + 2.5 + 2.5 + 2.5 + 2.5 + 2.5}{10} = 3.20\%
$$

Meanwhile the 2-year, averaging just the first two years, is roughly $\frac{5.0 + 5.0}{2} = 5.00\%$. So $y_2 = 5.00\% > y_{10} = 3.20\%$ — a deep inversion of −180 bp — produced entirely by the expectation that the Fed will cut hard after year 2. Add a small positive term premium and the 10-year ticks up a bit, but the inversion survives.

*The long yield is a forecast of the whole future path of short rates; when that path bends down, the long yield falls below the short yield and the curve inverts.*

## The bank-lending channel: how an inversion bites the real economy

We have treated the curve mostly as a *forecast* — the market predicting a slowdown. But there is a second story in which the inversion is not just predicting trouble; it is helping to *create* it. That is the **bank-lending channel**, and it is the most concrete link between the abstract shape of the curve and the real economy of jobs and spending.

A bank's business model is, at its heart, **borrow short, lend long**. The money a bank lends out — a 30-year mortgage, a 7-year business loan, a car loan — is funded by liabilities that are *short*: customer deposits that can be withdrawn any day, and short-term wholesale funding the bank rolls over constantly. The bank earns the long rate on its loans and pays the short rate on its funding. The difference between the two — the **net interest margin** — is a huge chunk of how a traditional bank makes money. And that difference is, almost exactly, *the slope of the yield curve*.

So watch what happens when the curve inverts. The bank's funding cost (the short rate) rises *above* the yield it can earn on new long-term loans. Making a new 10-year loan funded by deposits that cost more than the loan yields is a money-loser. Banks are not charities; they respond by **tightening credit**. They raise lending standards, demand bigger down payments, charge wider margins, approve fewer applications, and shrink their loan books. The Federal Reserve's quarterly Senior Loan Officer Opinion Survey, which asks banks directly whether they are tightening or loosening, reliably shows banks tightening after the curve inverts.

And tighter credit *is* a slowing economy. A business that cannot get a loan does not build the new plant or hire the new workers. A family that cannot get an affordable mortgage does not buy the house, so the builder, the realtor, the furniture store all see less business. Credit is the lubricant of a modern economy; when it thickens, the whole machine slows. This is why the inversion is partly self-fulfilling: by squeezing bank margins, it triggers the very credit contraction that pushes the economy toward the recession the curve was forecasting. The forecast and the cause are tangled together.

This channel also explains a timing nuance. Banks do not slam the brakes the instant the curve inverts; they tighten gradually, loan committee by loan committee, quarter by quarter. The credit squeeze builds slowly, which is one more reason the lag between inversion and recession is long — it takes time for tighter lending to work its way through borrowing, spending, and hiring. The smoke alarm (the inverted curve) goes off well before the fire (the recession) because the mechanism it triggers is itself slow-acting.

#### Worked example: an inverted curve eats a bank's margin

Consider a simplified bank, "Riverbank." It funds itself with \$1,000,000,000 (one billion dollars) of deposits and lends it all out. Two scenarios:

- **Normal curve:** deposits cost the bank 1.5% (a short rate), and its loan book yields 4.5% (a long rate). Net interest margin = $4.5\% - 1.5\% = 3.0\%$. On \$1B, that is **\$30,000,000** a year of net interest income. Healthy.
- **Inverted curve:** the Fed has hiked, so deposits now cost 5.0%, while the bank's *existing* loans — locked in earlier at 4.5% — still yield only 4.5%. Net interest margin = $4.5\% - 5.0\% = -0.5\%$. On \$1B, that is **−\$5,000,000** a year. The bank is now *losing money* on the spread.

Faced with that, Riverbank stops making new loans at 4.5% (they would lose money) and only lends at much higher rates, choking off borrowers. Multiply Riverbank across the whole banking system and you get a system-wide credit crunch — the inversion translating directly into less lending, less spending, and a weaker economy. (This same dynamic, when deposits flee and locked-in long bonds are underwater, is how rate moves can break a bank outright, as the [SVB and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) episode showed.)

*An inverted curve does not just predict a slowdown — by destroying the borrow-short-lend-long margin that funds bank lending, it helps cause one.*

## How long is the warning? Lead time, and why it varies

If the curve always inverted exactly six months before a recession, it would be a clock and we could all set our watches. It does not. The lead time — the gap between the *first sustained inversion* and the *official start of the recession* — has ranged from roughly six months to nearly two years across the modern episodes.

![A timeline comparing six inversion episodes from 1978 to 2022 with the lead time in months from first inversion to recession start](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-4.png)

The figure lays out the modern record. The 2s10s inverted in 1978 and the recession began in January 1980 — a lead of around 17 months. It inverted in early 1989 ahead of the July 1990 recession — about 18 months. It inverted in February 2000 before the March 2001 recession — roughly 13 months. It inverted in June 2006 before the December 2007 recession — around 18 months. The 2019 inversion was the short one: the 2s10s briefly dipped negative in August 2019, and the (pandemic-triggered) recession began in February 2020 — only about six months later, and arguably for reasons that had nothing to do with the inversion. And then 2022: the 2s10s inverted in July 2022, and as this is written the wait has run far longer than any prior episode, with no official recession yet declared.

Why is the lag so variable? Because the inversion marks the moment the market becomes *convinced* tight policy will bite — but the actual biting takes time and depends on the cycle. This is the famous "long and variable lags" of monetary policy, a phrase economists use precisely because the delay between cause (tight money) and effect (recession) is genuinely unpredictable. The economy has momentum: consumers keep spending savings, companies keep filling backlogs, employment stays strong for a while even as the forward-looking signals deteriorate. The inversion is the smoke alarm; the fire can smolder for a year or more before it spreads.

This variability is exactly why the curve is a lousy *timing* tool even though it is a good *risk* tool. If you sold all your stocks the day the 2s10s inverted in 2022, you would have missed one of the strongest equity rallies in years while you waited for a recession that, for a long stretch, did not come. The signal tells you the *odds* have shifted, not the *date*. For the macro view of how to read the slope as a regime indicator rather than a stopwatch, the sibling post [reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) takes the policy-and-positioning angle; the lesson there and here is the same — respect the signal, distrust the timing.

#### Worked example: the cost of treating the signal as a clock

Suppose in July 2022, when the 2s10s inverted, you held \$100,000 in a broad US stock index and decided to "get out before the recession." Over the following 18 months, instead of a crash, the index rallied substantially — call it +25% for the illustration. By going to cash earning, say, 5% in money-market funds, you earned about \$7,500 over that stretch (5% for roughly 1.5 years), while a stay-invested neighbor earned about \$25,000. You *underperformed by roughly \$17,500* by acting on the inversion as if it were a countdown.

Now flip it: if a recession *had* arrived on a typical timeline and the index had fallen 30%, you would have dodged a \$30,000 loss. That asymmetry — large opportunity cost if early, large protection if right — is the real decision the signal poses. It does not tell you which branch you are on.

*An inverted curve changes the probabilities, not the calendar; sizing your response to "elevated risk" is sane, betting on a specific date is not.*

### The re-steepening: the underrated second signal

Here is a subtlety that most casual commentary misses entirely, and it changes how you should read the curve in real time. The inversion is the *first* signal — the early warning. But the curve almost always *un-inverts* before the recession actually starts, and that re-steepening is the *second*, later, and arguably more urgent signal.

Think about why. The curve is inverted because the Fed has hiked the short end up while the market holds the long end down expecting cuts. When the economy finally starts to crack — when the data turns and a recession looks imminent rather than distant — the Fed pivots and begins cutting rates, often quickly. Cutting the policy rate drags the *short end* back down. As the 2-year plunges toward the cuts, the spread (long minus short) climbs back above zero. The curve re-steepens, and it usually does so via a "**bull steepener**" — short yields falling faster than long yields, the curve un-inverting from the bottom up.

So the full pattern is: curve inverts (early warning, recession 1–2 years out), curve stays inverted (the slow burn), then curve re-steepens sharply (the Fed is now cutting in earnest, recession is at the door or already here). An investor who breathes a sigh of relief when the curve "un-inverts" has it exactly backwards: in the historical episodes, the recession typically began *around or shortly after* the re-steepening, not during the inversion itself. The dis-inversion is not the all-clear; it is often the final approach. Watching the *transition* — inverted, then rapidly un-inverting — is a richer signal than watching the level alone.

#### Worked example: reading the inversion-then-steepening sequence

Trace the 2-year and 10-year through a stylized late cycle:

- **Month 0:** 2y = 4.9%, 10y = 4.3%. Spread = −60 bp. Deeply inverted; the early warning fired a year ago.
- **Month 6:** data weakens. 2y = 4.7% (market sniffing cuts), 10y = 4.2%. Spread = −50 bp. Still inverted, slightly less so.
- **Month 9:** the Fed starts cutting. 2y plunges to 3.8% as the market prices a rapid easing cycle; 10y eases only to 4.0%. Spread = $4.0\% - 3.8\% = +20 bp$. The curve has **re-steepened back to positive** — via the short end collapsing, a classic bull steepener.
- **Month 10–12:** the recession is officially underway.

If you had treated the Month 9 return to a positive spread as "danger over," you would have walked straight into the recession. The re-steepening was the late-stage confirmation, not the reprieve.

*The curve un-inverting because the Fed is slashing rates is not relief — historically it is the signal that the recession has arrived or is about to.*

## The two measures: 2s10s versus 3m10y

So far I have used the 2s10s — the 10-year minus the 2-year — because it is the one traders and headlines quote most. But it is not the only measure, and it is not the one the Federal Reserve's own researchers prefer.

![Two yield curve spread measures, the 2s10s and the 3m10y, plotted as lines over time both crossing below zero into inversion](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-5.png)

The figure shows the two side by side. The **2s10s** (10-year minus 2-year, the solid line) and the **3m10y** (10-year minus 3-month, the dashed line) tell the same broad story, but they invert at slightly different times and by slightly different amounts. The reason is what sits at the short end. The **2-year** yield already bakes in the market's expectations of Fed hikes and cuts over the next two years, so the 2-year can start *falling* — pricing future cuts — even while the Fed is still raising the overnight rate. That can make the 2s10s invert relatively early. The **3-month** bill, by contrast, hugs the *current* overnight rate almost exactly; it has very little forward-looking content. So the 3m10y inverts only once the Fed has actually hiked the front end up to or above the 10-year — typically a bit *later* than the 2s10s, but with arguably cleaner economic meaning.

The Federal Reserve's most-cited recession-probability model, developed by economists Arturo Estrella and Frederic Mishkin in the 1990s, uses the **3m10y** spread. Their argument is that the 3-month-to-10-year gap best captures the stance of monetary policy relative to long-run growth expectations — it directly contrasts "what borrowing costs right now" with "what the economy expects over the long haul." Many practitioners watch the 2s10s for its earlier signal and the 3m10y for confirmation. When *both* are inverted, the signal is considered broad-based and more reliable than either alone.

#### Worked example: when 2s10s and 3m10y disagree

Picture a moment early in a hiking cycle. The Fed funds rate is 3.0% and the market is sure more hikes are coming, then cuts. The yields might look like this:

- **3-month bill:** 3.05% (basically the current overnight rate).
- **2-year note:** 3.90% (averaging more near-term hikes, then leveling off).
- **10-year note:** 3.60% (pricing the eventual cuts and long-run growth).

Now compute both spreads:

$$
\text{2s10s} = 3.60\% - 3.90\% = -0.30\% \quad (\text{inverted})
$$
$$
\text{3m10y} = 3.60\% - 3.05\% = +0.55\% \quad (\text{still positive})
$$

The 2s10s is already screaming recession while the 3m10y is still calm. Six months later, after two more hikes lift the 3-month bill to 4.0%, the 3m10y finally flips to $3.60\% - 4.00\% = -0.40\%$ and confirms. The two measures agreed on direction but disagreed on *timing* by half a year — which is exactly why sophisticated readers watch both.

*The 2s10s inverts early because the 2-year sees the future; the 3m10y inverts later because the 3-month only sees the present — and the Fed trusts the slower, cleaner one.*

## What the curve does to the rest of your financial life

It is easy to treat the yield curve as an inside-baseball obsession of bond traders. It is not. The shape of the curve sets prices that reach into mortgages, stocks, the dollar, and the cost of nearly everything financed over time. This is the spine of the whole series: bonds are the price of money, and that price sets every other price. An inversion is the bond market re-pricing the future, and that re-pricing radiates outward.

**Mortgages.** The 30-year fixed mortgage rate is priced off the long end of the curve — most directly off the 10-year Treasury yield, plus a spread. (It tracks the 10-year rather than the 30-year Treasury because most mortgages are refinanced or paid off long before 30 years, giving them an effective life closer to a decade.) When the curve inverts because the long end is being held down by expected cuts, mortgage rates can actually be *lower* than the short-term rates the Fed has hiked to. That is why, during an inversion, you can see the strange situation of a savings account paying 5% while a 30-year mortgage costs 6.5% — a far smaller gap than usual, and occasionally even an inverted relationship in the financing world. The curve is quietly steering housing affordability.

**Stocks.** Equity prices are, in theory, the present value of a company's future profits, discounted back at a rate built on the curve. Two things happen when the curve inverts. First, the high short rates that caused the inversion raise the discount rate on near-term cash and make safe cash itself more attractive (why own a risky stock yielding 2% when a T-bill pays 5%?), which pressures valuations. Second, and more importantly, the inversion is a forecast of a coming slowdown — and slowdowns mean lower corporate profits. Historically, equities have often kept rising for months after an inversion (the lag again) before eventually falling as the earnings hit arrives. The bond market's pessimism and the stock market's optimism can coexist for a surprisingly long time, which is exactly the trap that punishes investors who treat the inversion as a sell-everything signal. The interplay is the subject of [the stock-bond correlation and the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

**The dollar and global money.** Because Treasuries are the world's reserve asset and the dollar is the world's reserve currency, the US curve is watched globally. High short-term US rates — the very thing that inverts the curve — tend to attract capital into dollars, strengthening the currency and tightening financial conditions for everyone who borrows in dollars, including emerging-market governments and companies. So a US inversion is not just a US recession signal; it is a global tightening signal. The curve's message ripples through exchange rates, capital flows, and credit conditions far beyond American borders.

#### Worked example: the inversion and your mortgage versus your savings

Suppose during a deep inversion the numbers line up like this for a household: a high-yield savings account pays **4.8%** (tracking the short rate the Fed hiked), while a new 30-year fixed mortgage costs **6.3%** (priced off a 10-year Treasury at ~4.0% plus a ~2.3% spread). Compare that to a normal-curve world where savings pay 1.0% and the mortgage costs 4.0% — a 3.0-point gap between what you pay to borrow and what you earn to save. In the inverted world that gap has compressed to just $6.3\% - 4.8\% = 1.5\%$.

What does that mean for you? Cash is unusually well paid (4.8% risk-free is a real return after modest inflation), which is part of why investors are content to hold T-bills and wait. And the cost of long-term borrowing, while high in absolute terms, is being held *down* relative to short rates by the same expected-cuts story that inverted the curve. The shape of the curve is literally setting the trade-off between saving and borrowing in your own household budget.

*The inverted curve is not abstract — it shows up as an unusually small gap between what your savings earn and what your mortgage costs, because the same expected-cuts story drives both.*

## Why it is a warning, not a guarantee

This is the section that earns the "mostly" in the title, and it is the most important part of the post. An inverted curve is one of the best recession signals we have, and it is *still* not a sure thing. Four reasons matter.

![A matrix contrasting the naive belief about yield curve inversion with the honest caveat across timing, false signals, distorted long end, and sample size](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-6.png)

The figure organizes the caveats. The naive belief, in each row, is that the signal is precise and infallible; the honest caveat is what actually limits it.

**Timing.** As we saw, the lag from inversion to recession runs from about six months to two years and varies every cycle. The signal says risk is elevated; it cannot tell you the quarter.

**False signals.** The record is strong but not spotless. The most-cited miss is the mid-1960s: a brief inversion around 1966 was followed by a slowdown but no official recession. Short, shallow inversions that quickly reverse have historically been less reliable than deep, sustained ones. The signal is a probability, and probabilities are sometimes wrong.

**A distorted long end.** The whole logic assumes the 10-year yield is a clean market forecast of future short rates. But the modern long end has been pushed around by forces that have nothing to do with US recession odds: massive central-bank bond buying (**quantitative easing**, where the Fed and others bought trillions in long-dated bonds and held their yields artificially low), enormous structural demand for safe US assets from foreign central banks and pension funds, and a compressed or even negative **term premium**. When the long end is artificially depressed, the curve can invert *more easily* and *for longer* without the usual recessionary meaning — which is one of the leading explanations for the strange 2022–2023 episode.

**Sample size.** Here is the quietly devastating point. There have only been about eight US recessions since 1960. Eight. Every confident statement about the curve "always" predicting recessions is built on a handful of data points. A pattern that holds in eight cases is suggestive, even strong — but it is not the law of gravity, and it is far too small a sample to rule out that the relationship could weaken or break as the structure of the economy and bond market changes.

Put those four together and the honest read emerges: an inverted curve says **recession risk is elevated**, not that a recession is *scheduled*. It is a genuinely informative signal sitting on top of a small sample, a variable lag, occasional false alarms, and a long end that modern policy has distorted. Respect it; do not worship it.

### The "this time is different" trap — in both directions

There is a deeper, almost philosophical hazard buried in the distortion caveat, and it is worth dwelling on because it has tripped up very smart people. Every cycle, when the curve inverts, a chorus arises to explain why *this* inversion does not count. In 2006, the argument was a global "savings glut" — Ben Bernanke's own phrase — flooding the world with capital that depressed long yields and supposedly broke the signal's meaning. The economy, the argument went, was fine; the inversion was a technical artifact. Then 2008 happened, and the inversion turned out to have been right and the dismissal wrong.

In 2022–2023, the identical argument returned in modern dress: quantitative easing and structural Treasury demand had crushed the term premium, so the inversion was distorted and meaningless. This time, for an unusually long stretch, the dismissal looked *correct* — no recession arrived on schedule. So which is it? Is "the long end is distorted, ignore the inversion" a wise caveat or a dangerous excuse?

The honest answer is: it is *both*, and you cannot know in real time which one you are living through. The distortion is genuinely real — central-bank buying really does depress term premia, and that really can make inversions less meaningful. But "this time is different" is also the most expensive phrase in finance, deployed at the top of every cycle to rationalize ignoring a warning that later proves correct. The disciplined posture is to hold the caveat and the signal *at the same time*: take the inversion seriously as elevated risk, acknowledge the distortion as a reason the probability might be lower than history implies, and refuse to let either consideration collapse into certainty. The moment you are *sure* the signal is broken is precisely the moment you are most exposed if it is not.

### Why the small sample is more damning than it sounds

It is worth sitting with the sample-size problem a beat longer, because it is the caveat people most want to wave away. There have been roughly eight US recessions since 1960. Statistically, eight observations is almost nothing. If you flipped a coin eight times and it came up heads seven times, you would not conclude the coin is rigged with any real confidence — eight trials simply cannot distinguish a strong relationship from a moderate one with a lucky streak. The yield curve's record is more impressive than that, because the relationship is *economically grounded* (we can explain the mechanism, not just observe the correlation), which is exactly what lets us trust it more than a random data-mined indicator. But the grounding is doing a lot of the work; the raw statistical evidence, on its own, is thin. Anyone who tells you the curve "has predicted every recession for sixty years" as if that were overwhelming proof is mistaking a small, suggestive, mechanistically plausible sample for a law of nature. It is good evidence. It is not certainty, and it cannot be.

#### Worked example: turning the signal into a probability, not a verdict

The grown-up way to use the curve is to translate it into an odds shift, not a yes/no. Suppose your baseline estimate is that in any given 12-month window there is a **15%** chance of a US recession starting. Now the 3m10y inverts deeply and stays inverted for months. Historically, a sustained 3m10y inversion has been associated with recession-within-12-months probabilities in the rough neighborhood of **two-thirds** (the New York Fed model has produced readings in that range during deep inversions).

So the signal moves your estimate from ~15% to perhaps ~65% — a large, decision-relevant shift. But notice what it is *not*: it is not 100%, and it does not name the month. A 65% chance of recession means a 35% chance of *no* recession — roughly one in three. Acting as if 65% were 100% is precisely the error that cost our hypothetical investor \$17,500 earlier.

*The right output of an inverted curve is a revised probability you can size decisions against, not a verdict you bet the farm on.*

## Reading the depth and duration of an inversion

Not all inversions are equal. A curve that dips to −5 bp for two weeks and pops back positive is a much weaker signal than one that plunges to −100 bp and stays there for a year. Three dimensions tell you how loud the alarm is.

![A grid showing how to read an inversion by its depth, its duration, and its breadth, applied to the 2022 to 2023 episode](/imgs/blogs/yield-curve-inversion-the-recession-signal-that-mostly-works-7.png)

**Depth** — how far below zero — measures the *strength* of the market's conviction that big cuts are coming. A −100 bp inversion says the market expects far more aggressive easing than a −10 bp one. **Duration** — how many months the curve stays inverted — measures the market's *commitment* to that view; a one-week flicker can be noise, while a sustained inversion is a settled belief. **Breadth** — whether multiple measures (2s10s *and* 3m10y) are inverted at once — measures how *broad-based* the signal is; when both the trader's gauge and the Fed's gauge agree, the warning is harder to dismiss.

The 2022–2023 episode scored high on all three, which is what made it so widely discussed. The 2s10s reached roughly **−108 bp** in mid-2023 — the deepest inversion since the early 1980s. It stayed inverted for well over a year — the longest sustained inversion on record. And both the 2s10s and the 3m10y were deeply negative simultaneously. By every traditional reading, this was the strongest recession warning the curve had given in four decades. And yet, for an unusually long stretch, no recession arrived — which is the single best modern illustration of why "mostly works" is the honest phrasing, and why the distorted-long-end and small-sample caveats deserve real weight.

#### Worked example: scoring two inversions side by side

Compare two hypothetical inversions to see how the dimensions combine.

- **Inversion A:** 2s10s dips to −8 bp, stays negative for 3 weeks, 3m10y never inverts. Depth: shallow. Duration: brief. Breadth: narrow. *Score: weak warning — plausibly noise from the 2-year jumping around on a single Fed meeting.*
- **Inversion B:** 2s10s reaches −90 bp, stays inverted 14 months, 3m10y also inverts to −120 bp. Depth: deep. Duration: long. Breadth: both measures. *Score: strong warning — this is the configuration that has preceded recessions.*

If you had to assign rough recession-within-the-next-18-months odds, you might leave Inversion A near your baseline (maybe nudge it from 15% to 20%) but lift Inversion B sharply (toward 60–70%). The depth, duration, and breadth are how you turn "the curve inverted" into "how much should I actually update."

*A real warning is deep, durable, and broad; a shallow, brief, single-measure dip is closer to noise — read all three dimensions before you react.*

## Common misconceptions

**"An inverted curve causes a recession."** Mostly no — it *forecasts* one, and only *partly* contributes to one. The curve inverts because the market expects a slowdown; the slowdown comes from the tight policy and weakening fundamentals the inversion is reflecting. The one real causal thread is the bank-lending channel: an inverted curve squeezes bank profit margins and tightens credit, which does drag on the economy. But the inversion is far more a thermometer than a cause. Selling the thermometer does not change the temperature.

**"The moment the curve inverts, the recession has started."** No — the recession typically begins *months to years later*, and the curve has often already *re-steepened* (climbed back above zero) by the time the recession officially starts. Counterintuitively, a curve that *un-inverts* rapidly — because the Fed is cutting in a panic — is often a *later-stage* warning, not an all-clear. The dis-inversion can be the more dangerous moment than the inversion itself.

**"It has never been wrong."** Not quite. The 1966 brief inversion did not deliver a recession, and the 2022–2023 inversion — the deepest in forty years — failed for an unusually long time to produce the recession it appeared to predict. The signal's record is excellent but not perfect, and "excellent over eight recessions" is a small-sample claim, not a physical law.

**"2s10s and 3m10y are interchangeable."** They tell the same story but not at the same time. The 2s10s inverts earlier because the 2-year already prices future cuts; the 3m10y inverts later but is the Fed's preferred measure because the 3-month tracks current policy directly. Watching only one means either jumping early or confirming late.

**"A small dip below zero is the same signal as a deep, sustained inversion."** No — depth, duration, and breadth all matter. A brief −5 bp flicker on the 2s10s is far weaker evidence than a −100 bp inversion sustained for a year across both measures. Treating a shallow flicker as a five-alarm fire is a classic over-reaction.

**"If the curve inverts, I should sell everything."** That is treating a probability signal as a timing signal, and it can be enormously costly. The lead time is long and variable; markets have historically continued to rise for many months after an inversion. The defensible response is to adjust risk gradually and treat recession odds as elevated — not to make an all-or-nothing bet on a date the signal cannot give you.

**"When the curve un-inverts, the danger has passed."** This is one of the most expensive misreadings, and it is exactly backwards. The curve typically re-steepens *because the Fed has started cutting rates aggressively in response to a cracking economy* — which means the recession is at the door, not receding. In the historical episodes the downturn usually began around or shortly after the re-steepening. A sigh of relief at the un-inversion is, more often than not, mistimed.

**"The yield curve is the only recession signal you need."** No single indicator is. The curve is unusually good, but it is one input. Sober recession-watching combines the curve with credit spreads, jobless claims, manufacturing surveys, and the breadth of the labor market, precisely because the curve's small sample, variable lag, and modern distortions mean it can be early, late, or — rarely — wrong. Treating any one number as the whole answer is how confident forecasters get embarrassed.

## How it shows up in real markets

**The 1980–82 Volcker inversions.** When Fed Chair Paul Volcker pushed the funds rate toward 20% to crush double-digit inflation, the curve inverted violently — short rates soared far above long rates as the market bet (correctly) that such punishing policy would force a sharp reversal. Two recessions followed in quick succession (1980 and 1981–82), and inflation was finally broken. This is the textbook case of the mechanism in its purest, most extreme form: aggressive hiking pins the front end sky-high while the long end prices the inevitable cuts. For the full macro story of that regime, see [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

**The 1989 inversion before the 1990–91 recession.** The Fed tightened in the late 1980s to contain inflation that had crept back after the mid-decade boom, and the curve inverted in early 1989. The recession arrived in July 1990, roughly 18 months later — a textbook lead time. What makes this episode instructive is how *quiet* it was at the time: there was no spectacular bubble like 1999's dot-coms or 2007's housing, just a gradual tightening that eventually tipped a slowing economy over the edge (helped along by the oil-price spike from the Gulf War). The curve flagged it anyway, which is the point — the signal does not require a dramatic bubble to work; it just reads the collision of tight policy and weakening expectations.

**The 2000 dot-com inversion.** The 2s10s inverted in early 2000 as the Fed tightened into a frothy, tech-driven boom. About a year later, the dot-com bubble burst and the 2001 recession arrived. The curve had quietly flagged the danger while equity investors were still euphoric — a clean reminder that the bond market and the stock market can disagree sharply about the future, and that the bond market's warning has historically aged better. The gap between an exuberant stock market and a worried bond market is itself a tell: when the two most important markets in the world disagree this sharply about the future, the disagreement is information, and history suggests the bond market's caution deserves the benefit of the doubt.

**The 2006–07 inversion before the Great Financial Crisis.** The curve inverted in 2006, well before the housing market cracked and the financial system seized in 2008. At the time, plenty of commentators dismissed it — "this time is different," they said, pointing to the same distorted-long-end arguments (a global "savings glut" depressing long yields) that we now use to *excuse* inversions. In hindsight the curve was right and the dismissals were wrong. That episode is a cautionary tale in both directions: the signal worked, *and* the "the long end is distorted, so ignore the inversion" argument was made then too — and was mistaken. Distortion is a real caveat, but it is also the excuse people reach for right before the signal turns out to be correct. The bank-lending channel was central to how the 2008 downturn metastasized; for the bank-stress angle, the case study of [SVB and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) shows how rate moves and balance sheets interact when the cycle turns.

**The 2019 inversion and the pandemic.** The 2s10s and 3m10y both briefly inverted in 2019, prompting a wave of recession warnings. A recession did arrive in early 2020 — but it was triggered by a pandemic, an exogenous shock no yield curve could have forecast. So 2019 is an awkward data point: the signal "worked" in that a recession followed within months, but the cause had nothing to do with the slowing the inversion was pricing. It is a reminder that correlation in a tiny sample can be partly luck.

**The 2022–2023 inversion: the great debate.** This is the episode that has most tested the signal's reputation. Driven by the fastest Fed hiking cycle in decades to fight post-pandemic inflation, the 2s10s plunged to about −108 bp — the deepest since 1981 — and stayed inverted for well over a year. By every traditional reading, a recession was nearly certain and imminent. Yet the US economy kept growing, unemployment stayed near record lows, and the recession that "had" to come kept not coming. The debate split into two camps. One side argued the signal was *distorted* — that quantitative easing, massive fiscal stimulus, and structural demand for Treasuries had artificially depressed the long end and broken the usual meaning of inversion. The other side argued the signal was merely *early* — that the long-and-variable lag was simply running longer this cycle because households and firms had locked in cheap pandemic-era financing and were unusually insulated from high rates. As of this writing the debate is not fully settled, and that ambiguity is itself the lesson: a signal built on roughly eight recessions can face a genuinely new structural environment and leave even experts unsure whether it is broken or just slow. For how the policy backdrop and bond supply shaped this period, see [deficits, debt, and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).

**Outside the United States.** The signal is strongest and best documented for US Treasuries, but inverted curves have flagged downturns elsewhere too — in the UK and parts of Europe — though with messier records, since other central banks, currency dynamics, and smaller, less liquid bond markets add noise. The cleanest version of the signal lives where the deepest, most liquid risk-free curve does: the US Treasury market. That is also why global investors watch the US curve even for non-US recession risk — the dollar's central role means US monetary conditions ripple worldwide.

## When this matters to you, and where to go next

Even if you never trade a bond, the yield curve touches your life. The long end of the curve drives 30-year mortgage rates, so an inverting curve is, indirectly, the bond market's forecast about future housing affordability and the cost of borrowing for everything from cars to corporate expansion. When you read a headline that "the yield curve just inverted," you now know what it actually means — short rates have risen above long rates because the market expects the Fed to be cutting before long — and, just as importantly, what it does *not* mean: it is not a dated prophecy, and it has been wrong, or merely early, before.

The honest takeaway is the one the title promises. Yield curve inversion is the recession signal that *mostly* works: economically grounded, historically reliable across a handful of cycles, and worth taking seriously — while resting on a small sample, a long and variable lag, a long end that modern policy has distorted, and at least one drawn-out modern episode that has tested it hard. Use it to update your sense of the *odds*, never to set your watch.

If you want a practical checklist for reading the curve like an adult rather than a headline, it comes down to five habits. **One: check both measures.** Look at the 2s10s for the early signal and the 3m10y for the Fed-preferred confirmation; treat a both-inverted reading as far stronger than either alone. **Two: weigh depth, duration, and breadth**, not just the binary of inverted-or-not — a deep, sustained, broad inversion is a real warning, a shallow flicker is closer to noise. **Three: translate the signal into a probability, not a verdict** — "recession odds are now elevated, maybe roughly two-in-three over the next year or so" is a usable thought; "the recession starts in March" is not. **Four: watch the re-steepening as a second, later signal** — the curve un-inverting because the Fed is cutting hard is the late-stage warning, not the all-clear. **Five: corroborate** — pair the curve with credit spreads, jobless claims, and labor-market breadth, because no single number, however good, should carry an entire forecast. Do those five things and you will be reading the curve the way the people who actually understand it do: as a powerful, imperfect probability gauge, not an oracle.

To go deeper, three directions in this series and its siblings build directly on this post. For the math of how the curve is constructed from individual bond prices, see [spot rates, the zero curve, and bootstrapping](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) and [forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be), which formalize the "long yield is an average of expected short rates" idea we used by hand here. For the policy machinery that moves the front end, [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) explains the hikes, cuts, and quantitative easing that distort the long end. And for the rigorous, model-based treatment of the entire term structure, [yield curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling) is where the intuition in this post becomes equations. This is educational material about how a market signal works, not financial advice — but understanding the signal is the first step to not being fooled by it.

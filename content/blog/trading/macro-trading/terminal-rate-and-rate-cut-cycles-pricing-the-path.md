---
title: "The Terminal Rate and Rate-Cut Cycles: Pricing the Path"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into why markets trade the expected path of interest rates — where they peak and how fast they fall — not today's policy rate, and how the futures curve and the 2-year yield let you read and trade that path."
tags: ["macro", "monetary-policy", "interest-rates", "terminal-rate", "rate-cuts", "fed-funds-futures", "sofr", "neutral-rate", "yield-curve", "2-year-treasury", "trading", "fixed-income"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Markets do not trade today's policy rate; they trade the expected *path*: where rates peak (the terminal rate) and how fast they come back down. It is the repricing of that path, not the level, that moves bonds and stocks.
>
> - The fed funds futures curve and the SOFR strip price the *whole future path* of the policy rate, month by month. Read together, that strip is the market's forecast of where rates go, not where they are.
> - The single most-watched number for the path is the **2-year Treasury yield**: it is roughly the market's average expected policy rate over the next two years, so it peaks and falls *before* the Fed actually cuts.
> - When the priced path reprices — say from 3 cuts to 6 cuts in the next year — bonds, stocks, and the dollar all move *now*, before a single cut happens, because every asset is discounted off the expected path.
> - The one number to remember: the Fed's terminal rate this cycle was **5.50%** (the upper bound, held from July 2023 to September 2024), and the entire 2024 trade was the market arguing about how fast it would come back down — it priced six cuts entering the year and got three.

In late 2023, the bond market threw a party. Inflation had rolled over from its 9% peak, the Federal Reserve had just held its policy rate steady after the fastest hiking cycle in forty years, and traders decided the hard part was over. By the end of December, fed funds futures — the contracts that let the market bet on where the policy rate will be in future months — were pricing in roughly **six quarter-point rate cuts for 2024**, starting as early as March. The 2-year Treasury yield, which had peaked above 5% in October, tumbled to 4.25%. The S&P 500 ripped to new highs. Everyone was positioned for the pivot.

Then 2024 actually happened. Inflation stopped cooperating — it stalled near 3%, then wobbled higher. The labor market refused to crack. And one by one, the Fed officials who were supposed to be cutting kept saying *not yet*. March came and went with no cut. So did June. So did July. The first cut of the cycle did not arrive until **September 2024** — and over the full year the Fed delivered just **three** cuts, half of what the market had priced eight months earlier. The 2-year yield, which had fallen on the dream of six cuts, climbed back toward 5% in the spring as those cuts got priced *out*, then fell again into the fall as they got priced back *in*. It was a full round trip, and almost none of it had anything to do with the policy rate the Fed was actually charging that day. The rate barely moved. The *expectations about the rate* moved enormously — and those expectations were what traders were long and short.

That episode is the whole thesis of this post in miniature. **Markets do not trade the current policy rate. They trade the expected path of the rate** — where it peaks (the *terminal rate*) and how fast it descends afterward (the *cutting cycle*). The fed funds futures curve and its close cousin the SOFR strip price that entire path. And it is the *repricing* of the path, not the level of today's rate, that moves bonds, stocks, currencies, and credit. Learn to read the path and you understand most rate-driven moves you will ever see. This post builds that skill from absolute zero, then hands you the playbook.

![The rate path climbs to the terminal rate, plateaus, then descends toward the neutral rate](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-1.png)

## Foundations: the terminal rate, the path, and how futures price both

Before any trading, we need a small stack of ideas, each built from scratch: what the *policy rate* is, what the *path* and the *terminal rate* are, how *fed funds futures* and the *SOFR strip* turn the path into tradable prices, and why the *expected* path matters more than the *realized* one. Everything in this post is a consequence of these five.

### The policy rate is one number; the path is its whole future

Start with the thing the central bank controls. In the United States, the Federal Reserve sets a target range for the **federal funds rate** — the rate banks charge each other to borrow reserves overnight. It is a single, blunt number, moved only at the eight scheduled meetings of the Federal Open Market Committee (the FOMC) each year, usually in steps of 0.25% (a "quarter point," or 25 basis points). For most of 2023 and 2024 the *upper bound* of that range — the figure most people quote — sat at 5.50%, then stepped down to 4.50%.

If that were the whole story, rates would be boring. The Fed announces a number eight times a year; you write it down; done. But here is the move that unlocks everything: **the market does not care very much about today's number, because today's number is already known and already priced. What the market trades is its best guess about *all the future numbers* — the policy rate three months from now, six months, a year, two years, three years.** That sequence of expected future policy rates is what we mean by **the path**.

Think of it like the weather. Right now it is 70 degrees outside; that is a fact, not a bet. What you can actually wager on is *tomorrow's* temperature, and next week's, and the seasonal trend. The current temperature is the policy rate. The forecast — the whole curve of expected future temperatures — is the path. Markets are a forecasting machine, so they spend all their energy trading the forecast, not the thermometer reading you can already see.

### The terminal rate is the peak of the path

The path has a characteristic shape over a cycle, and it has names for its parts. When inflation is too high, the Fed *hikes*: it raises the policy rate, meeting by meeting, to slow the economy down. The hikes do not go on forever — at some point the Fed judges that rates are high enough to bring inflation back to target, and it stops. **The level at which the Fed stops hiking is called the terminal rate.** It is the peak of the mountain, the highest point the policy rate reaches in a given cycle.

This cycle, the terminal rate was a 5.50% upper bound, reached at the July 2023 meeting. The Fed then *held* there — a plateau — for about fourteen months while it waited to be sure inflation was beaten. After the plateau comes the descent: the **cutting cycle**, where the Fed lowers the rate step by step back down toward something more normal.

The terminal rate matters enormously to traders for a reason that is easy to miss: **the market starts pricing and trading the terminal rate long before the Fed ever reaches it.** During the 2022 hiking cycle, every inflation report was secretly a referendum on one question — *how high does this peak go?* A hot inflation print did not just mean "one more hike"; it meant "the whole peak is higher than we thought," which repriced the entire path upward at once. We will see exactly how that played out later.

### Fed funds futures and the SOFR strip: turning the path into prices

So the path is the market's forecast of future policy rates. How does that forecast become a *price* you can read and trade? Through two closely related instruments.

The first is **fed funds futures**, traded on the CME exchange. Each contract is tied to one calendar month, and at expiry it settles based on the *average* effective fed funds rate that the Fed actually delivered during that month. The contract is quoted, by an old convention, as **100 minus the rate**. So if the market expects the average fed funds rate in, say, December to be 4.00%, the December contract trades at 100 − 4.00 = **96.00**. If expectations shift and the market now thinks December will average 3.50%, the contract rises to 96.50. The quoting is upside down — a *higher* price means a *lower* expected rate — but the logic is simple: **the futures price tells you exactly what rate the market expects for that month.**

The second instrument is the **SOFR strip**. SOFR — the Secured Overnight Financing Rate — is the modern benchmark for short-term dollar borrowing, based on actual overnight repo transactions (borrowing secured by Treasury collateral). It tracks the fed funds rate very closely because both are overnight dollar rates anchored by Fed policy. SOFR futures, traded as a *strip* of consecutive quarterly contracts stretching years into the future, price the expected path of SOFR — which, for our purposes, is essentially the expected path of the policy rate. The SOFR strip is the deeper, more liquid instrument the professionals actually use to express a view on the multi-year path; fed funds futures dominate the near term, out to about a year or two.

The key idea is the same for both: **line up the consecutive contracts and you have drawn the market's expected path of the policy rate, month by month and quarter by quarter, years into the future.** Each contract is one dot on the forecast curve. Together they *are* the path. When you hear an analyst say "the market is pricing four cuts by year-end" or "the curve has the terminal rate at 5.5%," they are reading exactly this strip of contracts.

It is worth being precise about *why* these futures are such clean readouts of the path, because it explains why traders trust them. A fed funds futures contract settles on the *realized average* fed funds rate over its contract month — a number that is, by the time the month is over, simply a fact the Fed delivered. There is no credit risk, no company earnings, no soft narrative buried in the settlement; the contract is a near-pure bet on what the Fed will do. That purity is what makes the strip a clean instrument: the only thing that moves it is a change in expectations about Fed policy. Compare that to a stock, whose price tangles together rates, earnings, sentiment, and a dozen other forces — when you want to isolate the market's view of the *rate path*, the futures strip and the 2-year are the cleanest windows you have.

There is also a second, official source for the expected path that you should know how it relates to the market's: the Fed's own **dot plot**. Four times a year, each of the nineteen FOMC participants writes down where they think the policy rate should be at the end of this year, next year, the year after, and in the "longer run." Plotted as dots, this is the *Fed's* forecast of its own path — including, in that longer-run dot, the committee's estimate of the neutral rate. The dot plot and the futures curve are two forecasts of the same path: one from the people who set the rate, one from the people who trade it. **The gap between them is itself a tradable signal** — when the market prices a much more dovish path than the dots show (more cuts, sooner), either the market thinks the Fed will capitulate to weaker data, or the market is offside and due to reprice toward the dots. Watching the two converge or diverge around each meeting is a core part of reading the path, and it is the subject of its own post in this series.

One practical note on SOFR versus fed funds, because beginners conflate them. They are not identical rates: fed funds is *unsecured* overnight lending between banks, while SOFR is *secured* by Treasury collateral in the repo market. Most of the time they move within a few basis points of each other, both anchored by the Fed's policy tools. But in moments of funding stress — a quarter-end when cash is scarce, or a Treasury-market dislocation — SOFR can spike above fed funds as the demand to borrow against collateral surges. For pricing the *policy path* over months and years, the distinction barely matters; both strips trace the same expected trajectory. For trading short-term funding stress, the spread between them is itself the signal. In this post we treat the SOFR strip and the fed funds strip as two views of the same path, which is true for everything except the very front, stress-sensitive contracts.

![Fed funds futures strip showing expected policy rate at each forward date, a plateau then a staircase of cuts](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-3.png)

### The realized path versus the expected path

Here is the last foundation, and it is the one that separates people who understand rates from people who merely follow them. There are two different paths, and confusing them is the most common mistake in macro.

The **realized path** is what the Fed *actually did* — the historical sequence of policy rates, the step chart you can plot after the fact. Hikes from 0.25% to 5.50%, a plateau, then cuts to 4.50%. It is a matter of record. It is also, for a trader, almost useless on its own, because by the time a rate move is realized it has been known and priced for weeks.

The **expected path** is what the market *thinks the Fed will do* from here — the forecast embedded in the futures curve at any given moment. This is what you trade. And it changes constantly, with every inflation report, jobs number, and Fed speech, because each new piece of data shifts the forecast.

The relationship between them is the engine of all rate-driven P&L: **prices move when the expected path reprices to a new shape, which usually happens when the realized data forces the market to revise its forecast.** When the realized path eventually arrives, it confirms or denies what was already priced — and the *surprise*, the gap between what happened and what was expected, is what moves markets in the moment. A cut that was fully priced moves nothing when it lands. A cut nobody saw coming moves everything. We will turn this into a worked example shortly.

Let me make the futures-pricing arithmetic completely concrete before we go further.

#### Worked example: pricing the terminal rate from a fed funds futures contract

Suppose you are looking at a fed funds futures contract for a month roughly a year out, and it is trading at a price of **96.50**. What is the market telling you?

Use the convention: implied rate = 100 − price.

```
implied average fed funds rate = 100 - 96.50 = 3.50%
```

So the market expects the effective fed funds rate to *average* 3.50% during that month. Now suppose today the policy rate's upper bound is 4.50% (effective rate around 4.33%, near the middle of the range). The contract is telling you the market expects the rate to be roughly **0.83% lower** a year from now than it is today. Since the Fed moves in 0.25% steps, that is about **three quarter-point cuts** priced in over that horizon:

```
expected drop  = 4.33% (today) - 3.50% (priced)  = 0.83%
cuts priced    = 0.83% / 0.25% per cut           ≈ 3.3 cuts
```

Now flip it: if a hot inflation report hits and the contract *falls* from 96.50 to 96.25, the implied rate rises from 3.50% to 3.75% — the market just *priced out* one cut. Each fed funds futures contract is worth \$25 per basis point of rate change per contract (a 0.25% move is \$1,250 per contract), so a trader long ten of these contracts just lost about \$6,250 on that 0.25% repricing. The Fed did nothing; the policy rate did not change; but one of the cuts the curve had penciled in for next year just evaporated, and anyone positioned for that cut just took a loss. **The takeaway: a single futures price is a direct, arithmetic readout of how many cuts (or hikes) the market has priced over that horizon, and the contract moves the instant the expected path is revised — long before the Fed acts.**

## How the market expresses a path: reading the futures curve

Now that one contract makes sense, step back and look at the whole strip at once. Plot the implied rate (100 minus price) of every consecutive monthly fed funds contract, or every quarterly SOFR contract, against its date. The resulting line *is* the market's expected path. Reading its shape is a core macro skill, so let us walk through what the shapes mean.

**An upward-sloping front of the curve** means the market expects hikes — each later contract prices a higher rate than the one before. This is what the curve looked like through most of 2022: contract after contract climbing, because the market kept revising the terminal rate higher as inflation surprised to the upside.

**A flat top** means the market expects a plateau at the terminal rate — the contracts for the next several months all price roughly the same rate, because the market thinks the Fed is done hiking but not yet cutting. This is what the front of the curve looked like through much of the 2023–2024 hold.

**A downward slope** means the market expects cuts — each later contract prices a lower rate, a staircase descending toward wherever the market thinks rates ultimately settle. The *steepness* of that descent is the market's view on how *fast* the Fed will cut, and the *level it flattens out at* is the market's view on the **neutral rate** (more on that below).

The single most useful thing you can extract from the curve is the **number of cuts priced over a horizon**, because it is the cleanest summary of the market's view and the easiest thing to bet against. "The curve prices 4 cuts by next December" is a complete, falsifiable statement: if the data comes in such that the Fed delivers fewer, the curve must reprice the rate *up* (cuts get removed), and the contracts fall. If it delivers more, they rise. You are not predicting the policy rate in the abstract; you are predicting whether the *priced number of cuts* is too high or too low.

A subtle but important point: the curve is not a forecast in the way a weather model is a forecast. It is a *probability-weighted average* of many possible paths. If the market thinks there is a 50% chance of a cut at the next meeting, the curve prices *half* a cut — it bakes in the expected value, not a yes/no. So when you read "1.5 cuts priced for the first half of the year," it might mean one certain cut plus a coin-flip on a second, or some other blend. This matters because it means the curve can move smoothly as probabilities shift, not just in discrete jumps when the Fed acts.

This probability structure is worth dwelling on, because it is how the most-quoted number in rates trading — "the market sees a 70% chance of a cut next meeting" — actually gets computed. The arithmetic is straightforward. Suppose the policy rate is 4.50% and the question is whether the Fed cuts 0.25% at the next meeting. The fed funds future for the meeting month settles on the *average* rate over that month, which blends the pre-meeting rate and the post-meeting rate. If the contract implies an average rate that is, say, 0.175% below the current rate, and a full cut would lower the post-meeting portion by 0.25%, you can back out the implied probability: the market is pricing roughly a 70% chance of that cut (because 0.70 × 0.25% ≈ 0.175%, adjusting for the fraction of the month after the meeting). This is exactly what tools like the "CME FedWatch" do — they invert the futures price into a probability of each possible Fed move. You do not need to do the algebra by hand, but you should understand that the headline probability is *derived from the same futures strip* that draws the path; it is not a separate poll of opinions.

Why does this matter for trading? Because it tells you how much is *already in the price*. If the market prices a 90% chance of a cut and the Fed cuts, almost nothing happens — the 90% was already there. If the market prices a 40% chance and the Fed cuts, the path reprices hard, because the surprise is large. The tradable quantity is never "will the Fed cut?" — it is "is the priced probability too high or too low?" A cut that surprises 60% of the market is a huge event; the same cut, fully priced, is a non-event. This is the probabilistic version of the realized-versus-expected distinction, and it is why experienced traders watch the *implied probabilities* shift in the days around each meeting far more closely than they watch the eventual decision.

### Why the front end and the long end price different things

The expected path is not flat in its *certainty*. The market is fairly confident about the next meeting or two — the Fed telegraphs its near-term moves heavily — so the front of the curve (the next few contracts) tends to track Fed communication closely. The further out you go, the more the contracts depend on guesses about inflation, growth, and the neutral rate, and the more they move on changing narratives rather than on the next data point. This is why a single inflation report can violently reprice the *whole* curve: it does not just change the odds of the next cut, it changes the market's belief about the entire trajectory, including the terminal rate and the pace of descent.

## Why repricing the path moves everything: the 2Y as a path proxy

We now arrive at the most important practical idea in the post. You do not have to watch a strip of forty futures contracts to track the expected path. There is one liquid, quoted-everywhere number that summarizes the front of the path beautifully: **the 2-year Treasury yield.**

Here is why. A 2-year Treasury note pays you a fixed yield for two years. An investor deciding whether to buy it is implicitly comparing it to the alternative of rolling overnight money — keeping cash in short-term instruments and re-investing at whatever the policy rate is, over and over, for two years. For these to be roughly equivalent (which arbitrage forces them to be), **the 2-year yield must be approximately equal to the market's *average expected policy rate over the next two years*.** That is exactly the average of the front of the path.

So the 2-year yield is a single, real-time, deeply liquid distillation of the market's expected rate path. When the market prices more cuts, the average expected rate over two years falls, and the 2-year yield falls with it. When it prices cuts out, the 2-year rises. The 2-year does not wait for the Fed; it moves the moment the *expectation* moves. This is why traders call the 2-year "the policy-sensitive part of the curve" and watch it like a hawk — it is the market's path, priced in one number you can see on any screen.

![The 2-year Treasury yield peaks and falls before the Fed cuts, leading the policy rate](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-4.png)

Look at the chart above. The chunky gray step line is the realized policy rate — the thing the Fed actually delivered, moving only at meetings. The smooth blue line is the 2-year yield — the market's priced path. Notice three things. First, the 2-year *peaked* (at about 5.05% in October 2023) almost a full year *before* the Fed's first cut in September 2024. The market priced the top of the cycle long before the policy rate started coming down. Second, the gap between the blue line and the gray line — the 2-year sitting *below* the policy rate — is precisely the cuts the market has priced in. When the 2-year is well under the policy rate, the market is screaming "cuts are coming." Third, the blue line is far smoother and more continuous than the gray steps, because expectations update every day while the policy rate only moves eight times a year.

This is the mechanism behind the whole post: **because the 2-year (and every other asset that discounts future cash flows) is priced off the *expected path*, the price moves when the path reprices — not when the Fed acts.** The Fed acting is usually the anticlimax; the move already happened when the market changed its mind.

### Why the path reaches all the way into stocks

It is easy to see why the path moves bonds — a bond is literally a stream of fixed payments discounted by rates. It is less obvious why a tech stock with no near-term profits should care about the expected path of the overnight policy rate. The link is *discounting*, and it is the reason rate-path repricing moves the entire market at once.

Every asset is a claim on future cash flows, and every future cash flow is worth less today the higher the rate you discount it by. The rate you discount by is not one number; it is built up from the *whole expected path* — a far-future cash flow is discounted by the compounded sequence of expected short rates between now and then, plus risk premiums. So when the expected path lifts (higher for longer), the discount applied to *all* future cash flows rises, and assets whose value sits mostly in the far future — high-growth stocks, long bonds, anything "long duration" — fall hardest, because their distant cash flows get marked down the most. When the path bends down (a pivot), the same machinery runs in reverse and those assets re-rate up.

This is why a profitless software company can lose half its value when the path reprices higher even though its business has not changed: the *value* of its far-off future profits, discounted back to today by a higher expected path, simply shrank. It is also why, in 2022, stocks and bonds fell *together* — the rising path lifted the discount rate underneath both at once, breaking the usual diversification. The path is the tide; when it rises, it lifts the discount rate beneath every boat in the harbor, and the longest-duration boats sink first. Understanding this collapses what looks like a dozen separate markets into one: they are all pricing off the same expected path, so they all reprice when the path does.

#### Worked example: the path reprices from 3 cuts to 6 cuts and a bond rallies

Let us put real numbers on the late-2023 repricing. Entering the fourth quarter of 2023, the market had a fairly hawkish view: the 2-year yield was around **5.05%** (October 2023), reflecting expectations of "higher for longer" — maybe three cuts over the following year, with the terminal rate held for a while.

Then inflation cooled and the Fed sounded done. Over about two months, the market flipped to a dovish view, pricing roughly **six cuts** for 2024. The 2-year yield fell from 5.05% to about **4.25%** by year-end — a drop of **0.80%**, which is exactly the extra three cuts (3 × 0.25% = 0.75%, close enough) getting added to the expected path.

Now the bond P&L. Take a holder of a 2-year note. The price sensitivity of a bond to yield changes is its *duration*; a 2-year note has a duration of roughly 1.9 years. A bond's price moves approximately by (−duration × yield change):

```
price change ≈ -duration × Δyield
            ≈ -1.9 × (-0.80%)
            ≈ +1.52%
```

So the 2-year note gained about **1.5%** in price purely from the path repricing — and crucially, *the Fed had not cut at all yet.* The policy rate was still pinned at 5.50%. Put it in dollars to feel it: on a \$10,000,000 position in 2-year notes, a +1.5% price move is a gain of about **\$150,000**, earned entirely from the *expected* path shifting from three cuts to six — not from a single cut actually happening. Scale it down to a retail-sized \$25,000 position and the gain is about \$375. Either way, a trader who was long the 2-year (or long fed funds futures, or short the dollar, or long rate-sensitive stocks) made money on the *forecast changing*, not on any rate actually moving. **The takeaway: the path repricing is the trade; by the time the cuts the market priced actually arrive, the money has long since been made or lost.**

## "Higher for longer" versus the pivot

The two phrases you will hear endlessly in rates trading are "higher for longer" and "the pivot," and they are nothing more than two *shapes* of the expected path. Understanding them as path shapes — rather than as vibes — is what lets you trade them.

**Higher for longer** means the market expects the terminal rate to be *held at the plateau for an extended time*, with cuts pushed far into the future or made shallow. On the curve, this looks like a long flat top and a gentle, delayed descent. It is a *hawkish* path. When the market shifts toward higher-for-longer — say a hot inflation print convinces everyone the Fed cannot cut soon — the front of the path reprices *up* (cuts get pushed out and removed), the 2-year yield *rises*, bond prices *fall*, and because the discount rate underneath every asset stays high for longer, long-duration assets (profitless tech, long bonds, anything whose value is in far-future cash flows) get hit hardest. The 2023 selloff from July to October — when the 10-year went from below 4% to nearly 5% — was largely a higher-for-longer repricing.

**The pivot** is the opposite shape: the market expects the Fed to *bend the path downward sooner and faster*, pulling cuts forward and steepening the descent. On the curve, this looks like a steeper, earlier staircase down. It is a *dovish* path. When the market shifts toward a pivot — say inflation rolls over and a Fed official sounds soft — the front of the path reprices *down*, the 2-year yield *falls*, bond prices *rally*, and risk assets re-rate *up* because the discount rate is coming down. The late-2023 "everything rally" — stocks and bonds surging together into year-end — was a pivot repricing, the mirror image of the summer's higher-for-longer selloff.

![Higher for longer versus the pivot shown as two path shapes and their effect on bonds and stocks](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-5.png)

The crucial insight from the figure: **the terminal *level* can be identical in both scenarios — 5.50% either way — and the market reaction is entirely about the *shape of the descent*.** Same peak, different path down, opposite outcomes for your portfolio. This is the cleanest possible proof that markets trade the path, not the level: hold the level fixed, vary only the expected pace of cuts, and bonds and stocks move violently in opposite directions. If the level were what mattered, nothing would happen when the level is unchanged.

This also explains a phenomenon that confuses beginners: why bonds can rally *while the Fed is still hiking*, or sell off *while the Fed is cutting*. It is never about what the Fed did at that meeting; it is about whether the meeting moved the *expected path* relative to what was already priced. A "hawkish cut" — the Fed cuts but signals fewer cuts ahead than the market expected — can send the 2-year *up* and bonds *down*, even though the Fed just lowered rates. The realized move was a cut; the path repriced higher; the path won.

#### Worked example: the hawkish cut that sends bonds down

Picture a meeting where the policy rate is 4.50% and the market is confident the Fed will cut to 4.25% — a 0.25% cut is 100% priced. Going in, the curve also prices *four more cuts* over the following year, ending the path around 3.25%. The 2-year, reflecting that average, sits near 3.90%.

The Fed delivers the cut, exactly as priced. But in the press conference and the new dot plot, the Chair signals that inflation is stickier than hoped and the committee now sees only *two* more cuts ahead, not four. The path the market had priced was too dovish. Watch what happens:

```
priced before meeting:  cut now + 4 more cuts  -> path lands ~3.25%
signaled at meeting:    cut now + 2 more cuts   -> path lands ~3.75%
the path repriced UP by roughly 0.50% in the out-quarters
```

The market must now price *out* two cuts. The 2-year yield, which averages the path, *rises* — say from 3.90% to 4.15%, a +0.25% move. Using the 2-year's ~1.9 duration:

```
price change ≈ -1.9 × (+0.25%) ≈ -0.48%
```

The 2-year note *fell* about half a percent in price — on a day the Fed *cut rates*. On a \$5,000,000 position that is a loss of roughly **\$24,000**, booked the moment the path repriced, before the next meeting even arrived. Stocks likely fell too, because the discount rate over the next year just got marked up. A trader who heard "Fed cuts!" and bought bonds lost money; a trader who watched the *path repricing higher* and faded the dovishness made money. **The takeaway: the direction of a single rate move tells you nothing — what moves prices is whether the meeting pushed the expected path up or down relative to what was already priced.**

## The neutral rate and where the path lands

We have talked about where the path *peaks* (the terminal rate) and how fast it *descends* (the cutting pace). The last piece is where it ultimately *lands*. That destination has a name: the **neutral rate**, often written **r-star** (r*).

The neutral rate is the policy rate that is neither stimulating nor restraining the economy — the rate at which monetary policy is in balance, neither pressing the gas nor the brake. If the policy rate is *above* neutral, policy is "restrictive" — it is slowing the economy to fight inflation. If it is *below* neutral, policy is "accommodative" — it is goosing the economy. The Fed hikes *above* neutral to fight inflation, then, once inflation is beaten, cuts back *toward* neutral. **The neutral rate is therefore the gravitational center the whole path is pulled toward — the level the descent is aiming for.**

This matters for trading because the neutral rate sets the *floor* of the cutting cycle in the market's mind. When the market prices cuts, it is implicitly pricing a destination. If the market believes neutral is around 3%, it will price the path descending toward 3% and flattening there. If the market revises its estimate of neutral *up* — decides the economy can tolerate higher rates without inflating, perhaps because productivity or deficits are higher — then even with the same number of near-term cuts, the *whole tail* of the path lifts, and long-dated yields rise. A large chunk of the 2024 rise in long-term Treasury yields, even as the Fed cut, was the market marking up its estimate of the neutral rate.

The neutral rate is invisible — you cannot observe it directly, the Fed only estimates it (its "longer-run" dot in the projections sits around 2.5–3.0%), and reasonable people disagree by a full percentage point. That uncertainty is itself tradable: when the market's belief about neutral shifts, the long end of the curve moves even if the front end (the near-term cuts) does not. The terminal rate is about the *peak*; the neutral rate is about the *destination*; and the cutting cycle is the journey between them. A complete view of the path needs all three.

A useful way to keep these three straight is to think of the path as a trip over a mountain. The *terminal rate* is the summit — how high you climb. The *cutting cycle* is the descent — how quickly you come down the far side. The *neutral rate* is the valley floor on the other side — where the descent levels off. A trader with a view on the path is really taking a view on one of these three: that the summit is higher or lower than priced, that the descent is faster or slower than priced, or that the valley floor is higher or lower than priced. Each maps to a different part of the curve and a different instrument, which is exactly why step six of the playbook matters. Naming *which* of the three your edge is about — peak, pace, or destination — is half of building a good rate trade, because it tells you where on the curve to express it and what data will move it.

#### Worked example: the realized path — a 5.50% terminal and the cuts that followed

Let us ground all of this in the actual realized path of 2022–2024, because the numbers are clean and worth memorizing. The Fed funds *upper bound* moved as follows (these are the real effective dates):

```
Mar 2022  0.50%   first hike off the zero floor
Jun 2022  1.75%   the 75bp hikes begin
...
Jul 2023  5.50%   TERMINAL RATE reached — hiking stops
                  (held flat for ~14 months: the plateau)
Sep 2024  5.00%   first cut (a 50bp cut)
Nov 2024  4.75%   second cut (25bp)
Dec 2024  4.50%   third cut (25bp)
```

Read the structure. The hiking phase took the rate up **+5.25%** (from a 0.25% to a 5.50% upper bound) in about sixteen months — the fastest tightening in forty years. The terminal rate of 5.50% was then *held for roughly fourteen months* — that long plateau *is* "higher for longer" made real. Then the cutting cycle began with an unusually large 50-basis-point cut in September 2024, followed by two 25s, for **−1.00%** of cuts over the final four months of the year.

Now connect it to the opening story. Entering 2024, the market had priced *six* cuts for the year. The Fed delivered *three*. The realized path was far less dovish than the expected path the market had entering the year — and that gap is exactly why the 2-year yield round-tripped: it fell on six cuts being priced, rose as the realized data forced cuts to be priced *out* (the market overshot dovishly), then partly fell again as the cuts finally came. **The takeaway: the realized terminal rate (5.50%) and the realized cuts (three in 2024) are facts you can plot, but the trades all happened in the *gap* between this realized path and the expected path the market kept revising.**

![The realized fed funds path as a step chart with the 5.50 percent terminal plateau and the cuts marked](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-2.png)

The step chart above is the realized path in one picture: the climb to the 5.50% plateau (shaded amber), the long hold, and the first cut in September 2024. Memorize this shape. Every hiking cycle has roughly this silhouette — a climb, a peak, a plateau, a descent — and the market's entire job is to price the *next* version of this shape before it happens.

#### Worked example: the 2Y as a path proxy — reading cuts off the gap

One more piece of arithmetic, because it is the single most useful real-time read you can do. At any moment, compare the 2-year Treasury yield to the current effective fed funds rate. The gap is, roughly, the average size of the rate cuts the market has priced over the next two years.

Take a concrete snapshot. Suppose the effective fed funds rate is **4.33%** (a 4.50% upper bound) and the 2-year yield is **3.73%**. The gap:

```
fed funds (today)      = 4.33%
2-year yield (priced)  = 3.73%
gap                    = 0.60%   (2Y is 0.60% below the policy rate)
```

The 2-year is the *average* expected policy rate over two years. For the average over two years to sit 0.60% below today's rate, the path has to descend meaningfully — the rate has to spend a good chunk of those two years *below* today's level. Roughly, an average that is 0.60% below today implies the path ending something like **1.0% to 1.5% lower** than today by the end of the window (because the average of a descending path is higher than its endpoint). That is on the order of **four to six quarter-point cuts** priced over two years.

Now watch what happens when the 2-year moves. If a strong jobs report pushes the 2-year *up* from 3.73% to 3.98%, the gap shrinks from 0.60% to 0.35% — the market just priced *out* about one cut, and anyone long the front end lost money, all without the Fed doing anything. **The takeaway: the gap between the 2-year and the policy rate is a free, real-time gauge of how many cuts the market has priced; widening gap = more cuts being priced (dovish, bond-bullish), shrinking gap = cuts being removed (hawkish, bond-bearish).**

![The 2-year yield leading the policy rate lower from 2023 to 2025 with the lead marked](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-6.png)

## Common misconceptions

A few beliefs are so common and so wrong that naming them explicitly will save you real money.

**Misconception 1: "What matters is the current interest rate."** No — what matters is the *expected path*, and the current rate is already priced into everything. When the Fed cut for the first time in September 2024, bonds barely moved on the day, because that cut had been priced for months. The information was already in prices. If you are reacting to the rate the Fed *just set*, you are reacting to old news. The number to watch is not the policy rate; it is the *change in the expected path*, which you read off futures or the 2-year. In 2023, the policy rate was flat at 5.50% for months while the 2-year swung from 5.05% to 4.25% and back — all the action was in the path, none in the level.

**Misconception 2: "Rate cuts are always bullish for stocks."** This one has killed many traders. Cuts are bullish *if they come for the right reason and roughly as expected*. But cuts that come *because the economy is collapsing* — emergency cuts into a recession — happen alongside crashing earnings and rising default risk, and stocks usually fall *through* the cutting cycle. The Fed cut aggressively in 2001 and 2007–2008, and stocks were in the middle of brutal bear markets the whole time. The path matters, but *why* the path is bending down matters just as much. A dovish pivot driven by falling inflation (a "soft landing") is bullish; a dovish pivot driven by a cracking labor market (a "hard landing") is not. Same cuts, opposite outcome.

**Misconception 3: "The market predicts the path well."** It does not. The market is a *probability-weighted forecast*, and it is frequently, badly wrong — it priced six cuts for 2024 and got three; it spent most of 2021 pricing essentially no hikes right before the fastest hiking cycle in forty years. The curve is not a crystal ball; it is the *consensus bet*, and consensus bets are exactly the thing you can profit from when you have a differentiated view. The value of the curve is not that it is right — it is that it tells you precisely what is *priced*, so you know what you are betting against.

**Misconception 4: "A higher terminal rate is always worse for risk assets."** Not necessarily — what matters is the terminal rate *relative to neutral and relative to what was priced*. A terminal rate of 5.50% that the market already expected is benign; a terminal rate of 5.50% when the market had priced 4.50% is a brutal repricing. And a higher terminal rate that reflects a *stronger economy* (higher neutral rate, strong growth) can coexist with rising stocks, because the earnings outlook is improving alongside the discount rate. Level alone tells you nothing; the *surprise relative to what was priced* tells you everything.

**Misconception 5: "The first cut is the big event."** The first cut is usually the *least* important move of a cutting cycle, because it is the most anticipated and most fully priced. The big repricings happen *before* the first cut (as the market builds in the cutting cycle) and *during* the cycle (as the pace surprises). The first cut itself is typically a non-event for prices. Trading the headline "Fed cuts!" is trading the one moment when the information is already fully in the market.

## How it shows up in real markets

Two case studies, both with real dates and numbers, show the path-trading dynamic in its purest form.

### The 2022 terminal-rate repricing

In early 2022, the market badly underestimated how high the Fed would have to go. Entering the year, fed funds futures priced a terminal rate around 1.5–2%. Then inflation kept surprising upward — it hit 8.5% in March, then 9.1% by June, a 40-year high. Each hot print did not just add one hike; it forced the market to mark up the *entire peak* of the path. By the autumn, the priced terminal rate had marched all the way up toward 5%, and it eventually realized at 5.50%.

This was a *terminal-rate repricing*, and it was the dominant macro trade of 2022. The 2-year yield, which began 2022 at about 0.73%, climbed relentlessly to about 4.43% by year-end — a +3.7% move — as the market kept revising the peak higher. Because the discount rate underneath every asset was lifting, and lifting by *more than anyone had priced*, stocks, bonds, and crypto fell together. The classic 60/40 portfolio had one of its worst years in a century. The lesson: **when the *terminal* repricing is the story — when the market is revising the peak, not just the pace — the move is large, broad, and relentless, because the whole path lifts at once.** A trader's edge in 2022 was simply to keep believing the peak was higher than priced, meeting after meeting, and to stay short the front end and short duration.

### The 2024 cut-pricing roundtrip

The opposite character of trade played out in 2024 — not a terminal repricing but a *pace* repricing, and a round trip at that. As we have seen, the market entered the year pricing six cuts. Through the first half, sticky inflation forced those cuts to be *priced out*: by April, the 2-year had climbed back to about 4.99% from its 4.25% December low, as the market gave up on the early, aggressive cuts. Then, in the second half, cooling data and a softening labor market brought the cuts back: the 2-year fell to about 3.66% by September, and the Fed finally delivered its first (50bp) cut.

The full-year picture for the 2-year was a *V*: down on the dovish dream, up as reality bit, down again as the cuts arrived. The policy rate, meanwhile, was a flat line at 5.50% until September. **Every dollar of P&L in the front end that year came from the path repricing back and forth, not from the level, which barely moved until the fourth quarter.** A trader who understood this did not try to guess the terminal rate (it was set); they traded the *gap* between the cuts the market had priced and the cuts the data would actually support — fading the six-cut euphoria in January, then fading the no-cut pessimism in the spring. The number to watch all year was the 2-year, and the question was always the same: *is the priced number of cuts too high or too low given the data?*

## How to trade it: the playbook

Everything above converges on one discipline. You are not forecasting the interest rate. You are trading the *gap* between two things: the path the market has *priced* (which you read off the futures curve and the 2-year) and the path the Fed's *reaction function* implies (what the Fed will actually do given the incoming inflation and jobs data). When those two disagree, there is a trade.

![The path-trading playbook comparing the priced path to the reaction function with the 2-year as the dial and the invalidation](/imgs/blogs/terminal-rate-and-rate-cut-cycles-pricing-the-path-7.png)

Here is the playbook, step by step.

**1. Read what the curve has priced.** Pull up the fed funds / SOFR strip (or just the 2-year versus the policy rate) and state the priced path as a sentence: "the curve prices N cuts over the next twelve months, with the terminal rate held until month M." Make it falsifiable. The 2-year-minus-fed-funds gap from the worked example is your quick gauge — a wide gap means many cuts are priced; a narrow or negative gap means few or none.

**2. Form a view on the reaction function.** The Fed's reaction function is its rule of thumb: cut when inflation is falling toward 2% *and* the labor market is softening; hold or hike when inflation is sticky or the labor market is hot. Ask whether the incoming data supports *more* cuts than the curve prices, or *fewer*. This is where you need a real view on inflation and employment — the rest of this series is largely about building that view. You are comparing two paths: priced versus warranted.

**3. Trade the gap with the 2-year as the dial.** If you think the curve prices *too many* cuts (it is too dovish versus what the data will support), the cuts will get priced *out*, the 2-year will *rise*, and you want to be **short the front end** — short fed funds/SOFR futures, short the 2-year, or long the dollar. If you think the curve prices *too few* cuts (too hawkish), the 2-year will *fall* and you want to be **long the front end**. The 2-year is your dial: it moves the instant the expected path reprices, so it is both your signal and, often, your instrument. You are positioning for the *path to reprice toward your view*, and you make money as the gap closes.

**4. Size it by conviction and define the invalidation.** This is the part amateurs skip. Your view is a bet that the priced path is wrong; the *data* is the referee. Before you put the trade on, name the print that proves you wrong: "I am short the front end because I think inflation stays sticky and cuts get priced out; if core CPI prints below 0.2% month-over-month for two consecutive reports, my thesis is dead and I cover." A path trade *must* have a data-defined invalidation, because the whole position is a claim about what the data will force the Fed to do. When the invalidating print lands, the path will reprice *against* you fast, and you want to be out before it does.

**5. Respect the asymmetry of the cycle's stage.** Where you are in the cycle changes the trade. Near a *suspected terminal rate*, the asymmetry favors fading additional hikes (the Fed is almost done, so priced hikes are likely to come out). Deep into a *cutting cycle*, watch for the pace surprising — and watch *why* the cuts are coming, because cuts into a recession (Misconception 2) mean the stock trade and the bond trade diverge violently. And always remember the first cut is the anticlimax; the money is made pricing the cycle *before* it starts.

**6. Choose the cleanest expression for your conviction.** The same path view can be expressed in several instruments, and the right one depends on what you are confident about. If your view is purely about *near-term cuts being mispriced*, the 2-year or the front fed funds/SOFR contracts are the cleanest — they isolate the front of the path with almost no other noise. If your view is about the *terminal rate* or the *neutral rate* (the peak or the destination), you want instruments further out the curve, where those beliefs live. If you are confident about the *direction* but not the magnitude, the foreign-exchange market often gives you a higher-conviction, lower-cost expression: a path that reprices more dovishly than other countries' paths tends to *weaken the dollar* (lower relative yields make the currency less attractive), so being short the dollar is a softer way to play a US dovish repricing. And if your view is really about *risk assets* — a soft-landing pivot that lifts stocks — you can express it in equities, but only if you are sure the pivot is the benign kind (Misconception 2). The discipline is to match the instrument to the part of the path you actually have an edge on, rather than reflexively trading whatever is in front of you. In every case, though, the 2-year remains your *dial* — the single number that tells you, in real time, whether the path is repricing toward your view or against it.

The discipline in one line: **find where the curve's priced path disagrees with the path the data will force, express it through the 2-year, and let a specific data print invalidate you.** You will never know the future policy rate. You do not need to. You only need to know whether what is *already priced* is too much or too little — and that is a question you can actually answer.

## Further reading & cross-links

- [Trading the FOMC: the statement, the presser, and the dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) — how the Fed communicates the path it intends, and how the market reprices around each meeting.
- [Interest rates: the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — why the discount rate sits underneath every asset, the foundation for why path repricing moves everything.
- [Reading the yield curve: slope, inversion, and recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how the 2-year, the 10-year, and the slope between them encode the expected path and the growth outlook.
- [The central-bank balance sheet: net liquidity, reserves, RRP, and the TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) — the other policy lever, and how liquidity interacts with the rate path.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the plumbing behind the policy rate the whole path is anchored to.
- [Paul Volcker and the 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the most extreme terminal rate in modern history and what it took to break inflation.

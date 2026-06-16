---
title: "How Monetary Policy Moves Currencies: Rate Differentials and the Carry Trade"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Policy moves a currency mainly through the rate differential — the faster-hiking central bank's currency offers a higher yield, pulls in capital, and appreciates; read the relative policy path and you read the currency."
tags: ["macro", "monetary-policy", "foreign-exchange", "rate-differentials", "carry-trade", "interest-rate-parity", "dollar", "yen", "dxy", "currencies", "central-banks", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Monetary policy moves a currency mostly through one channel: the **rate differential**. When a central bank hikes faster than its peers, its currency offers a higher yield and stronger expected returns, so global capital flows in and the currency appreciates.
>
> - **The faster hiker wins.** It is not the level of rates that matters, it is the *gap* between one central bank's path and its peers'. A currency strengthens when its central bank out-hikes the rest of the world.
> - **The carry trade is the mechanism.** Borrow a low-rate currency, hold a high-rate one, and collect the spread every day — until the gap narrows and everyone exits at once.
> - **Real beats nominal.** A currency rewards a *real* yield advantage (rate minus expected inflation), which is why a 5% rate with 8% inflation is not bullish for a currency at all.
> - **The one number to remember:** in 2022 the Fed out-hiked nearly the entire world and the US Dollar Index (DXY) rose **8.2%** — the wrecking ball that crushed the yen, the pound, and emerging markets in a single year.

In 2022, the most important price in global finance was not a stock, a bond, or a barrel of oil. It was the US dollar. Over twelve months the dollar rose against almost every currency on earth — the euro briefly fell *below* parity for the first time in twenty years, the British pound crashed toward \$1.03 after a botched budget, the Japanese yen lost a quarter of its value, and central banks from Cairo to Buenos Aires watched their currencies buckle. The US Dollar Index, which measures the dollar against a basket of major peers, climbed from about 96 at the start of the year to an intraday peak of **114.8** on 28 September 2022 — a twenty-year high — and closed the year up roughly 8%.

Nothing about the United States had suddenly become 8% more valuable. American factories were not 8% more productive; the trade deficit was, if anything, getting worse. So what drove the dollar to a generational high? One thing, almost entirely: the Federal Reserve was hiking interest rates faster and harder than nearly every other major central bank. The Fed took its policy rate from near zero to 4.5% in nine months, while the Bank of Japan held its rate at essentially zero, the European Central Bank moved late and slowly, and most emerging-market central banks could not keep pace. The dollar offered a rising yield while its peers did not. Capital chased that yield. The dollar went up.

This post is about that mechanism, built from absolute zero. We are going to answer one question with a precision the financial press rarely bothers with: **how, exactly, does a central bank's interest-rate policy move its currency?** The headline answer is blunt — *policy moves a currency mainly through the rate differential* — and by the end you will be able to look at two central banks' projected paths and read which way the exchange rate between them is being pulled, the way a sailor reads the wind.

![Rate differential channel central bank hikes raise yield capital flows in currency appreciates](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-1.png)

The figure above is the whole post on one page, and we will earn the right to read it left to right. A central bank hikes; the gap between its rate and its peers' widens; its bonds and deposits now pay more; global capital flows in to capture that yield; demand for the currency rises; the currency appreciates. Six boxes. Hold them in your head — everything else is mechanism, exceptions, and how to trade it.

## Foundations: the exchange rate, the rate differential, and interest-rate parity

Before we can talk about what *moves* an exchange rate, we have to be ruthless about what an exchange rate *is*, what a rate differential *is*, and why a no-arbitrage relationship called interest-rate parity ties them together. None of this requires finance background — only the willingness to think about money the way a trader does.

### An exchange rate is a relative price

An exchange rate is just the price of one currency expressed in another. Like any price, it is a ratio: how much of *this* money you pay for one unit of *that* money. The market convention writes a pair as `BASE/QUOTE`, and the number tells you **how many units of the quote currency buy one unit of the base.** For `USD/JPY`, the dollar is the base and the yen is the quote, so `USD/JPY = 150` means **one US dollar costs 150 yen.** The dollar is the thing being priced; the yen is the money you pay with.

That one sentence — "one dollar costs 150 yen" — is the most useful phrase in all of FX. Translate every quote into "one [base] costs [number] [quote]," and you will never get the direction backwards. If `USD/JPY` rises from 150 to 160, one dollar now costs *more* yen, so the dollar got **stronger** (appreciated) and the yen got **weaker** (depreciated). A *rising* USD/JPY is a *weakening* yen — which trips up every beginner, because the chart going up *sounds* like it should be good for the yen. It is not.

Here is the everyday-money version. Treat a dollar like an imported gadget and the yen like the local currency you buy it with. If the gadget cost 150 units of local money last month and 160 this month, the gadget got more expensive — the foreign thing (the dollar) went up, the local money (the yen) went down. The number rising means the base currency strengthened.

One more piece of vocabulary, because it is load-bearing for the rest of the post. A currency is a relative thing: it can only be strong or weak *against something else*. There is no such thing as the dollar being "strong" in a vacuum; it is strong against the yen, the euro, the peso. This is why the US Dollar Index (DXY) exists — it bundles the dollar against six major currencies (heavily weighted to the euro) into one number so traders can talk about "the dollar" as a single thing. When we say "DXY rose 8.2% in 2022," we mean the dollar rose 8.2% against that basket on average.

### The dollar as the global benchmark

There is a reason this post keeps circling back to the dollar, and it is not American chauvinism. The dollar is the *benchmark* currency of the entire system — the unit in which most of the world's trade is invoiced, most cross-border debt is issued, and most central-bank reserves are held. Roughly 58% of disclosed global FX reserves are still held in dollars; the next-largest, the euro, is around 20%. When you trade almost any currency, you are implicitly trading it *against the dollar*, because the dollar is on the other side of the most liquid pairs. EUR/USD, USD/JPY, GBP/USD, USD/CNY — the dollar is in all of them.

This benchmark status has a profound consequence for how policy moves currencies: **US monetary policy is the gravitational center, and every other currency's rate differential is measured, first and foremost, against the dollar.** When the Fed hikes, it does not just move the dollar against one peer; it moves the dollar against *everything at once*, which is why a single central bank's policy can reorder the entire FX market. A trader who wants to understand any currency starts by asking what it is doing *relative to the dollar*, and what the dollar is doing relative to *its* policy path. We will treat the dollar as the reference point throughout, not because other rate differentials don't matter, but because the dollar's is the one that radiates out to all the others.

### How a policy rate actually reaches the currency

It is worth pausing on the precise chain by which a central bank's single administered rate ends up moving a price set in a global market the central bank does not control. The policy rate — the federal funds rate in the US — is the rate at which banks lend reserves to each other overnight. The central bank sets it directly. But the currency is priced by millions of participants buying and selling in a \$7.5-trillion-a-day market. How does one connect to the other?

Through a chain of substitution. The overnight policy rate anchors the entire short end of the yield curve, because no one will lend for three months at less than they can earn rolling overnight deposits. The three-month rate anchors the two-year, the two-year shapes the ten-year, and the whole curve of government yields shifts when the policy path shifts. Those government yields are the *risk-free return on holding that currency* — the baseline an international investor compares across countries. When US yields rise relative to Japanese yields, holding dollars becomes more rewarding relative to holding yen, and the global pool of mobile capital re-weights toward dollars. To re-weight toward dollars, it must *buy* dollars in the FX market. That buying is the transmission: the administered overnight rate moves the curve, the curve moves the relative return on the currency, and the relative return moves the flows that set the price. The central bank never touches the exchange rate directly; it moves the *incentive* to hold the currency, and the market does the rest.

### The rate differential: the gap that matters

Every currency has an interest rate attached to it, set by its central bank. Park money in dollars and you can earn the US short-term rate; park it in yen and you earn the Japanese rate. The **rate differential** is simply the difference between two currencies' interest rates — for example, the US rate minus the Japanese rate.

This is the single most important number in this entire post, so let us be precise about why. Money is mobile. A global investor with a billion dollars can hold it in any currency. If dollars pay 5.3% a year and yen pay 0.1% a year, the dollar offers a **5.2-percentage-point yield advantage** — and that advantage is a magnet. All else equal, money flows toward the higher yield. To hold dollars, you must *buy* dollars, and buying dollars pushes the dollar up.

So the thing that moves a currency is not the *level* of its interest rate in isolation. It is the *differential* — how its rate compares to the rest of the world's. A central bank with a 5% rate is not automatically attached to a strong currency; if every peer also sits at 5%, there is no differential and no pull. The currency strengthens when its rate *rises relative to peers* — when its central bank out-hikes the others. That relative move is what the entire post is about.

### Interest-rate parity: covered and uncovered

There is a no-arbitrage relationship that ties the rate differential to the exchange rate, and it has a clean name: **interest-rate parity (IRP)**. It comes in two flavors, and the difference between them is where the whole story lives.

**Covered interest parity (CIP)** is an iron law, true by arbitrage. Imagine you have \$1,000,000 and a one-year horizon. You can do one of two things. Path A: keep the money in dollars and earn the US rate, 5.3%. Path B: convert the dollars to yen today at the spot rate, earn the Japanese rate, 0.1%, and *lock in today* the rate at which you will convert the yen back to dollars in a year. That locked-in future rate is called the **forward rate**. CIP says these two paths must produce the same dollar amount — otherwise there would be a free lunch, a riskless profit anyone could scoop up until the prices adjusted. Because Path A clearly earns more interest, the forward rate must be set so that the yen "gains back" exactly the interest shortfall. The forward rate is *pinned* by the rate differential. No guessing, no forecasting — it is mechanical.

![Covered interest parity no-arbitrage box two paths must end at the same value](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-3.png)

The figure shows the no-arbitrage box. Both paths start at \$1,000,000 and must end at the same place, \$1,053,000. That constraint forces the forward rate to `150 × (1.001 / 1.053) = 142.6` yen per dollar. The high-rate currency (the dollar) trades at a *forward discount* — it is cheaper to buy for future delivery — precisely because it pays more interest today. This is not a theory; it is enforced minute by minute in the multi-trillion-dollar FX swap market.

**Uncovered interest parity (UIP)** is the famous, and famously wrong, half. UIP says: if the dollar pays 5.2% more than the yen, then the *spot* exchange rate "should" move so the dollar *depreciates* by about 5.2% over the year — exactly offsetting the interest advantage, so that an investor ends up indifferent between the two currencies. In theory, the higher-yielding currency is expected to weaken by the size of the rate gap, and the two effects wash out.

Here is the punchline that makes traders rich and textbooks uncomfortable: **UIP reliably fails over the horizons traders care about.** Higher-yielding currencies do *not*, on average, depreciate by their rate advantage. Often they *appreciate*, because the same capital flows that chase the yield also bid up the currency. The gap between what UIP predicts (the high-yielder should weaken) and what actually happens (it often strengthens, and at minimum you pocket the interest) *is the carry trade's profit*. We will spend a whole section on this, because it is the engine of the biggest FX moves on the planet.

The mental model to carry forward: **CIP pins the forward rate exactly; UIP is supposed to pin the spot move but doesn't — and that failure is the opportunity.**

One practical bridge between the two, because traders meet it daily: **forward points.** The difference between the forward rate and the spot rate is quoted as "forward points," and by CIP those points are nothing but the rate differential expressed in exchange-rate terms. A high-yield currency trades at a forward *discount* (you can buy it more cheaply for future delivery), and that discount is exactly the carry you earn by holding it. So when a carry trader "earns the carry," the bookkeeping is precise: they are capturing the forward points, the same rate differential, just priced into the forward curve instead of paid as an interest coupon. This is why you cannot escape the carry by using forwards instead of spot — the forward already has the differential baked in. The only way to *make* money on the trade beyond the locked-in points is for the *spot* rate to behave better than UIP says it should, which, empirically, it usually does. That empirical "usually" is the entire edge, and it is also the entire risk.

## The policy-divergence channel: relative hiking paths drive FX

Now we can state the central mechanism precisely. A currency is moved by monetary policy through **policy divergence** — the difference between the path its central bank is on and the paths of its peers. Not the level. The *relative path*.

Think about what a central bank actually controls. It sets the short-term policy rate (in the US, the federal funds rate) and signals where that rate is heading. Everything downstream — the yield on a two-year government bond, the rate on a bank deposit, the return on holding that currency overnight — keys off that policy rate and the expected path of future policy rates. So when the Fed says "we are taking rates from 0.25% to 5.5% and holding them there," it is simultaneously raising the yield on every dollar-denominated asset *and* the yield on simply holding dollars. If, at the same moment, the Bank of Japan says "we are staying at zero," the *differential* between holding dollars and holding yen blows out. Capital does what capital does: it moves toward the higher return.

Critically, currencies are forward-looking. The exchange rate today reflects not just today's rate gap but the *expected* gap over the coming months and years. This is why a currency can jump on a central bank *meeting* even if the rate doesn't change — if the bank signals it will hike *more* than the market expected, the expected future differential widens, and the currency appreciates *immediately*, before a single additional hike has happened. The FX market trades the *path*, not just the *level*.

This forward-looking quality is the source of the most important practical rule in FX trading: **the currency moves on the *surprise*, not the *fact*.** If everyone already expects the Fed to hike 25 basis points and the ECB to hold, that expected divergence is *already in the price*. The currency moves when reality differs from the expected path — a hike that's bigger than priced, a dot plot that drifts higher, an inflation print that forces the market to add hikes to the curve. This is why you will see a central bank raise rates and the currency *fall*: if the market had priced a larger hike, the actual smaller hike is a *dovish surprise* relative to expectations, and the expected differential *narrows*. The level went up; the surprise was down; the currency fell. Beginners who watch only the headline rate get whipsawed by this constantly. The professionals are watching the gap between what happened and what was priced.

Two second-order effects make the divergence channel even more powerful than the simple story suggests. First, the move *feeds the flows that drive it*. As the dollar strengthens on a widening rate gap, dollar-denominated assets — Treasuries, US stocks — become more attractive to foreign investors who now also expect currency gains, so capital chases both the yield *and* the appreciation, which strengthens the dollar further. Momentum builds. Second, a strengthening dollar *tightens financial conditions globally*, which pressures other central banks to hike to defend their own currencies — and if they *can't* (because their economies are too weak), their currencies fall further, widening the very gap that started the move. Divergence, once underway, tends to *self-amplify* until something forces convergence. That is why dollar cycles, once they get going, tend to run for years rather than weeks — the 2022 surge was the violent leg of a multi-year strong-dollar regime, not a one-month spike.

![US Japan rate gap step chart versus USD JPY dual axis 2020 to 2026](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-2.png)

The figure makes the channel concrete with the cleanest natural experiment of the decade: the US versus Japan. The blue step line is the Fed funds rate marching from 0.25% in early 2022 to 5.50% by mid-2023. The dashed line is the Bank of Japan, pinned near 0.10% the entire time. The amber shaded area between them *is* the rate differential — and watch the red line, USD/JPY, track it almost perfectly. As the gap opened from near zero to about five points, the yen fell from 115 to the dollar all the way to 157. The yen did not weaken because Japan's economy collapsed. It weakened because the Bank of Japan refused to follow the Fed, and the rate gap pulled capital out of yen and into dollars, day after day.

#### Worked example: the rate differential as a carry on a \$1,000,000 position

Let us make the channel pay in dollars. Suppose you are a global macro fund. You borrow Japanese yen — the funding currency — at Japan's short rate of **0.1%** a year. You convert that borrowed yen into US dollars at the spot rate and park the dollars in US Treasury bills earning **5.3%** a year. The size of the position is **\$1,000,000**.

What do you earn just from the rate gap, before the exchange rate moves at all?

```
us_rate     = 0.053
jpy_rate    = 0.001
notional    = 1_000_000

rate_gap    = us_rate - jpy_rate          # 0.052, i.e. 5.2%
annual_carry = rate_gap * notional        # 52000.0
daily_carry  = annual_carry / 365         # 142.47
```

The rate gap is `0.053 − 0.001 = 0.052`, or **5.2%**. On a \$1,000,000 position that is `0.052 × 1,000,000 = \$52,000` a year — a steady **\$142 a day**, dripping in whether the exchange rate moves or not. You collected \$52,000 for doing nothing but holding the higher-yielding currency and funding it in the lower-yielding one. **The rate differential is not a textbook concept; it is a cash yield you pocket every single day the gap stays open.**

#### Worked example: policy divergence and the yen, 115 to 157

Now layer the exchange-rate move on top. Over 2021–2024, the Fed hiked aggressively and the Bank of Japan held. `USD/JPY` went from **115** at the end of 2021 to **157** by the end of 2024. What did that do to the dollar and the yen?

The rate is yen-per-dollar, and it rose from 115 to 157. A higher yen-per-dollar means each dollar now buys more yen, so **the dollar appreciated and the yen depreciated.** The magnitudes are not symmetric, because each is measured against a different base:

```
s0 = 115
s1 = 157

dollar_move = (s1 - s0) / s0 * 100      # +36.52%
yen_move    = (s0 - s1) / s1 * 100      # -26.75%
```

From the dollar's side, it went from buying 115 yen to buying 157 yen — a **+36.5%** appreciation. From the yen's side, it lost **−26.8%** of its value against the dollar. A trader who was long the dollar against the yen — long USD/JPY — over that stretch captured both the **5%-ish annual carry** *and* a **36%** spot appreciation. **When policy diverges and stays diverged, the currency move and the carry compound in the same direction — and that is why the faster-hiking central bank's currency appreciates.**

This is the whole policy-divergence channel in one trade. The faster hiker's currency goes up for two reinforcing reasons: you earn its higher yield (the carry), and the capital chasing that yield bids the currency up (the spot move). Both legs point the same way as long as the divergence persists.

## The carry trade: borrow low, hold high, collect the spread

We have now used the phrase "carry trade" several times. Let us build it from the ground up, because it is the single most important expression of how monetary policy moves currencies — and the source of the most violent FX accidents in markets.

The **carry trade** is the strategy of *borrowing a low-interest-rate currency and using the proceeds to hold a high-interest-rate currency (or asset).* The low-rate currency is called the **funding currency** — for years this was the yen, sometimes the Swiss franc, more recently anything where the central bank is on hold while others hike. The high-rate currency is the **target** or **investment currency**. The difference in their interest rates is the **carry** — the yield you collect for holding the position open.

Why does this work? Go back to uncovered interest parity. UIP *says* the high-yield currency should depreciate by the rate gap, leaving you indifferent. In practice, it usually doesn't — over months and years, high-yield currencies tend to hold their value or even strengthen as capital floods in. So you collect the rate gap as nearly free yield, and frequently get a currency gain on top. Decades of data show carry trades earn a positive average return with a particular, sinister shape: small, steady gains most of the time, punctuated by rare, enormous losses. Traders describe it as **"picking up nickels in front of a steamroller."** You pocket a nickel a day, every day — until the day the steamroller arrives.

![Carry trade earns the rate gap daily then the unwind reverses every leg](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-6.png)

The figure lays the two states side by side, because they are the *same position run in reverse*. On the left, the carry: borrow yen at 0.1%, convert to dollars, hold dollars at 5.3%, and collect ~5.2% a year as a slow daily drip — with the bonus that the yen is drifting weaker, so the FX move *adds* to the gain. On the right, the unwind: a shock arrives (the Bank of Japan finally hikes, or US data softens), the rate gap that paid the carry starts to shrink, everyone sells dollars and buys yen back to repay their loans at once — and because the trade was crowded, the exit is too small for the crowd. The yen spikes, the gain reverses, and years of accumulated nickels vanish under the steamroller.

The reason the unwind is so violent deserves a beat. The carry trade is *self-reinforcing on the way up*: as more funds borrow yen to buy dollars, they sell yen and buy dollars, which weakens the yen further, which makes the trade look even better, which attracts more funds. The position gets crowded. Everyone is on the same side. When the catalyst flips — even a small one — every participant tries to close at the same moment: buy back yen, sell the dollar assets. The same feedback loop that pushed the yen down now whips it up, fast. This is why FX volatility tends to *explode* on carry unwinds while it merely simmers on the way up. The classic phrase is **"the dollar takes the stairs up and the elevator down"** when it is the funding side; for the high-yielder it is the reverse — slow grind up, violent drop.

A subtle but crucial point: the carry trade is *information* about policy expectations. When the carry on a currency pair is large and the trade is crowded, it tells you the market is heavily positioned for the rate gap to *persist*. The risk, then, is not that the gap widens — that is already priced and you are paid for it — but that the gap *narrows faster than expected*. That is what you are really short when you put on a carry trade: you are short "policy convergence." Keep that framing; it tells you exactly what to watch.

#### Why the yen is the funding currency

Why has the yen, specifically, been the world's favorite thing to borrow for decades? Because the Bank of Japan has run the lowest policy rate in the developed world for the better part of thirty years. After Japan's asset bubble burst in 1990, the BOJ cut to zero, stayed there, and for years even ran *negative* rates — charging banks to hold reserves. Through that whole stretch, borrowing yen cost almost nothing. So the yen became the natural funding leg: you could borrow it for near 0%, convert to almost any other currency, and capture that currency's higher yield. The Swiss franc has played the same role for similar reasons. The funding currency is simply *whichever major currency has a central bank pinned at the bottom while others sit higher* — and for a generation, that has overwhelmingly been the yen. This is also why yen-pair volatility is the market's single best barometer of carry-trade stress: when USD/JPY lurches, it is usually the carry crowd moving, not Japan's fundamentals.

#### Leverage turns the nickel into a dime — and the steamroller into a freight train

The carry on a currency pair is small in absolute terms — 5% a year is not exciting for a hedge fund. So carry trades are almost always *levered*. Because the FX swap and forward markets let you control a large notional with a small amount of capital, a fund might run a carry position at five or ten times its capital. At 10x leverage, a 5% carry becomes a 50% return on capital — now it's exciting. But leverage cuts both ways with brutal symmetry: a 10% adverse spot move, which would be an annoying drawdown unlevered, becomes a *100% wipeout* at 10x. This is the deeper reason carry unwinds are catastrophic rather than merely painful. The crowd is not just long the carry; it is *levered* long the carry. When the unwind starts, levered positions hit margin calls, forced to sell *into* the move, which deepens the move, which triggers more margin calls. The leverage is the accelerant that turns a normal repricing into a cascade. The steady daily drip lulls people into adding leverage in the calm; the calm is exactly what makes the eventual unwind so violent.

#### The math of the skew: why it is selling insurance

There is a precise way to see why carry "works" yet is dangerous, and it is worth internalizing. Over any long sample, the carry trade has a *positive average return* but a *negative skew*: lots of small positive months, a few enormous negative ones. The average is positive because UIP fails — you keep the rate gap more often than not. But the distribution is lopsided because the losses, when they come, are concentrated in rare, severe risk-off episodes when *everyone* unwinds together. Statistically, this is the same return profile as *selling insurance* or *selling out-of-the-money options*: you collect a steady premium, and occasionally you pay a huge claim. The carry trader is, in effect, short global volatility. This is why carry returns and equity-market crashes are correlated — both blow up in the same risk-off moment — and why a carry book provides *no diversification* exactly when you need it most. The positive average return is real; it is your compensation for warehousing that tail risk. The mistake is treating the average as the whole story and the tail as a freak event. The tail is not a freak; it is the price of the premium.

## Real versus nominal differentials: the signal underneath the signal

So far we have talked about *nominal* interest rates — the headline rate a central bank sets, the number on the bond. But a currency does not ultimately reward a high nominal rate. It rewards a high *real* rate — the nominal rate minus expected inflation. This distinction separates traders who understand FX from those who get run over by it.

Here is the reasoning. Suppose Country A offers a 5% nominal rate but has 8% inflation, while Country B offers a 2% nominal rate with 1% inflation. Naively, A looks like the high-yielder — 5% beats 2%. But A's *real* rate is `5% − 8% = −3%`, while B's is `2% − 1% = +1%`. Hold A's currency and your purchasing power is *shrinking* by 3% a year even after collecting the interest; the currency is being inflated away faster than the yield compensates. Hold B's and you are actually getting ahead. Over time, capital recognizes this. The currency with the higher *real* rate — the one that preserves and grows purchasing power — is the one that attracts durable inflows.

This is why emerging-market currencies with eye-watering nominal rates — 15%, 20%, 40% — are not automatically a great carry. Those high rates usually exist *because* inflation is high; the real rate may be small, zero, or negative, and the currency is steadily depreciating to reflect it. The nominal carry looks juicy and the spot bleeds it all away. The trade that works is the one where the *real* differential is wide and positive, because that reflects a central bank that has genuinely gotten ahead of inflation — and *that* is what durable capital chases.

There is a further wrinkle that separates good FX traders from great ones: it is not just the *current* real differential that matters, but the *expected path* of real rates. A central bank that is *raising* its real rate — getting genuinely ahead of inflation, hiking nominal rates faster than inflation is rising — is doing something a currency rewards richly, because it signals credible inflation-fighting and a widening real-yield advantage. A central bank with a high but *falling* real rate (inflation catching up to its nominal rate) is doing the opposite, even if the snapshot looks attractive. The 2022 dollar is the case study: the Fed was not just running a positive real rate, it was *driving its real rate up hard* as it hiked into still-elevated inflation, and the market rewarded that trajectory. The signal is the *direction* of the real differential, not only its level.

For a deeper treatment of why the real yield is the cleaner signal across all of macro — not just FX — see [real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). The short version for currencies: when you compare two central banks' paths, subtract each country's expected inflation from its nominal rate. The currency you want to be long is the one whose central bank is delivering — and is expected to keep delivering — the higher *real* policy rate, not merely the higher *nominal* one.

#### Worked example: the real-rate trap on a nominal-carry trade

Suppose you are tempted by a currency offering a **12%** nominal one-year rate while the dollar offers **5%**. The nominal differential is a mouth-watering **7 points**. You consider borrowing dollars and holding the high-yielder for the carry. But that country's expected inflation is **14%**, while US expected inflation is **3%**. What is the *real* differential — and what is it telling you?

```
em_nominal = 0.12
us_nominal = 0.05
em_infl    = 0.14
us_infl    = 0.03

nominal_gap = em_nominal - us_nominal     # 0.07  -> +7.0%
em_real     = em_nominal - em_infl        # -0.02 -> -2.0%
us_real     = us_nominal - us_infl        # +0.02 -> +2.0%
real_gap    = em_real - us_real           # -0.04 -> -4.0%
```

The nominal gap is `+7%` — it looks like a fantastic carry. But the high-yielder's *real* rate is `12% − 14% = −2%`, while the dollar's real rate is `5% − 3% = +2%`. The *real* differential is `−2% − 2% = −4%` — it points the *wrong way*. The high nominal rate is just compensation for inflation eating the currency; on a real basis, the dollar is the higher-yielder. A trader who chases the 7% nominal carry is likely to watch the currency depreciate by *more* than 7% as inflation grinds it down. **A high nominal rate with higher inflation is a trap; the currency rewards the real differential, and on a \$1,000,000 position that −4% real gap can cost you \$40,000 a year that the headline 7% never warned you about.**

## Common misconceptions

Three beliefs about currencies and policy are widely held, intuitive, and wrong. Each one costs traders money. Here is the correction, with a number.

**Misconception 1: "A strong economy means a strong currency."** It feels obvious — good economy, good currency. But the link runs through *policy*, not directly through growth. A strong economy matters for the currency mainly because it lets the central bank *keep rates higher*. If a booming economy has a central bank that refuses to hike (Japan for most of the 2010s and 2020s), the currency can be *weak* despite the boom. Conversely, a *weak* economy whose central bank is forced to hike to fight inflation can have a *strong* currency in the short run. In 2022 the US economy was slowing and recession fears were everywhere — yet the dollar rose **8.2%**, because the Fed was hiking hardest. The currency follows the *rate path*, and the economy only matters insofar as it shapes that path. For how growth, inflation, and the labor market feed the policy path, see [what moves exchange rates: rates, flows, and carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry).

**Misconception 2: "Carry is free money."** The carry trade collects a positive yield most of the time, which seduces people into treating it as a riskless coupon. It is not. The return distribution is brutally skewed: years of small steady gains, then a single unwind that erases them in days. When the yen-carry unwound in early August 2024, USD/JPY fell from a **161.9** July peak toward **145** in a couple of weeks — a roughly 10% move that wiped out about *two years* of a 5% carry, and the associated risk-asset selloff sent Japanese stocks down 12.4% in one session. Carry is not free money; it is *insurance you are selling*, and occasionally the claim comes due all at once.

**Misconception 3: "The Fed only matters for the US."** Because the dollar is the world's reserve and invoicing currency, US monetary policy is *global* monetary policy. When the Fed hikes, the dollar strengthens against nearly everything, which imports inflation into other countries (their imports, priced in dollars, get more expensive), forces *their* central banks to hike to defend their currencies, and tightens financial conditions worldwide through the vast pool of dollar-denominated debt. In 2022 the Fed's hiking cycle effectively *exported* a currency crisis: emerging-market central banks had to hike into weakening economies just to slow their currencies' slide against the dollar. The Fed sets the tide for *everyone*. For why the dollar sits at the center of the whole system, see [the dollar system: why the USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).

**Misconception 4: "A trade deficit makes a currency weak."** Generations of commentary insist that a country running a trade deficit — importing more than it exports — must have a weak currency, because it is "sending money abroad." Over the horizons traders care about, this is nearly invisible. The United States has run a large, persistent trade deficit for decades, and the dollar has had some of its *strongest* years during those decades, including 2022. The reason is that the trade balance is a *slow, structural* force measured in years, while rate differentials and capital flows are *fast* forces measured in days and weeks. Worse, the causality often runs backwards from the textbook: a strong currency driven by high rates makes imports cheaper and exports dearer, *widening* the trade deficit. So a wide deficit can be a *symptom* of a strong currency, not a cause of a weak one. Day to day, the trade balance is a footnote; the rate path is the headline.

**Misconception 5: "Intervention can override the rate differential."** When a currency moves sharply, governments sometimes step in and *intervene* — buying their own currency in the FX market to prop it up. Japan did this in 2022 and 2024 to slow the yen's slide. Traders watch these episodes and assume the authorities can simply set the price. They mostly can't, not against the differential. Intervention can slow or briefly reverse a move, buying time and punishing one-way speculators, but it does not change the *underlying incentive* created by the rate gap. As long as dollars yield 5% more than yen, the structural pull on the yen is *down*, and no amount of spending reserves changes that arithmetic; Japan's 2022 interventions cost tens of billions of dollars and the yen kept falling until the *rate gap* itself began to narrow in 2024. Intervention fights the symptom; only a change in the relative policy path fixes the cause.

## How it shows up in real markets

Theory is cheap. Here are two episodes where the rate-differential channel was not a model but a wrecking ball, with the dates and numbers.

### The 2022 dollar surge: the Fed out-hikes the world

2022 is the cleanest demonstration of policy divergence in modern history. Coming out of the pandemic, inflation surged everywhere, but central banks responded at wildly different speeds. The Fed, after a late start, hiked with shock-and-awe pace: 25 basis points in March, then 50, then four consecutive 75-basis-point hikes — taking the funds rate from 0.25% to 4.5% by December, the fastest tightening in four decades. Meanwhile the Bank of Japan stayed at zero, the ECB didn't lift off until July and moved cautiously, the People's Bank of China was *easing*, and emerging markets scrambled.

The result: the dollar's rate advantage exploded, and capital poured in. DXY climbed from ~96 to that 114.8 September peak — a 20-year high — and closed the year up about 8%.

![DXY year-end close line chart 2014 to 2025 with 2022 peak annotated](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-4.png)

The figure puts 2022 in context. The dollar had spent years in a 90–103 range; the policy-divergence shock of 2022 drove the year-end close to 103.5 and the *intraday* peak to that dashed 114.8 line — territory not seen since the early 2000s. Notice what happened next: as other central banks caught up in 2023–2024 and the divergence narrowed, the dollar gave back ground, exactly as the channel predicts. The dollar rose when the Fed *out-hiked*, and softened when the gap *closed*.

The damage on the *other* side of those pairs is where the channel really shows its teeth. The euro fell below \$1.00 — *parity* — in the late summer of 2022 for the first time in two decades, because the ECB was hiking later and more timidly than the Fed while also absorbing an energy shock from the war in Ukraine; the rate gap and the growth gap both pointed the dollar's way. The British pound was even more dramatic: in late September 2022, after the UK government announced large unfunded tax cuts, the pound crashed toward an all-time low near \$1.03 intraday. Part of that was a fiscal-credibility crisis, but the underlying vulnerability was the same — the Bank of England was seen as unable to match the Fed's path without breaking its own economy, so the rate-differential pull was relentlessly against sterling. In a strong-dollar year, the currencies that fall hardest are the ones whose central banks *can't keep up* — and 2022 ranked them by exactly that.

The emerging-market story rhymes. A surging dollar means every country with dollar-denominated debt sees its debt burden rise in local-currency terms, and every importer pays more for dollar-priced goods. Central banks from Brazil to Indonesia hiked aggressively — often *ahead* of the Fed — not because their domestic economies demanded it, but to defend their currencies against the dollar's pull. Some succeeded; many watched their currencies slide anyway. The Fed's policy path was, functionally, setting monetary policy for the planet, which is the misconception-3 point made vivid: there is no such thing as a purely domestic Fed.

#### Worked example: the 2022 dollar on a \$1,000,000 FX book

Suppose at the start of 2022 you ran a **\$1,000,000** book that was long the dollar against a basket of major currencies — effectively long DXY. The index returned **+8.2%** on the year (per `ASSET_RETURNS_2022`). What did the policy-divergence trade earn you?

```
book          = 1_000_000
dxy_return    = 0.082            # +8.2% total return on the dollar basket, 2022

pnl           = book * dxy_return    # 82000.0
```

The dollar basket's 8.2% gain on a \$1,000,000 book is `0.082 × 1,000,000 = \$82,000` — in a year when the S&P 500 fell 18%, long Treasuries fell 31%, and a 60/40 portfolio fell 16%.

![Horizontal bar chart of 2022 asset returns with US dollar plus 8.2 percent highlighted](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-5.png)

The figure shows why being long the dollar was the trade of the year. In 2022, almost every asset class fell — stocks, bonds, crypto, even most commodities by year-end — yet the dollar (DXY) *rose* 8.2%, beaten only by raw commodities. **When the Fed out-hikes the world, the dollar is not just a currency bet; it is one of the only places to hide, and on a \$1,000,000 book that was \$82,000 of gain while everything else bled.**

### The yen's policy-divergence collapse and the August 2024 unwind

The yen is the textbook funding currency, and 2021–2024 was its textbook collapse. The Bank of Japan, fighting decades of deflation, kept rates pinned near zero and even capped its 10-year bond yield, while the Fed hiked into the 5s. The rate gap between dollars and yen blew out to roughly five points — the widest in decades. The yen did exactly what the channel says: it fell from **108** to the dollar in 2020 to a **161.9** intraday low in July 2024, the weakest in 38 years. Every leg of that move was the rate differential at work; Japan's economy and trade balance were almost irrelevant to the FX move.

The carry trade built on that gap became one of the most crowded positions on earth — borrow yen near zero, buy *anything* yielding more, from US Treasuries to Mexican pesos to US tech stocks. Estimates of the total yen-funded carry ran into the hundreds of billions of dollars, much of it levered. Then came the unwind. In late July 2024 the BOJ nudged its rate up and signaled more, while a soft US jobs report raised the odds of Fed *cuts*. The rate gap that powered the carry started to *converge from both ends at once* — the funding leg got more expensive *and* the asset leg's yield advantage was set to shrink. On Monday, 5 August 2024, the unwind detonated: the yen ripped higher toward 145, Japanese stocks fell 12.4% in a single session (their worst day since 1987), and the VIX briefly spiked above 65. Nothing had blown up. The trigger was two small pieces of *interest-rate* news. But the crowd all reached for the same exit.

The mechanics of that day are the carry-unwind feedback loop in pure form, and they are worth tracing because they recur in every such episode. Step one: the rate-gap news flips the expected differential, so the marginal carry trader's edge evaporates and the most levered players start to close. Step two: closing the trade means *buying yen* (to repay the yen loan) and *selling the assets* bought with it (US stocks, other high-yielders). Step three: the yen buying strengthens the yen, which is an immediate mark-to-market *loss* for everyone still long the carry, triggering *their* stops and margin calls. Step four: forced sellers dump assets and buy yen into a thinning market, the yen accelerates, and the loop tightens. It is the same self-reinforcing dynamic that drove the yen *down* for three years, now running in reverse at ten times the speed — the elevator down. Within days, USD/JPY had retraced months of carry gains, and the shock rippled into every asset the carry crowd had touched. The lesson is not that the BOJ's tiny hike was a big deal in isolation; it is that *a crowded, levered position built on a rate gap is a coiled spring, and convergence is the trigger.*

#### Worked example: the carry-unwind hit, 161.9 to 145

Suppose you were long the carry — long USD/JPY — with a **\$1,000,000** notional position when USD/JPY sat at its July 2024 peak of **161.9**. In the unwind it fell to about **145**. What was the hit, and how much carry did it erase?

```
notional   = 1_000_000
peak       = 161.9
trough     = 145.0

fx_return  = (trough - peak) / peak     # -0.1044  -> -10.44%
loss       = notional * fx_return       # -104385.0

annual_carry   = 0.052 * notional       # 52000.0  (the 5.2% gap)
years_erased   = abs(loss) / annual_carry   # ~2.0
```

The spot fell `(145 − 161.9) / 161.9 = −10.4%`, a **−\$104,385** loss on the \$1,000,000 position — and at a 5.2% carry worth \$52,000 a year, that single move erased about **two full years** of accumulated carry in a matter of days. **This is the steamroller: the carry drips in at \$142 a day for years, and the unwind takes it all back in a week — which is why the carry-unwind tail, not the carry itself, is the thing you actually have to manage.**

## How to trade it: the policy-divergence playbook

Everything above earns its keep here. If policy moves currencies through the rate differential, then trading currencies is, at its core, trading *relative central-bank paths* — and managing the violent tail when those paths converge. Here is the playbook, concrete enough to act on.

![FX policy divergence playbook decision tree read the path go long the faster hiker size for the tail](/imgs/blogs/how-monetary-policy-moves-currencies-rate-differentials-7.png)

The figure is the decision flow. Start by reading the *relative path*. Then branch on whether policy is diverging or converging, position accordingly, and pre-size for the unwind tail.

**Signal 1: read the relative policy path, not the level.** For any currency pair, the question is never "is this central bank hiking?" but "is it hiking *faster than its peer*, and is the *expected gap* widening or narrowing?" Build the comparison from the two banks' projected paths — the Fed's dot plot, the market-implied path from rate futures, the peer central bank's guidance. The pair is pulled toward the currency whose *expected* differential is widening. When the Fed's dots drift higher relative to the ECB's guidance, you lean long the dollar against the euro; when they converge, you lean the other way. Tools for reading those paths live in [monetary policy transmission: how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets).

A concrete way to *operationalize* the relative path: track the **two-year yield spread** between the two currencies. The two-year government yield is a clean, market-priced summary of the expected policy path over the next couple of years — it bakes in all the hikes and cuts the market expects. The *difference* in two-year yields between two countries is the single best high-frequency proxy for the rate differential that drives their exchange rate, and it updates in real time as the market reprices either central bank. When the US two-year yield rises relative to the German two-year, the dollar tends to firm against the euro within the same session. Put the FX pair and the two-year spread on the same chart; when they diverge, one of them is usually about to catch up to the other, and that gap is a trade. The *prints* that move the spread are the ones to mark on your calendar: inflation reports (CPI, core PCE), jobs reports, and above all the central-bank meetings themselves, because those are when the expected path gets repriced in size.

**The position: long the faster hiker against the laggard.** When policy is diverging, the trade is to be long the currency whose central bank is out-hiking, funded in the laggard's currency. In 2022 that was long the dollar against almost everything; against the yen specifically (long USD/JPY) it was the cleanest expression because the BOJ was nailed to zero. You earn the carry *and* the expected spot appreciation, both pointing the same way.

**Signal 2: watch the carry-unwind tail above all.** The carry pays you a slow daily yield, so the risk is never the gap *widening* — that is already your thesis and you are paid for it. The risk is the gap *converging faster than priced*, which triggers the crowded unwind. The tells: the peer central bank turning hawkish at the margin (a surprise BOJ hike), the leading bank turning dovish (soft US data raising cut odds), positioning getting crowded (CFTC futures showing record one-sided bets), and FX volatility starting to twitch higher even as spot still grinds your way. When two or more of those line up, the convergence trade — flip to long the laggard, or simply cut the carry — has an asymmetric payoff. For reading positioning and the flows that crowd these trades, see [following the flows: positioning, COT, dealer hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging).

**Sizing and the invalidation.** Because the unwind is violent and fat-tailed, size the carry *small* and place stops *above the spike*, not at a normal-volatility distance — a crowded carry can move 10% in days, far beyond a typical FX range. The invalidation of a long-faster-hiker view is specific: it is *not* a bad print in the leading economy, it is a print that forces the *peer* central bank to out-hike instead, reversing the expected differential. If Japanese inflation surprises high and the BOJ signals a real hiking cycle while the Fed is done, the gap reverses and your long-USD/JPY thesis is dead — get out, or flip. Read the relative path; when it reverses, so does the currency.

**The one exception that proves the rule: the dollar smile.** Almost everything above says the dollar strengthens when the Fed out-hikes. But there is a second, distinct reason the dollar can rally: a global panic. When markets convulse, capital flees *to* the dollar as the ultimate safe haven and source of global liquidity, regardless of the rate path — this is the "flight to safety" leg of what traders call the *dollar smile*. So the dollar tends to be strong at both ends: when the US economy and rates are running hot relative to the world (the rate-differential leg), *and* when the world is in crisis (the safe-haven leg), with a soft middle when things are calm and money rotates out to higher-yielding peers. The practical upshot for this post's channel: the rate differential is the dominant driver in *normal* conditions, but in a genuine risk-off shock it can be temporarily overwhelmed by the safe-haven bid — and, tellingly, a carry unwind *is* a risk-off shock, so the dollar's two strength-drivers can fire together and against the yen-carry crowd at the same moment. For the full shape of that behavior, see [trading the dollar: DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile).

**Putting the layers together.** Trade nominal divergence for the direction, check the *real* differential to make sure the high-yielder isn't just inflating its currency away, watch positioning and vol for the unwind tail, respect the safe-haven bid in a crisis, and size for the steamroller. That is the entire discipline: the rate differential gives you the trade, real yields keep you honest, the dollar smile reminds you which regime you're in, and respect for the unwind keeps you alive. Read the relative policy path — the gap between one central bank and its peers, and where that gap is expected to go — and you read the currency.

## Further reading & cross-links

- [What moves exchange rates: rates, flows, and carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — the broader picture of FX drivers, where the rate differential sits alongside capital flows and trade.
- [Trading the dollar: DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile) — how the dollar behaves across risk regimes, including when it strengthens for the opposite reason (a flight to safety).
- [The dollar system: why the USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — why US policy is global policy and the dollar sits at the center of the financial system.
- [Monetary policy transmission: how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) — the full chain from a policy-rate change to every asset price, of which FX is one channel.
- [Real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the *real* differential, not the nominal one, is what currencies ultimately reward.

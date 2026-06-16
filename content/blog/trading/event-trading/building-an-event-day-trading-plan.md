---
title: "Building an Event-Day Trading Plan"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Amateurs react to the print; professionals follow a plan written before it. The five-step event-day routine, from knowing what's priced to the post-event review, turned into one repeatable workflow."
tags: ["event-trading", "trading-plan", "cpi", "fomc", "nfp", "expected-move", "risk-management", "cross-asset", "macro", "playbook"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — On an event day, the winner is decided before the release, not after. Amateurs stare at the number and react; professionals run a one-page plan written the night before. This post pulls the whole series into one repeatable five-step routine.
>
> - **What the routine is:** (1) know what's scheduled and what's already priced; (2) read the expected move and set a risk budget; (3) write the if-then scenario map per asset; (4) pre-set entries, stops, and size; (5) execute and review. Five steps, one page, every event.
> - **Why it works:** you cannot predict the print, so you plan *both branches*. The same CPI report gave the S&P 500 **−4.32%** on the hot Sep 2022 day and **+5.54%** on the cool Nov 2022 day. A plan that pre-decides hot and cool wins either way; a guess loses on average.
> - **The trade discipline:** size to the **expected move** (a \$500 risk budget on a ±1.2% S&P move ≈ a \$41,000 position), stage your orders before 8:30, and let the plan execute. Then review process, not just profit.
> - **The one number to remember:** a plan is not a prediction. You do not need to know the number — you need to have already decided your response to every number.

Two traders sit down at 8:25 a.m. on a CPI Thursday. Same screens, same account size, same data. By 8:35 their mornings could not look more different.

The first trader has no plan. The print drops at 8:30:00 — inflation a tenth hotter than expected — and he starts reading the headline. By the time he has parsed "core came in above consensus," the S&P has already dropped 0.8%. He decides he is bearish, types in a short, gets filled at a worse price four seconds later, watches the market bounce, panics, covers for a loss, then flips long into the bounce just as it rolls over again. He has *round-tripped* a position — bought high, sold low, bought back high — and by 9:00 he is down money and shaken, with no idea what just happened. He traded the screen.

The second trader did her work the night before. She wrote a single index card: the event, the consensus, the expected move, three pre-decided scenarios, her entry-stop-size for each, and the line that would tell her she was wrong. When the print hit at 8:30:00, her bracket order was already resting on the book. It triggered itself, filled near the open, with the stop already attached. By 8:35 she is flat or holding a position with capped risk, sitting on her hands, sipping coffee. She did not predict the number. She had already decided what she would do for *any* number. She traded her plan.

Same print. Same market. The difference was not skill at forecasting inflation — neither of them knew the number. The difference was that one of them had a plan and the other had a reflex. This entire series has built the pieces of that plan: [why news moves markets](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework), [what's priced in](/blog/trading/event-trading/consensus-expectations-and-priced-in), [the expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options), [the reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently), and [cross-asset transmission](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market). This post assembles them into one workflow you can run on any event — CPI, the jobs report, an FOMC decision, a Vietnamese rate move — every single time.

![The event-day routine as a six-stage loop from calendar to expected move to if-then map to pre-set orders to execute to review and back](/imgs/blogs/building-an-event-day-trading-plan-1.png)

## Foundations: the anatomy of an event-day plan

Before we walk the five steps, we need to define what an event-day plan actually *is* — and, just as importantly, what it is not. Almost every losing event-day trader makes the same category error: they confuse a plan with a prediction. Let us take the whole thing apart, term by term, so the rest of the post has a foundation to stand on.

### What a trading plan is

A **trading plan** is a written, pre-committed set of decisions about what you will do under each thing the market can do to you. It is written *before* the event, when you are calm and the screen is quiet, precisely so that you are not making decisions *during* the event, when the screen is screaming and your heart rate is up. The plan is a contract with your future, panicking self. It says: "Dear me-at-8:30:00, here is exactly what to do. Do not improvise."

The key word is *pre-committed*. A plan you can rewrite in the moment is not a plan; it is a suggestion, and suggestions evaporate the instant the market moves against you. The whole value of the plan is that it was decided when you had no skin in the immediate move and could think clearly.

### What "priced in" means

The single most important idea in event trading is that **markets trade expectations, not levels.** Weeks before any scheduled release, professional forecasters publish estimates, and the median becomes the **consensus** — the number everyone expects. That consensus is already baked into prices. Traders have spent weeks positioning for it.

This means the consensus number itself moves nothing on release. The only thing that can move price at 8:30:00 is the gap between what actually prints and what was already priced. We call that gap the **surprise**:

> surprise = actual − consensus

If the print equals consensus, the surprise is zero and, in theory, nothing should move — it was already in the price. A hotter-than-expected print is a positive surprise on inflation; a cooler print is a negative one. The *sign* of the surprise sets the *direction* of the reaction; the *size* of the surprise (relative to the expected move) sets the *magnitude*. This is the engine of the whole series, and the full treatment lives in [the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) and [consensus and priced-in](/blog/trading/event-trading/consensus-expectations-and-priced-in).

### What the expected move is

The **expected move** is the size of swing the options market thinks is *normal* for this event. You do not have to guess it — the market quotes it for you. A short-dated **straddle** (buying both a call and a put at the current price) pays off if the market travels far in either direction, so its price is the market's own estimate of how far the underlying is likely to move. Divide the straddle price by the price of the underlying and you get the expected move as a percentage.

On a typical S&P 500 CPI day in a calm regime, that number is roughly **±1.2%** — the market thinks a one-standard-deviation reaction is about 1.2% in either direction. The expected move is your risk yardstick: it tells you how wide to set stops, how small to size, and when a move is "normal" (often a fade) versus a genuine outlier (often a trend). The full mechanics are in [the expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

### What the if-then scenario map is

The **if-then scenario map** is the heart of the plan. It is a table whose rows are the branches the surprise can take — **hot** (above consensus), **in-line** (at consensus), **cool** (below consensus) — and whose columns are the assets you care about: stocks, crypto, the dollar, gold, and bonds. Each cell is a *pre-decided action* for that asset on that branch.

The map exists because you cannot know which branch the print will light up. So you write all three, and then you simply *read off* the cell when the number arrives. The map turns a panicked, real-time decision into a lookup. Building it correctly requires the [reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) (which decides the sign and size for the current regime) and the [cross-asset map](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) (which decides how one print hits every market at once).

![The if-then scenario map template with rows for hot in-line and cool surprises and columns for stocks crypto dollar gold and bonds with a pre-decided action in every cell](/imgs/blogs/building-an-event-day-trading-plan-2.png)

### What pre-set orders are

**Pre-set orders** are entries, stops, and targets staged on the broker's book *before* the release, so that execution happens at machine speed instead of human speed. The most common form is a **bracket order**: an entry with a stop-loss and a take-profit attached, all submitted as one package. When the entry fills, the stop and target attach automatically. You never have to type anything during the chaos.

The alternative — watching the print, deciding, and then typing an order — is fatally slow. In the few seconds it takes a human to read a headline and react, the market has already made most of the move, and the spread (the gap between bid and ask) has blown out, so you pay extra **slippage** (the difference between the price you wanted and the price you got).

### What the review is

The **review** is the step everyone skips and the step that actually makes you better. After the event, you score four questions: what was priced, what surprised, did I follow my plan, and what should I adjust. Crucially, you score *process, not outcome*. A trade that followed the plan and lost money is a *good* trade; a trade that broke the plan and made money is a *bad* trade that got lucky. The review is what turns one event into a sharper plan for the next.

### The difference between a plan and a prediction

Here is the distinction that the whole post hangs on. A **prediction** is a guess about *what the number will be*. A **plan** is a pre-decided *response to every number*. A prediction can only be right or wrong, and it makes you fragile: if you predicted "cool" and it prints hot, you are now mentally fighting the market while it runs you over. A plan cannot be "wrong" about the print, because it never bet on the print — it bet on having a prepared response, and it has one for whatever shows up.

Amateurs predict. Professionals plan. The number is an input to the plan, not the plan itself. Keep that sentence in your head and the rest of this post is just the mechanics of building the plan well.

### Why the plan beats the reflex: the calm-versus-panic asymmetry

There is a deeper reason a written plan wins, and it is not about information — both traders in the hook had identical information. It is about *when* the decision is made. The version of you that writes the plan the night before is calm, rested, and has no immediate money on the line. The version of you that meets the print at 8:30:00 has a position, a racing pulse, and a screen flashing red or green. These are, functionally, two different decision-makers, and the calm one is vastly better at trading than the panicked one.

The plan is how the calm decision-maker overrules the panicked one. By writing the entry, the stop, and the invalidation in advance and *committing* to them, the calm version pre-empts the panicked version's worst instincts: chasing the last tick, moving the stop, doubling down on a hunch, freezing on a gap. This is the same logic as a pilot's checklist or a surgeon's protocol — not because pilots and surgeons are stupid, but because even experts make worse decisions under acute stress, and the way you protect against that is to make the critical decisions *before* the stress arrives. An event-day plan is a stress checklist for trading.

This asymmetry is also why the plan must be *written*, not merely thought about. A plan in your head is editable in the moment, which means the panicked version can quietly rewrite it ("the stop was probably too tight anyway"). A plan on paper, with the orders already resting on the book, is much harder to override on impulse. The friction of having to consciously cancel a staged order and type a new one is, deliberately, a speed bump between you and a bad decision.

## Step 1: the calendar and what's priced

Everything starts with knowing what is scheduled and what the market already expects. You cannot trade an event you did not know was coming, and you cannot trade a surprise without first knowing the consensus the surprise is measured against.

### Map the calendar a week out

The first move, every week, is to look at the economic calendar and mark the **tier-1 events** — the releases big enough to move the whole tape. In the US, that short list is CPI, the jobs report (nonfarm payrolls), the FOMC rate decision, PCE (the Fed's preferred inflation gauge), and the big business surveys (ISM/PMI). Globally, add the major central banks — the ECB, the Bank of Japan, the Bank of England — and for Vietnam, the State Bank of Vietnam's rate moves and the GSO's CPI release. The macro-trading companion lays out the full calendar in [the macro calendar](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi).

Marking the calendar does two things. First, it tells you when to *be flat or be careful* — you do not want to be carrying a casual position into an 8:30 print you forgot about. Second, it tells you which day to *prepare a plan for*. Not every event deserves a full plan; a second-tier regional survey usually does not. The tier-1 events do.

### Find the consensus and the whisper

Once you know an event is coming, you find the **consensus** — the median forecast — and, where you can, the **whisper number**, the figure that active desks are really positioned for, which can drift a little from the published consensus in the days before the print. Both are easy to find: financial data terminals, broker research, and the major financial press all publish them.

Why bother? Because the consensus is the reference point that defines the surprise. If you do not know consensus is 3.2%, then a 3.2% print looks like "high inflation" and you might trade it as bearish — when in fact it is a *zero surprise* and should move nothing. The number in isolation is meaningless. The number relative to consensus is everything.

### Read what's already priced beyond consensus

Consensus tells you the expected number. But there is a second, subtler layer: how is the market *positioned* going in? If everyone is already short stocks expecting a hot print, then even a hot print can rally stocks (the bad news is already in the price, and the shorts cover). This is the **pain trade** — the move that hurts the most people — and it is why "obvious" reactions sometimes fail. The macro-trading series covers the mechanics in [following the flows](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) and [risk-on, risk-off](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).

For the plan, the practical takeaway is a single line on your card: *what is the lean?* Is the market positioned for hot, for cool, or balanced? You do not need precision. You need to know whether a given branch is "priced and crowded" (where the reaction may be muted or reverse) or "unpriced and clean" (where the reaction can run).

### Not every event deserves a plan — rank them

A subtle part of Step 1 is triage. There are dozens of data releases every week, and writing a full plan for each one is exhausting and, worse, dilutes your attention away from the prints that actually matter. So rank the calendar into tiers. **Tier-1** events move the whole tape and deserve a full one-page plan: CPI, the jobs report, FOMC, PCE, and the big surveys. **Tier-2** events nudge their own corner of the market but rarely move everything: regional Fed surveys, secondary housing data, weekly jobless claims (except when the labor market is the market's main worry). **Tier-3** is noise you can ignore.

The tiering is not fixed — it shifts with the regime, which is the whole lesson of the [reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently). In an inflation-obsessed regime, CPI is the king and a strong jobs number can be *bad* for stocks. In a growth-scare regime, the jobs report and jobless claims jump to tier-1 because the market's fear is recession, not inflation. Your weekly ritual is to ask: *what is the market most afraid of right now, and which release speaks to that fear?* That release is your tier-1 event for the week, and it is the one you build a plan for.

A practical discipline falls out of this: **be flat or hedged into tier-1 events you do not have a plan for.** The worst way to meet a CPI print is to be carrying a casual, oversized position you put on for some unrelated reason, with no scenario map and no pre-set stop. If you did not prepare a plan, the correct position into the print is "small or none," not "whatever I happened to be holding."

#### Worked example: a weak jobs print, two regimes, opposite trades

The Jul 2024 jobs report (released Aug 2, 2024) came in weak — **+114k** jobs against ~175k expected, with unemployment ticking up to **4.3%**. The S&P fell **−1.84%** that day and the 2-year yield dropped 28 basis points. Why did *weak* jobs hurt stocks?

- In the inflation regime of 2022, weak jobs would have been *good* for stocks — it implied the Fed could ease off. On a \$30,000 long, a 2022-style relief reaction of +1% = **+\$300**.
- But by Aug 2024 the market's fear had flipped to recession. Weak jobs now meant the economy was rolling over, so stocks fell. The same \$30,000 long took the −1.84% hit = **−\$552**.
- A trader whose scenario map still had "weak jobs → buy stocks" from the old regime would have bought the dip into a falling knife and lost roughly **\$850** versus the trader whose map's signs had flipped with the regime.

The intuition: the same surprise sign flips its trade when the regime flips, so Step 1 must always pin down *what the market is afraid of now* before the map's signs can be trusted.

#### Worked example: why the level fools you and the surprise does not

Say CPI prints at **3.2%** year-over-year. Is that bullish or bearish for stocks? You cannot answer without consensus.

- If consensus was **3.4%**, this is a **cool** surprise of −0.2 points — inflation better than feared, less Fed tightening implied. Stocks should rally. On a \$25,000 S&P position, a typical +1.5% cool-day reaction = +\$375.
- If consensus was **3.0%**, this is a **hot** surprise of +0.2 points — inflation worse than feared. Stocks should fall. On the same \$25,000 position, a −1.5% reaction = −\$375.
- If consensus was **3.2%**, the surprise is **zero**: no edge, often a fade, expected P&L on a \$25,000 position ≈ \$0 before costs.

Same 3.2% print, three completely different trades, separated only by the consensus you wrote down the night before. The intuition: you are never trading the number; you are trading the gap between the number and what was already priced.

## Step 2: the expected move and the risk budget

Once you know the event and what's priced, the next question is *how big a swing is normal* — and therefore *how much should I risk and how big should my position be?* This is where most retail blow-ups happen: not from being wrong about direction, but from being too big for the move.

### Read the expected move off the options

As covered in the foundations, the options market quotes the expected move for you. The quickest read is the at-the-money straddle: straddle price ÷ underlying price = expected move in percent. On a normal S&P CPI day that is about ±1.2%; on a Bitcoin event day it is more like ±4%, because crypto is structurally more volatile. There is also a formula version — expected move ≈ S × IV × √(t / 365) — but the straddle shortcut is faster and is what the desk actually uses.

The expected move is doing three jobs in your plan at once. It tells you (1) how wide your stop has to be to survive *normal* noise without getting knocked out, (2) how big a move counts as a genuine *outlier* worth trending with, and (3) — most importantly — how big your position can be for a given risk budget.

### Set a risk budget first, derive size second

This is the single most powerful habit in the whole routine, so write it in capital letters on your card: **decide what you are willing to lose, then let that decide your size.** Beginners do the opposite — they pick a position size that feels exciting, and then discover after the fact how much they can lose. Professionals fix the loss first.

Your **risk budget** is the dollar amount you are willing to lose on this trade if it goes against you by roughly one expected move. Pick a number that is small relative to your account — many disciplined traders risk well under 1% of capital per event. Then size falls out by arithmetic:

> position size = risk budget ÷ expected move

The beauty of this is that it self-corrects for danger. A bigger, scarier event has a wider expected move, which automatically gives you a *smaller* position for the same risk budget. You never have to remember to "trade smaller around big events" — the formula does it for you.

![Step 2 sizing to the expected move showing risk budget divided by the one-sigma expected move equals the position size with a worked example of five hundred dollars over 1.2 percent](/imgs/blogs/building-an-event-day-trading-plan-4.png)

#### Worked example: sizing a stock position to the expected move

You decide your risk budget for this CPI trade is **\$500** — that is the most you are willing to lose if the trade goes one expected move against you. The S&P 500 expected move is **±1.2%** (the \$60 straddle on the 5000 index: \$60 ÷ 5000 = 1.2%).

- position size = risk budget ÷ expected move = \$500 ÷ 0.012 = **\$41,666**, call it **\$41,000** notional.
- Check: if the trade moves 1.2% against you, you lose 0.012 × \$41,000 = **\$492** — right at your \$500 budget. Good.
- If instead you had naively traded a \$100,000 position because it "felt right," a 1.2% adverse move = −\$1,200, more than twice your intended risk.
- And if you set a tighter stop at half the expected move (0.6%), you could carry a \$83,000 position for the same \$500 budget — but you would get stopped out by ordinary noise far more often.

The intuition: your size is not a feeling; it is your risk budget divided by how far the market normally travels.

#### Worked example: the same budget sizes Bitcoin much smaller

Now run the identical \$500 budget on Bitcoin, whose event-day expected move is **±4%** (the \$2,400 straddle on \$60,000 BTC: \$2,400 ÷ 60,000 = 4%).

- position size = \$500 ÷ 0.04 = **\$12,500** of BTC.
- Check: a 4% adverse move on \$12,500 = −\$500, exactly the budget.
- That is *one-third* the dollar size of the S&P position, for the same risk — because Bitcoin moves more than three times as far on the same news.

The intuition: identical risk budget, wildly different position size, set entirely by each asset's expected move. The formula keeps you honest across assets.

### Event stops are wider than normal stops — on purpose

A common rookie error is to bring a normal-day stop to an event day. On a quiet day you might use a tight stop because the market is barely moving; bring that same tight stop to a CPI print and you will get knocked out by the ordinary noise of the release before your thesis has a chance to play out. The fix is to size the stop *off the expected move*, not off a fixed dollar or point amount. A reasonable default is a stop at roughly **one expected move** against you, which is wide enough to survive a normal reaction in your favor but tight enough to define the loss.

The trade-off is direct and it is why the risk budget comes first. A wider stop means a smaller position for the same risk budget; a tighter stop means a bigger position but more frequent stop-outs. Decide the budget, pick a stop that matches the event's noise, and let the two together set the size. Do not pick the size first and back into a stop that fits your ego.

A second discipline that pairs with the wide event stop is **partial profit-taking**. Because event reactions often spike then fade, a sensible plan banks part of the position when the move reaches a multiple of the expected move (say, take half off at 1× the expected move) and lets the rest run with a trailing stop. This protects the win from the fade while keeping upside if the move turns into a trend.

#### Worked example: scaling out of a winning event trade

You are long **\$40,000** of the S&P on a cool-CPI branch, with a 1.2% expected move and a 1.2% stop (max risk ≈ \$480). The market rallies +1.5% in your favor.

- At +1.2% (1× the expected move), you bank half the position: 0.012 × \$20,000 = **+\$240** locked in.
- You trail the stop on the remaining \$20,000 up to breakeven, so the worst case on the rest is now \$0, not −\$240.
- If the move extends to +2.5% by the close, the remaining \$20,000 adds another (0.025 − 0.012) × \$20,000 ≈ **+\$260**, for a total of about **+\$500** — while your downside was capped at the original \$480 the whole time.

The intuition: scaling out turns a spiky, mean-reverting event move into realized profit without giving up the trend, all defined in advance by the expected move.

## Step 3: the if-then scenario map per asset

Now we build the centerpiece. You know the event, what's priced, the expected move, and your risk budget. The scenario map turns all of that into a pre-decided action for every branch × every asset.

### Three branches, written before the print

Start with the rows: **hot**, **in-line**, **cool**. For each, write what you believe the *current regime* implies. In the inflation regime of 2022-2023, the rule was "good news is bad news": a hot inflation surprise meant more Fed tightening, so stocks fell, the dollar rose, gold fell, Bitcoin fell, and yields rose. A cool surprise was the mirror image. But the sign is not a law of nature — it depends on the regime, which is exactly what the [reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) determines. In a growth-scare regime, a *weak* jobs number can be bad for stocks (recession fear) rather than good (rate-cut hope). So the first thing each row's logic must encode is *which regime are we in and why does the market care about this number now.*

### Five columns, one action per cell

The columns are the assets: stocks (S&P, and for Vietnam the VN-Index), crypto (Bitcoin), the US dollar (DXY), gold, and bonds (yields). Each cell holds a concrete action — *short*, *buy dips*, *fade the spike*, *no trade* — not a vague sentiment. The map in figure 2 shows the template; the key is that every cell is filled before 8:30 so that the print is a lookup, not a decision.

The cross-asset logic is not arbitrary. A hot inflation print pushes yields up (the bond market prices more Fed tightening), which strengthens the dollar (higher US yields attract capital), which pressures gold and crypto (they pay no yield, so they look worse when cash pays more) and stocks (higher discount rates lower equity valuations). One print, one surprise, propagating through every market in a coherent chain — that is the whole point of [cross-asset transmission](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).

### Why you need both branches: the same report, opposite outcomes

Here is the proof that you must plan both branches rather than predict one. The chart below shows the S&P 500's same-day move on three different CPI sessions. The headline inflation number was high every single time — any central banker would have called all three alarming. Yet the September 2022 print sent the S&P down **−4.32%** while the November 2022 and November 2023 prints sent it up **+5.54%** and **+1.91%**. The level was high all three days. Only the surprise sign differed. If your plan had predicted a single direction, it would have been catastrophically wrong on at least one of these. A plan that holds both the hot branch and the cool branch is *right on the process* every time, regardless of which way the number breaks.

![Bar chart of the S&P 500 same-day move on three CPI sessions with the hot September 2022 day red and down and the cool November 2022 and 2023 days green and up](/imgs/blogs/building-an-event-day-trading-plan-3.png)

#### Worked example: the plan in action on the hot branch

Suppose your scenario map's hot branch said: "if CPI is hot, the inflation regime says risk-off — be short equity exposure." You staged a **\$20,000** short S&P position (or a short hedge against a long book) as part of the plan. The September 13, 2022 print comes in hot, and the S&P falls **−4.32%** on the day.

- P&L on the short = 0.0432 × \$20,000 = **+\$864**.
- Because the position was pre-set as a bracket, it filled near the open rather than 0.8% lower after a human reaction — call that another ~\$160 of saved slippage on a \$20,000 position.
- Total ≈ **+\$1,024** captured by the hot branch, versus the trader with no plan who was still reading the headline while the move happened.

The intuition: the map did not need to predict that the print would be hot — it just needed a pre-decided action that paid off *if* it was hot, ready to fire the instant it was.

### Add the invalidation line to every branch

A scenario map without an invalidation line is a trap. The **invalidation** is the price or condition that tells you the read was wrong and you should be flat. The cleanest one for a knee-jerk trade is the 8:30 opening level: if you went short on a hot print and price reverses back *through* the open within a few minutes, the move was a head-fake (often the pain trade), and your plan should say "flatten." Writing the invalidation in advance is what stops a small, planned loss from becoming a panicked, unplanned one.

### The map travels: the same routine for a Vietnam event

The five-step routine is not US-specific — it works for any scheduled event in any market, because the engine (surprise, not level) is universal. The columns and the mechanism change; the structure does not. Consider a State Bank of Vietnam (SBV) policy-rate decision or a GSO CPI release. The assets you map are the VN-Index, the dong (USD/VND), the rate-sensitive banking and real-estate sectors, and — increasingly — foreign flows on HOSE, since foreign buying and selling swings the index. The mechanism is covered in the macro and Vietnam companions on [Vietnam monetary policy](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) and [foreign flows and the index effect](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).

The regime logic is just as important in Vietnam as in the US. In autumn 2022, the SBV *hiked* the refinancing rate from 4.0% to 6.0% to defend the dong as the dollar surged — a tightening shock. The VN-Index fell hard into that period, troughing at **911** on November 15, 2022, roughly −39% from its January 2022 peak near 1,528. Then across April-June 2023 the SBV *cut* three times back to 4.5%, and the index recovered toward 1,130 by year-end. A trader running the routine would have had a scenario map whose "surprise hawkish" branch said *risk-off the VN-Index, watch foreign selling* and whose "surprise dovish" branch said *risk-on, banks and real estate lead.* Same five steps, same both-branches discipline, different columns.

#### Worked example: sizing a VN-Index event trade in dollars

Suppose you trade a VN-Index ETF exposure worth **\$20,000** (so the dong move is hedged into your home currency for clarity) into an SBV decision, with a risk budget of **\$300** and a pre-print expected move of about ±1.5% for the index on a policy surprise.

- position size already set at \$20,000; check the risk: a 1.5% adverse move = 0.015 × \$20,000 = **−\$300**, exactly the budget. Good fit.
- On a surprise-hawkish print in the 2022-style regime, a −2.5% index reaction on the \$20,000 = **−\$500** if unhedged — which is why the plan's hawkish branch was *short or flat*, not long. A pre-set short of \$20,000 on that branch = **+\$500**.
- On a surprise-dovish cut like the 2023 easing, a +2% relief rally on \$20,000 = **+\$400** for the long branch.

The intuition: the routine is portable — swap the US assets for VN-Index, dong, and foreign flows, keep the surprise engine and the both-branches map, and the dollar math works identically.

## Step 4: pre-setting entries, stops, and size

The scenario map says *what* to do. Step 4 stages it on the broker's book so that *how* you do it is automatic. The enemy on an event day is the gap between deciding and executing, and the only way to close that gap to near-zero is to do the deciding in advance.

### Stage bracket orders, not bare entries

For each branch you actually intend to trade, build a **bracket**: the entry, a stop-loss, and a take-profit, submitted as one package. The stop is sized off the expected move (wide enough to survive normal noise, no wider) and the take-profit off a multiple of the expected move (a common rule is to risk one expected move to make 1.5). When the entry triggers, the stop and target attach automatically. You are now in a fully-defined trade with a known max loss the instant you fill.

Many platforms support **resting conditional orders** — orders that only activate if price breaks a level. That is ideal for event trades: you can stage a "short on a break of the pre-print low" and a "long on a break of the pre-print high" simultaneously, so whichever branch the market lights up, the correct order fires and the other can be cancelled.

### The pre-set advantage is mostly about slippage and stops

Two concrete edges come from pre-setting. First, **fill quality**: a resting order near the open captures the move instead of chasing it, and around a print the spread blows out for several seconds, so the difference between an instant fill and a four-second-late fill can be a meaningful fraction of a percent. Second, **the stop is already attached**: you are never in an open position without a defined exit, which is exactly the situation in which round-trips and blow-ups happen. The series covers the microstructure of these seconds in [liquidity and gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news).

![Before and after comparison of reacting to the print with no plan versus pre-setting bracket orders showing the no-plan trader chasing the screen and round-tripping while the pre-set trader is filled and done by 8:35](/imgs/blogs/building-an-event-day-trading-plan-6.png)

#### Worked example: a pre-set stop saves the position on a gap

Your plan has a **\$30,000** long position into an event, with a pre-set stop at **2%** below entry. The print is a shock and the market gaps down **−5%** before you could possibly react manually.

- With the pre-set stop, your stop order executes on the way down. Even allowing for some slippage past the 2% level, your loss is capped near **−\$600** to **−\$700** on the \$30,000 position.
- Without a pre-set stop, you are watching a −5% gap, frozen, and by the time you manually sell you are down the full 5% = **−\$1,500**.
- The pre-set stop saved roughly **\$800-\$900** on this single event by turning a −\$1,500 panic exit into a −\$600 planned exit.

The intuition: the stop you set when calm is the stop that actually protects you; the stop you "plan to set in the moment" does not exist when the moment arrives.

#### Worked example: the no-plan round-trip

Compare the disciplined trader to the no-plan trader on the same in-line-ish print. The no-plan trader buys a **\$25,000** long into the initial pop, riding it +1%, then watches it reverse, panics as it goes −2% from his entry, and dumps it.

- Entry on the pop, then he sells after a 1% gain has evaporated and turned into a 1% loss from entry... but he actually bought *after* the +1% pop, so his real entry was high. From his entry the market went +0% then −2%.
- Net on the \$25,000: he round-tripped −1% net = **−\$250**, plus widened-spread slippage on both the entry and the panic exit of roughly **\$100** = about **−\$350** total.
- Had he instead followed an in-line plan ("fade the first spike, small size, or no trade"), his expected loss was near \$0.

The intuition: with no pre-decided action, the trader's only strategy is to chase the most recent tick, which is exactly how you buy high and sell low.

## Step 5: execution discipline and the post-event review

The plan is written, sized, and staged. Step 5 is the discipline to *let it run as written* — and then the review that compounds your edge over months.

### Execute the plan, do not improvise

When the print hits, your job is almost boring: read the number, identify the branch (hot, in-line, or cool), confirm the cross-asset move agrees with that branch, and let the pre-set order do its work. The one decision left in real time is the *fade-or-trend* read in the first minute — is this a normal-sized move likely to revert (fade) or a multiple-of-expected-move outlier likely to run (trend)? The [anatomy of a news reaction](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) covers that spike-fade-trend microstructure in depth, but the plan should already have a default for each branch so even this is mostly pre-decided.

The discipline part is what you do *not* do. You do not add size because you feel strongly. You do not move your stop further away because the trade is "about to work." You do not invent a fourth scenario that you never wrote down. The moment you start improvising, you are back to being the first trader in the hook — reacting to the screen. If the print is genuinely outside everything your plan considered (a number so far from consensus that no branch fits), the correct improvisation is *to stand aside*, not to wing it.

### Honor the invalidation

The hardest discipline is taking the planned loss. When price hits your invalidation, you flatten — no negotiating. The reason the invalidation was written in advance is that the calm version of you knew the panicking version would want to "give it a little more room." A small planned loss is the cost of doing business; an unplanned loss that you let run because you abandoned the plan is how accounts die. Around fast events, missing the invalidation can be expensive: the [Aug 2024 carry cascade](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) saw the S&P fall −3.0% in a session and the VIX spike intraday to 65.73 — the traders who honored their stops survived it; the ones who "gave it room" did not.

### The five-minute pre-flight check

In the minutes right before the release, run a short pre-flight check so you execute the plan you actually wrote, not a half-remembered version of it. Confirm five things: (1) the card is in front of you and the consensus number is correct as of this morning — late-breaking revisions to the forecast do happen; (2) your bracket orders are staged and the sizes match your risk budget; (3) the pre-print high and low (the levels your conditional entries reference) are marked on the chart; (4) you are flat or hedged in everything *except* the planned trade, so no stray position muddies the read; and (5) the cross-asset confirmations you will look for — yields, the dollar — are on the screen. This check takes two minutes and prevents the most common execution failures: a stale consensus, a mis-sized order, or a forgotten position fighting your plan.

The reason this matters is that the print arrives in a burst, and you have only seconds to act. Anything you can resolve *before* the burst is one fewer thing to resolve *during* it. The pre-flight check moves the last few decisions out of the chaos window and into the calm one — the same calm-versus-panic asymmetry that justified the plan in the first place.

### The post-event review: score process, not outcome

After the dust settles — that day or that evening — you run the review. Four questions, every time:

1. **What was priced?** Was my consensus right? Was my expected move about right, too tight, or too wide? Did I correctly read the market's lean going in?
2. **What surprised?** What was the sign and size of the surprise? Which branch actually lit up? Did the cross-asset move confirm or contradict the branch?
3. **Did I follow the plan?** This is process, not P&L. Did I take the pre-set entry? Did I honor the stop and the invalidation? Did I improvise?
4. **What to adjust?** Pick *one* concrete change for next time — a wider stop, a tighter consensus source, a re-written cell in the scenario map, a smaller risk budget.

The discipline here is to grade yourself on process. A trade that followed the plan and lost \$500 is a *win* for your process; a trade that broke the plan and made \$500 is a dangerous *loss* — it taught your brain that breaking the plan pays, and next time it will not. Keep a written log so the adjustments accumulate. Over a year of events, the trader who reviews process turns a rough plan into a sharp one; the trader who only looks at P&L learns nothing and repeats the same mistakes.

![The post-event review loop with four questions what was priced what surprised did I follow the plan and what to adjust feeding back into the next plan](/imgs/blogs/building-an-event-day-trading-plan-7.png)

#### Worked example: a good loss beats a lucky win

Two outcomes, same event. Trader A followed her plan: pre-set short on the hot branch, honored a stop when the move faded, took a planned **−\$200** loss on a \$20,000 position (a 1% adverse move). Trader B abandoned his plan, doubled his size on a hunch into the same print, got lucky on the direction, and booked **+\$1,500**.

- Trader A's process score: clean. She executed exactly as written; the loss was within budget. Over 50 such events at her edge, this process compounds positively.
- Trader B's process score: failure. He risked roughly **\$3,000** of undefined downside (a 2% adverse move on a doubled \$150,000-equivalent exposure could have been −\$3,000) to make \$1,500. The reward was real but the risk was uncontrolled.
- Expected value over many repeats: A's disciplined −\$200-and-survive beats B's coin-flip, because B's eventual blow-up wipes out many lucky wins at once.

The intuition: you cannot control any single print's outcome, but you can control your process — and process is the only thing that compounds.

## How it works: a worked event day

Let us walk one full CPI day through all five steps, end to end, with one trader who has the plan. The numbers are illustrative but built on the real expected move and the real reactions from the series data.

**The night before (Steps 1-4).** Our trader checks the calendar: US CPI lands tomorrow at 8:30 ET. She finds consensus at **3.2%** year-over-year headline, with a whisper around 3.1% (a slight cool lean). She notes the market is positioned mildly long risk into the print, so a hot surprise is the *unpriced, clean* branch and a cool surprise is somewhat priced. She reads the S&P expected move off the straddle: **±1.2%**. She sets her risk budget at **\$500** and derives size: \$500 ÷ 0.012 = **\$41,000** notional for any equity branch she takes. She writes the if-then map:

- **Hot (> 3.4%):** short S&P, entry on a break of the pre-print low, stop 1.0% (−\$410), target 1.5 × expected move. Buy the dollar. Lighten gold and crypto.
- **In-line (3.1-3.3%):** no trade for five minutes, then fade the first spike back into the range, half size only.
- **Cool (< 3.0%):** long S&P and add small-cap (Russell 2000) exposure, target +1.5 × expected move. Sell the dollar; buy gold.
- **Invalidation:** if the move reverses through the 8:30 open within ten minutes, flatten — the read was wrong.

She stages the bracket orders for the hot and cool branches as resting conditionals and goes to bed. The whole thing fits on one card.

![The one-page event plan card filled in with the event consensus expected move three scenarios entry stop size and invalidation](/imgs/blogs/building-an-event-day-trading-plan-5.png)

**8:30:00 (Step 5, execute).** The print lands: **3.5%** — a hot surprise of +0.3 points, well outside consensus. The hot branch lights up. Her resting "short on a break of the pre-print low" triggers as the S&P breaks down, filling near the open with the 1.0% stop already attached. She does not type anything. The cross-asset tape confirms the branch: yields jump, the dollar rallies, gold and crypto soften. The branch agrees with itself, so she lets it run.

**The first minutes (fade-or-trend).** The move is roughly 1.8% in the first ten minutes — that is **1.5× the expected move**, a genuine outlier, not normal noise. Her plan's default for an outlier hot print is *trend, not fade*, so she holds toward the 1.5× target rather than taking the quick scalp. The move extends through the session toward a −2.5% close (a smaller cousin of the −4.32% September 2022 day).

**The P&L.** On the \$41,000 short, a captured move of about 2.0% (entry to a disciplined exit before the close) = **+\$820**, against a defined max risk of \$410. She risked \$410 to make \$820 — a clean 2:1 — and at no point was she guessing.

**That evening (the review).** She scores it: consensus and expected move were about right; the surprise was hot and large; she followed the plan exactly and honored the structure; the one adjustment is that her stop at 1.0% was a touch tight for a 1.5× day and she'll widen it to 1.2% (one full expected move) next time so she is not at risk of a noise-driven stop-out on a day the plan was right. One event, one improvement, logged.

Notice what she never did: she never predicted 3.5%. She did not know the number. She had simply pre-decided her response to every number, sized it to the expected move, staged it on the book, and executed without improvising. That is the entire routine.

## Common misconceptions

### "A plan means predicting the number"

This is the big one, and it is exactly backwards. A plan is the *opposite* of a prediction. A prediction commits you to one outcome and makes you fragile when reality differs; a plan pre-decides your response to *every* outcome and makes you anti-fragile. The proof is in the data: the same high-inflation CPI report gave the S&P **−4.32%** in September 2022 and **+5.54%** in November 2022. Anyone who "predicted" a direction was wrong on at least one of them by nearly ten percentage points. The trader with a both-branches plan was correct on process both times. You do not need to know the number — you need to have already decided what you will do for any number.

### "Bigger conviction deserves bigger size"

Conviction is a feeling, and feelings do not belong in the sizing formula. Size = risk budget ÷ expected move. That is it. On a \$500 budget and a ±1.2% move, your position is \$41,000 whether you feel "pretty sure" or "extremely sure," because your *certainty* about direction does not change how far the market normally travels, and the expected move is what determines your loss if you are wrong. The traders who size up on conviction are the ones who eventually meet a print that mugs their highest-conviction trade — and a doubled position on a −4.32% day is −\$3,500 on what should have been an \$864 loss.

### "I'll set my stop once I see how it trades"

No, you will not — or rather, the version of you watching a −5% gap will not set a sensible stop, because that version is panicking. The stop that protects you is the one you set when calm, before the print, attached to the order. As the worked example showed, a pre-set 2% stop on a \$30,000 position caps a −5% gap at roughly −\$600 instead of the −\$1,500 you eat when you freeze and exit manually. The "I'll decide in the moment" stop is a stop that does not exist when you need it.

### "In-line prints are boring, so I'll just trade them like a small surprise"

An in-line print is not a small surprise; it is *zero* surprise, and it behaves differently. With nothing new to price, the options market's pre-event premium collapses — the **vol crush** — and price tends to chop and fade rather than trend. Treating an in-line print like a directional trade is how you get chopped to death by noise inside the expected move. The plan's in-line branch should usually be "fade or no trade," not "small version of the hot trade." The series covers this in [event volatility and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush).

### "Reviewing is a waste of time if I made money"

The review is *most* important when you made money, because that is when you are least inclined to ask whether you made it correctly. A profitable trade that broke the plan is a process failure wearing a winning disguise, and if you do not catch it, your brain learns that breaking the plan pays. Grade process, not outcome, every time — especially after wins.

## The playbook: the one-page template

Here is the routine compressed into a one-page card you can fill in for any event. The whole point is that it fits on a single page and is written before 8:30.

**Top of card — the setup (Steps 1-2):**

- **Event & time:** which release, exact release time, who publishes it.
- **Consensus & whisper:** the median forecast, plus the whisper if you have it. *This defines the surprise.*
- **The lean:** is the market positioned for hot, cool, or balanced? Which branch is priced/crowded vs unpriced/clean?
- **Expected move:** ±__% (off the straddle). *Your risk yardstick.*
- **Risk budget:** \$___ (small relative to account). **Size = budget ÷ expected move.**

**Middle of card — the if-then map (Step 3):** for each branch, one pre-decided action per asset.

| If the surprise is | Stocks | Crypto | US dollar | Gold | Bonds (yields) |
|---|---|---|---|---|---|
| **Hot** (above consensus) | short / sell rallies | sell / aside | buy USD | lighten | yields up, short duration |
| **In-line** (at consensus) | fade the spike | no trade | flat | no trade | fade the wick |
| **Cool** (below consensus) | buy / add dips | buy / add | sell USD | buy | yields down, long duration |

*(These signs are for the inflation "good-news-is-bad" regime. In a growth-scare regime, re-derive them from the reaction function — the map's signs flip with the regime.)*

**Bottom of card — execution & exit (Steps 4-5):**

- **Pre-set orders:** bracket for each branch you'll trade — entry (on a break of the pre-print high/low), stop (≈ one expected move), target (≈ 1.5 × expected move).
- **Invalidation:** the line that says "I was wrong, flatten" — usually a reversal back through the 8:30 open within ~10 minutes.
- **Fade-or-trend default:** inside expected move → fade; outlier (> 1.5× expected move) → trend.
- **The review (after):** what was priced, what surprised, did I follow the plan, the *one* thing to adjust.

The discipline is in the order of operations: setup first, then the map, then the orders, then execution, then review — and never skip a step. The trader who runs this loop every event is the second trader in the hook, done by 8:35 with a defined trade and a clear head. The trader who skips it is the first, round-tripping a loss while still reading the headline. The number is the same for both of them. The plan is the difference.

## Further reading and cross-links

This post is the synthesis; each step has a full treatment elsewhere in the series and its companions.

- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the engine of Step 1: price contains the consensus, only the surprise moves it.
- [Consensus, expectations, and "priced in"](/blog/trading/event-trading/consensus-expectations-and-priced-in) — how to find consensus and read what's already priced (Step 1).
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — the risk yardstick and the sizing formula behind Step 2.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — how the regime sets the sign of each cell in your map (Step 3).
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — the columns of your map and why they move together (Step 3).
- [Anatomy of a news reaction: spike, fade, trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) — the fade-or-trend read inside Step 5.
- [Event volatility, implied vs realized, and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — why in-line prints behave differently.
- [Liquidity and gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news) — the microstructure that makes pre-set orders matter in Step 4.
- For the policy mechanism behind the events themselves, see the macro-trading companions on [the macro calendar](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi), [the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot), and [following the flows and positioning](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging).

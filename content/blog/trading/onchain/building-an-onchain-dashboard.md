---
title: "Building an On-Chain Dashboard: The Panels That Actually Matter"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How to design an on-chain dashboard on Dune or Flipside that drives daily decisions — which metrics earn a panel, how to lay them out, and three real templates for a token monitor, a protocol-health board, and a smart-money tracker."
tags: ["onchain", "crypto", "dashboard", "dune", "flipside", "data-analysis", "defi", "smart-money", "due-diligence", "monitoring"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A dashboard is a saved set of queries and charts that refresh on their own; a *good* one is the small set of panels that change what you do, not a wall of pretty charts you scroll past.
>
> - A **panel earns its place only if it answers a recurring question and changes a concrete action** (buy, trim, wait, avoid, investigate). Everything else is decoration.
> - The least useful panel is **price** — it is on every website and lags the chain. The edge lives in **flow, holders, fees, and unlocks**: who is moving what, where, and when.
> - Build for **signal density**: five panels that each map to a decision beat fifty that map to none. Lay top-line health at the top, drill-downs below, group panels by the question they answer.
> - The one rule to remember: **if you cannot name the action a panel changes, cut it.** A dashboard reflects yesterday; pair it with alerts for anything you need in real time.

On 22 May 2025, the Cetus AMM on the Sui chain was drained of roughly \$220M through a math-overflow bug in its pool accounting. The exploit itself took minutes. But the *signal* that something was wrong showed up on public dashboards long before most holders noticed: pool reserves cratered, an unfamiliar address started pulling liquidity in size, and the on-chain fee revenue that had been ticking up every day simply stopped. Anyone watching a protocol-health board with a TVL panel and a reserves panel saw a vertical red line appear in real time. Anyone watching only the token price saw a number that was, for a few critical minutes, still green.

That gap — between the people staring at price and the people staring at *flow* — is the whole argument for building your own dashboard. Price is a single lagging number that every exchange, aggregator, and news site already shows you. The blockchain shows you the inputs to price *before* they resolve into it: the supply moving toward sell venues, the whales quietly accumulating or distributing, the vesting cliff nine days out, the protocol that earns real fees versus the one renting deposits with token emissions. A dashboard is how you stop running the same fifteen queries by hand every morning and turn them into a single screen that refreshes itself.

This post is a build guide. It is the sibling of [Writing On-Chain Queries with Dune](/blog/trading/onchain/writing-onchain-queries-with-dune) — that post teaches you to write the SQL; this one teaches you to assemble those queries into a daily decision tool. We will define what a dashboard actually is from zero, separate exploration from monitoring, work out which metrics deserve a panel and which are noise, lay out the panels so the screen reads in five seconds, and then build three concrete templates you can fork and adapt: a **token monitor**, a **protocol-health board**, and a **smart-money tracker**. We will end with the limits — a dashboard reflects yesterday, so you wire alerts on top of it for anything time-sensitive.

![Decision dashboard mental model: a question flows to a metric, to a panel, to an action](/imgs/blogs/building-an-onchain-dashboard-1.png)

The figure above is the mental model for the entire post, so hold onto it. Every panel on a good dashboard begins life as a *recurring question* you ask. The question selects a *metric* that answers it. The metric becomes one *panel* with a threshold drawn in so it reads at a glance. And the panel exists only because it changes an *action* — measured in dollars, taken today. If a chart cannot trace itself back through that chain to a real decision, it does not belong on the board. That single test is what separates a dashboard you actually use from a screenshot you post to look busy.

## Foundations: what a dashboard is, and what it is for

Before we argue about which panels matter, we need shared vocabulary. None of this assumes you have built a dashboard before.

**A query** is a single question you ask of blockchain data, written in SQL (on platforms like [Dune](https://dune.com) or [Flipside](https://flipsidecrypto.xyz)) and answered as a table or chart. "How many unique wallets held token X each day for the last 90 days?" is a query. The [Dune SQL post](/blog/trading/onchain/writing-onchain-queries-with-dune) covers how to write these against decoded blockchain tables; here we assume you can produce them.

**A panel** (sometimes called a *visualization*, *widget*, or *tile*) is one query rendered as a chart, a counter, or a table, and placed on a screen. A line chart of those 90-day holder counts is a panel. A single big number showing today's net exchange flow is a panel. A panel is the atomic unit of a dashboard.

**A dashboard** is a saved arrangement of panels that **refresh automatically**. This is the load-bearing word. When you run a query by hand in an editor, you get a one-time snapshot. When you pin that query's chart to a dashboard, the platform re-runs it on a schedule — hourly, every few hours, daily — and the panel always shows the latest answer without you touching the SQL. A dashboard is, precisely, *a set of queries that keep answering themselves*. That is the difference between doing analysis and *having* analysis.

**Refresh cadence** is how often each panel re-runs. On Dune, free-tier dashboards typically refresh on a slower schedule and re-run when someone opens them; paid tiers let you set explicit refresh intervals. Cadence matters because it sets how stale your view can be. A holder-count panel that refreshes daily is fine — holder bases move slowly. An exchange-flow panel you are using to catch a sell-off needs to be much fresher, and if you need it to the minute, a dashboard is the wrong tool entirely (we will come back to that limit; the answer is alerts).

### Exploration versus monitoring — two different jobs

The single most common mistake is confusing the two distinct jobs that query platforms do, so let us draw the line clearly.

**Exploration** is open-ended investigation. You have a hypothesis — "I think this token's volume is faked" — and you write throwaway queries, slice the data five ways, follow surprises, and either confirm or kill the idea. Exploration is *one-off*. The queries are scratch work. Most of them you will never run again. This is the work covered in posts like [Detecting Fake Volume vs Organic Demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand) and [How to Trace a Transaction Flow](/blog/trading/onchain/how-to-trace-a-transaction-flow).

**Monitoring** is the opposite. You have already decided *which questions you ask every single day*, and you want their answers waiting for you when you sit down. Monitoring queries are permanent, scheduled, and stable. A dashboard is a monitoring tool. The discipline of building one is the discipline of figuring out which of your hundreds of exploratory queries deserve to become a permanent, refreshing panel — and ruthlessly cutting the rest.

A useful way to keep them apart: **exploration answers a question once; monitoring answers the same question forever.** If you find yourself re-running an exploratory query every morning, that is the universe telling you it should become a panel. If you find yourself never looking at a panel, that is the universe telling you it should go back to being an ad-hoc query.

### The metric-to-panel mapping

The core design act is mapping a metric to a panel — deciding which numbers earn a permanent home. The rule is brutally simple and we will repeat it until it is reflexive: **a metric earns a panel only if it answers a recurring question AND changes a concrete action.** Two tests, both required. A metric can be fascinating and still fail the second test. A metric can change an action but be a one-off and fail the first. Only metrics that pass both deserve the screen real estate, the refresh budget, and your attention every morning.

This is also where we kill the most popular panel of all. Price passes the first test — "what is it worth?" is certainly recurring — but it fails the second. Price is already on every exchange, every aggregator, every phone widget; it carries no on-chain *edge*, and it lags the flows that move it. Putting a price chart on your on-chain dashboard is like putting a clock on a dashboard whose entire job is to predict what time it will be. We will formalize this in the panel-selection section.

### Signal density: the property that matters most

**Signal density** is the share of panels on your screen that actually drive a decision. A dashboard with thirty panels where five drive decisions has a signal density of one in six — most of the screen is noise you have to look past to find the five that matter. A dashboard with exactly those five panels has a signal density of one. Higher is better. Always.

This runs against instinct. More panels *feel* like more information, more thoroughness, more value. They are the opposite. Every dead panel costs you twice: once in the refresh budget and load time it consumes, and again in the cognitive tax of scanning past it every morning to reach the live ones. A decision dashboard is defined by what you left off it. We will see exactly this contrast in the vanity-versus-decision figure shortly.

There is also a hard economic reason to keep the panel count low. Refreshing a query is not free: on Dune and Flipside, query execution consumes credits, and a board with thirty heavy panels re-running every few hours can burn through a free or modest tier quickly. A board with five well-chosen panels refreshes for a fraction of the cost and loads in a fraction of the time. Signal density and refresh economy point the same direction — fewer, better panels — which is a happy alignment: the discipline that makes the board *useful* also makes it *cheap*. When you are tempted to add a panel "just to have it," remember it costs credits forever, not just attention.

### Sharing and forking — you do not start from scratch

A final foundation: on Dune and Flipside, dashboards and queries are **public by default and forkable**. Anyone can open a dashboard, see the SQL behind every panel, copy ("fork") the whole thing into their own account, and edit it. This is the community's superpower. You almost never build a token-holder query from raw transaction logs — you fork a well-built community dashboard, swap in your token's contract address, and you are 80% done. We will cover forking and parameterizing properly at the end. For now, internalize that the on-chain analytics world is a giant shared library; the skill is knowing which book to copy and what to change inside it.

## Panel selection: which metrics earn a panel

This is the heart of dashboard design, so we will spend real time here. The question is not "what *can* I measure on-chain?" — the answer to that is "almost anything," which is useless. The question is "what should I put on a screen I look at every day?" Let us walk the candidate metrics one at a time and apply the two tests.

![Metric to panel mapping showing price and market cap earn no panel while flow holders fees and unlocks do](/imgs/blogs/building-an-onchain-dashboard-6.png)

The matrix above is the verdict for the most common candidate metrics, and it is worth reading row by row, because the cuts are as important as the keeps. The left column is the question each metric answers; the middle is whether it changes a decision; the right is the verdict. Notice the pattern: the metrics that earn a panel all answer a *forward-looking, on-chain-native* question that price cannot.

### Price and market cap — the panels to cut

Start with what does *not* belong. **Price** answers "what is it worth right now?" It is real, it is recurring, and it is utterly without edge: it is on CoinGecko, your exchange, your phone, and it is a lagging summary of flows you could be watching directly. **Market cap** (price × circulating supply) is worse — it is price wearing a hat, restating the same number scaled by a supply figure that is itself often wrong (circulating-supply estimates lag unlocks). Neither changes a decision you could not make better by looking one layer deeper. Both are cut.

This is counterintuitive enough that it deserves a name: **the price trap.** Beginners build dashboards that are 80% price charts because price is the number they emotionally care about. But the dashboard's job is not to tell you how you feel; it is to tell you what is about to happen. Price is the *output*. Your panels should be the *inputs*.

### Reported volume — conditional, with a filter

**Reported volume** answers "is there activity?" — but raw volume is one of the most easily faked numbers in crypto. Wash trading (a wallet trading with itself or a colluding partner to manufacture volume) is rampant, especially on low-cap tokens and some exchanges, as covered in [Detecting Wash Trading](/blog/trading/onchain/detecting-wash-trading). So volume earns a panel *only if* you filter it: deduplicate self-trades, exclude known wash-trading clusters, or restrict to DEX swaps you can verify against pool reserves. Unfiltered volume is a vanity number. Filtered, organic volume is a real demand signal. The panel is conditional on the cleaning.

### The metrics that earn a panel — flow, holders, fees, unlocks

Now the keeps. Every one of these answers a question price cannot, and every one changes an action.

**Exchange net flow** — deposits to exchanges minus withdrawals — answers "is supply moving toward sell venues?" A large net *inflow* to exchanges is supply arriving where it can be sold; it is bearish-leaning and a reason to trim or wait. A net *outflow* is coins leaving for self-custody or staking, removing sell-side supply; it is bullish-leaning. This is one of the cleanest forward signals on-chain, covered in depth in [Exchange Flows: Inflows and Outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows). It changes an action. It earns a panel.

**Holder concentration** — what share of supply the top wallets control — answers "who can dump and end the trade?" If one non-liquidity wallet holds 31% of supply, that wallet is your single point of failure; the panel directly sets your maximum position size. See [Supply Distribution and Holder Concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration). It earns a panel.

**Unlock calendar** — when vested tokens become sellable and how much — answers "is dilution coming in N days?" A 15%-of-supply cliff next week is a known future sell pressure you can hedge, wait out, or avoid. See [Token Unlocks, Vesting and Emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions). It earns a panel.

**Fees and revenue** — what users actually pay to use a protocol — answers "does anyone pay for this?" It is the single best separator of a real business from a subsidized farm. See [On-Chain Fundamentals: Fees, Revenue and TVL](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl). It earns a panel.

**Smart-money flow** — net buying and selling by a curated cohort of historically profitable wallets — answers "are the good wallets accumulating?" It is a *confirmation* signal, not a standalone buy trigger (more on that danger later), covered in [Following Smart-Money Wallets](/blog/trading/onchain/following-smart-money-wallets). It earns a panel, as a confirm leg.

#### Worked example: the exchange-flow panel that flagged a \$5M inflow

Take a mid-cap token you hold a \$50,000 position in. Your token-monitor dashboard has one panel: net exchange flow, refreshed every few hours, with a threshold line drawn at \$1M. For three weeks the panel hovers near zero — a few hundred thousand dollars in and out, normal churn. One morning the panel spikes: a net \$5M of the token has moved *to* centralized exchanges in 24 hours, against a token with only ~\$40M of daily real volume. That is supply equal to roughly 12% of a day's volume arriving where it can be sold, in one day.

What does that single panel change? It moves your action from "hold" to "trim or hedge." You sell a third of your position — \$16,667 — to lock gains ahead of the likely sell pressure, and set a tighter stop on the rest. If the inflow turns out to be one entity moving to an exchange for staking or an OTC desk (which the panel itself cannot distinguish), you have given up a little upside on a third of your stack. If it is what it usually is — a large holder preparing to sell — you have stepped out of the way of a 15–20% drop on \$33,333 of exposure, saving roughly \$5,000–\$6,600. **One panel, one threshold, one action measured in dollars: that is what earning a panel means.**

## Layout: arrange panels by the question they answer

A correct set of panels laid out badly is still a bad dashboard. Layout is not decoration; it is the difference between a screen that reads in five seconds and one you have to study. Three principles.

**Top-line health at the top.** The panels that tell you "is anything wrong, right now?" go where your eye lands first — the top row. For a token, that is holder growth, concentration, and liquidity-lock status. For a protocol, that is TVL, fees, and revenue trend. You should be able to glance at the top row and know whether to keep reading or close the tab. Everything that requires interpretation or drill-down goes below.

**Group by question, not by data source.** The natural-but-wrong instinct is to group panels by where the data comes from — all the DEX panels together, all the holder panels together — because that mirrors how you wrote the queries. The right grouping is by the *question a human asks*: "is supply leaving?" "who controls the float?" "is dilution coming?" A reader scanning your board thinks in questions, not in table joins. Put the exchange-flow panel next to the unlock calendar because both answer "is sell pressure coming?", even though they query completely different tables.

**Drill-downs below the summary.** Each top-line panel should have its detail directly beneath it. The concentration counter ("top-10 hold 38%") sits above the full holder-distribution chart that shows *how* that 38% is split. The reader gets the headline from the top, and drops down for the why only when the headline alarms them. This top-to-bottom reading order — summary, then detail — matches how attention actually works under time pressure.

### Vanity versus decision dashboards

We can now make the central distinction concrete, because it is fundamentally a layout-and-selection problem.

![Vanity dashboard of 28 redundant charts versus decision dashboard of 5 action-tied panels](/imgs/blogs/building-an-onchain-dashboard-2.png)

The before-after above is the difference in one picture. On the left is the vanity dashboard: 28 panels, dominated by price, market cap, and a dozen variations of the same volume chart — beautiful, screenshot-ready, and decision-free. You can stare at it all day and never know what to *do*. On the right is the decision dashboard: five panels, each one wired to an explicit action. Exchange net flow over +\$5M means trim. Unlock cliff in nine days means wait or hedge. Top wallet over 30% means size down or pass. Smart-money cohort net-positive with zero exits means hold. The vanity board maximizes panel count; the decision board maximizes signal density. The vanity board is for posting; the decision board is for trading.

The tell of a vanity dashboard is that you cannot finish the sentence "if this panel turns red, I will ___." If the honest answer is "I would look at it and feel something," it is vanity. If the answer is "I will trim 30% and tighten my stop," it is a decision panel. Run every panel through that sentence.

#### Worked example: a \$2M liquidity removal caught before the rug

A decision dashboard's value is clearest when it saves you from a loss. Say you are holding a \$10,000 position in a small DeFi token. Your token monitor has an LP-lock panel: it tracks the dollar value of liquidity in the main pool and whether the LP tokens are locked or held by the team. The panel shows a deep, *unlocked* pool — \$3M of liquidity, but the LP tokens sit in a team-controlled wallet, which the panel flags amber. You note the risk but stay in because momentum is strong.

One evening the panel updates: \$2M of the \$3M liquidity has been removed in a single transaction. The pool is now thin enough that your \$10,000 exit would move the price 30%+. This is the classic rug-pull mechanic from [Rug-Pull and Honeypot Detection](/blog/trading/onchain/rug-pull-and-honeypot-detection): the team pulls the floor, then the token collapses as holders scramble to sell into no liquidity. Because the panel surfaced the \$2M removal while ~\$1M of liquidity still remained, you exit immediately and recover roughly \$8,500 of your \$10,000 (eating ~15% slippage on the way out). The holders who only watched price saw it 40 minutes later — after the token had fallen 85% and the remaining liquidity was gone — and recovered \$1,500 or less. **The same starting position; a \$7,000 difference in outcome, decided entirely by whether the dashboard had an LP-lock panel.** For real-time protection you would wire this to an alert, which we cover at the end.

## Template A: the Token Monitor

Now we build, concretely. The token monitor is the dashboard you keep open for any token you hold or are watching. Its job is one question, decomposed: **is this token safe to hold, and is supply about to hit it?**

![Token Monitor dashboard layout with six panels for holders concentration LP lock exchange flow unlocks and smart money](/imgs/blogs/building-an-onchain-dashboard-3.png)

The layout above is the template. Six panels, arranged top-to-bottom by question, with no price chart anywhere. Let us go panel by panel — what each one queries and why it earns its place.

**Panel 1 — Holders and growth.** A line chart of unique holding addresses over time (say, 90 days). The query counts distinct addresses with a non-zero balance at each daily snapshot. *Why it earns its place:* a widening holder base is organic demand; a flat or shrinking one while price rises is a distribution-into-strength warning. It answers "is the base broadening or is this one whale's game?"

**Panel 2 — Concentration.** A counter (or small bar chart) showing the percentage of supply held by the top 10 non-liquidity, non-burn addresses. The query ranks balances, excludes known contract/LP/burn addresses, and sums the top 10. *Why it earns its place:* concentration is your direct cap on position size. If the top wallet holds 31%, you size down or pass — full stop. It answers "who can dump and end the trade?"

**Panel 3 — LP lock and depth.** A panel showing total liquidity in the main pool(s) in dollars and whether the LP tokens are locked (and until when). *Why it earns its place:* this is your rug tripwire, as the \$2M-removal example showed. It answers "can the floor be pulled this block?"

**Panel 4 — Exchange net flow.** The deposits-minus-withdrawals panel from the worked example, with a dollar threshold drawn in. *Why it earns its place:* the cleanest forward sell-pressure signal on-chain. It answers "is supply moving toward sell venues?"

**Panel 5 — Unlock calendar.** A small table or timeline of upcoming unlock events: date, percentage of supply, and the dollar value at the current price. *Why it earns its place:* known future supply you can position around. It answers "is dilution coming in N days?"

**Panel 6 — Smart-money holders.** A panel counting how many wallets from your curated smart-money cohort currently hold the token, and the net change over the last 7 days. *Why it earns its place:* a confirmation leg — good wallets accumulating is supporting evidence, not a buy trigger on its own. It answers "are the good wallets adding or leaving?"

Read top to bottom, those six panels give you the entire due-diligence picture in five seconds: base health (panels 1–3), supply pressure (panels 4–5), smart-money confirm (panel 6). And critically, *not one of them is a price chart*, because price is everywhere and carries no edge.

### The build, panel by panel

Here is the walkthrough — how you would actually assemble Panel 4 (exchange net flow) on Dune. The other panels follow the same shape: a query, a visualization, a threshold, a placement.

You start by *not* writing it from scratch. You search Dune for an existing "exchange flow" dashboard for the chain your token lives on, fork it, and find the query that computes net flow. The query, in simplified form, looks like this:

```sql
-- Net exchange flow for one token over the last 30 days.
-- Positive = net moving TO exchanges (sell-side); negative = leaving.
with labeled as (
  select
    t.evt_block_date as day,
    case when ex_in.address  is not null then 'inflow'
         when ex_out.address is not null then 'outflow'
         else 'other' end as direction,
    t.value / 1e18 as amount
  from erc20_transfers t
  left join exchange_addresses ex_in  on t."to"   = ex_in.address
  left join exchange_addresses ex_out on t."from" = ex_out.address
  where t.contract_address = {{token_address}}
    and t.evt_block_date > now() - interval '30' day
)
select
  day,
  sum(case when direction = 'inflow'  then amount else 0 end) as to_exchanges,
  sum(case when direction = 'outflow' then amount else 0 end) as from_exchanges,
  sum(case when direction = 'inflow'  then amount else 0 end)
    - sum(case when direction = 'outflow' then amount else 0 end) as net_flow
from labeled
group by 1
order by 1;
```

Notice the `{{token_address}}` — that double-brace is a **parameter**. On Dune you define it once at the dashboard level, and every parameterized query reads it. Change the parameter, and all six panels re-point at a new token. This is how one token-monitor dashboard serves every token you watch.

You then turn the `net_flow` column into a bar chart (positive bars red for sell-side inflow, negative bars green for outflow — color by *meaning*, the convention from the rest of this series). You draw a horizontal threshold line at the dollar amount that would change your action. You give the panel a one-line title that states the question: "Net exchange flow — is supply moving to sell venues?" And you place it in the second row of the layout, beside the unlock calendar, because both answer "is sell pressure coming?"

That is the whole loop: fork a query, parameterize it, visualize with meaningful color, draw the action threshold, place it by question. Repeat six times and you have a token monitor you can re-point at any token in seconds. Notice what the loop does *not* include: writing raw SQL against transaction logs, hand-labeling exchange addresses, or building a charting library. The community has done the hard parts; your job is selection, verification, and thresholds — the judgment layer, not the plumbing. That is why a careful analyst can stand up a genuinely useful token monitor in an afternoon, and why the temptation to over-build it with twenty more panels is the main thing standing between you and a board you will actually open every morning.

#### Worked example: sizing a position from the concentration panel

Concentration is the panel that most directly converts to dollars, so let us make the math explicit. Your maximum bankroll for a single speculative token is \$20,000. Your sizing rule, drawn from the concentration panel: full size only if the top non-LP wallet holds under 10% of supply; half size if 10–20%; quarter size if 20–30%; pass above 30%.

You open the monitor on a new token. Panel 2 reads: top wallet holds 24% of supply, top-10 hold 51%. That single number puts you in the quarter-size band. So your position is \$20,000 × 0.25 = \$5,000, not \$20,000. Six weeks later that top wallet begins distributing; the token falls 60%. On the \$5,000 you actually held, you lose \$3,000. Had the concentration panel not existed and you sized at the full \$20,000 on conviction, you would have lost \$12,000. **The concentration panel did not predict the dump — it sized you so the dump cost \$3,000 instead of \$12,000, a \$9,000 difference from one number on one panel.** That is the quiet, unglamorous value of a decision dashboard: not calling tops, but never being too big when the thing you cannot predict happens.

## Template B: the Protocol Health board

The token monitor asks "is this token safe and is supply hitting it?" The protocol-health board asks a deeper question: **is this protocol a real business, or is it renting growth with token emissions?** This is the board you run on DeFi protocols whose tokens you might hold for fundamentals, not just flow.

![Protocol Health dashboard layout with six panels for TVL fees price-to-fees active users emissions and revenue](/imgs/blogs/building-an-onchain-dashboard-4.png)

The layout above puts real cash flow on top and growth-plus-supply-risk below. Six panels.

**Panel 1 — TVL, ex-double-count.** Total value locked is the headline DeFi metric and the most abused one. Naively summed, TVL double-counts: deposit ETH into a lending market, borrow a stablecoin, deposit *that* back, and the same dollar gets counted twice (or five times in a recursive loop). A health board strips this — it counts net deposited capital, not recursive re-deposits. *Why it earns its place:* it answers "how much real capital is actually at work here?" The honest version of this metric is the whole subject of [Reading DeFi TVL Honestly](/blog/trading/onchain/reading-defi-tvl-honestly).

**Panel 2 — Fees and revenue.** *Fees* are what users pay in total; *revenue* is the slice the protocol (or its token holders) keeps. The query sums protocol fee events over time. *Why it earns its place:* this is the single best signal that a protocol is a business. People paying fees is demand you cannot fake the way you can fake volume. It answers "does anyone actually pay to use this?"

**Panel 3 — Price-to-fees (P/F).** Market cap divided by annualized fees — the on-chain cousin of a stock's price-to-earnings ratio. *Why it earns its place:* it tells you whether the market is pricing the protocol cheaply or for perfection. It answers "cheap or expensive relative to what it earns?"

**Panel 4 — Active users.** Daily unique active addresses interacting with the protocol, ideally with a retention overlay (how many of last month's users came back). *Why it earns its place:* it separates real demand from airdrop tourists who showed up for a reward and left. It answers "is there real, returning demand?"

**Panel 5 — Emissions vs revenue.** How many tokens the protocol mints to pay out as yield, valued in dollars, set against the fees it earns. *Why it earns its place:* this is the trap detector. A protocol paying \$10M of token emissions to attract deposits that generate \$1M of fees is *buying* its TVL with dilution — it is bleeding. It answers "is the yield real, or bought with inflation?"

**Panel 6 — Revenue trend (net of emissions).** Fees earned minus the dollar value of emissions, over time. *Why it earns its place:* it is the bottom line — is the protocol net cash-flow positive or subsidizing its own growth? It answers "is this a business or a Ponzi-shaped incentive program?"

The whole board exists to answer one decision: a 10× price-to-fees multiple on a protocol with real fees, sticky users, and shrinking emissions is a *position*; the same 10× on TVL that evaporates the day rewards stop is a *trap*. The board tells the two apart before you size in dollars.

#### Worked example: a \$50M-fee protocol at a \$500M cap

Let us run the board on a concrete case. A lending protocol's panels read as follows. Panel 1: \$2.1B TVL, ex-double-count — call it \$1.4B of genuinely net-new deposits after stripping recursive loops. Panel 2: \$50M in annualized fees, of which \$30M is kept as protocol revenue. Panel 3: market cap \$500M, so price-to-fees = \$500M ÷ \$50M = **10×**. Panel 4: 18,000 daily active addresses, with 60% month-over-month retention — sticky. Panel 5: \$8M of annual token emissions against the \$50M of fees. Panel 6: revenue net of emissions = \$30M − \$8M = **+\$22M**, and trending up.

Read together: this protocol earns \$50M, keeps \$30M, dilutes only \$8M, retains its users, and trades at 10× fees. That is a real business priced reasonably. Contrast a competitor whose board shows \$50M of fees but \$45M of emissions — net +\$5M, with TVL that would crater if rewards stopped. *Same headline fee number, opposite businesses.* The decision: the first protocol is a position you can size with conviction (say \$40,000 of a \$200,000 book at a 10× multiple you find fair); the second is a momentum trade you keep small (\$5,000) and watch the emissions panel like a hawk. **The P/F panel gave you the multiple; the emissions panel told you which 10× was real — a difference worth tens of thousands of dollars in how you allocate.**

## Template C: the Smart-Money Tracker

The third template follows the wallets that have historically been early and right. But it does so in the one way that survives contact with reality: **in aggregate, never as a single celebrity wallet.** This board answers "where is the smart cohort putting money, and is it a crowd or a mirage?"

![Smart-Money Tracker dashboard layout with six panels for cohort net flow rotation new positions breadth distribution and overlap](/imgs/blogs/building-an-onchain-dashboard-5.png)

The layout above tracks a curated *cohort* — say, 50 to 200 wallets you have tagged as historically profitable — and aggregates their behavior. Six panels.

**Panel 1 — Cohort net flow.** Total dollars the cohort bought minus dollars sold, daily, across all tokens. *Why it earns its place:* it is the headline — is the smart crowd risk-on or risk-off right now? It answers "is the cohort adding or distributing in aggregate?"

**Panel 2 — Sector rotation.** Cohort net flow broken down by narrative/sector (L2s, AI tokens, DeFi blue-chips, memecoins, etc.). *Why it earns its place:* it shows *where* the money is rotating, which is the actionable part. It answers "from which sector to which sector is the money moving?" This complements [Narratives and Sector Rotation On-Chain](/blog/trading/onchain/narratives-and-sector-rotation-onchain).

**Panel 3 — New positions.** Tokens the cohort first bought within the last 7 days. *Why it earns its place:* fresh positions are the early signal — what the smart wallets just started accumulating that is not yet obvious. It answers "what did the cohort buy that is still early?"

**Panel 4 — Cohort breadth.** How many distinct wallets are behind a given flow — not the dollar amount, the *count*. *Why it earns its place:* it is the bullshit detector. A \$20M flow from 40 wallets is a real rotation; the same \$20M from 3 wallets washing between each other is noise. It answers "is this a crowd or one wallet faking a trend?"

**Panel 5 — Distribution watch.** Which cohort wallets are *selling*, and what. *Why it earns its place:* the smart crowd quietly exiting is the most valuable warning the board can give. It answers "are the good wallets distributing?"

**Panel 6 — Conviction overlap.** Tokens currently held by more than, say, 5 cohort wallets independently. *Why it earns its place:* independent agreement is far stronger than one big bet. It answers "where is the cohort independently agreeing, not just one whale's position?"

#### Worked example: a \$20M cohort inflow into one sector

The board updates one morning and Panel 2 lights up: the cohort's net flow into "AI tokens" jumped to +\$20M over 48 hours, up from roughly zero the prior week. Before you act, you read the rest of the board the way it is designed to be read. Panel 4 (breadth): the \$20M came from 34 distinct wallets, not a handful — real crowd. Panel 3 (new positions): six AI tokens show up as fresh cohort buys this week. Panel 5 (distribution): zero cohort wallets are selling AI tokens. Panel 6 (overlap): two of those six tokens are now held by 8+ cohort wallets independently.

That is a clean rotation signal: broad (34 wallets), fresh (six new positions), one-directional (no distribution), and concentrated in agreement (two high-overlap tokens). The action: you allocate \$10,000 split across the two high-overlap names, front-running the narrative the cohort just started. You set your invalidation on Panel 5 — if cohort wallets start *distributing* AI tokens, you are out. Now compare the noise case: the same +\$20M, but Panel 4 shows it came from 3 wallets, and on inspection they are trading the same tokens *between each other* — wash flow dressed as conviction. Same \$20M headline, but breadth turned it from a \$10,000 trade into a pass. **The net-flow panel found the flow; the breadth panel told you whether \$20M meant 34 buyers or 3 — and that distinction was the whole trade.**

A serious caveat lives under this entire template, and the board is designed around it: following wallets blindly is a trap. Smart-money labels suffer survivorship bias (you see the wallet that 100×'d, not the 99 that died), wallets get used as bait, and naive copy-trading front-runs you into their exits. This is exactly why the tracker is built on *cohort* aggregates and *breadth*, not single wallets — and why [The Perils of Copy-Trading On-Chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) is required reading before you trade off this board.

## The panel admission test: a reusable design rule

We have selected metrics three times now, once per template. Let us extract the rule into something you can apply to *any* candidate panel, for any dashboard, forever.

![Panel admission test graph showing a candidate panel must answer a recurring question be verifiable and change an action or be cut](/imgs/blogs/building-an-onchain-dashboard-7.png)

The decision graph above is the test. Every proposed panel runs through three questions in order, and any "no" cuts it:

1. **Does it answer a recurring question I actually ask?** If it is a one-off curiosity, it belongs in an ad-hoc exploratory query, not on the board. *Cut.*
2. **Can I verify it on-chain?** If it is washable (raw volume), or it merely lags a number I already have (price), it is a vanity metric. *Cut.*
3. **Does it change an action measured in dollars?** If it is interesting but inert — nice to know, but you would do nothing differently — it fails the only test that matters. *Cut.*

Only a panel that clears all three ships. And when it ships, you do three things: add it, **draw the threshold** (the line at which the action changes — without it the panel makes you think every time instead of read at a glance), and **wire an alert** if the signal is time-sensitive. That last step is the bridge to the next section.

Apply this test honestly and most dashboards shrink by two-thirds. That is the goal. The discipline of the admission test is what turns the vanity wall into the decision board.

## How to read it: a daily walkthrough of the token monitor

Selecting and laying out panels is the build. *Using* the board is a daily ritual, and the ritual is what makes the panels pay off. Here is the five-second-to-five-minute pass through the token monitor for a token you hold.

**Second 1–5: the top row.** Glance at holders (panel 1), concentration (panel 2), LP lock (panel 3). Are holders flat or growing? Is concentration where it was, or did the top wallet's share jump? Is the LP still locked and deep? If all three are green and unchanged from yesterday, the base is fine — most mornings, you stop here and move on. The top row exists precisely so that a healthy token takes five seconds.

**Second 5–30: the supply row.** If you are still reading, look at exchange flow (panel 4) and the unlock calendar (panel 5). Did net flow cross your threshold? Is the next unlock cliff inside your time horizon? These are the panels that turn "hold" into "trim" or "hedge." Most of your actual trading decisions originate in this row.

**Minute 1–3: the drill-downs.** If a top-line panel alarmed you, drop into its detail. Concentration jumped? Open the full holder-distribution chart — is it one new whale, or did an exchange's hot wallet just reshuffle (a false positive)? Exchange flow spiked? Open the per-address breakdown — is it many small depositors (broad selling) or one entity (which might be OTC, staking, or a single seller)?

**Minute 3–5: the confirm.** Glance at smart-money holders (panel 6). Are the tagged wallets adding into the weakness you are worried about (reassuring) or leaving alongside it (confirming the worry)?

That is the discipline: the layout lets a healthy token cost you five seconds and a troubled one cost you five minutes, with the drill-downs sitting exactly where you need them. A board that makes *every* token cost five minutes — because the summary and detail are jumbled together — is a board you will stop opening. Reading speed is a design outcome, not an accident.

## Alerts from panels: when a dashboard is not enough

Here is the hard limit, and it is the most important sentence in this post: **a dashboard reflects yesterday.** Even a well-built board on a fast refresh shows you the chain on a delay — minutes to hours, depending on cadence and tier. For anything you slowly monitor (holder trends, fee growth, valuation), that is perfectly fine; those move on the scale of days. But for anything that can hurt you in minutes — a \$2M liquidity removal, a sudden \$5M exchange inflow, a contract upgrade — a dashboard you have to *open and look at* is structurally too slow.

The fix is to wire **alerts** on top of your panels. Every panel with a threshold line is, by definition, an alert waiting to be created: the threshold *is* the trigger condition. You take the SQL behind the exchange-flow panel, attach it to an alerting system, and have it ping you (Telegram, Discord, email, a bot) the instant net flow crosses +\$5M — instead of you noticing it the next time you happen to open the board. Building these is the subject of [On-Chain Alerts and Monitoring Bots](/blog/trading/onchain/onchain-alerts-and-monitoring-bots) and the anomaly-detection work in [Anomaly Detection: Building the Alerts](/blog/trading/onchain/anomaly-detection-building-the-alerts).

The clean division of labor: **the dashboard is for slow monitoring and context; the alerts are for fast triggers.** You design them together — each decision panel's threshold becomes an alert rule. The dashboard tells you the lay of the land each morning; the alerts tap you on the shoulder the moment something crosses a line while you are not looking. Build the board first (it forces you to define the thresholds), then promote the time-sensitive thresholds to alerts.

A useful way to decide which panels become alerts is to ask, for each one, "if this crossed its threshold at 3 a.m., would I want to be woken up?" Holder growth ticking down a little: no, that is a weekly read. A \$2M liquidity removal or a \$5M exchange inflow: yes, that is a wake-me-up event. The answer sorts your panels into the slow board and the fast alerts almost automatically. Panels that pass the "wake me up" test get an alert; the rest stay panels.

#### Worked example: promoting the LP panel into a \$1M alert

Return to the LP-removal scenario, but now wire it correctly. Your token monitor's LP panel has a threshold: flag any single liquidity removal over \$1M, or any removal that drops the pool below \$1.5M of remaining depth. On the dashboard alone, you would catch this only the next time you opened the board — which, on a busy day, might be hours later, long after the \$10,000 position was gone. So you promote the threshold to an alert: the same query that draws the panel now runs on a tight schedule and pings your phone the instant a removal over \$1M lands.

Three weeks later, at 2 a.m., \$2M leaves the pool. The alert fires; you wake, check, and exit while ~\$1M of liquidity still cushions your sell, recovering roughly \$8,500 of the \$10,000. The panel *defined* the threshold; the alert *delivered* it in time to act. Had you relied on the dashboard alone, the panel would have shown you the same \$2M removal at 8 a.m. — accurate, and \$7,000 too late. **The threshold is the same number on both; only the alert turns it into money saved, because a dashboard reflects yesterday and a rug does not wait for you to open a tab.**

## Forking and parameterizing community dashboards

You should rarely build a dashboard from a blank page, so let us close the build loop on the community-fork workflow that the foundations section promised.

**Forking** copies an existing public dashboard (and all its underlying queries) into your account, where you can edit it freely. On Dune, you open a well-built community dashboard — say, a generic "ERC-20 token analytics" board — and click fork. You now own a private copy with six working queries you did not have to write.

**Parameterizing** is what makes a forked board reusable. A good community board uses parameters (`{{token_address}}`, `{{chain}}`, `{{days}}`) so that one dashboard serves any token. After forking, you set the token-address parameter once, and all six panels re-point. The skill of forking well is reading the SQL to confirm it is parameterized (and correct) rather than hard-coded to someone else's token.

Two cautions. First, **verify the forked queries before you trust them.** A community dashboard's green numbers are someone else's logic, and that logic can be wrong, stale (using a deprecated table), or naive (summing TVL with double-counting). Read the SQL behind every panel you keep. The whole ethos of this series is *verify, do not trust a dashboard's green number* — that applies double to a dashboard you did not write. Second, **community labels drift.** The "exchange addresses" or "smart money" tables a forked board relies on are maintained by someone; confirm they are current, since stale labels silently corrupt every panel built on them. Labeling and attribution is its own discipline, covered in [Labeling and Attribution](/blog/trading/onchain/labeling-and-attribution).

The practical recipe: fork a board that is close to what you want, read every query, fix what is wrong or stale, swap the parameters to your target, cut the panels that fail the admission test, draw your thresholds, and promote the time-sensitive ones to alerts. That is faster than building from scratch and, done with the verification step, just as trustworthy.

## Common misconceptions

**"More panels mean a more thorough dashboard."** The opposite. Every dead panel lowers signal density and taxes your attention. A 28-panel vanity board is *worse* than a 5-panel decision board, not more thorough — you spend your morning scanning past 23 panels to find the 5 that matter. Thoroughness is in panel *selection*, not panel *count*. The best dashboards are mostly defined by what was left off.

**"A dashboard gives me real-time signals."** No — a dashboard reflects yesterday (or the last refresh, which on free tiers can be hours stale). It is a *slow monitoring and context* tool. Anything you need within minutes — a liquidity pull, a sudden exchange inflow — must be an *alert*, not a panel you have to open and look at. Confusing the two means you will see the rug on the dashboard 40 minutes after the alert would have fired.

**"Price is the most important panel."** Price is the *least* useful on-chain panel. It is on every site, it lags the flows that move it, and it carries no edge a one-layer-deeper metric could not carry better. Price is the output your other panels are trying to predict. Putting it front-and-center is mistaking the scoreboard for the game.

**"Higher TVL means a healthier protocol."** Not without two corrections. First, naive TVL double-counts recursive deposits — the same dollar counted many times. Second, TVL bought with token emissions is *rented*, not earned; it evaporates the day rewards stop. A health board strips double-counting and sets TVL against emissions and fees, precisely because the raw number lies. See [Reading DeFi TVL Honestly](/blog/trading/onchain/reading-defi-tvl-honestly).

**"Following smart-money wallets is a buy signal."** It is a *confirmation* leg at best, and a trap at worst. Smart-money labels suffer survivorship bias, wallets get used as bait, and naive copy-trading front-runs you into their exits. That is why the tracker template aggregates a *cohort* and weights *breadth* — and why you read [The Perils of Copy-Trading On-Chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) before trading off it.

## The playbook: what to do with it

The if-then checklist for turning a dashboard from a screen into decisions.

**Designing a panel — the admission test.**
- *Signal:* you are tempted to add a chart. → *Read:* run it through the three questions — recurring? verifiable? changes an action in dollars? → *Action:* if all three are yes, ship it (add, draw the threshold, wire an alert if time-sensitive); if any is no, cut it to an ad-hoc query. → *False positive:* an "interesting" panel that you would never actually act on — cut it however pretty it is.

**Token monitor — exchange inflow.**
- *Signal:* exchange net-flow panel crosses your dollar threshold (e.g. +\$5M against a \$40M-volume token). → *Read:* supply is moving toward sell venues. → *Action:* trim a third, tighten the stop on the rest; drop into the per-address drill-down to see if it is one entity or many. → *Invalidation/false positive:* the inflow is a single address that turns out to be an OTC desk or a staking move — re-add on confirmation it was not a sale.

**Token monitor — LP and concentration.**
- *Signal:* LP-lock panel shows a large liquidity removal, or concentration panel shows the top wallet above your size threshold. → *Read:* rug risk (liquidity) or single-point-of-failure risk (concentration). → *Action:* on liquidity removal, exit immediately while liquidity remains; on concentration, size down per your band (full/half/quarter/pass). → *Invalidation/false positive:* the "concentration" is a known locked-team or staking contract, not a sell-capable wallet — exclude it and re-read.

**Protocol health — the real-business test.**
- *Signal:* you are evaluating a protocol token for a fundamentals position. → *Read:* P/F panel for the multiple, fees-and-revenue for whether anyone pays, emissions-vs-revenue for whether growth is bought with dilution. → *Action:* size with conviction only if fees are real, users sticky, and net revenue (after emissions) positive and rising; otherwise keep it a small momentum trade and watch the emissions panel. → *Invalidation:* net revenue turns negative or active users roll over — the business is subsidizing itself; cut.

**Smart-money tracker — sector rotation.**
- *Signal:* cohort net-flow into a sector spikes (e.g. +\$20M). → *Read:* check breadth (how many wallets), new positions (is it fresh), distribution (is anyone selling), and overlap (independent agreement) *before* acting. → *Action:* if broad, fresh, one-directional, and overlapping, allocate to the high-overlap names front-running the rotation; set invalidation on the distribution panel. → *False positive:* the flow is 3 wallets washing between themselves — breadth exposes it; pass.

**The whole-board ritual.**
- *Signal:* it is morning and you sit down. → *Read:* five-second pass on the top row (health); thirty-second pass on the supply row (pressure); drill down only if something alarms. → *Action:* most mornings, nothing — the board's job is mostly to tell you it is safe to do nothing. → *Invalidation:* if every token takes five minutes, your layout is wrong (summary and detail are jumbled) — regroup by question and re-separate summary from drill-down.

The thread through all of it: a dashboard is not a feeling, a screenshot, or a wall of charts. It is the smallest set of panels where every single one finishes the sentence "if this turns red, I will ___" — with an action measured in dollars. Build that, draw the thresholds, promote the fast ones to alerts, and you have turned scattered queries into a daily decision tool.

## Further reading & cross-links

- [Writing On-Chain Queries with Dune](/blog/trading/onchain/writing-onchain-queries-with-dune) — the SQL behind every panel here; build the queries first, then assemble them into the dashboard.
- [On-Chain Alerts and Monitoring Bots](/blog/trading/onchain/onchain-alerts-and-monitoring-bots) — promote each decision panel's threshold to a real-time trigger; the dashboard's missing fast half.
- [The On-Chain Tooling Landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where Dune and Flipside sit among Nansen, Arkham, DeBank, and the rest.
- [Reading DeFi TVL Honestly](/blog/trading/onchain/reading-defi-tvl-honestly) — why the protocol-health board strips double-counting and sets TVL against emissions.
- [Following Smart-Money Wallets](/blog/trading/onchain/following-smart-money-wallets) and [The Perils of Copy-Trading On-Chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — how to build the smart-money cohort, and why you track breadth not celebrities.
- [Combining On-Chain with Off-Chain Signals](/blog/trading/onchain/combining-onchain-with-offchain-signals) — the next step: pairing your on-chain panels with price, social, and macro context.
- Supporting panels: [Exchange Flows](/blog/trading/onchain/exchange-flows-inflows-and-outflows), [Supply Distribution and Holder Concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration), [Token Unlocks, Vesting and Emissions](/blog/trading/onchain/token-unlocks-vesting-and-emissions), and [On-Chain Fundamentals: Fees, Revenue and TVL](/blog/trading/onchain/onchain-fundamentals-fees-revenue-and-tvl).

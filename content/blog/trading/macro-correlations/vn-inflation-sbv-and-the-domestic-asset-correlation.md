---
title: "VN Inflation, SBV Policy, and the Domestic Asset Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Alongside the imported correlation to US macro, the VN-Index has a strong domestic correlation: lower SBV rates plus a wider credit room lift Vietnamese banks, brokers and property, while rising VN CPI that forces tightening pulls them down."
tags: ["macro", "correlation", "vietnam", "vn-index", "sbv", "credit-growth", "inflation", "monetary-policy", "banks", "domestic-liquidity"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The VN-Index has a powerful *domestic* correlation that sits on top of its imported correlation to US macro: lower SBV policy rates plus an expanded credit room are strongly positive for Vietnamese equities, and rising VN CPI that forces the SBV to tighten is strongly negative.
>
> - Domestic liquidity is the home-grown version of the global-liquidity correlation. The State Bank of Vietnam (SBV) sets it with three levers — the **policy/refinancing rate**, the annual **credit-growth target (the "room")**, and open-market operations — and **credit growth is THE Vietnamese liquidity variable** because this is a bank-and-credit-driven market.
> - The centerpiece numbers: the VN-Index correlates about **−0.45** with a rise in the SBV policy rate, about **−0.30** with a rise in VN CPI, and about **+0.50** with foreign net buying. When the SBV cuts and widens the room, the highest-beta financials — **brokers, then property, then banks** — lead; the margin cycle amplifies the move both ways.
> - The home leg can override the imported leg. In 2023 the SBV cut from 6% to 4.5% even as the Fed held near its peak, and that domestic easing cushioned the VN-Index — Vietnam partly decoupled. In 2022 both legs pushed down at once and the VN-Index fell about **−33%**.
> - The one fact to remember: **in Vietnam, follow the credit room and the SBV rate the way you follow Fed liquidity in the US.** It is the single most reliable domestic driver of the index.

In the first half of 2023, two of the world's central banks were doing opposite things. The US Federal Reserve was finishing the most aggressive hiking campaign in forty years, dragging its policy rate toward 5.5% and telling markets it would stay "higher for longer." On paper, that is a hostile environment for a small, foreign-flow-sensitive, dollar-borrowing emerging market like Vietnam. The imported correlation — the VN-Index's roughly **+0.45** beta to the S&P 500 and its **−0.40** link to a stronger dollar, which the companion posts in this track measure in detail — said the VN-Index should have been under pressure all year.

It was not. After a brutal 2022, the VN-Index spent 2023 grinding *higher*, from a year-end 2022 close near 1,007 points back above 1,130 by year-end and on toward 1,290 by 2025. The reason was domestic. While the Fed held, the SBV went the other way: it *cut* its refinancing rate four times in the spring of 2023, from 6% back to 4.5%, and it kept the annual credit-growth target generous at roughly 14-15%. Cheaper money and more credit at home was pumping liquidity into a market whose biggest sectors — banks, brokers, property — live and die by exactly that. The home leg was pulling up even as the imported leg was, at best, neutral.

That is the whole thesis of this post. The VN-Index does not answer to one book; it answers to two. There is the **imported correlation** — to US risk, the dollar, and global liquidity — and there is a separate, often-stronger **domestic correlation** driven by the SBV, by credit growth, and by Vietnamese inflation. Most retail commentary watches only the imported leg ("the Fed did X, so VN-Index will do Y") and is repeatedly blindsided when the home leg dominates. By the end of this post you will be able to read both, know which one is in charge, and put numbers on the move.

![SBV rate and credit room feed domestic liquidity that lifts banks brokers and property and drives the VN-Index](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-1.png)

## Foundations: the SBV, the credit room, VN CPI, and what "domestic liquidity" means

Let us build this from absolute zero, because the payoff at the end depends on the plumbing being clear. We are going to define every actor, every lever, and the one statistical idea — correlation — that ties them to the index.

**What is a correlation, precisely?** Correlation, written *r*, is a single number between −1 and +1 that measures how tightly two things move together. An *r* of **+1** means they move in perfect lockstep the same direction; **−1** means they move in perfect opposite directions; **0** means no linear relationship. When this post says the VN-Index has a correlation of about **−0.45** with the SBV policy rate, it means: in the historical record, when the SBV *raised* its rate the index tended to *fall*, and when the SBV *cut* the index tended to *rise* — a moderately strong inverse relationship, but not a law of physics. (If correlation is genuinely new to you, the series opener [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) builds it from the ground up; here we take the definition as given.)

**Who is the SBV?** The State Bank of Vietnam is Vietnam's central bank — the rough equivalent of the US Federal Reserve, but with one important difference: it is an arm of the government, not an independent body. Its mandate is broad — price stability, currency stability, and supporting the government's growth targets all at once. That last part matters: the SBV is explicitly expected to help hit a GDP growth number, so its instinct, whenever inflation allows, leans toward *easy* money. The constraint is inflation and the exchange rate; when those are calm, the SBV eases, and when they flare, it tightens.

**The SBV's three levers.** The SBV moves domestic liquidity with three tools, and you must know all three because the most important one is not the one Western analysts instinctively watch.

- **The policy / refinancing rate.** This is the rate at which the SBV lends to commercial banks, and it anchors the cost of money in the system — the analogue of the Fed funds rate. It went from 4% in 2020-21, *up* to 6% across 2022 as the SBV defended the currency and fought imported inflation, then back *down* to 4.5% in the spring of 2023. The deep mechanism of why the policy rate is the master price of money lives in [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable); here we care about its *correlation* with the index.
- **The credit-growth target — the "room."** This is the lever with no clean US equivalent, and it is the most important one. Each year the SBV sets a target for how much total bank credit (loans) may grow across the system — typically around 14-15% — and it *allocates* that growth to individual banks as a quota, a "room" (the Vietnamese-English market term for the allowance). A bank that has used up its room literally cannot make new loans until the SBV grants more, regardless of how much capital or deposits it holds. The SBV can expand or tighten the room mid-year, and when it does, it is directly turning the system-wide credit tap. We will see that **credit growth is the single best domestic liquidity gauge** for the VN-Index.
- **Open-market operations (OMO).** Day to day, the SBV injects or drains short-term cash from the banking system by buying or selling paper on the open market — the same plumbing the Fed uses to keep the overnight rate on target. OMO sets the *short-term* tone (interbank rates) between the bigger policy-rate and credit-room moves.

**Why is credit growth THE Vietnamese liquidity variable?** Because Vietnam is a *bank-and-credit-driven* economy and a *bank-and-credit-driven* stock market. The capital markets are still developing; the corporate bond market is small and was itself a source of crisis in 2022; firms fund themselves overwhelmingly through bank loans, and households buy property and stocks heavily on bank-fed leverage. The three biggest blocks of the VN-Index by weight — banks, real estate, and securities brokers — are all, directly or indirectly, in the business of credit. So when the SBV widens the credit room, it is not a niche regulatory tweak; it is the equivalent of the Fed running quantitative easing straight into the sectors that dominate the index. This is why we call domestic liquidity the *home-grown version of the global-liquidity correlation* studied in [global liquidity and the everything-correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation): same mechanism (more liquidity lifts risk assets), different tap.

**What is VN CPI, and what is the SBV reaction function?** VN CPI is Vietnam's consumer price index, published monthly by the General Statistics Office (GSO); its year-over-year change is the headline inflation number. The "reaction function" is the rule of thumb the SBV is widely understood to follow: keep inflation roughly under a soft ceiling of about **4%**, and as long as CPI sits comfortably below that, prefer to ease (cut rates, widen the room) to support growth. When CPI threatens the ceiling — or when a falling currency is importing inflation — the SBV tightens. So VN CPI is not a direct driver of stocks; it is the *constraint* that decides whether the SBV is allowed to ease. Low, stable CPI gives the SBV room to cut, which is bullish; rising CPI removes that room and forces tightening, which is bearish. The chain is **VN CPI → SBV reaction → domestic liquidity → equities.**

**What is the margin-lending cycle?** One more piece of plumbing, because it is the amplifier that makes the Vietnamese domestic correlation so violent. *Margin lending* is when a broker lends an investor money to buy more stock than the investor's own cash would allow — the investor puts up, say, 50% and borrows the other 50% against the shares. In a market with heavy retail participation, margin is the turbocharger of a rally: as prices rise, the value of the collateral (the shares) rises, which lets investors borrow even more, which buys more stock, which pushes prices higher still. This is *reflexive* — the move feeds itself. The danger is that it runs in reverse just as hard: when prices fall, collateral values fall, brokers issue *margin calls* (demands to add cash or sell), forced selling drives prices down further, triggering more calls — a cascade. Because Vietnamese brokers fund their margin books with bank credit, the margin cycle is directly geared to the SBV's stance: easy money funds more margin, which amplifies the up-move; tight money starves margin, which amplifies the down-move. The margin cycle is why the same SBV rate change that moves the index a few percent can move the brokers two or three times as much.

**What is "domestic liquidity," then?** Pull the four pieces together. Domestic liquidity is the amount of cheap, available money sloshing through the Vietnamese financial system, set by the SBV's rate (the *price* of money), the credit room (the *quantity* of money the banks may create), and amplified by the margin cycle (the leverage layered on top). When rates are low, the room is wide, and margin is expanding, deposits earn little, credit is abundant, leverage is cheap, and money flows toward higher-returning assets — including stocks. When rates are high, the room is tight, and margin is unwinding, the flow reverses with a vengeance. The figure above shows the core chain: the SBV rate and credit room feed domestic liquidity, VN CPI acts as the brake that can shut the easing off, and the liquidity then lifts banks, brokers and property, which together carry the VN-Index.

## The centerpiece: the VN-Index domestic correlation in numbers

Now we put numbers on the home leg. The series data file records three domestic/flow correlations for the VN-Index: against a *rise* in the SBV policy rate, about **−0.45**; against a *rise* in VN CPI year-over-year, about **−0.30**; and against foreign net buying on the Ho Chi Minh exchange (HOSE), about **+0.50**. (These are researched approximations from public Vietnamese market data and published research, rounded for teaching — treat them as directionally solid, not tick-exact.)

![The VN-Index domestic correlation bars for SBV rate VN CPI and foreign flows](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-2.png)

Read the signs, because they encode the whole mechanism:

- **SBV policy rate, −0.45.** A rate *hike* (the rate rises) is associated with a *falling* index, and a *cut* with a *rising* index. The magnitude, −0.45, says the relationship is moderately strong — not as deterministic as a physics constant, but very much a relationship a trader must respect. This is the same direction as the global "rates down, risk up" rule, run through the SBV.
- **VN CPI YoY, −0.30.** Higher inflation is associated with a weaker index. The sign is negative for the reason built in the foundations: high CPI removes the SBV's room to ease and eventually forces tightening, which drains domestic liquidity. The magnitude is a touch weaker than the rate correlation because CPI acts *through* the SBV rather than directly — it is the constraint, not the lever.
- **Foreign net buy on HOSE, +0.50.** When foreigners are net buyers, the index rises; when they sell, it falls. This is the *flow* that connects the domestic and imported legs, and it is the strongest single correlation of the three. Foreign flows are themselves driven by the dollar and global risk (the imported leg) — the deep treatment is in the companion post [the dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — but they land *domestically*, on a market thin enough that a few hundred million dollars of foreign flow moves the whole index.

A correlation is a relationship, but a trader needs to translate it into an expected move. That conversion is what a *beta* does, and the next worked example does it.

#### Worked example: a VN-Index and bank move on an SBV rate cut

Suppose the SBV surprises the market with a 50 basis-point cut to the refinancing rate (0.50 percentage points), and that this is a genuine surprise — the market had priced no change. We want a back-of-the-envelope move for the VN-Index and for a bank stock.

Translate the correlation into a beta. With a correlation of about −0.45 between the *change* in the SBV rate and the index return, and using the historical ratio of typical index moves to typical rate moves, a reasonable rule of thumb is that a surprise 50bp cut is worth roughly a **+3% to +5%** move in the VN-Index over the following weeks (a cut is bullish, so the sign flips: rate down, index up). Take the midpoint, **+4%**.

Now size it in dollars. Say you hold a Vietnamese equity portfolio worth 2.4 billion VND. At roughly 25,000 VND per US dollar, that is about **\$96,000**. A +4% index move on a portfolio that roughly tracks the index is **+96,000,000 VND**, or about **+\$3,840**.

The bank sleeve moves more. Banks are the highest-weight domestic-cycle sector and carry a domestic beta above 1 — call it 1.3× the index. So a +4% index move implies roughly **+5.2%** on the bank holdings. If 800,000,000 VND (about **\$32,000**) of the portfolio is in banks, that sleeve gains about **+41,600,000 VND**, or about **+\$1,664** — nearly half the portfolio's total gain from less than a third of its weight.

The intuition: a single SBV cut does not just lower a discount rate; it widens the cheap-credit machine that the index's biggest sectors run on, so the highest-beta financials capture most of the move.

### How to measure it honestly: the full-sample number lies

A single correlation figure like −0.45 is an *average over a sample*, and like every correlation in this series, it is not constant — it strengthens and weakens with the regime, and a full-sample average can hide that. This is the central honesty of the whole series ([correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant)), and it applies to the domestic leg too.

Three traps to avoid when you measure the SBV-to-index correlation yourself:

- **Use changes, not levels.** The SBV rate sat at exactly 4.5% for three straight years (2023, 2024, 2025) while the index climbed. If you correlate the *level* of the rate against the *level* of the index over that window, you get noise, because the rate did not move. The correlation lives in the *changes* — the hike to 6%, the cut to 4.5% — which is why the data records the correlation against the *change* in the SBV rate (Δ), not its level. The same goes for VN CPI: correlate the *change* in inflation against returns, not the level against the level.
- **The relationship leads, it does not coincide.** A rate cut does not move the index on the same day and then stop; it works through credit growth and margin expansion over weeks and months. The index also *anticipates* — it often rallies on the *expectation* of a cut before the cut lands, then sells the news. So the measured correlation depends heavily on the lead/lag window you choose. The general framing of leading-versus-coincident relationships is in [lead, lag: leading, coincident and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators); for the SBV, treat the rate as a *leading-to-coincident* driver with effects that build over one to two quarters.
- **The window matters, and short windows whipsaw.** Compute the correlation over a rolling window and the number breathes: it can be strongly negative during a clean hike-or-cut cycle (2022-23) and near zero during a flat-rate stretch (2024-25, when the rate did not move). A too-short window gives a jumpy, unreliable estimate; a too-long one washes out the regime you care about. The trade-offs of window length are exactly those in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters). The practical rule: the domestic correlation is *real and strong during policy cycles* and *dormant during policy pauses*. Read it as a regime property, switched on when the SBV is actually moving.

The honest summary: −0.45 is the right number *for the regime in which the SBV is actively changing policy*. When the SBV is on hold, the domestic leg goes quiet and the imported leg, or a domestic non-liquidity shock, takes over. Knowing *when* the correlation is live is as important as knowing its sign.

#### Worked example: the margin cycle amplifying an SBV cut on a broker

Trace the amplifier explicitly. Take the same surprise 50bp SBV cut from before, worth about +4% on the index. We want the move on a broker, accounting for the margin cycle.

A broker's earnings have two engines geared to liquidity. First, *turnover*: an easing-driven liquidity surge pulls retail traders back in, and market-wide daily turnover can jump from, say, 15 trillion VND to 22 trillion VND — a +47% rise — lifting commission revenue proportionally. Second, *margin interest*: as the rally builds and collateral values rise, the broker's margin book expands; a book of 8 trillion VND (about **\$320 million**) growing to 11 trillion VND (about **\$440 million**) is +37% more interest-earning assets.

Both engines firing means the broker's *earnings* rise far more than the index's price. With a domestic beta around 1.8×, the +4% index move implies roughly **+7.2%** on the broker's stock in the first leg — and if the rally extends and margin keeps compounding, brokers routinely run +15% to +25% while the index does +8% to +12%. On a 600-million-VND (about **\$24,000**) broker position, that first-leg +7.2% is **+43.2M VND**, about **+\$1,730**; a full-cycle +20% would be **+120M VND**, about **+\$4,800**, from a single position.

The intuition: the SBV cut is the trigger, but the margin cycle is the multiplier — the broker's revenue is geared to *both* turnover and the size of its leveraged book, so the highest-beta sector captures a multiple of the index move (and gives it all back, plus more, when the cycle reverses).

## Why credit growth is the cleanest domestic gauge

The SBV rate gets the headlines, but the credit room is the lever that most directly turns the liquidity tap, because it controls the *quantity* of money the banking system can create. In a bank-funded economy, new credit is new money in the hands of firms and households — and a chunk of that money finds its way into property and the stock market. The deep mechanism of how lending literally creates money is worth reading once in full: [how credit creates money: the lending channel](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles) builds it from first principles. Here we use the result: **credit growth is the flow of new domestic liquidity, and the VN-Index tracks it.**

This is why the annual credit-room announcement, and any mid-year expansion of it, is a market event in Vietnam in a way it simply is not in the US. When the SBV lifts the room — say from a 14% target to 15%, or grants an extra tranche to the big banks in the fourth quarter — it is signing off on a wave of new lending that flows straight into the most index-heavy, most leverage-sensitive sectors. The market has repeatedly rallied on credit-room expansions for exactly this reason.

#### Worked example: sizing a credit-room expansion in VND and dollars

Let us put a number on what a credit-room bump actually injects. Suppose total system credit at the start of the year is 13,000 trillion VND (this is the right order of magnitude for Vietnam's banking system). The SBV sets the credit-growth target at 14%, then mid-year lifts it to 15% — a one-percentage-point increase in the room.

One extra percentage point of growth on a 13,000-trillion-VND base is **+130 trillion VND** of additional new credit the banks are now allowed to extend. At roughly 25,000 VND per dollar, 130 trillion VND is about **\$5.2 billion** of fresh credit released into the economy in a single policy decision.

Now scale it against the market. The total market capitalization of HOSE is on the order of 5,000 trillion VND (about **\$200 billion**). The extra **\$5.2 billion** of credit is roughly **2.6%** of total market cap — and that is just the *direct* first round; in a leveraged, reflexive market a meaningful fraction of new credit cycles into equities, and the margin channel (next section) multiplies it. A 1-point room bump is not a rounding error; it is a liquidity injection worth a few percent of the entire market's value.

The intuition: the credit room sets how fast the money supply that funds Vietnamese asset prices is allowed to grow, so an expansion is a direct, datable liquidity event the index reliably responds to.

### The credit-growth-to-equities transmission, channel by channel

It is worth being precise about *how* new credit becomes higher stock prices, because the channels are distinct and they explain why the financials lead. There are three, and they stack.

The first is the **direct leverage channel**. Some of the new credit is borrowed, directly or indirectly, to buy financial assets — through brokers' margin books, through holding-company structures, and through the simple fact that a household with cheaper access to a mortgage or a business loan has more free cash to put into the market. This is the most immediate channel and it lands first on the brokers, whose margin books are literally made of bank credit.

The second is the **earnings channel for the lenders**. The banks and brokers do not just *transmit* the credit; they *earn* on it. A bank that can grow its loan book 15% instead of 10% earns more net interest; a broker that funds a bigger margin book earns more margin interest. So the credit expansion lifts the *earnings* of the very sectors that dominate the index — a self-reinforcing loop where more credit both funds the buying and fattens the buyers' profits.

The third is the **funded-firm channel**. The non-financial companies that borrow the new credit — property developers, manufacturers, infrastructure builders — use it to fund projects that, if they pay off, raise those firms' future cash flows and therefore their stock prices. This channel is slower and less certain than the first two (it depends on the projects actually generating returns), which is exactly why property, the most credit-funded non-financial sector, has a high but *lagging and riskier* domestic beta: it benefits enormously when credit flows, but it is also where the credit-quality risk concentrates when the cycle turns, as 2022 showed.

Stack the three and you see why the domestic-beta ladder runs **brokers > property > banks > index > defensives**: the brokers sit on the fastest, most leveraged channel; banks earn on the volume; property gets the slower, riskier funded-firm channel; and defensives, which neither lend nor borrow heavily, barely participate. The credit room turns the tap; these three channels are the pipes.

## The SBV reaction function: VN CPI is the brake on easing

The reason 2023 could happen — the SBV easing into a Fed that was still tight — is that VN CPI stayed tame. Vietnamese inflation, despite the global inflation shock, never blew through the SBV's soft 4% ceiling: it ran about 3.2% in 2022, 3.3% in 2023, and stayed in the low-to-mid 3s through 2025. That calm is exactly what *gave* the SBV the room to cut. Had VN CPI spiked above 4-5%, the SBV would have been forced to hold or hike regardless of what it wanted to do for growth, and the domestic leg would have turned hostile.

![SBV refinancing rate and VN CPI YoY paths with the inflation comfort ceiling](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-3.png)

The figure above traces both series from 2019 to 2025. Two features carry the story. First, the SBV rate is *not* a smooth line: it sits at 4% through the easy 2020-21 period, spikes to 6% across 2022 as the SBV defends the dong against a surging dollar and fights imported inflation, then drops to 4.5% in 2023 — a clean hike-then-cut cycle. Second, VN CPI (the amber line) stays below the red 4% ceiling the entire time. The SBV's 2022 hikes were driven far more by the *exchange rate* — defending against a +9% surge in the dollar — than by runaway domestic inflation, which is a crucial subtlety: the imported leg (the dollar, the Fed) can force the SBV's hand even when domestic CPI is fine. That linkage is the subject of the companion post on [the USD/VND, DXY, and the rate-differential correlation](/blog/trading/macro-correlations/usdvnd-dxy-and-the-rate-differential-correlation).

The takeaway for the correlation: **VN CPI's −0.30 link to the index runs entirely through the SBV.** Inflation does not crush Vietnamese earnings the way it would, say, an airline's; it matters because it sets whether the central bank can keep the cheap-credit machine running. Watch CPI as the *permission slip* for easing, not as a direct earnings shock.

#### Worked example: the VN CPI to SBV to equities chain, step by step

Walk the chain with numbers so the indirect path is concrete. Start with a VN CPI print that comes in hot: say year-over-year inflation jumps from 3.4% to 4.6%, clearing the 4% ceiling.

Step 1 — the reaction function. With CPI above the ceiling, the SBV's room to ease evaporates; a market that had been pricing a 50bp cut over the next two quarters now prices a *hold*, and possibly a 25bp hike. The expected policy path shifts by roughly 75bp in the hawkish direction (from −50 to +25).

Step 2 — the rate-to-index beta. Using the same conversion as before (a surprise rate move of about 50bp is worth roughly 4% on the index in the opposite direction), a 75bp hawkish shift in the *expected* path is worth on the order of **−6%** on the VN-Index as the market re-prices.

Step 3 — the dollars. On a 2.4-billion-VND (about **\$96,000**) portfolio, a −6% move is about **−144,000,000 VND**, or roughly **−\$5,760**. The bank and broker sleeves, with betas above 1, would fall more — a broker sleeve at 1.6× beta loses closer to **−9.6%**.

The intuition: a single hot CPI print can cost the index several percent not because inflation hits company cash flows directly, but because it slams shut the SBV's easing window — the inflation correlation is really a *policy* correlation wearing an inflation hat.

## The sector cross-section: who has the highest domestic beta

A correlation with the index is an average over very different sectors. The domestic cycle does not lift all boats equally; it lifts the credit-sensitive financials first and hardest, and leaves the defensives behind. Understanding the cross-section is what turns "the SBV cut, so be bullish VN" into an actual portfolio.

![Matrix of how each sector behaves under SBV easing versus tightening](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-5.png)

The matrix above lays out five groups against the two regimes — SBV easing plus a wide credit room, versus SBV tightening plus a tight room. Read it as the playbook for which sector to own in which regime:

- **Brokers (securities firms) — the highest beta.** A securities broker's revenue is turnover (trading commissions) plus margin-lending interest. Both explode when domestic liquidity floods in: cheap money pulls retail traders into the market, turnover surges, and the brokers' own margin books — money they lend to clients to buy stocks — balloon. In an easing regime brokers lead the market up by a wide margin; in tightening, they are hit first as margin unwinds and turnover dries up. The dedicated sector treatment is [securities brokers: the highest-beta sector in Vietnam](/blog/trading/vietnam-stocks/securities-brokers-sector-vietnam-highest-beta).
- **Property — the rate-sensitive funder.** Real-estate developers are the most leveraged, most funding-dependent sector. When the SBV eases and credit reopens, presales recover and developer funding costs fall, so property re-rates hard. When the SBV tightens — and especially when, as in 2022, the *bond* market that funds developers freezes at the same time — property is the epicenter of stress.
- **Banks — the broad-market core.** Banks are the largest index weight, so as banks go, the index largely goes. Easing lifts credit volume and supports net interest margins and asset quality; tightening slows loan growth and raises bad-debt fears. Banks are the *broad* domestic-cycle proxy — the heartbeat of the index, treated in full in [the banking sector: the heartbeat of the VN-Index](/blog/trading/vietnam-stocks/banking-sector-vietnam-the-heartbeat-of-vn-index).
- **Margin balances — the accelerator.** Margin lending is not a sector but a force that runs *through* brokers and amplifies everything. In easing, margin balances expand and turn a rally into a melt-up; in tightening, forced margin unwinds turn a sell-off into a cascade. This is the reflexive amplifier of the whole domestic cycle.
- **Defensives (consumer staples, utilities) — the low-beta hideout.** These sectors have stable demand that does not swing with the credit cycle, so they lag badly in an easing boom but become relative winners (cash, dividends, lower drawdowns) when the SBV is tightening and the financials are falling.

![Sector beta ordering brokers property banks lead in easing](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-4.png)

The chart above shows the index-level expression of this: the VN-Index level (the dark line) plotted against the SBV-rate regime, with 2022's tightening shaded red and 2023-24's easing shaded green. The index fell about a third in the tightening window and recovered through the easing window — and within those moves, the brokers and property names did far more than the index, while defensives did far less. The ordering of domestic beta — **brokers > property > banks > index > defensives** — is the single most useful sector fact in Vietnamese macro trading.

#### Worked example: a USD-anchored domestic P&L across one easing cycle

Put the cross-section to work over a full easing cycle. Suppose at the start of 2023 you build a 2.4-billion-VND portfolio (about **\$96,000** at 25,000 VND/USD) tilted to the domestic cycle, anticipating the SBV cuts:

- 40% banks (960 million VND, about **\$38,400**)
- 25% brokers (600 million VND, about **\$24,000**)
- 20% property (480 million VND, about **\$19,200**)
- 15% defensives (360 million VND, about **\$14,400**)

Across 2023, say the VN-Index returns +12% as the SBV cuts and the room stays wide. Apply representative domestic betas to that index move: banks at 1.3× (+15.6%), brokers at 1.8× (+21.6%), property at 1.5× (+18%), defensives at 0.4× (+4.8%).

- Banks: 960M × 15.6% = **+149.8M VND** (about **+\$5,990**)
- Brokers: 600M × 21.6% = **+129.6M VND** (about **+\$5,180**)
- Property: 480M × 18% = **+86.4M VND** (about **+\$3,460**)
- Defensives: 360M × 4.8% = **+17.3M VND** (about **+\$690**)

Total gain: **+383.1M VND**, about **+\$15,320**, or **+16%** on the portfolio — meaningfully ahead of the index's +12%, entirely because the tilt loaded the high-domestic-beta financials in an easing regime. Had you run the same tilt into a *tightening* regime, the betas flip against you and that same tilt underperforms the index badly.

The intuition: in an easing regime the right trade is not "buy Vietnam," it is "buy the high-domestic-beta financials"; the margin and credit channels mean the brokers and property capture far more of the move than the index average suggests.

## Domestic versus imported: which leg is in charge?

We now have both halves. The VN-Index answers to an imported book (US risk, the dollar, global liquidity, foreign flows) and a domestic book (SBV rate, credit room, VN CPI). The full decomposition is worth seeing on one chart.

![Imported versus domestic decomposition of the VN-Index correlations](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-6.png)

The figure splits the VN-Index's full correlation set into the two legs. The imported leg: about **+0.55** to MSCI EM, **+0.45** to the S&P 500, **−0.40** to the dollar, **−0.25** to the US 10-year yield. The domestic leg: about **+0.50** to foreign net buying, **−0.45** to the SBV rate, **−0.30** to VN CPI. The striking thing is that the two strongest single correlations — MSCI EM at +0.55 and the SBV rate at −0.45, with foreign flows at +0.50 bridging them — are one from each book. Neither leg is clearly dominant on average; *which* one is in charge depends on the moment.

**Foreign flows are the bridge between the two books, and they are why Vietnam is so reactive.** Vietnam is a small, frontier-to-emerging market: total HOSE market cap is on the order of **\$200 billion**, a rounding error against the trillions in global equity. That smallness is the key to the whole imported-leg mechanism. When global risk appetite is high and the dollar is weak, foreign investors allocate more to emerging and frontier markets, and even a few hundred million dollars of net foreign buying — trivial globally — is a large flow relative to Vietnam's daily turnover, so it moves the index hard (the +0.50 correlation). When the dollar surges and global risk turns off, the same flow reverses and foreigners sell, dragging the index down regardless of what the SBV is doing domestically. So the foreign-flow channel is precisely the wire through which the imported leg (the dollar, the Fed, EM positioning) reaches the domestic market. The domestic leg, by contrast, does not need foreigners at all — it works through *domestic* credit and *domestic* margin, funded by *domestic* deposits. This is the structural reason the two legs can diverge: one is funded abroad and one is funded at home, and the SBV controls only the home tap.

A useful sizing intuition: because foreign holdings are only a slice of HOSE (and capped by foreign-ownership limits in many large-cap stocks), the *domestic* investor base — flooded with cheap credit and margin — can outvote foreign selling when the SBV is easing. That is exactly the 2023 mechanism: foreigners were net sellers for stretches of the year, but the domestic liquidity wave from the SBV cuts was large enough to absorb the selling and still push the index up. The home leg won because home money was bigger than the foreign outflow.

So when does the home leg override the imported leg? The cleanest way to think about it is a 2×2 of the two policy stances.

![Quadrant of when the domestic leg dominates the imported leg](/imgs/blogs/vn-inflation-sbv-and-the-domestic-asset-correlation-7.png)

The quadrant map above crosses the SBV stance (easing vs tightening) with the global stance (risk-on vs risk-off):

- **SBV easing into global risk-on (top-left).** Both legs push the same way — up. This is the strongest VN bull setup, exemplified by the 2020-21 liquidity boom when the SBV held rates at 4%, kept credit wide, *and* global liquidity was flooding everything. The VN-Index roughly tripled off its 2020 low.
- **SBV easing into global risk-off (top-right).** The two legs disagree, and the home leg can win — Vietnam *decouples* upward. This is 2023: the Fed held near its peak (imported leg neutral-to-negative) but the SBV cut from 6% to 4.5% and the domestic leg pulled the index higher. This is the quadrant retail commentary gets most wrong, because it is watching only the Fed.
- **SBV tightening into global risk-on (bottom-left).** The home leg drags and Vietnam lags a global rally — the domestic brake is on even as the world is bid.
- **SBV tightening into global risk-off (bottom-right).** Both legs push down at once. This is the worst case, and it is exactly 2022: the SBV hiked to 6% to defend the dong while the Fed hiked and the dollar surged, and the VN-Index fell about −33%.

The deeper machinery of why a 2×2 of macro states reorganizes the entire correlation map is the same as in [correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants), and the rotation of those regimes over time is [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock). The Vietnamese version simply adds a domestic policy axis on top of the global one.

#### Worked example: the 2023 decoupling, in dollars

Quantify the 2023 decoupling to show how much the home leg can be worth. Imagine two traders, each with a 2.4-billion-VND portfolio (about **\$96,000**), at the start of 2023.

Trader A watches only the imported leg. The Fed is holding near 5.5%, the dollar is firm, US 10-year yields are high — by the imported correlations alone (VN-Index roughly −0.40 to the dollar, +0.45 to the S&P which was choppy early in the year), Trader A expects a flat-to-down year for Vietnam and stays in cash, earning a deposit rate of about 5%: **+120M VND**, about **+\$4,800**.

Trader B also watches the domestic leg, sees the SBV cutting from 6% to 4.5% and the credit room staying wide, and stays invested with a domestic-cycle tilt. The VN-Index returns about +12% on the year; with the financials tilt from the earlier example, Trader B earns about +16%: **+384M VND**, about **+\$15,360**.

The gap is **+264M VND**, about **+\$10,560**, on a \$96,000 base — purely from recognizing that in the top-right quadrant the home leg, not the Fed, was driving the index.

The intuition: the single most expensive mistake in Vietnamese macro trading is to read only the imported leg; when the SBV eases into a hawkish Fed, the domestic correlation is in charge and the index decouples.

## Common misconceptions

**"Vietnam just follows the Fed / the S&P 500."** This is the imported-leg-only error, and it is wrong often enough to be dangerous. The imported correlation is real — about +0.45 to the S&P, −0.40 to the dollar — but it is only *one* of two books, and the domestic book is roughly as strong (−0.45 to the SBV rate). 2023 is the counterexample: the Fed was hostile all year and the VN-Index *rose*, because the SBV was easing. The correct mental model is two legs, with the quadrant deciding which one leads — not one leg that is always the Fed.

**"VN CPI directly drives Vietnamese stocks."** No — the −0.30 correlation runs almost entirely *through* the SBV. Inflation matters because it sets whether the central bank may ease, not because it directly compresses corporate earnings. The proof is in the 2022-23 contrast: VN CPI was actually a touch *higher* in 2023 (3.3%) than the index-crushing year of 2022 (3.2%), yet 2023 was an up year and 2022 was down −33%. Inflation per se did not move the index; the SBV's *policy response* did. Watch CPI as the SBV's permission slip, not as an earnings shock.

**"The SBV rate is the only domestic lever that matters."** It gets the headlines, but in a bank-and-credit-driven market the *credit room* is at least as powerful, because it controls the quantity of new money rather than its price. A market can rally on a credit-room expansion even with the policy rate unchanged, and it can stall despite a low rate if the room is exhausted and banks physically cannot lend. If you watch only the policy rate you will miss half the domestic liquidity signal.

**"Banks are the highest-beta way to play an SBV cut."** Banks are the *broadest* domestic-cycle proxy — the heaviest index weight — but they are not the highest beta. That title goes to the brokers, whose turnover-and-margin revenue model is geared even harder to a liquidity surge, with property close behind. In a sharp easing rally the brokers routinely double the move of the banks. Use banks for broad exposure; use brokers and property to maximize beta to the domestic cycle.

**"Foreign flows are a domestic signal."** Foreign net buying has the strongest single correlation with the index (+0.50) and it lands domestically, so it is tempting to file it under the home leg. But its *driver* is the imported leg — the dollar, global risk appetite, and EM positioning, as covered in [the dollar cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity). Foreign flow is the *transmission channel* through which the imported leg reaches the domestic market, not a home-grown signal. Treat it as the bridge between the two books.

**"A flat SBV rate means the domestic correlation is gone."** The opposite of the headline error, but just as wrong. When the SBV holds its rate flat for years (as in 2024-25), the *rate* correlation goes dormant — there is no change to correlate against — but the domestic leg is not dead. The credit room is still being set and adjusted, margin balances are still expanding and contracting, and the SBV can switch the rate lever back on at any time. The correct reading is that the domestic correlation is a *regime property*: strongly live when the SBV is actively moving, quiet when it is on hold, and ready to dominate again the moment policy turns. A dormant correlation is not an absent one — it is a coiled one.

## How it shows up in real markets

**2020-21: the liquidity boom (top-left quadrant).** Coming out of the COVID crash, the SBV held its refinancing rate at 4%, kept the credit room generous, and the world was awash in liquidity from global central banks. Both legs pushed up together. Vietnamese retail participation exploded — new brokerage account openings hit records — and margin balances ballooned as cheap credit poured into the market. The VN-Index roughly tripled from its March 2020 low near 660 to a January 2022 peak above 1,500. Brokers and property led; this is the textbook both-legs-up rally, and the sector-by-sector record is in [the 2020-21 liquidity bull](/blog/trading/vietnam-stocks/case-study-2020-2021-liquidity-bull-sector-by-sector).

**2022: the double hit (bottom-right quadrant).** Everything reversed at once. The Fed hiked the fastest in forty years and the dollar surged about +9%, dragging the dong down and forcing the SBV to hike its refinancing rate from 4% to 6% to defend the currency. At the same time, a domestic corporate-bond crisis erupted — regulatory tightening exposed fraud and over-leverage among property developers, and the bond market that funded them froze. Property and banks faced a contagion of funding stress; margin calls cascaded through the brokers. Both the imported and domestic legs pushed down simultaneously, and the VN-Index fell about **−33%** from its January peak to its November trough near 911. The anatomy of the bond-property-bank contagion is documented in [the 2022 bond crisis case study](/blog/trading/vietnam-stocks/case-study-2022-bond-crisis-property-bank-contagion). This is the regime where Vietnam's two legs reinforce on the downside — the most dangerous quadrant.

The 2022 episode is also the cleanest illustration of *why both legs matter*: an analyst watching only the SBV would have been puzzled, because domestic CPI was a tame 3.2% and yet the SBV was hiking — the hikes were a *currency defense* forced by the imported leg (the +9% dollar surge), not a domestic inflation fight. And an analyst watching only the Fed would have missed the domestic bond-fraud shock that turned a global sell-off into a Vietnamese collapse. You needed both books open to understand a −33% year that neither leg alone could explain.

**2023: the decoupling (top-right quadrant).** This is the post's headline case. The Fed held near its 5.5% peak all year — the imported leg was neutral-to-hostile. But VN CPI stayed tame in the low 3s, which *gave the SBV room*, and the SBV used it: four cuts in the spring took the refinancing rate from 6% back to 4.5%, and the credit room stayed generous. Domestic liquidity reflated. The VN-Index recovered from its 2022-end close near 1,007 back above 1,130, and the recovery was led — as the cross-section predicts — by brokers and the surviving property and bank names. Vietnam partly *decoupled* from a hawkish Fed because its home leg was easing. The divergence-by-sector record is in [the 2023-24 recovery case study](/blog/trading/vietnam-stocks/case-study-2023-2024-recovery-and-sector-divergence).

**The recurring credit-room rallies.** Beyond the big regime shifts, the Vietnamese market has a recurring micro-pattern: a rally on news that the SBV has expanded the credit room, often in the fourth quarter when the year's initial allocation is running low. Each room expansion is a datable liquidity injection (recall the worked example: a 1-point bump is roughly **\$5 billion** of new credit, a few percent of total market cap), and the brokers and property names — the most credit-sensitive — lead the pop. The day-of reaction mechanics of SBV and VN CPI events are covered in the event-trading companion [Vietnam events: SBV, VN CPI, and how the VN-Index reacts](/blog/trading/event-trading/vietnam-events-sbv-vn-cpi-and-how-vn-index-reacts).

## How to read it and use it

Here is the domestic-correlation playbook, distilled.

**Step 1 — locate the SBV stance.** Is the policy rate rising or falling, and is the credit room being widened or held tight? This is your single most important domestic read. Falling rate plus widening room is the green light; rising rate plus tight room is the red light. Watch the credit-room announcements as carefully as the rate decisions — the room is the quantity lever and it is uniquely Vietnamese.

**Step 2 — check the inflation permission slip.** Where is VN CPI relative to the ~4% soft ceiling? If CPI is comfortably below 4%, the SBV *can* ease and the green light is sustainable. If CPI is pushing toward or through 4% — or if the dong is falling and importing inflation — the SBV's room to ease is closing, and even a currently-low rate is at risk of reversing. CPI tells you whether the easing stance is *durable*.

**Step 3 — locate the global quadrant.** Is the world risk-on or risk-off (the Fed, the dollar, EM flows)? Combine it with the SBV stance to find your quadrant. Both-legs-up (SBV easing, global risk-on) is the strongest setup; both-legs-down (SBV tightening, global risk-off) is the most dangerous. The two off-diagonal quadrants are where the home leg and imported leg fight — and where reading both is worth the most, because the consensus (which usually watches only the imported leg) is most likely to be wrong.

**Step 4 — pick the sector beta.** Once you know the regime, choose your exposure along the domestic-beta ladder: **brokers > property > banks > index > defensives.** In an easing regime, tilt to the high-beta financials to maximize the move; in a tightening regime, rotate to defensives and cash, because the same betas that helped now hurt. The margin cycle means the financials' moves are amplified in *both* directions — that is the source of both the upside and the risk.

**Step 5 — size for the amplifier, not the average.** This is the step most people skip. The index-level correlations (−0.45 to the rate, +0.50 to flows) are *averages*; the sectors you will actually trade move two to three times as much because of the margin cycle. If you size a broker position as though it moves with the index, you will be carrying far more risk than you think on the way down. Set position sizes off the *sector* beta, not the index beta — a broker at 1.8× and a property name at 1.5× should be smaller positions, in risk terms, than the index allocation that the same conviction would imply. The reflexive amplifier that pays you in the up-move is exactly the thing that can wipe you out in the down-move, so respect it on the way in.

A note on stops and the margin cascade: in a tightening or risk-off turn, the danger in Vietnamese financials is not a slow grind but a margin-call *cascade* — forced selling that gaps prices down faster than a casual stop can fill. The practical defense is to cut high-beta exposure *early* when the regime flips (the moment the SBV signals a hold-or-hike or the dollar breaks higher), not to rely on a stop catching you mid-cascade. The domestic correlation is symmetric in sign but asymmetric in speed: it builds up gradually and unwinds violently.

**Putting it together — a live read.** Walk the four steps with the 2025 setup. The SBV rate sits at 4.5%, held flat for three years, and the credit room is generous at roughly 14-15% — a mildly green domestic light, though a *flat* rate means the domestic correlation is in its dormant, on-hold mode rather than firing on an active cut. VN CPI runs in the mid-3s, comfortably below the 4% ceiling, so the easing stance is durable and the SBV retains the *option* to cut if growth wobbles. Globally, the picture is mixed — the dollar softened to a 2025 close near 97 from 108 the year before, which is supportive for foreign flows. So the quadrant is roughly "SBV easy-on-hold into a softening dollar": a constructive backdrop, but one where you would lean on the *imported* leg (the weaker dollar pulling foreign flows back) more than the dormant domestic leg, and you would size your sector tilt modestly rather than loading maximum broker beta — because the domestic correlation is not switched fully on until the SBV is actively moving again. That is the discipline: the same checklist gives a *different* answer depending on which leg is live, and right now neither is at full throttle.

**What invalidates the signal.** The domestic correlation breaks down in two situations. First, when the imported leg is so violent it overwhelms everything — a global risk-off crash drags Vietnam down regardless of SBV easing, because in a true crisis all correlations go toward one (the cross-asset version is [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)). Second, when a *domestic* shock that is not about liquidity hits — the 2022 bond-market fraud crisis was a credit-quality shock, not a rate shock, and it hurt property and banks even before the SBV finished hiking. Always ask: is the move about the *price/quantity of liquidity* (the correlation holds) or about *credit quality / a specific shock* (the correlation can mislead)?

**The one number to carry.** The VN-Index correlates about **−0.45** with the SBV policy rate. In Vietnam, you follow the credit room and the SBV rate the way you follow Fed liquidity in the US — and when the home leg and the imported leg disagree, do not assume the Fed wins. In 2023 the home leg won, and it was worth about **\$10,000** on a **\$96,000** portfolio to have known it.

To close the loop with where the series started: a correlation is a property of a *regime*, not of an asset pair, and the Vietnamese domestic correlation is the clearest case of that lesson in the whole track. "Lower rates lift the VN-Index" is not a constant; it is a statement that is *strongly true while the SBV is actively easing*, *dormant while the SBV is on hold*, and *overridden when a domestic credit-quality shock or a global crisis takes the wheel*. Master the regime — the SBV stance, the credit room, the inflation permission slip, and the global quadrant — and the domestic correlation map falls out of it. That is the home-grown version of the single idea that runs through every post in this series: find the regime first, and the correlation follows.

## Further reading and cross-links

Within this series:

- [VN-Index and US macro: the imported correlation](/blog/trading/macro-correlations/vn-index-and-us-macro-the-imported-correlation) — the other leg: the VN-Index's beta to US risk and global growth.
- [USD/VND, DXY, and the rate-differential correlation](/blog/trading/macro-correlations/usdvnd-dxy-and-the-rate-differential-correlation) — how the dollar and the SBV-Fed rate gap move the dong and force the SBV's hand.
- [Global liquidity and the everything-correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation) — the global parent of the domestic-liquidity story.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — how the regime that selects correlations rotates over time.
- [Correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) — the 2×2 machinery behind the domestic-vs-imported quadrant map.

Mechanism and reaction (other series):

- [How credit creates money: the lending channel](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles) — why credit growth *is* new liquidity, from first principles.
- [Vietnam events: SBV, VN CPI, and how the VN-Index reacts](/blog/trading/event-trading/vietnam-events-sbv-vn-cpi-and-how-vn-index-reacts) — the release-day reaction mechanics for SBV and VN CPI.
- [The banking sector: the heartbeat of the VN-Index](/blog/trading/vietnam-stocks/banking-sector-vietnam-the-heartbeat-of-vn-index) and [securities brokers: the highest-beta sector](/blog/trading/vietnam-stocks/securities-brokers-sector-vietnam-highest-beta) — the two ends of the domestic-beta ladder.
- [The 2022 bond crisis: property-bank contagion](/blog/trading/vietnam-stocks/case-study-2022-bond-crisis-property-bank-contagion) and [the 2020-21 liquidity bull](/blog/trading/vietnam-stocks/case-study-2020-2021-liquidity-bull-sector-by-sector) — the two regimes, sector by sector.

---
title: "The Expectations Channel: Forward Guidance and Credibility"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How a central bank moves yields, spreads, currencies, and stock multiples with words alone — the reaction function, the rate path, the term premium, and why credibility is the cheapest policy tool of all."
tags: ["monetary-policy", "forward-guidance", "central-banks", "asset-valuation", "credibility", "term-premium", "reaction-function", "expectations", "yield-curve", "draghi", "taper-tantrum", "uk-gilt-crisis"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A central bank's most powerful lever is not the rate it sets today; it is what the market believes it will set tomorrow. Words move the expected *path* of rates, the path moves the discount rate on every asset, and the discount rate moves what assets are worth — all before a single bond is bought or a single rate is changed.
>
> - Markets price the **path**, not the spot rate. A 2-year yield is roughly the *average* of the overnight rate the market expects over the next two years, so guidance about the future reprices bonds today.
> - A long yield = **expected average short rate + a term premium**. Credible guidance leaves the path alone but squeezes the term premium — the compensation investors demand for *uncertainty* about the path — and that compression alone can move a 10-year yield 50-60bp with no rate change.
> - **Credibility is the cheapest policy tool there is.** A credible central bank moves markets for free: Draghi's three words in July 2012 — "whatever it takes" — cut Spain's 10-year yield from 7.6% to 1.6% with zero bonds bought. A credibility *loss* reprices everything violently: the 2022 UK mini-budget put 130bp on a 30-year gilt in days.
> - The one number to remember: **7.6% to 1.6%.** That is the entire expectations channel in a single before-and-after, and the bond-buying program that "did it" (OMT) was never once used.

On the morning of 26 July 2012, the euro was, in the cold language of the bond market, coming apart. Spain was paying 7.6% to borrow for ten years; Italy 6.6%. Those are not the yields of a solvent G7-sized economy — they are the yields a market charges when it is pricing a real chance that the borrower defaults or leaves the currency altogether. Investors were quietly building a "redenomination premium" into peripheral debt: the extra yield you demand for the risk that the euros you are owed get paid back in resurrected, devalued pesetas and lire. A self-fulfilling spiral was tightening — higher yields made the debt less sustainable, which justified still-higher yields.

Then Mario Draghi, the president of the European Central Bank, stood up at a conference in London and said nineteen words, of which three did the work: *"Within our mandate, the ECB is ready to do whatever it takes to preserve the euro. And believe me, it will be enough."* He bought nothing that day. He announced nothing concrete. The ECB's actual backstop — Outright Monetary Transactions, an open-ended promise to buy the bonds of a country under attack — was not unveiled until September, and over the following decade it was **never used, not once**. And yet Spanish and Italian yields began to fall that afternoon and kept falling for two years, until Spain was paying 1.6% — a level *below* what the United States paid at the time.

That is the expectations channel, in its purest possible form. No discount rate was cut. No liquidity was added. No currency was sold. A man changed what the market believed about what he *would* do, and the price of money in a quarter of the world's economy moved. This post is about how that works — the machinery by which words become valuations — and about its dark twin, the day the market stops believing you, when the same channel runs in reverse and disciplines a government in a matter of hours.

It is worth sitting with how strange this is. In every other channel, the central bank *does* something physical: it changes a rate, it buys a bond, it sells a currency. The action is the cause; the price move is the effect. In the expectations channel there is no action — the cause is a *change in belief*, and the central bank's only tool is its own credibility, accumulated over years and spendable in an instant. It is the closest thing in economics to magic, and like all magic it works only as long as the audience believes. The rest of this post is an attempt to take the magic apart and show you the gears: how a belief about the future becomes a number on a screen, why that number is sometimes worth more than any amount of money the bank could spend, and what happens on the day the trick stops working.

![Expectations channel chain: words to reaction function to rate path to discount rate to asset value](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-1.png)

This is the fourth of the four transmission channels this series tracks. The discount-rate channel moves asset prices by changing the actual cost of money; the liquidity channel moves them by changing how much money is sloshing around; the currency channel moves them through the exchange rate. The expectations channel is the strange one, because it moves prices through *belief* — and belief, it turns out, is the channel through which the other three are anticipated. The market does not wait for the rate cut; it prices the rate cut the instant it believes the cut is coming. Get the expectations channel right and you understand why the biggest market moves often happen on days when the central bank does *nothing at all*.

## Foundations: how an expectation becomes a price

Start with the single most important idea in all of asset pricing, stripped of its mathematics. **The value of any asset is the present value of the cash it will throw off in the future, discounted back at the rate of return you could earn elsewhere.** A bond pays coupons and principal; a stock pays dividends and buybacks; a building pays rent. To value any of them you do two things: you forecast the future cash, and you pick a *discount rate* to translate "a dollar in five years" into "a dollar today."

Everyday version: say a friend promises to hand you \$1,000 in one year, and you are completely certain they will. How much is that promise worth to you *right now*? Not \$1,000 — because if you had cash today you could put it in a savings account and grow it. If the safe rate is 5%, you would accept about \$952 today, because \$952 growing at 5% becomes \$1,000 in a year. The \$952 is the *present value*; the 5% is the *discount rate*. Now here is the entire point of this post in one sentence: **the central bank's words change that 5%** — and not the 5% you can earn today, but the average 5% the market expects to be able to earn over the *whole life* of the cash flow. Move that average, and you move the price of everything.

### The market prices the path, not the spot rate

A central bank sets one rate: the overnight rate at which banks lend to each other (in the US, the federal funds rate). That is a single number that applies for, effectively, one night. But almost nothing in finance has a one-night horizon. A 2-year note, a 10-year bond, a 30-year mortgage, a stock whose cash flows stretch out for decades — all of them care about the cost of money over *years*, not overnight.

So the market does something clever and relentless. It treats a longer-term interest rate as the market's best guess of the **average** of the overnight rate over that horizon. This is the *expectations hypothesis* of the yield curve, and while reality adds a wrinkle we will get to (the term premium), the core is exactly right: a 2-year yield is, to a first approximation, the average overnight rate the market expects over the next two years.

This is why guidance is so powerful. The central bank cannot reach out and set the 2-year yield directly. But it can change what the market *expects* the overnight rate to average — and the moment it does, the 2-year yield moves to match, automatically, with no purchase and no policy action. The 2-year note is, in a real sense, a *bet on the path* of the policy rate. Words that change the bet change its price.

There is a deeper reason this works, and it has a name: **rational expectations**. The assumption — borne out remarkably well in practice for liquid markets like government bonds — is that traders do not wait for the future to arrive before pricing it; they form a best guess using all available information and price *that guess today*, correcting it as new information lands. A market that prices the future the moment it forms a view about it is a market that responds to *information about the future* — and a central bank's words are exactly that. The whole reason a speech can move a yield is that the people who set yields are forward-looking. If markets were purely backward-looking — pricing only what has already happened — forward guidance would be powerless, because words about the future would carry no information a backward-looking trader would act on. They are not, so it does. This is also why the market often moves *before* the data: it is pricing the data it expects, and the actual release moves prices only to the extent it *differs* from what was expected. The "surprise" is the news; the expected part was already in the price.

#### Worked example: how a 2-year note prices an expected path of hikes

Suppose the overnight policy rate is 4.00% today, and the market expects the central bank to hike to 4.50% in six months, to 5.00% in a year, and hold there. What should the 2-year note yield be?

The 2-year yield is roughly the average overnight rate over the eight quarters. Lay out the expected path quarter by quarter:

- Quarters 1-2 (months 0-6): 4.00%
- Quarters 3-4 (months 6-12): 4.50%
- Quarters 5-8 (months 12-24): 5.00%

Average = (4.00 + 4.00 + 4.50 + 4.50 + 5.00 + 5.00 + 5.00 + 5.00) / 8 = 37.00 / 8 = **4.625%**.

So the 2-year note yields about 4.63% even though the overnight rate is 4.00% today. The extra 63bp is *entirely* the market pricing in the expected hikes. Now the central bank gives a speech that convinces the market the hikes will be *faster* — to 5.00% in six months, not twelve. Re-average with the new path and the 2-year yield jumps toward 4.75%+. **The bond repriced because the expected path changed — the spot rate never moved.** A trader who buys the 2-year before the speech and sells it after takes a loss not because anything happened, but because beliefs about the future happened.

### The reaction function: the market's model of the central bank

If markets price the expected path, then the single most valuable thing a trader can have is an accurate model of *how the central bank will set that path* — what data it watches, what it fears, what it will tolerate. Economists call this the central bank's **reaction function**, and the market spends enormous effort estimating it.

A reaction function is a rule of the form: *given inflation X, unemployment Y, and financial conditions Z, the central bank will set the rate to R.* The most famous textbook version is the Taylor rule, which says the rate should rise when inflation is above target and fall when output is below potential. No central bank follows a mechanical rule, but every central bank *behaves* as if it has one, and the market reverse-engineers it from speeches, votes, projections, and history.

Here is the crucial implication. **A piece of data does not move bond yields directly. It moves them through the reaction function.** A hot inflation print is not, in itself, bullish or bearish for bonds — it depends entirely on what the market believes the central bank will *do* about it. If the market is certain the bank will respond aggressively, a hot print sends the expected path higher and front-end yields jump. If the market believes the bank will look through it ("transitory"), the same print barely moves yields. The data is an *input*; the reaction function is the *machine*; the repriced path is the *output*.

![Reaction function graph: data surprise filtered through credibility into expected path and asset prices](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-3.png)

This is why two identical CPI reports can produce opposite market reactions a year apart: the reaction function in the market's head changed. It is also why "Fed-speak" is dissected like scripture. When an official uses the word "patient," "data-dependent," "vigilant," or "restrictive," the market is not parsing vocabulary for its own sake — it is updating its estimate of the reaction function, and therefore the expected path, and therefore the price of every rate-sensitive asset.

We cover *how to trade* these reaction-function shifts around the FOMC meeting in [trading the FOMC: statement, presser, and dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot), and the *statistical* tightness of the link between the expected path and front-end yields in [the Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation). Here we are after the *mechanism*: why words about the future are a policy lever at all.

## Forward guidance: turning words into an instrument

Once you accept that markets price the path, a remarkable possibility opens up. The central bank can ease or tighten *without touching the rate* — simply by changing what the market expects the future rate to be. This deliberate management of expectations is called **forward guidance**, and since the 2008 crisis it has been a first-class policy tool, not an afterthought.

Why did it become so important? Because central banks ran out of room on the rate itself. When the overnight rate hits zero (or thereabouts), you cannot cut it further in the normal way — you hit the *zero lower bound*. But the long rates that actually matter for the economy — mortgages, corporate borrowing, the discount rate on stocks — are averages of *expected future* short rates. If the bank cannot cut the spot rate below zero, it can still push long rates down by *promising to keep the spot rate at zero for a very long time*. That promise lowers the expected average, and the long rate falls. Guidance is how a central bank keeps easing after its main instrument is pinned.

There are two flavors, and the difference between them is the whole game.

**Calendar-based guidance** ties the promise to time: "we expect rates to remain exceptionally low *at least through mid-2015*." It is concrete and easy to price — the market just extends the flat part of the expected path out to the named date. But it is brittle. If the economy improves faster than expected, the bank is stuck choosing between breaking its word (a credibility hit) and keeping rates too low (an inflation risk).

**State-contingent guidance** ties the promise to economic conditions instead of dates: "we will keep rates near zero *until unemployment falls below 6.5%, so long as inflation stays under 2.5%*." This is the famous "Evans rule" the Fed adopted in December 2012. It is more robust — the guidance automatically adjusts as the economy evolves, so the bank never has to break a calendar promise. The cost is that it forces the market to forecast the *economy* in order to forecast the *policy*, which is harder and noisier. State-contingent guidance is, in effect, the central bank publishing a piece of its reaction function and saying "hold me to this."

There is a subtler axis that cuts across both flavors, and it is worth naming because it separates guidance that is merely *informative* from guidance that is genuinely *powerful*. Economists distinguish **"Delphic"** guidance from **"Odyssean"** guidance. Delphic guidance is a forecast: the central bank shares its honest view of where the economy and rates are headed, like the Oracle at Delphi describing the future. It is useful information, but it is not a commitment — if the forecast changes, the guidance changes, and the market knows it. Odyssean guidance is a *binding promise* — named for Odysseus lashing himself to the mast so he could not act on the Sirens' song. The bank deliberately commits to a path it might later be *tempted to abandon*, precisely so the market can rely on it. The classic Odyssean promise is "we will hold rates low even after the economy recovers and even after inflation rises a bit" — a pledge to be irresponsible *later* in order to ease *now*. Odyssean guidance is more powerful than Delphic guidance for exactly one reason: credibility. A forecast moves markets a little; a *believed commitment* moves them a lot. The entire potency of forward guidance scales with how credibly the bank can tie itself to the mast — which is to say, the expectations channel runs entirely on the fuel we turn to next.

The most extreme form of Odyssean guidance is to stop talking about the path and simply *fix a yield by decree*. The Bank of Japan did exactly this in 2016 with **yield-curve control (YCC)**: it pledged to pin the 10-year JGB yield near 0% and stand ready to buy unlimited bonds to enforce it. YCC is forward guidance taken to its logical end — the bank does not describe the expected path, it *guarantees the long yield itself*, and credibility does so much of the work that for long stretches the BOJ barely had to buy anything to hold the peg. The catch is the mirror image of OMT: a peg holds for free only while the market believes it, and when belief wavers, the bank must buy enormous quantities to defend it — which is why the BOJ ended up owning more than half of all outstanding JGBs before it began unwinding YCC in 2024. Credibility is cheap until it is challenged, at which point it can become very expensive indeed.

#### Worked example: how calendar guidance reprices the front end

Suppose the overnight rate is 0.25% and, with no guidance, the market expects the bank to start hiking in twelve months, reaching 2.00% by year three. The expected path averages, over three years, to roughly 1.20%, so the 3-year yield sits near 1.20%.

Now the bank issues calendar guidance: "rates on hold *at least through the end of year two*." The market erases the year-one and year-two hikes from its path. The new path is 0.25% for two full years, then a ramp to 2.00% in year three. Re-average:

- Years 1-2: 0.25% (held by guidance)
- Year 3: ramps from 0.25% toward 2.00%, averaging ~1.10%

Three-year average ≈ (0.25 + 0.25 + 1.10) / 3 ≈ **0.53%**.

The 3-year yield falls from ~1.20% to ~0.53% — a **67bp easing** — and the bank did not cut the overnight rate by a single basis point. It eased by *deleting expected hikes from the market's path*. That is forward guidance as a real, measurable instrument: the words did the work that a 67bp rate cut would have done, at the zero bound where a 67bp cut was impossible.

### The dot plot and the SEP: guidance as a published forecast

In 2012 the Fed began publishing the **dot plot** — a chart, released four times a year as part of the Summary of Economic Projections (SEP), in which each of the 19 FOMC participants anonymously marks where they think the appropriate policy rate will be at the end of each of the next few years and "in the longer run." It is the most literal form of forward guidance imaginable: the policymakers, individually, drawing the expected path on a chart and handing it to the market.

The dots are not a promise, and the Fed says so loudly — they are projections, conditional on each official's economic forecast, and they shift meeting to meeting. But the *median* dot is the closest thing the market has to the central bank's own published path, and it moves markets hard. When the median 2024 dot moves up by 25bp between two meetings, the market reads it as the committee collectively revising its reaction function tighter, and the front end sells off accordingly — no rate change required.

The dot plot's great virtue is transparency; its great vice is that markets sometimes treat projections as commitments. When the dots say one thing and the data then says another, the Fed has to choose between following its own dots (and looking mechanical) or overriding them (and looking unreliable). Managing that tension *is* the modern art of central banking — and it is why every dot-plot release is followed by a press conference whose entire purpose is to explain how literally to take the dots.

## The term premium: where uncertainty lives in a yield

We have been pretending a long yield is *just* the average of expected short rates. It is mostly that — but not exactly. The wrinkle is the **term premium**, and it is where the credibility story really bites.

The term premium is the extra yield investors demand for *committing their money for a long time* instead of rolling over short-term loans. Why demand extra? Because the future is uncertain, and locking in a fixed rate for ten years exposes you to the risk that rates turn out higher than you expected — that you are stuck holding a 4% bond while everyone else earns 6%. The term premium is the *compensation for bearing that uncertainty*. It is, in plain terms, the price of not knowing the path.

So the full decomposition is:

**Long yield = expected average short rate (the path) + term premium (the uncertainty about the path).**

This split is the key to understanding why guidance is so potent. Guidance works on *both* terms. It can shift the expected path (move the first term). But even when it leaves the path exactly where it was, it can compress the *term premium* (shrink the second term) — because the entire job of the term premium is to compensate for uncertainty about the path, and clear, credible guidance *removes that uncertainty*. If the central bank tells you, believably, exactly what it will do and why, you no longer need to be paid much to bear path risk, because the path is no longer risky.

![Term premium decomposition: a 10-year yield split into expected path and term premium, before and after guidance](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-5.png)

#### Worked example: decomposing a 10-year yield into expectations and term premium

A 10-year Treasury yields 4.50%. Decompose it. The market expects the overnight rate to average 3.50% over the coming decade — that is the *expectations* component, the path. The remaining **4.50% − 3.50% = 1.00%** is the *term premium*: the compensation investors demand for the risk that the path turns out higher than expected, plus the inconvenience of locking up money for ten years.

Now the central bank delivers a stretch of clear, credible communication. The market still expects the *same* 3.50% average path — guidance did not change where rates are going. But it removed much of the *uncertainty* about that path, so investors no longer need 1.00% of compensation to bear it; 0.40% is enough. Re-add the pieces:

New 10-year yield = 3.50% (unchanged path) + 0.40% (compressed term premium) = **3.90%.**

The yield fell 60bp. **No rate was cut, and the expected path is identical — the entire move came out of the term premium, paid for by credibility.** That 60bp flows straight into every long-duration valuation in the economy: mortgages reprice, corporate borrowing costs fall, and the discount rate on a 25-year stock cash flow drops, lifting its fair value by several percent. This is the channel through which calm, predictable central banking is *itself* a form of monetary easing — it lowers the price of bearing risk.

The reverse is the nightmare. When a central bank becomes *unpredictable* — when the market can no longer model its reaction function, or no longer trusts it to control inflation — the term premium *blows out*. Investors demand far more compensation to lend long, the long yield spikes even if the expected path has not moved, and every long-duration asset gets cheaper at once. A credibility shock is, mechanically, a term-premium explosion. Hold that thought; it is the engine of the UK gilt crisis we will dissect below.

## Long and variable lags: why guidance is the only fast tool

There is one more reason expectations are the channel that matters most, and it is about *time*. Milton Friedman's most durable phrase is that monetary policy works with "long and variable lags." When a central bank changes the actual overnight rate, the effect on inflation and employment takes — by most estimates — somewhere between twelve and twenty-four months to fully arrive, and the exact lag varies cycle to cycle. The rate you set today is fighting the economy of next year, through a fog.

That lag is a deep problem. It means a central bank that waits for inflation to actually appear before acting is already a year too late, and a bank that waits for inflation to actually fall before easing keeps policy too tight for a year too long. The only way out of the lag trap is to act on *expectations* — to shape what the economy expects *before* the slow machinery of actual rate changes grinds through. Guidance is the fast lever precisely because it works at the speed of belief, not at the speed of the real economy. A credible promise about the future tightens or eases *financial conditions* — the yields, spreads, and exchange rates that businesses and households actually face — on the day it is made, months before any rate move would have bitten.

This is also why credibility compounds. A central bank the market believes can *front-run its own lags*: it says what it will do, the market reprices immediately, financial conditions adjust, and the economy starts responding to the guidance long before the rate moves. A central bank the market does *not* believe has lost its fastest tool and is left with only the slow, lagged instrument of actual rate changes — fighting next year's economy through a fog, with one hand tied.

## How credibility is earned, measured, and lost

We have used "credibility" as if it were obvious. It is not — it is a specific, measurable thing, and understanding how it is built and tracked is what turns the expectations channel from a story into a tool.

Credibility, for a central bank, means one concrete thing above all: **the market believes you will bring inflation back to target.** When that belief is firm, *inflation expectations are anchored* — households, firms, and investors expect inflation to be roughly 2% (or whatever the target is) over the long run, regardless of what this month's print says. Anchored expectations are the single most valuable asset a central bank owns, because they are self-fulfilling: if everyone expects 2% inflation, workers ask for raises consistent with 2%, firms set prices consistent with 2%, and inflation tends to *be* 2%. The anchor does much of the central bank's job for it, for free.

How do you measure something as intangible as the market's belief about future inflation? You read it out of asset prices, because the expectations channel works both ways — if the market prices the future, then market prices *contain* the market's expectations. The cleanest gauge is the **breakeven inflation rate**: the gap between the yield on a normal nominal bond and an inflation-protected bond (a TIPS, in the US) of the same maturity. That gap is, almost by construction, the inflation rate the market expects over that horizon. Another is the **5-year, 5-year forward breakeven** — the expected average inflation rate over the five years *starting* five years from now — which strips out near-term noise and shows the *long-run anchor* specifically. When central bankers say they are watching whether "longer-term inflation expectations remain well anchored," this is the number they mean. As long as that forward measure sits near target, the central bank retains its credibility — and its cheapest, most powerful lever — no matter how ugly the current inflation print.

This is also the early-warning system. Credibility does not usually vanish in a single day (the UK gilt crisis is the violent exception); more often it *drifts*. The first sign of a credibility problem is the long-run inflation anchor starting to slip — the 5y5y forward breakeven creeping up, signalling that the market is beginning to doubt the bank's resolve. That drift is precisely what frightened the Fed out of "transitory" in late 2021: not the headline print alone, but the risk that *expectations* would un-anchor and a temporary inflation shock would harden into a permanent one. Once expectations un-anchor, the only way back is the Volcker way — crush the economy hard enough to prove resolve and re-earn the anchor — which is enormously more painful than guarding the credibility in the first place. The asymmetry is the whole reason central banks are so cautious: an intact anchor is a free, powerful tool; a lost anchor is regained only at the cost of a recession.

## Common misconceptions

**"The central bank moves markets by buying and selling."** Sometimes — but the biggest moves often come with *no transaction at all*. Draghi crushed peripheral spreads in 2012 with a sentence; the OMT program credited with "doing it" was never used. The 2013 taper tantrum sent the US 10-year yield up 136bp on a *hint* that purchases might slow — no purchase changed. The transaction is often just the thing that makes the words credible; the words do the repricing.

**"A rate decision is what matters; the press conference is commentary."** Backwards. On most FOMC days the rate decision is fully expected and barely moves anything — the market priced it weeks ago. The volatility comes from the *guidance*: the statement's wording, the dot plot, and the chair's answers, all of which update the reaction function and therefore the *path*. The decision is old news; the guidance is the news.

**"If inflation is high, bond yields must rise."** Only if the market believes the central bank will *respond*. High inflation under a credible inflation-fighter can leave long yields *low*, because the market trusts the bank to bring inflation back down — so the expected average future short rate, and the inflation compensation built into the yield, both stay anchored. Volcker's whole achievement was earning the credibility that let long yields fall *while* he was still fighting inflation. Yields follow the *reaction function*, not the inflation print.

**"Forward guidance is just talk; only actions count."** Talk *is* an action when it changes the expected path, because the path is what assets are priced off. A credible promise to hold rates low for two years eases financial conditions today exactly as a rate cut would — that is not a metaphor, it is the arithmetic of the worked examples above. The market does not discount cash flows against what the bank *did*; it discounts them against what the bank is *expected to do*.

**"A central bank can always talk markets into anything."** This is the most dangerous myth, and the case studies exist to demolish it. Words work *only* to the extent the market believes them. Credibility is a finite, hard-won, easily-destroyed asset. When it is intact, words are free and powerful. When it is gone, words are ignored and the same channel runs in reverse — violently. The expectations channel is not a magic wand; it is a *credibility-powered* lever, and the power supply can fail.

## Case studies: when words moved more than money

### Draghi, 2012: three words and a backstop never used

Return to where we started, now with the machinery to see it clearly. By July 2012 the euro-area periphery was in a self-fulfilling sovereign-debt spiral. The market had built a large *redenomination premium* into Spanish and Italian bonds — the term premium was exploding, driven not by an expected-path story but by an existential fear about the currency itself. Spain's 10-year yield hit 7.6%, Italy's 6.6%. At those levels the debt arithmetic does not close, which justifies still-higher yields, which makes the arithmetic worse: a classic doom loop.

![Draghi spread collapse: Spanish and Italian 10-year yields before the speech and by 2014](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-2.png)

Draghi's "whatever it takes" did something precise: it attacked the *term premium*, specifically the redenomination-risk piece of it. By promising an unlimited, credible backstop (formalized in September as OMT), he made the bet "the euro breaks up and I get repaid in pesetas" a losing one. The redenomination premium that had inflated peripheral yields had no reason to exist if the ECB stood behind the currency — so it collapsed. Spain's yield fell to 1.6% by 2014, *below* the United States. And the spending that supposedly accomplished this? Zero. OMT was never activated. The promise was self-enforcing: precisely *because* it was credible, it never had to be used.

#### Worked example: how Draghi's words lifted a peripheral bond's price

Take a Spanish 10-year bond with a 5% annual coupon and €100 face value. In July 2012, with the market yield at 7.6%, that bond does not trade at par — its price is the present value of its cash flows discounted at 7.6%. Discounting ten €5 coupons plus the €100 principal at 7.6% gives a price of roughly **€82** (the bond trades at a discount because its coupon is below the market yield).

Now the redenomination premium collapses and the yield falls to, say, 4.0% over the following two years (on its way to 1.6%). Re-discount the same cash flows at 4.0%, and the price rises to roughly **€108** — the bond now trades at a premium because its 5% coupon is above the new market yield.

That is a price gain of about **(108 − 82) / 82 ≈ 32%** on the bond, plus the coupons collected along the way — a huge total return on a *government bond*, a thing that is supposed to be boring. And the entire move was a repricing of the term premium, set in motion by a sentence. A holder of Spanish debt who believed Draghi on 26 July made one of the great macro trades of the decade by doing *nothing* — just holding through the credibility-driven re-rating. The currency channel and the discount-rate channel never touched it; the expectations channel did all the work. (We trace the full mechanics of this episode in the dedicated case study, [Draghi 2012: "whatever it takes" and the spread machine](/blog/trading/policy-and-markets/draghi-2012-whatever-it-takes-and-the-spread-machine).)

### The 2013 taper tantrum: a hint, not an action

If Draghi is the expectations channel's triumph, the 2013 taper tantrum is its warning. On 22 May 2013, Fed Chairman Ben Bernanke, in congressional testimony, suggested that the Fed *might*, in coming meetings, begin to slow ("taper") its pace of bond purchases — *if* the economy continued to improve. He changed no policy. He announced no taper. He attached an explicit condition. And the bond market detonated.

![Taper tantrum: the US 10-year yield jumps 136bp on a hint of slower bond buying](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-4.png)

The US 10-year Treasury yield went from 1.63% before the hint to 2.99% by early September — a **136bp** move, in four months, on *words about a possible future change to the pace of an ongoing program*. Why so violent? Because the market had built its entire expected path — and a very compressed term premium — on the assumption that the Fed's purchases would continue indefinitely. The faintest suggestion that the flow might slow forced a wholesale re-estimation of the reaction function: if the Fed will taper purchases when the economy improves, then it will also hike rates sooner, so the expected path steepens *and* the term premium that QE had suppressed comes flooding back. Both terms in the decomposition moved at once, and the long yield gapped.

The damage did not stay in the Treasury market. The taper hint sent a shock through the entire world that had been relying on the Fed's easy stance. Emerging-market currencies and bonds — which had attracted huge inflows during the zero-rate, full-QE years as investors reached for yield — sold off sharply as the expected US path steepened and the dollar firmed. The "Fragile Five" (Brazil, India, Indonesia, South Africa, Turkey) saw their currencies tumble and were forced into defensive rate hikes of their own. A single conditional sentence about the *pace* of US bond buying tightened financial conditions across a dozen economies that the Fed has no mandate over at all. That is the expectations channel operating globally: the world prices the Fed's *expected* path, so a change in that expectation reprices assets everywhere the dollar reaches.

The taper tantrum taught central banks a permanent lesson about the expectations channel: *the market reacts to the change in expected policy, not to the policy itself,* and a clumsy hint can move markets more than a deliberate action. Every central bank since has obsessed over communicating changes to its programs gradually, with heavy advance signaling, precisely to avoid re-running 1.63%-to-2.99% on a stray sentence. When the Fed *actually* began tapering in December 2013, it had pre-signaled the move so thoroughly that the market barely flinched — the same policy action, drained of surprise, moved nothing, because the expectation had already been priced. The contrast between the May hint (136bp) and the December action (a shrug) is the cleanest possible proof that it is the *change in expectation*, not the policy, that moves markets. The broader balance-sheet mechanics — how the *flow* of purchases sits underneath these expectations — are the subject of [QE vs QT: how balance-sheet policy moves markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets).

### The 2021 "transitory" call: the slow bleed of credibility

Not every credibility event is a single dramatic day. The most expensive one of the recent era was a *slow* one. Through 2021, as inflation climbed, the Fed repeatedly characterized it as **"transitory"** — a temporary artifact of pandemic supply snarls and reopening demand that would fade on its own. The reaction function the Fed was broadcasting said, in effect, *we will look through this; no aggressive response is coming.*

![The cost of transitory: inflation ran a year ahead of the policy rate before the fastest hiking cycle since Volcker](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-8.png)

For a while the market believed it, and that belief was itself stimulative: with the expected path anchored near zero, financial conditions stayed loose even as inflation rose. But inflation kept climbing — past 5%, past 7%, toward a 9.1% peak in mid-2022 — and the "transitory" framing curdled from a forecast into a *credibility liability*. The Fed formally retired the word in late November 2021, and over the following sixteen months it was forced to hike 525 basis points — from a 0.25% ceiling to 5.50% — the fastest tightening cycle since Volcker.

#### Worked example: the credibility cost of getting the reaction function wrong

Why did the Fed have to hike *so hard, so fast*? Because it had spent a year telling the market it would not, and the market had set the expected path accordingly. Lay out the cost.

Had the Fed begun signaling a normal tightening in early 2021, when inflation first ran hot, the market would have built a *gradual* path — say a steady climb to a 2.5-3.0% peak over two years. Instead, the "transitory" guidance held the expected path near zero for roughly a year (the shaded credibility gap in the chart, where inflation ran far above the policy rate). When the Fed finally capitulated, it had to do two jobs at once: catch the rate up to where it should already have been, *and* re-anchor inflation expectations that had started to drift — re-earn the credibility it had spent.

The arithmetic of the catch-up is brutal: **525bp in 16 months**, versus perhaps 275bp spread over 24 months on the gradual path that earlier honesty would have allowed. The extra speed and height of the hiking cycle — and the 2022 wreckage it caused, with the S&P down 19.4%, the aggregate bond index down 13.0%, and the classic 60/40 portfolio down 16% in a single year — was, in significant part, the *bill for a year of mispriced guidance*. Credibility is cheap to spend and expensive to rebuild: the Fed eased financial conditions for free in 2021 by promising restraint, then paid for it with a violent, faster-than-necessary tightening in 2022. The lesson the term-premium framework makes precise: guidance that the market later judges *wrong* does not just fail to help — it forces a larger correction than honest guidance ever would have required.

### The 2022 UK gilt crisis: credibility snaps in days

The expectations channel's most violent recent demonstration ran in reverse, and it disciplined a G7 government in under a week. On 23 September 2022, the new UK government under Prime Minister Liz Truss delivered a "mini-budget": roughly £45 billion of *unfunded* tax cuts — tax cuts with no offsetting spending cuts or revenue plan, to be financed entirely by additional borrowing — announced into an environment of already-high inflation, with the Bank of England tightening and, pointedly, *without* the customary independent fiscal forecast from the Office for Budget Responsibility.

![UK gilt crisis: the 30-year gilt yield jumps 130bp and sterling hits a record low after the mini-budget](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-6.png)

The market's verdict was instant and brutal. The 30-year gilt yield jumped roughly **130bp** in days; sterling fell to an all-time low of \$1.035 against the dollar on 26 September. This was not an expected-path story — the Bank of England had not changed its rate path. It was a pure **credibility snap**: the market abruptly demanded a much larger term premium to hold UK government debt, because it could no longer trust the UK's commitment to sound public finances. The fiscal lever (unfunded borrowing) and the monetary lever (a tightening central bank) collided, and the market repriced the *risk of holding gilts at all*. The term premium exploded exactly as the decomposition predicts.

It got worse through a hidden plumbing problem. UK pension funds had used "liability-driven investment" (LDI) strategies that were leveraged to gilt prices; as gilt prices crashed (yields spiked), those funds faced collateral calls, were forced to *sell gilts to raise cash*, which pushed gilt prices down further, which triggered more calls — a doom loop. On 28 September the Bank of England had to step in with up to £65 billion of emergency gilt buying to stop a disorderly collapse — the central bank intervening to clean up a *fiscal*-credibility shock. The policy was reversed within weeks; Truss resigned after 49 days, the shortest premiership in British history.

#### Worked example: the 2022 credibility snap on a 30-year gilt

Quantify the violence. Take a 30-year gilt with a 1.5% coupon — a very long-duration bond, so extremely sensitive to yield changes. A useful rule of thumb: a long bond's price moves by approximately *(−duration × change in yield)*, and a 30-year gilt at low coupons has a duration in the neighborhood of 20 years.

The yield jumped about **+130bp = +1.30%** in days. Price impact ≈ −20 × 1.30% = **−26%.** A "safe" government bond lost roughly a quarter of its value in *days* — not because the Bank of England changed its rate path, but because the market lost faith in the government's fiscal credibility and demanded a far larger term premium to hold the debt.

Now layer on the leverage. A pension fund holding that gilt at, say, 3x leverage in an LDI structure does not lose 26% — it loses roughly 3 × 26% ≈ **78%** of its equity in the position, triggering collateral calls it can only meet by selling gilts into a falling market, deepening the very move that is destroying it. That is the doom loop in numbers, and it is why the Bank of England had to intervene within 48 hours. The whole episode is the expectations channel's mirror image: credibility, which moves markets for free when intact, *destroys* markets at speed when it snaps. (The full LDI mechanics are the subject of [the 2022 UK gilt and LDI crisis: when credibility snapped](/blog/trading/policy-and-markets/the-2022-uk-gilt-and-ldi-crisis-when-credibility-snapped).)

### Putting the four together

![Credibility matrix: cost of guidance, term premium, and bond reaction across intact, strained, and snapped credibility](/imgs/blogs/the-expectations-channel-forward-guidance-and-credibility-7.png)

Lay the four cases side by side and the structure is unmistakable. Draghi shows credibility *intact*: words are free, the term premium compresses, yields fall in an orderly collapse, and the backstop is never used. The 2013 taper tantrum and the 2021 "transitory" episode show credibility *strained*: the market doubts the message, the term premium starts rebuilding, and policy has to do more (and faster) than it promised. The 2022 UK gilt crisis shows credibility *snapped*: the term premium blows out, yields spike disorderly, and the central bank is forced into a huge intervention to clean up someone else's fiscal mess. Same channel, three states, wildly different costs — which is exactly why credibility, not the rate, is the asset a central bank guards most jealously.

## What it means for asset values

The expectations channel reprices assets in a strict order, and knowing the order is the playbook.

**The front end of the yield curve moves first and most directly.** The 2-year note is the cleanest expression of the expected path; when guidance shifts the path, the 2-year yield moves almost mechanically. If you want to read what the market thinks a central bank just signaled, the front end *is* the readout — it is the reaction function, priced. A hawkish surprise sends the 2-year up; a dovish one sends it down, often within seconds of a statement crossing the wire.

The *shape* of the curve carries the message too. When guidance pushes the expected path higher in the near term but the long-run anchor holds, the curve *flattens* or even *inverts* — short yields rise toward and past long yields — because the market is pricing hikes now and cuts later. An inverted curve is, read through this lens, the market saying "the central bank is tightening hard today, which means it will have to ease tomorrow, probably because it is about to slow the economy." When credibility is *lost*, the opposite happens at the long end: the curve *steepens* as the term premium blows out and long yields spike independent of the near-term path — the bear-steepening that defined the 2022 UK gilt move and that always signals the market is repricing *risk*, not the path. Curve shape is the expectations channel's most information-dense readout: the front end shows the path, the long end shows the credibility, and the spread between them shows which of the two is moving.

**Long yields move through both terms — the path and the term premium.** Clear, credible guidance can pull a 10-year yield down even with the expected path unchanged, by compressing the term premium (the Draghi mechanism). Lost credibility blows the term premium out and spikes long yields even with the path unchanged (the UK gilt mechanism). When you see a long yield move *more* than the expected path justifies, you are watching the term premium — which means you are watching credibility.

**Equities reprice through the discount rate.** A stock is a long-duration cash-flow stream; a lower expected-rate path and a compressed term premium lower its discount rate and lift its multiple, especially for long-duration "growth" names whose cash flows sit far in the future. This is why a dovish guidance surprise can rip the equity market on a day the central bank does nothing — the discount rate on decades of future cash just fell. The full equity mechanics live in [how monetary policy moves stocks: discount rates and sectors](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).

#### Worked example: how a guidance shift reprices a growth stock

Take a stock whose fair value is the present value of a long stream of growing cash flows. A standard shortcut (the Gordon growth model) values it as *next year's cash flow divided by (discount rate − growth rate)*. Suppose the stock will throw off \$5 per share next year, those cash flows grow at 4% per year forever, and the market discounts them at 9% (a 5% real cost of capital plus the rate environment). Fair value = \$5 / (0.09 − 0.04) = \$5 / 0.05 = **\$100.**

Now the central bank delivers credible dovish guidance. It does not cut the spot rate — but it lowers the *expected path* and compresses the *term premium*, and the discount rate the market applies to this stock falls from 9% to 8.5%. Re-value: \$5 / (0.085 − 0.04) = \$5 / 0.045 = **\$111.** The stock is worth about **11% more** — on a day the central bank changed *nothing* but expectations.

Notice the leverage: a mere 50bp move in the discount rate produced an 11% move in fair value, because the cash flows are long-duration and the denominator (discount rate − growth) is small, so a small change in it swings the result hard. This is the precise mechanism behind the violent equity reactions to guidance surprises — long-duration growth stocks are *enormously* sensitive to the discount rate, and the expectations channel moves the discount rate without moving the spot rate at all. The further out a company's profits sit, the more its value is really a bet on the expected rate path — which makes the equity market, in part, a leveraged trade on the central bank's credibility.

**Credit spreads and the currency follow credibility.** A credible central bank compresses risk premia broadly — sovereign spreads, corporate spreads — because the future is more predictable and the tail risks are backstopped. A credibility loss does the reverse and, critically, *hits the currency*: a central bank the market no longer trusts to control inflation or defend financial stability sees its currency sold (sterling to \$1.035; the dollar's own 10.7% first-half slide in 2025 as policy credibility was questioned). Credit and FX are the *broad* readouts of credibility, where the bond market's verdict spills into every other asset.

**The signal to watch:** the gap between what the central bank is *saying* and what the data is *doing*. When guidance and data agree, the channel is calm and credibility accrues. When they diverge — when the bank says "transitory" while inflation accelerates, or "data-dependent" while the data screams — the market starts pricing the *resolution* of the gap, and that is when the term premium gets restless. **What would invalidate the read:** a central bank that consistently does what it said it would do, on the schedule it implied, rebuilds credibility and makes its words cheap and powerful again — at which point the violent repricings stop and guidance goes back to moving markets for free. The fuller treatment of how the whole monetary toolkit fits together is in [the monetary toolkit: rates, QE, QT, and forward guidance](/blog/trading/policy-and-markets/the-monetary-toolkit-rates-qe-qt-and-forward-guidance), and the institutional question of *who is even allowed to make these promises* is in [who actually sets policy: the Fed, Treasury, and Congress](/blog/trading/policy-and-markets/who-actually-sets-policy-fed-treasury-and-congress).

## The one idea to keep

Strip everything else away and the expectations channel reduces to a single, slightly unnerving truth: **a central bank's most powerful instrument is belief.** It can move the price of money across an entire economy without spending a cent, if — and only if — the market believes it. That "if" is credibility, and credibility is the cheapest policy tool in existence precisely because, when you have it, the market does the work for you: it prices your promises as though they were already actions. Draghi moved a quarter of the world's bond market with three words and a backstop he never used, because the market believed him.

But credibility is also the most dangerous tool, because it is the one you can lose in an afternoon. When it goes — when a government delivers £45 billion of unfunded promises into a skeptical market, or a central bank spends a year insisting inflation is "transitory" — the same channel that moved markets for free runs in reverse and reprices everything at once, violently, with the central bank reduced to cleaning up the wreckage. The expectations channel is the lever that costs nothing to pull when your credit is good and everything when it is not. That is why central bankers, who could in principle do anything, spend most of their energy doing the one thing that keeps the lever working: being believed.

## Further reading & cross-links

- [Who actually sets policy: the Fed, Treasury, and Congress](/blog/trading/policy-and-markets/who-actually-sets-policy-fed-treasury-and-congress) — the institutional question of who holds the levers and who is allowed to make credible promises.
- [The monetary toolkit: rates, QE, QT, and forward guidance](/blog/trading/policy-and-markets/the-monetary-toolkit-rates-qe-qt-and-forward-guidance) — where guidance fits among the central bank's other instruments.
- [Draghi 2012: "whatever it takes" and the spread machine](/blog/trading/policy-and-markets/draghi-2012-whatever-it-takes-and-the-spread-machine) — the full case study of the purest expectations-channel event.
- [The 2022 UK gilt and LDI crisis: when credibility snapped](/blog/trading/policy-and-markets/the-2022-uk-gilt-and-ldi-crisis-when-credibility-snapped) — the credibility-loss mirror image, with the LDI doom-loop mechanics in full.
- [Trading the FOMC: statement, presser, and dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) — the trader's positioning playbook around the reaction-function readouts.
- [The central bank toolkit: rates, QE, QT, forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — the trader's-lens companion on guidance and the "Fed put."
- [The Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) — the statistical tightness of the link between the expected path and front-end yields.
- [QE vs QT: how balance-sheet policy moves markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) — the balance-sheet flow that sits underneath the taper-tantrum expectations.
- [How monetary policy moves stocks: discount rates and sectors](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors) — how the repriced discount rate flows into equity multiples.

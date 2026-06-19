---
title: "The Dollar (DXY): Cross-Asset Gravity"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the US dollar is negatively correlated with almost everything priced in it or funded by it, how strong each link is, what drives the dollar, and when the relationship breaks."
tags: ["macro", "correlation", "dollar", "dxy", "currencies", "gold", "emerging-markets", "commodities", "crypto", "cross-asset", "rate-differential", "dollar-smile"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The dollar is the denominator of the global financial system, so its level is *negatively* correlated with almost everything priced in it or funded by it: gold (about −0.55), copper (−0.50), oil (−0.45), emerging-market equities (−0.55), Bitcoin (−0.35), and weakly the S&P 500 (−0.20). The one positive link is the US 10-year yield (+0.40), because higher US yields pull capital into dollars.
>
> - A **stronger dollar tightens global financial conditions** and a weaker dollar is a global risk-on tailwind; that single sentence explains most cross-asset moves.
> - The strongest negative link is **emerging markets**, because so much EM borrowing is in dollars — a rising dollar literally raises their debt burden.
> - The driver is the **rate differential** (higher US yields vs the rest of the world pull money into dollars) plus the **dollar smile** (USD rises in both a US-outperformance boom *and* a global risk-off bust).
> - **The one number to remember:** in 2022 the DXY surged about 28% from its 2020 low to a 20-year peak of **114.8**, and that single move helped drive Bitcoin −64%, long bonds −31%, and EM equities into the ground all at once.

In September 2022, a chart that almost nobody outside foreign-exchange desks usually watches became the most important picture in finance. The US Dollar Index — the DXY — printed **114.8**, its highest level in two decades. The yen had collapsed to 152 per dollar. The euro had fallen *below parity* for the first time since 2002. The British pound briefly traded near \$1.03 in a sterling crisis. And as the dollar went vertical, *everything else fell at the same time*: the S&P 500 was down 18% on the year, the Nasdaq down 32%, long Treasuries down 31%, gold flat-to-down, and Bitcoin down a staggering 64%.

To a beginner, that looks like chaos — a dozen unrelated markets all having a bad year for a dozen unrelated reasons. To anyone who understands the dollar, it was one trade. The dollar is the *unit of account* for the global financial system. When the price of that unit goes up, the price of nearly everything measured in it, or borrowed in it, goes down. The dollar isn't just another asset on the board. It's the gravity that bends the whole board.

That is the thesis of this entire post, and it is worth stating plainly before we build the machinery underneath it: **the dollar is cross-asset gravity.** Its level correlates negatively with commodities, gold, EM, and crypto, more weakly with US stocks, and positively with US yields — and once you can read the DXY, you have a single dial that tells you whether global financial conditions are tightening or loosening for *all* risk assets at once.

![The dollar at the center pulling commodities, EM, gold, and crypto down as it rises](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-1.png)

## Foundations: what the dollar index actually is and why everything is priced in it

Before we can talk about *correlation* — the statistical measure of how two things move together — we have to be precise about what "the dollar" even means as a number you can chart.

### Correlation, in one minute

A **correlation** is a single number between −1 and +1 that summarizes how two things tend to move together. **+1** means they move in perfect lockstep (one up, the other always up by a proportional amount). **−1** means they move in perfect opposition (one up, the other always down). **0** means no linear relationship at all. The symbol statisticians use is *r* (Pearson's correlation coefficient). When we say "DXY and gold have a correlation of about −0.55," we mean: across a long history of weekly or monthly moves, when the dollar rose, gold usually fell, and the relationship was moderately strong but far from mechanical. (For the precise machinery — Pearson vs. Spearman vs. beta, and why a single number hides regime shifts — see [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta).)

The single most important thing to internalize, and the theme of this whole series, is that **correlation is a regime, not a constant.** The −0.55 between gold and the dollar is a long-run average; in some years it's −0.8, in others it briefly goes positive. We will return to this honesty again and again.

### What the DXY is: a basket, not "the dollar"

There is no single price of "the dollar," because a currency is only ever worth something *relative to another currency*. So the market built an index. The **US Dollar Index (DXY)** is a weighted basket measuring the dollar against six other currencies. The weights, set when the index launched in 1973 and essentially frozen since, are:

- **Euro (EUR): ~57.6%** — by far the dominant component, so DXY is *mostly* a EUR/USD chart inverted.
- **Japanese yen (JPY): ~13.6%**
- **British pound (GBP): ~11.9%**
- **Canadian dollar (CAD): ~9.1%**
- **Swedish krona (SEK): ~4.2%**
- **Swiss franc (CHF): ~3.6%**

When you see "DXY = 100," it means the dollar is worth exactly what it was at a 1973 baseline against that basket. When it rises to 114.8, the dollar has *appreciated* — it buys more euros, more yen, more pounds. When it falls to 90, the dollar has *depreciated*.

Two warnings a beginner must hear immediately. First, the DXY is *Europe-heavy and developed-market only* — it has no Chinese renminbi, no Mexican peso, no emerging-market currencies at all. So "DXY up" really means "dollar up against rich-world currencies," which is *correlated* with but not identical to "dollar up against everything." Second, the Federal Reserve's broad **trade-weighted dollar index** (and the **DXY**) can sometimes disagree at the margin. For our purposes the DXY is the standard, liquid, tradable proxy, and it captures the gravity we care about.

It also helps to have a feel for the *scale* of the moves you'll see. The DXY is a slow-moving index by the standards of single currency pairs — a 1% daily move is large, a 5% monthly move is a major event, and a 10%+ move over a few months (like 2022) is a once-a-decade dislocation. That sounds small until you remember it's a *weighted basket of six exchange rates*, each of which is itself enormous. The reason a "small" DXY move matters so much is leverage and breadth: it moves *every* dollar-priced asset at once, and the most dollar-sensitive assets (EM, crypto) move several times the DXY's percentage. A 5% DXY rally is a non-event for the dollar itself and a near-bear-market for emerging markets. Hold that asymmetry in mind: *the cause looks tiny, the consequences look huge.*

One more foundational distinction. There is a difference between the dollar's **level** (is the DXY at 90 or 110?) and its **rate of change** (is it rising or falling, and how fast?). Most cross-asset correlations are with the *change*, not the level. An asset doesn't care that the DXY *is* 105; it cares that the DXY *went from* 100 *to* 105. A dollar that is high but *stable* is much friendlier to risk assets than a dollar that is lower but *rising fast*. This is why traders watch the dollar's momentum — its trend and its rate of change — at least as closely as its absolute level. When this post says "a stronger dollar," read it as "a dollar that is rising," because that is what the correlations are built on.

### Why so much of the world is priced in dollars

Here is the fact that makes the dollar special, the reason it is *gravity* and not just one currency among many: an enormous share of global finance is *denominated* in dollars even when the United States isn't involved at all.

- **Commodities are priced in dollars.** Oil, copper, gold, wheat, coffee — when a Korean refiner buys Saudi crude, the invoice is in dollars. A barrel of oil has a dollar price first; its price in won or yen is just that dollar price converted.
- **Cross-border debt is largely in dollars.** A Brazilian company, a Turkish bank, an Indonesian government — vast amounts of borrowing by non-US entities is issued in dollars. The Bank for International Settlements tracks tens of trillions of dollars of such "offshore dollar" debt. Those borrowers owe dollars, earn local currency, and must find dollars to repay.
- **The dollar is the world's reserve currency and the settlement currency of trade.** As of late 2024, the dollar was still **57.8%** of disclosed global FX reserves — more than the euro (19.8%), yen (5.8%), pound (4.7%), and renminbi (2.2%) *combined*. It is the currency central banks hold, the currency trade is invoiced in, the currency the global banking system clears in.

This is the "exorbitant privilege," and it has a direct market consequence: because the dollar is the denominator, *a move in the dollar reprices everything in the numerator at once.* That is the entire engine behind the figure above and the rest of this post. (For the deep mechanism of *why* the dollar holds this role — the network effects, the petrodollar, the safe-asset franchise — read the dedicated [dollar system: why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) post; this post is about the measurable *correlations* that role produces.)

It is worth pausing on just how unusual this is, because it's the thing that makes the dollar *gravity* rather than just a big planet. Most assets are *correlated* with each other through shared exposure to the business cycle — stocks and corporate bonds both like growth, so they move together. But the dollar's correlation to other assets isn't a shared-exposure correlation; it's a *denominator* correlation. When you measure copper in dollars and the dollar moves, copper's measured price *must* move even if copper hasn't changed at all, the same way every distance on a map changes if you redraw the scale bar. No other asset has this property. The euro is one currency among many; the dollar is the *ruler everything else is measured with*. That is why a chart of the DXY tells you something about *every* market simultaneously, and why FX traders — who watch the ruler for a living — often see cross-asset moves coming before equity traders do.

There's also a feedback loop that makes the gravity self-reinforcing. When the dollar rises and EM borrowers are squeezed, they must sell assets to raise dollars to service debt; that selling pushes their currencies *down further*, which raises their debt burden *more*, which forces *more* selling. A rising dollar can become self-fulfilling — a "dollar doom loop" — until a central bank intervenes or the Fed eases. The same loop runs in reverse on the way down: a falling dollar eases debt burdens, frees up dollar liquidity, lets borrowers re-leverage, and feeds the risk-on rally. This reflexivity is why dollar trends, once established, tend to *persist* for months or years rather than mean-revert quickly — and why the dollar's correlation to risk assets is one of the most durable in all of macro.

## The mechanism: why a stronger dollar is a global tightening

Why should a higher dollar *pull other prices down*? There are two distinct channels, and a beginner should hold them separately.

### Channel 1: the arithmetic of the denominator

Take a commodity priced in dollars — say copper at \$4.00 per pound. Now suppose the dollar strengthens 5% against every other currency, but nothing changes about copper supply or demand. For a European buyer paying in euros, copper just got 5% *more expensive* in euro terms, because each euro now buys fewer dollars. Faced with a more expensive input, non-US buyers demand a little less. To clear the market, the *dollar* price of copper drifts *down*. So the same physical metal, with unchanged fundamentals, falls in dollar terms simply because the unit it's quoted in got stronger. This is pure arithmetic, and it's why oil, copper, and gold all carry a structural negative correlation to the DXY.

### Channel 2: dollar funding and global financial conditions

The second channel is deeper and explains why the dollar is *tightening* and not just a quote conversion. Because so much of the world *borrows* in dollars, the dollar's level *is* a global financial condition.

Picture a Turkish company that borrowed \$10 million and earns Turkish lira. When the dollar strengthens against the lira, that \$10 million debt is now worth *more lira* — the company's liability just grew without it borrowing a cent more. Multiply that across tens of trillions of offshore-dollar debt and you see the problem: **a rising dollar simultaneously raises the real debt burden of every dollar borrower outside the US.** It drains dollar liquidity, widens credit spreads, and forces deleveraging — selling assets to raise the now-scarcer dollars. That selling is *global*, and it hits the riskiest, most dollar-funded assets first: emerging markets, then commodities, then crypto.

This is why traders say "**the dollar is the world's margin call.**" A strong dollar is the financial-conditions equivalent of the Fed hiking rates for the entire planet at once. Conversely, a *weak* dollar floods the world with cheap dollar liquidity — the single biggest macro tailwind a risk asset can have. (For the policy mechanism that sets the dollar's level in the first place, see [how monetary policy moves currencies via rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials).)

A concrete picture of channel 2 helps. Outside the United States, banks and corporations hold dollar liabilities (debts they must repay in dollars) against dollar assets (loans, receivables, reserves). The whole edifice depends on being able to *get dollars when you need them* — to roll over short-term dollar funding, to meet a margin call, to pay an invoice. When the dollar strengthens, dollars become more valuable and therefore scarcer to those who don't earn them; the cost of borrowing dollars in the offshore market (visible in things like the cross-currency basis and FRA-OIS spreads) widens. That is a *funding squeeze*, and it behaves exactly like a tightening of monetary policy, except the Fed didn't do it — the foreign-exchange market did. The Fed recognized this so explicitly that during both 2008 and 2020 it opened **dollar swap lines** with foreign central banks, lending them dollars to relieve the squeeze. When the world's central bank has to airlift dollars to other central banks to stop a strong-dollar funding crisis, you know the dollar's gravity is real and dangerous.

The two channels usually reinforce each other, which is why the dollar's negative correlation to risk is so reliable. A US-led growth boom (channel: rate differential) raises the dollar, which simultaneously squeezes foreign borrowers (channel: funding). A global panic (haven bid) raises the dollar, which simultaneously squeezes foreign borrowers. Almost every reason for the dollar to rise *also* tightens conditions for the rest of the world. There is no clean way for the dollar to rip higher without something else, somewhere, breaking.

Put the two channels together and you get the central correlation chart of this post.

![Bar chart of the correlation of a rising dollar with each asset](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-2.png)

Read that chart as a hierarchy of *how dollar-sensitive* each asset is. EM equities and gold sit at the bottom (−0.55), copper and oil next (−0.50, −0.45), Bitcoin in the middle (−0.35), and the S&P 500 only weakly negative (−0.20). The lone green bar — the US 10-year yield at +0.40 — is the *cause*, not a victim, and we'll come back to it.

#### Worked example: a dollar move into a commodity move

Suppose you hold a position in copper, and your macro view is that the DXY is about to rally from 100 to 105 — a **+5%** dollar move — with no change in copper fundamentals. The long-run correlation of copper to the dollar is about −0.50, and historically copper's *beta* to the dollar (the percentage move in copper per 1% move in the dollar) runs around −2: a 1% stronger dollar tends to knock roughly 2% off copper.

So a +5% DXY move implies copper falls about **5% × −2 = −10%.** On a \$4.00 copper price, that's a drop to about **\$3.60**. If you held \$50,000 of copper exposure, the dollar move alone — *before any copper-specific news* — would cost you about **\$5,000.** The intuition: when you're long a dollar-priced commodity, you are *implicitly short the dollar*, whether you meant to be or not.

## The driver: rate differentials and the lone positive correlation

Everything so far explains why the dollar *pulls other assets down*. But what pulls the *dollar* up? The single most important driver is the **rate differential** — the gap between US interest rates and those of the rest of the world.

### Money flows to where it earns the most

A currency is, among other things, a claim on a stream of interest. If US two-year Treasuries yield 5% and German two-year bunds yield 2%, a global investor can earn 3 percentage points more, in a safe asset, simply by holding dollars instead of euros (ignoring the cost of hedging the currency, which matters but doesn't kill the basic pull). That extra yield draws capital *into* dollars — investors sell euros, buy dollars, buy Treasuries — and that buying *bids the dollar up*. So:

- **US yields rise faster than foreign yields → dollar strengthens.**
- **US yields fall faster (Fed cutting while others hold) → dollar weakens.**

This is why the US 10-year yield is *positively* correlated with the DXY (+0.40 in our data): the same force — higher US rates relative to the world — pushes both up together. It's also why the **front end** of the curve (the 2-year, which tracks expected Fed policy) is so tightly linked to the dollar; see [the Fed funds path and the front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) for the policy-expectations leg, and [bond yields: the master correlation](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset) for how the same yield move radiates to every other asset.

We can see what actually moves the dollar by flipping the macro correlation matrix around and reading down the dollar's column.

![Bar chart of what drives the dollar from the correlation matrix](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-5.png)

The drivers line up exactly with intuition: a **hot CPI surprise** (+0.45) lifts the dollar because it implies more Fed hiking; a **rising 10-year yield** (+0.45) and a **rising real yield** (+0.40) lift it for the same reason; a **wider credit spread** (+0.30) lifts it because stress is risk-off and dollar-positive. The lone negative driver is **higher oil** (−0.40) — when energy prices spike, the US (a large energy producer) is hurt less than energy-importing Europe and Japan, but the cleaner reading is that an oil-driven terms-of-trade shock often hits the euro and yen harder, which is the same as the dollar rising; in the 1970s, oil and the dollar moved together, so this is one of the more regime-dependent links in the matrix.

#### Worked example: a rate differential into a dollar move

Suppose the Fed signals two more hikes while the European Central Bank signals it is *done*. The US two-year yield rises from 4.5% to 5.0% while the German two-year stays at 2.5%. The rate differential just widened by **0.5 percentage points**, from 2.0pp to 2.5pp.

A common rule of thumb on FX desks is that EUR/USD moves roughly 1% for every ~0.10pp shift in the two-year rate differential over short horizons. A +0.5pp widening therefore points to roughly a **−5% EUR/USD** move (euro weaker, i.e., dollar stronger). Since the euro is ~58% of the DXY, a −5% EUR/USD translates to roughly **+3% on the DXY** (0.58 × 5%). If you were positioned for a stable dollar with \$100,000 in EM equities (correlation to the dollar ~−0.55, beta near −1.2), that +3% dollar move alone would imply roughly **−3.6%** on your EM book, about **−\$3,600**, purely from the rate-differential shift. The intuition: the dollar is the rate differential wearing a costume.

There is a subtlety worth flagging: the dollar responds to *relative* policy, not absolute policy. The Fed can hike and the dollar can still *fall* — if other central banks are hiking *faster*, or if the market had already priced in even more Fed hiking than it delivered. In 2017, the Fed raised rates three times and the DXY *fell* about 10%, because the rest of the world's growth accelerated and the ECB began signaling the end of its own easing, narrowing the differential from the other side. This is one of the most common ways beginners get the dollar wrong: they watch the Fed in isolation. The dollar is always a *relative* trade. Always ask "compared to whom?"

#### Worked example: why a hot CPI print lifts the dollar

Suppose US core CPI comes in at +0.5% for the month versus a +0.3% expectation — a +0.2pp upside surprise. From our event-study betas, a +0.1pp core-CPI surprise moves the DXY about +0.35% in the inflation-fear regime, so a +0.2pp surprise points to roughly **+0.7% on the DXY** on the day. The mechanism: hot inflation means the market prices in *more Fed hikes*, which widens the expected rate differential, which pulls capital into dollars. If you held \$30,000 of gold (dollar beta near −1.5), that single CPI print would imply roughly **−1.0% on gold**, about **−\$300**, in the minutes after the release — not because inflation is bad for gold (it "should" be good), but because the *dollar and real-yield* response to the inflation dominates. The intuition: an inflation surprise reaches gold *through* the dollar and real yields, and that path is negative.

## The dollar smile: why USD rises in both booms and busts

There is one more piece of the mechanism that confuses beginners, and it is the most important nuance in the whole post. If a *strong economy* draws capital into dollars (boom), but a *crisis* also drives money into dollars (bust), then when does the dollar *fall*? The answer is captured in one of the most useful pictures in macro: the **dollar smile.**

![The dollar smile curve showing USD strong at both ends](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-4.png)

The horizontal axis is global growth, weak on the left and strong on the right. The vertical axis is the dollar's level, strong at the top. The curve looks like a smile because the dollar is *strong at both ends and weak in the middle*:

- **Left side — the risk-off bust.** When the world is scared (a financial crisis, a war shock, a pandemic), investors flee to safety. The deepest, most liquid safe asset on earth is the US Treasury, bought with dollars. So in a panic, money rushes *into* dollars regardless of US growth. The dollar is the global haven. (This is why the dollar *rose* in March 2020 even as the US economy was collapsing.)
- **Right side — the US-outperformance boom.** When the US economy is booming *faster than the rest of the world*, the Fed is hawkish, US yields are high, and the rate-differential magnet pulls capital into dollars. The dollar is strong because America is winning. (This is 2022 and much of 2024.)
- **The middle — calm, synchronized global growth.** The dollar is *weakest* when the world is growing together, calmly, and the US is *not* uniquely outperforming. Capital fans out to higher-returning markets abroad; risk appetite is healthy; dollar liquidity floods the world. This is the global risk-on regime — and it's the best environment for commodities, EM, and gold.

The dollar smile is why a single "dollar up = risk off" rule isn't quite enough. The dollar can rise for a *good* reason (US boom) or a *bad* one (global panic), and the cross-asset consequences differ in the details — but in *both* cases the dollar is rising and the rest of the world is being squeezed. The smile is the reason the dollar's negative correlation to risk assets is so robust: there are two independent reasons for it to rally, and almost only one (calm global synchronization) for it to fall.

You can map decades of dollar history onto the smile. The 2008 financial crisis was a textbook *left-side* rally: as Lehman fell and the world deleveraged, the DXY surged from about 71 to 88 *even though the crisis began in America*, because the panic drove a global scramble for dollars. The 2014–2015 period was a *right-side* rally: the US recovery outpaced Europe and Japan, the Fed prepared to hike while the ECB launched QE, and the DXY rose from about 80 to 100 — crushing commodities (oil fell from \$100 to \$30) and triggering an EM and Chinese-currency scare. The mid-2017 to early-2018 stretch was the *middle* of the smile: synchronized global growth, a soft dollar, and a roaring rally in EM, commodities, and crypto (Bitcoin's run to \$20,000 happened in exactly this soft-dollar window). Once you see the smile, every major dollar regime of the last twenty years slots cleanly into one of its three zones — and each zone came with a predictable cross-asset signature.

The practical upshot: the smile tells you that "the dollar is rising" is *necessary but not sufficient* for a forecast. You also need to know *which side* you're on, because the left side (panic) means even US stocks and gold can fall as everything is sold for cash, while the right side (US boom) means US large-caps can hold up fine while only non-US assets suffer. The cross-asset *breadth* of the damage is wider on the left side of the smile than the right.

#### Worked example: reading the smile in a panic

In March 2020, US growth was *collapsing* — you would naively expect the dollar to fall. Instead the DXY spiked from about 95 to **103** in under two weeks, a roughly **+8%** move, as the world scrambled for dollars to meet margin calls and hoard the safe asset. Gold, which "should" rally in a crisis, initially *fell* about 12% during that scramble, because it too is priced in dollars and was being sold to raise cash. An investor holding \$20,000 of gold who expected a crisis hedge instead watched it drop to about **\$17,600** in the first leg — *because the dollar, not gold, was the true haven that week.* The intuition: in a real liquidity panic, the only thing that goes up is the dollar, and even gold bows to it before recovering.

## The mechanism per asset: same gravity, different channels

The dollar pulls on every asset, but through a different channel and at a different strength. This is the heart of "cross-asset gravity," and it's worth walking asset by asset.

![Matrix of why each asset is negatively correlated with the dollar](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-7.png)

- **Emerging markets (−0.55, the strongest).** EM gets hit through *both* channels at maximum strength. EM commodities and exports are dollar-priced (channel 1), and EM borrowers are the most dollar-indebted on earth (channel 2). A rising dollar raises their debt burden, drains their dollar funding, and weakens their currencies — which then *imports inflation* and forces their central banks to hike, slowing their economies. This is the textbook "EM crisis" transmission, and it's why a strong dollar is feared in São Paulo and Jakarta more than anywhere.
- **Gold (−0.55).** Gold is the dollar's *rival as money*. It yields nothing, so its appeal rises when the dollar's appeal (and real yields) fall, and vice versa. Gold is also dollar-priced, so channel 1 applies. The cleanest way to think of gold is as the *anti-dollar*: it goes up when faith in the dollar (and in fiat money generally) goes down. (For the deeper story — gold tracks *real yields* more than inflation — see [inflation and gold: the real yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).)
- **Copper and oil (−0.50, −0.45).** Pure channel-1 commodities, with a twist: copper also has a strong *growth* signal of its own, so its dollar correlation is partly offset by the fact that a US-boom dollar (right side of the smile) coincides with strong demand. Oil is the most regime-dependent — in an oil-supply-shock regime, oil and the dollar can rise together.
- **Bitcoin (−0.35).** Crypto trades as a **global risk-and-liquidity** asset. A strong dollar means tighter global liquidity, and the highest-beta, most speculative assets shed value first. Bitcoin's dollar correlation rose sharply when it began trading as a macro-liquidity instrument around 2020–2022. (See [crypto as a macro asset: the liquidity correlation](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation) for the full liquidity story.)
- **The S&P 500 (−0.20, the weakest).** Here is the surprise for many beginners: large-cap US stocks are only *weakly* dollar-sensitive. The reason is that the S&P 500's earnings are *mostly domestic* — most US companies sell to US customers in dollars, so a strong dollar barely touches their core business. The negative link that *does* exist comes from the roughly 40% of S&P revenue earned abroad: a strong dollar shrinks the dollar value of those foreign earnings (a German sale converts to fewer dollars) and makes US exports pricier. So the S&P's dollar beta is real but modest, concentrated in multinationals and exporters — not the whole index.

The within-index dispersion is itself instructive. A strong dollar is a *headwind* for the big technology and consumer-staples multinationals that earn half their revenue overseas (think a software giant or a soft-drinks company), and roughly *neutral* for domestic-focused names — regional banks, homebuilders, US-only retailers, utilities. So even inside the S&P, "the dollar" sorts winners from losers. This is why a savvy equity manager who turns bullish on the dollar will tilt *toward* domestic small-caps and *away* from large-cap exporters, capturing the dollar effect *within* the equity allocation rather than betting on the whole market. The aggregate index correlation of −0.20 is the *average* of a meaningfully negative multinational sleeve and a roughly flat domestic sleeve.

A useful way to rank all of this is by *where each asset sits on the funding-vs-pricing spectrum*. Assets that are heavily *dollar-funded* (EM, crypto) get hit through the funding channel, which is the violent one — it shows up as deleveraging and gaps. Assets that are merely *dollar-priced* (oil, copper, gold) get hit through the arithmetic channel, which is steadier and more mechanical. And assets that are mostly *dollar-earning* (US domestic stocks) barely move at all. The further left on that spectrum (most funded), the stronger and more dangerous the negative correlation. That single ordering — funded, then priced, then earned — reproduces almost the entire ranking in the correlation bar chart.

## Lead, lag, and when the gravity breaks

Two questions remain before we can use the dollar correlation honestly: does the dollar *lead* other assets (so it gives an early warning), and *when does the relationship break*?

### Does the dollar lead?

On the funding channel, the dollar often *leads* the assets most sensitive to it. Because the dollar reprices global funding conditions in real time and the FX market is the largest, fastest, most liquid market on earth (trillions of dollars a day), a dollar move frequently shows up *before* the slower-to-react equity and credit markets fully digest it. EM equities, EM credit spreads, and high-beta commodities tend to follow the dollar with a short lag of days to a couple of weeks. This is why dollar-watchers often feel like they can see cross-asset stress coming: a sharp, persistent dollar rally is an early-warning siren for EM and risk, days before the headlines catch up. It is not a precise timing tool — the lead is noisy and varies by regime — but as a *directional* heads-up, a breaking dollar trend is one of the better leading signals in macro. (For how leading, coincident, and lagging indicators are formally classified, see [lead-lag: leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).)

### When the gravity breaks: the decoupling

The dollar's negative correlation to gold and commodities is robust but not eternal, and the most instructive recent break is **gold's 2023–2024 decoupling**. For roughly 2007 through 2021, gold's correlation to real yields was about −0.82 and its inverse link to the dollar was textbook clean — exactly what the "anti-dollar" story predicts. Then in 2022–2025 the relationship *broke*: gold climbed from about \$1,800 to over \$2,600 *even as* real yields rose to +2% and the dollar stayed firm. The old model said gold should have fallen hundreds of dollars; instead it rallied to record highs.

What happened? A *new, stronger driver* overpowered the dollar: relentless **central-bank gold buying**, accelerated after Western sanctions froze Russia's dollar reserves in 2022 and pushed many emerging-market central banks to diversify *out of dollars and into gold* as a sanctions-proof reserve. That structural demand was large enough to swamp the usual real-yield-and-dollar drag. The correlation didn't *reverse* permanently — it was *temporarily overpowered* by an idiosyncratic flow. This is the single most important lesson about every correlation in this series: a relationship breaks not because the old mechanism stopped existing, but because a *bigger* mechanism showed up. When the dollar and an asset stop moving opposite each other, don't conclude "the dollar doesn't matter anymore" — conclude "something larger is now driving this asset, and I need to find out what." The decoupling is *information*, not noise.

### Rolling windows hide the regimes

A final measurement caution. The headline "DXY-gold correlation = −0.55" is a *full-sample average* that smears together −0.8 regimes, near-zero regimes, and brief positive regimes into one misleading number. The honest way to study the dollar's correlation is with a *rolling window* — a 90-day or 24-month correlation recomputed continuously — which reveals the relationship breathing in and out across regimes. A trader who relies on the static full-sample number will be blindsided every time the regime shifts. (For why the window length itself changes the answer, see [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).)

## Common misconceptions

A handful of myths recur often enough that correcting them is the fastest way to actually *use* the dollar correlation.

**Myth 1: "A strong dollar is bad for the stock market."** Mostly false, and dangerously oversimplified. The S&P 500's correlation to the dollar is only about **−0.20** — weak. US large-caps are mostly domestic earners; a strong dollar barely dents them. What a strong dollar *is* bad for is *everything outside the US*: EM, commodities, gold, and the foreign-earnings slice of the index. People who shorted US stocks on "strong dollar" alone in 2024 (DXY closed at 108.5) missed a year where US equities did fine. The dollar is gravity for the *world*, not specifically for the S&P.

**Myth 2: "Gold is an inflation hedge, so a strong dollar (which fights inflation) should help gold."** False on both counts. Gold's primary correlation is to **real yields** and to the **dollar**, not to headline inflation. A strong dollar usually *accompanies* high real yields (the Fed is hiking), and both are *negative* for gold. In 2022, headline US CPI hit a 40-year high of 9.06% — and gold finished the year roughly *flat* (−0.3%), because the dollar surged and real yields went from −1% to +1.7%. Inflation was raging; gold did nothing. The dollar and real yields, not the CPI print, ruled gold.

**Myth 3: "The dollar and US yields should move opposite each other, like bonds and the dollar."** False — they're *positively* correlated (+0.40). It's natural to assume the dollar acts like a bond price (which falls when yields rise), but the dollar is the *opposite*: higher US yields *attract* foreign capital into dollars, so the dollar *rises* with US yields. Confusing these is one of the most common beginner errors. The dollar follows the *yield differential*, not the yield level alone.

**Myth 4: "A weak dollar is always good for risk assets."** Mostly true but with a critical exception. A weak dollar is usually a global risk-on tailwind — except when the dollar is weak because the *US itself* is in trouble (a US-specific crisis, a debt-ceiling scare, a loss of confidence in US policy). In that case a falling dollar can coincide with falling US assets, because the *cause* is bad. Always ask *why* the dollar is moving (the smile: boom, bust, or calm) before assuming the cross-asset sign.

**Myth 5: "The DXY is the dollar against everything."** False — it's the dollar against six *developed-market* currencies, 58% euro. The dollar can be flat against the DXY while ripping against EM currencies (the peso, the lira, the rand). For EM-focused traders, the DXY is a rough proxy; the more precise gauge is a broad trade-weighted or EM-currency index. Don't assume "DXY flat" means "no dollar pressure on EM."

**Myth 6: "If the dollar's correlation to gold is −0.55, then when the dollar rises 10% gold falls 5.5%."** False — this conflates *correlation* with *beta*. Correlation (−0.55) measures *how consistently* two things move opposite each other; beta measures *how much* one moves per unit of the other. Gold's correlation to the dollar is moderate (−0.55) but its *beta* is larger (around −1.5), so a 10% dollar rally points to roughly a −15% gold move *on average*, not −5.5%. A moderate correlation with a high beta means "they don't *always* move opposite, but when they do, gold moves a lot." Never plug a correlation number into a P&L calculation as if it were a beta. (For the precise difference between correlation, beta, and the rank-based Spearman measure, see [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta).)

## How it shows up in real markets

The cleanest way to feel the dollar's gravity is to watch what happens to *everything else* when the DXY makes a big move. We have three textbook episodes.

### 2022: the dollar surge that crushed everything

2022 is the purest demonstration of cross-asset gravity in modern history. The Fed hiked from near-zero to 4.5% in nine months — the fastest tightening in 40 years — while the ECB and Bank of Japan lagged badly. The rate differential blew out, and the DXY rose about **28%** from its 2020 low to a 20-year peak of **114.8** in September.

![DXY level 2014 to 2025 with the 2022 peak marked](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-3.png)

Now look at what the dollar's gravity did to the rest of the board that year.

![2022 total returns by asset showing the dollar as the only gainer](/imgs/blogs/the-dollar-dxy-cross-asset-correlation-6.png)

The dollar (+8.2% on the full-year DXY return) was the *only* major asset class that went up. Bitcoin fell 64%, the Nasdaq 100 fell 33%, long Treasuries fell 31%, the S&P fell 18%, and even the classic 60/40 portfolio fell 16% as stocks and bonds dropped together. Gold was the great escape, finishing roughly flat (−0.3%) — not because it "hedged inflation," but because its anti-dollar character partly offset the rising-real-yield drag. This is the year the phrase "the dollar is the world's margin call" stopped being a metaphor.

#### Worked example: sizing the 2022 dollar drag on an EM portfolio

Suppose at the start of 2022 you held a **\$100,000** emerging-market equity portfolio and the DXY rose 8.2% on the year. EM equity's correlation to the dollar is about −0.55, and its empirical beta to the DXY runs near −1.3 (EM is more dollar-sensitive than 1-for-1 because of the debt channel). The dollar move *alone* implies an EM drag of roughly **8.2% × −1.3 ≈ −10.7%**, or about **−\$10,700**, before any EM-specific earnings or local-rate news. EM equities actually fell about 20% in 2022 — so *more than half* of the loss was attributable to the dollar's gravity, not to anything happening inside those companies. The intuition: when you buy EM, you are taking a large, often unhedged, short-dollar bet whether you realize it or not.

### 2020: the dollar collapse and the everything-rally

The mirror image came in the second half of 2020. After the March panic spike, the Fed slashed rates to zero, launched unlimited QE, and flooded the world with dollar liquidity. The DXY fell from 103 to about **90** by year-end — and *everything* rallied. Stocks roared back to record highs, gold hit \$2,070, copper doubled off its lows, EM equities surged, and Bitcoin began the run from \$10,000 to \$69,000. A falling dollar is the single most powerful risk-on signal there is, because it loosens financial conditions for the entire planet at once. The everything-rally of late 2020 and 2021 was, at its root, a weak-dollar rally.

The lesson of placing 2020 and 2022 side by side is the most important takeaway in this post. These were two of the most extreme cross-asset years in living memory, and they were *mirror images* driven by the *same variable*. In 2020 the dollar fell 13% from its peak and almost every risk asset on earth soared; in 2022 the dollar rose 28% to a two-decade high and almost every risk asset on earth was crushed. If you had watched nothing but the DXY across those two years — not a single earnings report, not a single growth forecast — you would have correctly called the direction of commodities, EM, gold, and crypto in *both* years. That is what "cross-asset gravity" means in practice: one variable, read correctly, anticipates the *sign* of returns across half the asset universe.

#### Worked example: the weak-dollar tailwind on a commodity basket

Suppose at the June 2020 dollar peak you bought a \$40,000 basket of commodities and EM equities, and over the next twelve months the DXY fell about 12% (from ~99 to ~90). With a blended dollar beta near −1.4 for that basket, the dollar move *alone* implies a tailwind of roughly **12% × 1.4 ≈ +16.8%**, or about **+\$6,700**, before any commodity-specific or EM-specific gains. Copper roughly *doubled* over that window and EM equities rose ~40%, so the realized return dwarfed the \$6,700 — but a large, reliable chunk of it was simply the dollar deflating. The intuition: in a falling-dollar regime, being long anything dollar-priced gives you a tailwind for free; in a rising-dollar regime, the same positions fight a headwind every day.

### The yen carry and the dollar-smile episodes

The dollar smile shows up vividly in the yen. From 2021 to 2024, the Fed hiked aggressively while the Bank of Japan held rates near zero — the widest major rate differential in decades. The result: USD/JPY rocketed from **115** at the end of 2021 to an intraday peak of nearly **162** in July 2024, the weakest yen since 1986. This was the right side of the smile (US outperformance, huge rate differential) and it powered the "yen carry trade" — borrow cheap yen, buy higher-yielding dollar assets. When that trade violently unwound in early August 2024 (the BoJ hiked, US data wobbled), the *left* side of the smile briefly fired: a global risk-off spasm, a snap-back in the yen, and a sharp equity drawdown — all in a few days. The same currency pair traversed both sides of the smile within three years.

The yen carry unwind of August 2024 is also a perfect miniature of dollar gravity's *reflexivity*. The carry trade had quietly become enormous — leveraged positions funded in cheap yen, parked in higher-yielding dollar and global assets. When the yen suddenly strengthened, those positions became unprofitable, forcing traders to *buy back yen to close them*, which strengthened the yen *further*, forcing *more* unwinding. In two trading sessions the Nikkei fell over 12%, US tech sold off sharply, and crypto dropped double digits — a global deleveraging spasm triggered by a *currency funding* move, not by any change in earnings or growth. It was channel 2 (dollar/yen funding) firing in fast-forward, and it shows how a currency move can be the *cause* of a cross-asset selloff rather than a symptom. For anyone who still thinks of FX as a sleepy backwater detached from "real" markets, August 2024 was the rebuttal: the funding currency *is* the market's leverage dial.

### Crypto and the global-liquidity correlation up close

Bitcoin's relationship with the dollar deserves a closer look because it is the newest and most regime-dependent. Before 2020, Bitcoin's correlation to anything macro was essentially zero — it traded on its own crypto-native cycle. Then, as institutions arrived and crypto began trading around the clock as the highest-beta expression of global liquidity, its correlation to the dollar (and to the Nasdaq, and to the *negative* of real yields) climbed sharply, peaking in 2022 when Bitcoin behaved like a leveraged risk asset, falling 64% as the dollar surged and liquidity drained. By 2024–2025 that correlation faded back toward the low-0.3s as crypto-native catalysts (the Bitcoin ETF launch, the halving) reasserted themselves alongside the macro driver. The takeaway: crypto's dollar correlation is *real but unstable* — strongest precisely when macro liquidity is the dominant story (panics and tightening cycles) and weakest when crypto-specific news takes over. Treat Bitcoin as a *high-beta dollar-liquidity gauge* that occasionally goes off and does its own thing.

## How to read it and use it

Here is the payoff: how to actually *use* the dollar as a single dial for global financial conditions.

**The core signal.** Treat the DXY as your master gauge of global financial conditions. *DXY rising = tightening = headwind for commodities, gold, EM, and crypto.* *DXY falling = easing = tailwind for the same.* If you can only watch one chart to gauge global risk appetite, the dollar is a strong candidate — it summarizes the rate differential, the safe-haven bid, and global liquidity in one line.

**The regime check (use the smile).** Before you act on a dollar move, ask *why* it's moving:
- Dollar up because the **US is booming** (right side of the smile, high US yields, hawkish Fed) → classic risk-headwind for non-US assets, but US large-caps can still do fine.
- Dollar up because the **world is panicking** (left side, flight to safety) → broad risk-off; even US stocks fall, and *cash and Treasuries* are the only refuge.
- Dollar down in **calm, synchronized growth** (the middle) → the best regime for commodities, EM, and gold; lean risk-on.
- Dollar down because the **US itself is in trouble** → the exception; a falling dollar with falling US assets means the *cause* is bad, not good.

**Position sizing the hidden short-dollar bet.** Recognize that whenever you are long a dollar-priced or dollar-funded asset — a commodity, gold, an EM stock, Bitcoin — you are *implicitly short the dollar.* A naive "diversified" book of commodities + EM + gold + crypto is, in dollar terms, one giant short-dollar position that will all move together when the DXY moves. That's why these "diversifiers" all fell at once in 2022. True diversification against dollar gravity means holding the dollar itself (cash, short-dated Treasuries) or US domestic earners.

#### Worked example: the hidden dollar exposure in a "diversified" book

Suppose you build what feels like a well-diversified \$200,000 portfolio: \$50,000 each in gold, a commodity index, EM equities, and Bitcoin. It *looks* spread across four asset classes. But measure its dollar exposure: gold beta ≈ −1.5, commodities ≈ −1.5, EM ≈ −1.3, Bitcoin ≈ −2.5. The portfolio-weighted dollar beta is about **(−1.5 − 1.5 − 1.3 − 2.5) / 4 ≈ −1.7**. So a +5% DXY rally implies roughly **5% × −1.7 ≈ −8.5%** across the whole book, about **−\$17,000**, *all at once*, because all four "diversifiers" are really the same short-dollar trade in different clothes. The fix isn't more "asset classes"; it's adding something with *positive* dollar beta — cash, short Treasuries, or a long-dollar position — to actually offset the gravity. The intuition: diversification across assets that all share one hidden factor isn't diversification at all.

**Using the dollar to hedge.** The flip side of the hidden-exposure problem is that the dollar is an excellent *hedge*. If you run a portfolio of EM, commodities, and crypto and you can't or don't want to reduce it, a *long-dollar* overlay (long DXY futures, or simply holding more cash/short Treasuries) will rise precisely when your risk book falls, because the gravity that hurts your assets *helps* the dollar. This is why some macro funds treat a long-dollar position as portfolio insurance: it's a hedge that *pays you in a crisis* (the left side of the smile) and *also* in a US-boom selloff (the right side). Few hedges work on both ends of the growth spectrum; the dollar does, which is exactly the smile in portfolio form.

**What invalidates the signal.** The dollar's negative correlation is robust but not eternal. It weakens or flips when: (1) an asset has a *stronger* idiosyncratic driver overpowering the dollar (gold's central-bank-buying decoupling in 2023–24, when gold rose despite high real yields and a firm dollar); (2) oil-supply shocks make oil and the dollar rise together; (3) a US-specific crisis makes a falling dollar bad rather than good; or (4) you're measuring against the wrong basket (DXY flat but EM currencies cratering). Always pair the DXY with the *reason* and with the right currency benchmark for the asset you care about.

**The honest caveat.** Like every relationship in this series, the dollar's cross-asset correlations are *regimes, not constants.* The −0.55 to gold averaged across decades hides years of −0.8 and years of +0.3. Use the dollar as a powerful first-order map of global financial conditions, not as a mechanical formula — and when the dollar and an asset stop moving opposite each other, treat the decoupling as *information* (something idiosyncratic is now driving that asset), not as a broken rule.

**Pair the DXY with the right benchmark.** The last operational rule is to match the dollar gauge to the asset you care about. If you trade developed-market equities and commodities, the DXY is a fine proxy. If you trade emerging markets specifically, watch a broad or EM-weighted dollar index too, because the DXY can be lulled to sleep by a stable euro while the dollar is quietly grinding EM currencies into the dirt. And if you trade a single commodity with a strong supply-side story (oil during a war, copper during a mine strike), remember that the asset's *own* driver can temporarily swamp the dollar entirely — the gravity is always there, but a powerful enough rocket can fight it for a while. The discipline is to *decompose* every move: how much was the asset, and how much was the dollar deflating or inflating underneath it? Once you train your eye to subtract the dollar first, the rest of the move is the part that actually tells you something about the asset.

**A simple weekly routine.** You don't need a trading desk to use any of this. Once a week, look at three things: (1) *the DXY level and its trend* — is it rising, falling, or flat over the last month? (2) *the US-vs-rest rate differential* — is the Fed more or less hawkish than the ECB and BoJ, and is that gap widening or narrowing? (3) *which side of the smile you're on* — is the dollar moving because the US is booming, because the world is panicking, or because growth is calm and synchronized? Those three reads, together, give you a one-line global-conditions thesis: "dollar rising on US outperformance → headwind for EM and commodities, US large-caps okay," or "dollar falling on synchronized growth → risk-on tailwind, lean into commodities and EM." That single sentence will explain more cross-asset moves than a stack of individual stock forecasts.

The dollar is gravity. It is invisible until something big moves, and then you realize it was bending every price on the board the whole time. It is the denominator that reprices the numerator, the funding currency that becomes the world's margin call, the haven that rallies in a panic and the yield magnet that rallies in a boom. Its correlation to commodities, gold, EM, and crypto is one of the most durable in all of macro — not because of a shared business cycle, but because the dollar is the *ruler everything else is measured with*. Learn to read the DXY, ask the smile *why*, never confuse its correlation for its beta, and treat every decoupling as a clue rather than a contradiction — and you hold the single most powerful cross-asset dial in macro.

## Further reading and cross-links

- [The Fed funds path and the front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) — the policy-expectations engine behind the rate differential that drives the dollar.
- [Bond yields: the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset) — how the same yield move that lifts the dollar radiates to every other asset.
- [Crypto as a macro asset: the liquidity correlation](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation) — why Bitcoin trades as a dollar-liquidity instrument.
- [Oil prices, CPI, and the energy-equity correlation](/blog/trading/macro-correlations/oil-prices-cpi-and-the-energy-equity-correlation) — the dollar-priced commodity that sometimes rises *with* the dollar.
- [The macro asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full grid this post drew the dollar's row and column from.
- [Dollar system: why USD rules markets (DXY)](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the deep mechanism behind the dollar's reserve-currency role.
- [How monetary policy moves currencies via rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — the policy mechanism that sets the dollar's level.
- [The dollar: cross-asset gravity (allocator's lens)](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) — the portfolio-construction view of the same gravity.

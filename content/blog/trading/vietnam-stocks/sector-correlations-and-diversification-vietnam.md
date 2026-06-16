---
title: "Sector Correlations: What Really Diversifies a Vietnamese Portfolio"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero deep dive into correlation and diversification for VN-Index investors — why a basket of banks, property, brokers, steel and construction is one credit bet not five, why correlations spike to one in a crisis, and how to build a book that is actually diversified with defensives, structural growth and cash."
tags: ["vietnam-stocks", "sector-rotation", "vn-index", "diversification", "correlation", "portfolio-construction", "risk-management", "credit-cycle", "defensive-sectors", "position-sizing", "2022-bond-crisis"]
category: "trading"
subcategory: "Vietnam Stocks"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Owning five stocks is not diversification if they are all in the credit chain. On VN-Index the heavy cyclicals rise and fall together, so a "diversified" book of them is really one big macro bet.
>
> - The chain sectors — banks, real estate, securities brokers, steel, construction — are **highly correlated** (pairwise roughly 0.65 to 0.82) because they all run on one shared input: the credit cycle. Owning all five is owning the same trade five times.
> - **Real diversification** comes from adding sectors with low correlation to the chain: defensives (utilities, staples, pharma) around 0.30, structural growth (IT) around 0.45, plus cash, which is the only thing that is correlation zero with everything.
> - Correlations are **not stable**. In a credit shock they all rush toward one: positions that looked uncorrelated at 0.30 in calm markets jumped to 0.85 in the 2022 selloff, exactly when you needed the diversification most.
> - The one number to remember: a six-stock book of pure chain names has roughly **1.5 effective independent bets**, not six. A six-stock book spread across chain, defensives, growth and cash has roughly **4**. Count your real bets, not your tickers.

In the second half of 2022, a Vietnamese retail investor we will call Minh held what he proudly described as a "diversified" portfolio. He owned five large, liquid, blue-chip stocks: a big private bank, a top property developer, the largest brokerage, a leading steelmaker, and a major construction contractor. Five different companies, five different industries, five different tickers. On paper it looked like a textbook spread — no single name was more than a fifth of the book, and he had deliberately avoided putting everything into one stock. He felt safe.

Then the corporate-bond crisis hit. Over roughly twelve months from the early-2022 peak, his bank fell about 40%, his developer fell about 48%, his broker fell about 58%, his steelmaker fell about 52%, and his construction name fell about 50%. They did not fall on different days for different reasons. They fell *together*, week after week, as one. The "diversification" he had built evaporated precisely when he needed it. His five-stock book behaved as if it were a single, highly leveraged bet on one thing — and it was. It was a bet on the Vietnamese credit cycle, dressed up in five different costumes.

This post is about the single most misunderstood idea in portfolio construction: **diversification is not about how many stocks you own — it is about how independently they move.** Minh did not own five bets. He owned roughly one. By the end of this deep dive you will understand what correlation actually measures, why the chain sectors are welded together by a shared credit driver, why spreading across "different" sectors can be a complete illusion, what genuinely lowers the risk of a Vietnamese book, why correlations betray you in a crisis by all rushing to one, and how to count the number of *real* independent bets you are running. Then you will be able to build a book that is diversified in fact, not just in appearance.

![Flow diagram showing five chain sectors all driven by one credit input collapsing into one bet](/imgs/blogs/sector-correlations-and-diversification-vietnam-1.png)

## Foundations: what correlation actually means

Before we can talk about diversification, we have to nail down the one concept the whole subject rests on: **correlation**. It sounds technical, but the idea is something you already understand intuitively, and we will build it from zero with no math background assumed.

### Do two things move together?

**Correlation** is a single number, between minus one and plus one, that answers one question: *when one thing moves, does the other tend to move the same way, the opposite way, or unrelatedly?*

- **Correlation of +1** (perfect positive) means the two move in lockstep. Every time A goes up, B goes up; every time A goes down, B goes down, in proportion. Two share classes of the same company are near +1. A bank and a second bank in the same country are close to it.
- **Correlation of 0** (uncorrelated) means knowing what A did tells you *nothing* about what B did. Sometimes they move together, sometimes opposite, with no reliable pattern. Cash sitting in a deposit and the price of a steel stock are roughly zero — the steel stock swings around while the cash just earns its interest, indifferent.
- **Correlation of minus one** (perfect negative) means they move exactly opposite. When A goes up, B goes down. True minus-one pairs are rare in stocks; the closest real-world cousins are a long position and a short hedge of the same thing.

For Vietnamese stocks, almost everything sits between **0 and +1**, because every stock is partly riding the same broad market tide — when VN-Index has a great day, most stocks are green; when it craters, most are red. The interesting question is not *whether* two sectors are positively correlated (almost all are), but *how strongly*. A pair at 0.30 still has a lot of independent movement; a pair at 0.80 barely has any.

### The everyday intuition: umbrellas and sunscreen

Here is the mental model. Suppose you run a little beach kiosk and you can stock two products. If you stock two brands of sunscreen, your sales of both rise on sunny days and fall on rainy days — they are highly correlated, near +1. Stocking the second brand barely smooths your income, because both products are betting on the same thing: the weather being sunny. Your good days are great and your bad days are terrible.

Now suppose instead you stock sunscreen *and umbrellas*. Sunny days sell sunscreen; rainy days sell umbrellas. Their sales are *negatively* correlated. Whatever the weather, one of the two is selling, so your total income is far steadier. You did not change how many products you carry — you carry two either way. You changed how *independently* they respond to the underlying driver (the weather). That, in one image, is the entire point of diversification. The chain sectors are two brands of sunscreen. Real diversification means adding the umbrella.

### Why correlation, not count, is what matters

The thing that reduces the risk of a portfolio is the *averaging out* of independent ups and downs. If you hold two stocks whose bad days never coincide, then on any given day one is probably cushioning the other, and your combined book bounces around far less than either piece. But that cushioning only works if the bad days really are independent. If the two stocks always have their bad days on the *same* days — because they share a driver — then there is nothing to average out. Two stocks moving in perfect lockstep are, for risk purposes, just one stock you happen to own in two pieces.

This is why "I own five stocks" tells you almost nothing about how diversified you are. The number that matters is the *correlation between them*. We will make this precise with the portfolio-variance math shortly, but hold the intuition first: **diversification is purchased with low correlation, and you get exactly none of it from holding many highly-correlated names.**

### How correlation is measured, and what it misses

It helps to know roughly how the number is computed, because that tells you what it can and cannot capture. To measure the correlation between two stocks, you take their returns over many periods (say weekly returns over a year), and you ask: when stock A's return was above its own average, was stock B's return also above its average, and by how much, period after period? If they were consistently above-average together and below-average together, the correlation is near +1. If A being above average told you nothing about B, it is near zero. The technical formula divides the *covariance* (the average product of the two return deviations) by the product of the two standard deviations, which scales the result to sit cleanly between minus one and plus one — but the formula is less important than the question it answers: *do the deviations line up?*

Three things this number quietly misses are worth flagging, because each one bites real investors. First, **correlation only measures linear co-movement** — it can miss a relationship that is real but kinks at the extremes. Two sectors can look uncorrelated in normal ranges yet crash together in a tail event, which is exactly the crisis-correlation problem we tackle later. Second, **correlation is not causation**: two sectors can be highly correlated because one causes the other, or because a third thing (credit) drives both. For the chain, it is the third-thing case — credit drives all five — which is *more* dangerous than a direct causal link, because it means a single external event can hit all five at once. Third, **correlation is a backward-looking average** computed over whatever window you choose; pick a calm window and you get a comfortingly low number that will not survive the next storm. Always ask "measured over what period?" before you trust a correlation.

### One more intuition: rolling dice versus copying one die

A final image to lock in why independence is the whole game. Imagine your portfolio's daily return is decided by dice. If you hold one stock, you roll one die — your outcome swings the full range, one to six. If you hold six stocks that are perfectly *independent*, you roll six dice and take the average; by the law of averages, that average clusters tightly around 3.5, with the wild extremes (all sixes, all ones) almost never happening. The averaging of independent rolls is what crushes the swings — that is diversification working.

But now suppose your six "stocks" are six *copies of the same die* — whatever the first one rolls, the other five roll the same. Averaging six identical rolls gives you back exactly the single roll: one to six, full swings, no smoothing at all. That is a basket of chain stocks. They are not six independent dice; they are one die reported six times. Diversification is the difference between rolling six dice and copying one die — and correlation is precisely the dial that says how much each new "die" is a fresh roll versus a copy.

This is why "I own five stocks" tells you almost nothing about how diversified you are. The number that matters is the *correlation between them*. We will make this precise with the portfolio-variance math shortly, but the intuition is now locked: **diversification is the averaging of independent bets, and correlation measures how independent each bet really is.**

## Why the chain sectors are so highly correlated

So why are the heavy VN-Index cyclicals — banks, property, brokers, steel, construction — correlated as high as 0.65 to 0.82 with each other? Because they are not five independent businesses responding to five different drivers. They are five links in one chain, and the chain runs on a single shared input: **credit** (borrowed money — bank loans, corporate bonds, margin lines). This is the subject of its own dedicated deep dive, [the financial chain](/blog/trading/vietnam-stocks/the-financial-chain-banks-property-brokers-steel-construction); here we need just enough of the mechanism to see *why the correlation is so high*.

### One driver feeding five sectors

Walk the chain. When the State Bank of Vietnam (SBV) loosens credit, **banks** get more room to lend, so their loan books and profits grow. Developers borrow that cheap credit, so **real estate** can buy land, build, and presell apartments. A rising market and loose credit swell **margin** balances (*ky quy*, money brokers lend retail investors to buy more stock than their cash allows), so **securities brokers** earn more on lending and on the surge in trading volume. The building boom that credit funds consumes rebar and structural steel, so **steel** demand and prices rise. And the contractors doing the building — **construction** — see their order books fill. One input, credit, fans out into all five sectors' earnings at once.

The crucial consequence: when the *one shared driver* moves, *all five sectors move the same way at the same time*. Loosen credit and all five rise. Tighten credit, or freeze the bond market, and all five fall. There is nothing to average out between them, because they do not have independent good and bad days — they share the same calendar of good and bad days, dictated by the credit cycle. That shared driver is exactly what a high correlation number is detecting.

### The reflexive loop makes the correlation even tighter

There is a second-order effect that pushes the chain's internal correlation *higher* than the shared input alone would explain: a reflexive feedback loop between credit and collateral. When credit is loose, property prices rise; rising property prices are the *collateral* against which banks lend, so higher collateral values let banks lend even more, which pushes credit looser still, which lifts property further. The loop reinforces itself on the way up. Brokers amplify it: as the market rises, the value of stock posted as margin collateral rises, so brokers extend more margin, which buys more stock, which lifts the market. Credit feeds collateral feeds credit.

This reflexivity is why the correlation is not merely high but *self-tightening in a trend*. In a strong move, every link is simultaneously cause and effect for every other link, so they lock into near-lockstep. It is also why the reversal is so violent: the same loop runs backward, with falling collateral forcing credit to contract, forcing more selling, forcing collateral down further. A book of chain stocks does not just share a driver — it shares a *feedback loop*, which is the most correlated thing a group of stocks can possibly be.

### Lead-lag does not save you

A natural objection: "the links don't all turn on the exact same day — brokers turn before steel — so surely they're not perfectly correlated, and that gives me some diversification?" It is true that the chain has a lead-lag order: credit and brokers turn first, while construction and steel earnings lag by two to three quarters because a building boom takes time to show up in tonnes shipped and contracts billed. But lead-lag is *timing*, not *independence*. The links still all go the same direction over any cycle that matters; the broker just gets there a quarter early and the steelmaker a quarter late. Over a full down-leg they all end up deeply negative together. Lead-lag is useful for *anticipating rotation within the chain* — buying the link that turns first — but it provides essentially zero diversification, because the *destination* is shared even when the *timing* differs slightly. Do not mistake a small phase shift for genuinely independent movement.

### Reading the correlation matrix

The cleanest way to *see* this is a correlation matrix — a grid where each cell is the correlation between the row sector and the column sector. The diagonal is always 1 (every sector is perfectly correlated with itself). The off-diagonal cells tell the story.

![Heatmap of approximate sector return correlations across seven Vietnamese sectors](/imgs/blogs/sector-correlations-and-diversification-vietnam-2.png)

Read the figure as two neighborhoods. The top-left block — banks, property, brokers, steel — is a sea of high numbers: banks-property 0.78, banks-brokers 0.80, property-brokers 0.82, property-steel 0.70, brokers-steel 0.75. Those are *very* high correlations for things that are nominally "different industries." Now look at where utilities, staples, and IT sit relative to that block: utilities-to-chain around 0.30 to 0.40, staples-to-chain around 0.30, IT-to-chain around 0.45. Those are the cool cells — the sectors whose movements are substantially independent of the credit cycle.

That single picture is the whole post in one image. The chain sectors are hot together (a "diversified" basket of them is one bet), and the cool cells are where genuine diversification lives.

#### Worked example: the illusion priced out

Take Minh's book in concrete numbers. He put 500 million dong (about **\$19,700** at the mid-2026 rate of about \$1 = 25,400 VND) into the market, split equally — 100 million dong each — across a bank, a developer, a broker, a steelmaker, and a construction contractor. He believed he had five 100-million-dong bets, each one independent, so that even a disaster in one would cost him at most a fifth of the book.

But the average pairwise correlation among those five is roughly 0.75. When a credit shock hits, they do not fail one at a time — they fail together. In the 2022 down-leg they fell about 40%, 48%, 58%, 52%, and 50% respectively. The book did not lose a fifth. It lost about (40 + 48 + 58 + 52 + 50) / 5 = **49.6%** — nearly half — in one coordinated move, from 500 million dong to about 252 million dong (roughly **\$9,900**). His five "independent" bets were one bet that lost about half its value. **When the pairwise correlation is near 0.75, splitting one bet into five tickers does not split the risk — it just gives the same bet five names.**

## The diversification illusion: many positions, one bet

Now we can name the trap precisely. The **diversification illusion** is the gap between how diversified a book *looks* (many positions, many tickers, many industries) and how diversified it *is* (how independently those positions actually move). Minh's book looked maximally diversified — five sectors, equal weights — and was almost completely undiversified, because the five sectors were one credit bet.

### Why the illusion is so convincing

The illusion is seductive for three reasons, and each one is worth seeing clearly.

First, the names genuinely *are* different companies in different industries. A bank really does a different thing day-to-day than a steel mill. The labels "Banking," "Real Estate," "Securities," "Basic Resources," "Construction" appear in different boxes on the exchange's sector classification. Everything about the surface presentation says "diverse." Only the *return behavior* — the thing you actually care about — is identical, and that is invisible unless you go looking for the correlation.

Second, on calm days the illusion holds up. When credit is neither loosening nor tightening dramatically, the chain sectors *do* wiggle somewhat independently on company-specific news — a developer announces a project, a bank reports earnings, a steelmaker signs a contract. In quiet markets the measured correlation drops, and the book looks reasonably spread. The illusion only shatters in the moments that matter most — the big credit moves — when the shared driver overwhelms all the idiosyncratic noise and the correlation jumps. We will return to this crisis-correlation effect; it is the cruelest part of the whole story.

Third, the illusion feels like prudence. Minh did the responsible-sounding thing: he refused to put everything in one stock. "Don't put all your eggs in one basket" is genuinely good advice. The trap is that he put his eggs in five baskets that were all tied to the same rope. Spreading across positions *feels* like risk control, so he stopped looking — never asking the only question that mattered, which is whether the positions move independently.

### A "diversified" book that is one bet, vs. a real one

Here is the contrast in pictures. On the left, the naive book: five chain names, equal weight, which collapses to one credit bet. On the right, a genuinely diversified book: some chain exposure (you do want to participate in the credit upcycle), plus a defensive sector, plus a structural-growth sector, plus a cash buffer. Same number of slots, completely different risk.

![Before and after comparison of an all chain book versus a truly diversified book](/imgs/blogs/sector-correlations-and-diversification-vietnam-3.png)

The right-hand book is not magic and it is not free — by adding low-correlation pieces you give up some of the explosive upside you would get from being all-in on the chain when the credit cycle is roaring. That is the trade. Diversification trims the best case in exchange for cutting the worst case. For most investors, and especially for anyone who cannot perfectly time the credit cycle, that is a trade worth making. The point of this post is to make sure you are *actually* making it, and not just believing you are.

## What genuinely diversifies a Vietnamese book

If a basket of chain stocks is one bet, what actually adds independence? The answer is: sectors and assets whose earnings run on a *different driver* than the credit cycle. There are four broad buckets, in roughly increasing order of how reliably they help.

### Defensives: utilities, staples, pharma

**Defensive** sectors sell things people buy regardless of the economic cycle: electricity and water (**utilities**), food and basic household goods (**consumer staples**), and medicine (**pharma/healthcare**). Their revenue does not surge in a credit boom or collapse in a credit bust — people keep the lights on, keep eating, and keep taking their medication in good times and bad. Because their earnings driver (essential consumption) is largely independent of the credit cycle, their stock returns correlate only weakly with the chain — roughly **0.30 to 0.40**.

These are the umbrellas to the chain's sunscreen. They will not make you rich in a roaring bull market — a defensive utility plodding along at a steady dividend looks dull next to a broker tripling in a margin-fueled rally. But in a credit shock, when the chain is falling 50%, a regulated power utility with contracted cash flows might fall only 10 to 15%, and a staple food producer might be roughly flat. The companion deep dives on [late-cycle and defensive sectors](/blog/trading/vietnam-stocks/late-cycle-and-defense-utilities-staples-healthcare) and [utilities](/blog/trading/vietnam-stocks/utilities-power-water-gas-sector-vietnam-defensive-cash) go deep on why these cash flows are so stable.

### Structural growth: IT

**Information technology** is a different animal. Vietnamese IT — led by names like FPT — grows on a driver that is largely *secular* rather than *cyclical*: global demand for software outsourcing and digital transformation, denominated substantially in foreign currency. That demand keeps growing through domestic credit cycles, so IT correlates with the chain only around **0.45** — higher than pure defensives (because it is still a stock in the same market, sharing the broad market tide) but far below the chain-internal 0.75.

IT diversifies in a particular way: it adds a growth engine that is not the credit cycle. When domestic credit is tight and the chain is struggling, an exporter earning dollars from overseas clients can keep compounding. The [IT sector deep dive](/blog/trading/vietnam-stocks/information-technology-sector-vietnam-fpt-structural-growth) covers why this revenue base is so much more resilient than domestic cyclical earnings.

### Cash: the only true zero

**Cash** — money in a bank deposit or money-market fund — is the one holding that is genuinely correlation *zero* with every stock. It does not fall when the market falls. In a 49.6% drawdown of the equity book, the cash portion sits there, unchanged, earning its deposit rate. Cash is not exciting, and in a bull market it is a drag (it earns a few percent while stocks run double digits). But its diversification value is unmatched precisely *because* it is the only thing that holds its value when everything correlated rushes to one. Cash is also optionality: it is the dry powder that lets you buy the chain cheaply *after* the crash, which is where the real returns are made.

### Export and FX plays

A fourth, more advanced bucket: companies whose fortunes ride a *different* macro variable than domestic credit — for instance **exporters** earning foreign currency, who can actually *benefit* from a weakening dong (their dollar revenue translates into more dong) even as the domestic credit cycle turns down. This is a partial hedge: when a credit shock coincides with dong weakness, an export earner's FX tailwind can offset some of the equity-market headwind. It is not a clean negative correlation, but it introduces a genuinely different driver into the book.

### The hierarchy of diversifiers

It is worth ranking these four buckets by *how reliably* they help, because not all diversification is equal and the ranking is counterintuitive. The order is not by how low the calm-market correlation is — it is by how well that low correlation *survives a crisis*.

**Cash is first**, and it is in a class of its own, because its zero correlation is structural, not statistical. It cannot spike to one in a panic, because cash is the thing everyone is panicking *toward*. Every other diversifier on this list is a risky asset whose measured correlation can betray you exactly when you need it; cash is the only one whose diversification is guaranteed by construction.

**Defensives are second.** Their 0.30 to 0.40 correlation does rise in a crisis — a forced seller dumps the utility too — but it rises *less* than the chain's internal correlation, and the underlying businesses keep generating cash, so the *price* recovers faster once the panic passes. They are imperfect but genuinely helpful ballast.

**IT and export/FX plays are third.** They diversify the *growth driver* (global software demand, the currency) rather than the *risk-off behavior*, so they help most in a slow grind where the domestic cycle weakens but there is no acute panic. In a sharp liquidity crisis they get sold like everything else, but they bring a different earnings engine that compounds through the cycle.

The practical reading: build your *crisis* protection out of cash, your *steady* ballast out of defensives, and your *growth* diversification out of IT and exporters. Do not expect the third bucket to save you in a 2022-style panic — that is cash's job.

![Flow diagram showing chain exposure plus defensives growth cash and export plays building a diversified book](/imgs/blogs/sector-correlations-and-diversification-vietnam-5.png)

#### Worked example: the portfolio-variance math

Let us make precise *why* low correlation reduces risk, with the actual formula, because this is the load-bearing math of the whole subject. Portfolio risk is measured by **variance** (the square of volatility). For two equally-weighted positions A and B, each with volatility (standard deviation) sigma, and correlation rho between them, the variance of the combined book is:

```
var(portfolio) = (1/2)^2 * sigma^2  +  (1/2)^2 * sigma^2  +  2 * (1/2) * (1/2) * rho * sigma * sigma
              = 0.5 * sigma^2 * (1 + rho)
```

The volatility of the book is the square root of that: sigma_portfolio = sigma * sqrt(0.5 * (1 + rho)).

Now plug in numbers. Say each stock has 40% annual volatility. If you hold two chain stocks at **rho = 0.80**, your book volatility is 40% * sqrt(0.5 * 1.80) = 40% * sqrt(0.90) = 40% * 0.949 = **37.9%**. You barely reduced risk at all — from 40% in one stock to 37.9% in two. The second stock did almost nothing because it was the same bet.

If instead the second stock is a defensive at **rho = 0.30**, your book volatility is 40% * sqrt(0.5 * 1.30) = 40% * sqrt(0.65) = 40% * 0.806 = **32.2%**. That is a real reduction — from 40% to 32.2%, a fifth of the risk gone, for the same two-stock book. **At rho = 0.8 the second position barely diversifies; at rho = 0.3 it meaningfully cuts your volatility — the correlation, not the count, does the work.**

## How correlation rises in a crisis: everything goes to one

Here is the cruelest fact in the whole subject, and the one that ruins more "diversified" portfolios than any other: **correlations are not stable, and in a crisis they all rise toward one.** The diversification you measured in calm markets — the comforting 0.30 between your defensive and your chain stock — is exactly the number that fails you when you need it.

### Why the spike happens

In normal times, stocks move on a mix of two things: a *common* factor (the overall market, the credit cycle) and *idiosyncratic* factors (this company's earnings, that sector's news). On a quiet day, idiosyncratic noise dominates, so different sectors wander somewhat independently and measured correlations are moderate.

In a crisis, the common factor explodes and swamps everything. When a credit shock hits, *every* risky asset is suddenly being driven by the same one thing — fear, deleveraging, margin calls, a rush for cash — and the idiosyncratic stories stop mattering. A forced seller facing a margin call does not sell only their chain stocks; they sell *whatever they can*, including the defensive utility, because they need cash now. So even genuinely low-correlation pairs get yanked downward together. The correlation that was 0.30 in calm markets measures 0.85 in the panic. Diversification thins out exactly when the storm arrives.

![Line chart showing average pairwise correlation rising to a peak during a crisis then falling](/imgs/blogs/sector-correlations-and-diversification-vietnam-4.png)

The figure shows the stylized arc: average pairwise correlation drifting in the 0.45 to 0.52 range in the calm pre-crisis months, then spiking to 0.85 to 0.90 at the height of the selloff, then relaxing back toward 0.60 as the panic subsides. The diversification you counted on lives in the low part of that curve and disappears at the peak.

#### Worked example: the diversification that vanished

Minh's more sophisticated friend, Lan, thought she had fixed the problem. Her book was three chain stocks and two defensives, and she had checked the correlations in the calm market of early 2022: her defensives sat around **0.30** to the chain. She reasoned that when the chain fell, her defensives — only weakly correlated — would hold up and cushion the book.

Then the bond crisis hit, and the realized correlation of her defensives to the chain jumped to about **0.85**. Her power utility, which "should" have fallen far less than her broker, instead fell about 30% as forced sellers dumped everything liquid to raise cash. The cushioning she had budgeted for — a defensive falling 10% while the chain fell 50% — did not arrive; the defensive fell 30%. Her measured-in-calm 0.30 was a fair-weather number. **The correlations you can rely on are the ones that hold in a crisis — and most don't, so the only genuinely crisis-proof diversifier is the one with zero correlation by construction: cash.**

This is why cash is special. Every *stock-to-stock* correlation can spike in a panic, because all stocks are risky assets that forced sellers dump for cash. But cash *is* the thing they are rushing toward — its correlation to stocks does not drift to one in a crisis, it stays at zero (or even goes negative in real terms as cash becomes precious). The lesson is not "diversification is useless" — it is "size your true crisis cushion in the one asset whose correlation cannot betray you."

### The margin-call mechanism that synchronizes the crash

It is worth understanding the precise machinery that drives correlations to one, because it tells you *when* the spike is coming. In the Vietnamese market the dominant mechanism is the **margin-call cascade**. A large share of the market's buying in a bull is done on margin — borrowed money from brokers. When prices fall enough, the broker's risk system issues a *margin call*: the leveraged investor must add cash or have positions force-sold. If they cannot add cash (and in a broad selloff, few can), the broker sells the investor's stock automatically, at whatever price the market offers.

Here is the cruel part: the broker's algorithm does not sell only the falling chain stocks. It sells the investor's *most liquid* positions to raise cash fastest — and the most liquid positions are often the large defensive blue-chips. So a margin call triggered by a collapsing property stock results in the forced sale of an investor's *utility* position, dragging the defensive down for reasons that have nothing to do with the utility's business. Multiply this across thousands of leveraged accounts hitting margin calls in the same week, and you get the mechanical engine that yanks *every* liquid stock down together — the defensive's correlation to the chain spikes not because its fundamentals changed, but because it was collateral in someone else's blown-up trade. This is why the correlation spike is sharpest exactly at the bottom, when forced selling peaks, and why it relaxes as the deleveraging exhausts itself.

This is also a *signal*: when you see margin balances at record highs going into a credit-cycle turn, you are looking at the fuel for the next correlation spike. The companion deep dive on [the margin cycle](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam) covers how to read margin balances as an early-warning gauge.

This is why cash is special in a second sense. Every *stock-to-stock* correlation can spike in a panic, because all stocks are risky assets that forced sellers dump for cash. But cash *is* the thing they are rushing toward — its correlation to stocks does not drift to one in a crisis, it stays at zero (or even goes negative in real terms as cash becomes precious). The lesson is not "diversification is useless" — it is "size your true crisis cushion in the one asset whose correlation cannot betray you."

## How many real bets do you actually have?

We can now answer the question that should replace "how many stocks do I own?": *how many genuinely independent bets am I running?* This is the single most useful reframe in portfolio construction, and it has a rough quantitative answer.

### Effective bets, not ticker count

The intuition: if you hold six stocks that are perfectly correlated (rho = 1), you really have **one** bet — they are the same thing in six pieces. If you hold six stocks that are perfectly *uncorrelated* (rho = 0), you have **six** genuine bets, and the averaging-out of six independent risks shrinks your book volatility dramatically. Real portfolios sit in between, and the number of *effective independent bets* depends almost entirely on the average correlation, not the count.

A useful back-of-envelope: for N equally-weighted positions with average pairwise correlation rho, the number of effective independent bets is roughly **N / (1 + (N - 1) * rho)**. You do not need to memorize the formula; you need the consequence. Plug in N = 6 chain stocks at rho = 0.75: effective bets = 6 / (1 + 5 * 0.75) = 6 / 4.75 = **1.26** — barely more than one bet, despite six tickers. Now plug in N = 6 *mixed* stocks (chain + defensives + IT + a cash-like holding) with a much lower average correlation, say rho = 0.30: effective bets = 6 / (1 + 5 * 0.30) = 6 / 2.5 = **2.4**, and as you push correlations lower and add cash, the effective count climbs toward 4.

![Bar chart comparing effective independent bets for a chain book versus a mixed book](/imgs/blogs/sector-correlations-and-diversification-vietnam-6.png)

The figure makes the contrast brutal: six chain stocks deliver roughly **1.5** effective bets, while six mixed-sector stocks deliver roughly **4**. Same number of positions, the mixed book carries nearly three times the genuine diversification. *This* is what you are buying when you reach for low-correlation sectors — not more tickers, more independence.

### Why the effective-bets number is the one to internalize

The reason this single number is so powerful is that it collapses the entire diversification question into one figure that you can actually act on. Ticker count flatters you — it goes up every time you click "buy," regardless of whether you bought real diversification or just more of the same bet. The effective-bets number does not flatter you: it stays stubbornly near one no matter how many chain stocks you stack, and it only climbs when you add genuinely independent drivers. It is, in effect, your portfolio's honesty meter.

There is also a hard ceiling worth knowing. As you add more and more equally-correlated positions, the effective-bets number does not grow without limit — it converges to **1 / rho**. With an average correlation of 0.75, the most diversification you can *ever* squeeze out of chain stocks, even with a thousand of them, is 1 / 0.75 = about **1.33** effective bets. You simply cannot diversify your way out of a single driver by piling on more exposure to that driver; the math forbids it. The only escape is to lower the *average* correlation, which means adding positions tied to different drivers entirely. This is the quantitative proof of why "more stocks" is a dead end and "different drivers" is the only road.

#### Worked example: adding a diversifier to Minh's book

Let us fix Minh's book and quantify the improvement. He swaps two of his five chain names (say the broker and the steelmaker, the two highest-beta links) for a regulated power utility and an IT growth name. His new book: bank, developer, construction, utility, IT — three chain, two diversifiers.

Replay the 2022 shock. The three remaining chain stocks fall about 40%, 48%, 50%. The utility, even with crisis-elevated correlation, falls about 25%. The IT exporter, buffered by dollar revenue, falls about 30%. The equal-weighted book drawdown is now (40 + 48 + 50 + 25 + 30) / 5 = **38.6%**, versus the all-chain book's 49.6%. On his 500 million dong (about \$19,700), that is the difference between ending at about 302 million dong (about \$11,900) instead of 252 million dong (about \$9,900) — roughly **\$2,000** of capital preserved, about 11 percentage points less drawdown, by swapping two correlated names for two diversifiers. **Two lower-correlation positions did not just trim the average — they cut the worst-case drawdown by a fifth, which is exactly the protection you bought them for.**

## Common misconceptions

A few persistent myths cause more diversification damage than ignorance does, because they feel sophisticated. Each one is corrected with a number.

### Myth 1: "More stocks means more diversified"

The most common myth, and the one Minh believed. The number of stocks is almost irrelevant; the *average correlation* between them is what counts. Twenty chain stocks at rho = 0.75 give you about 20 / (1 + 19 * 0.75) = 20 / 15.25 = **1.31** effective bets — essentially the same one bet as six of them. Adding the seventh, tenth, twentieth chain stock buys you almost nothing, because each new name is more of the same credit exposure. Diversification is bought with low correlation, and you cannot buy it by stacking more of the same correlated thing. Two genuinely independent positions beat twenty correlated ones.

### Myth 2: "Spreading across the index diversifies"

"I'll just buy a broad VN-Index basket — surely that's diversified." Here is the trap specific to Vietnam: VN-Index is *itself* dominated by the chain. Banks alone are roughly a third of the index by weight; add property, brokers, steel, and construction and the credit chain is well over half the entire index. So a market-cap-weighted "diversified" index position is, in reality, a **majority bet on the credit cycle**. Buying the index does spread you across many names, but those names are concentrated in exactly the correlated sectors we are trying to diversify *away* from. To genuinely diversify, you have to *overweight* the under-represented low-correlation sectors relative to their small index weights — the index will not do it for you.

### Myth 3: "Correlations are stable, so I can set and forget"

We just dismantled this one, but it bears repeating as a myth because so many investors size their risk on calm-market correlations. The 0.30 you measured between your defensive and the chain in a quiet quarter is a fair-weather number; in the 2022 panic that same pair realized about 0.85. If you sized your "safe" cushion assuming 0.30 would hold, your actual drawdown was far worse than your model said. Correlations are *regime-dependent* — they rise in stress — so always stress-test your book assuming the low correlations break toward one, and hold your true safety in cash, whose zero does not break.

### Myth 4: "Diversification means giving up returns"

Half-true, and the half that is false is dangerous. Diversification trims the *best case* (you are not all-in on the chain when it triples) but it disproportionately cuts the *worst case*, and avoiding deep drawdowns is worth more to long-run compounding than capturing every last bit of upside. A book that falls 38% and recovers needs about a 61% gain to get back to even; a book that falls 50% needs a **100%** gain to recover. By cutting the drawdown from 50% to 38%, diversification roughly halves the climb back. Over a full cycle, the diversified book that avoided the deepest hole often *ends up ahead* of the concentrated one, despite capturing less of the boom. The asymmetry of drawdown recovery — losses needing larger gains to undo them — is the mathematical reason that "boring" diversification beats "exciting" concentration over time.

### Myth 5: "I diversified by holding two banks"

A subtler version of myth one, and very common among Vietnamese retail investors who own "a private bank and a state bank." Two banks are about as correlated as it is possible for two distinct stocks to be — easily 0.85 or higher — because they are the *same link* in the chain, with the same loan-book exposure to the same credit cycle. Splitting one sector position into two names within that sector buys essentially zero diversification; it only diversifies away *single-company* risk (the small chance that one specific bank has an accounting scandal), which is real but minor next to the *sector* risk you are still fully exposed to. The same applies to owning two property developers, or two brokers. Diversifying *within* a sector is rearranging deck chairs; diversification lives *between* uncorrelated sectors, not between two names in the same one.

## How it shows up on VN-Index

Theory is cheap; the Vietnamese market has run the live experiment twice in recent memory. Here is how correlation and diversification actually played out.

### The 2022 everything-down

We have used 2022 as the running example, and it is the canonical case. From the early-2022 peak near 1,498 on VN-Index to the late-2022 trough near 1,007 — a roughly 33% index drawdown — the *internal* correlations of the chain sectors spiked to extreme levels. There was nowhere to hide *inside* the chain: bank, property, broker, steel, construction all fell 40 to 58% in a tightly synchronized move, because the bond-market freeze that started in property propagated through the entire credit chain. A book built from any combination of those five was a single trade.

The investors who came through 2022 least damaged were not the ones who owned *more* chain stocks or who picked the "best" bank — they were the ones who carried genuine non-chain ballast: a meaningful cash allocation, defensive utilities and staples, and structural-growth IT. The lesson VN-Index taught in 2022 is exactly this post's thesis: in a credit shock, the only thing that protected a book was *low correlation to the credit chain*, and that protection had to be bought *before* the shock, when it looked like a boring drag on a roaring bull.

### The defensives that held

Look at the relative behavior within that 2022 selloff and the contrast is stark. While the chain fell about 50%, several defensive names fell far less. Regulated power utilities with contracted offtake — whose revenue is set by long-term agreements and barely flinched at the credit freeze — drew down only mildly. Large staple food and beverage producers, selling the same volume of product to the same consumers, held up dramatically better than any cyclical. And the largest IT exporter kept *growing earnings* through the whole episode, because its customers were overseas enterprises whose budgets had nothing to do with Vietnamese corporate-bond rollovers. These were not lucky stock picks. They were *structurally* low-correlation to the chain, and that structure showed up exactly when it counted.

### Foreign flows add a second synchronizing force

There is a Vietnam-specific wrinkle that makes the chain even more correlated than its credit linkage alone: passive foreign flows. Much foreign money enters and leaves VN-Index through ETFs and index funds that buy and sell the *whole index basket* at once. Because the chain dominates the index by weight, every wave of foreign inflow disproportionately buys the chain, and every wave of outflow disproportionately sells it — all the chain names together, on the same day, regardless of their individual fundamentals. When foreigners are net sellers for a stretch (as they were through large parts of 2021 to 2024), that selling pressure lands on the chain as a bloc, pushing its internal correlation higher still. A domestic investor who thought they had diversified across "different" chain sectors found those sectors moving in lockstep partly because a foreign ETF was redeeming the basket. The [foreign flows and index-effect deep dive](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) covers this mechanism; the takeaway here is that index-basket trading is a second force, on top of credit, welding the chain together.

### The 2020-21 bull: when correlation helped you and hid the risk

The flip side: in the 2020-21 liquidity-driven bull market, the chain's high internal correlation worked *for* holders. As SBV cut rates and margin balances swelled (see [liquidity and the margin cycle](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam)), the whole chain rose together, and a concentrated chain book delivered spectacular returns — brokers more than doubled, property and steel ran hard. This is the seductive part: high correlation in a *bull* feels like genius, because all your bets win at once. It lulls investors into believing their concentrated book is skill rather than a single leveraged bet on credit that happened to be going the right way. The same correlation that made you a hero in 2021 made you a victim in 2022. That symmetry is the whole point.

## The playbook: build a book that is actually diversified

Here is how to put all of this to work — concrete steps, with the signals and the invalidation.

### Step 1: count your real bets, not your tickers

Before adding anything, audit what you have. List your positions, group them by *driver* (not by sector label): how many are riding the domestic credit cycle (banks, property, brokers, steel, construction, and most domestic cyclicals)? If more than roughly half your book is in that one driver, you do not have a diversified portfolio — you have a leveraged credit bet, regardless of how many tickers it spans. Estimate your effective bets with the rough N / (1 + (N - 1) * rho) rule using an average chain correlation around 0.75; if the answer is close to one or two, you are concentrated.

### Step 2: add true diversifiers, sized to matter

Diversifiers only help if they are big enough to move the needle. A 2% position in a defensive utility next to a 90% chain book changes nothing. The buckets, in order of crisis-reliability:

- **Cash** — the only zero-correlation, crisis-proof holding. Hold a deliberate allocation (commonly 10 to 30% depending on where you think the credit cycle is) as both ballast and dry powder for buying the chain cheaply after a crash.
- **Defensives** — utilities, staples, pharma, at roughly 0.30 to 0.40 correlation. These are your equity ballast: they participate in the market's long-run rise but fall far less in a credit shock.
- **Structural growth (IT)** — around 0.45 correlation, a growth engine that runs on global software demand rather than domestic credit.
- **Export / FX plays** — a partial hedge whose driver is the currency, not the credit cycle.

Sizing these is the subject of its own deep dive on [sizing by sector beta](/blog/trading/vietnam-stocks/sizing-by-sector-beta-how-much-to-bet); the principle here is that the diversifier's *weight* must be large enough that its low correlation actually cushions the book.

#### Worked example: from one bet to four

Start from 500 million dong (about \$19,700) all in the chain — about 1.3 effective bets. Restructure to: 30% chain (150 million dong across a bank and a developer), 20% defensives (100 million dong in a utility and a staple), 15% IT (75 million dong), 15% an export/FX earner (75 million dong), and 20% cash (100 million dong). The average pairwise correlation across this book drops to roughly 0.35 *among the equities*, and the 20% cash is correlation zero with all of it.

Run the rough effective-bets math on the equity sleeve (about 80% of the book, five positions at average rho ~0.35): 5 / (1 + 4 * 0.35) = 5 / 2.4 = **2.1** effective equity bets, and the cash adds a fully independent unit, taking the whole book to roughly **3 to 4** effective bets. You went from owning one credit bet to owning three or four genuinely independent ones — *without adding a single extra dollar of capital*, just by reallocating toward low-correlation drivers. **The same \$19,700, re-pointed at different drivers instead of more of the same one, roughly tripled the number of real bets and cut the credit-shock drawdown from about 50% to under 30%.**

### Step 3: the signals to watch

- **Credit-cycle indicators.** Because the chain rides credit, watch the SBV's stance, system credit growth, and the corporate-bond market's health. When credit is loosening and margin balances are rising, the chain's high correlation works *for* you — leaning into it is fine, *as long as you know that is the bet you are making*. When credit is tightening or the bond market is stressing, the same correlation becomes your enemy, and you want more cash and defensives.
- **Realized correlation, not assumed.** Periodically check how your supposedly-diversified positions are *actually* co-moving. If your defensives start moving in lockstep with the chain, the crisis-correlation spike may be starting — that is a signal to raise cash, not to trust the calm-market number.
- **Your own concentration.** The simplest signal is your own book: if more than half of it shares one driver, that is the warning, full stop.

### Step 4: the invalidation

Every thesis needs a line that says "I was wrong." For this one: **the thesis is that the chain sectors are highly correlated and so do not diversify each other.** It would be invalidated if the chain sectors *decoupled* — if, in a genuine credit shock, banks rose while property fell, or brokers held while steel collapsed, persistently and not just for a few days of noise. That has not happened in the Vietnamese market's modern history; in every real credit shock the chain has moved as one. If you ever see a sustained decoupling — the links genuinely going their separate ways through a full credit cycle — then the chain has stopped being a chain, and a basket of its members would start to provide real diversification. Until you see that, treat any basket of chain stocks as a single bet, and build your diversification from the cool cells of the correlation matrix.

![Matrix comparing diversifiers and duplicators by correlation to chain and whether they diversify](/imgs/blogs/sector-correlations-and-diversification-vietnam-7.png)

The final matrix is your one-glance reference: chain cyclicals are duplicators (correlation to chain ~0.75, they do not diversify); defensives (~0.30 to 0.40) and IT (~0.45) and export/FX plays diversify partially; cash (correlation zero) is the only true, crisis-proof diversifier. Build from the bottom rows, not the top one.

### A note on rebalancing through the cycle

One last operational point: a diversified book is not a static thing you set once. The right mix shifts with where you are in the credit cycle. Early in a loosening cycle, when credit is cheapening and the chain's correlation is about to work *for* you, it is reasonable to lean heavier into chain exposure and lighter on cash — you are deliberately taking the credit bet because the odds favor it, and you accept the concentration with eyes open. Late in a tightening cycle, when the bond market is stressing and the next correlation spike is loading, you want to be shifting the other way: trimming chain, raising cash, leaning on defensives. The discipline is to make those shifts *deliberately and in advance*, based on credit-cycle signals, rather than being forced into them by a margin call after the crash has already started. The investor who raised cash to 30% in mid-2022 *before* the bond freeze had both protection on the way down and dry powder to buy the chain near the 1,007 trough — and buying a correlated bet *cheaply, after the crash* is where the credit chain actually pays you. Diversification is not only about surviving the down-leg; it is about having the independence and the cash to act when everyone else is forced to sell.

Put it all together and the discipline is simple to state, even if it takes real conviction to hold: stop counting tickers and start counting drivers. Every position you own is a bet on *something* — a credit cycle, an essential-consumption stream, a global software-demand curve, a currency, or nothing at all in the case of cash. A portfolio is diversified to exactly the degree that those underlying bets are independent, and not one bit more. Owning five chain stocks is owning the credit cycle five times; owning a chain stock, a utility, an IT exporter, and a slug of cash is owning four different things. The market does not reward you for the number of names on your statement. It rewards you for the number of genuinely independent bets behind them — and in a credit shock, it ruthlessly punishes anyone who confused the two.

## Further reading and cross-links

- [The financial chain: banks, property, brokers, steel, construction](/blog/trading/vietnam-stocks/the-financial-chain-banks-property-brokers-steel-construction) — the full mechanism of *why* the chain sectors share one credit driver, link by link, and how the 2022 contagion ran through it. This post is the "so what for your portfolio" companion to that one.
- [Sizing by sector beta: how much to bet](/blog/trading/vietnam-stocks/sizing-by-sector-beta-how-much-to-bet) — once you know what to own, how large to make each position given its volatility and its correlation to the rest of the book.
- [Late-cycle and defense: utilities, staples, healthcare](/blog/trading/vietnam-stocks/late-cycle-and-defense-utilities-staples-healthcare) — the low-correlation sectors that actually cushion a book in a credit shock, and when in the cycle to lean on them.
- [Liquidity and the margin cycle in Vietnam](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam) — why margin balances drive the chain's correlation higher in bull markets and how the margin-call cascade synchronizes the crash.
- [Anatomy of a stock sector: why industries move together](/blog/trading/vietnam-stocks/anatomy-of-a-stock-sector-why-industries-move-together) — the foundational post on what makes a whole industry's stocks co-move in the first place, which is the building block for everything here.

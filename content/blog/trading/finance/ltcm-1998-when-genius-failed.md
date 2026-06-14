---
title: "Long-Term Capital Management: When Genius and Leverage Nearly Broke the System"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a hedge fund run by Nobel laureates turned a tiny pile of equity into a position so large and so leveraged that a Russian default forced the Federal Reserve to organize a private rescue."
tags: ["ltcm", "hedge-funds", "leverage", "risk-management", "financial-crisis", "value-at-risk", "arbitrage", "systemic-risk", "case-study"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Long-Term Capital Management was a hedge fund run by the smartest people in finance, and it nearly took the global banking system down with it because it borrowed far too much against far too little.
>
> - In 1998 a fund with about \$5 billion of its own money controlled roughly \$125 billion of bonds and over \$1 trillion of derivatives.
> - It made dozens of tiny, "sure-thing" bets that two nearly identical prices would converge; leverage of about 25 to 1 turned those pennies into a fortune.
> - When Russia defaulted in August 1998, panicked investors fled to safety, the spreads diverged instead of converging, and the fund lost about \$4.6 billion in five weeks.
> - To stop a chain reaction through the banks that had lent to it, the Federal Reserve organized a private \$3.6 billion bailout by a consortium of 14 banks on September 23, 1998 - no taxpayer money was used.
> - The durable lesson: a correct model plus extreme leverage plus a liquidity crisis equals ruin, because in a panic everyone wants out at once and there is no door.

In January 1998, Long-Term Capital Management (LTCM) was arguably the most respected investment fund on Earth. Its founders included John Meriwether, the legendary bond trader who had built the most profitable desk at Salomon Brothers, and two economists - Robert Merton and Myron Scholes - who would share the 1997 Nobel Memorial Prize in Economic Sciences for inventing the mathematics of options pricing. The fund had returned more than 40 percent a year for three straight years. Banks lined up to lend it money on terms they offered no one else.

By the end of September 1998, that fund was effectively bankrupt, and the Federal Reserve Bank of New York had quietly summoned the heads of Wall Street's biggest banks into a room to arrange a rescue - not because they wanted to save LTCM, but because LTCM owed them so much money, and was tangled into so many trades with them, that its collapse threatened to detonate the entire financial system. The fund had lost about 90 percent of its capital in roughly five weeks.

The diagram above is the mental model: a fund that compounded quietly and spectacularly for four years, then unraveled almost overnight when one event - a Russian default - flipped the sign on every bet at once.

![Timeline from LTCM founding in 1994 to the September 1998 rescue](/imgs/blogs/ltcm-1998-when-genius-failed-1.png)

This is a story about leverage. Not about stupidity - the people involved were genuinely brilliant, and their core ideas were mostly correct. It is about what happens when correct ideas are scaled up with enormous borrowed money, and then the world does something the model assumed it almost never would. By the end of this piece you will understand exactly how a 4 percent move in the wrong direction can wipe out a fund, why "diversified" bets all lost together, why a calculation that said "we can lose at most \$45 million on a bad day" was off by a factor of ten, and why the people who lend money are the ones who decide when the music stops.

## Foundations: every term the story turns on

Before we walk through the blow-up, we need to build the vocabulary from zero. LTCM's story is dense with jargon, but every term is simpler than it sounds. We will define each one with a plain-language version and a small number so it sticks.

**A hedge fund** is a private investment pool that takes money from wealthy individuals and institutions and tries to make profits in ways an ordinary mutual fund cannot - by borrowing money, by betting that prices will fall (going "short"), and by using complex instruments like derivatives. "Hedge" originally meant reducing risk by offsetting one bet with another, but the modern industry uses the word loosely. LTCM was a hedge fund, and if you want a fuller tour of how these vehicles operate and charge fees, see [How hedge funds work: leverage, "2 and 20", and the carry](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20).

**Leverage** is using borrowed money to make a bigger bet than your own cash would allow. If you have \$1 and borrow \$24 to buy \$25 of something, you are "leveraged 25 to 1." Leverage multiplies gains and losses equally. If your \$25 of assets rises 4 percent, you make \$1 - a 100 percent return on your own \$1. If it falls 4 percent, you lose \$1 - and your own money is gone. Leverage is the single most important concept in this entire story.

**Basis points (bps)** are how finance measures tiny percentages. One basis point is one hundredth of one percent, or 0.01 percent. One hundred basis points equals 1 percent. When two bond prices differ by "10 basis points," they differ by 0.10 percent - a sliver. LTCM lived and died on slivers.

**A bid-ask spread** is the gap between the price at which you can sell something (the bid) and the price at which you can buy it (the ask). In calm markets this gap is tiny; in a panic it explodes, because dealers do not want to hold risky things, so they quote a low buy price and a high sell price. A widening spread is a sign that liquidity is drying up.

**Liquidity** is the ability to turn an asset into cash quickly without moving the price much. A U.S. Treasury bond is highly liquid; you can sell \$100 million of it in seconds. An obscure mortgage bond is illiquid; trying to sell it fast forces you to drop the price. The central trap in this story is that LTCM's positions were liquid in normal times and became wildly illiquid in a crisis, exactly when the fund needed to sell.

**Arbitrage** is, in its pure textbook form, a risk-free profit: if the same thing trades at two different prices, you buy the cheap one and sell the expensive one and pocket the difference. True risk-free arbitrage barely exists. What LTCM did is better called **relative-value** or **convergence arbitrage**: it found two assets that were *nearly* identical and whose prices *should* converge over time, bought the cheaper one, shorted the richer one, and waited for the gap to close. It was not risk-free - the gap could widen first - but historically it almost always closed.

**A convergence trade** is exactly that bet: you wager that the price gap between two similar securities will shrink toward zero. LTCM's signature trade was buying an "off-the-run" Treasury bond and shorting an "on-the-run" one. An **on-the-run** Treasury is the most recently issued bond of a given maturity; it is the most actively traded and therefore slightly more expensive (investors pay a small premium for its liquidity). An **off-the-run** bond is an older issue of almost the same maturity; it trades a touch cheaper because it is less liquid. The economic difference between them is almost nothing - they are both promises from the U.S. government - so over time their prices converge. LTCM bought the cheap off-the-run, shorted the rich on-the-run, and collected the gap as it closed.

**Repo** (short for repurchase agreement) is how the fund borrowed. In a repo, you sell a bond to a lender for cash today and agree to buy it back tomorrow for slightly more; the difference is the interest. Functionally it is a short-term loan secured by the bond. Because Treasuries are such safe collateral, lenders demanded almost no extra cushion - sometimes effectively zero "haircut" - which meant LTCM could borrow nearly the full value of its bonds and recycle that cash into more bonds. Repo is the engine that let a small amount of equity control a vast amount of assets.

**Derivatives** are contracts whose value derives from something else - an interest rate, a stock index, a currency. A **swap** is a common one: two parties agree to exchange one stream of payments for another, for example swapping a fixed interest rate for a floating one. Derivatives let LTCM take huge economic exposures while putting up only a small amount of cash (the "margin"), which is another form of leverage. The face value, or **notional**, of LTCM's derivatives ran to about \$1.25 trillion - though that headline number overstates the real risk, as we will see.

**Value-at-Risk (VaR)** is the risk model the whole industry used (and still uses). VaR answers the question: "On a normal bad day, how much could we lose?" A typical statement is "our one-day 99 percent VaR is \$45 million," meaning that on 99 days out of 100 the fund should not lose more than \$45 million. VaR is built on a statistical picture of how prices move - usually assuming a bell curve (the "normal distribution") and using recent volatility to estimate the future. Its fatal weakness, which LTCM discovered, is that the 1-in-100 day is exactly the day that matters, and real markets have far more extreme days than a bell curve predicts.

**Fat tails** describe that flaw precisely. A normal bell curve says enormous moves are vanishingly rare. Real markets have "fat tails": big moves happen far more often than the bell curve says. A "six-sigma" event - one the bell curve says should occur roughly once in several million years - actually shows up every decade or two. LTCM's models were calibrated on the calm, bell-curve middle and were blind to the fat tail that arrived in 1998.

**Correlation** measures whether two things move together. A correlation of zero means they are independent; a correlation of 1 means they move in lockstep. LTCM believed its many trades were mostly independent - that a loss in Italian bonds and a loss in U.S. mortgage spreads were unrelated, so diversification protected it. In the crisis, every trade lost at the same time. The correlations the model assumed were near zero went to nearly 1. That is the single most important sentence about why diversification did not save them.

**Short selling** is betting that a price will fall. To short a bond, you borrow it from someone who owns it, sell it today at the current price, and promise to return an identical bond later. If the price falls, you buy it back cheaper and keep the difference; if it rises, you lose. Every LTCM convergence trade had a short leg - the "rich" security it expected to fall in relative value. Shorting is essential to relative-value trading because it lets you isolate the *gap* between two prices while hedging away the direction of the overall market. It also has an asymmetry worth noting: your potential loss on a short is theoretically unlimited, because a price can rise without bound, whereas the most you can lose on something you own is what you paid.

**Mark-to-market** is the daily ritual that turned LTCM's paper losses into real, immediate danger. Every day, the fund and its lenders revalue every position at the current market price. If a position has dropped, the loss is recognized *that day*, even if the fund has not sold anything and still believes the price will recover. Mark-to-market is what makes leverage so unforgiving: you do not get to wait for your thesis to play out in peace. The market's daily verdict is binding, because it determines your collateral.

**A margin call** is the demand that follows a bad mark. When the value of your collateral falls, the lender calls and says, in effect, "post more cash, today, or we sell you out." A leveraged fund must always have enough liquid cash to meet margin calls. When LTCM's positions were marked down day after day, the margin calls poured in, and meeting them forced the fund to sell good assets at terrible prices - the mechanism that converts a drawdown into a death spiral.

**A haircut** is the safety cushion a lender keeps on collateral. If you post a \$100 bond and the lender gives you \$98 of cash, the \$2 difference is a 2 percent haircut - it protects the lender if the bond falls before they can sell it. In the calm 1990s, lenders gave LTCM almost no haircut on Treasury collateral, which is what let it borrow nearly the full value of its bonds. In the panic of 1998, lenders raised haircuts sharply, which is a hidden form of margin call: the same collateral suddenly supported far less borrowing, forcing the fund to find cash it did not have.

**Who were the people?** John Meriwether ran arbitrage at Salomon Brothers in the 1980s and was a near-mythical figure in bond trading; he left after a scandal on his desk and founded LTCM in 1994. Myron Scholes and Robert Merton were academic economists. With the late Fischer Black, Scholes had derived the Black-Scholes-Merton formula for pricing options - one of the most influential equations in finance. Merton extended the theory. In October 1997 Scholes and Merton received the Nobel Memorial Prize in Economics for that work. Both were partners at LTCM. The fund also recruited a deep bench of brilliant traders and PhDs, and even a former vice chairman of the Federal Reserve, David Mullins. The intellectual firepower was real, and it was part of the problem: the aura of genius made banks comfortable lending almost without limit.

With that toolkit, we can now look at what the fund actually built.

## The setup: the dream team and the machine they built

LTCM launched in February 1994 and raised about \$1.25 billion - then the largest hedge fund debut in history. Investors had to commit at least \$10 million and lock it up for years. The pitch was seductive: a team of the world's best traders and economists would harvest tiny, well-understood mispricings across the global bond markets, hedge away the obvious risks, and use leverage to turn small reliable edges into large returns.

The strategy rested on a deep insight. In most markets, prices are roughly efficient, so easy profits are gone. But there are persistent small inefficiencies created by structure rather than by anyone's mistake. The on-the-run versus off-the-run Treasury gap is one: institutions that need maximum liquidity overpay for the newest bond, leaving the slightly older bond a touch cheap. That gap is real, predictable, and tends to close as the new bond ages and becomes "off-the-run" itself. The edge on any single trade was minuscule - a handful of basis points - but it was about as close to a sure thing as bond markets offer.

The pipeline below shows the mechanics of a single one of these trades.

![Pipeline of one LTCM convergence trade from spotting the gap to profit](/imgs/blogs/ltcm-1998-when-genius-failed-2.png)

LTCM ran many flavors of this same idea:

- **On-the-run vs off-the-run Treasuries** - the canonical convergence trade, betting the liquidity premium on the newest bond would shrink.
- **Swap spreads** - betting on the gap between the interest rate on government bonds and the rate on interest-rate swaps, which historically traded in a tight band.
- **Sovereign convergence** - for example betting that the bonds of Italy, Spain, and other European countries would converge toward German bond yields as Europe moved toward a single currency. This trade actually worked and made money.
- **Merger arbitrage** - buying the stock of a company being acquired (which trades just below the offer price) and betting the deal closes, collecting the small gap.
- **Equity volatility** - selling options on stock indexes, betting that future market volatility would be lower than the high level the options market was pricing in. This trade was so large that dealers nicknamed LTCM "the central bank of volatility."

Each trade was a small, hedged, statistically attractive bet. The genius - and the danger - was in the scale. By early 1998, LTCM had grown its equity to roughly \$4.7 billion. Against that equity it held about \$125 billion of assets on its balance sheet, a leverage ratio of roughly 25 to 1. On top of that sat a derivatives book with a notional value of around \$1.25 trillion. The trades were individually tiny edges, scaled up massively until the tiny edges produced enormous dollar profits.

The leverage stack is worth seeing as a picture, because the proportions are the whole story.

![Stack showing equity under balance-sheet assets under derivative notional](/imgs/blogs/ltcm-1998-when-genius-failed-5.png)

#### Worked example: how 25-to-1 leverage works

Start with the headline. LTCM had about \$5 billion of equity (its own and investors' money) controlling about \$125 billion of assets. Divide: \$125 billion / \$5 billion = 25. So the fund was leveraged roughly 25 to 1. For every \$1 of real capital, it controlled \$25 of bonds, \$24 of which was effectively borrowed.

Now ask what a small market move does. Suppose the \$125 billion of assets falls in value by just 4 percent. That is a loss of 0.04 x \$125 billion = \$5 billion. But the fund only had \$5 billion of equity. So a 4 percent adverse move on the assets wipes out 100 percent of the equity. The fund is insolvent.

The intuition: at 25-to-1 leverage, the gap between "great year" and "bankrupt" is a 4 percent move - the kind of move a diversified bond portfolio can have in a single bad week.

#### Worked example: a single convergence trade on \$1 billion notional

Take the classic trade. An off-the-run 30-year Treasury yields, say, 10 basis points more than the on-the-run 30-year - meaning it is cheaper. LTCM buys \$1 billion (face value) of the cheap off-the-run bond and simultaneously shorts \$1 billion of the rich on-the-run bond. The two positions are almost perfectly hedged against interest-rate moves: if all rates rise, the long loses and the short gains by nearly the same amount. What is left is a pure bet on the 10-basis-point gap closing.

How much is 10 basis points worth on a long-dated bond? For a 30-year bond, a 1-basis-point change in yield moves the price by roughly \$1,800 per \$1 million of face value (this sensitivity is called "DV01"). For \$1 billion of face value, 1 basis point is worth about \$1,800 x 1,000 = \$1.8 million. So if the 10-basis-point gap closes completely, the trade earns roughly 10 x \$1.8 million = about \$1 million, net of the offsetting legs - actually the net DV01 of the spread position is smaller because the two bonds have similar sensitivities, so a realistic profit on the *spread* itself is on the order of a few hundred thousand to about a million dollars as the 10bp gap goes to zero.

The intuition: one such trade earns pennies on each dollar of face value, so to make real money you must do it on billions of face value, dozens of times over - which is exactly why the balance sheet swelled to \$125 billion.

#### Worked example: how leverage turns a 0.5 percent edge into a 12 percent return - and a 0.5 percent loss into ruin

Suppose a trade has an expected edge of 0.5 percent on the assets you put to work. With no leverage, you earn 0.5 percent - dull. Now borrow to lever the position 24 to 1, so your \$1 of equity controls \$25 of the trade (the same 25-to-1 ratio LTCM ran). The 0.5 percent edge is earned on the full \$25:

```
gross gain   = 0.5% x $25 = $0.125 per $1 of equity
return on equity = $0.125 / $1.00 = 12.5%
```

A boring 0.5 percent edge becomes a brilliant 12.5 percent return on equity. That is the magic that produced LTCM's 40-percent-plus years. But run the same arithmetic with a 0.5 percent *loss*:

```
gross loss   = 0.5% x $25 = $0.125 per $1 of equity, a 12.5% hit
a 4% loss    = 4% x $25 = $1.00 per $1 of equity, total wipeout
```

The intuition: leverage is perfectly symmetric. The exact machine that turns a half-percent edge into a 12 percent gain turns a 4 percent loss into a total loss of capital - and markets deliver 4 percent moves far more often than the models assumed.

For three and a half years, the machine ran beautifully. LTCM returned roughly 20 percent in its partial first year, then about 43 percent and 41 percent (net of fees) in 1995 and 1996, then about 17 percent in the harder year of 1997. It had grown so flush with capital that in late 1997 it actually returned about \$2.7 billion to outside investors - it had more money than it had good trades for, so it shrank its equity base while keeping its positions roughly the same size. That decision, sensible-looking at the time, quietly pushed its leverage *higher* going into 1998. The fund was now making the same bets on a thinner cushion.

It is worth pausing on why the banks were so eager to lend. LTCM was the most prestigious client on the street, and its trades generated enormous fees and flow for the dealers who financed them. Each bank could see only its own slice of LTCM's book, not the whole, and each assumed the others were applying the same discipline it was. The fund used this fragmented view deliberately: it played lenders against one another to secure near-zero haircuts and the most generous terms in the market. The collective result was that no single institution understood how leveraged LTCM truly was, and the safeguards that should have come from prudent lending - real haircuts, position limits, transparency - were competed away. This is a recurring feature of blow-ups: the discipline that is supposed to come from lenders evaporates when the borrower is glamorous and the lenders are competing for its business. The same dynamic would reappear with Archegos two decades later, when multiple prime brokers each extended huge leverage to a client none of them could see in full.

The fee structure compounded the incentive to lever up. Like most hedge funds, LTCM charged investors a management fee plus a large share of profits - in its case a 2 percent management fee and a 25 percent performance fee, somewhat richer than the standard "2 and 20." Performance fees reward the upside but do not claw back the downside: the partners shared generously in the gains of the good years and did not have to return those fees when the fund collapsed. That asymmetry quietly encourages more leverage, because more leverage means bigger returns in good years and the downside is partly someone else's problem. To be fair to the partners, they had reinvested most of their own fortunes in the fund and lost heavily - but the general incentive structure of leveraged funds tilts toward taking more risk than an outside investor would choose.

## The blow-up, step by step

The trouble began far from LTCM's offices, in the emerging markets.

In the summer of 1997, a currency crisis swept Asia - Thailand, Indonesia, South Korea. Capital fled, currencies collapsed, and investors got their first scare. LTCM weathered 1997 with a modest gain. But the episode planted a seed: investors around the world were learning to be afraid, and fear makes people want safe, liquid assets and shun risky, illiquid ones - the precise opposite of what LTCM's portfolio needed.

Then came the detonator. On August 17, 1998, Russia did something the models treated as almost unthinkable: it defaulted on its domestic government debt and devalued the ruble. A government defaulting on bonds denominated in its own currency was supposed to be nearly impossible - a government can always print its own money to pay. Russia chose not to. The shock was not the size of Russia's economy, which was small; it was the message. If Russia could default, what was truly safe?

The result was a global **flight to liquidity** (also called a flight to quality). Investors everywhere dumped anything risky or illiquid and crowded into the safest, most liquid assets - above all, on-the-run U.S. Treasuries. And here is the cruel mechanism: that is exactly the bond LTCM was *short*. The fund was long the cheap, less-liquid off-the-run bonds and short the rich, super-liquid on-the-run bonds, betting the gap would shrink. Instead, the panic made the liquid bond even more prized and the illiquid bond even more shunned. The gap did not close. It blew wide open.

The contagion ran like a chain reaction.

![Graph of Russia default leading through flight to liquidity to forced selling and losses](/imgs/blogs/ltcm-1998-when-genius-failed-3.png)

Every one of LTCM's "diversified" trades was, underneath, the same bet: that the premium for liquidity and safety would stay normal, and that risky-versus-safe spreads would converge. When the whole world fled to safety at once, every trade moved against the fund simultaneously. Swap spreads blew out. Mortgage spreads blew out. The European convergence trades, the merger-arb book, the volatility positions - all bled together. The diversification was an illusion, because the hidden common factor was liquidity, and liquidity vanished everywhere at once.

The losses came fast. On a single day - August 21, 1998 - LTCM reportedly lost about \$550 million. By the end of August the fund had lost roughly 45 percent of its capital. Over the five weeks following the Russian default, losses totaled about \$4.6 billion. The fund's equity collapsed from about \$4.7 billion at the start of the year toward a few hundred million dollars.

#### Worked example: VaR said "max \$45M/day" - reality was a multiple of that

LTCM's risk models estimated a daily Value-at-Risk in the neighborhood of \$45 million - the loss it expected not to exceed on 99 out of 100 days. That number was built on the volatility of the recent calm years and a bell-curve assumption about how prices move.

Then August 21, 1998 arrived and the fund lost about \$550 million in one day. Compare:

```
model "worst normal day" (99% VaR):  about $45 million
actual loss on Aug 21, 1998:         about $550 million
ratio:                               about 12x the modeled bad day
```

Under a bell curve, a day 12 times worse than the 99th-percentile day is so improbable it should essentially never happen in the life of the universe. It happened in week one of the crisis, and similar days kept coming. The model was not slightly wrong; it was wrong about the *shape* of risk - it assumed thin tails where reality had fat tails.

The intuition: VaR tells you about the ordinary bad day and is silent about the catastrophic one, yet the catastrophic day is the only one that can kill you.

As the losses mounted, a second, faster killer engaged: the leverage spiral. As the assets fell in value, the equity cushion beneath them shrank. But the assets stayed nearly the same size - you cannot instantly sell \$100 billion of bonds. So the *ratio* of assets to equity exploded. A fund leveraged 25 to 1 in January was, by late September, leveraged on the order of 250 to 1 or worse, because the denominator (equity) had nearly vanished while the numerator (assets) had not.

![Before-and-after of LTCM equity and leverage in January versus September 1998](/imgs/blogs/ltcm-1998-when-genius-failed-4.png)

At that point the fund was trapped. It could not raise its bets, and it could not unwind them either. Its positions were so large that trying to sell would crash the very prices it was selling into - and everyone in the market knew LTCM was in trouble, so other traders front-ran its expected sales, pushing prices further against it. The fund's lenders began demanding more collateral (margin calls) as the bonds fell. To meet the calls, LTCM had to sell into a market with no buyers, at fire-sale prices, which locked in more losses, which triggered more margin calls. This is the feedback loop that turns a bad week into an extinction event.

#### Worked example: how a rising haircut becomes a hidden margin call

Suppose LTCM has financed \$10 billion of bonds through repo at a 1 percent haircut, the kind of generous term it enjoyed before the crisis. A 1 percent haircut means the lenders advanced 99 percent of the bonds' value, so LTCM borrowed \$9.9 billion and had to fund only \$100 million itself.

```
bonds posted as collateral:   $10 billion
haircut (pre-crisis):         1%  -> lender keeps $100 million
cash LTCM could borrow:       $9.9 billion
```

Now the panic hits and the lenders, frightened, raise the haircut to 5 percent on the same bonds - even before the bonds have lost a cent of value.

```
same bonds:                   $10 billion
haircut (crisis):             5%  -> lender keeps $500 million
cash LTCM can now borrow:     $9.5 billion
sudden cash shortfall:        $9.9B - $9.5B = $400 million
```

LTCM must come up with \$400 million in cash immediately, on a position whose market value has not even moved yet, simply because the lender got nervous. Multiply this across a \$125 billion balance sheet and a few points of haircut, and the cash demands reach into the billions - cash a fund running on a \$5 billion equity sliver does not have.

The intuition: in a crisis, lenders tighten haircuts at the worst possible moment, and a rising haircut is a margin call in disguise that can drain a leveraged fund of cash before prices have even moved.

#### Worked example: the \$3.6 billion consortium recap split across 14 banks

By mid-September 1998, the Federal Reserve Bank of New York concluded that a disorderly LTCM failure could cascade through the banks that had financed and traded with it. The Fed did not lend its own money. Instead, it convened the heads of LTCM's major creditors and brokered a private rescue. On September 23, 1998, a consortium of 14 financial institutions agreed to inject about \$3.6 billion in exchange for 90 percent ownership of the fund, taking control to wind it down in an orderly way.

Most of the 14 contributed roughly \$300 million to \$350 million each. A rough split:

```
total recapitalization:    about $3.6 billion
participating banks:       14
average per bank:          $3.6B / 14 = about $257 million
the big contributors:      about $300-350 million each
```

Bear Stearns, notably, declined to put in equity (it had been LTCM's clearing agent and pushed back hard). Two firms that had been invited to a separate earlier buyout attempt - and a few others - did not join the final group on equal terms. The point of spreading it across so many banks was that each could absorb its slice, whereas a sudden default would have hit several of them with concentrated, simultaneous losses.

The intuition: the rescue was the banks bailing out their own exposure to LTCM, coordinated by the Fed - which is very different from a government writing a check.

The Fed-organized consortium recapitalized the fund on September 23, 1998. The positions were unwound over the following year, the original investors were nearly wiped out, and LTCM was closed for good in early 2000.

## The mechanism dissected: why genius was not enough

It is tempting to file LTCM under "the models were wrong." That is too easy and mostly false. The deeper mechanism is an interaction of four forces, each individually manageable, that together were lethal.

### Force 1: leverage amplifies a small move into a fatal one

We have already done the arithmetic, but it bears restating as a principle. At 25-to-1 leverage, your equity is a 4 percent buffer against the value of your assets. A diversified bond book does not usually move 4 percent against you - but "usually" is doing enormous work in that sentence. The whole danger of high leverage is that it converts a *survivable* market move into a *terminal* one. LTCM's positions, taken individually, were sound bets that would have made money if held for a couple of years. The fund simply did not have the capital to survive the path between here and there. Leverage shrinks the distance between "temporarily underwater" and "out of business" to almost nothing.

There is a useful way to think about this that risk managers now call "survival to the destination." A trade has two separate questions attached to it: is the destination right (will the spread eventually converge?), and can I survive the path (will an adverse move force me out before I get there?). An unleveraged investor only has to get the destination right; they can ride out any amount of temporary drawdown because no one can force them to sell. A leveraged investor has to get *both* right, and the second is far harder, because the path is governed by the panic of strangers, not by fundamentals. LTCM's destinations were almost all correct. Its paths killed it. The more leverage you carry, the more your fate depends on the path rather than the destination - and the path is the thing no model can predict.

This also explains why "being early" and "being wrong" are financially identical under leverage. If you put on a convergence trade and the spread widens for three months before it narrows, an unleveraged investor merely feels uncomfortable; a 25-to-1 investor gets margin-called into oblivion and never sees the narrowing. The market can stay irrational longer than a leveraged fund can stay solvent. LTCM had identified real mispricings, but it had no mechanism to guarantee it would still exist when the mispricings corrected, and so the correctness of its analysis was, in the end, irrelevant.

### Force 2: crowded trades and the no-exit problem

LTCM's strategies were not secret. Its alumni and imitators - the proprietary trading desks at the big banks, other arbitrage funds - ran versions of the same trades. So when the spreads moved against LTCM, they moved against everyone running those trades at once. And when those funds and desks tried to cut their risk, they were all selling the same positions at the same time. A crowded trade has a hidden property: the exit is only wide enough for one. When everyone heads for it together, the price gaps against all of them, and there is effectively no liquidity to get out. LTCM was not just betting on convergence; it was betting that it would never have to sell in a hurry. It was forced to, and discovered there was no one to sell to.

### Force 3: VaR assumed a calm world that no longer existed

The risk model had two specific blind spots. First, it assumed price changes followed a roughly normal distribution - thin tails - when markets have fat tails, so it radically underestimated the chance of an extreme day. Second, it estimated future volatility from *recent* volatility, and the mid-1990s had been unusually calm. Calibrating risk on a calm period is like judging how high a river can rise by measuring it during a drought. When the storm came, the model's "1-in-100-year" loss arrived in the first week, and kept arriving. VaR is a useful tool for the ordinary day; it gives a dangerously false sense of safety about the extraordinary one, and the extraordinary one is the only one that ends you.

### Force 4: correlations go to 1 in a crisis

This is the subtlest and most important point. LTCM believed its book was diversified across dozens of unrelated trades in different countries and markets, so that even if some lost, others would gain. In normal times that was roughly true; the correlations between, say, Danish mortgage spreads and U.S. swap spreads were low. But a liquidity crisis introduces a single dominant factor that overrides everything: the universal flight from risk to safety. Suddenly every risky-versus-safe spread is the same trade. The independent bets become one giant bet on "calm continues," and that bet loses everywhere at once. The matrix below lays out the strategies, what each quietly assumed, and how each failed.

![Matrix of LTCM strategies, their hidden assumptions, and how each failed](/imgs/blogs/ltcm-1998-when-genius-failed-6.png)

The deep lesson is that diversification protects you against *idiosyncratic* risk - one company, one country going bad - but not against *systematic* risk, the common factor that hits everything. Leverage assumes your diversification is real. When the correlations jump to 1, the diversification evaporates exactly when the leverage needs it most. That combination - high leverage resting on assumed-low correlations that spike in a crisis - is the structural flaw at the heart of nearly every great blow-up since.

### Why it was systemic, not just sad

A normal fund blowing up is a private tragedy for its investors. LTCM was different because of who stood on the other side of its trades. The fund had borrowed roughly \$120 billion through repo from a web of dealer banks and held its \$1.25 trillion derivatives book with those same banks as counterparties. If LTCM defaulted, those banks would simultaneously (a) be left holding collateral they would have to dump into a falling market, and (b) find their derivative hedges suddenly unhedged. Several major banks faced concentrated, correlated losses at the same moment - the recipe for a chain of failures.

![Graph of LTCM counterparty web pulling in 14 banks and the Fed](/imgs/blogs/ltcm-1998-when-genius-failed-7.png)

That is why the Federal Reserve got involved. The Fed had no legal obligation and put up no money, but it had a clear interest in preventing a disorderly unwind that could freeze lending across Wall Street. By gathering the creditors in one room, it solved a coordination problem: each bank would have preferred someone else to bear the cost of saving LTCM, and left alone they might have raced to liquidate first, guaranteeing the crash. The Fed's role was to make them act together. This was the first vivid, public demonstration of what would later be called "too interconnected to fail."

The coordination problem deserves a closer look, because it is the heart of why systemic crises require a coordinator at all. Each of LTCM's lenders faced the same logic: if LTCM is going to fail, I want to seize and sell my collateral *first*, before the others flood the market and prices collapse. But if all 14 banks act on that logic simultaneously, they all dump collateral at once, the prices crash for everyone, and each bank realizes far larger losses than if they had waited and unwound in an orderly way. It is a classic run - rational for each individual, ruinous for the group. No single bank could fix it, because any bank that held back while the others sold would simply be the one left holding the worthless collateral. Only an outside party with the authority to put everyone in the same room and say "we move together or not at all" could break the trap. That is the precise function the Fed served: not lender, but coordinator. The same logic explains bank-run interventions, deposit insurance, and the 2008 rescues - the problem is almost never that the assets are worthless, but that a disorderly stampede to exit makes them temporarily worthless for everyone.

There is a darker side to the rescue that critics raised at the time and that history has vindicated. By organizing a save, even a private one, the Fed arguably signaled to the market that very large, very interconnected funds would not be allowed to fail in a disorderly way - a belief that may have encouraged exactly the kind of leveraged, interconnected risk-taking that produced 2008. Economists call this "moral hazard": when actors believe they will be rescued, they take more risk. Whether the LTCM rescue was the right call remains debated. It almost certainly prevented an immediate crisis; it also may have taught the wrong lesson to a market that was about to repeat the experiment with ten times the leverage and no consortium large enough to absorb it.

## The aftermath: the wind-down and the lessons

After the September 23 recapitalization, the consortium took 90 percent of the fund and installed oversight. The portfolio was unwound slowly over 1998 and 1999. Critically, many of LTCM's trades *did* eventually converge - the spreads narrowed again once the panic faded, and the consortium actually recovered its \$3.6 billion and ended up roughly whole, even slightly ahead. This is the bitter irony: the fund's bets were largely correct. It was right about *where* prices would end up and catastrophically wrong about whether it could survive the journey. Being right too early, with too much leverage, is indistinguishable from being wrong.

The original LTCM investors were not so lucky. The fund's equity had fallen from a peak around \$4.7 billion to under \$400 million before the rescue diluted them to almost nothing. People who had invested their fortunes - including many of the partners themselves, who had reinvested their earnings and even borrowed to buy more of the fund - lost most of their wealth. The partners' personal stakes, worth roughly \$1.9 billion at the start of 1998, were largely destroyed. Meriwether went on to start new funds; one of them, JWM Associates, was itself badly hurt in the 2008 crisis. The pattern repeats.

The lessons that risk managers drew - and the ones they should have drawn but did not fully internalize until 2008:

- **Leverage is a ruin risk, not just a return amplifier.** A position that is correct on average can still bankrupt you if leverage means a normal adverse path forces you out before you are proven right.
- **Liquidity is a risk factor, not a given.** Models must account for the fact that you may be unable to sell, or able to sell only by crashing the price, exactly when you most need to.
- **VaR is necessary but not sufficient.** Firms added "stress tests" and "scenario analysis" that ask "what if the worst historical event repeated, and worse?" rather than only "what is a normal bad day?"
- **Correlations are not stable.** Risk models began to assume that in a crisis, correlations across risky assets jump toward 1, erasing the benefit of diversification.
- **Concentration and crowding are hidden leverage.** If your trade is everyone's trade, your true risk is far larger than your own position implies, because the exit will be jammed.

And yet, ten years later, the financial system did almost exactly the same thing on a far larger scale. LTCM was a preview of 2008: extreme leverage (this time at the banks themselves and at funds), assets that were liquid until they suddenly were not (mortgage securities), risk models that assumed a benign distribution (housing prices "never fall nationally"), and correlations that all went to 1 when the panic hit. If you want the full anatomy of the larger sequel, see [Lehman Brothers and the 2008 financial crisis](/blog/trading/finance/lehman-brothers-2008-financial-crisis). The mechanism is identical; only the zeros are different.

## Common misconceptions

**"The models were just wrong - they were arrogant quants who didn't understand markets."** This is the popular caricature and it is mostly false. LTCM's models were sophisticated, and its core trades were largely correct - the spreads it bet on did eventually converge, and the rescue consortium recovered its money. The error was not in the direction of the bets but in the assumptions about extreme events and liquidity, and above all in the leverage that left no room to survive a temporary adverse move. "The model was wrong" lets you off too easy; the more uncomfortable lesson is that a correct model plus too much leverage still kills you.

**"Diversification protected them - they had dozens of unrelated trades."** It did not, and this is the most important correction. The trades looked unrelated in calm markets, but they shared one hidden common factor: a bet that liquidity premiums and risk spreads would stay normal. When the world fled to safety, every trade lost at once. Diversification defends against company-specific or country-specific risk; it does nothing against a systematic shock that hits all risky assets together, which is precisely the shock that arrives in a crisis.

**"The Fed bailed out LTCM with taxpayer money."** No public money was used. The Fed organized a meeting and applied pressure, but the \$3.6 billion came entirely from 14 private banks, who did it to protect their own exposure to the fund. The distinction matters: this was a coordinated private rescue, not a government bailout. (The 2008 bank bailouts, by contrast, did use public funds - another reason not to blur the two episodes.)

**"Smart people don't blow up - this was a fluke."** The opposite is closer to true. Extreme intelligence can be a risk factor, because it breeds confidence, and confidence invites leverage. The aura of Nobel laureates is part of why banks lent so freely with such thin cushions. Blow-ups are not caused by stupidity; they are caused by leverage meeting an event the leveraged party was sure could not happen. The smartest people are perfectly capable of being sure of the wrong thing.

**"It was a one-day Russian event - bad luck."** Russia was the trigger, not the cause. The cause was a portfolio so leveraged and so crowded that *some* shock was eventually going to force it to sell into illiquidity. If not Russia in 1998, something else later. Calling it bad luck misses that the fund had engineered a structure where a fairly ordinary crisis would be fatal. The vulnerability was built in; the trigger was almost incidental.

**"Notional value means LTCM was risking \$1.25 trillion."** The trillion-dollar figure is the face value of the derivatives, not the money at risk. Most of those derivatives were hedged against each other, so the net economic exposure was a small fraction of the notional. The notional number is dramatic and worth knowing because it shows the fund's footprint in the plumbing of the market - and thus its systemic reach - but treating it as the loss exposure misunderstands how derivatives work.

## How it echoes in other markets

The LTCM mechanism - leverage plus crowding plus a liquidity shock that pushes correlations to 1 - is one of the most durable patterns in finance. It keeps recurring, with new instruments and new names but the same skeleton.

**The 2007 quant quake (August 2007).** In a single week of August 2007, many quantitative equity hedge funds - statistical-arbitrage strategies that, like LTCM, ran similar crowded bets with leverage - suffered sudden, severe, simultaneous losses. The trigger appears to have been one large fund deleveraging, which moved the crowded factors against everyone running the same models. Funds that looked diversified and market-neutral lost double-digit percentages in days, then mostly recovered within weeks - a near-perfect miniature of the crowding-plus-leverage trap. It was a warning shot, one year before the main event.

**The 2008 global financial crisis.** This is LTCM at system scale. Investment banks ran leverage of 30 to 1 or more; Lehman Brothers was leveraged roughly 31 to 1. The "safe, liquid" assets were AAA-rated mortgage securities, which turned out to be illiquid and not safe at all. The risk models assumed national house prices would not fall together; the correlations went to 1 when they did. The flight to liquidity froze the repo market - the same plumbing LTCM had relied on. The difference was scale: this time no consortium could absorb it, and governments had to step in with public money. The mechanism, though, is the one LTCM taught and the world failed to learn.

**Amaranth Advisors (2006).** A multistrategy hedge fund that grew enormously concentrated in natural-gas futures through one star trader. When gas prices moved sharply against its leveraged spread positions in September 2006, Amaranth lost about \$6.6 billion in a matter of days - more in absolute dollars than LTCM - and collapsed. The lesson rhymes: a leveraged, concentrated position in a market too thin to exit at scale, undone by a move the trader was sure would not happen. Because Amaranth was less interconnected with the banking system than LTCM, its failure did not threaten the system, and no rescue was organized; its positions were sold to other firms.

**Archegos Capital Management (2021).** A family office run by Bill Hwang built giant, concentrated, leveraged bets on a handful of stocks - but did it through total-return swaps with multiple prime brokers, so no single bank could see the whole position. Hidden leverage of roughly 5 to 1 or more on a concentrated book meant that when a couple of the stocks fell, the margin calls came all at once. The banks raced to liquidate, the dumping crashed the stocks further, and the losses cascaded. Archegos lost essentially everything, and its banks lost over \$10 billion combined - Credit Suisse alone took about \$5.5 billion. It is LTCM's crowding-and-no-exit problem, modernized with swaps that concealed the leverage from the very lenders providing it.

**The March 2020 Treasury basis-trade scare.** In the early days of the COVID-19 panic, a popular hedge-fund trade - a highly leveraged arbitrage between Treasury bonds and Treasury futures, the "basis trade," often levered 50 to 1 or more - came under acute stress. As volatility spiked and margin calls hit, funds were forced to sell Treasuries, the supposedly safest and most liquid asset on the planet, into a market that briefly could not absorb the flow. The world's deepest market seized up, and the Federal Reserve had to intervene massively to restore function. The echo of LTCM is exact: extreme leverage on a tiny convergence spread, in a "safe" market, undone by a liquidity shock that forced selling when there were no buyers. Regulators continue to watch this trade closely for precisely this reason.

**Carry trades and the 1998 ruble link.** The same flight to safety that killed LTCM unwound leveraged currency "carry trades" - borrowing in a low-yield currency like the yen to invest in higher-yield assets. When risk appetite collapses, these trades reverse violently and all at once, because everyone is in the same position. The yen carry-trade unwinds of 1998, of 2008, and again in August 2024 each show the LTCM skeleton: leverage, crowding, a shock, and a stampede for an exit too narrow for the crowd.

The through-line in every one of these is the same. The specific trade does not matter. What matters is the combination: borrow a lot, against a small mispricing, in a position lots of other people also hold, in a market that is liquid until it is not. Sooner or later a shock arrives, the correlations jump to 1, and the leverage forces you to sell into a vacuum. LTCM was the cleanest, most instructive instance because the people running it were the best in the world - which proves the failure was structural, not personal.

## When this matters to you and further reading

You are unlikely to run a \$125 billion bond book, but the LTCM lesson scales all the way down to a personal balance sheet, and it is one of the most useful mental models in all of finance.

Whenever you use borrowed money - a margin account, a leveraged ETF, options, a mortgage on an investment property, crypto bought on leverage - you are running a small LTCM. The questions to ask are exactly the ones LTCM got wrong. First: if the thing I own moves against me by a normal-but-bad amount, can I survive it, or am I forced to sell at the worst possible moment? Second: is my "safe, diversified" position actually a single bet in disguise - do all my holdings depend on the same thing (low rates, a rising market, continued calm)? Third: in a panic, will I be able to get out, or will everyone be trying to leave through the same door? If the honest answers are "I'd be forced out," "it's secretly one bet," and "the door will be jammed," then you have rebuilt LTCM in miniature, and you should reduce your leverage before the world reminds you why.

For practitioners and serious students, the durable takeaways are: hold enough capital to survive the path, not just to be right about the destination; treat liquidity as a first-class risk that disappears in crises; never trust a risk number that assumes a normal distribution and recent calm; and assume your diversification will fail at the exact moment you are counting on it. The market does not care how smart you are. It cares whether you can meet the next margin call.

To go deeper into the surrounding system, three companion pieces fit naturally here. To understand the fees, leverage, and incentives of the industry LTCM helped define, read [How hedge funds work: leverage, "2 and 20", and the carry](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20). To see the banks that sat on the other side of LTCM's trades and how they earn and risk their capital, read [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money). And to understand the institution that organized the rescue and sets the price of the borrowed money that makes leverage possible, read [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

The best long-form account of the episode remains Roger Lowenstein's book "When Genius Failed: The Rise and Fall of Long-Term Capital Management," which reads like a thriller and is the source most of the figures here trace back to. The Federal Reserve's own staff studies of the September 1998 events, and the later President's Working Group report on hedge funds and leverage, are the dry but authoritative complement. Read them together and one sentence will stay with you: the fund was right about almost everything except whether it would still be standing when it was proven right.

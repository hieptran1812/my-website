---
title: "Japan and the Zero Bound: ZIRP, QE, and YCC"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The 30-year case study of policy at the zero bound — how the Bank of Japan invented every modern unconventional tool, what each one did to bonds, the yen, and stocks, and the exit that finally came in 2024."
tags: ["monetary-policy", "central-banks", "bank-of-japan", "zero-bound", "quantitative-easing", "yield-curve-control", "negative-rates", "carry-trade", "japanese-equities", "asset-valuation"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — When a central bank cuts its policy rate to zero and the economy still won't move, the rate stops working as a lever — so the Bank of Japan spent 30 years inventing one new tool after another (ZIRP, QE, the QQE "bazooka," negative rates, yield curve control), and the whole experiment is the world's textbook on what policy can and cannot do at the zero bound.
>
> - The Nikkei 225 peaked at **38,957** on 29 December 1989 and did not close above that level again until **2024** — a 35-year round trip that defines what a burst asset bubble plus deflation can do.
> - At the zero bound the BOJ stopped targeting a *rate* and started targeting *quantities*: it bought bonds until it owned **over half** of the entire Japanese government bond (JGB) market, plus the largest equity ETF book in the country.
> - Near-zero yen became the world's funding currency. The **yen carry trade** — borrow cheap yen, buy a higher-yielding foreign asset — worked for years and then violently unwound in **August 2024** when the BOJ finally started hiking.
> - **The one number to remember:** ¥-trillions of newly created base money still could not lift Japanese inflation to its 2% target for two decades. Printing money is *necessary* but not *sufficient* — the lesson every other central bank studied before its own QE.

On 9 April 2013, a new Bank of Japan governor named Haruhiko Kuroda did something central bankers almost never do: he made a promise so big it sounded reckless. He pledged to *double* the amount of base money in the Japanese economy in two years — to buy government bonds at a pace of ¥60–70 trillion a year, hoover up risky assets, and keep going until inflation hit 2%. The press called it the "bazooka." The yen fell, the Nikkei ripped higher, and for a moment it looked like Japan had finally found the lever to escape two decades of deflation.

It didn't fully work. That is the first thing to understand about this story, and it is the most important. Japan threw the largest peacetime monetary experiment in history at a stagnant economy — more aggressive, relative to the size of the economy, than anything the United States or Europe ever attempted — and it still spent most of the next decade below its inflation target. The tools moved asset prices enormously. They moved the *real economy* far less. The gap between those two facts is the whole subject of this post.

Japan matters far beyond Japan because it went first. Every unconventional tool the Federal Reserve, the European Central Bank, and the Bank of England reached for after 2008 — zero rates held for years, large-scale bond buying, forward guidance, negative rates, even yield caps — the Bank of Japan had already tried, often a decade earlier. Japan is the laboratory. When you read about "QE" or "the zero lower bound," you are reading about ideas that were stress-tested in Tokyo first. This post walks that laboratory end to end: how each tool works, what it did to JGBs, the yen, and Japanese stocks, and what the whole 30-year arc teaches about the limits of policy.

![Five rung ladder of BOJ tools from ZIRP to QE to QQE to NIRP to YCC with a green exit box](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-1.png)

The mental model is the ladder above, and it runs through the entire post. A central bank's normal lever is a single short-term interest rate. When that rate is cut all the way to zero and the economy is still weak, the lever is jammed — you cannot cut below zero by very much, and you have run out of room. So the central bank climbs the ladder, reaching for a new and stranger tool at each rung: first quantity (QE), then a bigger quantity faster (QQE), then a *negative* rate (NIRP), then fixing the price of a longer-term bond directly (YCC). Each rung is added when the rung above it stops moving the economy. And after 25 years of climbing, the final, hardest move was getting *back down* — the 2024 exit. Hold this picture; every section fills in one rung.

## Foundations: what the zero bound is and why it breaks the normal lever

Start from zero, literally. A central bank's main job, in the modern framework, is to set one very short-term interest rate — in the US it is the federal funds rate, in Japan the overnight call rate. By controlling the price of overnight money, the central bank ripples its influence out across every other interest rate and asset price in the economy. We covered the full machinery in [the monetary toolkit](/blog/trading/policy-and-markets/the-monetary-toolkit-rates-qe-qt-and-forward-guidance) and how a rate change physically reaches markets in [the discount-rate channel](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows); here we only need the one idea that breaks.

When the economy is too hot — prices rising too fast — the central bank *raises* its rate. Borrowing gets more expensive, spending and investment cool, inflation comes down. When the economy is too cold — weak growth, falling prices — it *cuts* the rate. Borrowing gets cheaper, spending picks up, the economy warms. The rate is a thermostat, and for most of modern history it has plenty of room to move in both directions.

Now picture an economy so cold that the central bank cuts and cuts and cuts — and arrives at **zero**. This is the **zero lower bound** (often just "the zero bound"). Why can't it keep going? Because of a humble piece of paper: **physical cash**. Cash pays an interest rate of exactly 0% — a ¥10,000 note in your drawer is still ¥10,000 next year. If a bank tried to charge you a deeply negative rate, say −5%, for holding your money in a deposit, you would simply withdraw it all as banknotes and store them in a vault, earning your guaranteed 0%. Cash is an escape hatch. It puts a floor — *roughly* zero, a hair below in practice — under how negative interest rates can go before everyone flees into physical currency. (Storing and insuring billions in banknotes has costs, which is why a *slightly* negative rate like −0.1% is survivable; −5% is not.)

So at the zero bound, the thermostat is stuck at its lowest setting and the room is still too cold. The price lever — cutting the rate — has run out of room. The central bank's job now is to find a *different* lever. That search is exactly what Japan ran, in real time, for thirty years, while the rest of the world watched.

### Why Japan got there first: the bubble and the bust

![Nikkei 225 index level from the 1989 peak of 38957 down to the 2009 low of 7055 and back](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-2.png)

Japan reached the zero bound first because it had the biggest asset bubble of the late 20th century, and the biggest bust. Through the 1980s, ultra-loose policy — partly a consequence of the 1985 [Plaza Accord](/blog/trading/policy-and-markets/the-plaza-accord-1985-engineering-a-weaker-dollar), which pushed the yen sharply higher and led Japan to cut rates to cushion its exporters — inflated stocks and land to absurd levels. At the December 1989 peak, the Nikkei 225 closed at **38,957**. The land under the Imperial Palace in Tokyo was, by some estimates, worth more than all the real estate in California. People genuinely believed prices only went up.

Then it broke. The BOJ raised rates to pop the bubble, and the air came out fast. The Nikkei fell for years. By the 2009 trough it touched roughly **7,055** — a decline of about 82% from the peak. Land prices fell for two decades. And here is the part that made Japan the zero-bound laboratory: the bust did not produce an ordinary recession that healed in a couple of years. It produced **deflation** — a sustained, gentle *fall* in the general price level — that lasted, on and off, for the better part of 20 years.

Deflation is the zero bound's evil twin, and it is worth understanding why, because it is the trap the BOJ spent decades trying to escape. When prices are falling, holding cash *gains* you purchasing power for free — your money buys more next year. So why spend or invest today? Households delay purchases; firms delay investment; everyone waits for things to get cheaper, which makes demand weaker, which makes prices fall further. It is a self-reinforcing spiral, and it is brutal for debtors: if you owe a fixed ¥10 million and wages and prices are falling, that debt gets *heavier* every year in real terms.

The cruelest part is what deflation does to the central bank's own lever. The interest rate that actually drives borrowing and investment decisions is not the *nominal* rate you see quoted — it is the **real** rate, the nominal rate minus expected inflation. A 5% nominal rate when inflation is 3% is only a 2% real cost of borrowing; the same 5% nominal rate when prices are *flat* is a full 5% real cost. Now run that logic at the zero bound with deflation. Suppose the BOJ has cut the nominal rate to **0%** but prices are falling at **−1%** a year. The real rate is:

```
real rate  =  nominal rate  -  inflation
           =  0%  -  (-1%)
           =  +1%
```

The bank has cut as far as the nominal lever allows — all the way to zero — and the *real* cost of money is still a positive, economy-cooling **+1%**. It is, in real terms, holding policy *tight* while desperately trying to ease. This is the deflation trap in one line of arithmetic, and it is why the BOJ's entire 30-year project was, at bottom, a fight to lift inflation *expectations* — to flip that sign — when the conventional rate lever had no room left. Everything that follows is the BOJ trying to climb out of that box.

## ZIRP: cutting the rate to the floor (1999)

The first rung is the simplest: cut the rate all the way down and hold it there. The BOJ formally adopted a **Zero Interest Rate Policy (ZIRP)** in February 1999, guiding the overnight call rate to essentially 0%. This is still "conventional" policy — it is just the price lever pushed to its limit.

![BOJ policy rate step chart pinned near zero from 1999 then negative in 2016 then 0.50 percent in 2025](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-4.png)

The step chart above is the spine of the whole story: a policy rate flattened against the zero line for a quarter-century, dipping *below* it in 2016, and finally lifting off in 2024–25. Notice the small bump in the mid-2000s — the BOJ briefly tried to exit, raising the rate toward 0.25–0.5% in 2006–2008, only to slam back to zero when the global financial crisis hit. That failed exit matters: it taught the bank (and everyone watching) that leaving the zero bound is *much* harder than arriving, a lesson that would haunt the 2024 exit two decades later.

ZIRP alone did not fix deflation, and the reason exposes the core problem. A central bank promising "rates are zero *right now*" is not very powerful if everyone expects rates to go back up the moment the economy improves. What actually moves long-term borrowing costs and investment decisions is the *expected path* of rates over years — the **expectations channel**, which we cover in depth in [forward guidance and credibility](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility). So the BOJ paired ZIRP with a promise: it would hold rates at zero "until deflationary concerns are dispelled." That promise — talking about the *future* path of policy — was one of the first uses of what we now call **forward guidance**. Japan invented that too.

#### Worked example: why a 0% rate makes a long bond so dangerous to hold

To understand everything that follows — especially why the BOJ had to buy *unlimited* amounts of bonds under YCC — you have to feel how a bond at a near-zero yield behaves. Let's price one.

Take a Japanese government bond (a JGB): a 10-year bond with a **0.1%** annual coupon, face value **¥100** (about **\$0.65** at ¥154 to the dollar — and yes, a real JGB trades in millions, but the per-¥100 math scales perfectly). When the market yield equals the coupon, the bond is worth par, ¥100. Now suppose the market yield rises by just **half a percentage point**, from 0.1% to 0.6%.

The price of a bond is the present value of its cash flows discounted at the market yield. A useful shortcut is **duration**: the approximate percentage price change is roughly minus the duration times the change in yield. A 10-year bond at a near-zero coupon has a duration close to its maturity — call it about **9.9 years**. So:

```
price change  ~  - duration  x  change in yield
              ~  - 9.9  x  (+0.50%)
              ~  - 4.95%
```

The bond falls from ¥100 to about **¥95.05** — a loss of nearly ¥5, or about **\$0.032 per ¥100 of face**, from a yield move of only half a point. Scale that to a ¥100 million holding and you have lost roughly **¥4.95 million** (about **\$32,000**). The lesson: when the *coupon* is tiny, almost all of a long bond's value sits in getting the face value back years from now, so its price is *exquisitely* sensitive to yields. A move that would barely dent a high-coupon bond can gut a near-zero one. **This is why pinning a low long-term yield is so expensive — the tiniest upward drift in yield inflicts large losses, and someone with infinitely deep pockets has to keep buying to stop it.**

## QE: when you can't move the price, move the quantity (2001)

ZIRP set the *price* of overnight money to zero. But the central bank can also act on *quantities*. In March 2001 the BOJ took the next rung and launched the world's first modern **quantitative easing (QE)** program. (For the general mechanism — applied later to the Fed and ECB — see [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) and [the liquidity channel](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid).)

Here is the mechanic in plain terms. The central bank creates new money — **bank reserves**, electronic central-bank money — and uses it to buy assets, mostly government bonds, from banks and investors. The seller hands over a bond and receives freshly created reserves. The point is no longer to set a particular interest rate; the bank's overnight rate is already zero. The point is to flood the banking system with so much money that something — bank lending, asset prices, inflation expectations — finally stirs. The BOJ explicitly shifted its operating target from the *price* of money (the rate) to the *quantity* of money (the level of bank reserves). That is the conceptual leap of QE: from price-targeting to quantity-targeting.

### The transmission: how QE is supposed to reach the economy

QE is meant to work through several channels at once, and it is worth naming them because Japan tested every one:

- **Portfolio rebalancing.** The central bank buys safe bonds, leaving investors holding cash that earns nothing. To get a return, they are pushed *out the risk curve* — into corporate bonds, stocks, foreign assets. Prices of risk assets rise. This is the channel that works most reliably.
- **Lower long-term yields.** By buying long bonds, the bank pushes their prices up and yields down, lowering borrowing costs for governments, mortgages, and companies.
- **The signal.** A huge bond-buying program tells markets the central bank is serious about keeping policy loose for a long time — reinforcing forward guidance.
- **A weaker currency.** More yen sloshing around, lower yen yields — the yen tends to fall, helping exporters and importing some inflation. (More on the currency channel below, and in [rate differentials and carry](/blog/trading/policy-and-markets/the-currency-channel-rate-differentials-and-carry).)

Of these, **portfolio rebalancing** is the workhorse, so it is worth making concrete. Imagine you run a Japanese pension fund and you have always held safe JGBs yielding 2%. The BOJ buys up JGBs, driving their yield to 0%. Your old comfortable income is gone — a 0% JGB can no longer fund your retirees' payouts. You face a choice: accept zero, or move into something riskier that still pays. So you sell some JGBs to the BOJ and buy corporate bonds, dividend-paying stocks, foreign bonds, real estate. So does every other yield-starved investor in the country, all at once. That collective shove *up the risk curve* bids up the prices of every risk asset — which is exactly why QE so dependably lifts stocks and credit even when it does nothing for consumer prices. The central bank doesn't have to buy the stock directly (though the BOJ eventually did); it just has to make the safe alternative unbearable, and the market reallocates itself.

Notice what is *missing* from that list: a direct, reliable line from "more bank reserves" to "more bank lending to real businesses" to "more inflation." That link is the weakest one, because it depends on someone *wanting to borrow*, which a central bank cannot force. You can lead a horse to water; you cannot make a deleveraging Japanese corporation take out a loan it does not want. That broken link is the whole reason QE reflates portfolios but not paychecks, and Japan is the proof.

#### Worked example: QE without inflation — why ¥-trillions didn't lift CPI

![BOJ monetary base tripling while Japan core CPI stays near zero around the 2 percent target line](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-6.png)

This is the single most important worked example in the post, because it is the lesson the whole world took from Japan. Let's trace the money.

Under Kuroda's QQE (the supercharged QE we'll meet next), the BOJ's **monetary base** — the central-bank money it directly creates — went from about **¥138 trillion** at the end of 2012 to roughly **¥590 trillion** by 2020. That is a **¥452 trillion** increase, more than a **4x** expansion, equal to over **\$3 trillion** at ¥150 to the dollar — created from nothing and pushed into the banking system. By the crude "printing money causes inflation" story, prices should have exploded.

They didn't. Japan's core inflation, as the red line shows, barely cleared the 2% target even once (the 2014 blip was largely a one-off consumption-tax hike, not underlying demand), and spent most of the decade between roughly 0% and 1%. Why?

The answer is a balance-sheet identity. When the BOJ creates ¥1 of reserves to buy a bond, that ¥1 lands in a commercial bank's reserve account *at the BOJ*. It becomes spending in the real economy **only if the bank lends it out and a business borrows and spends it**. In a deflationary, ageing economy where firms were paying down debt rather than borrowing and households were saving, banks had no profitable loans to make. So the reserves just *sat there*, piled up at the central bank, inert. The monetary base ballooned; the broader money that households and firms actually spend grew slowly; and prices stayed flat.

```
+Y452 trillion base money   (BOJ creates it, buys bonds)
      |
      v
sits in bank reserve accounts at the BOJ   <-- the leak
      |
      x  (banks find few borrowers; firms deleverage)
      v
~flat growth in money households actually spend
      |
      v
~flat CPI  (target missed for ~20 years)
```

The intuition to carry away: **base money is the fuel, but lending is the engine, and you cannot make a stalled engine run by pouring in more fuel.** QE reliably lifts asset prices (the portfolio-rebalancing channel always works — there are always investors to push out the risk curve). It does *not* reliably lift inflation, because that requires the private sector to *want* to borrow and spend. Japan proved this with the largest QE in history, and it is why the Fed and ECB, when they did their own QE after 2008, lifted markets enormously while struggling for years to hit 2% inflation. Markets are downstream of liquidity; inflation is downstream of demand, and liquidity does not create demand by itself.

## QQE: the bazooka (2013)

By 2013, Japan had been at the zero bound for 14 years and QE had been running, on and off, for over a decade, and *still* deflation lingered. The BOJ concluded the problem was partly psychological: people simply did not *believe* inflation would ever come back, so they behaved in ways that kept it away. The fix, the new leadership decided, was **shock and awe** — a program so enormous and so explicitly committed to 2% that it would jolt expectations.

That was **Quantitative and Qualitative Easing (QQE)**, launched April 2013, the monetary centerpiece of what was branded "Abenomics" after Prime Minister Shinzo Abe — a three-part program of aggressive monetary easing, flexible fiscal spending, and structural reform (the "three arrows"). The first arrow, QQE, was the loud one. "Quantitative" meant doubling the monetary base in two years. "Qualitative" meant changing *what* the bank buys: extending purchases out to much longer-maturity JGBs (pulling down the entire yield curve, not just the short end) and, most strikingly, buying *risky* assets directly — equity ETFs and J-REITs (exchange-traded property funds). No major central bank had ever made buying its own stock market a standing policy. The pace was staggering: ¥60–70 trillion of JGBs a year initially, later raised to ¥80 trillion. To put that in proportion, the Fed's largest QE programs ran at a scale that, relative to the size of each economy, QQE comfortably exceeded. This was the most aggressive peacetime monetary easing any developed country had ever attempted.

The "qualitative" word is doing quiet but enormous work, so sit with it. By buying long bonds, the BOJ was deliberately taking *duration risk* off the market's hands — absorbing the interest-rate risk that private investors would otherwise demand to be paid for, and thereby flattening the yield curve. By buying equity ETFs, it was taking *equity risk* off the market's hands and putting a price-insensitive bid under the Nikkei. Each purchase pushed private investors further out the risk curve (the portfolio-rebalancing channel), and the cumulative effect, over years, was to turn the central bank into the marginal buyer of both Japan's bonds and a meaningful slice of its stocks. That choice — to ease "qualitatively," not just "quantitatively" — is what eventually made the BOJ *the market*, with all the exit problems we'll come to.

The asset-price effect was immediate and huge. The yen, which had been painfully strong (around 80 to the dollar), weakened past 100 and kept falling — a gift to exporters like Toyota and Sony, whose foreign earnings translated into far more yen. The Nikkei, which had languished near 10,000, began a multi-year climb that eventually carried it back toward its 1989 high. Japanese equities had their best stretch in a generation. This is the portfolio-rebalancing and currency channels firing on all cylinders: cheaper yen, more liquidity, investors pushed into stocks.

But — and this is the recurring "but" of the whole Japanese experiment — the *real-economy* result fell short. Inflation popped briefly toward 2% in 2014, then faded. Wages barely grew. The structural forces dragging on Japanese demand — a shrinking, ageing population, cautious firms, deleveraging households — were stronger than the monetary push. QQE proved, once again, that you can reliably move *asset prices* with enough money, and only unreliably move *prices in the shops*.

## NIRP: below the floor (January 2016)

When even the bazooka couldn't seal the deal, the BOJ reached for the next rung — and stepped *through* the floor we said couldn't be broken. In January 2016 it introduced a **Negative Interest Rate Policy (NIRP)**, charging **−0.1%** on a portion of the reserves banks parked at the central bank.

Read that again: the bank now *charged* commercial banks for the privilege of storing money at the BOJ. The logic was to make sitting on cash actively painful, to push banks to lend instead of hoard. How is this possible if cash pays 0%? Because converting ¥-trillions of electronic reserves into physical banknotes and storing them is itself costly and impractical for a big bank — so a *small* negative rate, like −0.1%, can hold without triggering a stampede into cash. A *large* negative rate could not, which is exactly why no central bank ever pushed deep into negative territory. The effective floor is just *slightly* below zero, not far below it.

NIRP was controversial and only partly effective. It did push the whole yield curve down — short-term JGB yields went negative, and even some long yields flirted with zero. But it also squeezed banks' profits, and this deserves a careful look, because it is the heart of why negative rates are a self-limiting tool. Banks make money on the *spread* between what they earn on loans and what they pay on deposits. The trouble is that the deposit rate has its own floor at roughly zero: a bank cannot easily charge ordinary households a negative rate on their savings, because the households would just withdraw cash (the same escape hatch again). So when the central bank pushes the *lending* side of the spread down toward zero while the *deposit* side is stuck at zero, the spread — the bank's profit margin — gets crushed. A less profitable bank builds less capital, and a bank with thin capital lends *less*, not more. The tool meant to spur lending can throttle it.

The Japanese banking sector, already weakened by decades of low rates and the bad-loan hangover of the 1990s bust, felt this acutely. Bank stocks fell on the NIRP announcement — the market understood immediately that squeezed margins were bad for bank earnings. This is the recurring pattern of the upper rungs of the ladder: each new tool has a sharper side effect than the last. ZIRP was nearly costless. QE inflated some asset bubbles. QQE distorted the bond market. NIRP attacked the banking system's profitability. The further the BOJ climbed, the more collateral damage each rung inflicted, which is precisely why it kept the negative rate to a token −0.1% and never went deeper. Negative rates are the rung where the cure starts fighting the disease, and the BOJ kept the dose tiny on purpose.

## YCC: fixing the price of a 10-year bond (September 2016)

The final and most distinctive rung. Months after NIRP, in September 2016, the BOJ unveiled **Yield Curve Control (YCC)** — and it is the boldest thing any major central bank has done at the zero bound. Under YCC, the BOJ stopped targeting a *quantity* of bond purchases and went back to targeting a *price* — but not the overnight rate. It targeted the yield on the **10-year JGB**, pledging to hold it "around 0%."

![Yield curve control flow showing the BOJ buying unlimited JGBs to hold the 10 year yield at the cap](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-7.png)

Here is the mechanism, and it is beautiful and terrifying in equal measure. Normally the 10-year yield is set by the market — by millions of buyers and sellers. To override that and *pin* the yield at a target, the central bank makes an open-ended promise: **whenever the yield tries to rise above the cap, the BOJ will buy whatever quantity of bonds is required to push it back down.** It offers a "fixed-rate operation" — a standing order to buy *any amount* of 10-year JGBs at the target yield. In principle, unlimited.

Why does fixing a *price* require *unlimited* buying? Recall the worked example from the ZIRP section: when yields drift up, bond prices fall, and at a near-zero coupon they fall *hard*. If the market wants the 10-year yield at 0.6% but the BOJ insists on ~0%, every seller who hits the BOJ's fixed-rate bid is offloading a bond at an artificially high price (artificially low yield) — and the BOJ must absorb *all of them*. There is no cap on how much the market might want to sell. So holding the price requires a buyer with an infinite balance sheet. The central bank is the only entity that has one — it creates the yen to pay with.

#### Worked example: the BOJ's ETF book marked to market

YCC dealt with bonds, but QQE had also made the BOJ a giant *equity* holder through its ETF purchases. By the early 2020s the BOJ's equity ETF holdings had a book (cost) value around **¥37 trillion** — about **\$240 billion** — making the central bank the single largest owner of Japanese stocks. What happens to that book when stocks move? Let's mark it.

Suppose the BOJ holds ETFs that cost **¥37 trillion** to acquire, bought at an average Nikkei level of roughly **22,000**. Now the Nikkei rises to **33,000** — a **50%** gain. The mark-to-market value of the book rises with it:

```
unrealized gain  =  cost  x  (new level / cost-basis level  -  1)
                 =  Y37 trillion  x  (33,000 / 22,000  -  1)
                 =  Y37 trillion  x  0.50
                 ~  Y18.5 trillion  (about $120 billion) of paper profit
```

A ¥18.5 trillion paper gain — about **\$120 billion** — sounds wonderful. But flip it: if the Nikkei instead *fell* 30%, the same book would show roughly an ¥11 trillion *loss* (about \$72 billion), and the central bank's own balance sheet — the institution that is supposed to be the bedrock of financial stability — would be sitting on a large equity loss tied to the very market it can move. That is the trap of "qualitative" easing: by buying stocks, the BOJ tied its own solvency optics to equity prices and made itself a permanent, price-insensitive bid that distorts the market it owns. The intuition: **a central bank that owns the stock market can prop it up, but it can never cleanly sell — any large sale would crater the prices it is marked against, so the book becomes a position it cannot easily exit.**

### The cost: the BOJ became the market

![Stacked bars showing the Bank of Japan owning over half of JGBs plus a large equity ETF book](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-3.png)

Add it all up and you reach the defining consequence of 30 years at the zero bound, shown in the figure above. By buying without limit to hold prices, the BOJ ended up *owning the market it was trying to set*. It came to hold **over half** of all outstanding JGBs — roughly **¥580 trillion** of a market it had, in effect, nationalized. It became the largest single owner of Japanese equities through its ETF book. It held a meaningful slice of the J-REIT (property) market.

The costs of becoming the market are real and they compound:

- **Price discovery dies.** A bond yield is supposed to be a *signal* — the market's collective judgment about growth, inflation, and risk. When the central bank pins it, that signal goes dark. Nobody knows what the "real" 10-year yield would be, because the price is administered, not discovered.
- **Liquidity withers.** With the BOJ holding most of the bonds and standing ready to buy more, there are days when barely any JGBs trade at all. The market becomes a queue to sell to one buyer.
- **The exit becomes terrifying.** If you own half the bond market and a huge equity book, *how do you ever leave* without crashing the prices you hold? Every step toward normal must be telegraphed, gradual, and gentle, or it triggers exactly the rout it is trying to avoid.

This is the deepest lesson of YCC: the tool works — the BOJ genuinely held the 10-year near zero for years — but the price of making it work was to absorb the market into the central bank's own balance sheet, with no easy way back out.

## The currency channel: the yen as the world's funding source

We need to step out of Japan for a moment, because the BOJ's zero-bound policy had a giant *global* side effect that became one of the most important trades on the planet: the **yen carry trade**.

When Japanese interest rates are pinned near zero while rates elsewhere — especially in the US — are far higher, the yen becomes the cheapest money in the world to borrow. So global investors do the obvious thing: they borrow yen at almost nothing, convert it to a higher-yielding currency, and buy assets that pay more. The profit is the interest-rate *gap* — the "carry." For background on the mechanics across currencies, see [the currency channel](/blog/trading/policy-and-markets/the-currency-channel-rate-differentials-and-carry); here we trace the specific yen version.

![Pipeline of the yen carry trade borrowing yen at 0.1 percent and buying a US asset yielding 5 percent](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-5.png)

#### Worked example: the yen carry trade and the FX risk that ends it

Let's run the trade with real numbers. You borrow **¥100 million** at a yen funding cost of about **0.1%** per year. At an exchange rate of **¥150** to the dollar, that's **\$666,667**. You buy a US asset — say short-term Treasuries — yielding **5.0%**.

```
USD interest earned  =  $666,667  x  5.0%   =  $33,333
yen interest owed    =  Y100,000,000 x 0.1% =  Y100,000  (~$667 at Y150)
net carry (FX flat)  =  $33,333  -  $667     ~  $32,666 per year
```

On **\$666,667** of borrowed-and-invested money, that's about a **4.9%** annual return *for taking no obvious risk* — you simply harvested the rate gap. Multiply by leverage and it becomes very attractive, which is why trillions of yen flowed into this trade for years.

But look at what you are exposed to: the **exchange rate**. Your debt is in yen; your asset is in dollars. If the yen *strengthens* — say from ¥150 to ¥140 to the dollar, a move of about 6.7% — your dollar asset is now worth fewer yen when you go to repay your yen loan:

```
to repay Y100,000,000 you now need:  Y100,000,000 / 140  =  $714,286
you have:                            ~$666,667 (plus the year's $32,666 carry)
FX loss on principal:                $714,286 - $666,667  ~  $47,619
```

A **6.7%** yen rally inflicts a roughly **\$47,600** loss on the principal — *more than a year and a half of carry* — wiped out in a single move. And here is the cruel dynamic: when the yen starts to strengthen, every carry trader rushes to unwind at once (sell the foreign asset, buy yen to repay the loan), and that buying *strengthens the yen further*, which forces more unwinds — a stampede. **The carry trade collects pennies in front of a steamroller: years of quiet rate-gap income, then a sudden FX move that takes it all back in days.** That is precisely what happened in August 2024, which we'll get to.

### Why this is everyone's story: Japan as the world's template

It would be easy to file all of this under "Japan is weird" and move on. That would be a serious mistake, because Japan is not the exception — it is the *preview*. Every major central bank that followed walked the same ladder, just later and usually less far.

When the 2008 global financial crisis hit, the US Federal Reserve cut its policy rate to near zero and held it there for *seven years* (2008–2015) — the Fed's own ZIRP. It launched three rounds of QE (QE1, QE2, QE3) buying Treasuries and mortgage bonds — the Fed's QE, lifted almost verbatim from the BOJ's 2001 playbook. The European Central Bank went further down the ladder than the US, adopting *negative* policy rates in 2014 (the ECB's NIRP, two years before Japan's) and running massive bond-buying. The Bank of England, the Swiss National Bank, the Swedish Riksbank, the Bank of Canada — all reached for the same tools. The entire developed world spent the 2010s clustered at or near the zero bound, doing what Japan had been doing alone since 1999.

And they all got the same result Japan got, which is the punchline of the whole series of episodes. The post-2008 QE programs reflated asset prices spectacularly — US stocks, having bottomed in March 2009, embarked on one of the longest bull markets in history; bond yields fell to multi-century lows; house prices recovered. Yet for most of the 2010s, *none* of these central banks could reliably hit their 2% inflation targets. Year after year, inflation ran *below* target across the US, Europe, and Japan despite trillions of dollars, euros, and yen of bond buying. The Japanese diagnosis — base money reflates assets, not necessarily prices, because inflation needs real demand — turned out to describe the entire developed world.

The contrast that proves the rule arrived in 2020–2022. The COVID response was different in one decisive way: this time, *fiscal* policy fired alongside monetary policy at enormous scale — direct cash to households, business support, the works (the US alone deployed roughly \$5 trillion in pandemic fiscal support). Monetary policy created the money; fiscal policy put it directly into people's hands, where it became *demand*. And *that* combination — money plus demand — finally produced the inflation that two decades of pure QE never could, surging to 9% in the US and over 10% in parts of Europe by 2022. The natural experiment is almost too clean: QE alone for a decade gave near-zero inflation; QE plus a fiscal bazooka gave the highest inflation in 40 years. Japan had been telling the world the answer the whole time. Monetary policy at the zero bound is powerful over asset prices and weak over the price level *by itself* — it needs a fiscal partner to close the loop, a theme we develop in [the fiscal toolkit](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits).

## Common misconceptions

**"Printing all that money was guaranteed to cause runaway inflation."** The single most-falsified prediction of the past 20 years. The BOJ expanded its monetary base more than fourfold and could not reliably reach *2%* inflation, let alone runaway inflation. As the QE worked example showed, base money only becomes spending if banks lend and firms borrow — and in a deflationary, deleveraging economy, they didn't. Money sat inert as reserves. Inflation is downstream of *demand*, not of the monetary base alone.

**"QE failed, so it did nothing."** Wrong in the other direction. QE and QQE were extraordinarily effective at one thing: lifting *asset prices*. The yen fell, the Nikkei tripled off its lows, JGB yields were crushed, risk assets rallied. The portfolio-rebalancing and currency channels worked exactly as designed. What failed was the leap from asset prices to wages and consumer prices. "QE moves markets but not Main Street" is the accurate summary — and it is *why* QE is sometimes criticized for worsening inequality (asset owners win; wage earners wait).

**"Yield curve control means the BOJ buys a fixed amount of bonds."** It's the opposite. YCC targets a *price* (the 10-year yield), so the *quantity* of buying is whatever it takes — potentially unlimited. In fact, in the early years of YCC, *because* the cap was credible, the BOJ sometimes had to buy *less* than under pure QQE: when markets believed the peg would hold, they did the BOJ's work for it. A credible price target can require fewer purchases than an explicit quantity target — credibility substitutes for cash, an echo of [Draghi's "whatever it takes"](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility).

**"Negative rates can go as low as the central bank wants."** No. The physical-cash escape hatch puts an effective floor only *slightly* below zero. The BOJ chose −0.1% precisely because anything much lower would push banks and savers toward hoarding banknotes. Negative rates are a *small* tool, not a bottomless one.

**"Japan is just a unique basket case; it has no lessons for the US or Europe."** Exactly backwards. Japan is the *template*. Every tool the Fed, ECB, and BOE used after 2008 was pioneered in Japan years earlier, and the central lesson — that QE reliably reflates assets but not necessarily the real economy — shaped how those banks set expectations for their own programs. When the next zero-bound episode arrives anywhere, the playbook will be the Japanese one.

## Case studies: the tools meeting real markets

### The asset arithmetic: what each tool did to each asset

Pull the threads together asset by asset, because the spine of this whole series is *policy lever → channel → asset value*.

**JGBs (the bond).** The cumulative effect of ZIRP, QE, QQE, NIRP, and YCC was to crush Japanese government bond yields to near zero and hold them there for years. The 10-year JGB yield, which had been several percent in the 1990s, was pinned around 0% from 2016. The mechanism is direct: the BOJ became the dominant buyer, and under YCC, the *guaranteed* buyer at the cap. For bondholders this was a one-way ride — yields could only fall or stay flat, so prices could only rise or hold. The cost, as we saw, was a market the central bank effectively owned and could not easily leave.

**The yen (the currency).** Each easing step pushed the yen weaker, because lower yen yields make the yen less attractive to hold and they fund the carry trade. The yen went from around 80 to the dollar in 2012 to past 150 by 2024. A weaker yen is a *transfer* within the Japanese economy: it boosts exporters' yen earnings (a windfall for the Nikkei's big multinationals — Toyota's foreign profits, translated back into a cheaper yen, swell enormously) while it punishes households and import-dependent firms with pricier energy, food, and raw materials. This is the most important thing to grasp about a currency move: it does not make a country uniformly richer or poorer, it *redistributes* — from importers and consumers to exporters and foreign-asset holders. The currency channel is also the most reliable transmission of all the easing levers: easier money almost always weakens the currency, far more dependably than QE lifts inflation. When you want to predict the *first* effect of an easing surprise, look at the exchange rate.

Here is the bite of that redistribution made concrete. Japan imports almost all of its energy. When the yen weakened from ~110 to ~150 against the dollar (about a 36% depreciation) while global oil was already elevated, the *yen* price of imported crude roughly doubled over the cycle, feeding straight into electricity bills, transport, and food. That is how, by 2022–23, Japan finally got the 2%-plus inflation it had chased for two decades — not from booming domestic wages, but from a weak currency making imports expensive. It was inflation of the *wrong kind*: a cost-of-living squeeze on households rather than the demand-driven, wage-led inflation the BOJ actually wanted. The lever delivered the *number* the bank had targeted while missing the *meaning* behind it.

**Japanese equities (the multiple).** Stocks were the big winner across the whole experiment. Three forces stacked up in their favor: a weaker yen lifted exporter earnings (the *E* in the price-to-earnings multiple), QE's portfolio-rebalancing pushed yield-starved investors out of zero-yielding bonds and into stocks (lifting the *P/E multiple* itself), and the BOJ's own ETF buying put a literal, price-insensitive central-bank bid under the market. The Nikkei climbed from near 10,000 in 2012 back toward — and in February 2024, finally *past* — its 1989 high of 38,957. Thirty-five years to reclaim the peak, and a large part of the last decade's climb was manufactured by policy rather than by underlying economic growth. That is the double edge of the equity result: the BOJ genuinely rescued the stock market, but it did so partly by buying it, which means a chunk of the gain reflects the central bank's bid rather than Japan's productive economy — and that bid, once placed, is nearly impossible to remove.

### The exit: widening, then ending YCC (2022–2024)

![Timeline of BOJ policy tools from ZIRP 1999 to QE QQE NIRP YCC and the 2024 exit](/imgs/blogs/japan-and-the-zero-bound-zirp-qe-and-ycc-8.png)

The timeline above is the whole arc in one frame: ZIRP in 1999, QE in 2001, the QQE bazooka in 2013, NIRP and YCC in 2016, and then — after a long, flat decade — the slow climb back out. For years the BOJ held the line while the rest of the world stayed at the zero bound too. Then the world changed. The post-COVID inflation surge of 2021–22 sent the Fed and ECB hiking aggressively — the Fed went from 0.25% to 5.50% in 16 months (see [the 2022 hiking cycle in context](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets)). Suddenly the gap between near-zero Japanese yields and ~4–5% US yields was enormous, and it dragged the yen down hard, toward 150 and beyond. Imported inflation finally pushed Japanese CPI above 2% — the thing the BOJ had wanted for 20 years arrived, but for the "wrong" reason (a weak yen and global commodity prices, not strong domestic demand).

Holding the 10-year at ~0% while everyone else's yields soared became unbearable. The pressure on the cap was immense — markets relentlessly tested it, betting the peg would break. The BOJ's response was a careful, telegraphed retreat:

- **December 2022:** it *widened* the YCC band, letting the 10-year trade up to ±0.5% instead of ±0.25%. The 10-year JGB had been doing essentially nothing for years; this was the first crack. Markets read it (correctly) as the beginning of the end.
- **2023:** further widening, treating the cap as a "reference" rather than a rigid line, letting the 10-year drift up toward and past 0.5%, then 1%.
- **March 2024:** the big one. The BOJ **ended NIRP and YCC together**, raising its policy rate above zero for the **first time since 2007** — the first hike in 17 years. It dismantled, in one meeting, the negative rate and the yield peg it had spent eight years defending.
- **January 2025:** it lifted the policy rate to **0.50%** — Japan's first genuinely positive policy rate in a generation.

The exit had to be gentle precisely *because* the BOJ owned the market. A sudden move would have crashed JGB prices (the bank's own holdings) and spiked the yen. So the bank crept out over more than two years, signaling every step, and even so it triggered turbulence — most spectacularly in the carry trade.

#### Worked example: the August 2024 carry unwind

The cleanest demonstration of how a tiny rate move at the zero bound can convulse global markets came in **early August 2024**. The BOJ had just raised its policy rate (to about 0.25%) and signaled more to come, while the Fed was signaling *cuts*. The yen rate gap that had powered the carry trade for years suddenly looked like it would *narrow* — yen rates up, dollar rates down. That is poison for the carry trade.

Trace the chain with the worked-example logic from before, and put a dollar figure on it. Suppose a fund ran our earlier carry trade at scale: borrow ¥10 billion (about **\$66.7 million** at ¥150) and hold US assets against it. The yen rallied from roughly ¥150 to about ¥142 in days — a move of about 5.3%. On \$66.7 million of principal, the FX loss alone is about:

```
FX loss  ~  principal  x  yen move
         ~  $66.7 million  x  5.3%
         ~  $3.5 million   (gone in days, vs. ~$3.3 million of annual carry)
```

A single week erased *more than a full year's carry*. Carry traders were short yen (they had borrowed it) and long foreign assets, so when the yen strengthened on the BOJ's hike their FX losses mounted exactly like this. To stop the bleeding they had to unwind: sell the foreign assets, buy yen to repay the loans. But *everyone* did it at once. The yen surged. Surging yen forced *more* unwinds. The feedback loop we described in the carry worked example ran in real time:

```
BOJ hikes  ->  yen rate gap narrows  ->  yen strengthens
      ->  carry traders take FX losses  ->  they sell assets, buy yen
      ->  yen strengthens MORE  ->  more forced unwinds  (the steamroller)
```

The result: on **5 August 2024**, the Nikkei fell **12.4%** in a single day — its worst point drop ever — and the shock rippled into US tech stocks and global risk assets. A *0.25%* rate move in Japan, the world's funding currency, briefly convulsed markets across the planet. The intuition the whole world relearned that day: **because zero-rate yen had become the cheap funding for global leverage, the BOJ's first real steps off the zero bound were not a local Japanese event — they were a global de-leveraging shock.** Then it stabilized; the BOJ soothed markets, and by 2025 the rate sat calmly at 0.50%. But the episode is the permanent reminder that exits from the zero bound are dangerous in proportion to how long and how deep the policy ran.

## What it means for asset values: the zero-bound playbook

Strip the 30-year story down to a set of repeatable reads — what to expect when *any* central bank approaches, sits at, or leaves the zero bound.

**When a central bank hits the zero bound, the lever shifts from price to quantity, and assets reprice in a predictable order.** The currency weakens first and most reliably (the carry/portfolio channels). Long-bond yields get crushed and pinned. Risk assets — equities, credit, property — get a sustained tailwind from portfolio rebalancing and, if the bank buys them directly, a literal central-bank bid. The signal to watch is the *announcement of a new tool*: each new rung (QE → QQE → NIRP → YCC) is a fresh, large impulse for the weaker currency and higher asset prices.

**Quantitative tools reflate assets far more reliably than they reflate the real economy.** When you see QE or its cousins announced, lean toward *asset-price* trades (long equities, long duration, short the currency) and be deeply skeptical of "this will finally cause runaway inflation" trades. Two decades of Japanese data say base money does not become consumer prices unless private demand is already there. What would invalidate this read: clear evidence that banks are lending and firms are borrowing and spending (broad money growing fast, credit expanding) — *then* inflation can follow.

**The deeper and longer the zero-bound policy, the more violent the exit.** A central bank that owns half its bond market and a chunk of its stock market cannot leave cleanly. Watch for the *first* sign of retreat (a band-widening, a hawkish guidance tweak) as the leading edge — it telegraphs the whole exit. And watch the *funding-currency* dimension: when the world's cheapest money (zero-rate yen) starts to get more expensive, leverage everywhere built on that cheap money is at risk of an unwind. The August 2024 Nikkei crash is the template. The signal that the exit is genuinely complete is a *positive, market-set* policy rate that the central bank can move up *and* down again — which Japan only really regained in 2025.

**The one lesson for every other central bank.** Japan ran the maximal experiment, and the verdict is humbling: monetary policy at the zero bound is *powerful over asset prices and nearly powerless over the price level by itself.* It can make money cheap, hold yields down, and lift markets for years. It cannot, on its own, make a deleveraging society want to borrow and spend, and it cannot manufacture the inflation that comes from real demand. That is why the modern consensus — visible in the COVID response, where monetary *and* [fiscal](/blog/trading/policy-and-markets/the-fiscal-toolkit-spending-taxes-and-deficits) bazookas fired together and *did* produce inflation — is that the zero bound is where monetary policy needs fiscal policy as a partner. Japan learned that the hard way, over thirty years, and wrote the textbook the rest of us now read.

There is a second, quieter lesson that markets internalized only slowly: **unconventional policy is far easier to start than to stop.** Each rung of the ladder was announced in a single dramatic meeting; the exit took more than two years of careful, telegraphed steps and *still* set off a global de-leveraging shock in August 2024. The asymmetry is structural, not a failure of nerve. When a central bank becomes the dominant holder of its own bond and stock markets, it cannot sell without moving the prices it is marked against, so it is trapped on the inside of a position it built one easy announcement at a time. For an investor, the practical implication is to treat the *first* hint of an exit — a band-widening, a hawkish word, a single small hike — with far more respect than its size suggests, because it is the leading edge of a slow, grinding unwind of everything the easy money inflated. Japan spent thirty years proving that the cheapest moment in monetary policy is the moment you ease, and the most expensive is the moment, years later, when you finally have to take it back.

## Further reading & cross-links

Within this series:

- [Currency Policy and FX Intervention](/blog/trading/policy-and-markets/currency-policy-and-fx-intervention) — the broader toolkit for setting and defending an exchange rate, of which the BOJ's yen management is one case.
- [The Liquidity Channel: QE, QT, and the Everything Bid](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) — the general mechanism by which balance-sheet policy floods markets, applied to the Fed and ECB.
- [The Currency Channel: Rate Differentials and Carry](/blog/trading/policy-and-markets/the-currency-channel-rate-differentials-and-carry) — the full theory behind the yen carry trade and why rate gaps move exchange rates.
- [The Plaza Accord 1985: Engineering a Weaker Dollar](/blog/trading/policy-and-markets/the-plaza-accord-1985-engineering-a-weaker-dollar) — the coordinated FX move that pushed the yen up and helped inflate the bubble whose bust started this whole story.
- [Bernanke's QE 2008–14: The Birth of Balance-Sheet Policy](/blog/trading/policy-and-markets/bernanke-qe-2008-14-the-birth-of-balance-sheet-policy) — how the Fed adapted Japan's playbook a decade later, and what was the same and different.
- [The Expectations Channel: Forward Guidance and Credibility](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility) — why a credible promise (and a credible peg) can do the work of actual purchases.

Elsewhere on the site:

- [Quantitative Easing Explained: Printing Money](/blog/trading/finance/quantitative-easing-explained-printing-money) — the beginner-level mechanics of QE from first principles.
- [The Monetary Toolkit: Rates, QE, QT, and Forward Guidance](/blog/trading/policy-and-markets/the-monetary-toolkit-rates-qe-qt-and-forward-guidance) — the full menu of levers a central bank can pull.
- [QE vs QT: How Balance-Sheet Policy Moves Markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) — the trader's-eye view of positioning around balance-sheet shifts.
- [The Discount-Rate Channel: How Rates Reprice Cash Flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — why a change in yields is, mechanically, a change in what every asset is worth.

---
title: "QE vs QT: How Balance-Sheet Policy Moves Stocks, Bonds, FX, and Crypto"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero deep dive into quantitative easing and tightening — what the Fed actually does when it buys or sheds bonds, the portfolio-balance and signaling channels, why QE is not printing money into the economy, and how stocks, bonds, the dollar, and crypto respond — with the 2020-2024 cycle as the clearest demonstration in history."
tags: ["macro", "monetary-policy", "quantitative-easing", "quantitative-tightening", "liquidity", "federal-reserve", "balance-sheet", "bonds", "crypto", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When the policy rate hits zero, the central bank stops cutting rates and starts working the **balance sheet**. Quantitative easing (QE) buys bonds, creates bank reserves, and lifts risk assets through the portfolio-balance and signaling channels; quantitative tightening (QT) lets those bonds roll off and drains the system. The mechanics are simple once you see them, and 2020-2024 was the clearest demonstration in history.
>
> - **QE is an asset swap, not cash handed to the public.** The Fed buys a bond and pays with newly created **reserves** — money that lives inside the banking system and can never be spent at a store. It works by *removing* safe assets and duration from the market, which pushes investors out along the risk curve.
> - **The headline number is huge.** The Fed's balance sheet went from about \$4.3T in March 2020 to a peak of **\$8.96T in April 2022** — it nearly doubled in two years — and the everything-rally that followed (stocks, bonds, credit, crypto) was the portfolio-balance channel in action.
> - **QT reverses the flow, but slowly and with a cap.** From 2022 to 2025 the Fed drained roughly \$2.3T at a cap of about \$95B/month. QT does *not* have to crash stocks: in 2022-2024 the draining reverse-repo facility cushioned the hit, and the S&P still made new highs.
> - **The one habit to build:** watch the *trajectory* of the balance sheet and reserves, not the headline level. Rising is a tailwind for risk; draining is a headwind; *scarce* reserves are the invalidation that turns the whole framework off.

On **March 23, 2020**, with the COVID panic at its worst and credit markets freezing solid, the Federal Reserve put out a short statement that contained two words the modern market had never seen together: it would buy Treasury and mortgage bonds **"in the amounts needed."** No cap. No fixed program size. Open-ended. The financial press instantly nicknamed it **infinite QE.**

What happened next is one of the most violent risk-asset reversals on record. The S&P 500, which had just fallen 34% in five weeks, bottomed that very day and went on to nearly double over the next two years. Corporate bonds that had been gapping lower stabilized within days. By the end of 2021, the Nasdaq had risen over 130% from the March low, the price of Bitcoin had gone from under \$5,000 to nearly \$69,000, and the riskiest, longest-duration, most speculative assets in the world — unprofitable tech, meme stocks, far-out crypto — had been the biggest winners of all. It was, almost literally, an **everything rally.**

Here is the puzzle that should bother you: the economy in mid-2020 was a catastrophe. Tens of millions were unemployed. Whole industries were shut. Earnings were collapsing. And yet asset prices *soared*. The standard story — "stocks reflect the economy" — fails completely. The thing that explained the rally was not the economy. It was the Fed's **balance sheet**, expanding by trillions of dollars through QE. By the end of this post you will understand exactly why that balance sheet is the master variable when rates are near zero, how each asset class responds to it, and how a trader reads its trajectory. We build everything from the ground up — no finance background assumed.

![Diagram of QE mechanics showing the Fed buying a bond, reserves created, and money rotating into risk assets](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-1.png)

## Foundations: QE, QT, reserves, and the zero lower bound

Before any trading signal, you need a clean mental model of four things: what a central bank's interest-rate lever *is*, why it runs out, what reserves are, and what the central bank does when the lever hits the floor. We define every term from zero.

### The normal tool: the policy rate

In ordinary times, a central bank like the Fed has exactly one main lever: a very short-term interest rate. In the US this is the **federal funds rate** — the rate at which banks lend each other money overnight. The Fed does not set it by decree; it steers it by managing the supply of overnight cash in the banking system. When the Fed wants to *stimulate* a weak economy, it **cuts** the rate, making borrowing cheaper, which encourages spending and investment. When it wants to *cool* an overheating economy, it **hikes**, making borrowing more expensive. This is the entire conventional toolkit, and we cover it in depth in [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

The everyday analogy: treat the policy rate as the thermostat for the economy. Too cold (recession, high unemployment)? Turn the dial down — cheaper money. Too hot (inflation)? Turn the dial up — more expensive money. For decades, this dial was all the central bank needed.

### The problem: the dial hits the floor

But the thermostat has a floor. You cannot cut the policy rate much below **zero**, because if banks charged you to hold cash, people would just hold physical banknotes, which pay exactly 0%. (A few central banks experimented with slightly negative rates, but the practical limit is right around zero.) This floor has a name every trader should know: the **zero lower bound (ZLB)**, also called the **effective lower bound**.

Here is the trap. Imagine a severe recession hits when the policy rate is *already* low — say 0.25%, as it was in early 2020. You cut to zero, you have used up your last 0.25% of room, and the economy is *still* in freefall. The thermostat is jammed against the floor and the room is still freezing. The conventional tool is exhausted. This is exactly the situation in 2008 and again in 2020. **The zero lower bound is the trigger.** It is the precise condition that makes a central bank reach for a different lever entirely: the **balance sheet.**

This is the single most important idea in the post, so let it land. QE is not a tool central banks use because they like it. It is the tool they reach for when the normal tool has run out of room. Rates at zero is the *signal* that balance-sheet policy is coming.

### The new tool: quantitative easing

**Quantitative easing (QE)** is the central bank buying large quantities of bonds — mostly government bonds (Treasuries) and, in the US, mortgage-backed securities — in the open market, paying for them with newly created money. "Quantitative" because the lever is now a *quantity* (how many bonds, how much money) rather than a *price* (the interest rate). The plain-English summary: when the central bank cannot cut the price of money any lower, it instead pumps the *quantity* of money higher.

We have a dedicated primer on the basics in [Quantitative easing explained: is it really printing money?](/blog/trading/finance/quantitative-easing-explained-printing-money) — and the short answer to that title, which we will earn in detail below, is *not in the way most people think.*

The mirror image is **quantitative tightening (QT)**: the central bank *shrinking* its bond portfolio, which withdraws money from the system. In practice QT is usually done passively — the Fed simply lets bonds **mature and roll off** without buying new ones to replace them, rather than actively selling. When a bond the Fed holds matures, the Treasury pays it back, the money is extinguished, and the balance sheet shrinks. We will see the exact mechanics later.

A short history helps anchor the idea. QE is not a 2020 invention; it was pioneered by the **Bank of Japan in 2001**, after Japan's own rates had been stuck near zero for years following its 1990s asset-bubble collapse. The tool went global after the **2008 financial crisis**, when the Fed cut rates to zero and then ran three successive QE programs (nicknamed QE1, QE2, and QE3) between 2008 and 2014, taking its balance sheet from under \$1T to about \$4.5T. The first real attempt at QT followed in **2017-2019** — and it ended badly, with the September 2019 funding-market blowup we will return to. Then COVID hit in 2020 and the cycle ran again, faster and bigger than ever. So 2020-2024 was not a one-off experiment; it was the *fourth* full QE cycle and the *second* QT attempt, which is exactly why the patterns are now well understood. Each cycle taught the same lesson in a louder voice: when rates are at zero, the balance sheet is the lever, and its trajectory drives risk assets.

One technical distinction the bond market argues about endlessly is worth flagging early: **the "stock versus flow" debate.** Does QE work through the *flow* (the rate at which the Fed is currently buying, e.g. \$80B this month) or through the *stock* (the total quantity it has accumulated and is holding off the market)? The honest answer is both, but the *stock* view dominates the modern understanding: even after the Fed *stops* buying, the duration it has *already* removed stays removed, so the easing effect persists. This is why the *end* of QE (the "taper") matters as much as its start — it is the change in the flow that the market reprices, against a still-large stock. Hold this distinction; it explains why markets can wobble the moment buying *slows*, long before the balance sheet actually shrinks.

### Reserves: the money QE actually creates

To understand QE you must understand the *kind* of money it creates, because this is where almost every misconception lives. QE does not create dollar bills, and it does not credit your bank account. It creates **bank reserves**.

A **reserve** is an electronic balance that a commercial bank — JPMorgan, Bank of America, and so on — holds in *its own account at the Fed*. Think of it as the banks' checking account at the central bank. Two properties are load-bearing for everything that follows:

- **Reserves never leave the banking system.** You and I cannot hold reserves. A reserve balance can move from one bank's Fed account to another's when banks settle payments with each other, but it can never be spent at a grocery store or wired to a non-bank household. Only banks (and a handful of other institutions) hold them. This is the deep reason QE is not "printing money into the economy" — the money it prints is trapped inside the banking system's plumbing.

- **Reserves are the system's liquidity fuel.** When reserves are *ample*, banks lend to each other freely, short-term funding is cheap, and the plumbing hums. When reserves get *scarce*, funding markets seize and rates spike. We come back to this — it is the single most important risk in the playbook.

The relationship between the asset side (the bonds the Fed buys) and the liability side (the reserves it creates) is just accounting: the two sides of the balance sheet are always equal, and they rise together in lockstep during QE and fall together during QT. We go deep on the full liability mix — reserves, the reverse-repo facility, the Treasury's cash account — in [Reading the central bank balance sheet: reserves, RRP, TGA, and net liquidity](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga). For this post, the one fact to hold is: **QE buys bonds with new reserves; QT lets bonds roll off and extinguishes reserves.**

## The mechanics: the Fed buys a bond — who actually got the money?

Let us slow down and walk through a single QE purchase, step by step, because the whole post depends on getting this exactly right. The cover figure at the top traces this flow; we now narrate it.

Suppose the Fed decides to buy a \$100 million Treasury bond from a bank (in practice from a *primary dealer*, a large bank authorized to trade directly with the Fed). Here is what happens, in order:

1. **The bank hands the Fed the bond.** The \$100M Treasury moves from the bank's books to the Fed's. On the Fed's balance sheet, the asset side grows by \$100M.

2. **The Fed pays with new reserves.** The Fed credits the bank's reserve account at the Fed with \$100M. It does not move this money from anywhere — it *creates* it, electronically, the way you create a number by typing it. On the Fed's balance sheet, the liability side grows by \$100M (it now "owes" the bank those reserves). The two sides grew together. Nothing was taken from taxpayers; nothing was printed as cash.

3. **The bank now holds reserves instead of a bond.** This is the crucial swap. The bank gave up an interest-bearing safe asset (the bond) and received reserves (which pay the Fed's interest-on-reserves rate). Its *total* assets are unchanged — it just swapped one safe asset for another.

So who got the money? Not the public. Not borrowers. A bank swapped a bond for reserves. If that were the whole story, QE would do almost nothing — it just changes the *composition* of safe assets a bank holds. The magic is in the **second-order** effect, and it is where the entire transmission mechanism lives.

### The non-bank case is where it gets interesting

The Fed mostly buys from dealers, but those dealers are often buying the bonds *from* non-banks — pension funds, insurance companies, asset managers, foreign central banks. Trace it through. A pension fund sells its \$100M Treasury into the QE program. The pension fund now holds \$100M of **cash** (a bank deposit) instead of a bond. The bank that intermediated holds \$100M of new reserves. So now there is a non-bank entity sitting on a pile of cash that used to be a 10-year bond yielding, say, 2%.

That pension fund has a problem. It has liabilities to pay — pensions to retirees — and it needs its assets to earn a return. It cannot earn a return sitting in cash at ~0%. It *must* redeploy that cash into something that yields more. But the safe bond it used to own has been bought up by the Fed and is no longer available at the old yield. So the fund is *forced down the risk ladder*: into corporate bonds, into equities, into anything that offers a return. **This forced rebalancing is the engine of QE.** The Fed did not give the pension fund money; it *took away its safe yielding asset* and left it holding cash it cannot afford to sit on.

### Why the reserves don't force banks to lend: interest on reserves

A natural objection at this point: "But the bank now has \$100M of fresh reserves — surely it lends them out, and *that* floods the economy with money?" This is the heart of the "money multiplier" intuition many people carry, and in the modern system it is wrong. Two reasons.

First, as we said, **reserves cannot leave the banking system.** A bank cannot lend a reserve to a household; reserves only move between banks' Fed accounts. When a bank makes a loan, it creates a *new deposit* out of thin air — it does not "lend out" its reserves. (We unpack exactly how lending creates money in [How credit creates money: the lending channel and cycles](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles).) So having more reserves does not mechanically force more lending; lending is constrained by capital, regulation, and the demand for creditworthy loans, not by the quantity of reserves.

Second, the Fed deliberately *pays banks interest on reserves* (the **IORB** rate, interest on reserve balances). This is a crucial and underappreciated mechanic. Since 2008 the Fed has paid banks a risk-free rate just to *hold* reserves at the Fed. That means a bank holding QE-created reserves earns a perfectly safe return doing nothing — so it has no desperate need to push those reserves into riskier lending. IORB is also how the Fed *controls* the policy rate in an ample-reserves world: by setting the rate it pays on reserves, it sets a floor under what banks will accept elsewhere. The upshot for QE is profound: because reserves are trapped *and* earn a safe return, QE reliably inflates *asset* prices (through the portfolio-balance channel on non-banks) without reliably inflating *consumer* prices (because the reserves never gush into real-economy lending). This single design choice is most of the answer to "why didn't \$3.6T of QE cause runaway inflation in the 2010s?"

#### Worked example: the 2020 QE flood scale and the rally it powered

Put real numbers on the flood. Before the pandemic, in early March 2020, the Fed's total assets stood at about **\$4.3 trillion** (`data.FED_ASSETS`, the \$4.31T print for 2020-03). Over the next two years of open-ended QE — buying roughly \$120B of bonds *per month* at the peak — the balance sheet expanded to its high of **\$8.96 trillion** in April 2022 (`data.FED_ASSETS_PEAK`). That is an increase of about **\$4.66 trillion**, more than doubling the size of the balance sheet in roughly 24 months.

Now line that up against the asset returns over the same window, from the March 2020 low to roughly the end of 2021:

- The S&P 500 roughly **doubled** (up about 100%).
- The Nasdaq 100 rose roughly **130%**.
- Bitcoin went from under **\$5,000** to nearly **\$69,000** — a gain of over **1,200%**.
- Investment-grade and high-yield corporate bonds rallied hard as spreads compressed to historic lows.

Map the magnitudes onto the risk ladder and a pattern jumps out: the further out on the risk curve an asset sat, the bigger its gain. Safe assets (Treasuries) barely moved in price; risky assets (stocks) doubled; the riskiest, longest-duration assets (speculative crypto) went up by orders of magnitude. **The intuition to keep:** QE pushes \$4.66T of cash off the safe-asset shelf and the money cascades down the risk ladder, lifting the riskiest assets the most.

![Area chart of the Fed balance sheet from 2019 to 2025 showing the QE ramp to a peak and the QT drain](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-2.png)

The chart above is the spine of the entire 2020-2024 story. The green ramp is the QE flood — the balance sheet nearly doubling from \$4.3T to the \$8.96T April-2022 peak. The slate line after the peak is QT: the slow, deliberate drain back toward \$6.66T by mid-2025. Every asset-class move we discuss is, at bottom, a response to the *direction* of this one line. When it rises, liquidity is being added; when it falls, liquidity is being drained.

## The portfolio-balance channel: safe assets removed, money pushed into risk

We have just seen the engine; now let us name it formally, because it is the primary way QE reaches markets. Economists call it the **portfolio-balance channel.**

The idea: every investor holds a *portfolio* — a mix of safe and risky assets balanced to their appetite for risk and need for return. When the central bank buys up a huge quantity of the safest, longest-duration assets (Treasuries) and removes them from the available supply, two things happen at once:

1. **The price of the remaining safe bonds rises, so their yield falls.** Bond prices and yields move inversely — when there is more demand for a bond (the Fed is a giant new buyer), its price goes up and the interest it pays relative to that price goes down. So QE *directly* pushes down the yield on safe government bonds.

2. **Investors are left holding cash they must redeploy.** As we traced above, the sellers end up with cash earning nothing. With safe-bond yields now *lower* (less attractive) and a pile of cash that has to work, investors rebalance their portfolios *toward riskier assets* — corporate credit, equities, real estate, crypto — bidding those up.

The net effect: QE *lowers the yield investors can earn on safety, and raises the price they are willing to pay for risk.* That is the portfolio-balance channel in one sentence. It is why QE is sometimes described as the Fed "reaching for risk" on investors' behalf — by removing safe assets, it makes everyone reach a little further out the curve.

![Diagram of the portfolio-balance channel showing safe assets removed, yields falling, and money rotating into credit, equities, and crypto](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-3.png)

### Duration: the second thing QE removes

There is a subtler dimension to what QE removes, and serious traders track it closely: **duration**.

**Duration** is a measure of how sensitive a bond's price is to changes in interest rates, driven mostly by how *long* until the bond matures. A 30-year bond has high duration — a small change in rates swings its price a lot. A 3-month Treasury bill has almost no duration — its price barely moves with rates. When the market has to hold a lot of long-duration bonds, it demands extra compensation for the interest-rate risk, called the **term premium** — the extra yield long bonds pay over short ones purely for the risk of holding them a long time.

Here is the key: when the Fed buys long-dated bonds through QE, it is not just removing *bonds* from the market — it is removing **duration risk**. The market collectively has less interest-rate risk to bear, so it demands a smaller term premium, so long-term yields fall *further* than the simple supply effect alone would suggest. This is why QE programs deliberately target longer maturities (the 2011 "Operation Twist" was explicitly about buying long and selling short to remove duration). **Removing duration from the market is a distinct, powerful lever, separate from the reserve creation.**

#### Worked example: the duration removed by \$80B/month of buying

Put numbers on the duration channel. Suppose the Fed buys **\$80 billion per month** of bonds with an average maturity around 10 years — roughly the pace of one phase of the 2020-2021 program. Each month, the market has \$80B *less* 10-year-equivalent duration to hold. Over a year that is nearly **\$1 trillion** of duration risk lifted off private balance sheets and parked at the Fed, which does not care about price swings.

How much does that move yields? Empirical estimates from the QE era are remarkably consistent: a large-scale asset-purchase program of a few percent of GDP tends to lower the 10-year Treasury yield by roughly **0.15 to 0.25 percentage points (15-25 basis points)** per \$500B or so of purchases, mostly through the term-premium channel. The cumulative 2008-2014 QE programs are estimated to have held the 10-year yield down by on the order of **1 full percentage point** versus where it would otherwise have been.

So when you saw the 10-year yield fall to **0.62%** in July 2020 (`data.UST10Y`, the 2020-07 print), a large slice of that was the duration the Fed had absorbed. Lower long yields ripple straight into mortgage rates, corporate borrowing costs, and the discount rate used to value stocks — which is precisely how the bond-market effect becomes a stock-market effect. **The intuition to keep:** QE does not just add money, it absorbs the *interest-rate risk* the rest of the market would have to hold, and that absorption is worth roughly a full percentage point of 10-year yield.

## The signaling channel: QE as a promise to stay easy

The portfolio-balance channel is the mechanical effect of bond-buying. But QE has a second, psychological channel that is just as important for markets: the **signaling channel.**

When a central bank launches QE, it is not just buying bonds — it is making a *statement of intent*. By committing to a multi-trillion-dollar program, the Fed signals: *we are deadly serious about supporting the economy, we are not going to raise rates anytime soon, and we will keep policy loose for a long time.* That promise, believed, is itself stimulative.

Why does the promise matter so much? Because asset prices depend not just on today's interest rate but on the *expected path* of rates for years into the future. A stock is worth the present value of all its future cash flows, discounted at expected interest rates. If the market believes rates will stay near zero for three years instead of one, the discount applied to those future cash flows is smaller, so the stock is worth more *today*. QE, by signaling "lower for longer," pulls down the *expected path* of rates, not just the current one — and that re-rates every long-duration asset upward.

This is why the *announcement* of QE often moves markets more than the actual buying. When the Fed said "in the amounts needed" on March 23, 2020, almost no bonds had been bought yet — but the *signal* alone (open-ended, unlimited, whatever-it-takes) reversed the market that day. The signaling channel front-runs the mechanical channel. This also links QE tightly to **forward guidance** — the Fed's explicit communication about the future rate path — which we treat as one toolkit in [The central bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).

The flip side is a known fragility: if the market *stops believing* the promise — if it suspects the Fed will have to tighten sooner than it claims — the signaling channel can reverse violently. The 2013 **"taper tantrum"** is the classic case: merely *hinting* that QE would slow caused a sharp spike in yields and a sell-off in risk, even though no actual tightening had happened. The signal works in both directions.

## QT in reverse: the drain, the cap, and the runoff

Now flip the whole machine around. **Quantitative tightening (QT)** is QE run backwards: instead of expanding the balance sheet, the central bank shrinks it, draining reserves and reversing the portfolio-balance and duration effects.

The mechanism, in modern practice, is **passive runoff**, and the detail matters. The Fed holds thousands of bonds, and some mature every month. Normally, when a bond matures, the Treasury pays the Fed back, and the Fed *reinvests* that money by buying a new bond — keeping the balance sheet flat. Under QT, the Fed simply *stops reinvesting*: it lets the bonds mature and the money disappears. When the Treasury repays a maturing bond the Fed holds, it does so by debiting the Treasury's account and extinguishing the corresponding reserves. The balance sheet shrinks; reserves drain. No bonds are actively sold — they just roll off.

To keep this orderly, the Fed sets a **cap** — a maximum dollar amount it will let roll off per month. If *more* than the cap matures in a given month, the Fed reinvests the excess. The 2022-2024 QT program ran at a combined cap of about **\$95 billion per month** (\$60B Treasuries + \$35B mortgage bonds) at full pace, later slowed. The cap makes QT a predictable, gradual drain rather than a cliff.

#### Worked example: the QT runoff math — draining \$1.7 trillion at a \$95B cap

Do the arithmetic that every macro desk did in 2022. The Fed announced QT at a cap of roughly **\$95 billion per month.** Suppose the goal is to drain **\$1.7 trillion** of assets (a plausible target to get the balance sheet from the \$8.96T peak back toward a "normalized" level). How long does that take?

```
target drain      = $1,700 billion
monthly cap       = $95 billion / month
months to drain   = 1,700 / 95 ≈ 17.9 months  (a bit under 1.5 years at full pace)
```

But that is the *ceiling* pace. In practice, mortgage bonds prepay slowly and rarely hit their \$35B cap, so the realized drain ran closer to **\$75-80B/month**, stretching the same \$1.7T drain to over **21 months.** And indeed, from the \$8.96T peak in April 2022, the balance sheet fell to about **\$6.66T by mid-2025** (`data.FED_ASSETS`) — a drain of roughly **\$2.3T over about 38 months**, averaging close to \$60B/month once the pace was deliberately slowed in 2024. **The intuition to keep:** QT is a slow, mechanical, *pre-announced* drain — you can literally calendar it — which is exactly why its surprises come not from the runoff itself but from *where* the drained liquidity comes from.

That last point is the entire subtlety of QT, and it is where the 2022-2024 story gets interesting. QT shrinks the balance sheet, but the drained liquidity can come out of *different* liability buckets. If it drains **bank reserves**, it tightens the core of the system and risk assets feel it. But if it drains a *parking lot* of idle cash instead — the reverse-repo facility — then bank reserves barely fall, and risk assets shrug. The whole question of "will QT crash stocks?" reduces to "which bucket does the drain come from?" We return to this with the real 2022-2024 data in the case-study section.

![Dual-axis chart of the Fed balance sheet versus the 10-year Treasury yield, showing assets up and yields down in QE, reversing in QT](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-4.png)

The dual-axis chart above makes the QE/QT relationship with yields visible. The blue line is the balance sheet; the red line is the 10-year Treasury yield. Read it left to right: during the QE expansion (green band), the balance sheet climbs and the yield is pinned near its lows. After the April-2022 peak, QT begins (amber band), the balance sheet falls — and the yield rises sharply, from under 1% toward 4.5%. The portfolio-balance channel running in reverse: as the Fed stopped absorbing duration and started letting it back into the market, the term premium rebuilt and long yields rose.

## Asset by asset: how each market responds

We now have the full machine. Let us walk each major asset class and state precisely how it responds to QE and QT, with the mechanism for each. This is the part you trade.

### Bonds — the most direct effect

Bonds respond first and most mechanically, because the Fed is literally buying them.

- **Under QE:** the Fed's buying pushes bond *prices up* and *yields down*, especially at the long end where it removes duration. Treasury yields fall; the term premium compresses. This is the most direct, first-order effect of the whole program.
- **Under QT:** the Fed stops buying and lets bonds roll off, so the market must absorb more supply and more duration. Yields *rise*, the term premium rebuilds. The 10-year going from 0.62% (2020) to 4.88% (October 2023) is the cleanest illustration in the dataset.

There is one famous counter-intuitive wrinkle: sometimes when QE is *announced*, long yields actually *rise* rather than fall. Why? Because the signaling channel says "the Fed will successfully reflate the economy," and a stronger expected economy means higher expected future inflation and growth, which lifts long yields. So the *growth-expectations* effect can briefly overwhelm the *supply* effect. But the dominant, durable effect of sustained QE is lower yields, and of QT is higher yields.

### Stocks — the portfolio-balance beneficiary

Equities are the headline beneficiary of QE, through two reinforcing routes:

- **The discount-rate route:** lower long-term yields lower the rate at which future earnings are discounted, which raises the present value of stocks — *especially* long-duration growth stocks whose value is concentrated far in the future. This is why unprofitable tech and speculative growth led the 2020-2021 rally: they are the longest-duration equities, so they are the most sensitive to the discount rate QE crushes.
- **The rebalancing route:** the forced search for yield (portfolio-balance channel) pushes the cash from bond sellers into equities, bidding up multiples.

Under QT, both routes reverse: higher discount rates compress multiples (most painfully for the longest-duration growth names), and the draining liquidity removes the marginal bid. The 2022 bear market — with the Nasdaq down over 30% and the longest-duration, most speculative names down 60-80% — was QT and rate hikes hitting the exact assets QE had lifted the most. *What QE lifts the most, QT hits the hardest.*

A practical refinement for traders: the discount-rate route is best understood through the lens of **real yields**, not nominal ones. A stock's valuation is most sensitive to the *real* (inflation-adjusted) discount rate, and the 10-year TIPS yield is the cleanest market read on it. During the 2020-2021 QE flood the 10-year real yield was deeply *negative* — around **−1.0%** (`data.UST10Y_REAL`, the 2021-08 print) — which is a powerful tailwind for long-duration equities, because future earnings are barely discounted at all. When QT and hikes drove the real yield from −1.0% up to roughly **+2.5%** by October 2023, that 3.5-percentage-point swing in the discount rate is most of the story behind the growth-stock derating. If you watch one valuation input around balance-sheet policy, watch the real yield; we make it the centerpiece of [Real vs nominal: inflation, real yields, and the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

### The dollar — the relative-policy effect

The dollar's response is about *relative* monetary policy, and it is less mechanical than bonds or stocks. The cleanest framing: QE tends to *weaken* the currency, and QT to *strengthen* it, **all else equal** — because QE floods the system with the currency and pushes its yield down, making it less attractive to hold, while QT does the reverse.

But "all else equal" rarely holds, because *every* major central bank uses these tools, and the dollar trades on the *difference* between US policy and the rest of the world. If the Fed is doing QE while the European Central Bank is *also* doing QE, the dollar may not weaken at all. The 2022 episode is instructive: the Fed was tightening (QT plus aggressive hikes) faster and harder than most peers, which is a big reason the dollar surged to a multi-decade high (the dollar index peaked near 114.8 in September 2022). The transmission to FX runs through *rate differentials and relative liquidity*, which we develop fully in [What moves exchange rates: rates, flows, and carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry).

### Crypto — the high-beta tail of the risk curve

Crypto is the purest, most leveraged expression of the portfolio-balance channel, for a simple reason: it is the longest-duration, highest-risk, most speculative asset class in the liquidity ecosystem, with no earnings to anchor it. So it responds to liquidity the way a long lever responds to a small push — amplified.

- **Under QE (2020-2021):** Bitcoin went from under \$5,000 to nearly \$69,000, and the broader crypto complex rose far more. With cash cascading down the risk ladder and the safest assets yielding nothing, the *furthest* point on the risk curve received the most amplified flow.
- **Under QT (2022):** crypto led the drawdown, with Bitcoin falling roughly 75% from its high and the most speculative tokens falling 90%+. The same lever that amplified the upside amplified the downside.

The lesson traders take from 2020-2022 is that **crypto behaves as a high-beta liquidity proxy** — it tends to lead risk both up and down as the balance-sheet trajectory turns. It is not a hedge against the system; in the QE/QT era it has been the most sensitive *bet on* the system's liquidity.

This has a useful practical consequence: because crypto sits at the very end of the risk curve, it often *turns before* the broad equity index does, which makes it an early-warning gauge for the liquidity tide. When the balance-sheet trajectory is about to bite — in either direction — the furthest, most reflexive point on the curve tends to move first. A crypto market that is grinding higher while liquidity is being added is confirmation the tide is rising; a crypto market rolling over while equities are still complacent is one of the first cracks in the liquidity story. Treat it not as an asset to forecast but as a *sensor* on the liquidity regime — the canary that reacts to the air before the rest of the mine notices.

#### Worked example: the asset cascade — one push, four responses

Trace a single QE impulse through all four assets to see the cascade as one connected event, using the real 2020-2021 prints:

1. **The Fed buys**, balance sheet rises from \$4.3T toward \$8.96T (`data.FED_ASSETS`).
2. **Bonds:** the 10-year yield is pushed down to **0.62%** (`data.UST10Y`, 2020-07) — duration absorbed, term premium crushed.
3. **Stocks:** with the discount rate at rock bottom and cash searching for yield, the S&P roughly doubles and long-duration growth leads.
4. **Crypto:** the furthest point on the risk curve, Bitcoin, rises over 1,200% to ~\$69,000.
5. **The dollar:** softens modestly into 2020-2021 (the dollar index fell from ~96 in 2019 to ~90 at end-2020) as the flood of dollars and zero yields make it less attractive.

Now run it in reverse for 2022: balance sheet peaks and turns down (QT), the 10-year surges toward **4.88%** (`data.UST10Y`, 2023-10), stocks fall (Nasdaq −33%), crypto craters (Bitcoin −75%), and the dollar *strengthens* to a multi-decade high as the Fed out-tightens the world. **The intuition to keep:** QE and QT are a single dial, and turning it sends a correlated wave through all four asset classes — same direction of cause, amplified by position on the risk curve.

![Diagram comparing QE and QT showing reserves up and risk-on versus reserves down and risk-off](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-5.png)

The before/after figure above is the whole framework on one page. The left column (QE) is the expansion: the Fed buys, reserves are created, yields fall, duration is removed, and money rotates *into* risk. The right column (QT) is the contraction: bonds roll off, reserves drain, yields drift up, duration returns to the market, and money rotates *out* of risk. Every section of this post is one of those arrows.

## Common misconceptions

Balance-sheet policy attracts more confident misunderstanding than almost any topic in macro. Here are the four worth correcting, each with a number.

### Misconception 1: "QE directly funds the government deficit"

The claim: the Fed prints money and hands it to the Treasury to pay for government spending. The reality is more indirect. The Fed buys Treasuries in the **secondary market** — from dealers and investors who already own them — not directly from the Treasury at issuance (it is generally prohibited from the latter). So QE does not *directly* finance the deficit. What it *does* do is lower the government's borrowing costs by pushing down yields, and by being a giant buyer it makes it easier for the market to absorb new issuance. There is a real economic link — QE and large deficits often coincide, and QE makes deficits cheaper to fund — but the mechanical claim "the Fed prints money and gives it to the Treasury" is false. The money the Fed creates is reserves that go to *bond sellers*, not to the Treasury's spending account. The interplay between Treasury issuance and Fed liquidity is its own deep topic, covered in [Treasury issuance: bills, coupons, and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain).

### Misconception 2: "QE is automatically, mechanically inflationary"

The claim: the Fed created trillions, so massive consumer inflation is guaranteed. But look at the record. The Fed did roughly **\$3.6 trillion** of QE from 2008 to 2014, and US inflation stayed *below* the Fed's 2% target for most of that decade. Why? Because QE creates **reserves**, which are trapped in the banking system and do not mechanically become spending. For QE to cause consumer inflation, banks must *lend* the reserves into the real economy and borrowers must *spend* — and after 2008, with the economy weak and banks cautious, that lending simply did not happen at scale. The 2021-2022 inflation surge (CPI peaked at **9.06%** in June 2022 per `data.CPI_PEAK`) was real and partly liquidity-driven, but it had a crucial *extra* ingredient that the 2010s lacked: enormous **fiscal** transfers — stimulus checks, expanded benefits — that put money directly into households' hands to *spend*, on top of supply-chain shocks. QE was a contributor, not the sole cause. The lesson: *QE alone inflates asset prices reliably; it inflates consumer prices only when paired with money that actually reaches and is spent by the public.*

### Misconception 3: "QT must crash stocks"

The claim: QE pumped stocks up, so QT must pump them back down symmetrically. The 2022-2024 record refutes this directly. The Fed drained roughly **\$2.3 trillion** of assets over that period — the largest tightening of its balance sheet in history — and yet the S&P 500 made **new all-time highs** in 2024. How? Because, as we noted, *where* QT drains matters more than *how much* it drains. Through 2023-2024, the drain came overwhelmingly out of the **reverse-repo parking lot** (a pool of idle cash that fell from \$2.55T toward zero) rather than out of bank reserves. Bank reserves — the fuel that matters for risk — stayed ample. So the system tightened on paper while the liquidity that actually matters for markets barely fell. QT is a headwind, not an automatic crash; its impact depends on the reserve buffer.

### Misconception 4: "The size of the balance sheet is what matters"

The claim: just watch whether the balance sheet is big or small. But the *level* is far less informative than the *direction and composition*. A \$7T balance sheet that is *rising* is a tailwind; a \$7T balance sheet that is *draining* is a headwind — same level, opposite signal. And the composition (how much is in reserves versus parked in the reverse-repo facility or the Treasury's cash account) determines how much of the headline actually reaches markets. This is exactly why pros compute **net liquidity** — Fed assets minus the reverse-repo facility minus the Treasury cash account — rather than watching the headline. We build that number step by step in the [central bank balance sheet deep dive](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga).

## How it shows up in real markets

Theory is cheap. Here is how balance-sheet policy actually played out in the clearest case study in financial history: 2020-2024.

### The 2020-2021 QE flood

The setup: in March 2020 the policy rate was slammed to zero (the emergency cut to 0.25% on March 15, 2020) and the conventional tool was instantly exhausted. The Fed reached for the balance sheet, announced open-ended QE on March 23, and over the next two years expanded its assets from \$4.3T to \$8.96T.

The result was the everything-rally we opened with. But the *texture* of that rally is the lesson: it was sorted perfectly by position on the risk curve. Treasuries, the safest assets, barely moved in price (their yields just fell). Investment-grade credit rallied. Equities doubled. The Nasdaq, full of long-duration growth, rose 130%. And crypto, the furthest point on the curve, went up over tenfold. The *ordering* of the gains is the signature of the portfolio-balance channel: money cascading down the risk ladder lifts the bottom rung the most. If you understood the mechanism, the *ranking* of winners was predictable even if the exact magnitudes were not.

### The 2022-2024 QT drain (and the RRP cushion)

In 2022 everything reversed. Inflation had surged to a 40-year high (CPI 9.06% in June 2022), and the Fed pivoted to the most aggressive tightening in four decades: rate hikes from 0.25% to 5.50%, *plus* QT draining the balance sheet at up to \$95B/month. The portfolio-balance channel ran in reverse — duration came back to the market, the 10-year surged from under 1% to 4.88%, the discount rate spiked, and the exact assets QE had lifted the most fell the hardest. Nasdaq −33%, Bitcoin −75%, speculative tech −60% to −80%. Picture-perfect reversal.

But then came the twist that humbled anyone who expected QT to keep crushing stocks. From late 2022 through 2024, the QT drain came almost entirely out of the **overnight reverse-repo facility (RRP)** — a parking lot where money-market funds had stashed over **\$2.55 trillion** of idle cash at the December-2022 peak. As QT progressed and the Treasury issued mountains of new bills paying more than the RRP rate, that parked cash poured *out* of the RRP and into bills, *refilling* the system even as the Fed drained it. Because the drain came from the parking lot rather than from bank reserves, the liquidity that actually matters for risk stayed ample. The result: the Fed shrank its balance sheet by \$2.3T, and the S&P made new highs in 2024. **The RRP was the cushion that absorbed the QT blow.** This is the single most important nuance in modern balance-sheet trading, and it is why "watch the reserve and RRP composition, not the headline" is the core discipline.

#### Worked example: why QT drained \$2.3T but reserves barely fell

Trace the composition with the real numbers, because this is the calculation that separates people who understood 2022-2024 from people who got run over by it. The headline drain in gross assets from the April-2022 peak to mid-2025 was about **\$2.3 trillion** (\$8.96T → \$6.66T, `data.FED_ASSETS`). If that had all come out of bank reserves, it would have been brutal — reserves are the fuel that matters for risk.

But look at where it actually came from. The reverse-repo parking lot fell from its **\$2.55 trillion** peak (December 2022) toward **near zero** by 2025 — a drop of roughly **\$2.4 trillion** all by itself. So:

```
gross assets drained         ≈ -$2.3 trillion   (the QT runoff)
reverse-repo parking lot fell ≈ -$2.4 trillion   (cash leaving the lot into bills)
therefore bank reserves       ≈ roughly flat / even slightly higher
```

The arithmetic is almost poetic: the parking lot deflated by *more* than the entire QT drain, so the cash that the Fed pulled out of one liability bucket was *more than replaced* in the bucket that matters. Bank reserves, around \$3.0-3.5T through the period (`data.BANK_RESERVES`), stayed firmly in the "ample" zone. **The intuition to keep:** QT is only as tight as the *reserve* drain it causes — and when a parking lot of idle cash is there to absorb the runoff, a \$2.3T headline tightening can leave the fuel tank essentially full.

![Line chart of the 10-year Treasury yield showing the 2020 QE-era low near 0.6 percent and the 2022 to 2023 QT-era rise above 4.8 percent](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-6.png)

The yield chart above is the price-side echo of the whole cycle. The green trough at the left is the QE-era floor — the 10-year pinned at **0.62%** in July 2020 as the Fed absorbed duration. The amber surge to **4.88%** in October 2023 is the QT-and-hikes era, the term premium rebuilding as duration flowed back to the market. If you only watched one price to read the balance-sheet regime, the 10-year yield would not be a bad choice — though, as we will see in the playbook, watching the quantity directly is better.

### The broader pattern: a global phenomenon

This is not a US-only story. Every major central bank — the European Central Bank, the Bank of Japan, the Bank of England — ran QE after 2008 and especially after 2020, and global risk assets have tracked the *sum* of global central-bank balance sheets remarkably well. When the world's central banks are collectively expanding, global liquidity rises and risk assets grind higher; when they collectively tighten, the tide goes out. We develop this worldwide view in [Global liquidity: the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide). The Bank of Japan is the extreme case — it ran QE so large and so long that it ended up owning over half the entire Japanese government bond market, removing nearly all the duration from that market for years.

The global dimension also creates the most important *coordination* signal for a trader. Because the dollar is the world's reserve currency, US balance-sheet policy spills across borders: US QE floods the world with cheap dollars that flow into emerging markets and global risk assets, while US QT pulls those dollars back home and tightens conditions *everywhere*, not just in the US. This is why an emerging-market crisis often coincides with US tightening — the dollars that funded the boom are being repatriated. The flip side is that a *divergence* between central banks (one easing while another tightens) is what powers the biggest currency moves: in 2022-2024 the Fed was tightening hard while the Bank of Japan kept rates pinned near zero and stayed in QE, and the yen weakened relentlessly from around 115 to over 157 per dollar as a result. When you read the balance sheet, read it *relative* to the rest of the world's central banks — the gap is where the trade lives.

## How to trade it: the balance-sheet playbook

Everything above earns its keep here. This is how a trader actually uses balance-sheet policy. The discipline is not to predict the Fed — it announces its plans well in advance — but to *position correctly* around the trajectory it has already laid out.

![Pipeline diagram of the balance-sheet playbook showing read the trajectory, compute net liquidity, tailwind versus headwind, and the reserves-scarce invalidation](/imgs/blogs/qe-vs-qt-how-balance-sheet-policy-moves-markets-7.png)

**1. Read the trajectory, not the level.** The single most important habit. Watch the *direction* of the Fed's balance sheet (FRED series **WALCL**, updated weekly). Rising = liquidity being added = tailwind for risk. Draining = liquidity being withdrawn = headwind. A \$7T balance sheet rising and a \$7T balance sheet falling are opposite signals. The Fed pre-announces QE and QT programs, so the *direction* is usually known months ahead — your edge is positioning for it, not forecasting it.

**2. Compute net liquidity, not the headline.** Subtract the two big parking lots from the headline: **net liquidity = Fed assets − reverse-repo balance (RRPONTSYD) − Treasury cash account (WTREGEN)**. This is the cash that actually reaches markets, and it tracks risk assets far better than the gross number. The 2022-2024 lesson is the whole reason: gross assets fell \$2.3T, but because the drain came out of the RRP, net liquidity barely moved and stocks rose. Three public series, one subtraction, weekly — this is the core dashboard, built fully in the [balance-sheet deep dive](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga).

**3. Lead and lag the risk assets correctly.** When liquidity is rising, the *highest-beta* assets respond most — lean into the long-duration end (growth equities, crypto) for the most amplified upside. When liquidity is draining, the same assets fall hardest, so trim the high-beta tail *first* and rotate toward shorter-duration, cash-flowing assets and the safety of the front end. The ordering is consistent: crypto and speculative growth lead both the rally and the drawdown. Use them as the *early-warning* assets — they turn before the index does.

**4. Watch the bond market for the duration signal.** The 10-year yield and the term premium are the cleanest live read on whether the Fed is absorbing or releasing duration. Falling long yields with a compressing term premium confirm QE is biting; rising long yields with a rebuilding term premium confirm QT is biting. When the balance-sheet direction and the yield direction agree, the signal is strong; when they diverge, growth or inflation expectations are overriding the supply effect — a flag to dig deeper.

**5. The invalidation: scarce reserves.** This is the one thing that breaks the whole framework, and it is non-negotiable to monitor. Net liquidity is a clean tailwind/headwind dial *only while reserves are ample.* If QT drains reserves *past* the lowest comfortable level, the relationship snaps: small further drains cause large funding-rate spikes, the plumbing seizes, and the Fed is forced to *reverse* QT — exactly what happened in September 2019, when the overnight repo rate exploded from ~2% to nearly 10% in a single morning and the Fed had to restart purchases within weeks. The watch signals are the spread of repo rates (SOFR) over the Fed's floor, and any sign reserves are approaching the estimated scarce level (around \$3T for the post-2020 system). When reserves go scarce, the playbook inverts: a liquidity *drain* stops being an orderly headwind and becomes the trigger for a forced Fed *easing* — which is, paradoxically, bullish for risk once it arrives.

**6. Front-run the announcement, not the buying.** Because the signaling channel front-runs the mechanical channel, the *announcement* of a regime change (a QE launch, a QT taper or end) moves markets more than the flows themselves. The March 2020 "in the amounts needed" reversed the market the day it was said, before any meaningful buying. So the trade is around the *communication* — the FOMC statements, the meeting minutes, the chair's signals about the balance-sheet path — not the weekly flow data alone. The flow data confirms; the words lead.

Put it together and the discipline is simple to state and hard to execute: **read the trajectory of the balance sheet and the composition of reserves, compute net liquidity weekly, lean risk-on when it rises and trim the high-beta tail first when it drains, confirm with the bond market's duration signal, and treat scarce reserves as the hard invalidation that flips the whole framework.** Balance-sheet policy is the master variable of modern macro precisely because, when rates are at zero, it is the *only* lever left — and 2020-2024 showed, with brutal clarity, exactly how far that lever can move every asset you trade.

## Further reading & cross-links

- [Reading the central bank balance sheet: reserves, RRP, TGA, and net liquidity](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) — build the net-liquidity dashboard step by step from three public series.
- [Global liquidity: the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) — why risk assets track the *sum* of global central-bank balance sheets, not just the Fed's.
- [Quantitative easing explained: is it really printing money?](/blog/trading/finance/quantitative-easing-explained-printing-money) — the beginner's primer on the "printing money" question this post answers in depth.
- [The central bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — how the balance-sheet tools fit alongside the rate lever and communication.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the conventional tool that QE replaces when it hits the zero lower bound.
- [Treasury issuance: bills, coupons, and the liquidity drain](/blog/trading/macro-trading/treasury-issuance-bills-coupons-liquidity-drain) — how government borrowing interacts with Fed liquidity and the reverse-repo cushion.
- [What moves exchange rates: rates, flows, and carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — the relative-policy mechanism behind the dollar's QE/QT response.

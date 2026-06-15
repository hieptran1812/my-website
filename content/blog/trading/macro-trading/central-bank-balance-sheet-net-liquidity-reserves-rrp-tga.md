---
title: "Reading the Central Bank Balance Sheet: Reserves, RRP, TGA, and Net Liquidity"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into the Fed's balance sheet — what its assets and liabilities really are, what bank reserves, the reverse-repo facility, and the Treasury's cash account do — and how to compute the one derived number, net liquidity, that tracks risk assets startlingly well."
tags: ["macro", "monetary-policy", "liquidity", "federal-reserve", "balance-sheet", "reverse-repo", "quantitative-tightening", "net-liquidity", "central-banks", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The Fed's balance sheet is the most-watched plumbing gauge in macro, and one derived number — **net liquidity** (Fed assets minus the reverse-repo facility minus the Treasury's cash account) — tracks risk assets startlingly well. When net liquidity rises, stocks and crypto tend to grind higher; when it drains, they struggle.
>
> - The asset side is what the Fed bought (Treasuries and mortgage bonds). The liability side is who holds the cash it created: **bank reserves**, the **overnight reverse repo (RRP)** facility, the **Treasury General Account (TGA)**, and physical currency. The two sides are always equal — that is what "balance sheet" means.
> - **Net liquidity = Fed assets − RRP − TGA.** Reserves and currency are liquidity *in* the system; RRP and TGA are cash *parked away* from markets. From Dec 2021 to Dec 2022, gross assets barely moved (\$8.76T → \$8.55T) but net liquidity fell about \$0.67T (≈\$6.21T → \$5.54T) because the RRP balloon swelled to \$2.55T.
> - QT does not have to crash stocks. From 2022 to 2024 the Fed shrank assets by roughly \$2T, yet the S&P made new highs — because the draining RRP (\$2.55T → near zero) refilled bank reserves and *cushioned* the hit. Where QT lands matters more than its size.
> - The one number to watch weekly: **net liquidity**. Track WALCL (assets), RRPONTSYD (RRP), and WTREGEN (TGA) on FRED, subtract, and watch the trend — rising is a tailwind for risk, draining is a headwind. A \$500B TGA refill is a real, datable liquidity drain.

In December 2022, a single line in an obscure Federal Reserve data series quietly hit **\$2.55 trillion**. It was not the policy rate. It was not the size of the Fed's bond portfolio. It was the balance in something called the **overnight reverse repo facility** — the RRP — a financial parking lot that almost nobody outside a trading desk had heard of two years earlier. In mid-2021 it held close to nothing. Eighteen months later it held more cash than the entire annual budget deficit of most large economies. Money-market funds were shoving over two and a half trillion dollars into it every single night and pulling it back out every morning.

Why does that matter to anyone holding stocks, bonds, or crypto? Because that \$2.55 trillion was *liquidity that had left the building*. It was cash that existed on the Fed's books but was sitting in a vault rather than chasing assets. And when, over the following two years, that balloon deflated back toward zero, the cash it released cushioned one of the most aggressive monetary tightening campaigns in modern history — the Fed shrinking its balance sheet by roughly \$2 trillion — and helped explain why stocks, which "should" have collapsed under quantitative tightening, instead made new all-time highs.

This is the secret hiding in plain sight on the central bank's balance sheet. Most people look at one number: how big is it? Is the Fed "printing" or "shrinking"? But the desk looks at a derived number — **net liquidity** — that subtracts the cash parked in the RRP and in the Treasury's checking account from the headline total. That derived number tracks risk assets far better than the headline does. By the end of this post you will be able to compute it yourself, from three public data series, and read it like a professional. We build everything from zero.

![The Fed balance sheet drawn as two balancing columns of assets and liabilities](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-1.png)

## Foundations: what a central-bank balance sheet actually is

Before any trading signal, you need a clear mental model of what a balance sheet *is*. Almost every confused argument about "money printing" comes from people who have never looked at the two sides of the Fed's books. So we start there, in plain language, with no formulas.

### A balance sheet is just a list of what you own and what you owe

Every business — a corner shop, Apple, a bank, a central bank — has a balance sheet. It has exactly two sides, and they are always equal.

- **Assets** — everything the entity *owns*: cash, buildings, loans it has made, bonds it holds.
- **Liabilities** — everything the entity *owes*: debts, deposits it must honor, IOUs it has issued.

The two sides are equal by construction, because of a third quantity called **equity** (assets minus liabilities = the owner's stake). For the Fed, equity is tiny and we can mostly ignore it, so to a first approximation: **the Fed's total assets equal its total liabilities.** Every dollar of bonds it owns is matched by a dollar of money it owes to someone.

That last sentence is the whole game, so read it twice. When the Fed buys a bond, it does not pay with money that already existed. It *creates* new money — electronically, with a keystroke — and that new money becomes a liability on its books. The bond goes on the asset side; the freshly created money goes on the liability side. The two sides rise together, in lockstep, always equal. This is the deep reason the cover figure shows two columns of the same height: it is not a coincidence or a design choice, it is an accounting identity.

Here is the everyday analogy that makes it click. Imagine you run a tiny private bank in your garage. A neighbor walks in with a \$1,000 government bond and wants cash for it. You do not actually have \$1,000 in a drawer — but you have something better: a checkbook on *your own bank*. You take the bond (it goes on your asset side, "bonds owned: \$1,000") and in exchange you write the neighbor a deposit slip crediting their account at your garage-bank with \$1,000 (that goes on your liability side, "deposits owed: \$1,000"). You created \$1,000 of money out of nothing, and your balance sheet grew by \$1,000 on both sides simultaneously. The neighbor now has a \$1,000 claim on you; you now own a \$1,000 bond. That is *exactly* what the Fed does, except the Fed's "deposit slips" are bank reserves, and unlike your garage-bank, the Fed can never run out of them because they are denominated in the very currency it issues. This is why phrases like "the Fed is out of ammunition" are misleading — the Fed can always expand its own liabilities. The constraints on the Fed are about inflation and credibility, not about running out of money.

One more consequence worth internalizing: because the two sides are always equal, *you can read the same event from either side*. "The Fed did \$1 trillion of QE" (an asset-side statement) and "the banking system gained \$1 trillion of reserves" (a liability-side statement) are the same sentence viewed from two angles. The asset side tells you *what tool* the Fed used; the liability side tells you *where the liquidity went*. Pros read the liability side, because that is where the money that moves markets actually sits. Amateurs quote the asset side, because that is the number on the news.

### The asset side: what the Fed bought

The asset side of the Fed's balance sheet is short and boring. It is overwhelmingly two things:

- **US Treasury securities** — government bonds the Fed bought in the open market. As of 2025 this is roughly \$4.2 trillion.
- **Mortgage-backed securities (MBS)** — bundles of home loans the Fed bought to support the housing market, especially after 2008 and 2020. Roughly \$2.3 trillion.

There are a few small extras (loans to banks during crises, foreign-exchange holdings, gold certificates), but for trading purposes the asset side *is* Treasuries plus MBS. When you hear "the Fed's balance sheet is \$6.5 trillion," that is the total of this column.

The Fed does not buy these bonds because it wants the interest. It buys them as a *tool*. Buying bonds is how it injects money into the system (we will see exactly how in a moment). Letting them mature without replacing them is how it withdraws money. The asset side is the lever; the action happens on the liability side.

### The liability side: where the money actually lives

This is the side almost nobody studies, and it is where all the action for a trader is. The liability side answers the question: *the Fed created all this money — who is holding it right now?* There are four main answers, and learning to read the mix between them is the entire skill this post teaches.

1. **Currency in circulation** — physical dollar bills in wallets, tills, and mattresses. Roughly \$2.3 trillion. This is slow-moving and boring; it grows gently with the economy and rarely matters for short-term trading.

2. **Bank reserves** — electronic balances that commercial banks (JPMorgan, Bank of America, and so on) hold in their accounts at the Fed. Roughly \$3.3 trillion. This is the single most important liability for liquidity, and we devote a whole section to it below. Reserves are the raw fuel of the financial system.

3. **The overnight reverse repo facility (RRP)** — a balance where money-market funds and similar institutions park spare cash at the Fed overnight. This is the \$2.55 trillion balloon from the introduction. Crucially, cash sitting here is *removed* from the rest of the system.

4. **The Treasury General Account (TGA)** — the US Treasury's own checking account at the Fed. When the government collects taxes or sells bonds, the money sits here until it is spent. Cash sitting here is also *removed* from the rest of the system.

Look again at the cover figure. The left column (assets) is one undifferentiated blue block — Treasuries and MBS. The right column (liabilities) is split into four colored bands. The two big ones for liquidity are **reserves** (cash that *is* in the system) and the two parking lots, **RRP and TGA** (cash that has *left* the system). The whole concept of net liquidity is just: take the total, subtract the parking lots, and you have the cash that actually reaches markets. We will make that precise. But first, three terms deserve their own treatment, because traders trip over them constantly.

## Bank reserves: the fuel that never leaves the banks

Bank reserves are the most misunderstood object in monetary policy, so let us be very careful.

A **reserve** is an electronic balance that a commercial bank holds in its account at the central bank. Think of it as the banks' own checking account at the Fed. When JPMorgan needs to pay Bank of America — to settle a wire, a check, an ACH transfer — it does not ship a truck of cash. It instructs the Fed to move reserves from its account to Bank of America's account. Reserves are the medium banks use to settle with each other.

Two properties of reserves are load-bearing for everything that follows:

- **Reserves never leave the banking system.** You and I cannot hold reserves. A reserve balance can move from one bank's Fed account to another's, but it can never be spent at a grocery store or wired to a non-bank. Only banks (and a handful of other institutions) hold them. When people say "the Fed printed \$3 trillion," they almost always mean *the Fed created \$3 trillion of reserves* — and those reserves are stuck inside the banking system. That is why money printing does not mechanically become consumer inflation. It is a topic we develop in depth in [What money really is: base money, broad money, and why traders watch both](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders).

- **Reserves are the system's liquidity buffer.** When reserves are *ample*, banks lend freely to each other, repo markets are calm, and short-term funding is cheap and frictionless. When reserves become *scarce*, the plumbing seizes — overnight funding rates spike, banks hoard cash, and the Fed has to intervene. This actually happened in September 2019: reserves had been drained too far by the previous round of tightening, and the overnight repo rate briefly spiked from about 2% to nearly 10% in a single morning. The Fed had to restart asset purchases just to refill reserves. That episode is the reason the Fed now obsesses over keeping reserves "ample," and it is the single most important risk we will flag in the playbook.

So when you read the liability side, the question "how much is in reserves?" is really the question "how much fuel is in the system's tank?" More reserves equals more lubrication, easier funding, and — empirically — more appetite for risk. Fewer reserves equals tighter funding and, past a certain point, stress.

### Ample versus scarce: the regime that decides everything

There is a critical non-linearity hiding in reserves, and it is the single most important subtlety in this entire post. The relationship between reserves and funding stress is *not* a smooth line. It is a hockey stick.

When reserves are **ample** — well above the level the system needs — adding or removing a few hundred billion dollars barely moves overnight funding rates. The system has plenty of buffer; QT just trims the excess. This is the "abundant reserves" regime the Fed has deliberately run since 2008, and it is why most of the 2022-2024 QT was painless. In this regime, reserves are like the water in a large reservoir: you can drain a lot before anyone downstream notices.

But there is a level — call it the **lowest comfortable level of reserves (LCLoR)** — below which the system suddenly becomes reserve-*scarce*. Cross it, and the hockey stick kicks: small further drains produce large spikes in overnight rates, because banks start hoarding reserves to meet their own liquidity rules rather than lending them out. The exact LCLoR is unknown and moves over time (it grows as the banking system grows), but for the post-2020 system it is estimated somewhere around \$3 trillion. The Fed's entire QT-design philosophy is to drain reserves *toward* this level without crossing it — to get as close as possible to scarce without tipping in.

The reason the Fed is so paranoid about this is **September 2019**, which is the cautionary tale every desk knows. By mid-2019, the Fed's previous (2017-2019) QT had drained reserves from a peak of about \$2.8T down toward \$1.4T. That turned out to be below the comfortable level for that era's system. Then on September 16-17, 2019, a corporate-tax payment date and a large Treasury settlement drained reserves and lifted the TGA on the same day — a double liquidity hit. With reserves already scarce, there was no buffer. The overnight repo rate, which normally tracks the Fed's roughly 2% policy rate, **exploded to nearly 10% intraday.** Funding markets seized. The Fed had to inject tens of billions of reserves overnight and restart asset purchases within weeks — an emergency reversal of QT it had insisted was on "autopilot" only months earlier.

The lesson, burned into every macro trader's memory: **the cost of draining reserves is roughly zero until it is suddenly catastrophic.** There is no gentle warning. This is precisely why net liquidity is a *tailwind/headwind dial while reserves are ample*, and a *broken framework the moment they go scarce*. Keep that fork in mind; it is the invalidation in the playbook.

## RRP and TGA: the two parking lots

Reserves are cash *in* the system. The RRP and the TGA are the two big places where cash gets *parked away* from the system. Understanding them is the key that unlocks net liquidity.

### The overnight reverse repo facility (RRP)

A **repo** (repurchase agreement) is a fancy name for a one-night secured loan: I give you cash, you give me a bond as collateral, and tomorrow we reverse it with a small interest payment. A **reverse repo** is the same thing from the other side. The Fed's **overnight reverse repo facility (RRP)** lets money-market funds and similar cash-rich institutions do exactly this with the Fed itself: each evening they hand the Fed cash, the Fed gives them a Treasury as overnight collateral and pays them a fixed interest rate, and the next morning it reverses.

Why would a money-market fund do this? Because it is a perfectly safe place to earn the Fed's overnight rate on idle cash. When a fund has billions of dollars it cannot deploy into anything better, the RRP is a risk-free overnight savings account backed by the Fed. The repo and reverse-repo plumbing is explored more broadly in [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).

Here is the liquidity point. **Cash in the RRP is sitting on the Fed's balance sheet, not in the markets.** When a money-market fund moves \$100 billion from buying Treasury bills into the RRP, that \$100 billion is no longer chasing assets — it has been pulled out of circulation and parked at the Fed overnight. The RRP is a *sponge*. It soaks up cash. A rising RRP means cash is leaving the system; a falling RRP means cash is being released back into it.

The RRP went from near-zero in early 2021 to **\$2.55 trillion in December 2022** because after the 2020-2021 money flood, the financial system was awash in cash with nowhere to go, and the Fed's RRP rate made it the most attractive overnight home for trillions of idle money-fund dollars. Then, as the Fed raised rates and Treasury issued mountains of new bills paying *more* than the RRP, that cash drained out of the parking lot and into bills — pushing the RRP back toward zero by 2025.

It is worth understanding *why* the balloon inflated, because the mechanism is the same one that later deflated it, and the switch between the two is the heart of the 2022-2024 story. By 2021 the system had two problems at once. First, the 2020-2021 QE and fiscal stimulus had stuffed money-market funds with cash — investors parked savings there, and the funds had to invest it somewhere overnight, safely. Second, there was a *shortage* of the safe short-term assets those funds normally buy, because the Treasury had temporarily cut its bill issuance. So you had trillions of dollars hunting for a safe overnight home and not enough Treasury bills to absorb them. The RRP was the relief valve: the Fed offered to take that cash overnight at a guaranteed rate, and the funds poured in. The balloon was a *symptom of excess cash chasing too few safe assets* — a glut, parked at the Fed.

What deflated it was the mirror image. As the Fed hiked aggressively through 2022-2023 (the fastest cycle in 40 years), and as the Treasury ramped up bill issuance massively — especially after the 2023 debt-ceiling resolution — Treasury bills started paying *more* than the RRP rate. Suddenly the money-market funds had a better, still-safe option: buy the flood of new bills instead of parking at the Fed. So they pulled cash *out* of the RRP and into bills. This is why the RRP could fund the Treasury's huge 2023-2024 borrowing *without* draining bank reserves — the Treasury was, in effect, borrowing back the very cash that had been parked at the Fed. The RRP balloon and its deflation were two phases of the same supply-and-demand dance between idle cash and safe-asset supply.

### The Treasury General Account (TGA)

The **Treasury General Account (TGA)** is the federal government's operating checking account, held at the Fed. Every dollar of tax the IRS collects flows into the TGA. Every dollar the Treasury raises by selling bonds flows into the TGA. And every dollar the government spends — Social Security checks, defense contracts, interest on the debt — flows *out* of the TGA into the economy.

The same liquidity logic applies. **Cash sitting in the TGA has left the rest of the system.** When the Treasury collects \$300 billion in April tax payments, that money moves out of bank deposits and into the TGA — it drains liquidity. When the Treasury then spends it, the money flows back out into bank deposits — it adds liquidity. The TGA is a second sponge, and it is far more volatile than the RRP because it swings on tax dates, spending cycles, and — most dramatically — debt-ceiling fights.

During a debt-ceiling standoff, the Treasury is legally barred from issuing new debt, so it must run down its existing cash to keep paying the bills. That is why the TGA fell to just **\$0.05 trillion (about \$50 billion)** in June 2023, days before a default deadline — a near-empty checking account. The instant the ceiling was lifted, the Treasury rebuilt the account by issuing a tidal wave of new bills, refilling the TGA and *draining* hundreds of billions of liquidity from the system in a matter of weeks. That refill is one of the cleanest, most datable liquidity shocks a trader will ever see, and we work through its mechanics below.

## QE and QT: the asset side moves the levers

We now have all four liabilities. Next we connect the asset side (the lever) to the liability side (where liquidity lives), because that connection is **quantitative easing (QE)** and **quantitative tightening (QT)** — the two operations that dominate the modern Fed.

![Fed total assets from 2019 to 2025 with the QE peak and QT phase annotated](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-2.png)

The chart above is the headline balance sheet — the thing CNBC means by "the Fed's balance sheet." It tells a dramatic story. Pre-COVID, the Fed held about **\$4.3 trillion** of assets. Then in March 2020 the pandemic hit, and the Fed bought bonds on a scale never seen before, driving total assets to a peak of **\$8.96 trillion in April 2022** — roughly \$4.6 trillion of new asset purchases in two years. That is QE. From mid-2022 onward, the Fed reversed course, letting bonds mature without replacing them, shrinking the balance sheet back toward \$6.6 trillion by 2025. That is QT.

### What QE actually does, step by step

**Quantitative easing** is the Fed buying bonds in the open market with newly created money. Here is the mechanical sequence, which most explanations skip:

1. The Fed buys, say, \$1 billion of Treasuries from a bond dealer. The dealer's bank is JPMorgan.
2. The Fed creates \$1 billion of new reserves — out of thin air, with a keystroke — and credits them to JPMorgan's reserve account.
3. JPMorgan credits \$1 billion to the dealer's deposit account.

Now look at both sides of the Fed's books. **Assets** rose by \$1 billion (the new Treasuries). **Liabilities** rose by \$1 billion (the new reserves at JPMorgan). The balance sheet grew by \$1 billion on both sides, and \$1 billion of fresh reserves — fresh liquidity — entered the banking system. That is QE in one transaction. Repeat it \$4.6 trillion times and you get the 2020-2022 explosion. For a fuller treatment of the mechanism and its history, see [Quantitative easing explained: printing money](/blog/trading/finance/quantitative-easing-explained-printing-money).

### What QT actually does, step by step

**Quantitative tightening** is the reverse, and it is more subtle. The Fed does not usually *sell* bonds; it simply lets them mature and does not reinvest the proceeds. Here is the sequence:

1. A \$1 billion Treasury the Fed holds matures. The Treasury must repay the \$1 billion face value.
2. The Treasury pays by drawing down its TGA, which debits reserves out of the banking system.
3. The Fed's bond disappears from its asset side, and the matching reserves disappear from its liability side.

Both sides shrink by \$1 billion, and \$1 billion of reserves — liquidity — is destroyed. The diagram below lays the two operations side by side so you can see they are the same plumbing run in opposite directions.

![Before and after panels showing how QE adds reserves and QT removes them](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-5.png)

The key insight, and the reason this section exists, is that **QE and QT move the liability side, not just the asset side.** A trader who only watches the headline asset number (the previous chart) sees the *size* of the operation. But the liquidity that matters is on the liability side — and as we are about to see, the *same* QT can hit reserves hard or barely at all, depending on what the RRP and TGA are doing at the time. That is why the headline number can mislead you, and why net liquidity exists.

#### Worked example: the QE multiple — sizing the 2020-2022 flood

Let us put a number on the COVID QE. Take the Fed's total assets before the pandemic and at the QE peak, straight from the data:

- Pre-COVID (March 2020): assets ≈ **\$4.31 trillion** (`data.FED_ASSETS` at 2020-03).
- QE peak (April 2022): assets ≈ **\$8.96 trillion** (`data.FED_ASSETS_PEAK`).

The increase is \$8.96T − \$4.31T = **\$4.65 trillion** of new asset purchases in about 25 months. As a multiple, the balance sheet went from \$4.31T to \$8.96T — it **roughly doubled** (×2.08). To feel the scale: \$4.65 trillion is more than the entire annual GDP of Japan, created and deployed into bond markets in two years. Almost all of that landed on the liability side as new bank reserves, which is why reserves ballooned from about \$1.5T (2019) to over \$4T (2021) — see [How credit creates money: the lending channel and cycles](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles) for what that flood of reserves did and did not do downstream.

**Intuition:** the Fed doubled its balance sheet in two years, and that \$4.65 trillion of new money is the tide that lifted essentially every asset — stocks, bonds, housing, crypto — into 2021.

## The liability side is where liquidity actually lives

We have hinted at this repeatedly; now we make it the centerpiece. The reason the headline balance-sheet number is a *weak* guide to markets is that it ignores how the created money is distributed across the four liabilities. The same \$8.5 trillion of assets can correspond to very different amounts of *usable* liquidity depending on how much cash is parked in the RRP and TGA versus circulating as reserves.

Think of it as a bathtub. The asset side is the total volume of water the Fed has poured in. But two drains — the RRP and the TGA — can hold water in side-tanks, away from the main tub. What the market "feels" is the water level *in the main tub* (reserves and currency), not the total volume including the side-tanks. When the side-tanks fill, the main tub drains even if total volume is unchanged. When the side-tanks empty, the main tub rises even if the Fed adds nothing.

Let us watch the two side-tanks in action. First the RRP.

![Reverse repo balance from 2021 to 2025 showing the rise to 2.55 trillion and the drain to zero](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-3.png)

The RRP chart is the balloon-and-drain. Amber bars (2021-2022) show the parking lot *filling* — cash leaving markets and the banking system to sit at the Fed. Green bars (2023-2025) show it *draining* — cash returning to the system. The peak is the \$2.55 trillion from our introduction. Now the TGA.

![Treasury General Account balance from 2020 to 2025 including the 2023 near-zero](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-4.png)

The TGA chart is pure volatility. The lavender line lurches between roughly \$1.8 trillion and near-zero. The two annotated points tell the debt-ceiling story: the near-empty \$0.05T checking account in June 2023, and the rapid refill to \$0.68T by September 2023 that drained liquidity right back out as soon as the ceiling was lifted.

Put these two charts next to the headline-assets chart and a profound point emerges. From the asset chart, 2022 looks like a year when the Fed had barely started shrinking — assets were still near \$8.5T. But the RRP chart shows the parking lot ballooned to \$2.55T over that same year. So even though the Fed's total balance sheet was huge, a giant and *growing* chunk of the created money was sitting idle in the RRP, not circulating. The *usable* liquidity was draining even as the headline number stayed enormous. This is exactly the kind of divergence that fools traders who watch only the headline.

#### Worked example: the same QT, two very different reserve outcomes

Imagine the Fed runs \$95 billion of QT this month (its actual cap for much of 2022-2024). Mechanically, \$95B of bonds mature and the Treasury repays them. Where does the \$95B come from? It can come from two places on the liability side:

- **Scenario A (RRP cushions it):** money-market funds, seeing Treasury issue lots of new bills, move \$95B *out of the RRP* into those bills. The Treasury gets its cash from RRP money, not from bank reserves. Result: RRP falls \$95B, **bank reserves are untouched**, and the banking system feels no liquidity loss. QT happened, but the main tub did not drop — the side-tank drained instead.
- **Scenario B (RRP is empty):** the RRP is already near zero, so there is no parked cash to tap. The \$95B must come straight out of bank reserves. Result: **reserves fall \$95B**, the main tub drops, funding tightens, and risk assets feel it.

Same \$95B of QT, completely different market impact — purely because of *where on the liability side* the cash came from.

**Intuition:** QT only bites when the RRP buffer is empty. While the RRP is draining, QT is nearly painless; once it hits zero, every dollar of QT comes out of reserves and the pain begins. This single mechanism explains the entire 2022-2024 market.

## Net liquidity: assets minus RRP minus TGA

We are ready for the payoff. **Net liquidity** is the desk's name for the usable liquidity in the system — the water level in the main tub. The standard trader proxy is dead simple:

```
Net liquidity = Fed total assets  −  RRP balance  −  TGA balance
```

Read it as: start with all the money the Fed created (assets), then subtract the two parking lots where cash is held away from markets (RRP and TGA). What remains is, roughly, the cash that is actually circulating and available to support asset prices. It is a *proxy*, not a law of physics — but it has tracked risk assets remarkably well since 2021, which is why every macro desk now watches it.

![Net liquidity computed as assets minus RRP minus TGA showing the 2022 drain and stabilization](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-6.png)

The chart above plots both the gross assets (dashed gray) and net liquidity (solid blue). Notice how *differently* they behave in 2022. Gross assets are nearly flat — the Fed had barely begun QT. But net liquidity falls sharply, because the RRP balloon was soaking up cash faster than the headline suggested. Then in 2024-2025, the opposite: gross assets keep falling under QT, but net liquidity *stabilizes and even ticks up*, because the draining RRP releases cash back into the system faster than QT removes it. The blue line — net liquidity — is the one that lined up with the market: stocks struggled in 2022 as it fell, and rallied in 2023-2024 as it stabilized.

### Why should net liquidity track risk assets at all?

It is fair to be skeptical. Why would a back-of-envelope subtraction of three Fed data series have any business predicting the S&P 500? There are three honest answers, and a caveat.

The first is the **direct portfolio-balance channel**. When the system holds more usable cash (reserves), that cash does not sit idle — banks and investors deploy it. With short-term safe assets yielding little, the marginal dollar of reserves pushes investors out the risk curve: into corporate bonds, then equities, then the most speculative assets like crypto. More usable liquidity literally bids up asset prices because the cash has to go somewhere. Drain that liquidity and the bidding pressure reverses. This is not magic; it is supply and demand for assets, with central-bank liquidity setting the supply of cash on the demand side.

The second is the **funding-and-leverage channel**. Ample reserves mean cheap, frictionless overnight funding. Leveraged players — hedge funds, dealers, basis traders — finance their positions in repo markets that run on reserves. When reserves are abundant, leverage is cheap and plentiful, which supports asset prices; when reserves tighten, funding costs rise, leverage gets pulled, and forced de-leveraging hits prices. Net liquidity is, in part, a proxy for how easy it is to be leveraged.

The third is **reflexivity and positioning**. Because so many traders now *watch* net liquidity, it has become partly self-fulfilling: when the number turns up, desks lean risk-on, and that flow itself lifts prices. A signal that enough people trade becomes a flow.

The caveat — and it is important — is that net liquidity is a *correlation that has held strongly since 2021, not an iron law*. It worked beautifully because 2021-2024 was a period where liquidity was the dominant macro driver. In a period dominated by an earnings collapse, a credit event, or a geopolitical shock, net liquidity can rise while stocks fall. Treat it as a powerful regime dial — usually the most important single one for risk appetite — but not the only variable, and never a same-day timing tool.

#### Worked example: computing net liquidity at two dates

Let us actually compute it, using the curated series, at two endpoints: end of 2022 and end of 2024.

**December 2022:**
- Fed assets (`data.FED_ASSETS` at 2022-12): **\$8.55T**
- RRP (`data.ON_RRP` at 2022-12, the peak): **\$2.55T**
- TGA (`data.TGA`, nearest reading, 2023-01): **\$0.46T**
- Net liquidity = 8.55 − 2.55 − 0.46 = **\$5.54 trillion**

**December 2024:**
- Fed assets (`data.FED_ASSETS` at 2024-12): **\$6.87T**
- RRP (`data.ON_RRP` at 2024-12): **\$0.16T**
- TGA (`data.TGA` at 2024-12): **\$0.72T**
- Net liquidity = 6.87 − 0.16 − 0.72 = **\$5.99 trillion**

Here is the punchline. Between these two dates, **gross assets fell by \$1.68 trillion** (8.55 → 6.87) — a massive QT campaign. Yet **net liquidity actually *rose* by about \$0.45 trillion** (5.54 → 5.99). The headline screamed "the Fed is draining \$1.7 trillion, sell everything." Net liquidity quietly said "usable liquidity is flat to higher, the drain is being absorbed by the emptying RRP." The market followed net liquidity: the S&P 500 rose roughly 50% across 2023-2024.

**Intuition:** the headline balance-sheet number and net liquidity told *opposite* stories over 2022-2024. The traders who watched the headline got the regime wrong; the traders who watched net liquidity got it right.

#### Worked example: a \$500B TGA refill draining reserves dollar-for-dollar

Now a clean, datable shock. Suppose a debt-ceiling fight ends and the Treasury must rebuild its checking account from a near-empty \$50 billion up to a \$550 billion target — a **\$500 billion refill**. How does that hit liquidity?

The Treasury raises the \$500B by issuing new bills. Investors — money-market funds, banks, individuals — buy those bills. They pay for them, and that payment flows *into the TGA*. Trace the dollars:

- If buyers pay with cash that was sitting in the **RRP** (\$X of the \$500B), then RRP falls \$X and reserves are untouched — a painless refill.
- If buyers pay with cash from **bank deposits/reserves** (the remaining \$500B − \$X), then **reserves fall by that amount, dollar-for-dollar.** The TGA rose \$500B; whatever portion did not come from the RRP came straight out of the system's reserves.

In net-liquidity terms it is automatic: TGA went *up* \$500B, so `Net liquidity = assets − RRP − TGA` falls by \$500B minus whatever the RRP absorbed. If the RRP is empty (as it largely was by 2024-2025), the entire \$500B comes out of reserves and net liquidity drops a full \$500 billion. That is a serious, foreseeable headwind — and unlike a vague "QT is bearish" claim, you can see the refill schedule coming in the Treasury's quarterly refunding announcements.

**Intuition:** a TGA refill is a liquidity drain you can put on your calendar; when the RRP buffer is gone, every dollar of refill comes out of reserves and net liquidity, one-for-one.

#### Worked example: the RRP cushion that made QT painless (2022-2024)

Finally, let us quantify the cushion that defined the whole tightening cycle. Over 2022-2024 the Fed drained gross assets by roughly **\$2 trillion** of QT. If that had all come out of bank reserves, the system would have lurched toward the scarcity that caused the September 2019 repo spike. It did not — because the RRP did the heavy lifting.

- RRP peak (`data.ON_RRP_PEAK`, Dec 2022): **\$2.55T**
- RRP by late 2024 (`data.ON_RRP` at 2024-12): **\$0.16T**
- RRP drained ≈ **\$2.39 trillion**

So while QT pulled roughly \$2 trillion off the asset side, the RRP released roughly \$2.4 trillion of parked cash back into the system over the same window. The two roughly cancelled: bank reserves, instead of collapsing, stayed broadly ample (around \$3.0-3.5 trillion through the period, per `data.BANK_RESERVES`). The QT that "should" have crushed liquidity was almost entirely absorbed by the draining parking lot.

**Intuition:** the \$2.55T RRP balloon was, in hindsight, a giant liquidity shock-absorber. The Fed could run aggressive QT for two years with minimal reserve pain *because* there was a parking lot to drain first. The danger comes *after* it empties.

## Common misconceptions

The balance sheet attracts more confident-but-wrong takes than almost any other macro topic. Here are the big ones, each corrected with a number.

### Misconception 1: "QT must crash stocks"

The intuition is that the Fed withdrawing money should sink risk assets. The data says otherwise. From the QT start in mid-2022 through the end of 2024, the Fed shrank gross assets by roughly **\$2 trillion** — and the S&P 500 *rose* to new all-time highs across 2023-2024. Why? Because, as our worked example showed, net liquidity barely fell (and at points rose) since the draining \$2.55T RRP offset the QT. **QT crashes stocks only when it drains *reserves*, which only happens once the RRP and other buffers are exhausted.** The size of QT is not the signal; its impact on net liquidity is.

### Misconception 2: "The balance sheet size is all that matters"

People quote the headline number — "the Fed is at \$6.9 trillion, down from \$8.96 trillion" — as if that alone tells you the liquidity stance. It does not. Between Dec 2022 and Dec 2024, gross assets fell \$1.68T but **net liquidity rose \$0.45T** because the RRP and TGA mix shifted. Two days with identical headline balance sheets can have wildly different net liquidity depending on how much cash is parked in the RRP and TGA. **Always look at the composition of the liability side, not just the asset total.**

### Misconception 3: "A draining RRP is bearish"

Some commentators panicked as the RRP fell, treating a shrinking Fed-facility balance as a sign of stress. Backwards. A draining RRP means cash is *leaving* the parking lot and *re-entering* the system — it is a liquidity *tailwind*. The RRP fell from \$2.55T to near zero across 2023-2024, and that drain *added* roughly \$2.4 trillion of usable liquidity, cushioning QT. A rising RRP (cash fleeing into the parking lot, as in 2021-2022) is the bearish-for-liquidity move; a falling RRP is supportive — right up until it hits zero and the buffer is gone.

### Misconception 4: "The TGA is just government accounting, not a market variable"

The TGA looks like dull plumbing — the government's checking account — so traders ignore it. That is a mistake worth hundreds of billions. A TGA *refill* (after a debt-ceiling fight, or seasonal tax inflows) pulls cash out of the system dollar-for-dollar once the RRP is empty. The 2023 post-debt-ceiling refill alone moved the TGA from \$0.05T to \$0.68T — a **\$630 billion** swing — and that is a direct, calendar-able liquidity drain. The TGA is one of the most *forecastable* liquidity variables there is, because the Treasury publishes its cash targets in advance.

### Misconception 5: "Money printing immediately means inflation"

This one bridges to the broader money story. The Fed created roughly \$4.6 trillion of reserves in 2020-2022, but reserves *cannot leave the banking system* — you and I never get to hold or spend them. They became consumer inflation only to the extent that they supported *broad* money creation (bank lending and the fiscal transfers of 2020-2021), not because reserves themselves are spendable. The relationship between base money (reserves) and the inflation that actually hits prices is loose and lagged; we unpack it fully in [What money really is](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders). For the balance sheet specifically: a bigger liability side is a *liquidity* signal for asset prices, not an automatic *inflation* signal for consumer prices.

## Beyond the Fed: the same plumbing everywhere

Net liquidity is a Fed-specific recipe, but the underlying logic — *a central bank's liabilities are where system liquidity lives* — generalizes to every major central bank, and a complete macro trader watches the *global* picture, not just the US one. Three points make this practical.

First, **other central banks have the same balance-sheet structure**, even if the parking-lot facilities differ. The European Central Bank (ECB), the Bank of Japan (BOJ), and the Bank of England (BOE) all hold bonds on the asset side and owe bank reserves on the liability side. The BOJ in particular ran QE so extreme that by the mid-2020s it owned a balance sheet larger than Japan's entire economy and held over half of all outstanding Japanese government bonds — a scale that dwarfs the Fed's relative to GDP. When the BOJ finally began normalizing and let its yield-curve-control peg go, the resulting shifts rippled through global bond and currency markets, because Japanese liquidity had been a key funding source for the worldwide "carry trade."

Second, **global liquidity is additive and fungible at the margin.** Money created by the BOJ or the People's Bank of China does not stay home; some of it chases higher returns abroad, including into US stocks and bonds. So a sophisticated version of net liquidity sums the major central banks' balance sheets (converted to a common currency and adjusted for their parking lots) into a *global* liquidity aggregate. When global central-bank balance sheets are collectively expanding, risk assets worldwide tend to have a tailwind; when they are collectively shrinking, a headwind. The Fed is the biggest single driver because the dollar is the world's reserve currency — a dominance explored in [Petrodollar and dollar dominance](/blog/trading/finance/petrodollar-and-dollar-dominance) — but it is not the only one.

Third, **the dollar exchange rate is part of the liquidity transmission.** When the Fed drains liquidity and the dollar strengthens, it tightens financial conditions globally, because so much of the world's debt is dollar-denominated: a stronger dollar makes that debt harder to service. This is why a Fed liquidity drain often shows up first not in US stocks but in *emerging-market* stress and a rising dollar index. A trader watching net liquidity should also glance at the dollar — they are two windows onto the same tightening.

The takeaway is not that you must track six central banks every week. It is that net liquidity is one instrument in a global orchestra, and the loudest non-Fed risks — a BOJ normalization, a Chinese liquidity injection, a sharp dollar move — can override the US signal. When the Fed's net liquidity and the global picture disagree, respect the bigger force.

## How it shows up in real markets

Theory is cheap. Here is how net liquidity played out in three concrete, datable episodes that defined 2022-2024.

### 2022: the headline lied, net liquidity told the truth

Through 2022, the Fed's headline balance sheet looked nearly unchanged — assets hovered near \$8.5-8.9T, and QT had only just begun in the summer at a slow pace. A trader watching only the headline would have concluded "the Fed has barely started tightening its balance sheet, liquidity is fine." But net liquidity was *falling hard*: the RRP ballooned from \$1.60T (Dec 2021) to \$2.55T (Dec 2022), soaking up nearly a trillion dollars even as gross assets barely moved.

Compute it: net liquidity went from about \$6.21T (Dec 2021: 8.76 − 1.60 − 0.95) to about \$5.54T (Dec 2022: 8.55 − 2.55 − 0.46) — a drain of roughly **\$0.67 trillion** that the headline completely hid. And what did risk assets do? The S&P 500 fell about 19% in 2022, its worst year since 2008. Bitcoin fell about 65%. The headline said "fine"; net liquidity said "draining"; the market followed net liquidity. This is the single best argument for computing the proxy yourself. The rate side of this story — the fastest hiking cycle in 40 years, which made the RRP so attractive — is in [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

### 2023-2024: QT drained \$1.7T but the RRP cushioned it

This is the episode that breaks most people's mental model. Across 2023-2024 the Fed ran QT relentlessly, shrinking gross assets by roughly **\$1.7 trillion** (from \$8.55T to \$6.87T). Conventional wisdom said this should starve markets of liquidity and tank stocks.

Instead, the RRP drained from \$2.55T to \$0.16T over the same window — releasing about **\$2.4 trillion** of parked cash back into the system. That release *more than offset* the QT. Net liquidity, as we computed, actually rose modestly from \$5.54T to \$5.99T. Bank reserves stayed ample (roughly \$3.0-3.5T). And risk assets boomed: the S&P 500 gained roughly 24% in 2023 and another ~25% in 2024; Bitcoin roughly tripled. **QT with a draining RRP is a paper tiger.** The whole cycle is a case study in why composition beats headline size.

### The TGA refill shocks

Layered on top were the TGA swings. The 2023 debt-ceiling episode drove the TGA to a near-empty \$0.05T in June 2023 — which, perversely, was *supportive* for liquidity while it lasted, because a low TGA means cash is in the system rather than parked. Then the deal passed, the Treasury flooded the market with bills, and the TGA refilled to \$0.68T by September 2023 — a roughly **\$630 billion** drain over a few months. Markets wobbled into the autumn of 2023 (the S&P fell about 10% from its July high) partly on this drain plus a Treasury-supply scare, then resumed their climb once the refill was complete and the RRP kept absorbing the supply. The TGA is the variable that explains the *timing* of liquidity air-pockets within the broader trend.

## How to trade it: the net-liquidity playbook

Everything above resolves into a weekly routine. This is the playbook — the signal, the position, and crucially the invalidation.

![A four-gauge dashboard turning Fed assets, RRP, and TGA into a risk-on or risk-off lean](/imgs/blogs/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga-7.png)

The dashboard above is the whole post in one picture: three input gauges (assets, RRP, TGA) feed one derived number (net liquidity), which feeds a risk-on or risk-off lean, with an explicit invalidation. Here is how to run it.

### The weekly routine

Every week — the relevant Fed data updates Thursdays (balance sheet, the H.4.1 release) and the RRP/TGA are effectively daily — pull three numbers from FRED:

- **WALCL** — Fed total assets (the headline).
- **RRPONTSYD** — the overnight reverse repo balance.
- **WTREGEN** — the Treasury General Account balance.

Compute `Net liquidity = WALCL − RRPONTSYD − WTREGEN`. Plot the trend over the last 6-12 months. **You do not care about the absolute level; you care about the trend and the rate of change.** Rising net liquidity is a tailwind for risk assets (stocks, credit, crypto); draining net liquidity is a headwind. Treat it as a slow *regime dial*, not a same-day timing trigger — it tells you which way the wind is blowing, not when to enter.

A concrete way to operationalize it: compute net liquidity each week and look at its **4-week and 12-week change**. A persistent positive 12-week change with a positive 4-week change is a clean risk-on tailwind. A 12-week change that has turned negative — especially with the RRP near its floor — is your warning that the easy regime is ending. Overlay the net-liquidity line on the S&P 500 or your risk benchmark and you will usually see them move together with the liquidity line leading by a few weeks; the rare *divergences* (liquidity up, price down, or vice versa) are the most informative moments, because one of the two is about to correct. When they diverge, ask *why* — is an earnings or credit shock overriding liquidity (respect the shock), or is the market lagging a liquidity turn it has not priced yet (lean with liquidity)?

### Watch the two scheduled drains

Net liquidity does not move randomly. Its two biggest swing factors are largely *forecastable*, which is the edge:

- **TGA rebuilds.** The Treasury publishes its cash-balance targets in its **Quarterly Refunding Announcements** and Daily Treasury Statements. A planned TGA refill (after a debt-ceiling resolution, or around quarterly tax dates in April, June, September, and January) is a scheduled liquidity drain. Mark these on your calendar and expect a liquidity air-pocket — especially now that the RRP buffer is gone.
- **The RRP floor.** The single most important regime question is: *is there still cash in the RRP to cushion QT and bill supply?* While the RRP is draining (it fell from \$2.55T toward near-zero through 2024-2025), QT and TGA refills are absorbed painlessly. **Once the RRP is empty, the buffer is gone, and every dollar of QT or TGA refill comes straight out of bank reserves.** That transition — RRP hitting its floor — is the moment the regime flips from "QT is painless" to "QT bites." It is the most important thing to watch in the entire framework.

### The position and the lean

- **Net liquidity rising, RRP still has buffer:** risk-on lean. The tide is coming in and QT is cushioned. Favor beta — equities, credit, crypto. This was the 2023-2024 regime.
- **Net liquidity draining, RRP empty:** risk-off lean. The buffer is gone and drains hit reserves directly. Trim risk, raise cash, expect funding stress to show up in widening spreads and a rising VIX. This is the regime to fear.
- **A scheduled \$300-500B+ TGA refill with an empty RRP:** brace for a datable drain. Lighten up *before* the refill window, not during it.

### The invalidation

Every framework needs a line that says "I am wrong, or the regime has fundamentally changed." For net liquidity, it is **reserve scarcity**. Net liquidity is a tailwind/headwind framework *while reserves are ample*. If bank reserves fall toward the scarcity zone — historically estimated around **\$3 trillion** for the current system, though the exact level is uncertain and the Fed watches it closely — the framework flips from "smooth liquidity dial" to "funding-stress alarm." The September 2019 repo spike is the warning: reserves got drained too far, overnight rates exploded to nearly 10%, and the Fed had to abruptly restart purchases. **If you see reserves approaching scarcity and repo rates twitching above the Fed's policy band, the gentle net-liquidity dial no longer applies — that is the signal to de-risk hard, because the plumbing itself is at risk.** The Fed knows this, which is why it ended QT and stands ready to add reserves the moment scarcity threatens. Watch the SOFR rate relative to the Fed's interest-on-reserves rate: persistent upward pressure there is the canary.

### Putting it together

The discipline is simple to state and hard to follow: **ignore the headline balance-sheet size; compute net liquidity; watch its trend and its two scheduled drains (TGA refills and the RRP floor); lean risk-on when it rises with buffer and risk-off when it drains with no buffer; and abandon the whole framework the instant reserves approach scarcity.** A trader running this in 2022 would have respected the net-liquidity drain and stayed defensive while the headline looked calm. A trader running it in 2023-2024 would have leaned risk-on while the crowd feared QT. That is the entire edge of reading the central bank's balance sheet the way a desk does — not as a headline, but as a composition.

## Further reading & cross-links

- [What money really is: base money, broad money, and why traders watch both](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) — why reserves (the dominant balance-sheet liability) are base money that cannot leave the banking system, and why that breaks the naive "printing equals inflation" chain.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the rate side of the 2022-2024 story, and why the Fed's RRP rate made the parking lot so attractive that it swelled to \$2.55T.
- [Quantitative easing explained: printing money](/blog/trading/finance/quantitative-easing-explained-printing-money) — the deep history and mechanics of QE that built the \$8.96T asset side in the first place.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — how the RRP and interest-on-reserves rates form the floor of the Fed's rate-control system, the same tools that drive the balance-sheet flows here.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the repo plumbing the RRP plugs into, and where reserve scarcity shows up first as funding stress.
- [How credit creates money: the lending channel and cycles](/blog/trading/macro-trading/how-credit-creates-money-lending-channel-cycles) — what the flood of reserves did (and did not do) once it reached the banking system.

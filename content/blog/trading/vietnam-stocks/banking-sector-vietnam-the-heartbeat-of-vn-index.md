---
title: "Banking: The Heartbeat of VN-Index"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero deep dive into how a Vietnamese bank actually makes money, why banks are about a third of VN-Index, how to value one with P/B and ROE, and how to trade the sector through the credit cycle."
tags: ["vietnam-stocks", "sector-rotation", "vn-index", "banking-sector", "nim", "casa", "credit-growth", "asset-quality", "price-to-book", "bank-valuation", "vietcombank", "credit-ceiling"]
category: "trading"
subcategory: "Vietnam Stocks"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank is a margin machine: it borrows cheaply (deposits), lends dearly (loans), and pockets the spread (NIM); banks are about 34% of VN-Index, so when banks move, the index moves.
>
> - A bank's profit is mostly **net interest income** — its lending rate minus its funding cost, multiplied by the size of its loan book — plus fee income, geared to how fast credit can grow.
> - In Vietnam, loan growth is rationed every year by the State Bank's **credit ceiling**, the cheapest funding comes from **CASA** (current and savings accounts), and the biggest risk is **asset quality** — bad loans (NPLs) and the provisions banks must set aside against them.
> - You value a bank on **price-to-book versus return on equity**: a bank earning an 18% ROE deserves a higher multiple of its book value than one earning 11%, which is why VCB trades at 2.4x book while a thinner-margin bank trades near 1.0x.
> - The one number to remember: **banks are about a third of VN-Index**, so you cannot have a strong, durable index rally in Vietnam without the banks taking part.

On the last trading day of 2024, a Vietnamese investor checking the year-end scoreboard would have seen a familiar split-screen. The VN-Index had finished the year up about 12% — a respectable, unremarkable number. But the banking sector had run roughly twice as hard: the eighteen banks listed on the Ho Chi Minh exchange (HOSE) returned around 26% as a group, more than doubling the index. The year's headline gain was, to a large degree, a banking story dressed up as a market story. Strip the banks out and the rest of the tape looked tired.

This is not a one-off. Across the last decade, the single most reliable way to predict whether VN-Index will have a good year has been to ask one question first: *what are the banks doing?* Because banks are about 34% of the index by market value — more than a third of the entire weight sits in one industry — the banks are not merely a sector that participates in rallies. They are, mechanically, the largest single driver of whether a rally happens at all. When market commentators in Ho Chi Minh City call banks *"nhom co phieu vua"* — the "king" group of stocks — they are not being poetic. They are describing an arithmetic fact about how the index is built.

So if you are going to understand Vietnamese stocks at all, you have to understand banks first. The trouble is that banking is the sector most people understand *least*, because a bank's business is invisible. A steelmaker has furnaces; a property developer has land; a retailer has shops. A bank has a balance sheet, and almost everything that matters about it happens in the abstract space between an interest rate it pays and an interest rate it charges. This post takes that abstraction apart from absolute zero — how a bank actually earns a dong, what makes one bank far more profitable than another, why the State Bank of Vietnam rations how much each bank can grow, how to put a price on a bank, and finally how to trade the whole sector as it moves through the credit cycle and drags VN-Index along with it.

![Flow diagram of the bank profit engine from deposits through NIM and fees to profit and equity](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-1.png)

## Foundations: what a bank actually does

Start with the only thing a bank really is: a business that **buys money cheap and sells it dear.** Everything else is detail on top of that one sentence.

### Intermediation: borrowing short, lending long

When you deposit 100 million dong in a Vietnamese bank, you are not "keeping your money there" the way you keep socks in a drawer. You are *lending* that money to the bank. The bank owes it back to you — that is why your deposit is a *liability* on the bank's books, something it owes — and in exchange the bank pays you a small interest rate, perhaps 4.5% a year on a one-year term deposit. The bank then takes your money (pooled with millions of other depositors' money) and lends it back out: a mortgage to a family buying an apartment, a working-capital loan to a manufacturer, a credit line to a property developer. Those borrowers pay the bank a *higher* interest rate, perhaps 8% a year. The loans the bank makes are its *assets* — money other people owe *it*.

This is **financial intermediation**: the bank sits in the middle, between people with spare money (savers) and people who need money (borrowers), and it earns its living on the gap between the two rates. The saver gets 4.5% and the convenience of a safe, liquid account; the borrower gets 8% financing and the convenience of not having to find a lender personally; the bank keeps the 3.5-point gap as its reward for taking the risk and doing the matching. A bank is, at its core, a *spread* business. (If you want the deeper machinery of how this multiplies money through the whole economy, the companion post on how money is created walks through the money multiplier; here we stay focused on the single bank.)

The reason this is risky — the reason a bank earns a spread rather than nothing — is the mismatch in time. Your deposit can be withdrawn soon; the 20-year mortgage cannot be called back early. The bank is **borrowing short and lending long**, and that maturity mismatch is the original sin of all banking. It works beautifully right up until everyone wants their deposit back at once, which is what a bank run is. The whole apparatus of capital rules, central-bank backstops, and deposit insurance exists to keep that mismatch from blowing up.

### The balance sheet: where the whole business lives

Because a bank's product is money, you cannot understand it from a factory tour — you understand it from its **balance sheet**, the two-column ledger of what it owns and what it owes. On the *asset* side (what the bank owns, the money other people owe it) sit the loans — by far the biggest line — plus government bonds, cash held at the central bank, and a small amount of premises and equipment. On the *liability* side (what the bank owes) sit the deposits — the biggest line — plus money borrowed from other banks and any bonds the bank itself has issued. The gap between total assets and total liabilities is the **equity**: the shareholders' stake, the bank's own capital, the cushion. Everything in banking is a story about how these two columns are shaped.

The shape of that ledger explains why a bank is a *leveraged* business in a way a normal company is not. A typical Vietnamese bank holds assets worth roughly ten to fifteen times its equity — that is, for every 1 dong of shareholders' money, it controls 10-15 dong of loans and securities, funded mostly by depositors. This leverage is the source of the bank's high return on equity *and* its fragility. If a normal company's assets fall 8% in value, it has a bad year; if a bank's assets (its loans) fall 8% in value, it can wipe out its entire equity cushion, because that cushion was only ~8-10% of assets to begin with. This is why a few percentage points of bad loans frighten bank investors so much: there is very little equity standing between a loan loss and insolvency. The regulator's capital rules exist precisely to keep that thin cushion thick enough.

A second feature of the bank balance sheet matters for valuation later: it is **marked close to fair value.** A factory on a manufacturer's books might be carried at a decades-old cost bearing no relation to its real worth, but a bank's loans and deposits are financial contracts carried at amounts that track their current value reasonably well. That is why "book value" — equity — is a *usable* measure of a bank's worth, and why we value banks on price-to-*book* rather than the price-to-sales or asset-replacement measures we might use elsewhere. The honesty of bank book value is, of course, only as good as the honesty of the loan values behind it — and in a property crash, when the market suspects those loans are worth less than stated, it stops trusting book value, which is exactly when low-P/B "cheapness" turns into a trap.

### Building the full profit: from NII down to net income

Net interest income is the engine, but it is not the whole car. A bank's profit-and-loss statement stacks up in a specific order, and knowing the order tells you where earnings can break. Start with **net interest income** (NII) — the NIM-times-assets number from above, the dominant line for most Vietnamese banks, typically 70-80% of total operating income. Add **non-interest income**: fees, foreign-exchange gains, and trading income (more on fees below). Together these make **total operating income** — the bank's "revenue."

From that revenue, subtract **operating expenses** — staff, branches, technology — measured by the **cost-to-income ratio (CIR)**, the share of income eaten by running costs. A well-run Vietnamese bank keeps CIR around 30-40%; a bloated one runs higher. What is left after costs is **pre-provision operating profit** — the profit the bank would make if no loans ever went bad. Then comes the line that makes banking cyclical: **provision expense**, the charge for expected loan losses. Subtract provisions, subtract tax, and you reach **net income** — the bottom line that drives ROE and the share price.

The crucial insight is *where the volatility lives.* Net interest income and fees are relatively stable year to year; operating costs barely move. It is the **provision line** that swings wildly — near zero in a good year, enormous in a bad one — and that single line is what turns a steadily profitable business into a violently cyclical stock. When analysts argue about a bank's next year, they are almost always arguing about provisions, which is to say about asset quality, which is to say about the credit cycle. Hold that thought: *the bank's margin is predictable; its bad-debt charge is not, and that is the whole game.*

### NIM: the spread, made precise

The single most important number in banking is **NIM — net interest margin.** It is the spread above, expressed as a percentage of the bank's earning assets (mostly its loan book). Informally: of every 100 dong the bank has lent out, how many dong of pure interest profit does it keep after paying its depositors?

The arithmetic is: take the interest the bank *earns* on its loans, subtract the interest it *pays* on its deposits and other funding, and divide by the size of the earning assets. A NIM of 3.5% means that for every 100 dong of loans, the bank nets 3.5 dong of interest income a year before any costs or bad debts. That sounds thin — and it is — but a bank lends out a *gigantic* multiple of its own capital, so a thin margin on an enormous book becomes a large profit. This leverage is the whole trick of banking, and it is also why a small change in NIM swings a bank's earnings violently.

What actually *moves* NIM is the tug-of-war between loan yields and funding costs, and the two do not move in lockstep. When the SBV cuts rates, deposit costs tend to fall *faster* than loan yields — depositors accept lower rates quickly, while many loans are on fixed or slow-resetting rates — so NIM *widens* in a rate-cutting cycle. That is the mechanical reason banks are an early-cycle trade: easing money expands the margin before it does anything else. When rates rise, the reverse bites — funding costs reprice up before loan yields catch up, squeezing NIM. On top of the rate cycle sits the *lending mix*: a bank that tilts toward high-yield retail and small-business loans (consumer finance, motorbike loans, SME working capital) earns a fatter NIM than one stuffed with low-yield, low-risk lending to blue-chip corporates and the government — but it also takes more credit risk for that wider margin. NIM, in other words, is never free: a high NIM is either a reward for a great funding franchise (cheap CASA) or compensation for taking more risk, and telling the two apart is half the job of analyzing a bank.

#### Worked example: turning a loan book and a deposit base into net interest income

Take a mid-size Vietnamese bank with a **400 trillion dong** loan book yielding **8.0%** a year, funded by **350 trillion dong** of deposits costing **4.5%** a year. (The gap between 400tn of assets and 350tn of deposits is filled by the bank's own equity and other funding — that is part of why margins look better than the raw rate spread.)

Interest earned: 400tn x 8.0% = **32.0tn dong** a year. Interest paid: 350tn x 4.5% = **15.75tn dong** a year. Net interest income (NII) = 32.0 − 15.75 = **16.25tn dong**. To get NIM, divide NII by earning assets: 16.25 / 400 = **4.06%** — call it a ~4% NIM, a healthy figure for Vietnam. In dollars, at roughly \$1 = 25,400 VND, that 16.25tn dong of net interest income is about **\$640mn** a year — from a single bank, before fees. The intuition: a bank's core earnings are just a margin times a balance-sheet size, and because the balance sheet is huge, even a 3-4% margin throws off enormous absolute profit.

### CASA: why some banks fund themselves almost for free

Not all deposits cost the same. A **term deposit** (you lock 100 million dong away for twelve months) pays a high rate — that money is "expensive" funding for the bank. But the balance sitting in your everyday checking account, or a basic savings account you dip into constantly, pays almost nothing — often near 0.1-0.2% a year. Banks call these low-cost balances **CASA**: *current account and savings account* money. CASA is the cheapest raw material a bank can get, and the share of a bank's deposits that is CASA — its **CASA ratio** — is one of the sharpest dividing lines between a great bank and a mediocre one.

Why does it matter so much? Because two banks can charge the *exact same* 8% on their loans and still earn wildly different margins, purely because one funds itself cheaply and the other does not. A bank with 50% CASA is funding half its book at near-zero; a bank with 15% CASA is paying full term-deposit rates on almost everything. Same product, very different cost of goods. In Vietnam, the CASA leaders — Techcombank (TCB), MB Bank (MBB), and state-owned Vietcombank (VCB) — have spent years building payroll relationships and slick banking apps precisely to capture this sticky, free money.

![Before and after comparison of a low CASA bank and a high CASA bank with the same loan yield](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-6.png)

### Fee income: profit without credit risk

There is a second profit stream that the best banks lean on hard: **fee income** — money earned not from lending but from *services.* Payment and card fees, foreign-exchange conversion, trade finance, asset management, and especially **bancassurance** (selling life-insurance policies through the bank's branches and app for a commission) all generate income that carries *no credit risk* and consumes *no balance-sheet capital.* A fee dong is, in a sense, a higher-quality dong than an interest dong: it does not require the bank to take on a loan that might go bad, and it does not eat into the capital ratio.

This is why the market prizes banks with a high and rising share of non-interest income. A bank whose profit is 80% net interest income is a pure spread-and-leverage bet, fully exposed to the credit cycle; a bank that has built fee income to 25-30% of operating income has a sturdier, more diversified earnings base that holds up better when lending slows or NPLs rise. In Vietnam, the private banks have pushed hardest into fees — bancassurance deals, card ecosystems, and digital-payment volumes — partly to lift ROE and partly to earn the premium valuation that diversified income commands. When you see a bank's fee income growing faster than its loan book, that is usually a quality signal worth paying up for.

#### Worked example: the CASA gap between two identical lenders

Two banks each lend at **8.0%**. Bank A has a **50% CASA** ratio: half its funding costs ~0.2% and half costs the 4.5% term rate, so its blended funding cost is roughly (0.5 x 0.2%) + (0.5 x 4.5%) = 0.1% + 2.25% = about **2.35%** — round up to ~3.5% once you add wholesale funding and reserve costs. Bank B has only **15% CASA**: its blended funding cost is roughly (0.15 x 0.2%) + (0.85 x 4.5%) = 0.03% + 3.83% = about **3.86%**, and with friction it lands near **5.2%**.

On the same 8.0% loan yield, Bank A nets a NIM in the neighborhood of **4.5%** while Bank B nets closer to **2.8%** — a gap of more than 1.5 percentage points, *with no difference whatsoever in lending strategy.* On a 400tn dong book, that 1.5-point margin gap is about 6tn dong a year, roughly **\$236mn**, conjured purely out of cheaper deposits. The intuition: CASA is the closest thing in banking to free money, and the bank that gathers the most of it wins the margin war before a single loan is even priced.

### Credit growth and the credit ceiling: Vietnam's defining constraint

A bank's profit is NIM times the *size* of the book — so growing the book grows the profit. In most countries a bank grows its loans as fast as it can find good borrowers and capital. In Vietnam, there is a hard governor on the engine: the **credit ceiling** (*room tin dung*).

Every year, the **State Bank of Vietnam (SBV)** — the central bank — sets a target for total credit growth across the whole banking system, typically somewhere around 14-16%, and then *rations* that growth bank by bank. Each bank is granted a quota: it may grow its loan book by, say, 14% this year, and not a dong more without SBV's permission to expand the quota mid-year. This is a deliberate macro tool. Credit growth in Vietnam is the main valve controlling how much money flows into the economy, and the SBV uses the ceiling to keep inflation and asset bubbles in check. (The companion post on Vietnam's monetary policy unpacks how the SBV sets and adjusts these quotas in detail.)

For an investor, the credit ceiling changes everything about how you read the sector. A bank cannot simply outgrow its peers by being aggressive — it is capped. The banks that win extra quota are usually the healthiest ones (strong capital, low bad debt, sometimes those that take over a failing bank), so the *allocation* of the ceiling becomes a quality signal in itself. And the *level* of the annual ceiling is a top-down macro dial for the whole market: a year when the SBV lifts the system target toward 16% is a tailwind for every bank's earnings; a year of tight, defensive credit is a headwind. Watching the SBV's credit-growth announcements is, for the banking sector, like watching a central bank set the speed limit on the entire economy.

The mechanics of the quota create a distinct trading rhythm. The SBV typically hands out an initial quota at the start of the year, deliberately conservative, and then *adds* to it mid-year for the banks that have lent prudently and kept their balance sheets clean. Those mid-year quota grants are market-moving events: a bank that gets its ceiling lifted from 14% to, say, 18% suddenly has more earnings runway, and its stock often pops on the news. So part of reading the sector is tracking *who is near their quota* (and thus likely to slow lending into year-end) versus *who has fresh room to grow.* It also produces a familiar fourth-quarter pattern: banks that have used up their quota go quiet on new lending, while the anticipation of next year's allocation builds. None of this exists in a free-credit market; it is a uniquely Vietnamese layer that a sector investor must track.

There is also a subtle quality filter buried in the quota system. Because the SBV rewards clean, well-capitalized banks with bigger ceilings and sometimes asks strong banks to *absorb* weak ones (granting the acquirer extra quota as a sweetener), the credit ceiling quietly sorts the sector. A bank consistently winning above-system quota is being told, in effect, that the regulator trusts its balance sheet — a signal that often precedes the market's own re-rating. Conversely, a bank kept on a tight leash may be one the regulator is worried about. The quota is not just a growth cap; read carefully, it is a regulator's report card on each bank's health.

### NPLs and provisions: where profit leaks out

Lending is easy; getting paid back is the hard part. Some borrowers stop paying. A loan where the borrower has missed payments for more than 90 days is classified as a **non-performing loan (NPL)** — a "bad loan," *no xau* in Vietnamese. The **NPL ratio** is the share of the loan book that has gone bad, and it is the single best gauge of a bank's *asset quality.* A 2% NPL ratio means 2 dong of every 100 lent are in trouble.

Bad loans hit profit through **provisions.** Accounting rules force a bank, the moment a loan looks shaky, to set aside money — a provision — against the expected loss, *before* the loss is even confirmed. That provision is booked as an expense, straight out of this year's profit. So a wave of bad loans hurts a bank twice: it stops earning interest on the dead loans, *and* it must take provisioning charges that can wipe out a big chunk of earnings. A bank's reported profit in a bad year is often "what NIM and fees earned, minus what provisions destroyed."

#### Worked example: what a one-point rise in NPLs costs

A bank has a **400 trillion dong** loan book with a **2% NPL ratio** — so 8tn dong of loans are already non-performing. Now a credit downturn pushes its NPL ratio up by **one percentage point**, to 3%. That extra 1% of a 400tn book is **4tn dong** of newly problem loans — about **\$157mn** at \$1 = 25,400 VND.

The bank will not lose all 4tn (it usually recovers something from collateral), but it must *provision* against the expected loss. If it provisions, say, 70% of the new bad loans, that is 4tn x 70% = **2.8tn dong** of provisioning expense — roughly **\$110mn** charged straight against this year's pre-tax profit. For a bank that earns 16tn dong of net interest income, a 2.8tn provision charge erases a sixth of its core earnings in a single line. The intuition: a bank's margin is the engine, but rising NPLs are the brake, and in a bad year the brake can overpower the engine — which is exactly why asset quality, not margin, is what frightens bank investors.

### ROE and CAR: profitability and safety, the two gauges

Two more terms complete the foundation, and they pull in opposite directions.

**ROE — return on equity** — is the bank's annual profit divided by its shareholders' equity (its book value). It answers: for every 100 dong of owners' money in the bank, how many dong of profit does it earn a year? Vietnamese banks are unusually profitable by global standards; the best earn ROEs above 20%, the sector averages around 17%, and a struggling bank might manage 10-11%. ROE is the headline measure of how good a bank is at turning capital into profit, and — as we will see — it is the number that ultimately justifies a bank's valuation.

**CAR — capital adequacy ratio** — pulls the other way. It is the bank's capital measured against its risk-weighted assets, and it is the regulator's safety buffer: the cushion that absorbs losses before depositors are at risk. Vietnam runs on a Basel II framework with a minimum CAR around 8%. The tension is fundamental: a bank can boost ROE by lending out more relative to its capital (more leverage), but that *lowers* CAR and makes it more fragile. A bank with a thin capital buffer cannot grow its loan book even if it wins the credit quota, because growth consumes capital. So CAR quietly caps how much of its ROE potential a bank can actually deploy. The best banks earn high ROE *and* keep a comfortable CAR — they are profitable and safe at once, and the market pays up for that rare combination.

## The Vietnamese banking landscape

With the mechanics in hand, look at who the actual players are, because in Vietnam the *ownership structure* of a bank tells you a great deal about how its stock will behave.

### State-owned giants vs private dynamos

Vietnamese banks split into two camps. The **state-owned commercial banks** — Vietcombank (VCB), BIDV (BID), and VietinBank (CTG), sometimes called the "Big 3" or "Big 4" with Agribank (unlisted) — are majority-owned by the government. They are vast: enormous balance sheets, nationwide branch networks, the deposits of state enterprises and government payroll, and a quasi-sovereign aura that makes them the safest place to park money. VCB in particular is the crown jewel: it has the lowest NPL ratio in the system (often under 1%), a high CASA ratio, and the market's deepest trust — which is why it commands the richest valuation of any Vietnamese bank.

The **private joint-stock banks** — Techcombank (TCB), MB Bank (MBB), ACB, VPBank (VPB), HDBank (HDB), and others — are nimbler, hungrier, and faster-growing. They run higher NIMs (they lend more to higher-yielding retail and SME borrowers), chase CASA aggressively through technology, and generally post higher ROEs and faster credit growth than the state banks. They are also higher-beta: they re-rate harder in a bull market and sell off harder when asset-quality fears strike, because their loan books tilt toward riskier, higher-margin lending.

![Comparison matrix of state owned versus private Vietnamese banks across margin funding growth and foreign room](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-7.png)

This split has a direct trading consequence. State banks are the *ballast* of VN-Index — their huge weights mean they move the index even when they barely lead it — while private banks are the *sprinters* that lead the sector higher in an up-cycle. When you want broad, safe exposure to "Vietnam is doing fine," you buy VCB; when you want to express "the credit cycle is turning up and risk appetite is rising," you buy the high-NIM private names.

### A quick tour of the names that move the index

It helps to attach faces to the categories, because each major bank has a distinct personality that shapes how its stock trades. **Vietcombank (VCB)** is the bellwether: the most valuable bank in the country, the lowest NPL ratio in the system, a high CASA franchise built on state-enterprise and payroll deposits, and the richest valuation (around 2.4x book) because the market treats it as a quasi-sovereign safe haven. When foreigners want "core Vietnam," they want VCB — and its foreign room is locked, which only adds to the scarcity premium. **BIDV (BID)** and **VietinBank (CTG)** are the other two listed state giants: enormous balance sheets, big index weights, but thinner margins and historically higher bad-debt baggage than VCB. They are the heavy ballast — they move the index by sheer size, but they rarely *lead* a rally.

On the private side, **Techcombank (TCB)** is the CASA-and-fee machine: one of the highest CASA ratios in the system, a strong bancassurance and digital franchise, and deep ties to the property developer Vingroup/Vinhomes ecosystem — which is both its growth engine and its concentration risk. **MB Bank (MBB)** pairs a high CASA ratio (helped by military and corporate payroll relationships) with strong fee income and disciplined growth, often screening as one of the best risk-adjusted franchises. **ACB** is the conservative private bank — clean retail and SME book, low NPLs, steady high-teens-to-low-twenties ROE, less drama. **VPBank (VPB)** is the high-NIM, high-risk extreme: a large consumer-finance arm (FE Credit) that earns enormous margins on motorbike and consumer loans but carries the highest credit losses, making it the most cyclical of the big names. **HDBank (HDB)** rounds out the group with fast growth and a high ROE. Knowing these personalities is the difference between "buying banks" and *choosing* the right bank for the part of the cycle you are trading — the conservative ACB/VCB end for defense, the high-beta VPB/TCB end for an up-cycle.

### Foreign room: the 30% wall

One Vietnam-specific quirk shapes who can even buy these stocks. Foreign investors are capped at owning **30% of a bank's shares** — the *foreign ownership limit*, known in the market as *room ngoai* (foreign room). For the most coveted banks, foreigners have already filled that quota, so the room is "full": a foreign fund that wants in must buy from another foreigner exiting, often at a premium, because it cannot buy fresh shares on the open market. VCB's foreign room is effectively locked; that scarcity supports its price. Other banks — TCB and VPB at various times — have kept some room open or deliberately set their foreign cap below 30% to preserve flexibility. For a foreign-flow-driven market, *which bank has open room* determines where international money can actually go, and a bank opening up new room can see a flood of foreign buying. (The dynamics of foreign flows and the index are the subject of a dedicated post in this series.)

### Why banks dominate the index

Step back to the arithmetic that opened this post. Banks are about 34% of VN-Index by market capitalization — the largest sector by a wide margin, with real estate a distant second around 17%. Several of the ten largest stocks on HOSE are banks. This concentration is not an accident: banking is the most capital-intensive industry in the economy, banks carry the savings of the whole nation on their balance sheets, and Vietnam's listed market is unusually finance-heavy because the banking system was among the first to equitize and list.

![Horizontal bar chart of VN-Index sector weights with banks the largest at about 34 percent](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-2.png)

The practical upshot is that **VN-Index is a bank index wearing a market's clothes.** No durable rally happens without banks; no serious correction spares them. If you hold a VN-Index ETF or an index fund, roughly a third of your money is already in banks whether you chose it or not. And on any given day, a "banking day" — when the whole bank group runs 2-4% — can single-handedly produce a strong green session for the index while two-thirds of the market sits still. Reading the banks is, to a first approximation, reading the index.

## Valuing a bank: P/B and ROE

Now the question every investor eventually asks: is a given bank cheap or expensive? For most companies you reach for the price-to-earnings ratio. For banks, the workhorse metric is **price-to-book (P/B)** — the stock price divided by the bank's book value (its equity) per share. And the key to using it is that P/B only makes sense *paired with ROE.*

### Why book value, and why pair it with ROE

A bank's assets and liabilities are mostly financial — loans and deposits, marked at fairly current values — so its book value (assets minus liabilities, i.e. equity) is a reasonably honest measure of what the bank is "worth" on paper. A P/B of 1.0x means you are paying exactly book value; 2.0x means you are paying twice book.

Why would anyone pay twice book for a bank? Because of ROE. Book value is the equity that *generates* the bank's profit, and ROE is the rate at which it does so. A bank earning a 20% ROE is compounding its owners' capital at 20% a year; a bank earning 10% is compounding it at 10%. The first is worth a far higher multiple of its book than the second, because each dong of book equity is producing twice the profit. The clean way to say it: **P/B should rise with ROE.** A high-ROE bank deserves a high P/B; a low-ROE bank deserves a low one. The mistake beginners make is to call a low-P/B bank "cheap" without checking whether its low ROE *justifies* the low multiple.

![Scatter plot of Vietnamese bank return on equity against forward price to book](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-4.png)

The scatter above plots the major Vietnamese banks. The broad upward tilt is the ROE-P/B relationship at work: the highest-quality franchises command the richest multiples. VCB sits in the top-right premium corner — high ROE *and* the highest P/B in the sector at around 2.4x — because the market pays up for its rock-bottom NPLs, high CASA, and state backing. Several private banks cluster lower, around 1.0-1.4x book, despite respectable ROEs, because the market discounts them for higher perceived asset risk or thinner capital. The sector as a whole trades around 1.3x forward book on a ~17% ROE.

#### Worked example: the justified P/B for an 18% ROE bank

There is a simple formula linking the two. The justified price-to-book of a bank is approximately (ROE − g) / (COE − g), where **g** is the bank's long-run growth rate and **COE** is the cost of equity — the return shareholders demand for the risk.

Take a solid Vietnamese bank with **ROE = 18%**, a **cost of equity of 13%** (reasonable for an emerging market with Vietnam's risk), and a sustainable **growth rate of 5%**. Justified P/B = (18% − 5%) / (13% − 5%) = 13 / 8 = **1.6x book.** So a bank like this trading *below* ~1.6x might genuinely be cheap, and one trading well above it is pricing in either higher growth or lower risk than the inputs assume.

Now apply it to VCB at **2.4x**. To justify 2.4x on a 13% cost of equity, the market must believe VCB's (ROE − g) / (COE − g) is far richer — for instance a ~20% sustainable ROE with the market assigning it a *lower* cost of equity (say 11%) because of its quasi-sovereign safety: (20 − 6) / (11 − 6) = 14 / 5 = 2.8x, which comfortably supports 2.4x. The intuition: you are not "overpaying" for VCB at 2.4x book — you are paying for a durably higher ROE earned at lower risk, and the P/B-ROE math says that is exactly what a premium franchise should cost.

### Where the P/B-ROE link breaks: the asset-quality discount

The formula assumes the book value is *real.* That assumption is the soft spot, and it is where most bank-valuation mistakes happen. Book value is loans-minus-deposits-plus-other-equity, and if the loans are worth less than the bank claims — because bad debts are under-provisioned, or restructured loans are quietly rotting — then the "book" you are buying at 1.0x is a fiction, and you are really paying 1.3x or 1.5x of *true* book. This is why two banks with the same reported ROE and the same P/B can be priced completely differently by a careful investor: the one with clean, fully-provisioned assets is genuinely cheap at 1.0x, while the one with a property-loan time bomb and thin provisions is expensive at 1.0x because its book is overstated.

So the practical valuation routine for a Vietnamese bank is two-step. First, sanity-check the *reported* P/B against the *reported* ROE using the formula — is the multiple roughly in line with the profitability? Second, and more important, interrogate the *quality of the book* behind that ROE: the NPL ratio, the **loan-loss coverage ratio** (provisions set aside divided by NPLs — above 100% means the bank has reserved more than its current bad loans, a sign of conservatism), the share of restructured and special-mention loans, and the exposure to property and developer bonds. A bank with a 200%+ coverage ratio and low property exposure earns the right to its multiple; a bank with sub-80% coverage and heavy developer loans does not, no matter how low its P/B screens. The single sentence to carry: *a low P/B is only cheap if the book value is honest, and in banking the book value is exactly what a credit downturn calls into question.*

## Asset quality and the 2022 property exposure

If margin is what makes a bank profitable, asset quality is what can destroy it — and in Vietnam, the defining asset-quality episode of the last decade was the **2022 corporate-bond and property crisis.** Understanding it is essential to understanding why bank stocks behave the way they do.

### How property risk gets onto a bank's balance sheet

Vietnamese banks are deeply entangled with the property sector, in three ways. First, they make direct **real-estate loans** — to developers building projects and to homebuyers taking mortgages. Second, a large share of *all* lending in Vietnam is **collateralized by property**: even a manufacturing loan is often secured against land, so a property crash impairs collateral across the whole book. Third, banks were big holders and distributors of **corporate bonds** (*trai phieu doanh nghiep*) — many issued by property developers — which they held on their own books and sold to retail customers.

So when the property and corporate-bond market seized up in late 2022 — after a regulatory crackdown on bond issuance and the high-profile arrests of major developers — the stress flowed straight into bank balance sheets. Developers could not roll over maturing bonds; some could not service their bank loans; collateral values wobbled. The market's fear was not subtle: if property defaults cascaded, the banks holding property loans and property bonds would face a wave of NPLs and provisions that could gut their profits and even threaten their capital.

### What the data showed — and what it did not

Here is the important nuance. The *fear* in late 2022 was of a banking crisis; the *realized* damage was real but contained. Sector NPLs did rise — from around 1.5% in 2021 to roughly 1.9% in 2022 and on to about 2.2% in 2023-2024 — a meaningful deterioration, but nowhere near a systemic blowup. The SBV intervened: it allowed banks to *restructure* troubled loans (deferring the NPL recognition), eased some rules, and engineered a path that let developers and banks work through the stress over time rather than all at once.

![Line chart of Vietnamese banking sector NPL ratio and system credit growth from 2019 to 2024](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-5.png)

The chart traces it: NPLs climbing steadily off the 2021 low as the property stress filtered through, while system credit growth kept chugging along in the 13-15% range — the SBV never slammed the credit brakes hard enough to cause a deflationary spiral. The dispersion *within* the sector was the real story: VCB's NPL ratio barely moved and stayed best-in-class near 1%, while banks with heavy property and developer-bond exposure saw their bad-loan ratios and provisions jump much more. This is why "all banks are the same" is such a dangerous assumption: in a stress event, the gap between the cleanest and the most-exposed banks blows wide open.

For an investor, 2022 left two durable lessons. One: bank stocks are a *leveraged bet on the property cycle*, whether you like it or not, because of how intertwined the balance sheets are. Two: in a Vietnamese bank crisis, the SBV's instinct is to *forbear and restructure* rather than force mass write-downs — which softens the downside for the system but also means reported NPLs understate the true stress for a while, so you watch *restructured loans* and *special-mention loans*, not just the headline NPL number.

## Common misconceptions

Three beliefs reliably cost beginners money in this sector. Each falls apart on contact with a number.

### "A low P/B bank is always cheap"

The most expensive mistake in bank investing is buying the lowest P/B on the screen and calling it value. P/B is meaningless without ROE. A bank at **1.0x book earning an 11% ROE** is arguably *more expensive* than a bank at **2.4x book earning a 20% ROE**, because you are paying full book value for capital that compounds slowly, versus a premium for capital that compounds fast. Worse, a rock-bottom P/B is often the market *correctly* pricing hidden bad debt: the book value itself is suspect because the loans behind it are worth less than stated. A genuinely cheap bank is one trading *below its ROE-justified P/B* — a 1.0x bank earning 18% with clean assets — not simply the lowest multiple in the table. Always ask "cheap relative to what ROE and what asset quality?"

### "All banks are the same"

They are not, and the differences are huge and persistent. NIM ranges from under 3% at the big state banks to nearly 5% at the high-yield private lenders. CASA ranges from the low teens to roughly 50%. NPL ratios in the 2022-23 stress ranged from VCB's ~1% to multiples of that at property-heavy banks. ROE ranges from ~11% to ~24%. These are not noise — they reflect durable differences in funding franchise, lending mix, and risk discipline that persist for years. Treating the banking sector as one undifferentiated block means you buy the weak banks and the strong ones together, and in any stress event the weak ones are what hurt you. The sector moves together as a *tide* (co-movement), but *which bank you own inside the tide* still decides much of your return — a theme developed in the anatomy-of-a-sector post.

### "High credit growth is always good"

It feels intuitive that a bank growing its loans 25% a year must be a better investment than one growing 12%. Often the opposite is true. Rapid credit growth is how banks *manufacture* future bad loans: to grow fast, a bank must lower its standards, lend to weaker borrowers, and pile into hot sectors like property right before they turn. Some of the worst NPL waves in the 2022 stress hit banks that had grown their developer and bond exposure fastest in the preceding boom. Fast growth also burns capital (lowering CAR) and is often only possible because the SBV granted extra quota — which can be a sign the bank is being pushed to absorb a weaker rival. *Quality-adjusted, sustainable* credit growth funded by cheap CASA at a high NIM is wonderful; aggressive growth into risky collateral is a time bomb. Look at *what* is being lent and *how it is funded*, never just the headline growth rate.

## How it shows up on VN-Index

The abstractions become concrete when you watch the banks actually move the index across three episodes.

### The 2020-2021 liquidity bull: banks lead the charge

When the SBV slashed rates and flooded the system with liquidity during the pandemic, credit growth accelerated, NIMs widened (deposit costs fell faster than loan rates), and asset-quality fears were dormant. Banks were the perfect vehicle for that environment, and they led VN-Index from its March 2020 low through its late-2021 peak. High-beta private names like TCB, VPB, and MBB multiplied; the whole sector re-rated as ROEs hit record highs and the market extrapolated the good times. This is the textbook **early-cycle bank trade**: falling rates plus rising credit plus benign NPLs equals a powerful bank-led index rally.

### The 2022 NPL fear: banks lead it down

Then the cycle turned. Rates spiked, the bond and property crackdown hit, and the market's mood about banks flipped from "margin machines" to "property time bombs." Bank stocks de-rated hard in 2022 — many fell 30-50% from their highs — dragging VN-Index down with them, precisely because banks are a third of the index. The fall was a *re-rating of risk*, not a collapse in current earnings: banks were still profitable, but the market slashed the P/B multiple it was willing to pay because it feared the NPLs coming down the road. This is the mirror image of the early-cycle trade and the reason bank valuations are so cyclical: the multiple swings on *expected* asset quality long before the NPLs actually show up.

### The 2024 divergence: banks carry a tired index

By 2024, the picture had stabilized. The feared systemic blowup had not materialized (the SBV's forbearance worked), NPLs had risen but plateaued around 2.2%, and credit growth was healthy. Banks — especially the clean, high-CASA names — re-rated off their 2022 lows even as much of the rest of the market stayed sluggish. The result was the split-screen we opened with: the eighteen HOSE banks up about 26% on the year while VN-Index managed only about 12%.

![Grouped bar chart of 2024 returns for HOSE banks versus VN-Index](/imgs/blogs/banking-sector-vietnam-the-heartbeat-of-vn-index-3.png)

That 14-point gap is the whole thesis of this post in a single chart. The index's modest 12% was *carried* by the banks; without their 26%, the year would have been roughly flat. When you hear "the market went up in 2024," the honest translation is "the banks went up, and they are a third of the market." Banks are the heartbeat of VN-Index in the most literal sense: the index's pulse is, to a large degree, the banks' pulse.

One amplifier is worth naming, because it shows up repeatedly in these episodes: **margin lending** (*ky quy*) — borrowing from your broker to buy more shares than your cash allows. Bank stocks, being large, liquid, and held by nearly every Vietnamese investor, are among the most heavily margined names on the exchange. In an up-cycle, rising bank prices let investors borrow more against their now-bigger positions and buy still more banks — a self-reinforcing loop that exaggerates the rally. In a downturn, the loop runs in reverse: falling bank prices trigger *margin calls*, forcing leveraged holders to sell, which pushes prices lower and triggers more calls. This is part of why bank-led moves on VN-Index are so sharp in both directions, and why the *liquidity and margin cycle* is a sector signal in its own right — the more margin debt is piled into banks, the more violent the eventual unwind. A bank rally riding on swelling margin balances is more fragile than one driven by genuine earnings re-rating, and learning to tell them apart is part of trading the sector well.

## The playbook: how to trade the banking sector

Everything above earns its keep here. The banking sector is the most *cyclical-on-policy* group in the market — it lives and dies on rates, credit, and asset-quality perception — and that makes it tradable in a disciplined, signal-driven way.

### When to overweight banks

You want to be overweight banks when the credit cycle is turning *up* and asset-quality fear is *receding* — the early-cycle setup. The concrete signals, in roughly the order they appear:

The **SBV cutting policy rates** or signaling easier money. Lower rates widen NIM (deposit costs fall fast) and revive credit demand; this is the single biggest tailwind for the sector. A **rising annual credit-growth ceiling** — when the SBV lifts the system target toward 16%, every bank's earnings runway expands. **Stabilizing or falling NPLs** — when the bad-loan ratio plateaus and provisioning charges ease, the market re-rates the P/B multiple upward, and that multiple expansion is where the big returns live. **A recovering property market** — because banks are a leveraged play on property, the first signs of developer health and presales recovery flow straight into bank sentiment. And **foreign room opening or foreign buying returning**, since the most-coveted banks need foreign flow to push their constrained-supply shares higher.

When those align, the trade is to overweight the sector and, within it, tilt toward the *high-beta, high-NIM private banks* (TCB, MBB, VPB, ACB) that re-rate hardest in an up-cycle — while holding VCB as the lower-volatility anchor. Size it remembering that banks are already a third of any index position you hold: an "overweight" is a tilt on top of an exposure you partly own by default.

### When to underweight, and what invalidates the view

You want to *underweight* banks when the cycle is turning down: rates rising, the SBV tightening credit, NPLs climbing with no plateau in sight, and property stress building. The 2022 setup is the template — and crucially, the *multiple* (P/B) falls before the *earnings* do, so you cannot wait for reported profits to drop. By then the stock has already fallen. You trade the bank cycle on *expected* asset quality, watching restructured and special-mention loans, the property-bond maturity calendar, and the SBV's tone.

The **invalidation** of a bullish bank thesis is specific and worth writing down before you enter: a *re-acceleration of NPLs* (the bad-loan ratio breaking back above its recent range and provisions jumping), or a *macro shock that forces the SBV to tighten* (an inflation or currency scare that flips policy from easing to defense). Either one inverts the early-cycle math — NIM compresses, credit growth gets capped, and the P/B multiple the market will pay collapses. If you see NPLs turning back up while you are long the sector, the thesis is broken regardless of how cheap the stocks look, because *cheap on a deteriorating book is a value trap.* Conversely, the bearish view is invalidated when NPLs visibly plateau and the SBV pivots to easing — that is the bottom signal that has marked every bank-led recovery in VN-Index's history.

### The five numbers to watch every quarter

If you do nothing else, track five things each quarter and you will be reading the sector better than most of the market. **NIM** — is the margin expanding (a tailwind, usually early-cycle) or compressing? **CASA ratio** — is cheap funding holding up, which protects the margin when rates rise? **Credit growth versus quota** — how much of the annual ceiling has the bank used, and is it winning extra room? **NPL ratio and the loan-loss coverage ratio together** — are bad loans rising, and is the bank reserving enough against them (coverage above ~100% is comfort, below ~70% is a warning)? And **the property/developer-bond exposure** — the single biggest swing factor in a Vietnamese bank's downside. Layer the macro on top — the SBV's policy-rate path and the annual system credit-growth target — and you have a complete dashboard. A bank with rising NIM, stable high CASA, fresh quota, plateauing NPLs with full coverage, and modest property exposure is the early-cycle long; a bank with compressing NIM, slipping CASA, exhausted quota, rising NPLs with thin coverage, and heavy developer loans is the one to avoid no matter how cheap it looks.

#### Worked example: how much profit a bank's ROE throws off

Tie ROE back to real money to feel its weight. Take a bank with **120 trillion dong** of equity (book value) earning a **20% ROE.** Annual profit = 120tn x 20% = **24tn dong** — roughly **\$945mn** a year at \$1 = 25,400 VND, from a single bank. If the market values that bank at **2.0x book**, its market capitalization is 120tn x 2 = 240tn dong, about **\$9.4bn.** Now suppose a credit downturn knocks the ROE from 20% down to 12% (provisions eating the difference): profit falls to 120tn x 12% = 14.4tn dong, a **40% drop in earnings** — and because the market also cuts the P/B multiple it will pay on a now-riskier, slower-compounding bank (say from 2.0x to 1.3x), the market cap can fall far more than 40%. The intuition: ROE is leverage on the cycle in both directions — a bank's profit and its valuation multiple both swing with asset quality, and they swing *together*, which is why bank stocks are so violently cyclical and why timing the cycle is the whole game.

The discipline that ties it together: banks are not a stock you buy and forget. They are a *cyclical instrument* whose value swings on a handful of policy and asset-quality dials. Learn to read those dials — the SBV's rate and credit-ceiling decisions, the NPL trend, the loan-loss coverage, the property cycle, the foreign flow — and you are reading not just the banking sector but the heartbeat of VN-Index itself.

## Further reading & cross-links

- [Why a Whole Industry Moves as One: The Anatomy of a Vietnamese Stock Sector](/blog/trading/vietnam-stocks/anatomy-of-a-stock-sector-why-industries-move-together) — the foundation for *why* all bank stocks co-move, and how to think about leaders versus laggards inside the tide.
- [The Four Macro Dials: Rates, Credit, FX, Commodities](/blog/trading/vietnam-stocks/the-four-macro-dials-rates-credit-fx-commodities) — the top-down dials (especially rates and credit) that drive the whole banking sector at once.
- [Valuation by Sector: P/E, P/B, NAV, EV/EBITDA](/blog/trading/vietnam-stocks/valuation-by-sector-pe-pb-nav-ev-ebitda) — why banks are valued on P/B-versus-ROE while other sectors use different metrics entirely.
- [Real Estate: Land Banks, Presales, and Bonds](/blog/trading/vietnam-stocks/real-estate-sector-vietnam-land-banks-presales-bonds) — the sector on the other side of the bank balance sheet, and the source of the 2022 asset-quality stress.
- [Vietnam Monetary Policy: The State Bank, the Dong, and the Credit Ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) — how the SBV sets the rates and the credit quota that govern every bank's earnings.

---
title: "Valuing Vietcombank: A Complete P/B and DDM Case Study"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A full step-by-step valuation of Vietnam's largest bank using justified price-to-book and the dividend discount model, so you can value any bank in an emerging market."
tags: ["valuation", "asset-pricing", "bank-valuation", "price-to-book", "dividend-discount-model", "cost-of-equity", "vietnam", "emerging-markets", "roe", "case-study"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank is valued on its equity, not its enterprise value, so you price Vietcombank with a justified price-to-book multiple driven by ROE and with a dividend discount model, not with EV/EBITDA.
>
> - For banks, debt is the raw material, so EV multiples are meaningless; use equity-centric models: justified P/B and DDM.
> - The master formula is justified P/B = (ROE − g) / (r − g). Plug in VCB's ROE of about 22%, growth of 8%, and a cost of equity near 15.4%, and you get a fair P/B of roughly 1.9x.
> - VCB trades at about 1.65x book versus that justified 1.9x, a slight undervaluation, with the gap mostly explained by credit-cycle and state-ownership risk.
> - The one number to remember: when ROE comfortably exceeds the cost of equity, a bank *should* trade above book value — and the size of that premium is the whole valuation question.

In mid-2024, shares of Vietcombank — Joint Stock Commercial Bank for Foreign Trade of Vietnam, ticker VCB on the Ho Chi Minh Stock Exchange — changed hands around 88,000 to 92,000 VND. The bank reported book value per share of roughly 55,000 VND. Divide one by the other and you get a price-to-book ratio of about 1.65x: the market was paying 1.65 dong for every dong of accounting net worth. Is that expensive? Cheap? Fair?

You cannot answer that with the tools most people reach for. You cannot run an EV/EBITDA multiple, because a bank has no meaningful enterprise value in the usual sense — its "debt" is customer deposits, and those deposits are not financing to be netted out, they are the raw material the bank turns into loans. You cannot cleanly run a free-cash-flow DCF either, because for a bank, cash flow and capital are tangled together in ways that make "free" cash flow almost undefinable. Banks need their own toolkit.

This post builds that toolkit from zero and then applies it, end to end, to one real company. By the end you will have computed VCB's justified price-to-book from first principles, valued it with a two-stage dividend discount model, decomposed its return on equity with the Du Pont identity, stress-tested the answer with a sensitivity grid, and bracketed it with bull, base, and bear price targets. The numbers are approximate — drawn from VCB 2024 Annual Report estimates and market data — but the method is exactly what a professional bank analyst does.

![Bank valuation method selection flow for Vietcombank](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-1.png)

## Foundations: why banks need their own valuation methods

Start with a term-by-term build, because almost everything that follows rests on understanding what a bank *is* as a financial object.

A **bank** is a business that borrows money cheaply and lends it out at a higher rate. The money it borrows shows up as **liabilities** — overwhelmingly **customer deposits**, the balances in everyone's checking and savings accounts. The money it lends shows up as **assets** — mostly **loans** to households and companies, plus some securities and cash. The difference between what the bank earns on its assets and what it pays on its liabilities, scaled by the size of its loan book, is the **net interest margin (NIM)**. For VCB, NIM runs around 3.0% to 3.2%: it earns roughly three percentage points more on its loans than it pays on its deposits.

Here is the crucial structural fact. For an ordinary company — a factory, a software firm — debt is a *financing choice*. The business exists (it makes widgets), and separately it decides how much to borrow to fund itself. That separation is exactly why we can compute an **enterprise value (EV)**: the value of the whole operating business, independent of how it is financed, equal to the market value of equity plus net debt. We then divide EV by **EBITDA** (earnings before interest, taxes, depreciation, and amortization) to get a multiple that is comparable across companies regardless of their debt loads.

For a bank, that separation collapses. The "debt" — deposits — is not financing sitting underneath the business; it *is* the business. A bank with no deposits is not an unlevered version of a bank, it is not a bank at all. Netting deposits against assets to get an enterprise value would net away the entire operation. So EV/EBITDA, EV/Sales, and their cousins are not just imprecise for banks — they are meaningless.

What is left is the **equity**. Equity is what shareholders actually own: total assets minus total liabilities, the residual after every depositor and creditor is paid. For VCB, total assets are around 1,900,000 billion VND (often written 1,900 trillion VND), and total equity is around 150,000 billion VND. That thin sliver — equity is under 8% of assets — is the only thing a shareholder has a claim on, and it is the only thing valuation needs to price. So bank valuation is **equity-centric**: we value the equity directly. Two methods dominate.

The first is **price-to-book (P/B)**: the ratio of the equity's market price to its accounting book value. The second is the **dividend discount model (DDM)**: the present value of the cash the bank will hand to shareholders over time. Both are built on equity, and both, as we will see, reduce to the same underlying driver — how much return the bank earns on each dong of equity relative to what shareholders require.

![Vietcombank balance sheet assets versus liabilities and equity](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-4.png)

One more term to nail down: **return on equity (ROE)**, net income divided by shareholders' equity. It answers "for every dong of equity, how much profit did the bank make this year?" VCB's ROE is exceptional by global standards — around 22% to 23%. A bank earning 22% on equity is, in one year, generating profit equal to 22% of everything its shareholders own. Hold that number; it is the engine of the entire valuation.

It is worth pausing on why ROE, of all the metrics a bank reports, is the one that drives value. A factory's earnings can be lumpy and its asset base hard to pin down, so analysts often value it on cash flow. A bank is different: its assets are financial, marked at or near book, and its earnings are a fairly direct function of the equity it deploys. Equity is both the regulatory constraint (Basel rules cap how many assets a bank can hold per dong of equity) and the engine of profit (more equity supports more loans, which generate more net interest income). Because equity is simultaneously the binding constraint and the profit driver, the *return on that equity* is the single most informative number about a bank — it tells you how efficiently the scarce resource is being turned into profit. Everything downstream in this post is, in one way or another, a transformation of ROE.

A second reason banks demand their own approach is **accounting transparency**, and it cuts both ways. A bank's balance sheet is almost entirely financial instruments — loans, deposits, securities — that are carried close to their economic value, far more so than a manufacturer's goodwill, brand, or plant. That makes book value unusually *meaningful* for a bank: when you pay 1.65x book for VCB, you are paying 1.65 dong for a dong of reasonably-measured net financial assets, not for an accountant's estimate of intangible worth. But the same financial assets hide a trapdoor: loan values depend on assumptions about who will repay. A loan booked at full value today can be worth a fraction tomorrow if the borrower defaults. So book value is trustworthy *only as far as the loan-loss assumptions are honest* — which is exactly why the NPL ratio and provisioning policy sit at the heart of bank analysis, and why VCB's best-in-class 1.1% NPL ratio is such a load-bearing fact in its valuation premium.

Finally, banks are **leveraged by design** in a way ordinary firms are not. VCB runs roughly 12.7 dong of assets for every dong of equity. That leverage magnifies ROE in good times — a modest spread on a huge asset base becomes a large return on a small equity base — but it also magnifies fragility: a loss of just 8% on assets would wipe out the entire equity cushion. This is the deep reason a bank's required return (its cost of equity) is higher than a comparable industrial firm's, and why the gap between ROE and that required return is the whole ballgame. A bank is a leveraged bet on its own loan book; valuing it is valuing how reliably that bet pays off.

## The P/B framework for banks, derived from first principles

People treat price-to-book as a crude rule of thumb — "banks trade around book, so 1x is fair." That is lazy. The justified P/B is a precise quantity you can derive, and the derivation reveals exactly *why* a great bank deserves to trade well above book and a weak one below.

Begin with a fact about any business that pays out what it does not reinvest. If a company earns ROE on its equity and reinvests a fraction *b* (the **retention ratio**) back into the business, then its equity — its book value — grows at rate **g = b × ROE**. The rest, fraction (1 − b), is paid out as dividends. This is the **sustainable growth rate**: the pace at which a bank can grow its book value using only retained earnings, without issuing new shares.

Now value the equity as the present value of all future dividends, growing forever at rate *g*, discounted at the shareholders' required return *r* (the **cost of equity**). That is the **Gordon growth** form of the dividend discount model:

Price per share = D₁ / (r − g)

where D₁ is next year's dividend per share. Next year's dividend equals next year's earnings per share times the payout ratio: D₁ = EPS₁ × (1 − b). And next year's earnings per share, in terms of *this* year's book value per share (BVPS), is EPS₁ = ROE × BVPS (the bank earns ROE on the equity it starts the year with). Substitute:

Price = [ROE × BVPS × (1 − b)] / (r − g)

Divide both sides by BVPS to turn price into a price-to-book multiple, and recall g = b × ROE so that (1 − b) = (ROE − g) / ROE. After the algebra cancels the ROE term:

**Justified P/B = (ROE − g) / (r − g)**

This is the master equation of bank valuation, and it is worth reading slowly. The fair price-to-book is the *spread* between what the bank earns (ROE) and the growth it bakes in (g), divided by the *spread* between what shareholders demand (r) and that same growth. If ROE exactly equals r — the bank earns precisely what its owners require — the formula collapses to P/B = 1.0x: the bank is worth exactly its book value, no more. Every dong of reinvested equity earns just enough to justify itself, creating no extra value. If ROE exceeds r, P/B rises above 1: the bank turns each dong of equity into more value than it costs, so the market pays a premium. If ROE falls below r, P/B drops below 1 and the bank trades at a discount to book — the market is saying its equity is being *destroyed*, not compounded.

That is the entire reason banks trade where they do. A bank's premium or discount to book is a direct, almost mechanical readout of whether it earns more than its cost of equity.

It pays to dwell on each lever in the master equation, because each maps onto a real business question. **ROE** is the quality of the franchise: cheap deposits, disciplined lending, and fee income all push it up. **Growth g** is the bank's expansion runway: a bank in a fast-growing economy with rising credit penetration can reinvest at high rates, so g is large; a bank in a saturated market has little to reinvest profitably, so g is small. The required return **r** is the market's price of risk for this particular equity: a safe, transparent, well-capitalized bank in a stable currency commands a low r, while a fragile, opaque, or politically exposed bank commands a high one. The justified multiple is just these three forces resolved into a single number. When you argue about whether VCB is cheap, you are really arguing about one of these three inputs — usually r or g — and the formula forces that argument into the open.

There is a subtle trap in the growth term worth flagging now, because it will recur in the dividend model. Growth *helps* value only when ROE exceeds r. If a bank earns less than its cost of equity, faster growth makes the discount to book *worse*, not better, because every reinvested dong destroys value and growth compounds the destruction. You can see this in the formula: when ROE < r, raising g (which appears with a minus sign in both the numerator and denominator) pushes P/B further below 1. Growth is not free; it is only valuable for a bank that already clears its cost-of-capital hurdle. VCB, earning 22% against a ~15% required return, clears it comfortably, so its growth is genuinely accretive — which is precisely why retaining earnings to grow, rather than paying them out, creates value for VCB shareholders even though it starves them of cash dividends today.

#### Worked example: deriving VCB's justified P/B from CAPM

First we need *r*, the cost of equity, which we estimate with the **Capital Asset Pricing Model (CAPM)**: r = risk-free rate + beta × equity risk premium. In Vietnam, a frontier-to-emerging market, the inputs are higher than in the US. Take the risk-free rate as the Vietnamese government bond yield, about 5.5%. Take the **equity risk premium (ERP)** — the extra return investors demand for holding stocks over bonds — as about 9%, reflecting Vietnam's country risk. Take VCB's **beta** (its sensitivity to the market) as about 1.1, slightly more volatile than the index. Then:

r = 5.5% + 1.1 × 9% = 5.5% + 9.9% = **15.4%**

Each input deserves a sanity check, because the cost of equity is where most valuation errors hide. The **risk-free rate** of 5.5% is the yield on a long-dated Vietnamese government bond — the return you can earn with (near) certainty in dong, which sets the floor for any risky dong investment. In the US this number is closer to 4.5%; Vietnam's is higher because its inflation and sovereign risk are higher, and that difference flows straight through to a higher cost of equity for every Vietnamese stock. The **equity risk premium** of 9% is the steepest input and the most debatable: developed-market ERPs run around 4.5% to 5.5%, but frontier and early-emerging markets carry a country-risk premium on top, often 3 to 5 extra points, to compensate for currency volatility, weaker investor protections, and lower liquidity. The **beta** of 1.1 says VCB moves slightly more than the Vietnamese market as a whole — reasonable for a large, liquid bank that the index leans on heavily. Change any of these and the cost of equity moves; that sensitivity is why we will not trust a single point estimate.

Now the growth rate. VCB retains most of its earnings to fund growth and meet capital rules. With ROE around 22% and a long-run sustainable growth assumption of g = 8% (well below b × ROE, deliberately conservative for a terminal rate in a maturing economy), plug everything in:

Justified P/B = (ROE − g) / (r − g) = (0.22 − 0.08) / (0.154 − 0.08) = 0.14 / 0.074 = **1.89x**

So the model says VCB's equity is fairly worth about 1.89 times its book value. The market is paying 1.65x. The takeaway: on these inputs VCB looks modestly *undervalued*, by roughly 15%, because its ROE towers over its cost of equity by nearly seven percentage points.

#### Worked example: from justified P/B to a price target

Translating a multiple into a price is one multiplication. Book value per share is about 55,000 VND. At the justified 1.89x:

Fair price = 1.89 × 55,000 VND ≈ **104,000 VND per share** (≈ \$4.16 at 25,000 VND/USD)

Against a mid-2024 market price near 90,000 VND (\~\$3.60), that is roughly 15% upside. Notice how levered this is to the inputs: the justified multiple sits on the *difference* between two small spreads, (ROE − g) over (r − g). Shift the cost of equity by a single percentage point and the answer moves materially — which is exactly why we will build a sensitivity grid before trusting any single number. The takeaway: a price target is only as solid as the spread between ROE and the cost of equity that produced it.

## Reading VCB's price-to-book through history

A justified multiple means little without context, so look at where VCB has actually traded. Over 2019 to 2024 its price-to-book swung from 2.8x down to a COVID-era 2.1x, spiked to 3.4x in the 2021 liquidity-fuelled rally, then compressed steadily to 2.0x, 1.8x, and finally about 1.65x by 2024.

![VCB price to book ratio time series 2019 to 2024](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-2.png)

Before reading the trend, fix what the chart measures. The vertical axis is the multiple of book value the market was willing to pay at each year-end — not the share price, which can rise even as the multiple falls if book value grows faster. VCB's book value per share roughly doubled over this window as the bank retained and compounded earnings, so the *price* held up far better than the *multiple*. That distinction matters: a falling multiple on a fast-growing book is a very different animal from a falling price on a shrinking book. The former is a re-rating of an improving business; the latter is decline. VCB is firmly the former.

The de-rating from 3.4x to 1.65x is the single most important fact about VCB's valuation today. It did not happen because the bank got worse — ROE stayed pinned near 22% throughout. It happened because the *denominator of value* changed: Vietnamese government bond yields rose, the risk-free rate climbed, real-estate-sector worries fattened the risk premium, and global emerging-market multiples compressed. In the language of our master formula, ROE held but *r* rose, so justified P/B fell, and the market price followed. A multiple is not a verdict on a company in isolation; it is a relationship between a company's returns and the prevailing cost of capital. When you see a bank's P/B fall while its profitability holds, suspect the discount rate, not the business.

#### Worked example: backing out the market's implied cost of equity

We can run the master formula in reverse. If the market prices VCB at 1.65x book with ROE = 22% and g = 8%, what cost of equity *r* is the market implicitly using? Set justified P/B equal to the observed 1.65 and solve:

1.65 = (0.22 − 0.08) / (r − 0.08) = 0.14 / (r − 0.08)

So r − 0.08 = 0.14 / 1.65 = 0.0848, which gives **r ≈ 16.5%**.

The market is discounting VCB at about 16.5%, a full percentage point above our 15.4% CAPM estimate. That gap *is* the valuation debate in one number: bulls argue 15.4% is right and the stock is cheap; bears argue the market's 16.5% correctly prices in extra credit and ownership risk. The takeaway: every market price embeds an implied required return, and reverse-engineering it turns a vague "is it cheap?" into a precise "do I think the right discount rate is 15.4% or 16.5%?"

## ROE under the lens: the Du Pont decomposition

ROE is the engine, so we should open it up. The **Du Pont identity** factors return on equity into three drivers, each telling a different story about *how* the bank earns its return. For a bank the cleanest three-factor form is:

ROE = Net profit margin × Asset turnover × Equity multiplier

where net profit margin = net income / total revenue, asset turnover = total revenue / total assets, and the equity multiplier = total assets / equity (a direct measure of leverage). Multiply them and revenue and assets cancel, leaving net income / equity — ROE.

#### Worked example: decomposing VCB's 22% ROE

Use approximate VCB 2024 figures. Say total operating revenue (net interest income plus fees) is about 68,000 billion VND, net income after tax is about 33,000 billion VND, total assets are 1,900,000 billion VND, and equity is 150,000 billion VND.

- Net profit margin = 33,000 / 68,000 ≈ **0.485** (48.5% of revenue drops to the bottom line — very high, reflecting VCB's low funding costs and low loan losses)
- Asset turnover = 68,000 / 1,900,000 ≈ **0.0358** (revenue is about 3.6% of assets — low, because banking is a thin-spread, high-volume business)
- Equity multiplier = 1,900,000 / 150,000 ≈ **12.7x** (assets are 12.7 times equity — the leverage inherent to banking)

Multiply: 0.485 × 0.0358 × 12.7 ≈ **0.220**, or 22% ROE — matching the headline. The decomposition shows VCB's elite ROE is *not* built on reckless leverage (12.7x is conservative for a bank; many peers run 14x to 16x). It is built on a fat profit margin: low-cost deposits and a best-in-class **non-performing loan (NPL) ratio** of about 1.1% mean little of VCB's revenue leaks away to interest costs or bad-debt provisions. The takeaway: a high ROE earned through margin and asset quality is far more durable than the same ROE earned through leverage, and that durability is what justifies paying up on P/B.

This is also why the model deserves trust. A 22% ROE produced by a 48.5% margin and modest leverage is a *quality* ROE; if VCB hit 22% by running 18x leverage on thin margins, you would slash the justified multiple for fragility. The Du Pont decomposition is the bridge between a number and a judgment.

The decomposition also tells you *where to look for the next move in ROE*, which is the same as where to look for the next move in valuation. Of the three factors, the equity multiplier is the most constrained: Basel II rules cap how much leverage VCB can run, so this lever is near its regulatory limit and cannot stretch further to lift ROE. Asset turnover, the revenue-to-assets ratio, is structurally low for any bank and moves slowly with the NIM; a tightening of the net interest margin as deposit competition heats up would nudge it down. That leaves the net profit margin as the swing factor, and within it, the **provision for credit losses** — the amount the bank sets aside for loans that may go bad. In a benign year, provisions are small and the margin is fat; in a credit downturn, provisions balloon and the margin collapses. So the entire fragility of VCB's elite ROE concentrates in one line item tied to the credit cycle. That is the analytical payoff of Du Pont: it does not just confirm the 22% number, it points a finger at the exact place the number is most likely to break.

It also reframes the peer comparison. When VCB out-earns BID or CTG on ROE, Du Pont lets you ask *which factor* drives the gap. If VCB's edge came purely from higher leverage, it would be a fragile, borrowed advantage. In fact its edge comes from a fatter margin — cheaper funding and lower loan losses — which is a *structural* advantage rooted in its deposit franchise and underwriting discipline. Structural margin advantages persist; leverage advantages reverse the moment a regulator tightens. This is the difference between a premium you should pay for and one you should fade, and only the decomposition reveals which you are looking at.

![VCB return on equity versus cost of equity 2019 to 2024](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-5.png)

The chart makes the value-creation case visually: in every year from 2019 to 2024, VCB's ROE (around 21% to 23%) sat far above its estimated cost of equity (around 13% to 15.5%). The vertical gap between the two bars is **economic profit** — return earned in excess of what capital costs. A bank that lives above the line, year after year, is compounding shareholder wealth, and the master formula rewards exactly that persistence with a P/B above 1.

## The dividend discount model for VCB, and the stock-dividend wrinkle

The P/B method and the DDM are two windows on the same building, but the DDM forces you to confront a peculiarly Vietnamese complication: **VCB pays most of its "dividends" in stock, not cash.**

A **cash dividend** sends money to your bank account. A **stock dividend** issues you new shares instead — you own more shares of the same total pie, so each share is worth proportionally less and your stake's value is unchanged on the day. Vietnamese banks lean heavily on stock dividends because the State Bank of Vietnam pressures them to retain capital to meet **Basel II** capital-adequacy requirements. Paying out cash would deplete the regulatory capital the bank needs to keep growing its loan book. So VCB's reported dividend yield of 2% to 3% is mostly *stock*, which is economically closer to retained earnings than to a cash payout.

This matters for a naive DDM, which discounts *cash* to shareholders. If you mechanically plugged in VCB's tiny cash dividend, you would value the bank at almost nothing — absurd for a business compounding equity at 22%. The fix is to value the bank on its **dividend-paying capacity**, not its actual stingy cash payout. We model the cash VCB *could* sustainably distribute once its growth matures and its capital needs ease, even though today it reinvests nearly everything. This is standard practice for high-growth franchises in capital-hungry, fast-growing economies.

There is a clean way to see why retained earnings are not lost to the shareholder. When VCB keeps a dong instead of paying it out, that dong does not vanish — it joins the equity base and immediately starts earning the bank's ROE of 22%. As long as 22% exceeds the 15.4% the shareholder requires, the shareholder is *better off* having that dong reinvested than received as cash, because the bank can compound it faster than the shareholder could elsewhere at equivalent risk. This is the deep logic behind the stock-dividend regime: it is not the regulator robbing shareholders of income, it is the regulator (with shareholders' interests incidentally aligned) forcing a high-return compounder to keep compounding. The value shows up not as dividends today but as a larger book value — and therefore a higher share price — tomorrow. The DDM captures this by pushing the payout into the future, where it eventually arrives once growth slows and the bank no longer needs every dong of retained capital.

This also explains a feature of the model that surprises beginners: most of VCB's value comes from cash flows many years out, in the mature stage, not from the next few years. That is the correct economic picture of a young compounder. A mature, slow-growing bank pays out most of its earnings now, so its value is front-loaded and its DDM is dominated by near-term dividends. A high-growth bank deliberately defers the payout to fund expansion, so its value is back-loaded into the terminal value. Neither is "riskier" on its face, but the back-loaded profile means the high-growth bank's valuation is far more sensitive to long-run assumptions — the terminal growth rate and the eventual payout — than to anything happening in the next reporting period. Keep that in mind when you see the two-stage arithmetic below: the small near-term dividends barely move the answer, while the terminal value carries almost all of it.

#### Worked example: a two-stage DDM for VCB

A **two-stage DDM** splits the future into a high-growth phase and a mature phase. Stage one: VCB keeps compounding book value at a high rate while reinvesting heavily, so explicit cash dividends stay small. Stage two: growth settles to a sustainable long-run rate and the bank can pay out a normal share of earnings.

Set it up per share. Current EPS ≈ ROE × BVPS = 0.22 × 55,000 ≈ 12,100 VND. Cost of equity r = 15.4%.

*Stage one (years 1–5), high growth at 14%, low payout 15%.* EPS grows 14% a year; the bank pays out only 15% as cash. Year-1 dividend ≈ 12,100 × 1.14 × 0.15 ≈ 2,069 VND, rising 14% annually to about 3,495 VND by year 5. Discounting those five rising dividends at 15.4% gives a present value of roughly **9,000 VND** — small, because the cash payout is deliberately throttled.

*Stage two (year 6 onward), mature growth g = 8%, payout rises to 50%.* By year 6, EPS ≈ 12,100 × 1.14⁵ × 1.08 ≈ 25,160 VND (\~\$1.01), and at a 50% payout the year-6 dividend ≈ 12,580 VND (\~\$0.50). Apply the Gordon terminal value: TV at end of year 5 = D₆ / (r − g) = 12,580 / (0.154 − 0.08) = 12,580 / 0.074 ≈ 170,000 VND (\~\$6.80). Discount that back 5 years at 15.4% (divide by 1.154⁵ ≈ 2.04): present value ≈ **83,300 VND** (\~\$3.33).

Intrinsic value ≈ 9,000 + 83,300 ≈ **92,300 VND per share.**

That lands almost exactly on the mid-2024 market price near 90,000 VND, and close to the 104,000 VND the justified-P/B method produced. Two independent equity-centric methods converging within a sensible band is the strongest signal a valuation can give. The takeaway: the bulk of a high-growth bank's value lives in the terminal value, so your long-run growth and payout assumptions matter far more than the next few years of throttled dividends.

Notice the dependency the terminal value exposes: it sits on (r − g) = 0.074 in the denominator, the same fragile spread as the P/B formula. Nudge g from 8% to 9% and the terminal value jumps by more than 15%. Both methods are, at heart, bets on the same two numbers — the return the bank earns and the return its owners require — which is reassuring, because it means we only have a handful of assumptions to defend.

## Relative valuation: VCB against its peers

A model gives you an absolute anchor; the market gives you a relative one. VCB does not trade in a vacuum, so compare its 1.65x P/B against the other major Vietnamese banks and a couple of regional reference points.

![Vietnam bank price to book comparison VCB versus peers](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-3.png)

Among the big Vietnamese names, VCB sits at the top: about 1.65x book versus CTG (VietinBank) at roughly 1.4x, BID (BIDV) at about 1.3x, MBB (Military Bank) near 1.3x, and TCB (Techcombank) around 1.2x. VCB's premium to its peers is *earned*, and the master formula explains why: VCB pairs the highest ROE with the lowest NPL ratio (1.1% versus 1.5% to 2.5% for many peers) and the strongest state-backed deposit franchise, which keeps its funding cheap and its risk low. A higher, safer ROE justifies a higher P/B — the premium is not sentiment, it is arithmetic.

There is a tidy way to confirm that the peer ranking is rational rather than arbitrary: pair each bank's P/B with its ROE and check that the two line up. A bank with double the ROE should, all else equal, command a higher multiple, and the *ratio* of P/B to ROE is a rough "price per unit of profitability." VCB at 1.65x book on 22% ROE prices each point of ROE at about 0.075x of book; a peer at 1.2x on 18% ROE prices each point at about 0.067x. The numbers are close, which tells you the market is broadly pricing these banks consistently on profitability — VCB's higher absolute multiple is bought with higher absolute returns, not paid as a blind premium. Where the ratios diverge sharply, you have a candidate mispricing worth investigating; where they cluster, as here, the market has done its arithmetic and any edge has to come from disagreeing with the *inputs* (will VCB's ROE hold?) rather than spotting a lazy multiple.

This is also the right place to note what relative valuation *cannot* do. Comparing VCB to its peers tells you whether it is cheap or dear *relative to Vietnamese banks*, but if the entire sector is mispriced — too cheap because the market over-fears the property cycle, or too rich in a liquidity bubble like 2021 — relative valuation will happily call VCB "fairly valued versus peers" while the whole group is wrong in absolute terms. That is exactly why we anchored first on the *absolute* methods (justified P/B and DDM, both tied to a cost of equity built from macro inputs) and only then cross-checked against peers. Relative valuation is a sanity check on the absolute work, never a substitute for it. The two together — an absolute anchor and a relative cross-check — are far more convincing than either alone, and when they agree, as they roughly do for VCB, your confidence in the conclusion should rise.

Stretch the lens regionally and the picture is humbling. Singapore's DBS trades around 1.8x book on an ROE in the high teens; Thailand's Kasikornbank trades near 1.0x on a lower ROE. VCB's 1.65x with a 22% ROE looks, on a return-per-unit-of-price basis, *better* than DBS — but the comparison is not apples to apples. DBS earns its return in a hard currency, a developed market, and with a free float that lets large investors actually build positions. VCB's return comes with Vietnamese-dong currency risk, frontier-market liquidity, and a state ownership stake of roughly 74% that leaves a thin tradable float. Relative valuation across borders is only honest once you adjust for those structural differences — which brings us to risk.

One concrete way to make the cross-border comparison honest is to compare the *spread* each bank earns over its own cost of equity, rather than the raw P/B. DBS earns perhaps 17% ROE against a developed-market cost of equity near 9%, a spread of about 8 points. VCB earns 22% against a 15.4% cost of equity, a spread of about 7 points. Once you adjust for the much higher hurdle VCB must clear, the two banks create value at broadly similar *rates* — DBS edges it — even though VCB's raw ROE looks far higher. This is the single most important discipline in emerging-market valuation: a headline 22% return is not impressive in isolation, only relative to the 15%-plus that investors rightly demand to bear Vietnamese risk. Strip away the higher hurdle and VCB's apparent superiority over a Singapore megabank largely dissolves, which is exactly why a naive yield-or-ROE screen would lead a global investor astray. The cost of equity is not an accounting nicety; it is the great equalizer that lets you compare a Hanoi bank to a Singapore one on the same footing.

## A third cross-check: P/E, and why P/B is the better bank lens

We have valued VCB two ways — justified P/B and DDM — and both pointed near 90,000 to 104,000 VND. A natural third lens is the **price-to-earnings (P/E)** ratio: share price divided by earnings per share. P/E is the multiple most investors reach for, so it is worth seeing where it lands and why it is less reliable for a bank than P/B.

#### Worked example: VCB's P/E and its link to P/B

EPS is about ROE × BVPS = 0.22 × 55,000 ≈ 12,100 VND. At a 90,000 VND price:

P/E = 90,000 / 12,100 ≈ **7.4x**

VCB trades at roughly 7.4 times earnings. Is that cheap? Compare to a justified P/E from the same Gordon machinery: justified P/E = payout ratio × (1 + g) / (r − g). With a *normalized* payout of, say, 40%, g = 8%, r = 15.4%: justified P/E = 0.40 × 1.08 / 0.074 ≈ 5.8x. On that basis 7.4x looks slightly *full*, the opposite of what P/B suggested. The discrepancy is not a contradiction — it is the payout assumption doing the work. P/E is hostage to the payout ratio, and VCB's payout is artificially suppressed by the stock-dividend regime, so a normalized-payout P/E understates fair value while a low-actual-payout P/E overstates it. The takeaway: P/B sidesteps the payout problem entirely by anchoring on equity rather than distributed earnings, which is exactly why it is the primary lens for a bank whose payout policy is distorted by regulation.

There is a clean identity linking the two multiples that makes the point unavoidable: P/B = P/E × ROE. Rearranged, P/E = P/B / ROE. Because ROE is high and stable for VCB, P/B carries the same information as P/E but without depending on how earnings are split between dividends and retention. When two banks have the same ROE, ranking them by P/B and ranking them by P/E give the same answer; when their payout policies differ — as VCB's does from a Western bank's — only P/B stays honest. This is the technical reason every serious bank analyst leads with price-to-book and treats P/E as the supporting actor.

## The risks that bend the multiple

Every input in our model is a place where reality can intrude. The honest way to value VCB is to name the risks explicitly and see how each one would move the answer.

**Credit-cycle risk.** A bank's worst enemy is a recession that turns performing loans into defaults. VCB's loan book has meaningful exposure to **real estate**, roughly 25% to 30% of loans directly or indirectly. Vietnam's property sector wobbled badly in 2022–2023, and a deeper downturn would lift the NPL ratio, force higher provisions, and crush ROE. In our model, a drop in sustainable ROE from 22% to 18% — entirely plausible in a hard credit cycle — would slash the justified P/B from 1.89x toward 1.2x. Credit risk is the single biggest swing factor.

The mechanism is worth tracing because it shows how a single bad-debt shock cascades through every part of the valuation. When loans sour, the bank must raise its **loan-loss provision**, an expense that reduces net income directly — so ROE falls. Lower ROE shrinks the (ROE − g) numerator of the justified P/B, pulling the fair multiple down. Worse, a rising NPL ratio makes the market question the *book value itself*: if loans carried at full value are actually worth less, then the equity those loans support is overstated, so the denominator of P/B (book value) is also suspect. And as fear spreads, investors demand a higher risk premium, lifting r and shrinking the (r − g) denominator of the multiple from the other side. A credit shock therefore hits the valuation through three channels at once — lower ROE, distrusted book value, and a higher discount rate — which is why bank stocks fall so violently in credit crises and why VCB's 1.1% NPL ratio, the lowest among major Vietnamese banks, is genuinely a valuation asset rather than a footnote. The buffer it provides against this triple cascade is precisely what justifies a chunk of its premium to peers.

**Regulatory and capital risk.** Vietnam's phased adoption of Basel II (and eventual Basel III) capital standards forces banks to hold more equity against risky assets. That is good for safety but it caps how fast VCB can grow and how much it can ever pay in cash, which is precisely why the stock-dividend wrinkle exists. Tighter rules lower sustainable growth *g* and the eventual payout, both of which trim intrinsic value.

**Currency risk.** A foreign investor earns returns in dong but spends in dollars. If the VND depreciates against the USD — a recurring feature of emerging markets running trade and rate differentials against the US — a perfectly good dong return shrinks once converted. This effectively raises the dollar-based cost of equity a foreign holder should demand, lowering the price they will pay.

**State-ownership overhang.** With the state holding about 74%, VCB's free float is small. Thin float means lower liquidity, larger price impact when big holders trade, and a persistent governance question — the controlling shareholder is the government, whose priorities (financial-system stability, directed lending) may not always align with minority shareholders' returns. Markets typically apply a modest discount for this, which partly explains why the *market's* implied cost of equity (16.5%) exceeds our clean CAPM estimate (15.4%).

The state-ownership story is genuinely two-sided, and an honest valuation holds both halves at once. On the negative side, the government as controlling shareholder can push the bank toward goals that serve the financial system rather than minority owners — lending to favored sectors, supporting weaker banks, or holding back on dividends to preserve capital. The thin float also caps how much institutional money can flow in, which suppresses the multiple a large foreign fund is willing to pay because it simply cannot build a meaningful position without moving the price. On the positive side, state ownership is a powerful *implicit guarantee*: VCB is effectively too important to fail, which lowers its true default risk and, in a crisis, its funding cost. That guarantee is part of why VCB enjoys the cheapest deposits in the system — savers trust it implicitly. So the same 74% stake that justifies a governance discount also underwrites the deposit-funding advantage that drives the elite ROE. The net effect on the multiple is a judgment call, but the analytically clean way to handle it is exactly what the market does: bake a slightly higher required return into r, which is the channel through which all of these soft, hard-to-quantify risks should flow.

A fourth risk deserves a mention because it is easy to overlook: **interest-rate and margin risk**. A bank's net interest margin depends on the gap between what it earns on loans and pays on deposits, and that gap moves with the rate environment and with competition. If the State Bank of Vietnam cuts policy rates to support growth, VCB's loan yields may fall faster than its deposit costs, compressing the NIM and with it the ROE. Conversely, intensifying competition for deposits — as fintechs and smaller banks chase savers with higher rates — raises funding costs and squeezes the margin from the other side. The NIM of 3.0% to 3.2% is not a constant; it is a contested spread, and a sustained half-point of compression would knock more than a point off ROE, feeding straight into a lower justified multiple through the now-familiar channel.

#### Worked example: a justified-P/B sensitivity grid

Because the answer is so levered to two inputs, never report a single number — report a grid. Vary ROE across 18% to 24% and the required return *r* across 13% to 17%, holding g at 8%, and recompute justified P/B = (ROE − 0.08) / (r − 0.08) in each cell.

![Justified P/B sensitivity grid by ROE and required return](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-6.png)

Read the grid and the whole valuation snaps into focus. At our base case (ROE 22%, r ≈ 15%), justified P/B is about 1.9x — above today's 1.65x, so cheap. But slide right along the bear path (ROE falls to 18% as the credit cycle bites, r rises to 16% as the risk premium fattens) and justified P/B collapses to about 1.2x — *below* today's price, so the stock would be expensive. The market's 1.65x is, in effect, a weighted average across this grid: the market is hedging between a benign base case and a sour bear case. The takeaway: a sensitivity grid converts a fragile point estimate into an honest map of which assumptions make the stock cheap and which make it dear, and it shows you exactly which variable to watch (here, ROE durability through the credit cycle).

## Bull, base, and bear: turning the model into price targets

Pull every thread together into three coherent scenarios. Each one is a *self-consistent story*: a view on ROE, on sustainable growth, and on the required return, which together imply a justified P/B and therefore a price target against book value of 55,000 VND.

![Bull base and bear case price targets for Vietcombank](/imgs/blogs/valuing-vietcombank-vcb-pb-ddm-case-study-7.png)

**Bear case.** The property cycle deteriorates, NPLs climb, provisions eat into profits, and sustainable ROE settles at 18%. Simultaneously, risk aversion pushes the required return to 16% and growth slows to 6%. Justified P/B = (0.18 − 0.06) / (0.16 − 0.06) = 0.12 / 0.10 = 1.20x. Target price ≈ 1.20 × 55,000 ≈ **66,000 VND** (\~\$2.64) — about 27% downside from 90,000.

**Base case.** VCB keeps doing what it has done: ROE near 22%, growth 8%, cost of equity 15.4%. Justified P/B = 1.89x, target ≈ **104,000 VND** (\~\$4.16) — roughly 15% upside, consistent with the DDM's 92,000 VND once you split the difference.

**Bull case.** Rates ease as Vietnamese inflation stays tame, the risk premium compresses as the market re-rates toward emerging-market status, beta falls, and the required return drops to 14% while ROE pushes to 24% and growth to 9%. Justified P/B = (0.24 − 0.09) / (0.14 − 0.09) = 0.15 / 0.05 = 3.00x, target ≈ **165,000 VND** (\~\$6.60) — the kind of multiple VCB actually commanded in 2021.

The asymmetry is the punchline. Base and bull cases offer 15% and 80% upside; the bear case threatens about 27% downside. At 1.65x book, with ROE durably above the cost of equity, the model says you are being paid more for the upside than you are risking on the downside — *provided* you believe the credit cycle stays manageable. That single belief is what the entire valuation reduces to.

It is worth being explicit about how to *use* three scenarios rather than collapse them into one. The temptation is to average the three price targets and call that "fair value," but that throws away the most useful information the scenarios carry, which is the shape of the distribution. A better discipline is to attach rough probabilities — say 25% bear, 55% base, 20% bull — and compute a probability-weighted value: 0.25 × 66,000 + 0.55 × 104,000 + 0.20 × 165,000 ≈ 16,500 + 57,200 + 33,000 ≈ **106,700 VND**. That expected value sits above the market price, but the more important output is the *spread*: the outcomes range from 66,000 to 165,000, a band wide enough that position sizing, not just the central estimate, should drive the decision. A stock with a tight band of outcomes can be held with conviction; one with this wide a band — driven by a credit cycle nobody can forecast precisely — should be sized for the possibility that the bear case is the one that arrives. The scenarios are not three guesses to be averaged; they are a map of how wrong you can be, and in which direction.

Notice, too, that each scenario is internally consistent rather than a free mix of inputs. In the bear case, the same deteriorating credit cycle that drags ROE to 18% also fattens the risk premium that lifts r to 16% and slows the growth that drops g to 6% — these move *together* because they share a cause. A common amateur error is to stress one input while holding the correlated ones fixed, producing a scenario that cannot actually happen. Good scenario design changes inputs in coherent bundles tied to a single narrative, which is exactly what makes the resulting price targets believable rather than arithmetic artifacts.

## Common misconceptions

**"A P/B below 1 means a bank is cheap."** No — it means the market expects the bank to earn *less* than its cost of equity. A bank trading at 0.6x book with a 6% ROE against a 12% cost of equity is correctly priced for value destruction, not mispriced. Cheapness is P/B *relative to justified* P/B, and justified P/B is (ROE − g)/(r − g). VCB at 1.65x against a justified 1.89x is cheap; a struggling peer at 0.8x against a justified 0.7x is expensive.

**"You can value a bank with EV/EBITDA like any other company."** No — a bank's deposits are its raw material, not its financing, so enterprise value is undefined in any useful sense. Forcing an EV multiple onto a bank produces a number with no economic meaning. Banks are always valued on equity: P/B, P/E, and DDM.

**"VCB's low dividend yield means it's a poor income stock, so it's overvalued."** No — VCB pays mostly *stock* dividends because the State Bank of Vietnam requires it to retain capital under Basel II. The retained earnings compound book value at 22%, which is worth far more than a cash payout. The proper valuation uses dividend-paying *capacity*, not the throttled current cash payout; do otherwise and a 22%-ROE compounder looks worthless.

**"VCB's premium to BID and CTG is just brand hype."** No — the premium is arithmetic. VCB carries the highest ROE (22% vs ~18% for some peers), the lowest NPL ratio (1.1% vs 1.5%–2.5%), and the cheapest deposit funding. A higher, safer ROE mechanically earns a higher justified P/B; the market is pricing fundamentals, not a logo.

**"The model gives a precise fair value, so I can trust the single number."** No — the answer sits on the spread (ROE − g)/(r − g), and small shifts in r or ROE move it a lot. The justified value is a *range* (66,000 to 165,000 VND across scenarios), not a point. Anyone quoting a bank's intrinsic value to the dong without a sensitivity grid is selling false precision.

## How it shows up in real markets

The clearest real-market lesson from VCB is the **2021-to-2024 de-rating**, and it is a case study in separating price from value. In early 2021, in a flood of post-COVID liquidity and near-zero global rates, VCB traded above 3.0x book. By 2024 it had fallen to about 1.65x — a roughly 45% multiple compression — even though its ROE barely moved, holding near 22% the whole way. An investor who watched only the falling price would conclude the business was deteriorating. An investor who held the master formula in hand would see the truth: the *cost of capital* rose. Vietnamese bond yields climbed, the risk premium fattened on property-sector fear, and global emerging-market multiples compressed. ROE held; *r* rose; justified P/B fell; the price obeyed. The business was fine; the discount rate changed.

A second real-market lesson is what the **stock-dividend convention does to naive screeners**. Many automated value screens rank stocks by cash dividend yield. VCB, paying mostly stock dividends to satisfy the State Bank of Vietnam's capital rules, screens as a low-yield, unexciting income name — and gets passed over by investors who never look past the screen. Yet it is one of the highest-quality compounders in the region, growing book value at over 20% a year. The lesson generalizes across emerging markets: regulatory and capital regimes distort the surface metrics, and you have to value the *capacity* to pay, not the reported payout, or you will systematically misprice the best franchises.

The third lesson is **why the regional comparison is a trap if taken literally**. On paper VCB's 1.65x P/B with 22% ROE looks like better value than Singapore's DBS at 1.8x with high-teens ROE. But VCB's return arrives in a soft currency, through a thin free float, under a 74% state owner, in a market that periodically seizes up. The extra return is compensation for exactly those risks, not a free lunch. Honest cross-border valuation always converts the comparison into a *risk-adjusted* required return — which is just our cost-of-equity input doing its job. The frontier-market premium in *r* is not a bug in the model; it is the model correctly demanding more return for more risk.

A fourth, more practical lesson is about **timing the inputs, not the price**. Because a bank's valuation is so tightly bound to its cost of equity, the most reliable signal for when VCB is likely to re-rate is not a chart pattern but a turn in the macro inputs that feed r. When Vietnamese government bond yields peak and start falling, the risk-free rate in our CAPM drops; when the property-sector fear that fattened the equity risk premium fades, the premium compresses; both pull the justified P/B up, and history shows the market price follows with a lag. An investor armed with the master formula watches those inputs — the bond yield, the credit-spread proxies, the NPL trend across the sector — rather than the share price itself, because the inputs lead and the price lags. This is the difference between valuing a bank and merely trading its stock: the valuation tells you *what* it is worth under a set of conditions, and tracking the conditions tells you *when* the market is likely to agree.

Value Vietcombank well and you can value any bank: estimate the cost of equity honestly, anchor on a justified P/B driven by the ROE-versus-cost-of-equity spread, cross-check with a dividend model that respects the local payout regime, decompose the ROE to judge its quality, and then — always — replace the single number with a grid of scenarios. The arithmetic is universal; only the inputs change.

## Further reading & cross-links

- [Price-to-book ratio: valuing equity on net worth](/blog/trading/asset-valuation/price-to-book-ratio-pb-valuation-equity) — the general P/B framework this case study specializes to banks.
- [Discount rates in practice: WACC, cost of equity, unlevered beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta) — how to build the cost-of-equity input that drives every number here.
- [The dividend discount model for equity valuation](/blog/trading/asset-valuation/dividend-discount-model-ddm-equity-valuation) — the multi-stage DDM machinery used in the VCB worked example.
- [Vietnam stock market valuation: VN-Index P/E dynamics](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics) — the market context and country-risk backdrop for the cost-of-equity inputs.
- [Bank stock analysis with financial ratios](/blog/trading/equity-research/bank-stock-analysis-financial-ratios) — the analyst workflow and ratio toolkit that complements this valuation method.

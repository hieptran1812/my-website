---
title: "Underwriting a Property Deal: How Pros Price a Building Before They Buy"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Underwriting is the discipline of pricing a property before you buy it: build the pro-forma, compute the return three ways, stress every assumption, and buy only when the number still works when you are wrong."
tags: ["real-estate", "property", "underwriting", "pro-forma", "cap-rate", "cash-on-cash", "irr", "noi", "stress-testing", "margin-of-safety", "vietnam", "investing"]
category: "trading"
subcategory: "Real Estate"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Underwriting is the discipline of pricing a building *before* you buy it: you build a projected income statement (the **pro-forma**), compute the return three different ways, and stress-test every assumption until you know exactly where the deal breaks.
>
> - The pro-forma walks the rent *down*: gross potential rent, minus vacancy, minus operating expenses, equals **NOI** (net operating income); minus the mortgage equals your **cash flow**. Every return is built on those two lines.
> - There is no single "return." A **cap rate** measures the building, a **cash-on-cash return** measures your cash in year one, and an **IRR** measures the whole hold including the sale — and they can disagree wildly.
> - The single number that moves the return most is the one you can see least: the **exit cap rate** you assume five years out. Lift it from 6% to 7% in our example and a 14.3% return falls to 6.0% — the rent never changed.
> - The one habit to keep: the gap between the asking price and your underwritten value is your **margin of safety**. Amateurs buy on a story and a monthly payment; pros buy on a spreadsheet that still works when the assumptions go wrong.

A friend of mine — call him **Minh** — almost bought a small rental building in Ho Chi Minh City in early 2024. The broker's pitch was clean and confident: a ₫7 billion (about \$270,000) property, "rents itself," "this area only goes up," and a monthly mortgage payment Minh could just about cover from his salary plus the rent. He had the deposit ready. He was, by his own description, about a week from signing. Then he did something almost nobody does before buying the single largest asset of their life: he built a spreadsheet. Not a fancy one — gross rent at the top, a few subtractions, a price at the bottom. And the building that "rented itself" turned out to throw off a *4.5% yield* while the mortgage that funded it cost *12%*. Every month, the building would quietly cost him money. The story was true; the number was a trap.

That spreadsheet is called **underwriting**, and it is the difference between buying a building and gambling on one. The word comes from insurance — an underwriter is the person who decides whether to take a risk and at what price, and who literally *writes their name under* the policy to own the decision. In real estate it means the same thing: before you commit capital, you build the deal's projected income, price its risks, and decide what it is actually worth to you — a number you can defend line by line. Most people never do it. They buy on three things a broker hands them: a story ("the area is hot"), a comp ("the unit next door sold for X"), and a monthly payment they think they can stomach. None of those three is a valuation. Underwriting turns a gut feeling into a number, and then asks the only question that matters: *does that number survive being wrong?*

The diagram below is the mental model for the whole post — the **pro-forma**, the projected income statement that sits at the heart of every underwrite. We start at the top with the rent the building could collect, subtract our way down through the things that eat it, and arrive at the cash flow that everything else is measured against.

![A vertical pro-forma flow showing gross potential rent of 420 million dong per year, minus vacancy of 21 million, minus operating expenses of 84 million, equals net operating income of 315 million, minus debt service of 240 million, ending at cash flow which feeds the cap rate, cash-on-cash, and IRR returns](/imgs/blogs/underwriting-a-property-deal-1.png)

We will build the whole thing from zero. By the end you will be able to take any income property — Minh's HCMC building, an American duplex, a strip mall, a warehouse — and in a page of arithmetic say what it earns, what it is worth, how much that value moves when interest rates or vacancy go against you, and how far below the asking price you would have to buy for the deal to still work when you are wrong. That is the entire skill, and it is teachable.

A quick note before we start: this is educational, not financial advice. We are going to explain how property deals are priced and where the model bites — not tell you to buy or sell anything.

## Foundations: what underwriting is, and the pro-forma it produces

Let's meet our two running characters and define every term before we use it. **Minh** is buying a small rental building in Ho Chi Minh City; we price everything for him in Vietnamese đồng (₫), with US-dollar equivalents in parentheses so an international reader stays oriented (USD/VND ≈ 25,900 as of mid-2026). **Dana** is a US investor buying a small two-unit building — a *duplex* — in a mid-sized American city; we price her deal in dollars (\$). Reusing the same two people lets the examples compound instead of restarting each time.

**Underwriting** is the process of building a forward-looking financial picture of a specific deal and deciding what it is worth to *you*, at *your* cost of money, under *your* assumptions. It is not appraisal (what a bank's valuer thinks the building is worth today for lending purposes) and it is not a comp (what the unit next door sold for). It is your own model of the deal's future cash, discounted by your own required return and stressed by your own pessimism. Two people can underwrite the same building and reach different values — and both can be right, because they have different costs of capital and different views of the future. The number that comes out is not "the price"; it is *your ceiling* — the most you can pay and still hit your return.

The central artifact underwriting produces is the **pro-forma**. *Pro forma* is Latin for "as a matter of form" — it means a projected, hypothetical financial statement, as opposed to one reporting what already happened. A property's pro-forma is just its projected annual income statement: how much rent it should collect, what it costs to run, what's left after the mortgage. Everything in underwriting is some manipulation of the pro-forma. Let's build it one line at a time, top to bottom, because the whole edifice is only as honest as these lines.

### Gross potential rent

**Gross potential rent (GPR)** is the rent the building would collect if every unit were occupied, every month, at full market rent, with every tenant paying on time. It is the theoretical ceiling — the most the building could ever produce from rent. For Minh's building, with several rentable units totalling a market rent of ₫35 million per month, GPR is `₫35M × 12 = ₫420 million per year` (about \$16,200). For Dana's duplex, each side rents for \$1,250 a month, so her GPR is `\$1,250 × 2 × 12 = \$30,000 per year`.

GPR is where the broker's spreadsheet usually starts *and stops* — they quote you the gross rent as if it were income. It is not. No building is ever full, on time, at full rent, all year. The whole rest of the pro-forma is the discipline of subtracting reality from that fantasy.

### Vacancy and credit loss

**Vacancy loss** is the rent you don't collect because a unit sits empty — between tenants, during renovations, in a soft market. **Credit loss** is the rent you don't collect because a tenant *should* have paid but didn't (they're late, they default, they skip). Together they are usually quoted as a single percentage of GPR — the **vacancy and credit loss** allowance, often just called "vacancy" for short.

What number? It depends on the market and the asset. A stabilized apartment in a tight rental market might run 3–5%; a building in a soft market or with high tenant turnover might run 8–12%. The honest discipline is to use the *market's* vacancy rate, not the building's current one, because the current one reflects today's specific tenants and can flatter a deal. For both Minh and Dana we'll use a baseline of **5%**. For Minh: `₫420M × 5% = ₫21 million` of lost rent. For Dana: `\$30,000 × 5% = \$1,500`.

The reason vacancy matters far more than its small percentage suggests: it comes straight off the top, and as we'll see, a small move in vacancy is a *large* move in the bottom line, because expenses don't fall when the rent does.

### Operating expenses

**Operating expenses (OpEx)** are the recurring costs of running the building: property tax, insurance, property management, repairs and maintenance, utilities the owner pays, common-area cleaning, landscaping. The critical word is *operating* — these are the costs to keep the building running and the rent flowing. Three big things are deliberately **excluded** from operating expenses, and getting this boundary right is the single most common place beginners and even brokers cheat:

- **The mortgage is not an operating expense.** Debt is how *you* chose to finance the building, not a cost of the building itself. It comes out lower down.
- **Capital expenditures (capex) are not operating expenses.** A new roof, a new HVAC system, repaving the car park — these are large, lumpy, occasional costs. They are real, but they're handled with a *reserve* (more on this shortly), not the operating line.
- **Income tax and depreciation are not operating expenses.** They depend on your tax situation, not the building's operations.

For Minh, operating expenses (property tax, management at ~8% of rent, insurance, repairs, building fees) run about ₫84 million a year — roughly 20% of GPR, a typical figure for a managed residential building. For Dana, US operating expenses run heavier — American property taxes and insurance are high — so we'll use **45% of GPR**, a common rule of thumb for US small residential: `\$30,000 × 45% = \$13,500`. (We'll use a clean **\$13,500 → NOI of \$30,000** version of Dana's numbers in a moment; for now hold the structure.)

A warning that will recur: a seller's pro-forma almost always *understates* operating expenses, because every dollar of understated expense becomes a dollar of overstated NOI, which — through the cap-rate machinery — becomes many dollars of overstated value. The professional move is to take the seller's expense numbers, then add a contingency (often ~10%), and *never* use a building's expenses below the market norm for its type.

### NOI: the number everything hangs on

**Net operating income (NOI)** is what's left after vacancy and operating expenses but *before* the mortgage, before capex, before income tax, before depreciation:

$$\text{NOI} = \text{GPR} - \text{Vacancy} - \text{Operating Expenses}$$

NOI is the single most important number in income real estate. It is the building's true earning power, independent of how *you* financed it or what *your* tax bracket is. Because it strips out the mortgage, two buyers — one paying all cash, one borrowing 80% — compute the *same* NOI for the same building. That's the point: NOI is a property of the *building*, not of the buyer. We cover NOI and the cap rate in depth in [Cap Rate, NOI, and the Income Approach](/blog/trading/real-estate/cap-rate-noi-and-the-income-approach); here we just need it as the trunk that the three returns branch from.

#### Worked example: the full pro-forma on Minh's ₫7bn building

Let's run Minh's deal top to bottom, exactly as in Figure 1.

- **Gross potential rent:** ₫35M/month × 12 = **₫420 million/yr** (≈ \$16,200).
- **− Vacancy & credit loss (5%):** −₫21 million. *Running total: ₫399M.*
- **− Operating expenses:** −₫84 million (tax, ~8% management, insurance, repairs, fees). *Running total:* **₫315 million.**
- That ₫315 million is the **NOI**.

Now the headline number a broker would never volunteer. Minh is being asked to pay **₫7.0 billion** for this building. His **cap rate** — NOI divided by price — is:

$$\text{cap rate} = \frac{\text{₫315M}}{\text{₫7{,}000M}} = 4.5\%$$

The building, run perfectly, yields **4.5%** on the price — *before* a single đồng of mortgage. Hold that 4.5% in your mind, because we are about to compare it to the cost of the money Minh would borrow to buy it, and the comparison is the whole game.

*The pro-forma is just honest subtraction: every line you skip is a line the seller gets to keep in their favour.*

#### Worked example: the same pro-forma on Dana's US duplex

Run the identical structure for Dana, in dollars, so you can see the machine is universal and only the numbers change.

- **Gross potential rent:** \$1,250/month × 2 units × 12 = **\$30,000/yr** — wait, that's the *net-of-vacancy* simplification we'll use for the returns; let's do it properly. Her two sides at \$1,667/month each give a true GPR of `\$1,667 × 2 × 12 = \$40,000`.
- **− Vacancy & credit loss (5%):** −\$2,000. *Running total: \$38,000.*
- **− Operating expenses (~20% of GPR for a self-managed duplex with US taxes and insurance):** −\$8,000. *Running total:* **\$30,000.**
- That \$30,000 is the **NOI**, and on a \$500,000 price it's a **6.0% cap rate.**

Notice that Dana's duplex (6.0% cap) yields *more* than Minh's building (4.5% cap) on the same kind of asset. That is not because American buildings are better — it's because American buyers demand a higher yield, since US mortgage rates and US bond yields are higher than they were, and a cap rate is fundamentally the bond yield plus a risk-and-growth premium. The *same* building would trade at a lower cap rate (a higher price per dollar of income) in a low-rate market and a higher cap rate in a high-rate one. We'll carry Dana's \$500,000 / \$30,000-NOI / 6%-cap duplex as the running deal through every return calculation below.

*A pro-forma is the same machine in every country and currency — gross rent in at the top, reality subtracted line by line, the building's true earning power out at the bottom.*

## Building the pro-forma: from NOI down to the cash in your pocket

NOI is the building's income. It is *not* your income. To get from the building's NOI to the cash that actually lands in your bank account, you have to subtract the things that are specific to *your* deal — chiefly the mortgage, and a reserve for the big repairs that NOI politely ignores.

### Debt service: the mortgage, made explicit

**Debt service** is the total of principal and interest you pay on the mortgage over a year. If you put no money down it would be huge; if you pay all cash it is zero. It is entirely a function of *your* financing choice — the loan amount, the interest rate, and the amortization (how the loan is paid down over time).

Suppose Minh borrows **₫4.0 billion** of the ₫7.0 billion price (a 57% loan-to-value) at a Vietnamese mortgage rate. Here we hit a brutal local reality the global textbooks gloss over. Vietnamese home loans are typically sold with a **teaser rate** — a preferential ~8% for the first 12–24 months — that then **resets to a floating rate of 12–14%**. The teaser is the trap: the deal "pencils" at 8% and then quietly stops working when the rate resets. An honest underwrite models the *reset* rate, not the teaser. At a fully-reset ~12% on ₫4.0 billion, Minh's annual debt service (interest-heavy in the early years) is roughly **₫240 million a year** (₫20 million a month). We unpack exactly how mortgages amplify property returns in [Leverage and the Mortgage](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property); the number we need here is that ₫240 million.

### Before-tax cash flow

**Before-tax cash flow (BTCF)** is the NOI minus the debt service — the cash the deal produces *for you*, before you account for income tax:

$$\text{Before-tax cash flow} = \text{NOI} - \text{Debt service}$$

For Minh: `₫315M − ₫240M = ₫75 million a year` (≈ \$2,900). That is the cash the building actually drops into his account in year one, after the mortgage. Notice it is *positive* — but barely, and only because he put 43% down. The building earns 4.5% and the debt costs 12%; the deal survives the gap only because most of the price is Minh's own (cheap, 0%-interest) equity rather than the bank's expensive money. This is **negative leverage** — borrowing at a rate *higher* than the asset yields — and we'll see in a moment how it eats returns. The instant Minh's teaser rate resets and his payment jumps, that ₫75 million can flip negative.

### The capex reserve: the line that makes pro-formas honest

There is one more subtraction that separates a real underwrite from a broker's spreadsheet: the **capital expenditure reserve**, or *capex reserve*. Roofs wear out. Lifts fail. HVAC systems die. Car parks need repaving. These costs don't appear every year, so they're invisible in any single year's income statement — which is exactly why amateurs forget them and then get blindsided by a ₫400-million roof in year six. The discipline is to set aside a reserve *every* year — commonly **5–10% of gross rent**, or a per-unit figure — so the money is there when the lumpy bill arrives.

A reserve of, say, 5% of Minh's GPR is `₫420M × 5% = ₫21 million a year`. Strictly, this should come out of cash flow too, which would cut his ₫75 million to ₫54 million — and that matters. For clarity in the worked examples that follow we'll fold the capex reserve into our stress cases rather than the base case, but a real underwrite carries it as a standing line. *The capex reserve is the difference between a building's income and your income: the building doesn't pay for its own new roof — you do.*

Now we have the full vertical pro-forma of Figure 1: **GPR → − vacancy → − OpEx = NOI → − debt service = cash flow**, with a capex reserve standing alongside. From those two anchor lines — NOI and cash flow — we can finally compute the returns.

## The three returns: cap rate vs cash-on-cash vs IRR

Here is the single most clarifying idea in this whole post: **there is no such thing as "the return."** A property throws off at least three different returns, each answering a different question, each ignoring something different. Conflating them is the most expensive mistake in amateur real estate — the source of "but it yields 9%!" pitches that quietly lose money. Figure 2 lays the three side by side.

![A three-by-three matrix comparing cap rate, cash-on-cash, and IRR across what each one measures, what each one ignores, and when to use each one](/imgs/blogs/underwriting-a-property-deal-2.png)

### Cap rate: the return on the building

The **capitalization rate (cap rate)** is NOI divided by price — the unlevered yield on the building itself, as if you paid all cash:

$$\text{cap rate} = \frac{\text{NOI}}{\text{Price}}$$

It **measures** the building's earning power relative to its price. It **ignores** your loan entirely, ignores time (it's a single-year snapshot), and ignores the eventual sale. You **use it** to compare two buildings quickly and to sense whether a price is rich or cheap relative to the market's prevailing cap rate. Minh's building, at ₫315M NOI on a ₫7,000M price, has a **4.5% cap rate**. Dana's duplex, at \$30,000 NOI on a \$500,000 price, has a **6.0% cap rate**. The cap rate is the lingua franca of property — but it tells you nothing about what *you* will earn, because you're not paying all cash.

### Cash-on-cash: the return on your money in year one

The **cash-on-cash return** is your before-tax cash flow divided by the actual cash you put in (your down payment plus closing costs):

$$\text{cash-on-cash} = \frac{\text{Before-tax cash flow}}{\text{Cash invested}}$$

It **measures** how hard the cash *you personally tied up* works in year one. It **ignores** every year after the first, ignores the sale, and ignores the time value of money. You **use it** to check whether a deal feeds itself from day one or bleeds you while you wait for appreciation. This is where leverage enters: the same building has *one* cap rate but *many* possible cash-on-cash returns depending on how much you borrow.

For Minh: his cash in is ₫3.0 billion (the ₫7bn price minus the ₫4bn loan), and his before-tax cash flow is ₫75 million. So his cash-on-cash is `₫75M / ₫3,000M = 2.5%`. His money earns a measly 2.5% in cash terms in year one — *worse* than the building's 4.5% cap rate, because his expensive 12% debt is dragging the return down. That is negative leverage in a single number.

### IRR: the return on the whole hold

The **internal rate of return (IRR)** is the one number that ties everything together: the equity you put in at the start, every year's cash flow, and the lump sum you collect when you sell. Formally, the IRR is the single annual discount rate that makes the **net present value** of all those cash flows equal to zero — the rate at which the money you get back exactly justifies the money you put in, accounting for *when* each dollar arrives.

$$0 = -C_0 + \sum_{t=1}^{T}\frac{\text{CF}_t}{(1+\text{IRR})^t} + \frac{\text{Sale}_T}{(1+\text{IRR})^T}$$

Here $C_0$ is the equity you invest at time zero, $\text{CF}_t$ is the cash flow in year $t$, and $\text{Sale}_T$ is the net proceeds when you sell in the final year $T$. The IRR is the value that balances the equation.

It **measures** the complete return on a multi-year hold, properly weighting early cash more than late cash. It **ignores** almost nothing about the deal — but it carries a subtle assumption (that interim cash can be reinvested at the IRR itself, which inflates very high IRRs) and, crucially, it depends entirely on the **sale price you assume**, which is a guess about the future. You **use it** to rank deals with different shapes — a low-cash-flow / high-appreciation deal against a high-cash-flow / flat one — on a single comparable axis. We'll compute a full IRR in the next section; the headline is that IRR is the most honest of the three returns *and* the most assumption-dependent, because the biggest input is the sale price you can't yet see.

The three returns can disagree violently. Here are all three, side by side, for our two running deals:

| Metric | What it asks | Minh's HCMC building | Dana's US duplex |
|---|---|---|---|
| **Cap rate** | What does the building yield, all-cash? | 4.5% | 6.0% |
| **Cash-on-cash** | What does my cash earn in year one? | 2.5% | 6.5% |
| **5-year IRR** | What's my full return including the sale? | depends on the exit | ~14.3% (base case) |

Read across Minh's row and the whole tragedy of his deal is visible at a glance: his cash-on-cash (2.5%) is *below* his cap rate (4.5%), the unmistakable fingerprint of negative leverage — his expensive 12% debt is dragging his return *below* what the all-cash building would have earned. Borrowing made him *poorer* in cash terms, not richer. His only path to a respectable IRR runs entirely through appreciation, which is to say through an exit assumption he cannot control. Dana's row tells the opposite story: her cash-on-cash (6.5%) sits slightly *above* her cap rate (6.0%) — mild positive leverage — and her 14.3% IRR is anchored by a real, defensible sale. A buyer who looks only at cash-on-cash would reject Minh and like Dana; a buyer who looks only at the appreciation story might overpay for Minh. The underwriter looks at all three and asks which one the deal actually depends on. *Three returns, three questions: the cap rate prices the building, cash-on-cash prices your year one, and the IRR prices the whole journey including the exit.*

One more honest caveat about the IRR, because it's the metric people most often misuse. The IRR's headline number assumes you can *reinvest* every interim cash flow at the IRR itself — so a deal quoting a 25% IRR is implicitly assuming you'll find *other* 25% homes for the rent as it dribbles in, which you usually won't. This is why very high IRRs flatter deals that return their money slowly and why sophisticated investors pair IRR with an **equity multiple** — total cash returned divided by cash invested — which doesn't make that reinvestment assumption. Dana's deal returns about \$273,000 (five years of cash flow plus the \$218,000 net sale) on \$150,000 in, an equity multiple of roughly **1.8×**. The IRR tells you the *rate*; the multiple tells you the *amount*; you want both before you believe a deal.

## Financing the deal: how leverage amplifies the return — both ways

**Leverage** — using borrowed money to control an asset larger than your cash — is the engine of real-estate returns and the source of most real-estate ruin. The mechanism is simple and worth internalizing precisely: when you borrow, you keep *all* of the building's income above the loan's cost, but on a *smaller* slice of your own cash. If the building out-earns the loan, that's wonderful — your return on equity rises. If the building under-earns the loan, it's the reverse, and brutally so. Figure 4 shows both directions on a clean example.

![A before-and-after comparison of the same deal, unlevered on the left earning a seven percent cap rate that falls to five point six percent in a downside, and levered with a seventy percent loan on the right earning nine point three percent cash-on-cash that falls to four point seven percent in the same downside](/imgs/blogs/underwriting-a-property-deal-4.png)

To isolate exactly what leverage *does*, take a clean teaching version of Dana's duplex: price \$500,000, NOI \$35,000 — a **7% cap rate** — and a loan at **6%**. Here the building yields *more* than the loan costs: a **positive 1-point spread**. Watch what borrowing does.

#### Worked example: cap rate vs cash-on-cash with 70% leverage

**Unlevered (all cash).** Dana pays the full \$500,000 in cash. Her income is the full \$35,000 NOI (no mortgage). Her return is just the cap rate:

$$\frac{\$35{,}000}{\$500{,}000} = 7.0\%$$

**Levered (70% loan).** Now Dana borrows 70% — \$350,000 at 6% — and puts in only \$150,000 of her own cash. Her annual debt service is `\$350,000 × 6% = \$21,000`. Her before-tax cash flow is `\$35,000 − \$21,000 = \$14,000`. Her cash-on-cash return is:

$$\frac{\$14{,}000}{\$150{,}000} = 9.3\%$$

Leverage lifted her return from **7.0% to 9.3%** — a 2.3-point jump — purely because she borrowed at 6% to buy something that earns 7%. She captured the 1-point spread on the borrowed \$350,000 and added it to her own \$150,000.

**Now the downside.** Suppose NOI falls 20% — a recession, a soft rental market — to \$28,000. Unlevered, Dana's return drops to `\$28,000 / \$500,000 = 5.6%`: unpleasant, but she still clears more than 5%. Levered, her cash flow is `\$28,000 − \$21,000 = \$7,000`, and her cash-on-cash is `\$7,000 / \$150,000 = 4.7%` — now *below* the unlevered return. The debt service is fixed; when income falls, the loan eats a bigger and bigger share of a shrinking pie, and the leveraged owner falls behind the all-cash owner. Push NOI down far enough and the cash flow goes negative — the building stops feeding the mortgage, and Dana has to feed it from her pocket.

*Leverage doesn't change the building; it changes the bet — it stretches the same outcome wider in both directions, so it rewards a deal that works and punishes a deal that doesn't.*

### Positive, zero, and negative leverage

The single fact that determines whether leverage helps or hurts is the **spread between the cap rate and the loan rate**:

- **Positive leverage** — cap rate *above* the loan rate (Dana's 7% cap vs 6% loan). Borrowing lifts your return. More leverage is more return (and more risk).
- **Zero leverage** — cap rate *equal to* the loan rate. Borrowing does nothing to your cash-on-cash; it just equals the cap rate. Dana's *running* duplex, at a 6% cap with a 6% loan, sits here: her cash-on-cash is 6%, identical to the unlevered yield. Leverage adds only risk, no reward.
- **Negative leverage** — cap rate *below* the loan rate. This is Minh's situation: a **4.5% cap** against a **12% loan**. Every borrowed đồng *lowers* his return. More leverage is *less* return. The only thing that rescues such a deal is appreciation — the bet is no longer on income but on price growth, which is a fundamentally different and riskier wager.

This is why the same "buy with as little down as possible" advice that builds wealth in a positive-leverage market (much of the US in a normal rate environment) *destroys* it in a negative-leverage one (Vietnam in 2024, where mortgage rates of 12% tower over residential yields of 2–5%). The Vietnamese buyer who maximizes leverage is maximizing a *negative* spread — turbocharging a loss per đồng borrowed — and betting the whole thing on prices rising fast enough to bail them out. The 2022–23 freeze, when prices stalled and that appreciation bet failed, is exactly what negative leverage looks like when the rescue doesn't come.

## The exit: going-in vs exit cap, and why it dominates everything

Here is the assumption that quietly determines whether your deal is brilliant or ruinous — and the one almost every amateur pro-forma either ignores or fudges optimistically: the **exit cap rate**, the cap rate at which you assume you'll *sell* the building years from now.

When you buy, you buy at the **going-in cap rate** — today's NOI over today's price. When you sell, the buyer prices the building off *their* expected NOI and *their* required cap rate — the **exit cap rate**. Because value equals NOI divided by cap rate, your sale price is roughly `(future NOI) ÷ (exit cap)`. A tiny change in that exit cap rate moves the sale price enormously, and because the sale is usually the largest single cash flow in the whole deal, it moves your IRR more than anything else you can control.

The professional rule of thumb is conservative and worth tattooing on the inside of your eyelids: **assume your exit cap rate is equal to or higher than your going-in cap rate.** Cap rates tend to *rise* over a long hold as a building ages, and assuming they'll *fall* — that you'll sell at a richer yield than you bought — is betting that the market will pay you more for the same income later. Sometimes it does. But an underwrite that *requires* cap-rate compression to work is an underwrite betting on luck.

Let's make the dominance concrete with a full IRR, using Dana's running duplex (the 6%-cap version): \$500,000 price, \$30,000 going-in NOI, a \$350,000 loan at 6% (debt service \$21,000), \$150,000 of equity, NOI growing 2.5% a year, a five-year hold, and 2% selling costs. Figure 3 lays out the cash flows on a timeline.

![A five-year timeline of Dana's deal: year zero shows the equity outflow, years one through five show growing annual cash flows, and year five adds the sale proceeds, all feeding into a single internal rate of return](/imgs/blogs/underwriting-a-property-deal-3.png)

#### Worked example: the IRR over a five-year hold

Lay out every cash flow:

- **Year 0:** Dana puts in **−\$150,000** of equity.
- **Years 1–5:** her NOI grows 2.5% a year (\$30,000 → \$30,750 → … → \$33,941), and her debt service is a flat \$21,000. So her before-tax cash flows climb from about **+\$9,800** in year 1 to **+\$12,900** in year 5.
- **Year 5 — the sale.** The buyer prices off the *year-6 forward NOI* of about \$34,800. At a **6% exit cap rate** (equal to the going-in cap — the honest base case), the sale price is `\$34,800 ÷ 6% = \$580,000`. After 2% selling costs (\$11,600) and repaying the \$350,000 loan, Dana's net sale proceeds are about **+\$218,000**.

Feed those into the IRR equation — the equity out, the five climbing cash flows, the big sale in year five — and the single rate that balances it is:

$$\text{IRR} \approx 14.3\% \text{ per year}$$

A 14.3% annual return over five years. Notice where it comes from: the five years of cash flow total only about \$55,000, while the sale alone returns \$218,000 against \$150,000 in. **The exit is the deal.** The cash flow is almost a rounding error next to the sale — which means the *assumption* about the sale is where the entire return lives. Change that one assumption and you change everything, as we're about to see. *The IRR feels like a fact about the building, but most of it is a forecast about a sale that hasn't happened — and a forecast is only as good as the cap rate you plug into it.*

## Stress-testing the assumptions: where the deal breaks

A single-point estimate — "this deal returns 14.3%" — is not an underwrite; it's a hope with a decimal place. The real work is **stress testing** (also called *sensitivity analysis*): deliberately moving each assumption against you, one at a time and then together, to find out *where the deal breaks* and *which assumption breaks it*. You are not trying to predict the future. You are trying to find out how wrong you can be and still be fine — and which mistake would hurt most.

Figure 5 isolates the single most important sensitivity: the IRR as a function of the exit cap rate, holding everything else at the base case.

![A line chart showing Dana's five-year levered IRR falling steadily as the assumed exit cap rate rises, from about twenty-three percent at a five percent exit cap down through fourteen point three percent at the six percent base case to six percent at a seven percent exit cap and below zero past seven point eight percent, with the region above the going-in cap shaded to mark cap-rate expansion](/imgs/blogs/underwriting-a-property-deal-5.png)

Look at the slope. Every half-point of exit cap rate costs roughly four points of IRR. The deal returns 14.3% if Dana sells at a 6% cap, but only **6.0% if she sells at a 7% cap** — and a 7% exit cap is not a disaster scenario, it's just "interest rates were a bit higher when I sold than when I bought." Push the exit cap to 7.8% and her IRR hits *zero*: she held a building for five years, collected rent, repaid a loan, and made nothing — because the cap rate expanded one and a half points. This is precisely what happened to leveraged owners worldwide in 2022 when rates spiked: NOI was *rising*, and values *fell*, because exit cap rates blew out. We trace that cap-rate-and-yield mechanism in [Cap Rate, NOI, and the Income Approach](/blog/trading/real-estate/cap-rate-noi-and-the-income-approach).

Now widen the stress test to every assumption at once. Figure 6 runs the base case against four downside scenarios.

![A stress-test matrix listing five scenarios — base case, rate up, exit cap up, vacancy up, and all three together — each with the change made, the resulting five-year IRR, and a verdict ranging from healthy to broken](/imgs/blogs/underwriting-a-property-deal-6.png)

#### Worked example: the stress test that crushes the return

Start from Dana's 14.3% base case and move one lever at a time:

- **Rate up (6% → 8%).** Her loan resets higher; debt service jumps from \$21,000 to \$28,000, cutting every year's cash flow by \$7,000. IRR falls to about **10.2%**. Painful but survivable — the cash flow buffer absorbs it.
- **Exit cap up (6% → 7%).** Nothing about the building changes; she just assumes the next buyer demands a 7% yield instead of 6%. Her sale price drops from \$580,000 to about \$497,000. IRR falls to **6.0%** — the single largest hit from a single assumption, and the one she has the *least* control over.
- **Vacancy up (5% → 15%).** A soft rental market. This is the cruelest because of operating leverage: her gross rent is \$40,000, of which a 5% vacancy was \$2,000; a 15% vacancy is \$6,000, and since her expenses don't shrink, that extra \$4,000 comes *straight* off an NOI that was only \$30,000 — a 13% NOI cut to \$26,000. IRR falls to **3.7%**.
- **All three together — rate to 8%, exit cap to 7%, vacancy to 15%.** Now the downsides compound: lower NOI, higher debt service, and a lower sale price all at once. The IRR goes **negative, about −13.4%.** Dana doesn't just earn a poor return; she *loses money* — the building she held for five years returns less than she put in.

That last row is the entire point of stress testing. A deal that returns a glorious 14.3% in the base case turns into a 13.4%-a-year *loss* when three ordinary, plausible things go wrong at once — none of them a crash, just a worse-than-hoped rate, a slightly higher exit cap, and a softer rental market. *Stress testing doesn't tell you what will happen; it tells you how much pessimism the deal can absorb before it stops being an asset and starts being a liability.* For the full discipline of managing these downside paths, see [Risk Management in Real Estate](/blog/trading/real-estate/risk-management-in-real-estate).

#### Worked example: vacancy and the exit cap crushing Minh's HCMC deal

Run the same stress on Minh's building, where the starting point is already weaker because of negative leverage. His base pro-forma: GPR ₫420 million, vacancy 5% (−₫21M), expenses ₫84M, **NOI ₫315 million**, on a ₫7.0 billion price — a 4.5% going-in cap, financed with a ₫4.0 billion loan whose 12% reset rate costs ₫240M a year, leaving ₫75 million of before-tax cash flow.

Now move two levers against him.

- **Vacancy 5% → 15%.** HCMC's rental market softens — perhaps a wave of new supply completes nearby, a real risk in a city building apartments at full speed. His vacancy loss jumps from ₫21 million to `₫420M × 15% = ₫63 million`. Because his operating expenses don't shrink, the extra ₫42 million comes straight off NOI: `₫315M − ₫42M = ₫273 million`. His cap rate on the ₫7.0 billion price falls from 4.5% to `₫273M / ₫7,000M = 3.9%`. And his cash flow? `₫273M − ₫240M = ₫33 million` — less than half of what he started with, and one bad month from negative.
- **Exit cap 5% → 6%.** Suppose Minh planned to sell in five years at a 5% cap. If, instead, rates are higher when he sells and buyers demand a 6% cap, watch the sale price. On a roughly ₫340 million forward NOI, a 5% exit cap implies a sale of `₫340M / 5% = ₫6.8 billion`; a 6% exit cap implies `₫340M / 6% = ₫5.67 billion` — a **₫1.13 billion**, or 17%, haircut on the largest cash flow in the entire deal, from a one-point move in an assumption he set years earlier. That single change can turn a modest appreciation gain into an outright loss, because Minh's whole thesis — remember, this is a negative-leverage deal — *rested* on the exit. The income never justified it; only the sale could, and the sale just shrank 17%.

Stack both stresses together — a softer rental market *and* a one-point-higher exit cap — and Minh's deal, which limped along on appreciation hope at the base case, becomes a clear money-loser: thinner cash flow every year and a much smaller exit. *The stress test is where Minh's deal reveals its true nature: it was never an income deal that might appreciate; it was an appreciation bet that barely covered its costs, and the moment either the income or the exit disappoints, there is nothing holding it up.*

## The margin of safety: price vs underwritten value

Everything so far produces a number: your **underwritten value** — the most you can pay for this building and still hit your required return, given honest assumptions. The final, decisive act of underwriting is to compare that number to the **asking price**. The gap between them is your **margin of safety** — a term borrowed from value investing that means exactly what it sounds like: the cushion of room you leave for being wrong.

The logic is humbling and liberating at once. You will be wrong about *something* — vacancy, the rate reset, the exit cap, a surprise roof. Your assumptions are educated guesses, not facts. The margin of safety is what lets you be wrong and still be fine. If your honest underwriting says the building is worth ₫6.5 billion to you and the asking price is ₫7.0 billion, you do not have a margin of safety — you have a *negative* one. You'd be paying ₫500 million *above* your own value, betting your whole return on every optimistic assumption coming true. If instead you can buy at ₫6.0 billion against a ₫6.5 billion underwritten value, you have a ₫500-million cushion: the deal can disappoint and still work.

#### Worked example: underwritten value vs asking price

Return to Minh's HCMC building. The broker's asking price is **₫7.0 billion**. Minh's honest underwrite — NOI of ₫315 million, a required going-in cap rate of, say, **5.0%** to compensate him for the risk and his 12% cost of debt — says the building is worth, to him:

$$\text{underwritten value} = \frac{\text{NOI}}{\text{required cap}} = \frac{\text{₫315M}}{5.0\%} = \text{₫6.3 billion}$$

His number is **₫6.3 billion**; the asking price is **₫7.0 billion**. The margin of safety is **negative ₫700 million** — he'd be paying 11% *over* his own value. And that's before the stress test, which shows the deal turning cash-flow-negative the moment his teaser rate resets. The disciplined conclusion isn't "don't buy property"; it's "*this* building, at *this* price, with *this* financing, has no margin of safety — so either negotiate to ₫6.0 billion, find cheaper debt, or walk." Minh walked. Six months later the same building was still listed, now at ₫6.6 billion. The discipline didn't just protect him from a bad deal; it gave him the patience to wait for a better entry — or a better deal entirely.

This is the deepest lesson of underwriting. The margin of safety converts a forecast you can't trust into a decision you can defend. You don't need to predict the future correctly; you need to *buy cheaply enough that being wrong is survivable.*

How big a margin is enough? There's no universal number, because the right cushion scales with how uncertain your inputs are. A stabilized, fully-leased building in a deep, liquid market with verified rents needs less margin than a half-empty building in a thin market with a story attached. A rough professional instinct: the riskier and less verifiable the deal, the wider the gap you demand between your underwritten value and the price you'll pay — sometimes 5%, sometimes 20% or more. The margin also lives in your *assumptions*, not only your price: an underwrite that already used market vacancy, a contingency on expenses, a real capex reserve, and an exit cap *above* the going-in cap has built conservatism into the value itself, so even buying *at* that conservative value carries an implicit cushion. The danger is the reverse — stacking optimistic assumptions *and* paying full price, which is buying with a negative margin of safety twice over.

There's a discipline question hiding here too. The hardest part of underwriting isn't the arithmetic — it's having a number *before* you fall in love with the building, and then honouring it. Brokers create urgency ("another buyer is circling") precisely to get you to commit before you've underwritten, or to abandon your number once you have. The professional answer is to walk. A deal you underwrote at ₫6.0 billion and lost to someone who paid ₫7.0 billion is not a deal you lost; it's a loss you avoided. The market produces new buildings every week; it does not refund overpayments. For the broader question of when buying makes sense at all versus renting, see [Rent vs Buy: The Real Math](/blog/trading/real-estate/rent-vs-buy-the-real-math). And Figure 7 collects the whole discipline into a checklist you can run on any deal.

![An underwriting checklist laid out as a grid of six items — rent roll, operating expenses, capex reserve, debt terms, exit assumption, and margin of safety — each with the specific thing to verify before trusting the price](/imgs/blogs/underwriting-a-property-deal-7.png)

## Common misconceptions

Underwriting is mostly a set of habits that correct the intuitions amateurs bring to property. Here are the costly ones, named and corrected.

**"A low price is a good deal."** Price is meaningless without income. A building at ₫4 billion that nets ₫120 million (a 3% cap) is *more expensive* than a building at ₫6 billion that nets ₫360 million (a 6% cap), because you're paying more *per đồng of income*. Underwriting prices the income stream, not the sticker. A "cheap" building with collapsing rents or hidden capex can be the most expensive purchase you ever make; an "expensive" building with secure, growing income can be the bargain. The price tag is the start of the question, never the answer.

**"Cash-on-cash is the return."** Cash-on-cash is a *year-one* snapshot of *one* slice of the return — the cash, before the sale, before time. A deal can have a thin or even negative cash-on-cash (Minh's 2.5%) and a fine IRR because the return lives in appreciation; another can have a fat cash-on-cash and a poor IRR because it never appreciates and you sell for what you paid. Treating cash-on-cash as "the return" makes you systematically overpay for high-current-yield deals (often the riskiest, in the worst locations) and ignore lower-yield deals where the real money is in the exit. Always pair it with an IRR.

**"The rent roll is the truth."** The **rent roll** — the seller's list of units, tenants, and rents — is a sales document, not an audited fact. Sellers quote *asking* rents as if they were *signed* rents, omit concessions ("first month free"), bury arrears, and show a building "fully occupied" by a cousin who'll move out at closing. The professional move is to verify: demand the actual signed leases, the bank statements showing rent actually received, and the lease end dates. Underwrite the rent you can *prove*, not the rent you're *told*. One inflated rent line, run through the cap-rate machinery, can overstate the building's value by tens of thousands.

**"The seller's pro-forma is the deal."** Every seller's pro-forma is built to *sell*, which means it understates expenses, understates vacancy, ignores capex, and assumes a rosy exit. It is a starting point for your own model, never a substitute for it. The discipline is to rebuild the pro-forma from scratch with *market* vacancy, *market* expenses (plus a contingency), a *real* capex reserve, and a *conservative* exit cap — and see if the deal still stands. If it only works on the seller's numbers, it doesn't work.

**"If I can afford the monthly payment, I can afford the building."** The monthly mortgage payment is what the *bank* needs from you; it is not what the *building* costs to own. The payment ignores vacancy (the months with no rent), capex (the roof), the rate reset (the teaser expiring), and the opportunity cost of your down payment. People who buy on affordability of the payment get blindsided by every line the payment leaves out. Underwriting is precisely the habit of pricing all those lines *before* you find out about them the hard way.

**"Appreciation will bail me out."** In a negative-leverage market like Vietnam in 2024, a deal that loses money on income every month is implicitly a bet that prices will rise fast enough to cover the bleed and then some. Sometimes they do — HCMC prices have risen sharply for years. But "appreciation will save it" is not an underwrite; it's a hope, and it's the exact bet that vaporized in 2008, in Japan after 1990, and in Vietnam's own 2022–23 freeze. An honest underwrite makes the income deal stand on its own and treats appreciation as upside, not as the plan.

## How it shows up in real markets

**Overpaying on an optimistic exit cap (US offices, 2020–2024).** In the cheap-money years up to 2021, buyers underwrote office and apartment deals assuming exit cap rates *equal to or below* their low going-in caps — sometimes 4% or less — because cap rates had compressed for a decade. When the US 10-year Treasury yield rose from under 1% in 2020 to above 4% by 2024, exit cap rates blew out by 150–200 basis points. Deals underwritten to sell at a 4.5% cap were suddenly worth what a 6%+ cap implied — a value cut of roughly a quarter on the same income, exactly the slope of Figure 5. Highly leveraged owners who'd assumed the exit cap would hold (or fall) found their equity wiped out even though their buildings' NOI had *risen*. The lesson the survivors learned: the exit cap is not a detail, it's the deal.

**The deal that only pencils at the top (Vietnam, 2021–2022).** At the peak of Vietnam's last cycle, investors bought đất nền (land plots) and pre-sale apartments on underwrites that *required* continued double-digit annual price gains to work — the income (often zero, for raw land) couldn't service the debt, so the entire return depended on flipping to the next buyer at a higher price. When the corporate-bond market froze in late 2022 after the Tân Hoàng Minh and Vạn Thịnh Phát arrests, and the State Bank tightened credit, the next buyer vanished. Deals that "penciled" only at the top of the market — only if prices kept rising — became unsellable and unserviceable at once. This is the signature failure of underwriting to a best-case exit: the whole structure stands only while the music plays. We tell that story in full in [the Vietnam 2022 freeze case study](/blog/trading/real-estate/case-study-vietnam-2022-bond-and-property-freeze).

**The teaser-rate reset (Vietnam, ongoing).** A specific, local, and brutal underwriting trap: a buyer underwrites a HCMC apartment at the 8% preferential teaser rate the bank offers for the first 18 months, computes a payment they can afford, and signs. Then the rate floats to 13–14%, the payment jumps by half, and the cash flow that was thin at 8% goes deeply negative at 14%. The deal was never affordable; it was affordable *for 18 months*. An underwrite that models the reset rate from day one — as a professional's would — either rejects the deal or sizes the loan small enough to survive the reset. The teaser is designed to make the buyer skip exactly the line that matters most. We unpack the master-variable role of rates in [Interest Rates and Mortgages](/blog/trading/real-estate/interest-rates-and-mortgages-the-master-variable).

**The 2008 subprime underwrite (United States).** The entire US subprime collapse was, at root, an underwriting failure at industrial scale. Loans were underwritten to a borrower's ability to pay the *teaser* rate, with the explicit assumption that ever-rising house prices would let them refinance before the reset — the exit assumption *was* "prices keep rising." When prices stopped rising in 2006–2007, the refinance exit closed, the resets hit, and underwriting that depended on appreciation cascaded into the worst financial crisis since the 1930s. The mechanism is identical to a single bad property deal, just multiplied by trillions: underwrite to an optimistic exit and a teaser rate, and the whole thing unwinds the moment reality declines to cooperate. We cover it in [the 2008 subprime case study](/blog/trading/real-estate/case-study-us-2008-subprime-crisis).

**Disciplined institutional underwriting (the survivors).** The other side of every blowup is the buyer who underwrote conservatively and was *ready* when prices fell. Funds that underwrote 2015–2019 acquisitions to exit caps *above* their going-in caps, stress-tested for higher rates, and bought with a margin of safety simply earned a bit less in the boom — and then had dry powder and unlevered staying power when distressed assets came to market in 2023–2024. Underwriting discipline is not glamorous; it costs you deals in the boom that the optimists win. Its entire payoff is that you are still standing, and solvent, when the optimists are not.

## When this matters / Further reading

Underwriting matters the moment you are about to commit real money to a specific building — which, for most people, is the single largest financial decision of their lives. You do not need to be an institutional investor for it to apply. If you are buying a flat to rent out, a duplex to live in one half of, a plot to develop, or even your own home as a long-term asset, the same discipline protects you: build the pro-forma, compute the return more than one way, model the rate reset and the real expenses, assume a conservative exit, and refuse to pay above your own honest value. The whole apparatus exists to do one thing — turn a story and a monthly payment into a number you can defend, and then buy with enough room that being wrong doesn't ruin you.

The deepest habit to carry away is the margin of safety. You will never underwrite the future correctly, because nobody can. What you *can* do is buy cheaply enough, and finance conservatively enough, that the gap between your price and the building's true worth absorbs your inevitable errors. Minh almost learned that the expensive way; the spreadsheet taught him the cheap way. A week of arithmetic saved him from a decade of feeding a building that quietly cost more than it earned.

To go deeper into the pieces that feed an underwrite:

- [Cap Rate, NOI, and the Income Approach](/blog/trading/real-estate/cap-rate-noi-and-the-income-approach) — the engine behind the value-equals-NOI-over-cap machinery, and why property is a bond made of bricks.
- [Leverage and the Mortgage: How Debt Amplifies Property](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property) — the full mechanics of positive, zero, and negative leverage.
- [How Property Is Valued: Three Approaches](/blog/trading/real-estate/how-property-is-valued-three-approaches) — how the income approach you've just used sits alongside the sales-comparison and cost approaches.
- [Risk Management in Real Estate](/blog/trading/real-estate/risk-management-in-real-estate) — turning the stress test into a standing discipline for managing the downside.
- [Rent vs Buy: The Real Math](/blog/trading/real-estate/rent-vs-buy-the-real-math) — the same underwriting logic applied to the most personal property decision of all.

This is educational material, not financial advice. Underwriting is a tool for thinking clearly about a deal's risks and price; it does not tell you whether to buy, and it cannot make an uncertain future certain. What it can do is make sure that when you commit, you commit to a number you understand — and that you've already looked, on paper, at exactly how the deal would break.

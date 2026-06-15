---
title: "Real Estate and REITs: Income, Leverage, and the Rate Cycle"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Real estate is an income asset bought with debt, so its price is set by one gear: the cap rate, which moves with interest rates. This is how REITs actually work, why they trade like leveraged stocks, and when in the cycle to own them."
tags: ["asset-allocation", "cross-asset", "real-estate", "reits", "cap-rate", "interest-rates", "leverage", "income-investing", "inflation-hedge"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Real estate is an *income asset* — you buy a building to collect its rent — and almost everyone buys it with debt. That makes its price exquisitely sensitive to one number: the *cap rate*, which moves with interest rates. When rates rise, cap rates rise, and the same rent is suddenly worth less.
>
> - The master equation is **value = net operating income ÷ cap rate**. A building earning \$1,000,000 of rent is worth \$20,000,000 at a 5% cap rate and only \$16,000,000 at a 6.25% cap rate — a 20% haircut from the *same rent*, purely because the discount rate moved.
> - Real estate is bought 50–70% with borrowed money, so a 20% fall in the building can be a 50% loss on your equity. That leverage math is why 2022 was brutal: listed REITs fell **−24.9%**, *more* than stocks (−18.1%).
> - Listed REITs are marketed as a diversifier, but they trade like leveraged equity: their correlation with stocks is about **+0.75**. Private real estate only *looks* smoother because it's appraised a few times a year, not priced every second.
> - The one number to remember: REITs are a bet on the *direction of rates*. Own them for income when rates are falling or stable and growth is steady — and respect that a rate shock is their kryptonite.

In late 2021, a REIT investor was living a charmed life. Interest rates were pinned near zero, the economy was reopening, and listed US real estate had just returned **+41.3%** for the year — beating the S&P 500's +28.7%. Apartments were full, warehouses couldn't be built fast enough, and the dividend checks kept arriving. Real estate felt like the perfect asset: a steady income stream that also happened to be soaring in price.

Then 2022 happened. The Federal Reserve, fighting the worst inflation in forty years, raised its policy rate from near zero toward 5% in the space of a year. The 10-year Treasury yield — a number we'll see is the master driver of real estate values — climbed from 0.93% at the end of 2020 to over 4% by October 2022. Nothing about the *buildings* changed. The apartments were still full; the warehouses still couldn't be built fast enough; the rent checks still arrived. And yet listed US REITs fell **−24.9%** that year — a bigger drop than the stock market's −18.1%. The income asset that felt so safe lost a quarter of its value because of something happening entirely in the bond market.

That whole whipsaw — up 41% on falling rates, down 25% on rising rates, with the actual rents barely moving — is real estate in one breath. It is an income asset whose *price* is governed not by the income but by the rate you discount that income at. The diagram below is the mental model we'll build the entire post around: a building's value is its rent divided by a single gear called the cap rate, and that gear is bolted directly to interest rates.

![Cap rate valuation gear showing value equals NOI divided by cap rate](/imgs/blogs/real-estate-reits-income-leverage-and-rates-1.png)

This post is the real-estate chapter of the cross-asset playbook. By the end you'll understand what a building actually earns, what a REIT is and why it pays such fat dividends, how the cap rate turns rents into a price, why leverage amplifies everything, why "real estate" is really a dozen different asset classes wearing one name, why listed REITs betray you as a diversifier exactly when you need one, and — the payoff — when in the cycle you actually want to own them. None of this is investment advice; it's the mechanism, the history, and the decision framework.

## Foundations: what real estate earns and what a REIT is

Let's build from zero. Real estate, as an *investment*, is not really about buildings. It's about the cash a building throws off. A building is just the machine; the rent is the product. So the first thing to understand is exactly what cash a property produces, what's left after costs, and how you can own a slice of that without becoming a landlord yourself.

### Net operating income: what a building actually earns

When you own a commercial property — an apartment complex, a warehouse, an office tower — tenants pay you rent. But you don't keep all of it. You have to pay property taxes, insurance, maintenance, the cost of the staff who run the building, and so on. The rent you collect *minus* the costs of operating the building is called the **net operating income**, or **NOI**. It is the single most important number in real estate.

$$\text{NOI} = \text{rental income} - \text{operating expenses}$$

Here the symbols mean: *rental income* is all the cash tenants pay you in a year; *operating expenses* are the recurring costs of running the building (taxes, insurance, repairs, management) — but, by convention, **not** the cost of the mortgage and **not** depreciation. NOI is the income the building produces *before* you've paid the bank and before any accounting deductions. It's the building's earnings, pure and operational.

#### Worked example: the NOI of one apartment building

You own a small apartment building with 20 units. Each rents for \$1,500 a month, so if every unit is occupied, your *gross potential rent* is 20 × \$1,500 × 12 = \$360,000 a year. But buildings are never 100% full — let's assume a 5% vacancy, so you actually collect about \$342,000.

Now subtract the costs of running it:

- Property taxes: \$40,000
- Insurance: \$12,000
- Maintenance and repairs: \$30,000
- Property management (a company that handles tenants): \$28,000
- Utilities and other: \$22,000

Total operating expenses: \$132,000. So your NOI is \$342,000 − \$132,000 = **\$210,000**. That \$210,000 is what the building earns each year, before any mortgage payment. Notice we did *not* subtract the loan — that's deliberate, and it's the key to everything that follows. NOI is the property's cash flow independent of how it's financed; whether you bought it with all cash or with a giant mortgage, the building still earns \$210,000.

The one-sentence intuition: NOI is a building's salary — what it brings home from doing its job — before the bank takes its cut.

### A REIT is a company that owns buildings and must pay out its income

Most people can't buy a \$50 million office tower. But you can buy a *share* of a company that owns a hundred of them. That company is a **REIT** — a *real estate investment trust*. A REIT is a business whose entire job is to own (and sometimes operate) income-producing real estate: shopping centers, apartments, warehouses, data centers, cell towers, hospitals. You buy its shares on a stock exchange just like you'd buy shares of Apple, and the share price goes up and down all day.

What makes a REIT special — and the reason it exists as a legal structure — is a bargain it strikes with the tax authorities. To qualify as a REIT in the US, the company must pay out **at least 90% of its taxable income to shareholders as dividends**. In exchange, the REIT itself pays **no corporate income tax** on the income it distributes. Compare that to a normal company like a manufacturer: it pays corporate tax on its profits, *then* you pay tax again on the dividends — that's the famous "double taxation". A REIT skips the first layer. The income passes through the company almost untouched and lands in your account.

This pass-through deal has one huge consequence for you as an investor: **REITs pay high dividends.** Because they're legally required to distribute almost all their income, REIT dividend yields are typically 3–5%, well above the ~1.5% yield of the broad stock market. A REIT is, first and foremost, an income vehicle. People buy stocks like Amazon for *growth* — the company reinvests its profits to get bigger. People buy REITs for *income* — the company hands you the rent.

#### Worked example: where your REIT dividend comes from

Suppose a REIT owns 50 buildings that, combined, produce \$500 million of NOI a year. After paying interest on its debt and its corporate overhead, say it has \$300 million of distributable income left. By law it must pay out at least 90% of taxable income, so it distributes roughly \$270 million to shareholders.

If the REIT has 100 million shares outstanding, that's \$2.70 per share in dividends. If the shares trade at \$60 each, your *dividend yield* is \$2.70 ÷ \$60 = **4.5%**. Buy 1,000 shares for \$60,000 and you collect about \$2,700 a year in dividends — your share of the rent from 50 buildings you'll never visit.

The one-sentence intuition: a REIT is a way to be a landlord without the toilets — you collect a pro-rata slice of the rent, and the 90% payout rule guarantees most of it actually reaches you.

### FFO: the earnings number that actually matters for a REIT

Here's a quirk that trips up every newcomer who tries to value a REIT with normal stock tools. For an ordinary company, *net income* (profit after all expenses) is the headline earnings number, and you value the stock off it with a price-to-earnings ratio. For a REIT, net income is almost useless — and the reason is *depreciation*.

Accounting rules force a company to treat a building as a wasting asset: every year, it must deduct a slice of the building's value as *depreciation*, an expense that reflects the idea that the building is wearing out. For a factory machine, that's reasonable. For a well-maintained office tower or apartment complex, it's mostly fiction — good real estate usually *appreciates* over time, not depreciates. Yet the accounting rules make the REIT book a huge depreciation expense every year, which crushes its reported net income far below the actual cash the buildings generate.

So the real-estate industry uses a different number: **funds from operations**, or **FFO**. The definition is simple — take net income and *add back* the depreciation (and a couple of other non-cash adjustments):

$$\text{FFO} = \text{net income} + \text{depreciation} + \text{amortization} - \text{gains on property sales}$$

The intuition: FFO strips out the fake depreciation expense to reveal the genuine recurring cash the portfolio throws off. It's the REIT equivalent of cash earnings. When you see a REIT trading at "15× FFO," that's the real-estate version of a price-to-earnings ratio, and it's the right tool. (A refinement called *adjusted FFO*, or AFFO, also subtracts the recurring capital spending needed to keep buildings competitive — leasing commissions, tenant improvements — and is an even better proxy for the cash actually available to pay dividends.)

#### Worked example: why net income lies about a REIT

A REIT collects \$210,000,000 of NOI across its portfolio. After \$60,000,000 of interest on its debt and \$10,000,000 of corporate overhead, it has \$140,000,000 of pre-depreciation cash income. Now the accountants insist on a \$120,000,000 depreciation charge on the buildings. Reported *net income* is therefore \$140,000,000 − \$120,000,000 = **\$20,000,000**.

A naive investor looks at that \$20 million net income, sees a REIT paying \$130,000,000 in dividends, and panics — the company is "paying out six times its earnings," surely a dividend on the brink of collapse. But add the \$120,000,000 of depreciation back: FFO is \$20,000,000 + \$120,000,000 = **\$140,000,000**. The \$130,000,000 dividend is comfortably covered by \$140,000,000 of FFO — a healthy ~93% payout of *cash* earnings. The buildings aren't actually wearing out; the depreciation was an accounting ghost. The one-sentence intuition: never judge a REIT by net income or by a P/E ratio — use FFO, or the depreciation ghost will make a perfectly healthy company look like it's bleeding to death.

### Three flavors: public REITs, private real estate, and mortgage REITs

"Real estate" the asset class comes in several wrappers, and the differences matter enormously for how it behaves:

- **Public (listed) equity REITs** — shares of building-owning companies that trade on a stock exchange. *Equity* here means the REIT owns the buildings (the equity in them), not the loans. This is what most of this post is about, because it's how ordinary investors get exposure. The key trait: you can buy or sell in one second, and the price moves every second — which, we'll see, makes it behave like a stock.

- **Private / direct real estate** — actual buildings owned directly, or shares in a *private* fund that isn't listed on any exchange. This is what pension funds and the very wealthy hold. The buildings are the same; the wrapper is different. The crucial difference: there's no live market price. The value is *appraised* (estimated by a professional) only a few times a year. That single fact — appraised, not traded — creates the "smoothing" illusion we'll dismantle later.

- **Mortgage REITs (mREITs)** — these don't own buildings at all. They own the *loans* against buildings. A mortgage REIT borrows short-term money cheaply and lends it long-term against real estate, pocketing the spread. They're really leveraged bond funds in a real-estate costume, and they behave very differently (and usually worse in a crisis) than equity REITs. When this post says "REIT" without qualification, it means an *equity* REIT.

Keep the equity-REIT-vs-private distinction especially firmly in mind. They own the *same kind of buildings* and earn the *same kind of rent*, but one is marked to market every second and the other is appraised twice a year. That difference is responsible for one of the biggest myths in all of investing — that private real estate is "less risky" than listed real estate. It isn't. It just looks calmer because nobody's quoting you a price.

## The valuation gear: the cap rate

Here is the heart of the whole post. Everything about how real estate behaves — its rate sensitivity, its booms and busts, the 2022 crash — flows from one little equation and one little number: the *capitalization rate*, universally shortened to **cap rate**.

### What a cap rate is

When you buy a building, you're buying its stream of future NOI. How much should you pay for, say, \$1 of annual NOI? The answer is set by the *cap rate*, which is just the building's NOI expressed as a percentage of its price:

$$\text{cap rate} = \frac{\text{NOI}}{\text{property value}} \quad\Longleftrightarrow\quad \text{property value} = \frac{\text{NOI}}{\text{cap rate}}$$

The two forms say the same thing. The first defines the cap rate as the yield you earn on the price you pay (like a dividend yield, but for a building). The second — the one that matters — says **a building's value is its NOI divided by the cap rate.** Flip the cap rate and you get the price.

A cap rate is just a *yield*. A 5% cap rate means you pay \$20 for every \$1 of annual NOI (because \$1 ÷ 0.05 = \$20); the building yields you 5% of its price each year before financing. A 7% cap rate means you pay only ~\$14.30 for that same \$1 of NOI — the building is "cheaper" relative to its income. **Low cap rate = expensive building; high cap rate = cheap building.** This trips people up at first because it's backwards from a price: a *lower* cap rate means a *higher* price.

### Why the cap rate is glued to interest rates

So what determines the cap rate? Why is it 5% for one building and 8% for another, or 5% in 2021 and 6.5% in 2023? You can decompose the cap rate into three pieces:

$$\text{cap rate} \approx \underbrace{r_f}_{\text{risk-free rate}} + \underbrace{\text{property risk premium}}_{\text{extra yield for the risk}} - \underbrace{g}_{\text{expected NOI growth}}$$

where: $r_f$ is the *risk-free rate* — what you can earn with no risk, anchored by the 10-year Treasury yield we covered in [government bonds and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration); the *property risk premium* is the extra yield investors demand for taking on real-estate risk (vacancy, illiquidity, the building burning down); and $g$ is the expected growth rate of the building's NOI (rents tend to rise over time, and growth makes a building worth more, which *lowers* the cap rate you'll accept).

This decomposition is the whole ballgame, so let's read it carefully. The first term is the risk-free rate. **When interest rates rise, the risk-free rate rises, and — holding the risk premium and growth fixed — the cap rate rises right along with it.** And a higher cap rate, by our master equation, means a *lower* property value for the same NOI. That is the mechanism. That is why real estate is a rate-sensitive asset. It is not a vague "rising rates are bad for everything" hand-wave; it is an arithmetic link running straight from the bond market into the price of a building.

The intuition is one you already have. If a totally safe Treasury pays 1%, then a building yielding 5% (a 5% cap rate) looks fantastic — you're getting 4 extra percentage points for taking real-estate risk. But if the Treasury now pays 4.5%, that same building yielding 5% barely pays you anything extra for all that risk and hassle. Nobody will pay the old price. Buyers demand a higher yield — a higher cap rate — which means they'll only pay a lower price. The building's value falls until its NOI, divided by the new higher cap rate, gives the new lower price. The rent never changed; the *required yield* changed; the price had to follow.

![Cap rates track the 10-year Treasury yield over time](/imgs/blogs/real-estate-reits-income-leverage-and-rates-3.png)

The chart above shows the linkage in real numbers. As the 10-year Treasury yield climbed from 0.93% at the end of 2020 to nearly 5% in late 2023 — a move of about 365 basis points (a *basis point* is one hundredth of a percent, so 0.01%; 365 bps is 3.65%) — average commercial cap rates climbed too, from around 5% to roughly 6.5%. The spread between them stayed roughly stable, because the risk premium and growth expectations didn't change much. What changed was the floor the whole thing sits on: the risk-free rate. And that move in cap rates, by itself, knocked roughly a fifth off the value of a lot of commercial real estate.

The same decomposition also explains why different buildings carry *different* cap rates at the same moment in time — it's the risk-premium and growth terms doing the work. A brand-new logistics warehouse leased to a blue-chip tenant for 15 years in a booming submarket might trade at a 4.5% cap rate: low risk premium (the tenant is reliable, the building is modern) and high expected NOI growth (rents are rising). A tired suburban office building with leases expiring soon and an uncertain future might trade at a 9% cap rate: high risk premium (will the tenants renew?) and *negative* expected growth (rents are falling). Same risk-free rate underneath both; the gap between a 4.5% and a 9% cap rate is the market pricing the building's specific quality, location, tenant credit, and lease structure. A lower cap rate isn't a bargain — it's the market saying "this is a safer, faster-growing income stream, and I'll pay up for it." When you hear a real-estate investor say a deal "trades at a 6 cap," they've just told you, in two words, the market's entire risk-and-growth verdict on that building.

And the growth term, $g$, deserves one more beat, because it's the lever that can rescue real estate from rising rates. If a building's rents are growing at, say, 4% a year — because it's in a supply-constrained market with short leases — that growth *subtracts* from the cap rate, holding the price up even as the risk-free rate climbs. This is precisely why the structural-winner sectors (industrial, data centers, apartments) survived the 2022 rate shock far better than the no-growth sectors (office): rising NOI partly cancelled the rising risk-free rate inside the cap-rate formula. The rate cycle sets the headwind for *all* real estate; growth determines which buildings can lean into it and which get blown over.

#### Worked example: the cap rate turns a rate move into a price move

You're looking at an office building that produces \$1,000,000 of NOI a year — a nice, round, friendly number. In 2021, with rates near zero, the market cap rate for buildings like this is 5%. So the building is worth:

$$\text{value} = \frac{\$1{,}000{,}000}{0.05} = \$20{,}000{,}000$$

You pay \$20 million. Now roll forward to 2023. The 10-year Treasury yield has risen by about 1.25 percentage points, and cap rates for this kind of building have moved from 5% to 6.25%. The building still earns exactly \$1,000,000 of NOI — same tenants, same leases, same rent. But it's now worth:

$$\text{value} = \frac{\$1{,}000{,}000}{0.0625} = \$16{,}000{,}000$$

The building lost \$4,000,000 — a **−20% drop** — without a single tenant leaving or a single dollar of rent declining. The entire loss came from the discount rate. This is the single most important calculation in real estate: a 1.25-point rise in the cap rate, driven by rising interest rates, cut the value by a fifth.

The one-sentence intuition: in real estate, *you can lose 20% of your money while your tenants are paying you exactly what they always did* — because the price is the rent divided by a rate, and the rate moved.

### The duration analogy: real estate is a very long bond

If you've read the [government bonds](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) chapter, this should feel familiar — and it should, because it's the same mathematics. A bond is a stream of fixed cash flows whose price falls when rates rise. A building is a stream of (roughly) fixed cash flows whose price falls when rates rise. The cap rate plays the role of the bond's yield.

How sensitive? A building's value behaves like a bond with very long *duration* — duration being the measure of how much a price moves for a 1% change in rates. Our worked example showed a roughly 20% value loss for a ~1.25-point rate move, which is a sensitivity of about 16% per percentage point. That's the duration of a 20-to-30-year bond. **Real estate, valued off a cap rate, has the rate sensitivity of a long-dated bond — with the added wrinkle that, unlike a bond, it's also bought with leverage.** Hold that thought; leverage is next, and it's where the real damage lives.

## Leverage: how debt amplifies everything

Almost nobody buys real estate with cash. The defining feature of property as an asset is that it's bought with *debt* — typically the buyer puts up 30–50% of the price and borrows the rest with a mortgage. This is so normal it has a name: the **loan-to-value ratio**, or **LTV**. A 60% LTV means you borrowed 60% of the purchase price and put up 40% of your own money (the *equity*).

Leverage is wonderful when prices rise and a wrecking ball when they fall, because the debt is *fixed*. The bank lent you a fixed number of dollars and wants exactly that many back, regardless of what happens to the building. So every dollar the building gains or loses lands entirely on your equity slice — the thin layer between the building's value and the bank's claim.

### Why a 20% property fall is a 50% equity loss

Let's take the building from before and add a realistic mortgage. This is the most important arithmetic in the post after the cap-rate equation itself.

![Leverage turns a 20 percent property fall into a 50 percent equity loss](/imgs/blogs/real-estate-reits-income-leverage-and-rates-4.png)

The figure tells the story: the building is worth \$20 million, financed with \$12 million of debt (60% LTV) and \$8 million of your equity. When the cap rate rises and the building falls to \$16 million, the debt doesn't move — the bank is still owed exactly \$12 million. So the entire \$4 million loss comes out of your equity, which drops from \$8 million to \$4 million. A 20% fall in the *building* became a 50% fall in *your money*.

#### Worked example: the leverage amplifier in full

You buy the \$20,000,000 building with 60% leverage:

- Mortgage debt: 60% × \$20,000,000 = \$12,000,000
- Your equity: \$20,000,000 − \$12,000,000 = **\$8,000,000**

Now rates rise, the cap rate moves from 5% to 6.25%, and the building falls 20% to \$16,000,000 (the same calculation as before). The debt is unchanged. Your equity is now:

$$\text{equity} = \$16{,}000{,}000 - \$12{,}000{,}000 = \$4{,}000{,}000$$

You've gone from \$8 million to \$4 million — a **−50% loss on your equity** from a −20% fall in the building. The leverage ratio amplified the loss by 2.5×. And this works in both directions: if the building had *risen* 20% to \$24 million, your equity would have jumped to \$12 million — a +50% gain. That asymmetric amplification — big wins in good times, ruinous losses in bad times — is exactly why real estate fortunes are made *and lost* so spectacularly, and why 2022 hurt levered owners far more than the headline property declines suggested.

The one-sentence intuition: leverage doesn't change the building, it changes *who absorbs the building's swings* — and with the debt fixed, the equity holder absorbs all of them, magnified.

### Refinancing risk: the bomb on a timer

There's a second, sneakier way rates hurt levered real estate: **refinancing risk.** Most commercial mortgages aren't 30-year fixed loans like a house. They're shorter — often 5, 7, or 10 years — with a *balloon* at the end, meaning the whole principal comes due and must be paid off, usually by taking out a *new* loan. That's called *refinancing*.

Here's the trap. Suppose you borrowed \$12 million at 3.5% in 2019 on a 5-year loan. In 2024, that loan comes due. You don't have \$12 million in cash, so you need a new loan. But rates have risen — the new loan costs 7%. Worse, the building's value has fallen (higher cap rate), so the bank, applying its 60% LTV limit to a *lower* value, will only lend you, say, \$9.6 million. You're now \$2.4 million short and facing an interest bill that's doubled. You either inject fresh cash, sell the building into a weak market, or hand the keys to the bank.

This is the "maturity wall" or "refinancing wall" you hear about in real-estate downturns. It's why a rate shock doesn't just cut values on paper — it forces *actual* distressed sales as owners hit refinancing deadlines they can't meet. The 2023–2024 office market was a slow-motion version of exactly this. We'll return to it in the case studies.

#### Worked example: the refinancing squeeze

You own a building worth \$20 million in 2019, financed with a \$12 million loan at 3.5%, due in 2024. Your annual interest is 3.5% × \$12,000,000 = \$420,000, comfortably covered by your \$1,000,000 NOI.

In 2024 the loan matures. The situation has changed:

- The building's value has fallen to \$16,000,000 (higher cap rate).
- New loans are priced at 7%.
- The bank will lend at most 60% LTV against the *new* value: 60% × \$16,000,000 = \$9,600,000.

So you must repay \$12,000,000 but can only borrow \$9,600,000 — a \$2,400,000 cash gap you have to fill out of pocket *or* lose the building. And even on the smaller new loan, your interest jumps to 7% × \$9,600,000 = \$672,000, eating two-thirds of your \$1,000,000 NOI versus the old \$420,000. The deal that was comfortably profitable is now barely breaking even, and you had to write a \$2.4 million check just to keep it.

The one-sentence intuition: in levered real estate, you don't only lose when the value falls — you can be forced to *sell at the bottom* simply because a loan came due at the wrong moment.

## Sectors matter: "real estate" is a dozen asset classes

One of the biggest mistakes a beginner makes is treating "real estate" as a single thing. It isn't. The economics of a warehouse and the economics of a downtown office tower have almost nothing in common, even though both are "commercial real estate." The dispersion between property *sectors* is enormous — often bigger than the dispersion between entire asset classes — and it has only grown since the COVID pandemic rewired how people work, shop, and live.

![REIT sector dispersion showing industrial and data centers up and office down](/imgs/blogs/real-estate-reits-income-leverage-and-rates-5.png)

The chart shows the spread, and it's stark. Over the 2020–2024 window, the structural winners and losers pulled apart by something like a hundred percentage points. Let's walk the major sectors:

- **Industrial / warehouses** — the e-commerce winner. Every online order needs a warehouse near a city to ship from, and the explosion of online shopping created insatiable demand for "logistics" space. Rents rose for years; vacancy stayed near record lows. A boring metal box near a highway turned out to be one of the best real-estate investments of the decade.

- **Data centers** — the digital-infrastructure winner, and increasingly the AI winner. These are the windowless buildings full of servers that run the internet and, lately, train AI models. Demand for computing power has exploded, and data-center REITs have ridden it. The constraint is now electricity, not space — a fascinating wrinkle that makes the best data-center sites genuinely scarce.

- **Apartments (residential)** — the steady-demand sector. People always need somewhere to live, and a chronic US housing shortage keeps apartments well-occupied. Apartments aren't a moonshot, but they're resilient: short leases (typically one year) mean rents can reset upward with inflation quickly, which makes apartments one of the better *inflation hedges* within real estate.

- **Retail** — the bifurcated sector. The "retail apocalypse" narrative crushed weak malls as shoppers moved online, but well-located shopping centers anchored by groceries and services held up fine. Within "retail," a class-A mall and a dying regional mall are practically different asset classes.

- **Office** — the post-COVID loser. Remote and hybrid work permanently cut how much office space companies need. Vacancy in many cities hit record highs, leases didn't renew, and — fatally — office buildings carry heavy leverage that came due into a market with rising rates and falling values. Office is the sector that turned the 2022–2024 rate shock from painful into catastrophic, and it's where most of the genuine real-estate distress of that period lives.

#### Worked example: same rate shock, opposite outcomes

Two REITs each own buildings producing \$1,000,000 of NOI, both bought at a 5% cap rate for \$20 million, both with 60% leverage. Then the rate shock hits and the cap rate rises to 6.25% for both. But the *sectors* diverge:

- **The warehouse REIT** signs new leases at 15% higher rents as old ones expire (logistics demand is booming). Its NOI rises to \$1,150,000. New value = \$1,150,000 ÷ 0.0625 = \$18,400,000 — down only 8% despite the rate shock, because growing rents partly offset the higher cap rate.
- **The office REIT** loses a major tenant who went remote; occupancy drops and NOI falls to \$800,000. New value = \$800,000 ÷ 0.0625 = \$12,800,000 — down 36%. On 60% leverage, the warehouse owner's equity fell ~25% while the office owner's equity was *wiped out* (the \$12,800,000 value barely covers the \$12,000,000 debt).

The one-sentence intuition: the cap rate hits every sector the same way, but rent *growth* is what separates a survivable rate shock from a catastrophic one — which is why sector selection inside real estate matters as much as the rate call itself.

## How real estate behaves: the appraisal illusion

Now we can answer the question that matters for a portfolio: how does real estate actually *move*? And the honest answer is uncomfortable, because it depends entirely on which wrapper you hold — listed or private — even though the underlying buildings are identical.

### Listed REITs: equity-like volatility, in real time

A listed REIT is a stock. It trades on an exchange, its price updates every second, and that price reflects the market's real-time, leveraged view of those buildings' value. So listed REITs are *volatile* — they have the price swings of equities, often a bit more because of the embedded leverage. Look back at the year-by-year returns: +41.3% in 2021, −24.9% in 2022. Those are stock-sized — actually, larger-than-stock-sized — moves. Listed real estate is not a placid income stream that sits quietly collecting rent; it's a leveraged, rate-sensitive equity that happens to pay a fat dividend.

![REIT total returns swinging with the rate cycle from 2014 to 2024](/imgs/blogs/real-estate-reits-income-leverage-and-rates-2.png)

The bars above make the volatility visceral. In a single decade, listed US REITs ranged from +41.3% to −24.9%. The two extremes — the 2021 boom and the 2022 bust — are almost mirror images, and they line up exactly with the rate cycle: 2021 had collapsing rates and reopening optimism; 2022 had the fastest rate-hiking cycle in forty years. The buildings inside the index barely changed across those two years. The *price* changed because the *rate* changed. If you take one picture away from this post, make it this one: REIT returns are a rate-cycle chart wearing a real-estate costume.

### Private real estate: smooth because it's appraised, not marked

Here's where it gets subtle, and where most of the marketing happens. Private real estate funds report *much* smoother returns than listed REITs. In a year when listed REITs fall 25%, a private real-estate fund holding similar buildings might report a small loss or even a small gain. Brochures point to this and say: see, private real estate is *less risky*, *less correlated*, a *true diversifier*. It is one of the most expensive misunderstandings in investing.

The smoothness is an *artifact of how private real estate is valued*. A listed REIT is priced by the market every second. A private building is *appraised* — a professional estimates its value — only quarterly or even annually, and appraisers are conservative and backward-looking. When the market moves, appraisals lag, often by quarters. So the *reported* value of private real estate moves in slow, gentle steps while the *actual* economic value (the price a building would fetch today) jumps around just like the listed market.

This is called **appraisal smoothing**, and it has a clean tell: when listed REITs (priced instantly) and private real estate (priced lazily) diverge sharply, the listed market is almost always *right* and the private market is just *late*. In 2022, listed REITs fell 25% in months; private real-estate funds reported modest losses, then kept "discovering" the decline through 2023 and 2024 as appraisals slowly caught down and as forced sales finally printed real transaction prices. The risk was identical — same buildings, same rate shock, same leverage. One wrapper showed it immediately; the other hid it in stale appraisals.

The investing consequence is profound. Private real estate's low reported volatility and low reported correlation are *measurement artifacts*, not genuine diversification. If you "un-smooth" the appraisals — statistically reconstruct what the values were really doing between appraisal dates — private real estate's volatility and its correlation with stocks both jump to roughly the listed REIT level. **You do not get a free lunch by holding the same buildings in a wrapper that prices them less often. You just get a smoother-looking statement and a nasty surprise when the appraisals finally catch up.**

#### Worked example: the smoothing illusion in numbers

Imagine the *true* economic value of a portfolio of buildings falls 20% over a year, in roughly equal monthly steps. A listed REIT holding those buildings would report something close to that −20%, with the price visibly sliding each month. Its measured volatility would be high — the monthly returns would swing widely.

A private fund appraises the same buildings once a year. At appraisal time it might mark them down only 8% (appraisers are conservative and lean on the prior, lagging comparable sales), reporting a gentle −8% for the year while the rest of the −20% sits invisible, to be "discovered" in next year's appraisal. Its measured monthly volatility is near zero — because between appraisals, the reported value is a flat line.

Same buildings. Same true −20% economics. One reports −20% with high volatility; the other reports −8% with near-zero volatility — *this year*. The −12% gap doesn't vanish; it's deferred. The one-sentence intuition: low private-real-estate volatility is a statement about appraisal frequency, not about risk — the risk is identical, just reported on a delay.

## How real estate correlates: the diversification that isn't

We've now seen what drives real-estate prices (the cap rate, hence rates) and how it behaves (equity-like for listed, deceptively smooth for private). The last piece — and the most important one for a portfolio — is how it moves *relative to everything else you own*. Because the entire pitch for real estate is diversification: "stocks and bonds and real estate, three different things." Let's test that claim against the data.

![Listed REITs track stocks not bonds across the cycle](/imgs/blogs/real-estate-reits-income-leverage-and-rates-6.png)

The grouped bars compare listed REITs, stocks, and bonds year by year, and the visual verdict is unambiguous: **REITs move with stocks, not against them.** When stocks have a great year (2019, 2021, 2023), REITs do too. When stocks have a terrible year (2018, 2022), REITs do too — and in 2022, REITs fell *harder*. The monthly correlation of listed REITs with stocks over 2015–2024 is about **+0.75** — the same high correlation we found for high-yield credit in the [corporate credit](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) chapter. A correlation of +0.75 means they mostly move together; the diversification benefit is real but modest, and it *evaporates in a crash*, which is precisely when you wanted it.

Here's the full correlation profile of listed REITs, from monthly data over 2015–2024:

| REITs vs… | Correlation | What it means |
|---|---|---|
| US stocks | +0.75 | Move together most of the time; REITs are basically a stock sector |
| High-yield credit | +0.70 | Both are leveraged, risk-on, "reach for yield" assets |
| US bonds | +0.45 | Meaningfully rate-sensitive — the long-duration link from the cap rate |
| Cash | ~0.00 | Cash is the one true diversifier; it doesn't move with anything |

Read that table as a diagnosis. The +0.75 with stocks says listed real estate is, for portfolio purposes, *another flavor of equity*. The +0.45 with bonds says it *also* carries serious interest-rate risk — it's the rare asset that's simultaneously equity-like (it's a leveraged claim on the economy) and bond-like (it's discounted off a rate). That dual exposure is exactly why 2022 was so painful: it was a year when both stocks *and* bonds fell, and REITs, sitting at the intersection, got hit from both directions.

### The partial inflation hedge: real, but conditional

There's one genuinely good diversifying property of real estate, and it's worth stating precisely because it's usually overstated. Real estate is a **partial inflation hedge.** The logic is sound: real estate is a *real asset* — a physical building producing real cash flow — and rents tend to rise with the general price level over time. If inflation pushes up the cost of everything, it also tends to push up rents (especially in sectors with short leases that reset frequently, like apartments and hotels) and the replacement cost of building new supply. Over the long run, real estate has roughly kept pace with inflation, which is more than cash or long bonds can say.

But "over the long run" and "roughly" are doing heavy lifting. The inflation hedge is *partial* and *slow* for two reasons. First, leases lock in rents — a building with 10-year fixed leases can't raise rents when inflation spikes; it has to wait for renewals. So the hedge works much better for short-lease sectors (apartments, hotels, self-storage) than long-lease ones (net-lease retail, some office). Second, and more importantly, inflation usually comes with *rising interest rates*, and rising rates raise the cap rate, which *cuts* values — the very mechanism we've spent the whole post on. So in a real inflation shock like 2022, the cap-rate hit (from rising rates) overwhelmed the rent-growth benefit (from inflation), and real estate fell. The inflation hedge is a *long-run, gradual* property that gets temporarily buried by the *short-run, violent* rate sensitivity. Connect this to the [real vs nominal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) framework: real estate hedges inflation over years, but it's hostage to *real rates* over months.

The one-sentence intuition: real estate protects your purchasing power across a decade of inflation, but it can absolutely crater in a single year of *rising rates* — the slow hedge and the fast risk live in the same asset.

## Common misconceptions

**"Real estate is a safe, stable income asset."** Half true and dangerously incomplete. The *income* (the rent) is relatively stable. The *price* is not — listed REITs swing like volatile stocks (+41% then −25% in two consecutive years). You can be collecting your dividend on schedule while your principal drops 25%. The stability is in the cash flow, not the capital value; conflating the two is how people get blindsided.

**"REITs are a great diversifier — they're a different asset class from stocks."** Listed REITs have a +0.75 correlation with stocks. They are, statistically, an equity sector with extra leverage and extra rate sensitivity. They diversify a stock portfolio about as much as adding a new stock sector does — modestly in normal times, and barely at all in a crash, when correlations rush toward one. Cash and, in growth-driven recessions, Treasuries are real diversifiers; listed REITs are not.

**"Private real estate is less risky than listed REITs."** This is the appraisal illusion. Private and listed real estate own the same buildings with the same leverage. Private just gets appraised a few times a year instead of priced every second, so its reported returns look smooth. Un-smooth the appraisals and the risk is the same. You're not buying less risk; you're buying a less frequent — and lagging — price quote.

**"Real estate is a reliable inflation hedge."** Over the long run, yes, partially. In a typical inflation shock, no — because inflation brings rising rates, rising rates raise cap rates, and the cap-rate hit usually swamps the rent-growth benefit in the short term. In 2022, the worst inflation in forty years, REITs *fell* 25%. A reliable hedge doesn't do that. Call it a slow, partial, real-rate-dependent hedge and you'll be calibrated correctly.

**"A REIT's high dividend means high total return."** Not necessarily. A REIT's total return is dividend *plus* price change. The dividend can be a fat 5% while the price falls 25% (2022) for a deeply negative total return. The high yield is a feature of the 90%-payout legal structure, not a promise of good performance. Worse, a high yield sometimes signals the *market* expects the dividend to be cut — a yield that looks too good can be a warning, not a gift.

## How it shows up in real markets

**The 2022 rate shock — the defining lesson.** This is the case study every REIT investor should have tattooed on their arm. Entering 2022, listed REITs were riding high on near-zero rates. Then the Fed hiked at the fastest pace since the 1980s, the 10-year Treasury yield blew out from ~1.5% to over 4%, and cap rates followed. The result: listed REITs returned **−24.9%** for the year, *worse* than the S&P 500's −18.1% and far worse than what the buildings' fundamentals (still-rising rents in most sectors) would suggest. Nothing was wrong with the rent. Everything was wrong with the discount rate. And the leverage — REITs typically carry meaningful debt — turned the value decline into an even bigger equity decline. The single cleanest demonstration in modern markets that real estate is, at heart, a leveraged bet on the level of interest rates.

**The 2021 reopening melt-up.** The mirror image. In 2021, with rates pinned near zero and the economy reopening from lockdowns, listed REITs returned **+41.3%**, beating the S&P 500. Apartments and industrial led; the low discount rate inflated every cap rate to historic lows (meaning historically high prices). It was a vivid lesson in the upside of the same mechanism: when rates fall, cap rates fall, and the same rent becomes worth dramatically more. Investors who mistook this for a permanent property boom — rather than a temporary gift from the rate environment — were set up for the brutal reversal in 2022.

**The office sector's slow-motion collapse, 2022–2024.** The most painful real-world story, because it combined every risk in this post. Remote work permanently cut office demand (a *fundamental* hit to NOI), rising rates raised cap rates (a *valuation* hit), and the heavy leverage on office buildings meant owners faced refinancing walls — loans maturing into a market with lower values and higher rates. The result was genuine distress: owners handing keys back to lenders, office towers in major cities selling for fractions of their prior value, and some buildings appraised at less than their outstanding debt. It was the refinancing-risk and sector-dispersion sections playing out simultaneously, in public, over two years.

**The 2007–2009 financial crisis.** The original real-estate-driven catastrophe, and a different mechanism worth contrasting. The 2008 crisis began in *residential* mortgages (subprime loans), and listed equity REITs fell about **−37%** as the entire financial system seized, credit froze, and leveraged real estate owners couldn't refinance. This wasn't primarily a cap-rate-from-rising-rates story (rates actually *fell* as the Fed cut); it was a *credit and leverage* story — the funding for real estate disappeared, forcing fire sales. The lesson: real estate has two big risk channels, rising rates (2022) and freezing credit (2008), and a severe enough event can trigger both. In both crises, the "diversifier" failed exactly when needed: REITs fell alongside stocks.

**The 2020 COVID divergence.** A useful nuance. In the 2020 calendar year, listed REITs returned **−5.1%** while the S&P 500 gained +18.4% — a rare year of meaningful underperformance. But the *average* masked enormous sector dispersion: industrial and data-center REITs soared on the e-commerce and work-from-home boom, while hotel, retail, and office REITs collapsed on lockdowns. The headline "real estate" number was a blend of a boom and a bust happening simultaneously inside the same asset class — the clearest argument there is for why "real estate" as a single allocation is a fiction, and sector selection is everything.

## When to own it: the real-estate allocation playbook

Here's the payoff. After all the mechanism, when do you actually want REITs in a portfolio, how much, and what tells you you're wrong? Real estate earns its place as an *income asset with a partial inflation hedge* — but only if you respect that its master risk is interest rates. (This is educational framing, not personalized advice.)

![Real estate allocation playbook by rate and growth regime](/imgs/blogs/real-estate-reits-income-leverage-and-rates-7.png)

The matrix above maps the regimes. Read it as a decision grid: the rows are what rates and growth are doing, and the cells tell you whether to own REITs, which sectors, and what would invalidate the call. Let's walk the logic.

**The best regime: rates falling or stable, growth steady.** This is REIT heaven. Falling rates pull cap rates down, lifting values directly (2021 in miniature); steady growth keeps rents and occupancy healthy; and you collect a 3–5% dividend the whole time. When the rate cycle is turning down and the economy is fine, listed real estate is one of the most attractive income assets available — you get the dividend *and* the cap-rate tailwind. This is when to lean in.

**The income regime: rates stable, growth steady.** Even without a rate tailwind, REITs are a legitimate *income* holding here. You're being paid a ~4% dividend yield to own a real, inflation-linked cash flow stream. The thing to check is the *spread*: is the cap rate (or the dividend yield) high enough above the risk-free Treasury yield to compensate for the risk and leverage? When that cushion is thin — when REITs yield barely more than safe bonds — the asset is expensive and the margin of safety is gone. Connect this to [interest rates as the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable): the level of rates sets the bar every other asset, including real estate, must clear.

**The danger regime: rates rising fast.** This is the one to respect. Rising rates raise cap rates and crush values, and leverage amplifies the damage — the 2022 lesson. When the rate cycle is turning *up* sharply, levered real estate is exactly where you don't want to be. If you hold REITs into a rate-hiking shock, expect equity-like-or-worse drawdowns. The defensive move within real estate is to favor low-leverage REITs and short-lease sectors (apartments, storage) whose rents can reset upward to partly offset the cap-rate hit. The thing that would invalidate even that caution: rents repricing up *fast* enough (a genuine inflation surge in short-lease assets) to outrun the cap-rate damage — possible, but rare.

**The recession regime: growth shock.** Here it's selective. A recession cuts rates (good for cap rates) but also cuts demand — occupancy falls, some tenants default, and leverage can force distressed sales (the 2008 channel). Lean toward the defensive, recession-resistant sectors: apartments (people always need housing), healthcare (demographics), self-storage. Avoid the cyclical, high-leverage corners (hotels, lower-quality office and retail).

On *sizing and pairing*: because listed REITs are ~0.75 correlated with stocks, they are **not** a substitute for the bond or cash ballast in a portfolio — don't count them as your "safe" sleeve. Think of REITs as an *equity sector tilt* with an income kicker and a partial inflation hedge, typically a single-digit percentage of a diversified portfolio, sized as part of your equity risk, not on top of it. They pair naturally with the rest of the [cross-asset map](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own): you hold genuine diversifiers (cash, Treasuries) for crash protection and add REITs for income and inflation-linked real exposure — never confusing the two jobs. And whichever way you go, the *first* question to ask before buying real estate is never "how nice are the buildings" — it's "where is the rate cycle, and how much leverage am I taking?"

On *public versus private and the smoothing trap*: a sophisticated allocator chooses the wrapper deliberately, not by which one's statement looks calmer. Private real estate's main genuine advantages are access (some property types and deals simply aren't available in listed form) and the *behavioral* benefit of not being shown a live, panic-inducing price every day — which can stop you from selling at the bottom. But you pay for that smoothness with illiquidity (you often can't get your money out for months or years, sometimes exactly when you most want it) and with the risk of buying or selling at a *stale appraisal* that hasn't caught up to reality. The listed market is the honest one: when listed REITs are down 25% and private funds still show small losses, the listed price is telling you the truth about the rate shock, and the private appraisal is simply late. A useful discipline is to treat the listed REIT market as a *real-time barometer* for what your private real estate is actually worth — if listed industrial REITs have repriced down 20%, your private warehouse is worth roughly 20% less too, no matter what last quarter's appraisal says. Never let a slow-moving appraisal lull you into believing you own a low-risk asset; you own the same leveraged, rate-sensitive buildings either way, just priced on a delay.

The one decision that matters most: own REITs for income when rates are falling or stable and growth is steady; trim or avoid them when the rate cycle is turning up hard. Everything else — sector selection, public vs private, leverage — is detail on top of that single, rate-driven call.

## Further reading & cross-links

Real estate is the asset where the cross-asset connections are densest, because it sits at the intersection of equity risk and rate risk. To go deeper into the pieces this post leaned on:

- [Government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — the duration mathematics that makes real estate behave like a long bond, and the risk-free rate that sets the floor under every cap rate.
- [Equities: owning a slice of growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth) — why listed REITs, at a +0.75 correlation, are best understood as a leveraged, income-heavy equity sector rather than a separate asset class.
- [Corporate credit: investment grade, high yield, and the spread](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) — the other "yieldy, leveraged, +0.75-to-stocks" asset, and the same lesson about diversifiers that fail in a crash.
- [The map of asset classes: what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — where real estate fits in the full menu, and why it's an equity-risk tilt rather than a ballast holding.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the single force that drives cap rates and therefore real-estate values; the rate cycle is the REIT cycle.

The takeaway to carry into the rest of the playbook: real estate is an income asset bought with leverage, priced off a cap rate that's bolted to interest rates. That makes it a partial inflation hedge over the long run and a fast rate-risk over the short run — a useful income holding when the rate cycle is friendly, and a leveraged liability when it isn't. Watch the rate cycle, respect the leverage, and never mistake a smooth appraisal for low risk.

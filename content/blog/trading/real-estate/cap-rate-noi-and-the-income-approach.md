---
title: "Cap Rate, NOI, and the Income Approach: A Property Is a Bond Made of Bricks"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How income property is really priced: net operating income is the coupon, the cap rate is the yield, and value moves inverse to the cap rate exactly like a bond price moves inverse to its yield."
tags: ["real-estate", "property", "cap-rate", "noi", "income-approach", "valuation", "yield", "interest-rates", "rental-yield", "vietnam", "investing"]
category: "trading"
subcategory: "Real Estate"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An income property is a bond made of bricks: its rent is the coupon, the cap rate is the yield, and its price moves *inverse* to that yield exactly like a bond's price moves inverse to its own.
>
> - **NOI** (net operating income) is the annual rent left *after* vacancy and operating expenses but *before* the mortgage, capex, income tax, and depreciation. It is the coupon the building pays.
> - The **cap rate** is just NOI ÷ price — the unlevered yield on the building. Master three equations and you can value, compare, and stress-test any income property: `Value = NOI ÷ cap`, `cap = NOI ÷ Value`, `NOI = Value × cap`.
> - Value moves **inverse** to the cap rate. Hold the rent fixed and lift the cap rate from 5% to 6% and the value falls about **17%** — the rent never changed. This is why the 2022 rate hikes repriced property worldwide.
> - The one number to remember: a cap rate is mostly **the risk-free rate (the 10-year government bond) plus a risk-and-growth premium**. When the bond yield rises, cap rates follow, and property prices fall.

A friend of mine — call him Minh — bought a small apartment in Thủ Đức, on the eastern edge of Ho Chi Minh City, in 2021. He paid ₫3.0 billion (about \$116,000) for it and rented it out for ₫15 million a month. He never thought of it as anything but "a flat I rent out." Then, in late 2022, a relative who works in finance asked him a question that genuinely stumped him: *"What's the yield on it?"* Minh had no idea. He knew the price. He knew the rent. He had never connected the two into a single number — and so he had no way to answer the only question that actually mattered: was his flat cheap, fair, or expensive?

That single number is the **cap rate**, and the framework around it — the **income approach** — is the most important idea in all of real estate that almost nobody outside the industry is ever taught. It is the bridge between a building and a bond. Once you see that bridge, a stack of seemingly unrelated facts snaps into one picture: why property prices fell across the world in 2022 even though rents were rising; why a parking garage in a boring suburb can be a *better* deal than a glamorous tower downtown; why "this place yields 9%!" is sometimes a warning sign rather than a brag; and why Vietnamese apartments — which barely yield 2.6% in rent — are priced on a completely different logic than American ones that yield 6%.

The diagram below is the mental model for the whole post: how you walk down from the rent a building *could* collect to the **net operating income** it actually keeps — the coupon that everything else is built on.

![Waterfall chart showing gross potential rent of 180 million dong per year, minus 9 million dong of vacancy and minus 21 million dong of operating expenses, arriving at net operating income of 150 million dong per year, with mortgage, capex and tax noted as not subtracted](/imgs/blogs/cap-rate-noi-and-the-income-approach-1.png)

We will build the whole thing from zero. By the end you will be able to do what Minh couldn't: take any income property — his flat, a US apartment building, a strip mall, a logistics warehouse — and in two lines of arithmetic say what it's worth, what yield it throws off, and how much that value would move if interest rates went up a point. That is genuine valuation power, and it fits on the back of a napkin.

A quick note before we start: this is educational, not financial advice. We are going to explain how property is priced and where the model bites — not tell you to buy or sell anything.

## Foundations: from gross rent to NOI, and what a cap rate actually is

Let's meet our two running characters. **Minh** owns that flat in Ho Chi Minh City; we'll price everything for him in Vietnamese đồng (₫), with US-dollar equivalents in parentheses so an international reader stays oriented (USD/VND ≈ 25,900 as of mid-2026). **Dana** is a US investor who owns a small apartment building in a mid-sized American city; we'll price her property in dollars (\$). Reusing the same two people lets the examples compound instead of restarting.

Everything in the income approach is built up from one raw ingredient — the rent — by subtracting things one layer at a time. Let's define each layer precisely, because the whole edifice is only as honest as these definitions.

### Gross potential rent

**Gross potential rent** (sometimes "gross scheduled income") is the rent the property would collect in a full year if every unit were occupied at the going rate, with nobody behind on payments. It's the *theoretical maximum* the building's leases could produce.

For Minh's flat rented at ₫15 million a month, gross potential rent is ₫15M × 12 = ₫180 million per year (about \$6,950). Note this is the ceiling, not what lands in his pocket — it assumes the flat is rented every single month with no gap and no deadbeat tenant. Reality always shaves a bit off, which is the next layer.

### Vacancy and credit loss

No property is rented 100% of the time. Tenants move out and the unit sits empty for a month while you find the next one; a tenant stops paying and you eat the loss while you evict. The allowance for this is **vacancy and credit loss**, almost always quoted as a percentage of gross potential rent.

A healthy residential market runs at maybe 5% vacancy; a struggling office building might be 20% or worse. For Minh's flat, a 5% vacancy allowance is ₫180M × 5% = ₫9 million per year. Subtract it and we have what's called **effective gross income**: ₫180M − ₫9M = ₫171 million.

The phrase to remember: gross rent is the rent the building *could* earn; effective gross income is the rent it *realistically* collects.

### Operating expenses — and the four things they explicitly exclude

**Operating expenses** ("opex") are the recurring costs of running the property as a property: management fees, building insurance, property tax or land-use fees, routine repairs and maintenance, common-area utilities, the cost of leasing up empty units. These are the costs you'd pay whether you owned the building with cash or with a mortgage — the costs of being a landlord.

Here is the single most important thing to internalize about NOI, and the place beginners most often go wrong. **Four big costs are deliberately *not* operating expenses:**

1. **The mortgage** (principal and interest). NOI is measured *before* any financing. This is on purpose — it lets you compare a building to itself regardless of how it was bought. The yield on the *building* shouldn't change just because you borrowed to buy it.
2. **Capital expenditures (capex)** — big, lumpy, infrequent spending that extends the building's life: a new roof, replacing the elevators, a full HVAC system. Routine maintenance is opex; a new roof is capex. (Many careful investors *do* subtract a capex reserve, and we'll flag where that bites.)
3. **Income tax** — the tax *you* owe on the profit. That depends on your personal situation, not the building's.
4. **Depreciation** — a non-cash accounting deduction. It's a tax concept, not a real cash cost, so it has no place in NOI.

For Minh's flat, suppose annual operating expenses — building management, insurance, the small Vietnamese land-use fee, and routine repairs — run ₫21 million.

### Net operating income (NOI) — the coupon

Now we can define the star of the show. **Net operating income (NOI)** is what's left after vacancy and operating expenses, but before the mortgage, capex, income tax, and depreciation:

$$ \text{NOI} = \text{Gross potential rent} - \text{Vacancy} - \text{Operating expenses} $$

For Minh:

$$ \text{NOI} = ₫180\text{M} - ₫9\text{M} - ₫21\text{M} = ₫150 \text{ million per year} $$

That ₫150 million (about \$5,800) is the building's **coupon** — the cash the bricks throw off every year, independent of how Minh financed them or what tax bracket he's in. This is exactly the waterfall in the figure above: gross rent at the top, two red subtractions, the green NOI at the bottom, and a gray reminder that the mortgage, capex, and tax all sit *below* the NOI line. Hold onto that number; the entire rest of the post is about turning it into a price.

#### Worked example: building NOI for a ₫7 billion HCMC flat

Let's do a fresh one, the kind a Vietnamese buyer actually faces in 2025. A modest 70 m² apartment in Hanoi or HCMC now runs around ₫7.0 billion (about \$270,000) at current primary prices. Say you, the buyer, can rent it for ₫15 million a month.

- Gross potential rent: ₫15M × 12 = **₫180 million per year**.
- Vacancy at 5%: −₫180M × 5% = **−₫9 million**.
- Effective gross income: ₫180M − ₫9M = **₫171 million**.
- Operating expenses (management, insurance, repairs, land-use fee): **−₫21 million**.
- **NOI = ₫171M − ₫21M = ₫150 million per year.**

So this ₫7 billion flat produces a ₫150 million coupon. Notice we did *not* subtract a mortgage — even if the buyer borrowed ₫5 billion of the ₫7 billion price, the NOI is unchanged. The financing changes the buyer's personal cash flow; it does not change the building's coupon.

*The intuition: NOI is the rent the bricks earn before you and your bank divide it up — it belongs to the building, not to the buyer.*

#### Worked example: building NOI for Dana's US apartment building

Let's do the same exercise in dollars so the method is unmistakably currency-agnostic. Dana owns a small 8-unit apartment building. Each unit rents for \$1,200 a month.

- Gross potential rent: 8 units × \$1,200 × 12 = **\$115,200 per year**.
- Vacancy and credit loss at 6% (a unit or two turns over each year, and a tenant occasionally pays late): −\$115,200 × 6% = **−\$6,912**.
- Effective gross income: \$115,200 − \$6,912 = **\$108,288**.
- Operating expenses — property taxes (\$14,000), insurance (\$4,000), property management at 8% of collected rent (\$8,663), repairs and maintenance (\$9,000), water/sewer/trash for the common areas (\$4,000): total **−\$39,663**.
- **NOI = \$108,288 − \$39,663 = \$68,625 per year.**

Two things to notice. First, the operating-expense ratio here is about \$39,663 ÷ \$108,288 ≈ **37%** of effective gross income — a typical figure for US apartments, which run anywhere from 35% to 50% depending on whether tenants pay their own utilities. A broker quoting you a building with a suspiciously low expense ratio (say 20%) is almost certainly leaving real costs out, which inflates the NOI and hides a worse deal. Second, Dana's mortgage — say she borrowed \$1.2 million at 6.5% for roughly \$78,000 a year in payments — does *not* appear anywhere above. Her NOI is \$68,625 whether she paid all cash or borrowed most of the price. (And notice: \$68,625 of NOI against a \$78,000 mortgage payment means the building's coupon doesn't even cover the debt — a danger the cap rate alone won't show you, which is exactly why we treat financing separately.)

*The intuition: the operating-expense ratio is a sanity check — if a deal's expenses look too light, someone has scrubbed real costs out of the NOI to make the price look cheap.*

### The cap rate — the yield on the bricks

We have a price (₫7 billion) and a coupon (₫150 million). Connect them and you get the **capitalization rate**, universally shortened to **cap rate**:

$$ \text{Cap rate} = \frac{\text{NOI}}{\text{Price (or Value)}} $$

For our ₫7 billion flat: cap rate = ₫150M ÷ ₫7,000M = **2.14%**. For Minh's older flat that he bought at ₫3.0 billion with the same ₫150M NOI, cap rate = ₫150M ÷ ₫3,000M = **5.0%**.

That's it. The cap rate is the *unlevered, current yield on the building* — what you'd earn on your money each year, as a percentage, if you bought the building outright with cash and pocketed the NOI. It is the real-estate twin of a bond's yield. A bond paying \$50 a year on a \$1,000 price yields 5%; a building paying ₫150 million on a ₫3 billion price "yields" — caps at — 5%. Same idea, different asset.

The act of turning a stream of income into a single price by dividing by a yield is called **capitalization** — you are *capitalizing* the income. "Cap rate" is short for "the rate at which we capitalize the income." Don't let the jargon intimidate you; it's one division.

### Gross rental yield — the Vietnamese shortcut, and why it lies a little

Before we leave the foundations, one more term, because it's the one most Vietnamese (and most casual) buyers actually use: **gross rental yield**. It's the simplest possible income measure — annual rent ÷ price — and it skips the entire NOI calculation:

$$ \text{Gross rental yield} = \frac{\text{Annual gross rent}}{\text{Price}} $$

For our ₫7 billion flat collecting ₫180 million in annual rent: gross yield = ₫180M ÷ ₫7,000M = **2.57%**, call it 2.6%. It's quick and quotable, which is why it dominates conversations at coffee with a HCMC agent. But it's a flattering number, because it pretends rent is free to collect — no vacancy, no management, no insurance, no repairs. The honest measure, the cap rate, subtracts all of that first. The gap between the two is exactly the cost of being a landlord, and it's never zero. Keep both in your head: gross yield is the headline; the cap rate is the truth.

#### Worked example: cap rate vs gross yield on the ₫7 billion flat

Let's put the two side by side on the same flat — same ₫7.0 billion (≈ \$270,000) price, same ₫15M/month rent — so you can see precisely how much gross yield overstates the return.

- **Gross rental yield** = ₫180M ÷ ₫7,000M = **2.57%**.
- **Cap rate** = ₫150M (NOI) ÷ ₫7,000M = **2.14%**.
- **The gap** = 2.57% − 2.14% = **0.43 percentage points** — about a sixth of the headline yield, lost to vacancy and operating costs that gross yield ignored.

That 0.43-point haircut is the cost of being a landlord made visible. And both numbers are *strikingly low* — a Vietnamese bank deposit pays more than either. Which raises the puzzle we'll resolve later: why does anyone buy a 2.1%-cap flat? (Answer: they're not buying the coupon — they're buying the expected price appreciation. Hold that thought.)

*The intuition: gross yield is the brochure number; cap rate is the bank-statement number — and the difference is every cost the brochure forgot.*

## The three equations and the bond analogy

Everything in the income approach is one little triangle of three quantities — **Value**, **NOI**, and **cap rate** — bound by a single relationship. Rearrange that relationship three ways and you get three equations. Know any two of the three quantities and the third falls out instantly.

![Triangle diagram of the three income-approach equations: NOI equals value times cap rate at the top, value equals NOI divided by cap rate on the lower left showing 3.0 billion dong, and cap equals NOI divided by value on the lower right showing 5.0 percent, all connecting to a central property income hub](/imgs/blogs/cap-rate-noi-and-the-income-approach-2.png)

The three equations are:

$$ \text{Value} = \frac{\text{NOI}}{\text{cap rate}} \qquad \text{cap rate} = \frac{\text{NOI}}{\text{Value}} \qquad \text{NOI} = \text{Value} \times \text{cap rate} $$

They are the same statement written three ways, exactly like `distance = speed × time` is also `speed = distance ÷ time` and `time = distance ÷ speed`. Let's give each a job:

- **`Value = NOI ÷ cap`** is the *valuation* equation. You have a building's NOI and you know what cap rate similar buildings trade at, and you want a price. This is the workhorse — it's how appraisers, lenders, and buyers actually put a number on income property.
- **`cap rate = NOI ÷ Value`** is the *comparison* equation. You have a price and an income and you want to know if it's a good deal. Computing the cap rate lets you line up wildly different properties on one scale.
- **`NOI = Value × cap`** is the *target* equation. You know what you want to pay and the going cap rate, and you back out how much income the building must produce to justify it.

Take Minh's flat, ₫150M NOI at a 5% cap. The triangle says: Value = ₫150M ÷ 5% = **₫3.0 billion**. That matches what he paid. Good — the price is "fair" at a 5% cap, which is the market cap rate for that kind of flat in that area.

### Why "a bond made of bricks" is exactly right

Let's make the bond analogy precise, because it's not a loose metaphor — the math is genuinely the same.

A **bond** is a loan you make to a government or a company. It pays you a fixed annual amount, the **coupon**, and trades at a **price**. Its **yield** is coupon ÷ price. A \$1,000 bond with a \$50 coupon yields 5%. If interest rates in the economy rise and new bonds start paying \$60 on \$1,000, nobody will pay \$1,000 for your old \$50 bond anymore — they'll only pay a price low enough that *your* \$50 coupon represents a competitive 6% yield. That price is \$50 ÷ 6% = \$833. Your bond's price *fell* even though its coupon never changed. (We unpack this seesaw in depth in [price and yield, the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) — the single most useful idea you can carry from bonds into real estate.)

Now read that paragraph again and swap the words: **NOI** is the coupon, **cap rate** is the yield, **building value** is the price. A building with a ₫150M coupon at a 5% market cap rate is worth ₫3.0 billion. If the market cap rate rises to 6% — because interest rates rose — nobody will pay ₫3.0 billion anymore. They'll only pay ₫150M ÷ 6% = ₫2.5 billion. The building's value *fell* even though its rent never changed.

| | Bond | Income property |
|---|---|---|
| The cash it pays | Coupon (\$50/yr) | NOI (₫150M/yr) |
| What you pay for it | Price (\$1,000) | Value (₫3.0bn) |
| The yield | Coupon ÷ price = 5% | NOI ÷ value = cap rate = 5% |
| When market yields rise | Price falls | Value falls |
| The driver | Risk-free rate + credit spread | Risk-free rate + risk-and-growth premium |

A property is a bond made of bricks. That's the whole post in five words, and the rest is just consequences.

## Why value moves inverse to the cap rate

This is the single most counterintuitive — and most important — consequence of the framework. **When the cap rate goes up, value goes down, and vice versa.** The rent can be completely unchanged and the value still moves, purely because the cap rate moved. New investors find this baffling: how can a building be worth less if it's collecting exactly the same rent? Because price is rent *divided by* yield, and dividing a fixed number by a bigger denominator gives a smaller answer. It's arithmetic.

![Before-and-after comparison showing the same 150 million dong net operating income divided by a 5.0 percent cap rate gives a value of 3.0 billion dong, while dividing it by a 6.0 percent cap rate gives 2.5 billion dong, a 17 percent decline of 500 million dong](/imgs/blogs/cap-rate-noi-and-the-income-approach-4.png)

The figure above is the **cap-rate seesaw**, and it's worth staring at. On the left, ₫150M NOI ÷ 5.0% = ₫3.0 billion. On the right, the *exact same* ₫150M NOI ÷ 6.0% = ₫2.5 billion. The cap rate rose one percentage point and ₫500 million of value evaporated — a 17% drop — without a single đồng of rent changing hands.

### Cap-rate compression and expansion

The market gives these moves names.

**Cap-rate compression** is when cap rates *fall* — yields squeeze tighter, so prices rise. This is what made every property owner feel like a genius from roughly 2012 to 2021: cheap money pushed yields down everywhere, cap rates compressed from ~6% toward ~5%, and the same NOI was suddenly worth 20% more. Compression is a tailwind; you can make money even if your rents are flat, simply because the market is willing to pay a lower yield.

**Cap-rate expansion** is the reverse: cap rates *rise*, prices fall. This is what mauled property in 2022–2023 when interest rates spiked. The same buildings producing the same (or even rising) NOI lost value because buyers demanded a higher yield. Expansion is a headwind that can wipe out years of rent growth in a single repricing.

Here is the unsettling part: the cap rate is set by *the market*, not by your building. You can be a perfect landlord, raise rents every year, fix every leak — and still watch your property's value fall because the market cap rate expanded. Your coupon went up; the yield buyers demand went up more.

#### Worked example: the seesaw on Minh's flat (−17%)

Let's run the seesaw on a real holding. Minh's flat: NOI ₫150 million, bought at a 5.0% cap, so worth ₫3.0 billion (≈ \$116,000).

- **At a 5.0% cap:** Value = ₫150M ÷ 0.050 = **₫3.00 billion**.
- **At a 6.0% cap:** Value = ₫150M ÷ 0.060 = **₫2.50 billion**.
- **Change:** ₫2.50bn − ₫3.00bn = **−₫500 million**, a fall of −₫500M ÷ ₫3.00bn = **−16.7%**.

Notice the asymmetry of percentages: the cap rate rose 1 point, from 5% to 6%, which is a *20%* relative increase in the yield (6/5 = 1.20). The value fell ₫3.0bn → ₫2.5bn, which is the reciprocal: ₫2.5bn ÷ ₫3.0bn = 0.833, a −16.7% move. The lower the cap rate you start from, the more violent the price move for a given change in rate — a property at a 4% cap is far more sensitive to a 1-point rate jump than one at an 8% cap. This is the real-estate version of bond **duration**: low-yield assets have long duration and swing hardest when yields move.

*The intuition: at low cap rates, you are paying a very high price for each đồng of rent, and a small rise in the yield buyers demand knocks a huge chunk off that price — the bond seesaw, in bricks.*

## What sets the cap rate: risk-free rate plus risk plus growth

If the cap rate is the lever that moves property values, the obvious question is: what sets the cap rate? Why 5% and not 8%? The answer is a small, honest formula that you can carry everywhere:

$$ \text{cap rate} \approx \text{risk-free rate} + \text{risk premium} - \text{expected NOI growth} $$

Let's define each piece from zero.

**The risk-free rate** is the return you can earn with essentially no risk of not being paid — in practice, the yield on a long-term government bond (the 10-year US Treasury in America, the 10-year government bond in Vietnam). It's the *floor* under every other yield. If a safe government bond pays 4.5%, no rational investor accepts *less* than 4.5% to hold a risky building — they'd just buy the bond. So the cap rate starts at the risk-free rate and builds up from there. (For why this one rate prices every asset on Earth, see [real yields, the variable that prices everything](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)

**The risk premium** is the extra yield investors demand for taking on everything that makes a building riskier than a government bond: the tenant might leave, the roof might fail, the neighborhood might decline, and — crucially — you can't sell a building in a day the way you can sell a bond. That last one, **illiquidity**, is a real cost and a real part of the premium. Riskier, harder-to-sell income earns a higher premium.

It's worth dwelling on **illiquidity** because beginners systematically underprice it. A government bond can be sold in seconds at a price you can see on a screen. A building takes months to sell, costs 2–6% in transaction fees and taxes to transact (in Vietnam, a 2% personal income tax on the transfer plus a 0.5% registration fee, on top of agent commissions), and the price you'll actually get is uncertain until a buyer signs. That gap — the difference between an asset you can exit instantly and one you might be stuck holding through a downturn — is a genuine cost, and investors demand to be paid for it in the form of a higher cap rate. It's a large part of *why* even a perfectly safe building yields more than a Treasury: not because the rent is risky, but because your *exit* is.

**Expected NOI growth** *subtracts* from the cap rate. Here's the subtle part: a bond's coupon is fixed forever, but a building's rent tends to *grow* over time with inflation and the economy. If buyers expect the NOI to rise 3% a year, they'll accept a lower current yield today, because they're being paid partly in future growth. A building everyone expects to grow fast (think prime apartments in a booming city) trades at a *low* cap rate not because it's safe, but because buyers are pricing in the growth. This is the deep reason the same cap rate can mean opposite things: a 5% cap on a stagnant building is a 5% coupon and nothing more, while a 5% cap on a fast-growing one is a smaller coupon today plus a promise of growth — and only one of those survives a recession with its value intact.

So a cap rate is really: *the safe rate, plus a premium for the building's risk and illiquidity, minus the growth investors expect.* Two buildings can have the same cap rate for opposite reasons — one safe-and-stagnant, one risky-but-fast-growing.

#### Worked example: decomposing Dana's 6% cap rate

Dana, our US investor, is looking at her apartment building, which trades at a 6.0% cap rate. The 10-year Treasury yields 4.5%. Let's decompose the cap into its parts:

- **Risk-free rate (10Y Treasury):** 4.5%.
- **Plus risk-and-illiquidity premium for an apartment building:** call it +3.5% (apartments are moderately risky and far less liquid than a Treasury).
- **Minus expected NOI growth:** apartment rents in her city are expected to grow ~2%/yr, so −2.0%.
- **Net cap rate:** 4.5% + 3.5% − 2.0% = **6.0%.** ✓

Now stress it. Say the Treasury yield jumps to 5.5% (a 1-point rise) and nothing else changes. The cap rate becomes 5.5% + 3.5% − 2.0% = **7.0%**. Her building's value, at the same NOI, falls from NOI ÷ 6% to NOI ÷ 7% — a drop of 6/7 − 1 = **−14.3%**. A one-point move in a government bond yield, transmitted straight through the cap-rate equation, knocked a seventh off her building's value. She did nothing wrong; the floor under all yields rose.

*The intuition: a cap rate is mostly the government bond yield in disguise — change that yield and you change every property price, which is why real estate lives or dies by the bond market.*

## Cap rate vs interest rates: the seesaw that repriced the world in 2022

Now we can answer the question that motivated this whole post: **why did property prices fall across the world in 2022 even as rents were rising?** Because cap rates are tethered to the risk-free rate, and in 2022 central banks yanked the risk-free rate up at the fastest pace in forty years.

![Line chart of the US all-property cap rate against the 10-year Treasury yield from 2015 to 2026, with the cap rate falling to about 5.0 percent in 2021 then rising back toward 6.0 percent, the Treasury yield rising from under 1 percent in 2020 to about 4.5 percent by 2026, and the amber shaded gap between them labeled as the risk-and-growth premium that narrows from about 390 basis points to about 180 basis points](/imgs/blogs/cap-rate-noi-and-the-income-approach-3.png)

The chart above is the relationship in real US data. The gray line is the 10-year Treasury yield; the blue line is the all-property cap rate; the amber band between them is the **risk-and-growth premium** — the spread. Three things jump out.

First, **the two lines move together over time.** Cap rate ≈ Treasury yield + spread, just like the formula says. They're not glued, but they're tethered.

Second, **2021 was the era of compression.** Money was historically cheap — the Fed had pinned short rates near zero and the 10-year fell to ~0.9% in 2020 — and that cheap money pushed cap rates down to a trough of about **5.0%**. Property prices soared. The US Case-Shiller home price index ran from ~134 at its post-crash trough in 2012 to over **308 by mid-2022**, more than doubling. Compression made everyone a genius.

Third, **2022–2023 was the great repricing.** As inflation surged, the Fed hiked aggressively and the 10-year Treasury leapt from under 1.5% to over 4%. The risk-free floor rose, and cap rates expanded right behind it, climbing from ~5.0% back to ~6.0%. Through the seesaw, that 1-point cap expansion lopped roughly a sixth off the value of income property nationwide — even though rents, in most sectors, were *rising*. US 30-year mortgage rates told the same story from the financing side: a record-low 2.65% in January 2021 became 7.08% by October 2022 and peaked near 7.79% in October 2023. The cost of money tripled, and property repriced.

Notice one more thing in the chart: the spread (the amber band) **narrowed** from about 390 basis points in 2014 to roughly 180 basis points by 2025. (A *basis point* is one-hundredth of a percent; 100 bps = 1%.) When the spread is thin, property is offering only a slim premium over the risk-free bond — which means investors are not being paid much for the extra risk and illiquidity of owning bricks. Thin spreads are a yellow flag: they often precede repricing, because there's little cushion if the risk-free rate rises further.

This is the seesaw at economy scale. *Rising rates crush property values not by lowering rents but by raising the yield buyers demand.* If you remember one mechanism from this post, make it this one.

## Cap rate by asset type and risk: why riskier means cheaper per đồng of income

A cap rate is not one number — it's a different number for every *kind* of property, and the pattern is consistent: **riskier, less liquid income earns a higher cap rate, which means a lower price for each đồng of NOI.** That sounds backwards until you remember the formula. Higher cap rate = lower value per unit of income = the income is "cheaper." Safe income is expensive; risky income is cheap. The market is just demanding more yield to compensate for more risk.

![Matrix comparing typical cap rates, income risk, and liquidity across five property types: prime city office at 4.5 to 5.5 percent with low risk and high liquidity, grade-A apartment at 5.0 to 6.0 percent with low to moderate risk and high liquidity, suburban retail at 7.0 to 8.5 percent with moderate to high risk and moderate liquidity, logistics and industrial at 6.0 to 7.5 percent with moderate risk and moderate liquidity, and raw land or dat nen with no cap rate, high risk and low liquidity](/imgs/blogs/cap-rate-noi-and-the-income-approach-6.png)

Read the matrix above top to bottom and the logic is clean:

- **Prime city-center office or trophy apartment (4.5–5.5%):** the safest, most liquid income there is — long leases, blue-chip tenants, a deep buyer pool. Investors accept a low yield because the income is reliable and they can always sell. We call this **core** real estate. Low cap rate = high price per đồng of NOI.
- **Grade-A apartment (5.0–6.0%):** people always need somewhere to live, leases are short but renew constantly, and demand is broad. Low-to-moderate risk, high liquidity, modestly higher cap rate than trophy office.
- **Logistics / industrial warehouses (6.0–7.5%):** boomed with e-commerce, but more tied to the economic cycle and to single large tenants. Moderate risk, moderate cap rate. A boring box in an industrial park can out-yield a glamorous downtown tower.
- **Suburban retail / strip malls (7.0–8.5%):** higher tenant risk (shops fail, e-commerce competes), thinner buyer pool, slower to sell. Investors demand a fat yield — a high cap rate — to take it on. The income is "cheap" because it's risky.
- **Raw land / đất nền (no cap rate at all):** bare land produces *no NOI*. There's no rent to capitalize, so the income approach simply doesn't apply — you can't divide a nonexistent coupon by anything. Land is priced purely on what someone will pay for future development or appreciation. It is the opposite of a bond: all hope, no coupon. This is why much of Vietnam's most-traded real estate (đất nền, raw plots in satellite provinces) is *speculation*, not income investing — there's no coupon underneath it.

The crucial mindset flip: **a high cap rate is not a "better deal" — it's a higher-risk deal.** A 9% cap and a 5% cap aren't a good property and a bad property; they're a risky income and a safe income, each priced to its risk. We'll hammer this in the misconceptions section, because it's where beginners lose money.

#### Worked example: Minh's flat vs a suburban strip mall

Suppose both throw off the same ₫150 million NOI. Minh's grade-A city flat caps at 5.0%; a suburban strip mall, riskier and harder to sell, caps at 8.0%.

- **Flat:** Value = ₫150M ÷ 0.050 = **₫3.00 billion** (≈ \$116,000).
- **Strip mall:** Value = ₫150M ÷ 0.080 = **₫1.88 billion** (≈ \$72,500).

Same coupon, but the strip mall is worth ₫1.1 billion *less*. A naïve investor sees the strip mall yielding 8% vs the flat's 5% and thinks "the mall is the better deal — more yield!" But the market priced that extra 3% of yield as *compensation for risk*: the mall's tenants are more likely to fail, its income more likely to drop, and it's far harder to sell in a hurry. You're not getting a free 3%; you're being paid to take a real chance of losing money. The flat's lower yield *is* its safety.

*The intuition: a higher cap rate is the market handing you a bigger yield precisely because it thinks you'll need it — risk and yield rise together, always.*

## Going-in vs exit cap, and the assumption risk hiding in plain sight

So far we've used one cap rate. But a real investment has at least two, and the gap between them is where fortunes are made and lost. Buy a building today and you live with two different cap rates across your hold:

- The **going-in cap rate** is the cap rate at which you buy — today's NOI ÷ today's price. It's a fact; you can compute it the day you sign.
- The **exit cap rate** (or "terminal cap rate") is the cap rate at which you *expect to sell*, years from now. It's an *assumption* — you're guessing what yield the next buyer will demand. And small changes in that guess swing your return enormously.

When the cap rate holds steady across your hold, the math is friendly: as NOI grows, value grows in lockstep. The figure below shows a clean five-year hold at a steady 6% cap, with NOI growing ~5% a year.

![Timeline of a five-year hold at a steady 6 percent cap rate, starting at year zero with NOI of 150 million dong and value of 2.50 billion dong, and growing each year to year five with NOI of 191 million dong and value of 3.19 billion dong as the rent compounds and the value follows](/imgs/blogs/cap-rate-noi-and-the-income-approach-5.png)

Read across the timeline: NOI ₫150M → ₫158M → ₫165M → ₫174M → ₫182M → ₫191M, and at a constant 6% cap the value tracks it perfectly: ₫2.50bn → ₫2.63bn → ₫2.76bn → ₫2.89bn → ₫3.04bn → ₫3.19bn. Every 5% of rent growth lifts value by the same 5%. The coupon compounds, and because the yield is fixed, the price follows it up. *This* is the benign case — and the one brokers always pitch.

But the exit cap is an assumption, and the world doesn't promise it stays put. If interest rates rise over your hold, your exit cap *expands* above your going-in cap, and that expansion fights your rent growth. Sometimes it wins.

#### Worked example: going-in vs exit cap and the hit to IRR

Dana buys a building today producing \$100,000 of NOI, at a **going-in cap of 6.0%**. So she pays \$100,000 ÷ 0.06 = **\$1,666,667**. She plans a 5-year hold and expects to grow NOI 5% a year through good management, reaching \$100,000 × 1.05⁵ ≈ **\$127,600** by year 5. The question is what she can sell for — and that hinges entirely on the **exit cap rate** she'll face.

**Scenario A — rates flat, exit cap = going-in cap of 6.0%:**
Sale price = \$127,600 ÷ 0.06 = **\$2,126,700**. She bought at \$1,666,667, so the property gained about \$460,000, roughly +27.6% over five years — a healthy result driven by the compounding NOI.

**Scenario B — rates rose, exit cap expands to 7.0%:**
Sale price = \$127,600 ÷ 0.07 = **\$1,822,900**. The *same* building, the *same* grown NOI, but the next buyer demands a 7% yield instead of 6%. Her exit value is roughly \$304,000 *lower* than Scenario A. Her five-year gain shrinks from +27.6% to about +9.4% — most of her hard-won rent growth was eaten by the one-point cap expansion.

The lesson on **internal rate of return** (IRR — the annualized return that accounts for the timing of every cash flow): in Scenario A, with steady caps and growing rent, Dana's IRR might run around 10–12% a year once you include the rent collected along the way. In Scenario B, the same operating performance but an expanded exit cap can drag the IRR down toward 5–6% — *half*. Nothing about how she *ran* the building changed. The entire difference is the exit-cap assumption, which is really a bet on where interest rates go.

*The intuition: your buy-side cap rate is a fact you can verify, but your sell-side cap rate is a guess about future interest rates — and that single guess can swing your return more than years of skilled management.*

This is **assumption risk**, and it's the quiet killer in real-estate underwriting. A pro forma (the projected-returns spreadsheet a broker hands you) almost always assumes the exit cap equals or even *beats* the going-in cap. Flip that one cell up a point and watch the promised returns collapse. Always ask: *what exit cap is this deal assuming, and what happens if rates rise instead?*

## Common misconceptions

The income approach is simple arithmetic, but it generates a remarkable number of confident, expensive errors. Here are the ones that cost people the most.

### "A high cap rate is a good deal."

The most common and most dangerous mistake. A high cap rate means the market is pricing the income *cheaply* — and it's cheap for a reason: the income is riskier or less liquid. A 10% cap in a declining town with a shaky tenant is not a bargain; it's the market warning you that this income may not last. Conversely, a 4.5% cap on a prime, fully-leased trophy building isn't "expensive" — you're paying up for safety and liquidity. The cap rate isn't a quality score; it's a risk gauge. *Higher cap = higher risk, not better deal.* The only fair comparison is cap rates *within the same risk class*: a 6.5% cap on one strip mall vs a 7.5% cap on a comparable strip mall is a meaningful comparison; a strip-mall cap vs a trophy-office cap is not.

### "The cap rate is my return."

It is not. The cap rate is the *unlevered, current* yield on the building — what you'd earn in year one if you paid all cash and the NOI never changed. Your actual **total return** has three more parts the cap rate ignores: **NOI growth** (rents rising over time lifts both your income and your value), **leverage** (a mortgage can amplify your return on equity — and your losses; see [leverage and the mortgage, how debt amplifies property](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property)), and **cap-rate movement** (compression lifts your value, expansion crushes it). The cap rate is the *starting yield*, not the finish line. A 5% cap building with 4% rent growth and modest leverage can deliver a low-double-digit total return; a 9% cap building with falling rents and an expanding exit cap can lose money. *Cap rate is your day-one coupon yield, not your IRR.*

### "Gross rental yield is the cap rate."

These get conflated constantly, especially in residential markets, and the gap between them is huge. **Gross rental yield** is just annual rent ÷ price — it ignores vacancy *and every operating expense*. The **cap rate** is *net* operating income ÷ price — it subtracts vacancy and opex first. Gross yield always flatters the property because it pretends the rent is costless to collect. For Minh's ₫7 billion flat: gross yield = ₫180M ÷ ₫7,000M = **2.6%**, but the cap rate = ₫150M ÷ ₫7,000M = **2.1%**. The half-point gap is the vacancy and operating costs that gross yield conveniently ignores. In Vietnam, where buyers almost always quote *gross* yield, this matters enormously — your true net yield is meaningfully lower than the headline.

### "If the rent didn't change, the value can't change."

We've spent the whole post killing this one, but it dies hard because it *feels* true. Value = NOI ÷ cap. If the cap rate moves — because interest rates moved, or the market's risk appetite shifted — the value moves even with rent frozen. A building producing identical rent was worth ₫3.0 billion in 2021 and ₫2.5 billion in 2023 purely because cap rates expanded with rising rates. *The market reprices your yield even when it can't touch your rent.*

### "Cap rates and interest rates are independent — real estate is its own thing."

Real estate feels local, tangible, separate from the abstract bond market. But cap rates are tethered to the risk-free rate, because every investor can choose between a building and a government bond. When the bond yield rises, the bar every property must clear rises with it. Real estate is *not* an island; it's a high-yield bond with a roof, and it moves with the bond market more than most owners want to admit.

### "đất nền (raw land) has a great yield."

Bare land produces no rent and therefore *has no cap rate at all* — there's no coupon to capitalize. When someone pitches a land plot on its "yield," they're talking about *expected appreciation*, which is a forecast of future price, not income you can bank today. That's speculation, not income investing. It can absolutely pay off, but it's a fundamentally different (and riskier) game than buying a coupon. Don't apply income-approach logic to an asset with no income.

## How it shows up in real markets

### The global repricing of 2022–2023

This is the cleanest real-world demonstration of the seesaw at scale. From 2021 into 2023, the US 10-year Treasury yield rose from under 1.5% to over 4%, dragging the risk-free floor up beneath every asset. Commercial real-estate cap rates expanded from a ~5.0% trough back to ~6.0%, and through `Value = NOI ÷ cap`, that one-point expansion meant property values fell on the order of 15–20% *even where rents were rising*. Office was hit hardest — both because cap rates expanded *and* because remote work cut NOI by emptying buildings, a double blow. The lesson the income approach predicts exactly: when the risk-free rate jumps, property reprices downward regardless of rents, and the assets with the lowest going-in cap rates (the "safest," lowest-yield trophies) fall the most in percentage terms, because low-yield assets have the longest duration.

### Vietnam: priced for appreciation, not yield

Vietnamese residential property runs on a logic that baffles foreign investors: the yields are *tiny*. The chart below puts Vietnam's gross rental yields next to income-driven markets.

![Bar chart of gross rental yields and cap rates showing HCMC apartments at 2.6 percent and Hanoi apartments at 3.5 percent tinted as low-yield appreciation plays, Singapore at 3.0 percent and London at 3.8 percent in the middle, and US apartment cap rates at 5.5 percent and US retail cap rates at 7.5 percent tinted as income markets priced for yield](/imgs/blogs/cap-rate-noi-and-the-income-approach-7.png)

A HCMC apartment yields about **2.6%** gross; a Hanoi flat about 3.5%; compared with a US apartment building capping at ~5.5% or US retail at ~7.5%. Why would anyone buy a 2.6%-yield flat when a bank deposit in Vietnam pays more? Because Vietnamese buyers aren't buying the *coupon* — they're buying the *appreciation*. With primary HCMC apartment prices rising from ~₫91M/m² in 2024 to ~₫111M/m² in 2025 (roughly +33% then +40% year-on-year in the hottest segments), the income is almost an afterthought; the bet is on price. In the language of our formula, buyers accept a rock-bottom cap rate because they're pricing in enormous *expected NOI growth* (and price growth). That's a coherent strategy in a fast-urbanizing, supply-constrained market — but it's a momentum bet, not an income bet, and momentum bets reverse. When growth expectations falter, a property priced for appreciation has almost no coupon to cushion the fall. Compare that with the income-market apartment, where a 5.5% coupon keeps paying you while you wait.

### The 2022–23 Vietnamese bond freeze

Vietnam's developers fund themselves heavily through corporate bonds, and in 2022 that market seized. The arrests at Tân Hoàng Minh (April 2022) and Vạn Thịnh Phát / SCB (October 2022) froze the corporate-bond market overnight, choking developers' financing. Projects stalled, presale buyers were stranded, and transaction volumes collapsed. The income-approach lens explains the price action: even where rents held, the *risk premium* on Vietnamese property spiked (default risk, liquidity risk, counterparty risk all rose at once), pushing implied cap rates up and values down. The SBV cut policy rates four times in 2023 and Decree 08 gave bond-restructuring relief, which began pulling the risk premium back down — and as that premium compressed, prices recovered. Same seesaw, Vietnamese edition: the move came through the *premium* term, not the rent.

### Japan's lost decades and the low-cap-rate trap

Japan's 1980s property bubble is the textbook warning about ultra-low cap rates. At the peak, prime Tokyo commercial property reportedly traded at cap rates as low as 2% — buyers pricing in endless growth. When growth stopped, there was almost no coupon underneath the price to break the fall. Land values fell roughly 70–80% from peak over the following two decades. The income approach explains both the bubble and the bust: a 2% cap rate is a bet that NOI will grow forever; when that bet failed, the price had nothing to stand on but the (tiny) actual income. *The lower the cap rate, the more of the price is hope rather than coupon — and hope is the part that can vanish.*

### The 2008 US housing crash

US national home prices, per the Case-Shiller index, fell from a peak of ~184.6 in July 2006 to a trough of ~134 in February 2012 — roughly **−27%** peak-to-trough, the worst nationwide housing decline since the Depression. The income approach was screaming a warning years earlier: home prices had risen far faster than rents, which meant implied yields (rent ÷ price) had collapsed to absurdly low levels. Price had detached from the coupon. When credit tightened and the risk premium snapped back, prices reverted toward what the income could justify. Where the coupon and the price diverge, the price eventually moves toward the coupon — and the further they've diverged, the harder the snap.

### REITs: the cap-rate seesaw, traded by the second

A **REIT** (real estate investment trust) is a company that owns income property and trades on the stock exchange like a share — a way to own a slice of a portfolio of buildings without buying one yourself. Because REIT prices update every second the market is open, they make the cap-rate seesaw visible in real time, where private property prices lag by months. A REIT essentially *is* a basket of NOI being capitalized at a market yield, and its dividend yield behaves like a cap rate.

Watch what happens when the 10-year Treasury yield jumps: REIT prices fall almost immediately, often *more* than the underlying buildings will eventually be marked down, because the stock market reprices the same NOI at the higher yield instantly. In the 2022 rate shock, listed REIT indices in the US fell roughly 25% even as the private appraised values of the very same kinds of buildings had barely begun to move — the public market simply applied the higher cap rate first. The lesson cuts both ways. REITs let small investors buy income property in liquid, dividend-paying form (the income-and-rates intuition is unpacked in [real estate, REITs, income, leverage and rates](/blog/trading/cross-asset/real-estate-reits-income-leverage-and-rates)) — but that liquidity comes with the bond seesaw running live: when rates rise, you watch the value fall in real time rather than in a slow-motion appraisal. Same arithmetic — `Value = NOI ÷ cap` — just with the price tag updating every second.

### Singapore: a disciplined, income-priced market

Singapore is a useful contrast to Vietnam: a mature, transparent, heavily-regulated property market where prices stay much closer to what the income justifies. Prime Singapore residential gross yields run around 3% and office cap rates have historically sat in the 3.5–4.5% range — low, but tethered to the city-state's low risk-free rate and its reputation as a safe-haven store of value. When global rates rose in 2022–2023, Singapore cap rates expanded too, and the government's cooling measures (stamp duties on foreign and multiple-property buyers) deliberately leaned against speculation to keep prices anchored nearer the income. The contrast with Vietnam is instructive: both are fast-growing Asian markets with low yields, but Singapore's low cap rates rest on genuinely *low risk* (a safe rate, deep liquidity, strong rule of law), while Vietnam's low yields rest more on *high expected growth and appreciation*. Same low number, very different foundations — and the foundation determines how far the price can fall if the story changes.

## When this matters / Further reading

Here's where this touches your life. If you ever buy a property to rent out, the income approach is how you tell *cheap* from *expensive* — and the only way to do it is to build the NOI honestly (subtract real vacancy and real operating costs, not the broker's optimistic ones) and compute the cap rate. If you're comparing two rentals, the cap rate puts them on one scale — but only if you compare *within the same risk class*. If you own property already, the cap-rate seesaw tells you why your home's value moves with interest rates even when your neighborhood hasn't changed: the cap rate buyers demand is mostly the government bond yield in disguise. And if you ever hear a pitch promising a fat 9% yield, you now know to ask the only question that matters: *what risk am I being paid that 9% to take?*

The three equations — `Value = NOI ÷ cap`, `cap = NOI ÷ Value`, `NOI = Value × cap` — fit on a napkin and will let you value, compare, and stress-test any income property for the rest of your life. The one mechanism to carry above all: **value moves inverse to the cap rate, and the cap rate is mostly the risk-free rate plus a risk-and-growth premium.** That is why rising interest rates crush property values, and why a property is, at bottom, a bond made of bricks.

To go deeper, the natural next reads:

- [Price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) — the same inverse relationship, derived cleanly in its purest form. Master it on bonds and real estate becomes obvious.
- [How property is valued: the three approaches](/blog/trading/real-estate/how-property-is-valued-three-approaches) — the income approach is one of three; see where it fits alongside the sales-comparison and cost approaches, and when each one rules.
- [Leverage and the mortgage: how debt amplifies property](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property) — the cap rate is *unlevered*; add a mortgage and your return on equity (and your risk) changes dramatically. This is where the cap rate stops and your actual return begins.
- [Real yields: the variable that prices everything](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the macro view of *why* the risk-free rate is the floor under every cap rate, every bond, and every asset price on Earth.

*This article is educational, not financial advice. Cap rates, yields, prices, and rates cited are as of mid-2026 and go stale quickly — always check current numbers before acting.*

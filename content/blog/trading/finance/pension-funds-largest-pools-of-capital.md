---
title: "Pension Funds: The Largest Pools of Capital on Earth and the Promises They Must Keep"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A plain-English guide to what a pension fund is, how a promise to pay retirees decades from now becomes a number called a liability, why the discount rate controls everything, and how a quiet corner of finance nearly broke the UK bond market in 2022."
tags: ["pension-funds", "retirement", "liability-driven-investing", "defined-benefit", "defined-contribution", "funded-ratio", "discount-rate", "financial-institutions", "asset-management", "uk-gilt-crisis"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A pension fund is a giant pot of workers' retirement money invested against promises that stretch decades into the future; the size of those promises depends almost entirely on one assumption called the discount rate, and the relentless hunt for returns to keep the promises quietly shapes prices across every market — until, as in the United Kingdom in 2022, the matching goes wrong and the fund's own hedges nearly break the bond market.
>
> - Pension funds hold tens of trillions of dollars worldwide; Japan's GPIF alone manages around \$1.5 trillion, making it the single largest pool of investible capital on the planet.
> - The central distinction is defined benefit versus defined contribution: in a DB plan the employer promises a fixed check and bears the risk; in a DC plan the worker bears it. The world has spent forty years shifting from the first to the second.
> - A pension's promise to pay retirees is a "liability," and its size is computed by discounting future payments back to today; cut the discount rate by one percentage point and the liability can balloon by 10 to 20 percent without a single new promise being made.
> - To control that risk, big DB plans use liability-driven investing — buying long bonds, often with leverage — which worked quietly for years and then, when UK yields spiked in September 2022, triggered a margin-call spiral that forced gilt selling, drove yields higher still, and pulled the Bank of England into an emergency rescue.
> - The number to remember: a saver who puts \$500 a month into a defined-contribution plan for 40 years at a 6% return ends up with roughly \$1 million — but bears every bump in the market along the way, which is the whole point of the DB-to-DC shift.

Here is a question almost nobody can answer correctly. Who is the largest single investor in the world — bigger than any sovereign wealth fund, any hedge fund, any bank's trading book? It is not a Wall Street name. It is a government agency in Tokyo called the Government Pension Investment Fund, the GPIF, which manages around \$1.5 trillion of Japanese workers' retirement money. And GPIF is just one of thousands of pension funds, which together control somewhere north of \$50 trillion globally — more than the annual economic output of the United States, China, and the European Union combined. That ocean of money exists for one mundane, profound reason: ordinary people retire and need an income, and someone promised to provide it. The diagram above is the mental model: a pension fund is a machine that takes small contributions in today, invests them for decades, and pays out a stream of benefits to retirees far in the future — and everything dangerous about pensions lives in the gap between the money it has and the promises it owes.

![Before and after diagram of who bears risk in defined benefit versus defined contribution pensions](/imgs/blogs/pension-funds-largest-pools-of-capital-1.png)

By the end of this post you will understand what a pension fund actually is, the all-important difference between a defined-benefit and a defined-contribution plan, how a promise made today becomes a number called a "liability," why a single assumption — the discount rate — controls the entire game, how the four largest pension funds on earth invest their trillions in wildly different ways, why so many public pensions are chronically underfunded, and how a sleepy technique called liability-driven investing nearly detonated the United Kingdom's bond market in the autumn of 2022. We will work through real dollar arithmetic at every step, and we will name the risk beside every benefit. None of this is investment or retirement advice; it is an attempt to make the largest and least understood pool of money in the world legible.

## First principles: what a pension fund actually is

A **pension** is a promise of income in retirement. A **pension fund** is the pool of money set aside and invested to make good on that promise. Strip away the jargon and that is the whole of it: money collected from working people (and usually their employers) today, invested to grow, and paid back out as a monthly income once those people stop working.

The reason a *fund* is needed at all is timing. Contributions arrive over a worker's career — say from age 25 to 65 — but the benefits are paid out over their retirement — say from 65 to 90. The fund is the reservoir that bridges those two flows. It must take in a steady trickle for forty years, grow it through investing, and then release it as a stream of checks for another twenty-five. That is why a pension fund is, at its core, a long-horizon investor: it has a clearer view of the distant future than almost any other institution, because it knows roughly how many people it must pay, how much, and for how long.

### The pension fund's place among financial institutions

A pension fund sits alongside banks, insurers, mutual-fund companies, hedge funds, and sovereign wealth funds in the larger ecosystem of finance — and it is worth seeing how it differs from each. (For the full map of who's who, see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions).) A bank takes deposits it owes back on demand and makes loans; its liabilities are short and its assets are illiquid, which is why banks are fragile. A pension fund is almost the mirror image: its liabilities are extraordinarily *long* — it will not owe most of its money for ten, twenty, or forty years — which means it can afford to hold illiquid, long-dated assets that a bank never could. That long horizon is the pension fund's single greatest structural advantage. The danger, as we will see, is when a fund forgets that its liabilities are long and starts behaving as if its money is needed tomorrow.

### Contributions, returns, and benefits: the three flows

Three cash flows define a pension fund, and keeping them straight is essential.

- **Contributions** are money flowing *in* — from workers' paychecks, from employers, and sometimes from the government. These are the deposits that fill the reservoir.
- **Investment returns** are the growth of the pool while the money sits invested — interest from bonds, dividends and price gains from stocks, profits from real estate and private companies.
- **Benefits** are money flowing *out* — the monthly checks paid to retirees.

A healthy pension fund is one where contributions plus investment returns, compounded over decades, are enough to cover all the benefits it has promised. An unhealthy one is where they are not — where the promises exceed the resources. The entire discipline of running a pension is the discipline of keeping those three flows in balance across a horizon longer than most companies, and many governments, survive.

![Pipeline of money flowing from contributions through the fund and investments to benefits paid](/imgs/blogs/pension-funds-largest-pools-of-capital-3.png)

The figure above is that machine drawn out: contributions from workers and employers flow into the pool, the pool is deployed into bonds, stocks, and alternative assets, the returns compound over the decades, and the result flows out as benefit checks to retirees. The arrows only balance if the green inflows and compounded returns are large enough to cover the red outflow — and the size of that red outflow is the deepest question in the whole business, because it is a promise about a future that has not happened yet.

### Assets under management (AUM)

The standard way to size any investment institution is by its **assets under management**, or **AUM** — the total market value of everything it holds and invests. When you read that GPIF manages "\$1.5 trillion" or CalPERS "\$500 billion," that is AUM. It is the size of the reservoir at a point in time. But for a pension fund, AUM is only half the story. A fund with \$100 billion in assets sounds enormous until you learn it has promised \$140 billion in benefits. The number that matters is not the assets alone but the assets *measured against the promises* — and to measure the promises, we need to turn a stream of future checks into a single number today.

## How it works: turning promises into a liability

This is the conceptual heart of the entire post, so we will build it carefully from zero. A pension's promise — "we will pay you \$40,000 a year, every year, from retirement until you die" — is a stream of future cash payments. To manage a fund, you must express that whole stream as a single number today: how much money would you need *right now* to be able to fund all those future checks? That number is called the **present value** of the liability, and computing it requires one idea: the **time value of money**.

### The time value of money

A dollar today is worth more than a dollar in a year, because a dollar today can be invested and grow. If you can earn 5% a year, then \$100 today becomes \$105 in a year. Run that backwards: a promise to pay \$105 *next* year is worth only \$100 *today*, because \$100 invested at 5% would produce exactly that \$105. We say the \$105 future payment has a present value of \$100 when **discounted** at 5%.

The arithmetic is mechanical. To find the present value of a payment due in *n* years, you divide it by (1 + r) raised to the power *n*, where *r* is the discount rate:

```present value = future payment / (1 + r)^n
```

A payment of \$1,000 due in 10 years, discounted at 5%, is worth 1,000 / (1.05)^10 = 1,000 / 1.629 = about \$614 today. The further away the payment and the higher the discount rate, the smaller its present value. That single relationship — distant payments discounted at a higher rate are worth less today — is the lever that controls a pension fund's entire reported health. Let us put real numbers on it.

#### Worked example: the present value of a \$40,000-a-year pension for 25 years

Take a worker retiring today who is promised \$40,000 every year for the next 25 years (a simplified single retiree; real funds sum this across millions of people). The fund needs to know: how much money must we hold today to fund this promise, assuming we can earn 5% a year on whatever we set aside?

We discount each of the 25 annual payments back to today and add them up. Discounting a level stream like this is an **annuity** calculation, and the shortcut formula for the present value of \$1 per year for *n* years at rate *r* is:

```PV factor = (1 - (1 + r)^-n) / r
```

Plugging in r = 5% (0.05) and n = 25:

- (1.05) raised to the -25 power is about 0.2953.
- 1 minus 0.2953 = 0.7047.
- Divide by 0.05: 0.7047 / 0.05 = 14.094.

So \$1 a year for 25 years is worth about \$14.09 today. Multiply by the \$40,000 annual payment: 40,000 x 14.094 = about \$563,760.

The fund must hold roughly **\$564,000 today** to fund a \$40,000-a-year pension for 25 years at a 5% discount rate — even though the total dollars eventually paid out are 40,000 x 25 = \$1,000,000. The gap between the \$1 million of nominal payments and the \$564,000 you need today is the work that compounding does over 25 years.

The one-sentence intuition: a pension promise that *sounds* like a million dollars only requires a little over half that today, because the fund expects investment returns to do the rest of the lifting — and that expectation is exactly what can go wrong.

### The funded ratio: the single most important pension number

Now we can define the number that pension watchers obsess over. A fund's **liabilities** are the present value of all the benefits it has promised. A fund's **assets** are the present value of everything it owns. The **funded ratio** is simply assets divided by liabilities, expressed as a percentage:

```funded ratio = assets / liabilities
```

A fund that is **100% funded** has exactly enough assets to cover its promises. A fund that is **80% funded** is short — it has only 80 cents of assets for every dollar of promise. A fund **over 100%** has a surplus. The funded ratio is the pension world's equivalent of a report card, and it is the number that triggers political fights, contribution increases, and benefit cuts.

#### Worked example: a funded ratio of 80%

Take a pension fund — call it a mid-sized US public plan — that holds \$80 billion in assets and has promised benefits with a present value of \$100 billion.

- Assets: \$80 billion.
- Liabilities: \$100 billion.
- Funded ratio: 80 / 100 = **80%**.
- Funding shortfall (the "unfunded liability"): 100 - 80 = **\$20 billion**.

That \$20 billion gap is not an accounting technicality. It is real money the fund does not have but has promised to pay. Closing it requires some combination of three things: higher contributions from workers and the employer, higher investment returns, or — the politically explosive option — reduced benefits. Many large US public pensions sit somewhere between 70% and 85% funded, which is why pension shortfalls are a perennial fixture of state-budget crises.

The one-sentence intuition: the funded ratio is a fraction whose numerator (assets) is buffeted by markets and whose denominator (liabilities) is set by an assumption — so a pension can swing from "healthy" to "in crisis" without any cash changing hands, just because one of those two numbers moved.

### Two risks hiding inside the liability: longevity and inflation

Before we move on, two forces quietly inflate the liability beyond what the discount-rate arithmetic alone captures, and a practitioner thinks about them constantly.

The first is **longevity risk** — the risk that retirees live longer than the fund assumed. Our \$40,000-for-25-years example baked in a 25-year payout. But if people on average live three years longer than projected, the fund owes three extra years of checks to millions of retirees, and the present value of the liability rises accordingly. Across the developed world, life expectancy at 65 has risen steadily for decades, and almost every increase has been a one-way addition to pension liabilities that funds did not fully reserve for. A DB plan that underestimates lifespans is silently underfunded no matter what its assets do.

The second is **inflation**. Many DB pensions — most public ones — are **indexed**, meaning benefits rise each year with the cost of living so that retirees' purchasing power is protected. That indexation is generous to retirees and brutal to the fund: a 2% promised benefit growing at 3% inflation compounds into a far larger nominal payout over a 25-year retirement. A fund with inflation-linked liabilities must therefore hold inflation-linked assets (such as index-linked government bonds) to match them, which is one more reason the safe-bond layer of the portfolio matters so much. When inflation surprised to the upside in 2021 and 2022, indexed pension liabilities jumped — even as the rising interest rates that accompanied that inflation pulled liabilities back down through the discount rate. The two effects partly offset, which is exactly the kind of subtle, opposing-forces calculation that makes pension management genuinely hard.

#### Worked example: how three extra years of life inflate a liability

Return to the \$40,000-a-year promise, valued at about \$564,000 for a 25-year payout at 5%. Suppose mortality tables are updated and the expected payout period rises to 28 years.

- The annuity factor for 28 years at 5% is (1 - (1.05)^-28) / 0.05. (1.05) raised to the -28 power is about 0.2551; 1 minus that is 0.7449; divided by 0.05 gives about 14.898.
- New present value: 40,000 x 14.898 = about **\$595,900**.
- Increase from the original \$564,000: about **\$32,000**, or roughly 6% more, per retiree — for a demographic change, not a richer promise.

Now scale that across a fund paying millions of retirees, and a few extra years of average lifespan can add tens of billions to the liability. The one-sentence intuition: longevity is a slow, relentless tax on every DB pension, because living longer is wonderful for retirees and expensive for the funds that must pay them.

## The discount rate: the most powerful number in pensions

We have used the discount rate casually; now we must confront it directly, because it is the single most consequential — and most contested — number in the entire pension world.

Recall that the liability is the present value of future benefits, and present value depends on the discount rate. Here is the crucial, counterintuitive fact: **a lower discount rate makes the liability bigger, and a higher discount rate makes it smaller.** Why? Because discounting at a high rate assumes you can earn a lot on the money you hold today, so you need less today to reach a given future sum. Discounting at a low rate assumes you can earn little, so you need more today. The discount rate is, in effect, an assumption about future investment returns — and the higher you assume your returns will be, the smaller your promises look today.

This sets up one of the deepest tensions in finance. Pension managers and the politicians who oversee them have every incentive to choose a *high* discount rate, because it shrinks the reported liability, raises the funded ratio, and reduces the contributions the employer must make this year. A more conservative discount rate — closer to the yield on safe government bonds — produces a larger, more honest liability and an uncomfortable funded ratio. US public pensions have historically used discount rates of 7% or higher (tied to their assumed long-run portfolio return), while economists and accounting standards in the private sector argue the rate should reflect the safe, bond-like nature of the promise — often 3% to 5%. The difference is worth trillions of dollars in reported liabilities.

#### Worked example: dropping the discount rate from 7% to 6%

Take a pension that has promised a stream of benefits and currently discounts them at 7%, producing a reported liability of \$100 billion. A board votes to lower the assumed return — and therefore the discount rate — to 6%, judging 7% to be unrealistically optimistic. What happens to the liability?

The effect depends on the *duration* of the liability — roughly, the average number of years until the promises come due. A typical pension liability has a duration of around 15 years. A useful rule of thumb: the percentage change in present value is approximately the duration times the change in rate.

- Change in discount rate: 7% down to 6% = a drop of 1 percentage point (0.01).
- Duration: about 15 years.
- Approximate increase in liability: 15 x 0.01 = 0.15, or **about 15%**.

So lowering the discount rate by a single point inflates the \$100 billion liability to roughly **\$115 billion** — a \$15 billion increase — without anyone promising a single extra dollar of benefits. If the fund's assets were \$85 billion, its funded ratio just fell from 85% (85/100) to about 74% (85/115). One assumption, changed by one point, and the fund went from "a bit underfunded" to "seriously underfunded."

The one-sentence intuition: the discount rate is a dial that lets a pension make its promises look bigger or smaller at will, which is exactly why fights over it are really fights over how much pain to recognize today versus push onto the future.

### Why the discount rate links pensions to the Fed

There is a second, subtler consequence. The "honest" discount rate for a safe promise is tied to the yield on long-term government bonds, and those yields are heavily influenced by central banks. When the Federal Reserve holds interest rates near zero, as it did for much of the 2010s, the safe discount rate falls — and pension liabilities, measured honestly, balloon. (For how central banks move those rates, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).) The decade of ultra-low rates after the 2008 crisis was a slow-motion disaster for pension funding precisely because it inflated liabilities while suppressing the safe returns funds could earn. This is also why, when rates *rose* sharply in 2022, pension liabilities actually *shrank* — a piece of good news that, paradoxically, helped trigger the worst crisis in the sector's history, as we will see.

## The two species: defined benefit vs defined contribution

Every pension plan in the world belongs to one of two families, and the difference between them is the most important thing to understand about retirement finance. It comes down to one question: **who bears the risk?**

A **defined-benefit (DB)** plan promises a specific, *defined* income in retirement — for example, "2% of your final salary for every year you worked." If you worked 30 years and finished on \$60,000, you are promised 60% of \$60,000, or \$36,000 a year, for life, regardless of how the fund's investments performed. The *benefit* is defined and guaranteed; the employer (or the government) must invest, manage, and top up the fund to make good on it. **The employer bears the investment risk and the longevity risk** — the risk that markets disappoint, or that retirees live longer than expected.

A **defined-contribution (DC)** plan promises only the *contributions* that go in — for example, "we will deposit 5% of your salary into your account each year." What that account is worth at retirement depends entirely on how the investments performed. There is no guaranteed income; there is a pot of money that might be large or small. **The worker bears the investment risk and the longevity risk** — if markets crash the year before they retire, or if they live to 100 and the pot runs dry, that is their problem. The famous American **401(k)** plan (named after a section of the tax code) is the archetypal DC plan.

![Tree diagram of pension plan types branching into defined benefit and defined contribution](/imgs/blogs/pension-funds-largest-pools-of-capital-7.png)

The tree above sorts the whole family. The two great branches are DB and DC. Under DB sit *public* DB plans (the state pensions for teachers, police, and civil servants — CalPERS, GPIF) and *corporate* DB plans (company pensions, many of which are now "frozen," meaning closed to new accruals). Under DC sit individual plans like the US 401(k), where each worker manages their own account, and the distinctive Dutch model of *collective* DC, where workers pool risk but bear it collectively rather than the employer guaranteeing it. The risk owner changes as you move across the tree, and that is the only thing about a pension plan that ultimately matters to the person depending on it.

### Why the world shifted from DB to DC

For most of the twentieth century, the DB plan was the norm: you worked for a company or a government for decades, and they promised you a pension for life. Then, over the past forty years, the rich world quietly dismantled that model. In 1980, the overwhelming majority of US private-sector workers with a retirement plan had a DB pension. Today the overwhelming majority have a DC plan, mostly 401(k)s. The shift is one of the great unannounced transfers of risk in economic history.

Why did employers abandon DB? Because DB promises are open-ended liabilities that grow with longevity, fall with interest rates, and must be funded regardless of business conditions. A company carrying a large DB plan is, in effect, also running a pension fund and a bond portfolio on the side, and a bad decade in markets can saddle it with a crippling top-up bill. DC plans eliminate all of that: once the employer makes its contribution, it is done — no open-ended promise, no liability on the balance sheet, no risk. The cost of that relief is borne by the worker, who now faces the full uncertainty of markets and lifespan with no guarantee.

#### Worked example: a DC saver putting \$500 a month for 40 years at 6%

Consider a worker who contributes \$500 a month into a 401(k) from age 25 to 65 — 40 years, or 480 monthly contributions — and earns a steady 6% annual return (0.5% a month). How much do they end up with?

This is a **future value of an annuity** calculation. The formula for the future value of a regular monthly contribution *C* over *N* months at monthly rate *i* is:

```future value = C x [((1 + i)^N - 1) / i]
```

Plugging in C = \$500, i = 0.005 (6% / 12), N = 480:

- (1.005) raised to the 480th power is about 10.957.
- Subtract 1: 10.957 - 1 = 9.957.
- Divide by 0.005: 9.957 / 0.005 = 1,991.5.
- Multiply by \$500: 500 x 1,991.5 = about **\$995,700**.

So \$500 a month for 40 years at 6% grows to roughly **\$1 million** — even though the worker only contributed 480 x \$500 = \$240,000 of their own money. The other ~\$756,000 is compounded investment return. This is the seductive promise of DC: the magic of compounding can turn modest savings into a seven-figure pot.

The one-sentence intuition: a DC plan can build real wealth, but the worker is fully exposed to the path of returns — a 6% *average* with a crash in year 39 produces a very different (and far more painful) result than a smooth 6%, and unlike a DB pensioner, the DC saver eats that risk entirely.

## The giants: how the world's largest pension funds invest

The trillions sitting in pension funds are not managed uniformly. The four largest and most influential funds embody four distinct philosophies, and comparing them is the best way to see the range of what a pension fund can be. (All AUM figures below are approximate and as of recent reporting; they fluctuate with markets and currencies.)

![Matrix comparing GPIF, CalPERS, CPPIB, and ABP by size, type, and style](/imgs/blogs/pension-funds-largest-pools-of-capital-2.png)

The matrix above lays the four side by side. They range from about \$500 billion to \$1.5 trillion in assets, all four are (broadly) defined-benefit plans, but their investment *styles* diverge sharply — from GPIF's near-total reliance on cheap index funds to CPPIB's army of in-house dealmakers. That divergence in style, not in size, is what makes them interesting.

### GPIF (Japan): the \$1.5 trillion index investor

Japan's **Government Pension Investment Fund** is the largest pool of pension assets on earth, managing roughly **\$1.5 trillion** of the retirement savings of Japanese workers. Its scale is hard to overstate: it is so large that it cannot help but *be* the market in many of the assets it touches. If GPIF decides to raise its allocation to foreign stocks by a few percentage points, that single decision moves hundreds of billions of dollars across the globe.

GPIF's philosophy is the opposite of the swashbuckling stereotype. It is overwhelmingly a **passive** investor — it buys broad index funds tracking entire markets rather than trying to pick winners. Its target allocation is famously simple: roughly a quarter each in domestic bonds, domestic stocks, foreign bonds, and foreign stocks. The logic is sound for an institution its size: at \$1.5 trillion, beating the market through clever stock-picking is nearly impossible (you *are* the market), and the fees and risks of active management are not worth it. GPIF's job is not to be brilliant; it is to be cheap, diversified, and to capture the long-run return of global capital markets at minimal cost. When the world's largest investor chooses indexing, it is a powerful endorsement of the idea that, at scale, simple and cheap beats clever and expensive.

### CalPERS and CalSTRS (United States): the public-pension battlegrounds

The **California Public Employees' Retirement System (CalPERS)** and its sibling for teachers (**CalSTRS**) are the largest US public pension funds, with CalPERS managing around **\$500 billion** for roughly two million current and retired California public workers. They are DB plans, and they are the public face of the American public-pension predicament: large, politically governed, chronically short of fully funded, and locked in a perpetual argument over the discount rate.

CalPERS' defining struggle is over its **assumed rate of return**, which doubles as its discount rate. For years it assumed it could earn 7.5% a year; under pressure it cut that to 7%, and then to about 6.8%. Every cut, as our worked example showed, inflates the reported liability and forces California's state and local governments to contribute more — which is why each reduction is a bruising political fight. Pull the assumed return too high and the fund looks healthy on paper while quietly digging a deeper hole; pull it down to an honest level and the funded ratio collapses and contribution bills explode. CalPERS lives permanently between those two bad options, which is the natural condition of a large, mature, politically governed DB plan.

### CPPIB (Canada): the "Canada model" of in-house active investing

The **Canada Pension Plan Investment Board (CPPIB)** manages roughly **\$550 billion** for the national pension plan of Canada, and it pioneered an approach now copied worldwide and known simply as the **"Canada model."** Where GPIF outsources almost everything to cheap index funds, CPPIB does the opposite: it builds large in-house teams of investment professionals who directly buy and manage private assets — entire companies, toll roads, airports, office towers, infrastructure projects — the way a giant private-equity or sovereign-wealth fund would.

The Canada model rests on three pillars: pay competitive (private-sector-level) salaries to attract top investment talent; insulate the fund from political interference through an arm's-length governance structure; and use the fund's enormous size and long horizon to invest directly in illiquid private assets that smaller investors cannot access. The bet is that a pension's decades-long horizon is the perfect match for illiquid assets that pay off slowly, and that doing deals in-house captures returns that would otherwise leak away as fees to external managers. The model has been broadly successful and widely imitated, but it carries real risks: illiquid assets are hard to value and hard to sell in a crisis, in-house teams are expensive, and the more a pension behaves like a private-equity firm, the more it takes on private-equity-style risks. (For how the fee-hungry active-management world works, see [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20).)

### ABP (Netherlands): the Dutch collective model

The Dutch civil-service pension **ABP** manages around **\$500 billion** and represents a third path between pure DB and pure DC. The Netherlands runs one of the best-funded and most sophisticated pension systems in the world, built on **collective** schemes where risk is pooled across all members rather than guaranteed by an employer or borne by lonely individuals. In the traditional Dutch arrangement, benefits are targeted but not absolutely guaranteed: if the collective fund's funded ratio falls below regulatory thresholds, benefits can be trimmed and contributions raised, sharing the pain across the whole membership. This is sometimes called **collective defined contribution (CDC)**, and the Netherlands has been reforming its system further in this direction.

The Dutch model is fascinating because it is an explicit, honest compromise on the risk question. Rather than pretending (as some DB plans do) that benefits are perfectly safe, or dumping all risk onto individuals (as DC plans do), it spreads risk collectively and adjusts benefits transparently when funding falls short. It demands sophisticated funded-ratio rules and a high level of public trust, but it has produced one of the most robustly funded pension systems anywhere. It is the clearest real-world example that the DB/DC binary is not the only option.

### Who pension funds answer to

A pension fund is not a private business chasing profit; it is a **fiduciary**, meaning it is legally bound to act in the sole interest of its members — the workers and retirees whose money it holds. That fiduciary duty is the bedrock of pension governance, and it constrains everything the fund does: it cannot take risks for the manager's benefit, it cannot favor the sponsor over the beneficiaries, and it must invest prudently. But the structures that enforce that duty differ sharply by country, and the differences matter.

US private pensions are governed by a 1974 law called **ERISA** (the Employee Retirement Income Security Act), which sets funding rules and fiduciary standards and created the **Pension Benefit Guaranty Corporation (PBGC)** — a government-backed insurer that pays (capped) benefits if a corporate plan fails, funded by premiums from the plans it covers. US *public* pensions, by contrast, are governed by state law and overseen by politically appointed or elected boards, which is precisely why their discount-rate choices are so vulnerable to short-term political pressure: the people setting the assumption are accountable to voters on a four-year cycle, while the liabilities run for forty. The Canada-model funds deliberately broke that link, placing CPPIB at arm's length from the government with an independent professional board specifically to insulate investment decisions from politics. The single biggest predictor of whether a DB pension is honestly funded is often not its investment skill but its governance — whether the people choosing the assumptions are shielded from the temptation to flatter them.

## How a pension allocates: bonds, equities, and alternatives

Whatever its philosophy, every pension fund must answer the same question: how do we split the pool across different kinds of assets? This is the **asset allocation** decision, and it is where the abstract goal — "earn enough to meet our liabilities" — becomes a concrete portfolio.

![Stacked asset allocation of a large pension fund from government bonds up to alternatives](/imgs/blogs/pension-funds-largest-pools-of-capital-4.png)

The stack above shows a representative allocation for a large DB pension. Read it from the bottom up, because that is how a pension thinks about risk. At the base sit **government bonds** (roughly 35%) — the safest assets, whose steady payments most closely resemble the steady benefit payments the fund owes. These bonds are the **liability-matching** layer: their job is not to maximize return but to behave like the liabilities, so that when liabilities rise or fall, this part of the portfolio rises or falls with them. Above the bonds sits **credit** (roughly 10%) — corporate and private bonds that pay a little more in exchange for a little more risk. Then come **public equities** (roughly 30%) — stocks, the engine of long-run growth, accepted with their volatility because over decades they have delivered the highest returns. At the top sit **alternatives** (roughly 25%) — private equity, real estate, and infrastructure, the illiquid, hard-to-value, higher-fee assets that funds increasingly chase in search of extra return.

### The relentless hunt for yield, and the shift to alternatives

That top layer has grown dramatically over the past two decades, and the reason traces straight back to the discount rate. A US public pension that needs to earn 7% a year to meet its assumed return cannot do it with government bonds yielding 2% or 3%. The math is brutal: if the safe assets yield far below the target return, the rest of the portfolio must reach for far more. So pensions have piled into equities and, increasingly, into **alternatives** — private equity, real estate, infrastructure, private credit, hedge funds — in search of the extra return the assumed rate demands. This is the **hunt for yield**, and it is one of the most important forces in modern markets. Trillions of dollars of pension money flowing into private assets has helped inflate the private-equity and venture-capital industries, lifted real-estate and infrastructure valuations, and quietly shaped asset prices everywhere. When a CalPERS raises its private-equity target, the ripples reach every buyout fund on earth. (For the bird's-eye view of how all these pools connect, see [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system).)

The danger is that the hunt for yield is, structurally, a hunt for risk. A pension that needs 7% in a 3% world is being pushed, by its own discount-rate assumption, into ever-riskier and ever-less-liquid assets. The higher return is real, but so is the higher chance of loss and the difficulty of selling in a crisis. The assumed return, set partly for political convenience, ends up dictating how much risk millions of retirees' money is exposed to.

## Liability-driven investing and the role of leverage

We now arrive at the technique at the center of the 2022 crisis, so we will build it carefully. **Liability-driven investing (LDI)** is an approach in which a pension fund invests with the explicit goal of *matching its liabilities* rather than simply maximizing returns. The insight behind LDI is genuinely clever, and to see it you must remember the discount-rate lesson: a pension's liability behaves like a very long-dated bond. When interest rates fall, the liability swells (lower discount rate, bigger present value); when rates rise, it shrinks. That is exactly how a long-dated bond behaves too — bond prices rise when rates fall and fall when rates rise.

So LDI says: hold long-dated bonds whose value moves in lockstep with the liability. If rates fall and the liability balloons by \$5 billion, the matching bonds also rise by about \$5 billion, and the funded ratio is protected. The fund has *hedged* its single biggest risk — the risk that falling rates inflate its promises. For a fund that cares about its funded ratio above all, this is a beautiful solution. It is why LDI became the dominant strategy for UK corporate pension funds in particular over the 2010s.

### Where the leverage comes in

Here is the catch that turned a prudent hedge into a systemic hazard. A typical UK DB pension might have, say, 50% of its assets in return-seeking equities and growth assets and only 50% available for the bond-matching hedge. But to fully hedge a liability that behaves like 100% of the fund's value in long bonds, it would need *more* long-bond exposure than it has cash for. The solution the industry adopted was **leverage**: instead of buying \$50 of physical gilts (UK government bonds), the fund would use derivatives or repurchase agreements (**repo** — borrowing cash against the bonds as collateral) to get the price exposure of, say, \$100 or \$150 of gilts while only tying up a fraction of cash. This let the fund keep its equities working for growth *and* hedge its full liability with the bond exposure — the best of both worlds, achieved with borrowed money.

This worked quietly for a decade. But leverage always carries the same hidden bill: when the borrowed-against asset moves the *wrong* way, the lender demands more collateral — a **margin call** or **collateral call**. A leveraged LDI fund is fine as long as gilt prices drift gently. If gilt prices *fall* sharply (because yields rise sharply), the fund's leveraged gilt position loses value fast, and its counterparties demand cash collateral immediately to cover the loss. Where does a fund that is fully invested find emergency cash? It sells assets — and the most liquid asset it holds is often the very gilts at the center of the trade. That is the trapdoor, and in September 2022 the UK fell straight through it.

## How it shows up in real markets

The mechanics above are not academic. They have detonated in public, repeatedly. Here are the episodes that show pensions shaping — and occasionally breaking — markets.

### The 2022 UK gilt and LDI crisis

On 23 September 2022, the new UK government announced a "mini-budget" of large, unfunded tax cuts. Markets recoiled, and the yield on long-dated UK government bonds — gilts — spiked violently, rising more in days than it normally would in months. For the legions of UK pension funds running leveraged LDI strategies, this was the precise nightmare the strategy never expected: gilt prices were collapsing, their leveraged hedges were generating enormous mark-to-market losses, and counterparties were issuing collateral calls demanding billions in cash, immediately.

To raise that cash, the funds did the only thing they could: they sold gilts. But mass selling of gilts pushed gilt prices *down further*, which pushed yields *up further*, which generated *more* losses on the leveraged positions, which triggered *more* collateral calls, which forced *more* selling. This is the **doom loop**, a self-reinforcing spiral in which the act of meeting margin calls makes the margin calls worse.

![Graph of the 2022 UK LDI doom loop from yield spike to forced selling to Bank of England rescue](/imgs/blogs/pension-funds-largest-pools-of-capital-6.png)

The graph above traces the loop. A gilt yield spike causes LDI hedges to lose value, which triggers collateral calls, which force gilt selling into a thin market at fire-sale prices, which drives yields higher still — and around again. The loop only breaks when something external intervenes. On 28 September 2022, the Bank of England did exactly that, announcing emergency purchases of long-dated gilts to stabilize prices and stop the spiral. It explicitly framed the intervention as a financial-stability action to prevent a fire sale that threatened the functioning of the gilt market — the market that underpins the entire UK financial system. It was, put simply, a central bank rescuing the pension system from the consequences of the pension system's own hedge.

#### Worked example: the LDI margin-call spiral on a leveraged gilt position

Let us make the doom loop concrete with arithmetic. Suppose an LDI fund has \$10 billion of its own cash but, through repo and derivatives, controls \$30 billion of long gilt exposure — three-to-one leverage. The lenders require the fund to keep posting collateral so that the borrowed exposure stays covered.

- **Starting point:** \$30 billion of gilt exposure, \$10 billion of cash collateral behind it. Long gilts have a duration of roughly 20 years, meaning a 1-percentage-point rise in yields cuts their price by about 20%.
- **Yields rise 1 point:** the \$30 billion gilt position loses about 20% of its value = a **\$6 billion** loss. The counterparties issue a collateral call for roughly that \$6 billion to keep the position covered. The fund's \$10 billion cushion is now down to about \$4 billion.
- **The fund raises cash by selling gilts** — but it is selling into a market where everyone else is doing the same, so prices fall further. Say yields rise *another* 1 point.
- **Second leg:** the remaining position loses roughly another 20% of its (now smaller) value — call it another **\$4.8 billion** loss — triggering another collateral call. The \$4 billion cushion is now wiped out. The fund is forced to sell still more gilts to find cash it does not have.

In a matter of days, a fund that looked perfectly solvent — fully hedged, prudent, conservative — was forced to dump gilts to survive, and its dumping was part of what was driving the losses. Multiply this across the whole UK LDI industry, which collectively ran well over a trillion pounds of such positions, and you have a market-wide cascade that no single fund could escape.

The one-sentence intuition: leverage turns a sound hedge into a forced seller exactly when selling is most damaging, which is how a strategy designed to *reduce* risk ended up nearly breaking the bond market it relied on.

### The same playbook, a different decade: 1992 and 1998

The 2022 episode rhymes with earlier crises where leveraged bets on government bonds and currencies met a sudden move and unwound violently. When George Soros broke the Bank of England in 1992, it was a leveraged bet against a currency peg colliding with a central bank's limits — see [Soros and Black Wednesday](/blog/trading/finance/soros-bank-of-england-1992-black-wednesday) for that mirror image of a central bank pushed past its breaking point. The structural lesson repeats: leverage plus a one-way crowded position plus a sudden price move equals a forced unwind, whether the player is a hedge fund or, in 2022, the entire pension industry.

### CalPERS and the perpetual return-assumption war

CalPERS' decades-long fight over its assumed rate of return is the slow-motion American counterpart to the UK's fast-motion crisis. Each time CalPERS has cut its assumed return — from 7.75% to 7.5% to 7% to about 6.8% — it has triggered immediate, painful increases in the contributions required from California's cities, counties, and school districts, some of which have warned the rising pension bills threaten core services. The fight is fundamentally about honesty versus pain: a higher assumed return is less honest but defers the pain; a lower one is more honest but inflicts it now. There is no escape from the trade-off, only a choice about timing — and because the people making the choice are elected officials with short horizons and the liabilities stretch out for decades, the structural temptation is always to assume a little too much and pass the bill to the future.

### The US public-pension shortfall

Across the United States, state and local public pensions carry a combined unfunded liability measured in the *trillions* of dollars — estimates run from roughly \$1 trillion to several trillion depending on the discount rate used (and remember from our worked example how violently that choice swings the number). Some states, such as Illinois and New Jersey, have funded ratios well below 50% on conservative measures, meaning they hold less than half the assets needed to cover their promises. These shortfalls do not blow up in a single dramatic day like the UK crisis; they grind, year after year, as a structural drag on state budgets, crowding out spending on schools and roads, and occasionally forcing genuine benefit cuts or, in extreme cases like Detroit's 2013 bankruptcy, court-supervised reductions to pensions once thought untouchable. The US public-pension shortfall is the largest slow-moving fiscal problem in the developed world, and it exists precisely because DB promises were made with optimistic discount rates and then under-funded for decades.

### The migration into private assets

The final market-shaping episode is not a crash but a tide. Over the past two decades, pension funds — led by the Canada-model funds and large US plans — have moved trillions of dollars out of public stocks and bonds and into private equity, real estate, infrastructure, and private credit. This migration is the single largest source of capital for the entire private-markets boom. The buyout industry, the explosion of private credit replacing bank lending, the bidding wars over airports and toll roads — much of it is ultimately pension money reaching for the extra return its assumed rates demand. The consequence is that retirees' security is now tied to assets that are illiquid, infrequently priced, and largely opaque. In a benign decade this looks like sophistication; in a severe downturn, when private assets cannot be sold and their stale valuations finally catch up to reality, it could look very different. The hunt for yield that began with a discount-rate assumption has quietly rewired where the world's long-term capital lives.

## Common misconceptions

Pensions are widely misunderstood even by sophisticated people. Here are the wrong beliefs worth correcting, and *why* they are wrong.

**Misconception 1: "A pension fund is just a big savings account."** No. A savings account has no liability — the bank simply owes you your balance. A pension fund's defining feature is its *liability*: a stream of promised future payments whose present value depends on a discount rate. The whole difficulty of pensions — the funded ratio, the discount-rate fights, the LDI hedging — exists *because* there is a liability on the other side of the assets. A pension is assets *minus a promise*, and the promise is the hard part.

**Misconception 2: "My defined-contribution 401(k) is a pension, so my retirement is guaranteed."** It is a pension *plan*, but in a DC plan nothing is guaranteed. You are promised only the contributions; the final value depends entirely on markets and on how long you live. The word "pension" carries a connotation of a guaranteed lifelong income that simply does not apply to DC plans. Confusing the two leads people to under-save and to misjudge how much risk they are personally carrying.

**Misconception 3: "If a pension is 80% funded, it's 80% safe, and a small top-up fixes it."** The funded ratio is computed using a discount rate, and that rate is an assumption, not a fact. An 80%-funded plan on a 7% assumed return might be only 60% funded on an honest, bond-based rate. The reported ratio can flatter a plan that is in deep trouble, and the gap can be far larger than the headline percentage suggests. The ratio is a model output, not a measured truth.

**Misconception 4: "Higher assumed investment returns make a pension stronger."** Exactly backwards in terms of honesty. A higher assumed return is also a higher discount rate, which *shrinks* the reported liability and flatters the funded ratio *on paper* — but it does not put a single extra dollar in the fund. If the optimistic return fails to materialize, the gap was real all along and merely hidden. A higher assumed return makes the plan look stronger while potentially making it weaker.

**Misconception 5: "Leverage in a pension fund is reckless gambling."** Not necessarily — and this is subtle. In LDI, leverage is used to *hedge*, to make the assets track the liabilities more closely, which can genuinely *reduce* the fund's overall risk in normal conditions. The 2022 crisis was not caused by leverage being inherently reckless; it was caused by leverage creating a hidden need for emergency cash that the funds could not meet without becoming forced sellers in a crowded, falling market. Leverage was a prudent tool with a fatal failure mode under stress.

**Misconception 6: "Pension funds are passive bystanders in markets."** With over \$50 trillion in assets, pension funds are among the most powerful price-setting forces in the world. Their asset-allocation decisions move whole markets, their hunt for yield inflated the private-markets boom, and their forced selling in 2022 nearly broke the UK gilt market. They are not bystanders; they are, collectively, one of the largest movers of capital on earth.

## When this matters to you / further reading

If you have a retirement plan at work — and especially a 401(k) or any defined-contribution arrangement — then everything in this post is about *your* money, and the single most important thing to internalize is that **you, not your employer, bear the risk.** That reframes the choices you make: how much to contribute, how aggressively to invest as you age, and how to think about the path of returns rather than just the average. The DC saver in our worked example reached \$1 million on a smooth 6%, but real markets are not smooth, and a crash near retirement can be devastating to someone with no employer guarantee behind them. Understanding that you carry the longevity risk and the market risk is the beginning of taking it seriously.

If you are a citizen of a place with large public pensions — most US states, most rich countries — then the discount-rate fights and funded-ratio reports are arguments about your future taxes and public services. When a CalPERS cuts its assumed return, the bill lands on local budgets; when a state runs a 50%-funded plan, that shortfall is a claim on tomorrow's taxpayers. The discount rate is not a technicality; it is a political choice about who pays and when, dressed up as an actuarial assumption.

And if you simply want to understand how markets move, pensions are an under-appreciated key. The hunt for yield driven by optimistic return assumptions has reshaped where the world's capital flows, inflating private markets and bidding up illiquid assets. And the 2022 UK episode is the cleanest modern case study of how a strategy designed to reduce risk can, through leverage and crowding, become a systemic hazard that pulls a central bank into an emergency rescue — the same structural pattern that recurs across financial history whenever leverage meets a one-way crowded trade and a sudden move.

To go deeper into the surrounding system, three companion pieces fit naturally beside this one: the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) places pensions in the full ecosystem of banks, insurers, and funds; [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) explains the rate machinery that drives discount rates and pension liabilities; and [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20) covers the leveraged, fee-driven active-management world that the Canada-model pensions increasingly resemble. Read together, they reveal a single truth: the largest, slowest, most patient pool of money in the world is also, when the matching between its assets and its promises goes wrong, one of the most dangerous.

![Timeline from DB dominance through the DC shift to the 2022 LDI crisis and Bank of England rescue](/imgs/blogs/pension-funds-largest-pools-of-capital-5.png)

The timeline above is the forty-year arc in one image: from a world of defined-benefit promises, through the slow migration to defined contribution that shifted risk onto workers, to the LDI-and-leverage build-up of the 2010s, and finally to the four-day crisis of September 2022 that ended with the Bank of England buying gilts to save the system. The promises a pension fund makes are measured in decades; the crises that test them arrive in days. Holding both timescales in mind at once is what it means to understand the largest pools of capital on earth.

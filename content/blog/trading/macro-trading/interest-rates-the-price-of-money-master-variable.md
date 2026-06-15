---
title: "Interest Rates: The Price of Money and the Master Variable Behind Every Asset"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into why an interest rate is the price of money and time, how discounting puts that one number underneath every asset price, and how to map the rate sensitivity of everything you own."
tags: ["interest-rates", "macro", "monetary-policy", "discounting", "present-value", "yield-curve", "duration", "risk-free-rate", "bonds", "federal-reserve", "valuation", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An interest rate is the price of money and time, and because the risk-free rate is the discount rate sitting under every asset price, when rates move, everything reprices at once.
>
> - A rate is the rent on money: what you pay to use someone else's cash for a year, or what you earn for lending yours. **The policy rate is the one the Fed pins; market rates are everything the world prices off it.**
> - Every asset is a claim on future cash flows, and every future dollar is discounted back to today by a rate. Raise the rate and the present value of the future shrinks — which is why **2022 saw stocks, bonds, and crypto fall together** as the Fed hiked from 0.25% to 5.50%.
> - **Duration** measures how much a price moves per 1% change in rates. A 10-year bond drops roughly 8-9%; a profitless long-duration tech stock or a no-cash-flow crypto token drops far more. Map every position's duration and you know who gets hurt when rates rise.
> - The one number to remember: in 2022 the Fed ran the **fastest hiking cycle in 40 years**, +525 basis points in about sixteen months, and a classic 60/40 stock-bond portfolio lost about 16% in a single year — the discount rate is the tide, and that year it went out for everyone at once.

In October 2021 you could borrow money almost for free. The Fed's policy rate was pinned near zero, the 10-year Treasury — the bedrock interest rate of the global financial system — yielded about 1.5%, and a generation of investors had been trained by a decade of cheap money to believe that the only direction for asset prices was up. Profitless software companies traded at twenty times revenue. A picture of a cartoon ape sold for the price of a house. Long-dated government bonds, the supposedly "safe" asset, had been bid up to prices that made sense only if rates stayed at the floor forever.

Then inflation arrived, and the Federal Reserve did something it had not done in four decades: it raised the price of money, fast. Between March 2022 and July 2023 the Fed lifted its policy rate from a target range topping out at 0.25% all the way to 5.50% — a +525 basis point move, the fastest tightening cycle since Paul Volcker's war on inflation in the early 1980s. And as that one number climbed, something happened that surprised a lot of people who thought they understood diversification: *everything fell at the same time.* The S&P 500 dropped about 19% over 2022. Long-term Treasuries fell more than 30%. Bitcoin lost about two-thirds of its value. The classic 60/40 portfolio — 60% stocks, 40% bonds, the textbook "balanced" allocation that is supposed to have one leg up when the other is down — had one of its worst years in a century, falling roughly 16%, because both legs fell together.

That synchronized selloff was not a coincidence and it was not bad luck. It was the single most important lesson in all of macro made visible in real time: **an interest rate is the price of money and time, and that price is the discount rate that sits underneath every asset on earth.** When it moves, every price tied to a future cash flow has to move with it. Understand rates and discounting and you are not holding one fact about bonds — you are holding the single thread that connects bonds, stocks, currencies, real estate, and crypto. This post builds that understanding from absolute zero, then hands you the playbook for reading it.

![Risk-free rate at the center with every asset class discounted by it](/imgs/blogs/interest-rates-the-price-of-money-master-variable-1.png)

## Foundations: what a rate actually is, present value, and the yield curve

Before any trading, we need three ideas, each built from scratch: what an interest rate *is*, what *present value* means, and what the *yield curve* shows. Everything else in this post is a consequence of these three.

### An interest rate is rent on money

Forget finance for a second and think about renting an apartment. You pay rent because you are using something — living space — that belongs to someone else, and the owner wants compensation for letting you use it instead of using it themselves. An interest rate is exactly that, except the thing being rented is *money*.

When you take out a loan, you are renting money. The lender hands you cash today and you agree to give it back later plus a fee. That fee, expressed as a percentage per year, *is* the interest rate. If you borrow \$100 for a year at a 5% interest rate, you pay back \$105: the \$100 you rented plus \$5 of rent. Flip it around and the same thing is true from the lender's side: if you put \$100 in a savings account paying 5%, the bank is renting your money and paying you \$5 a year for the privilege.

So at its most basic, an interest rate is **the price of borrowing money over a period of time.** It has two ingredients baked into it, and naming them both is the whole game:

- **The price of time.** Even with zero risk, a dollar you have now is worth more than a dollar you get in a year, because you could use the now-dollar — invest it, spend it, earn rent on it. Time itself has a price.
- **The price of risk.** A dollar promised by the U.S. Treasury is more certain than a dollar promised by a shaky startup. The riskier the promise, the more rent the lender demands. That extra rent is a *premium*.

The cleanest way to think about it: an interest rate is what the world charges you to move money across time. Borrowing pulls future money into the present (and you pay rent for the privilege). Lending or saving pushes present money into the future (and you collect rent). The rate is the exchange rate between *now* and *later*.

### The policy rate versus market rates

Here is the first thing almost everyone gets wrong: there is not one interest rate. There are thousands. The rate on your mortgage, the rate on a 3-month Treasury bill, the rate on a junk bond, the rate your savings account pays, the rate two banks charge each other overnight — these are all different numbers, and they move somewhat independently.

But they are not unrelated. They are organized around one anchor: **the policy rate**, the single overnight rate that the central bank — in the U.S., the Federal Reserve — directly controls. In America this is the **federal funds rate**, the rate banks charge each other to borrow reserves overnight. The Fed does not set your mortgage rate or the 10-year Treasury yield. It pins this one overnight number, and every other rate in the economy is priced as some spread above (or, rarely, below) it, adjusted for how long the loan lasts and how risky it is. (How the Fed actually pins that one number — paying interest on reserves as a ceiling and offering reverse repos as a floor — is its own deep topic; see the cross-link at the end on [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

So the mental model is a tree. At the root: the policy rate, set by committee. Growing out from it: market rates, set by millions of buyers and sellers who take the policy rate as their starting point and add premiums for time and risk. When the root moves, the whole tree sways — but the branches also bend on their own, blown by the market's own expectations about growth, inflation, and fear.

### Present value: a dollar later is worth less than a dollar now

This is the most important idea in finance and most people never have it explained cleanly, so here it is, slowly.

You are offered a choice: \$100 today, or \$100 in one year. Everyone picks today. Why? Because with the \$100 today, you could put it in a savings account at, say, 5% and have \$105 in a year — so \$100-today is strictly better than \$100-in-a-year. The dollar in the future is *worth less to you right now.*

Now turn that around into a question. If \$100 today grows to \$105 in a year at 5%, then how much money today would grow to exactly \$100 in a year? The answer is \$100 / 1.05 = \$95.24. So **a promise of \$100 in one year is worth \$95.24 to you today**, at a 5% rate. That \$95.24 is called the **present value** (PV) of that future \$100. We discounted the future amount back to today by dividing by (1 + the rate).

For a single cash flow `N` years out, the formula is just that division compounded once per year:

```
PV = future_amount / (1 + r) ** N
```

where `r` is the annual interest rate (the **discount rate**) and `N` is the number of years. The `(1 + r) ** N` in the denominator is the same compounding that grows your savings account, run in reverse: it shrinks future money back to a today-value.

The single most useful intuition in this entire post lives in that formula: **the higher the discount rate `r`, the smaller the present value.** A bigger denominator means a smaller number. And the further out the cash flow (the bigger `N`), the more violently the present value shrinks when `r` rises, because `r` gets compounded more times. Hold that thought — it is the seed of everything about duration and why long-dated assets get crushed when rates rise.

The reason the denominator has that exponent `N` rather than a simple multiplication is **compounding**, and compounding is what makes time so powerful in finance. Simple interest would just add the rate once per year linearly: \$100 at 5% simple for 10 years would be \$100 + 10 × \$5 = \$150. But real money compounds — you earn interest on your interest. \$100 at 5% *compounded* for 10 years is \$100 × 1.05^10 = \$162.89. That extra \$12.89 is interest earned on previously-earned interest, and over long horizons the gap between simple and compound growth explodes. Run that compounding backwards — which is exactly what discounting does — and the same exponent that makes money grow faster than you expect makes future money worth *less* than you expect. The asymmetry of compounding cuts both ways: it is your friend when you are saving and the market's weapon against long-dated assets when rates rise. The further out a cash flow sits, the more years of compounded discounting it endures, which is the entire reason a 30-year promise is so much more rate-sensitive than a 2-year promise. Time is not linear in finance; it is exponential, and the interest rate is the base of that exponent.

#### Worked example: the present value of \$100 in 10 years at 1% versus 5%

Take a promise of \$100 to be paid in exactly 10 years. What is it worth today?

At a 1% discount rate:

```
PV = 100 / (1.01 ** 10) = 100 / 1.1046 = $90.53
```

At a 5% discount rate:

```
PV = 100 / (1.05 ** 10) = 100 / 1.6289 = $61.39
```

The future cash flow did not change — it is \$100 in both cases. The *only* thing that changed is the discount rate, from 1% to 5%. And that single change knocked the present value down from \$90.53 to \$61.39, a drop of about 32%. **Nothing happened to the company, the bond, or the economy in this example — only the rate moved, and a third of the value evaporated.** That is the engine that powered the entire 2022 selloff, scaled across every asset on the planet.

![Present value of one hundred dollars in ten years falling as the discount rate rises](/imgs/blogs/interest-rates-the-price-of-money-master-variable-3.png)

### The yield curve: rates for different lengths of time

We said there are many rates. One of the cleanest ways to organize them is by *how long* the loan lasts. If you lend money to the U.S. government, you can do it for 3 months, 2 years, 10 years, or 30 years, and each of those gets a different rate. Plot those rates — yield on the vertical axis, time-to-maturity on the horizontal axis — and you get the **yield curve.**

Normally the curve slopes *upward*: longer loans pay higher rates, because lending for 30 years means more uncertainty (inflation could erode your money, the borrower's situation could change) than lending for 3 months, so you demand more rent. That extra rent for length is the **term premium.**

Sometimes the curve slopes *downward* — short rates are higher than long rates. This is an **inverted yield curve**, and it is one of the most-watched recession signals in all of macro, because it means the market expects the central bank to cut rates in the future (which only happens when the economy weakens). The yield curve was deeply inverted through 2022 and 2023 as the Fed hiked the short end hard while the long end priced in eventual cuts.

For the purposes of this post, the one number on the curve to obsess over is the **10-year Treasury yield.** It is long enough to reflect the market's view of growth and inflation, liquid enough to be the world's reference rate, and it is the discount rate that most directly sits underneath stocks, mortgages, and risk assets. When people say "rates went up" and mean it for the stock market, they almost always mean the 10-year moved.

It helps to see the yield curve as the market's answer to two separate questions stacked on top of each other. The first question is *where will the Fed set the overnight rate, on average, over the life of this bond?* — pure expectations about future policy. The second is *how much extra do I demand for locking my money up that long, given the uncertainty?* — the term premium. A 10-year yield of, say, 4% might be "the market expects the average overnight rate to be 3.5% over the next decade, plus a 0.5% term premium for the risk of holding ten years." This decomposition is why the curve can move even when the Fed does nothing: if the market suddenly expects higher inflation, the expectations piece of every long yield rises, and the whole long end of the curve lifts without a single Fed meeting. The Fed nails down today's overnight rate; the curve is the market pricing every future overnight rate plus the rent for time.

The shape of the curve is one of the most information-dense pictures in macro. An **upward-sloping** curve (long rates above short) is the normal, healthy state: the economy is expected to grow, the Fed is expected to keep rates roughly where they are or higher, and lenders earn a term premium for going long. A **flat** curve says the market is unsure. An **inverted** curve (short rates above long) is the alarm bell: it says the market expects the Fed to be *cutting* rates in the future, which historically only happens when a recession forces the Fed's hand. The spread between the 10-year and the 2-year Treasury yield, and the spread between the 10-year and the 3-month, are the two most-watched recession indicators in the business precisely because an inversion has preceded nearly every U.S. recession of the past half-century. Through much of 2022 and 2023 that spread was deeply negative — the curve was inverted by the most in four decades — as the Fed jacked the short end above 5% while the long end stayed lower in anticipation of eventual cuts. That inversion was the bond market's way of saying "the Fed is hiking us into a slowdown," and it is a signal every macro trader reads first thing in the morning.

## Discounting: why a dollar later is worth less than a dollar now

We have the formula. Now let us internalize *why* discounting is the mechanism that links one rate to the whole market, because this is the conceptual heart of the post.

Every asset — every single one — is a claim on future cash flows. A bond is a claim on fixed coupon payments plus the return of your principal. A stock is a claim on a company's future earnings (paid out as dividends or reinvested to grow those earnings). A rental property is a claim on future rent. Even a no-yield asset like gold or a non-dividend crypto token is, implicitly, a claim on a future *price* — what someone will pay you for it later.

To value any of these, you do the same thing every time: you take all those future cash flows, discount each one back to today using the appropriate rate, and add them up. That sum is the asset's present value, which is what it is "worth." Write the whole thing out and it is just our PV formula, summed over every future payment:

```
Value = CF_1/(1+r) + CF_2/(1+r)**2 + CF_3/(1+r)**3 + ... 
```

where `CF_n` is the cash flow in year `n` and `r` is the discount rate. **This single equation is the deepest idea in markets.** It says the value of anything is its future cash flows divided by a discount rate. And it tells you immediately: if `r` goes up, *every* term in that sum shrinks, so the *value of everything* shrinks. There is no asset that escapes the denominator.

The image to fix in your head: a future cash flow is a fixed size, but the discount rate is a magnifying glass held in reverse — turn it up, and the same future amount looks smaller today. The further in the future the cash flow, the more powerful that shrinking effect, because the rate compounds against it more times.

![Same future cash flow shrinking as the discount rate rises](/imgs/blogs/interest-rates-the-price-of-money-master-variable-4.png)

#### Worked example: a perpetual cash flow repricing when the discount rate jumps from 2% to 5%

Some assets pay forever — or close enough. A stable, mature company paying a steady dividend, a piece of farmland generating steady rent, a perpetual bond (a "consol"). For a cash flow that goes on forever, the math simplifies beautifully. The present value of \$1 per year, forever, discounted at rate `r`, is just:

```
Value = annual_cash_flow / r
```

This is the **perpetuity formula**, and it is the cleanest way to see rate sensitivity, because the rate is the *entire* denominator with no compounding to muddy it. Suppose you own an asset paying \$10,000 a year, forever. Discount it at 2%:

```
Value = 10,000 / 0.02 = $500,000
```

Now the discount rate rises to 5%. The cash flow has not changed — still \$10,000 a year. But the value:

```
Value = 10,000 / 0.05 = $200,000
```

The asset just lost **60% of its value** — \$500,000 to \$200,000 — purely because the discount rate went from 2% to 5%. Nothing about the cash flow changed. This is why long-duration assets (assets whose value comes mostly from cash flows far in the future) are so brutally sensitive to rates. **A perpetuity is the most rate-sensitive asset there is, and it shows you the mechanism in its purest form: value moves inversely with the discount rate, hard.**

A small refinement that matters for stocks: most companies grow their cash flows over time. The version of the perpetuity formula that accounts for a constant growth rate `g` is the **Gordon growth model**: `Value = cash_flow / (r − g)`. The intuition is the same but more extreme — a high-growth company's value lives in the gap `(r − g)`, and when `r` rises while `g` stays put, that gap widens and the value collapses even faster than for a no-growth asset. That is precisely why unprofitable, fast-growing tech (high `g`, all the cash flows far in the future) got annihilated in 2022 while a boring high-dividend utility (low `g`, cash flows soon) barely flinched.

## The risk-free rate: the foundation of all valuation

We keep saying "the discount rate." Where does it come from? It is built in layers, and the bottom layer is the most important number in finance: the **risk-free rate.**

The risk-free rate is the return you can earn with essentially zero chance of losing your money. In practice this means lending to the U.S. government, because the U.S. Treasury can always pay back dollars (it issues them), so a U.S. Treasury bond is treated as the closest thing to a risk-free promise that exists. The yield on short-term Treasuries is the *risk-free rate* for short horizons; the 10-year Treasury yield is the risk-free rate for long horizons.

Why does this one rate matter so much? Because it is the **opportunity cost of all money.** If you can earn 5% risk-free by lending to the government, then you will not accept less than 5% on anything riskier — you would just buy the Treasury instead. So every risky asset must offer *more* than the risk-free rate to compensate you for the risk. The risk-free rate is the floor under all required returns, the baseline every other asset is measured against. When the risk-free rate rises, the floor rises, and every asset above it has to reprice to clear that higher bar.

This is why the discount rate `r` in our valuation formula is never just one number — it is the risk-free rate plus a stack of premiums, each compensating you for a specific risk you take by holding something other than a Treasury bill:

- **The risk-free rate** — pure price of time, the foundation.
- **+ Term premium** — extra rent for lending longer (locking your money up for years means more inflation and rate uncertainty).
- **+ Credit premium** — extra rent for lending to a borrower who might default (a corporation, a homeowner, a government that is not the U.S.). This is the "spread" over Treasuries.
- **+ Liquidity premium** — extra rent for holding something hard to sell quickly (a thinly traded bond, real estate, private equity).
- **+ Equity risk premium** — for stocks specifically, the extra return demanded for bearing the wild uncertainty of owning a business rather than lending to it.

Each asset's discount rate is the risk-free rate plus whichever of these premiums apply. A 10-year corporate bond's discount rate is the 10-year Treasury yield (risk-free + term) plus a credit spread. A stock's discount rate is the risk-free rate plus the equity risk premium. **Here is the punchline that makes the risk-free rate the master variable: the premiums are relatively stable, but the risk-free rate moves with monetary policy — so when the Fed shifts the risk-free rate, it shifts the bottom of every single one of those stacks at once.** Raise the foundation and every floor above it rises, every present value above it falls. One number, the whole building.

![Ten year Treasury yield from 2020 to 2026 with the 2020 low and 2023 high marked](/imgs/blogs/interest-rates-the-price-of-money-master-variable-5.png)

The chart above is the risk-free rate (the 10-year Treasury) doing exactly this over the cycle we are studying. In July 2020 it bottomed at 0.62% — the discount rate under the entire market was almost zero, which is why asset prices were so high (divide future cash flows by something near zero and you get something near infinity). By October 2023 it had climbed to 4.88%, a 16-year high. That rise in the foundation, four percentage points, is the same four-point move that in our worked examples cut present values by a third to two-thirds. Scale that across every stock, bond, and house in the country and you have the macro story of 2022-2023.

## Everything is a bond: how rates transmit to stocks, FX, real estate, and crypto

Now we connect the master variable to each asset class. The unifying claim — the title of the cover figure — is that **everything is a bond**: every asset is a stream of future cash flows discounted by a rate, which is exactly what a bond is. The differences between asset classes are differences in *how far out* the cash flows are and *how certain* they are, which is to say differences in duration and risk premium. But the discounting machinery is identical. Let us walk the transmission, asset by asset.

![Policy rate fanning out through the Treasury curve into bonds stocks credit housing currencies and crypto](/imgs/blogs/interest-rates-the-price-of-money-master-variable-6.png)

### Bonds: the purest case

A bond *is* a stream of cash flows: fixed coupons every period and the face value at maturity. When market rates rise, newly issued bonds pay those higher rates, which makes your old lower-coupon bond less attractive — so its price falls until its yield matches the new market rate. The relationship is mechanical and exact: **bond prices move inversely to yields.** A bond is rate sensitivity in its most transparent form, which is why we study it first and use it as the template for everything else. The longer the bond's maturity, the bigger the price move for a given rate change — that is duration, which gets its own section.

It is worth making the inverse relationship concrete because so many beginners find it counterintuitive. Suppose you bought a bond last year that pays a fixed \$3 coupon per \$100 of face value — a 3% yield, which was the going rate at the time. Now market rates have risen and brand-new bonds pay \$5 per \$100 — a 5% yield. Nobody will pay you \$100 for your old bond that only pays \$3 when they can buy a new one paying \$5 for the same \$100. So the price of your bond has to fall — to roughly \$60-something — until its \$3 coupon represents a competitive 5% yield on the *lower* price. You did not do anything wrong; the rate rose and your bond's price fell to keep its yield in line with the market. This is the entire bond market in miniature: a fixed coupon, a price that moves opposite to the prevailing rate, and a yield that the two together must keep consistent with the world. Every other asset in this section is a more complicated version of this same trade — a fixed-ish future cash flow whose present value moves opposite to the discount rate.

### Stocks: long-duration claims on earnings

A stock is a claim on a company's earnings stretching out indefinitely into the future — effectively a perpetuity that grows. Using the same discounting machinery, a stock's value is the present value of all its future earnings. So when the discount rate rises, that present value falls, *for the exact same reason a bond's does.* The catch: how *much* a given stock falls depends entirely on *when* its earnings arrive.

- A mature, profitable, dividend-paying company (think a consumer-staples giant) has most of its value in earnings happening *soon*. Its cash flows are "short duration." A higher discount rate stings but does not devastate.
- A high-growth, currently-unprofitable company (think a speculative software or biotech name) has *almost all* of its value in earnings expected far in the future — it is losing money today and the story is entirely about 2035. Its cash flows are extremely "long duration." A higher discount rate, compounded over all those years, is catastrophic.

This is exactly why 2022 saw a violent **rotation from long-duration growth into short-duration value.** The Nasdaq-100, packed with long-duration growth names, fell about 33% in 2022 while the more value-tilted Dow Jones fell only about 9%. Same market, same rate move — but the long-duration assets, whose cash flows live furthest in the future, got discounted the hardest. *Long-duration tech is just a very long bond wearing a different costume.*

### Currencies (FX): rate differentials pull capital

Money flows toward higher returns. If U.S. rates rise while Japanese rates stay near zero, global capital moves into dollars to capture the higher yield — selling yen, buying dollars, pushing the dollar up. So **the dollar tends to strengthen when U.S. rates rise relative to the rest of the world.** This is the "rate differential" or "carry" channel. In 2022, as the Fed hiked aggressively while the Bank of Japan held rates pinned at zero, the dollar index (DXY) surged to a peak near 114.8 in September 2022, its highest in two decades, and the yen collapsed from about 115 to over 150 per dollar. Rates did not just move stocks and bonds; they moved the price of money *against other monies*. (The dollar's special role as the world's reserve currency amplifies all of this — covered in the cross-linked piece on the [dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).)

The mechanism here is subtle enough to be worth one more pass, because currencies are where the "price of money" framing is most literal. An exchange rate *is* the relative price of two monies, and an interest rate *is* the price of holding one money over time. So a currency pair is really a comparison of two interest rates plus expectations about where those rates are headed. When a global investor decides where to park cash, they compare the yield on dollars (the U.S. rate) against the yield on yen (the Japanese rate) and lean toward the higher one — but they also have to weigh the risk that the higher-yielding currency *depreciates* and gives back the yield advantage. In calm markets, the rate differential wins and capital piles into the high-yielder; this is the famous **carry trade**, borrowing in a low-rate currency to invest in a high-rate one. The catch is that carry trades are short volatility: they earn a steady drip of yield differential and then, every few years, blow up violently when the high-yielder suddenly weakens, as happened to the yen carry trade unwind in 2024. For a macro trader the lesson is direct — when one central bank is hiking and another is holding, the rate differential is a tailwind for the hiker's currency, and the trade works right up until the differential narrows or the market panics. The currency, like every other asset in this post, is just rates wearing a different costume.

### Real estate: rates are the monthly cost

Most property is bought with a mortgage, and a mortgage payment is overwhelmingly a function of the interest rate. When the 30-year mortgage rate jumps from 3% to 7% — which is roughly what happened from 2021 to late 2023 in the U.S. — the monthly payment on the same-priced house roughly doubles. Buyers can suddenly afford far less house for the same monthly budget, so either prices fall or transactions freeze. In 2022-2023, U.S. existing-home sales fell to their lowest level in nearly thirty years, not because anyone's job disappeared, but because the *price of the money used to buy houses* roughly doubled. Real estate is a bond too: its "cash flow" is the rent or the imputed value of living there, and its price is set by what monthly payment a buyer can afford at the prevailing rate.

### Crypto: the longest duration of all

Bitcoin and most crypto pay no cash flows at all. There is no coupon, no dividend, no rent — the entire value is a bet on a *future price.* In discounting terms, that is the longest-duration asset imaginable: all of the "cash flow" is at some indefinite point far in the future, so the discount rate hits it with maximum force. On top of that, crypto is the textbook "risk-on" speculative asset, the thing people buy with the *cheap, abundant money* that exists when rates are near zero. Raise rates — make safe money pay 5% — and the appetite for a no-yield speculative bet drains away. Both channels (long duration plus risk appetite) point the same direction, which is why Bitcoin fell about 65% in 2022, far more than stocks, in lockstep with the rate move. **When rates are the master variable, crypto is the most leveraged bet on rates staying low, even though it has nothing to do with bonds.**

That is the whole map. One rate, six asset classes, one mechanism. Now the tool that quantifies *who gets hurt most*.

## Duration: who gets hurt most when rates rise

We have used the word "duration" loosely. Let us pin it down, because it is the single most useful number for measuring rate risk.

**Duration is a measure of how sensitive an asset's price is to a change in interest rates.** Formally it is approximately the percentage change in price for a 1% (100 basis point) change in yields. A bond with a duration of 10 will fall roughly 10% if rates rise by 1%, and rise roughly 10% if rates fall by 1%. The handy approximation is:

```
percent_price_change = - duration * change_in_yield
```

The minus sign encodes the inverse relationship: yields up, prices down. The bigger the duration, the bigger the move for a given rate change. Duration is, intuitively, a weighted average of *how far in the future* an asset's cash flows arrive — which is why it lines up perfectly with everything we just said about stocks, real estate, and crypto. Cash flows far in the future = long duration = high rate sensitivity. The word "duration" literally means "how long until you get your money," and that length is exactly what determines the pain.

![Bond price drop for a one percent yield rise across two five ten and thirty year durations](/imgs/blogs/interest-rates-the-price-of-money-master-variable-7.png)

The chart makes the linearity vivid: a 2-year-duration bond barely moves (about −2% for a 1% rate rise), a 10-year drops about 10%, and a 30-year-duration bond gets demolished, falling roughly 30%. Same rate move, wildly different damage, all governed by one number: duration.

#### Worked example: a 10-year bond falling about 8-9% when the 2022 yield jumps

Let us ground duration in the real 2022 move. At the start of 2022 the 10-year Treasury yielded about 1.5%; by October 2022 it had climbed to about 4.05% — a rise of roughly 2.5 percentage points (250 basis points) in well under a year. A 10-year Treasury bond has a duration of roughly 8.5 years (slightly less than its maturity because the coupons return some money sooner). Plug into the approximation for the first 1% of that move:

```
percent_price_change = - 8.5 * 0.01 = -0.085 = -8.5%
```

So a +1% jump in the 10-year yield knocks roughly 8.5% off the price of a 10-year Treasury — the supposedly "safe" asset. But the 2022 move was about +2.5%, not +1%, so the damage was roughly `−8.5 × 2.5 ≈ −21%` just on the 10-year, and longer Treasuries (20- and 30-year, with durations near 17-19) fell over 30%. **The "safe" leg of the portfolio fell harder than it had in modern history, not because the U.S. might default, but because its duration ran headfirst into the fastest rate rise in 40 years.** This is the precise mechanism behind the 2022 bond crash — and behind the bank failures of March 2023, when banks like Silicon Valley Bank discovered their long-duration Treasury and mortgage portfolios had quietly lost a fortune to exactly this math.

Duration is the variable that turns "rates went up" into "this specific position lost this specific amount." Memorize it as the rate-pain coefficient: multiply the rate move by the duration and you have the price hit. Cash and very short bills have near-zero duration (they barely move). Long bonds, long-dated growth stocks, and no-cash-flow crypto have enormous effective duration (they move a lot). Your job as a trader is to know the duration of everything you own.

## Common misconceptions

A handful of beliefs about rates are widespread, intuitive, and wrong. Each one costs people money. Correct each with a number.

### "Interest rates only matter for bonds"

This is the costliest misconception, and the entire post so far is the rebuttal. Rates are the discount rate under *every* asset, so they matter for stocks, real estate, currencies, and crypto just as much as bonds — often *more*, because those assets carry longer effective duration. The proof is 2022: the asset that fell the most in the great rate selloff was not a bond, it was **Bitcoin, down about 65%** — an asset with no coupons, no relationship to the bond market on its surface, and yet the single most rate-sensitive thing in the entire market, because it is the longest-duration, most risk-on bet there is. If you think rates are a "bond thing," you did not understand why your tech stocks and your crypto fell together in the same year.

### "The Fed sets all interest rates"

The Fed sets *one* rate — the overnight policy rate — directly. It does not set the 10-year Treasury yield, the mortgage rate, or the corporate bond rate; those are set by the bond market, which trades on its own expectations of future growth and inflation. The proof: in late 2024 the Fed *cut* its policy rate by a full percentage point (from 5.50% to 4.50% over three meetings), and yet the 10-year Treasury yield *rose* over the same window, from about 3.78% in September 2024 to about 4.58% by December, because the bond market got more worried about inflation and deficits. **The Fed steers the short end; the market steers the long end, and the long end is the one that discounts your stocks.** Confusing the two will get you positioned exactly wrong.

### "Low rates are always bullish"

Low rates lift asset prices through the discounting channel — that part is true. But low rates are not a free lunch, for two reasons. First, rates are usually low *because the economy is weak or in crisis* — the Fed slashed to zero in 2008 and 2020 precisely because the world was on fire, and "rates are at zero" coincided with the worst of those crashes, not the top of a boom. Second, persistently low rates inflate bubbles that then burst spectacularly when rates normalize — the everything-bubble of 2021, built on near-zero rates, produced the everything-crash of 2022 when rates rose. **"Low rates are bullish" is true on the way down and a trap on the way back up.** The direction and the *change* in rates matter far more than the level.

### "If rates are high, just hold cash and wait"

Tempting, and not crazy when cash pays 5% — but two traps. First, high rates do not last; the moment the market smells cuts, long bonds and rate-sensitive equities rip higher *before* the Fed actually cuts, and cash-holders miss the move (you earn the 5% but forgo a 20% bond rally). Second, "high" is relative to inflation — earning 5% nominal while inflation runs 4% (as it briefly did) leaves you a real return of about 1%, far less impressive than the headline. The level of the rate tells you little; what matters is the rate *relative to inflation* (the real rate) and the *direction* of the next move. (The real-versus-nominal distinction is its own master signal, covered in the cross-link on [real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)

## How it shows up in real markets

Theory is clean; markets are where it bites. Three concrete episodes, with the real numbers, show the master variable at work.

### The 2022 repricing: the discount rate going up for everyone

2022 is the cleanest natural experiment in modern markets for the thesis of this post, because the dominant variable was *one thing* — the discount rate — moving in *one direction* — up — fast. Walk the tape. Inflation, which had been dismissed as "transitory," hit 7.5% year-over-year in January 2022 and peaked at 9.06% in June, a 40-year high. The Fed, badly behind, began hiking in March and did not stop until July 2023, taking the policy rate from a 0.25% ceiling to 5.50%.

![Fed funds rate step chart from 2019 to 2024 with the ZIRP era and the 2022 hiking cycle marked](/imgs/blogs/interest-rates-the-price-of-money-master-variable-2.png)

The step chart above is the master variable itself — the policy rate — making its move. Every other price in 2022 was a reaction to that staircase. As the discount rate climbed, the present value of every future cash flow shrank, and the damage sorted itself neatly by duration:

- **Long-duration government bonds** (20-30 year Treasuries): down more than 30%.
- **Long-duration growth stocks** (Nasdaq-100): down about 33%.
- **The broad stock market** (S&P 500): down about 19%.
- **Short-duration value stocks** (Dow): down about 9%.
- **Crypto** (Bitcoin, the longest duration of all): down about 65%.
- **The dollar** (DXY, which *gains* when U.S. rates rise faster than the rest of the world): *up* about 8% on the year, peaking near 114.8.

Read that list as a duration ladder and it explains itself perfectly. The assets with cash flows furthest in the future (or no cash flows at all) fell the most; the assets that *benefit* from higher domestic rates (the dollar) rose. One variable, sorted by duration, produced the entire cross-asset map of the year.

### The 60/40 portfolio's worst year in a century

The "balanced" 60/40 portfolio — 60% stocks, 40% bonds — is supposed to be diversified: when stocks fall in a recession, bonds usually rise (rates get cut, bond prices go up), cushioning the blow. That cushion depends on stocks and bonds being *negatively correlated.* But in 2022 the shock was not a recession — it was a *rate shock*, and a rate shock hits stocks and bonds *the same direction*, because both are discounted by the same rate that just rose. So both legs fell together, the diversification evaporated, and the 60/40 lost about 16% — one of its worst calendar years in a century. **The lesson the entire investment industry relearned in 2022: when the dominant macro force is the discount rate, diversification across asset classes does not save you, because every asset class shares the same discount rate.** Diversification protects you from idiosyncratic, asset-specific shocks; it does *nothing* against a move in the master variable that sits under all of them.

#### Worked example: the cumulative tightening and a 60/40 portfolio

Let us total up the master-variable move and tie it to the portfolio. The policy rate went from a 0.25% upper bound (March 2020 through March 2022) to 5.50% (July 2023) — a cumulative tightening of **525 basis points, or +5.25 percentage points.** Now build a stylized 60/40 portfolio worth \$100,000 and walk it through 2022 using the real asset returns:

```
Stock sleeve:  60% x $100,000 = $60,000  ->  -19% (S&P 500)  ->  -$11,400
Bond sleeve:   40% x $100,000 = $40,000  ->  -13% (Agg bond) ->  -$5,200
Total loss = -$11,400 + -$5,200 = -$16,600  ->  about -16.6%
```

The portfolio ends 2022 worth about \$83,400, down roughly 16.6%. The damning detail is the bond sleeve: it lost \$5,200, when its entire job in the portfolio is to *make money* (or at least hold steady) when stocks fall. The 40% "safe" allocation, made of intermediate-duration bonds with a duration around 6, fell about 13% because `−6 × the ~2%+ yield rise ≈ −13%` — pure duration math. **Both legs of the portfolio shared one discount rate, that rate rose 525 basis points, and so both legs fell — there was no hiding place inside the portfolio, only outside it (cash, the dollar, or being short duration).**

### Long-duration tech versus value: the rotation as a duration trade

Within the stock market, 2022 was less "stocks fell" and more "long duration fell, short duration held up" — a *rotation*. Money rushed out of expensive, profitless, growth-at-any-price names (long duration) and into cheap, profitable, cash-generating value names (short duration), because the rate move hammered the former far harder. This is the same trade as selling a 30-year bond and buying a 2-year bond when rates rise: you are shortening your duration to reduce rate pain. Traders who recognized that the equity rotation was *just a duration trade in disguise* — that "growth versus value" in 2022 was a polite way of saying "long-duration versus short-duration" — were positioned correctly, while those who thought of it as a vague "style" preference were guessing. **Reframe every rotation as a duration question and the right side of the trade often becomes obvious.**

## How to trade it: the playbook

Everything above collapses into one operating principle and a handful of concrete moves. This is the payoff.

**The master principle: rates are the tide.** Individual stocks, sectors, and tokens are boats. On any given day a boat can move on its own — earnings, news, a narrative. But the tide moves *every* boat, and over weeks and months the tide dominates. The single most important thing you can know about the macro environment is *which way the discount rate is going.* Get the tide right and you can be sloppy about the boats; get the tide wrong and the best boat in the world still goes down. Most macro positioning is, at bottom, a bet on the direction of the master variable.

Here is how to operationalize that, step by step.

**1. Map every position's rate sensitivity (its duration).** This is the core discipline. For every position you hold, ask: *how long until its cash flows arrive?* Sort your book on a duration ladder:

- **Near-zero duration** (rises hurt little): cash, T-bills, money-market funds, very short bonds, profitable cheap "value" stocks with cash flows now.
- **Medium duration** (moderate pain): the broad stock market, intermediate bonds, dividend stocks, mature companies.
- **High duration** (severe pain on a rate rise): long-dated Treasuries, profitless growth and tech, biotech, anything valued on a far-future story, crypto.

Once you can see your whole book on that ladder, you know in one glance how a rate move will hit you. A book stuffed with high-duration assets is a giant leveraged bet that rates fall — make sure that is a bet you actually want.

**2. Trade the direction of the discount rate, not the level.** The level of rates tells you where present values *are*; the *change* tells you which way they are about to move. The big money is made positioning *before* the move:

- **If you expect rates to fall** (the Fed is about to cut, inflation is rolling over, growth is weakening): *lengthen duration.* Buy long bonds, rotate toward long-duration growth, add risk assets. Falling discount rates lift the present value of everything, and the longest-duration assets rip the hardest. The bond and growth-stock bottoms in late 2022 came *before* the Fed's first cut, because the market front-runs the tide.
- **If you expect rates to rise** (inflation is hot, the Fed is hiking, the economy is overheating): *shorten duration.* Move to cash and T-bills (which now actually pay you), rotate from growth to value, trim or hedge long bonds and crypto. You want the lowest-duration book you can tolerate.

**3. Watch the right rate.** For risk assets, the variable that matters most is usually the **10-year Treasury yield**, not the Fed funds rate, because the 10-year is the actual discount rate under stocks and the long end the Fed does not directly control. A useful daily habit: when stocks lurch and you do not know why, *check the 10-year first.* A sharp move up in the 10-year is the most common hidden cause of a sudden equity selloff, especially in long-duration tech. Also watch the *real* yield (the 10-year minus expected inflation), because that is the truest discount rate — a topic in its own right.

The practical discipline is to build a small dashboard of the rates that actually drive your book and glance at it before you look at any single stock. At minimum it holds four numbers: the **Fed funds rate** (where policy is now), **market expectations of the next move** (read off Fed funds futures — the market's bet on the next meeting), the **10-year Treasury yield** (the discount rate under risk assets), and the **2-year-versus-10-year spread** (the curve's recession signal). The relationship between these four tells you the regime. Fed funds high and rising, 10-year rising, curve inverting: tightening regime — shorten duration, favor cash and value, expect risk assets to struggle. Fed funds peaking, futures pricing cuts, 10-year starting to fall: the pivot is coming — start lengthening duration *before* the first cut, because the bond and growth-stock rallies front-run the Fed by months. The single most expensive mistake in rate trading is waiting for the Fed to actually move; by the time the press release hits, the market has already repriced. You are trading the *change in expectations*, and expectations move on every inflation print, every jobs report, every word from the Fed chair — which is why those releases are the highest-volatility moments in all of markets.

**Reading the prints that move rates.** Because the rate path is set by expectations, the calendar events that *update* those expectations are the ones that move everything. Learn the hierarchy. The monthly **CPI inflation report** is usually the highest-stakes release in the cycle, because inflation is what forces the Fed's hand — a hot print pushes rate expectations up and hammers long-duration assets within seconds; a cool print does the reverse. The monthly **jobs report** (nonfarm payrolls) is second, because a strong labor market lets the Fed keep rates high while a weakening one pulls cuts forward. And the **FOMC meeting** itself — eight times a year — is where the Fed confirms or surprises the market's bet, with the chair's press conference often moving rates more than the decision. A macro trader does not need to forecast these perfectly; the discipline is simply to *know they are coming, know which way they push rates, and size your duration exposure accordingly going into them.* Carrying a maximally long-duration book into a CPI print you cannot predict is not a view — it is a coin flip on the master variable.

**4. Know what invalidates the view.** A rate-based thesis is invalidated when the rate path you bet on reverses. If you are positioned for falling rates (long duration) and inflation re-accelerates — exactly the risk live in 2025-2026, with CPI ticking back up toward 4% — the Fed may stop cutting or hike again, and your long-duration book gets hurt. Set the invalidation explicitly: "I am long duration because I expect the 10-year to fall below X; if it breaks above Y on a hot inflation print, I am wrong and I cut." The rate level is your stop.

**5. Respect what cash is telling you.** When the risk-free rate is 5%, cash is not "sitting on the sidelines" — it is a *competing asset paying 5% with zero duration*, and it raises the bar every risky asset must clear. High risk-free rates are a headwind for everything else by definition. When rates are near zero, cash is trash and the discounting tailwind pushes you out the risk curve; when rates are high, cash is a real opponent and a legitimate position. Let the risk-free rate tell you how hard you have to work to justify owning anything riskier than a Treasury bill.

**The one-sentence playbook: figure out which way the discount rate is going, map your whole book onto a duration ladder, lengthen duration when rates are set to fall and shorten it when rates are set to rise, and never forget that in a year like 2022 the tide goes out for everyone at once — there is no boat that floats when the master variable turns against the whole harbor.** Master rates and discounting, and you are not trading one asset class; you are trading the single thread that runs through all of them.

## Further reading and cross-links

- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the mechanics of how the central bank pins the one policy rate that anchors this entire post: the floor system, interest on reserves, and why the Fed controls the short end but not the long end.
- [Real versus nominal: inflation and the real yield as the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the next layer down. The *real* rate (rate minus inflation) is the truest discount rate; this post takes nominal rates as given, that one strips out inflation.
- [What money really is: base money, broad money, and what traders track](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) — the thing whose price an interest rate is. Understand money itself before you trade its price.
- [The dollar system: why the USD rules markets and how to read DXY](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the FX transmission channel in depth: why rate differentials move currencies and why the dollar is the global price of money.
- For the duration mechanics in a portfolio context, the bond-market and risk-parity literature (Bridgewater's "all-weather" framing) is the canonical practitioner reference for thinking in terms of the discount rate as the dominant cross-asset driver.

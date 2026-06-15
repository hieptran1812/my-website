---
title: "Real Yields: The Variable That Prices Everything"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The real interest rate, roughly the TIPS yield, is the master discount rate underneath every asset price. Learn why one variable explains so much of how stocks, bonds, gold, and crypto move together."
tags: ["asset-allocation", "cross-asset", "real-yields", "tips", "duration", "discount-rate", "interest-rates", "gold", "valuation", "portfolio-construction"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The *real* interest rate (the nominal yield minus expected inflation, which is roughly what the inflation-protected "TIPS" bond pays) is the single discount rate that sits underneath the price of almost everything. When it rises, the present value of every future cash flow falls — and the further in the future an asset's cash flows are, the harder it gets hit. That one variable explains a huge share of cross-asset moves.
>
> - **Every asset is priced as cash flows divided by `(1+r)^t`.** Raise `r` (the real yield) and you shrink the value of every future dollar at once. This is why a single rate move can reprice long bonds, growth stocks, gold, and real estate together.
> - **Duration is everything.** How much an asset falls when real yields rise depends on how far out its cash flows are. Long bonds, profitless growth stocks, gold (zero cash flow), and long-lease property are the most exposed; cash, value stocks, and floating-rate debt are the least.
> - **2022 was the cleanest demonstration in decades.** The 10-year real yield went from about **−1% to +1.7%** in ten months. That one move helped sink US bonds (**−13.0%**), the S&P 500 (**−18.1%**), REITs (**−24.9%**), and Bitcoin (**−64%**) in lockstep, while cash *gained* about 1.5%.
> - The one number to remember: a real yield near **−1%** is rocket fuel for long-duration assets; a real yield near **+2%** is gravity. Watch the 10-year TIPS yield and you are watching the dial that sets the weather for the whole portfolio.

In the spring of 2020, the world's most important interest rate went negative — and not the nominal rate you see quoted on the news, but the one that actually matters for valuing things. The yield on a 10-year US Treasury bond that is *protected against inflation* fell below zero and kept falling, bottoming near **−1.06% in January 2021**. Read that literally: an investor was willing to lock up money for a decade and accept a *guaranteed loss* of about 1% a year in purchasing power. That sounds insane until you understand what it does to every other price in the market. When the safest real return on earth is negative, every risky asset that offers *anything* above that looks brilliant by comparison. Money flooded into long-dated bonds, into tech stocks whose profits sat years in the future, into gold, into crypto, into anything with a long tail of cash flows. Asset prices everywhere inflated. It felt like a lot of separate bull markets. It was mostly *one* variable.

Then the dial turned. Over ten months in 2022, that same 10-year real yield climbed from about **−0.1% in April** to **+1.74% in October** — a swing of nearly two full percentage points in the most important number in finance, the most violent repricing of the real cost of money in decades. And the bull markets that had looked separate revealed themselves to be a single trade. US bonds had their worst year in modern history. The S&P 500 fell 18%. Long-duration tech fell far more. Gold, supposedly a safe haven, went *nowhere*. Bitcoin lost almost two-thirds of its value. Real estate trusts dropped a quarter. People reached for a dozen different explanations — a war, a tech bubble, a crypto fraud, a Fed mistake. But underneath the noise, one number had moved, and it had repriced the present value of every future dollar in the economy at the same time.

That number is the **real yield**, and learning to watch it is the closest thing in cross-asset investing to a master key. It does not explain everything — nothing does — but it explains *more* of how different assets move together than any other single variable. The diagram below is the mental model we will build the whole post around: the real yield is the discount rate that converts every asset's future cash flows into a price today, so a move in it ripples out to stocks, bonds, gold, and real estate at once.

![Real yield as the discount rate feeding into long bonds, growth stocks, gold, and real estate, all hit together in 2022](/imgs/blogs/real-yields-the-variable-that-prices-everything-1.png)

This is the deep-dive on real yields in the *Cross-Asset Playbook* series — the post that sits underneath all the others, because the variable it describes sits underneath all the other asset prices. We have looked at [government bonds and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration), at [gold](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock), at [stocks as a claim on growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth), and at [the stock-bond correlation that powers the 60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). This post is the thread that ties them together. (For the macro mechanics — how the real yield is constructed from the bond market and why it is the master *signal* — the macro series has a dedicated companion piece on [real versus nominal rates](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal); we will lean on it rather than re-derive it.)

A quick honesty note before we start: this is educational, not investment advice. The point is to teach you a mechanism that explains market behavior, not to tell you what to buy.

## Foundations: nominal versus real yield, and why the real yield is the discount rate in every valuation

Let us build this from absolute zero, because the whole post rests on three ideas — the difference between nominal and real, what "discounting" actually means, and why one rate does so much work. Get these and the rest is detail.

### Nominal versus real: the rate before and after inflation

Start with the rate everyone quotes. The *nominal* interest rate is the headline number: the bond yields 4%, the savings account pays 5%, the loan charges 7%. It is "nominal" because it is measured in plain dollars, with no adjustment for the fact that those dollars are losing value over time.

The trouble is that dollars *do* lose value. If a bank pays you 5% but prices are rising 4% a year, your money grows by 5% in dollar terms while the things you want to buy get 4% more expensive — so your *actual* gain in buying power is only about 1%. That 1% is the **real** interest rate: the rate after you strip out inflation.

The relationship is, to a close approximation:

$$ r_{\text{real}} \approx r_{\text{nominal}} - \pi^e $$

where `r_real` is the real yield, `r_nominal` is the quoted nominal yield, and `π^e` (the Greek letter pi, with a superscript *e*) is *expected* inflation — what the market thinks inflation will average over the life of the bond. (The exact relationship is `(1 + r_nominal) = (1 + r_real)(1 + π^e)`, but the subtraction is close enough for intuition and we will use it throughout.)

The real yield is the one that tells you the truth about whether you are getting richer. A 12% nominal yield with 13% inflation, as Americans faced in 1980, is a real yield of about **−1%** — you are going *backwards* in purchasing power despite the headline number looking generous. A 2% nominal yield with 0% inflation is a real yield of **+2%** — modest in dollars, genuinely good in buying power. The nominal number is the costume; the real number is the body underneath.

#### Worked example: the same 5% means two completely different things

Suppose you put \$1,000 in a one-year deposit paying **5%**. After a year you have \$1,050 in your account — a \$50 gain in plain dollars. So far so good.

Now ask what those \$1,050 actually *buy*. Take two worlds:

- **World A — low inflation (1%).** A basket of goods that cost \$1,000 last year now costs \$1,010. Your \$1,050 buys that basket and leaves \$40 of real surplus. Your *real* return is about `5% − 1% = 4%`. You genuinely got richer.
- **World B — high inflation (6%).** The same basket now costs \$1,060. Your \$1,050 cannot even buy it — you are \$10 short. Your real return is about `5% − 6% = −1%`. The 5% was an illusion; in buying power you went backwards.

Same nominal 5%, same \$50 of dollar interest, opposite outcomes. The intuition: nominal yields tell you how many dollars you will have; real yields tell you whether those dollars are worth more than the ones you started with — and only the second question matters for wealth.

### The real yield you can actually buy: the TIPS yield

Here is the elegant part. You do not have to *estimate* the real yield from a nominal yield and a guess about inflation. The US Treasury sells a bond that pays it to you directly. It is called a **TIPS** — Treasury Inflation-Protected Security. A TIPS works like a normal Treasury bond, except its principal is adjusted up with the Consumer Price Index. If inflation runs 3% over a year, the bond's principal grows about 3%, and the coupon is paid on the larger principal. The investor is therefore made whole against inflation automatically — whatever inflation turns out to be, it is added back.

Because inflation is handled for you, the yield quoted on a TIPS *is* the real yield. When the market says the 10-year TIPS yields **1.74%**, it is saying: lock up money for ten years and you are promised about 1.74% a year *above whatever inflation does*. That is a real return you can actually buy and hold. The series the whole financial world watches for this is the Federal Reserve's daily 10-year TIPS yield (its data code is `DFII10`), and it is the number we will track for the rest of this post.

So we now have a clean, observable definition: **the real yield ≈ the TIPS yield ≈ the true, after-inflation cost of money.** It is what a saver actually earns, in buying power, on a safe ten-year loan to the government. And it is the floor under everything else, because every risky asset has to beat *that* to be worth owning.

There is a neat bonus in having both bonds quoted side by side. Subtract the TIPS yield from the ordinary nominal Treasury yield of the same maturity and you get the *breakeven inflation rate* — the market's implied forecast for average inflation over the bond's life. If the 10-year nominal Treasury yields **4.0%** and the 10-year TIPS yields **1.7%**, the breakeven is `4.0% − 1.7% = 2.3%`: the bond market is collectively betting inflation will average about 2.3% over the next decade. This is why we can cleanly *decompose* any move in nominal yields into its two parts. When the nominal yield rises, you can look at the TIPS yield to see how much of the rise was the *real* component (a genuine discount-rate shock that hurts every asset) versus the *inflation-expectations* component (which is roughly a wash for real assets). That decomposition — done live in the market every day — is exactly how we will diagnose 2022 as a real-rate shock rather than an inflation scare. It is also the single most useful daily read in cross-asset macro: a nominal yield rising on *real* yields is dangerous for long-duration assets; the same nominal move driven by *inflation expectations* is far more benign.

### Why the real yield is *the* discount rate

Now the keystone idea — the reason this one variable matters so much.

Almost every asset is, underneath, a claim on *future cash*. A bond pays you coupons and then your principal back. A stock gives you a slice of a company's future profits (paid out as dividends or reinvested to grow the business). A rental property pays you rent for decades. Even a commodity, which throws off no cash, is a claim on a future *sale price*. In every case, owning the asset means owning a stream of dollars that will arrive at various points in the future.

But a dollar in the future is worth less than a dollar today — and not because of inflation alone, but because a dollar today can be *invested* to become more than a dollar later. If the safe real rate is 3%, then \$1.00 invested today grows to \$1.03 in a year. Run that backwards: a dollar you will receive in one year is worth only `\$1 ÷ 1.03 ≈ \$0.971` today, because that is the amount you would have to set aside *now*, at 3%, to have exactly \$1 in a year. This "running it backwards" is called **discounting**, and the rate you discount at is the **discount rate**.

The value of any asset is the sum of all its future cash flows, each discounted back to today:

$$ \text{Price} = \sum_{t=1}^{T} \frac{\text{cash flow}_t}{(1+r)^t} $$

where `cash flow_t` is the dollars you expect to receive in year `t`, `r` is the discount rate, and `T` is the final year. This formula is called *discounted cash flow*, and it is the backbone of how serious investors value almost everything.

Look hard at where `r` lives: in the *denominator*, raised to the power of *time*. That placement is the entire story of this post. Because `r` is in the denominator, **a higher discount rate means a lower price** — mechanically, for every asset, all else equal. And because `r` is raised to the power `t`, **the further out a cash flow sits, the more a change in `r` moves its value.** The exact same rate move barely touches a dollar arriving next year but heavily reprices a dollar arriving in twenty years.

And what *is* the discount rate `r`? At its core it is **the real yield plus a risk premium**. The real yield is the safe, after-inflation return you could earn instead; the risk premium is the extra return you demand for taking on uncertainty. When the real yield rises, the *entire* discount rate for every asset rises with it — risk premium or not — because the safe alternative just got better. That is the channel. The real yield is the common term in the discount rate of every asset on earth, which is precisely why moving it moves everything at once.

### The perpetuity: where the discount rate's power is starkest

There is one special case that makes the discount rate's grip on prices unmistakable, and it happens to be the closest model for assets like stocks and long-lease real estate that pay cash *forever*. Imagine a claim that pays you the same amount every year, with no end date — an everlasting stream. Finance calls this a *perpetuity*. Its value collapses to a strikingly simple formula:

$$ \text{Price} = \frac{\text{cash flow per year}}{r} $$

where the numerator is the fixed annual payment and `r` is the discount rate. That is it: an infinite stream of cash, valued by a single division. (The infinite sum of `cash ÷ (1+r)^t` over all future years converges to exactly `cash ÷ r` — the far-off terms shrink to nothing fast enough that the total is finite.)

Look at what this does. If the payment is \$50 a year and `r` is **2%**, the perpetuity is worth `\$50 ÷ 0.02 = \$2,500`. Now nudge `r` up to **4%** — a change of just two percentage points — and the value *halves* to `\$50 ÷ 0.04 = \$1,250`. The cash flow never changed; doubling the discount rate halved the price. This is the discount rate at its most ruthless, and it is not a curiosity: it is roughly how the most long-duration assets behave. A growing perpetuity (the standard textbook model for a stock, where the payment grows at rate `g`) is worth `cash ÷ (r − g)`, and when `r − g` is small — a fast-growing company discounted at a low rate — the value becomes *violently* sensitive to `r`. A move from `r − g = 3%` to `r − g = 5%` cuts the value by 40%. This is the algebra behind why the longest-duration, fastest-"growth" stories get repriced the hardest when the real yield moves. The perpetuity is the duration amplifier in its purest mathematical form.

#### Worked example: a single dollar, ten years out, at two different real rates

You are promised one payment: **\$10**, arriving in **year 10**, certain. What is it worth today?

- **At a 1% real discount rate:** `\$10 ÷ (1.01)^10 = \$10 ÷ 1.1046 ≈ \$9.05`. The decade of waiting costs you about 95 cents.
- **Raise the real rate to 3%:** `\$10 ÷ (1.03)^10 = \$10 ÷ 1.3439 ≈ \$7.44`. Now the same \$10 is worth \$7.44 today.

A two-point move in the real rate — from 1% to 3% — cut the present value of that single far-off dollar by `(\$9.05 − \$7.44) ÷ \$9.05 ≈ 18%`. Nothing about the \$10 changed. The company did not get worse, the bond did not default, the world did not end. Only the rate moved, and a fifth of the value evaporated. The intuition: the discount rate is a tax on the future, and raising it makes every future dollar shrink — with no help from any "real-world" bad news.

We will return to this exact calculation when we talk about 2022, because it is — almost literally — what happened to the whole market.

## Duration is everything: why the same rate move hurts some assets far more than others

If the real yield is the discount rate under all assets, why did some fall 13% in 2022 while others fell 64%? Because assets differ in *how far out* their cash flows sit — and that, precisely, is what determines their sensitivity to the discount rate. The word for this sensitivity is **duration**.

### What duration really means

In bond-land, *duration* has a technical definition (the weighted-average time until you receive a bond's cash flows, which also equals roughly how much its price moves for a 1% change in yield). But the deeper, portable idea is simpler and applies to *every* asset: **duration is how far in the future an asset's value lives.** An asset whose payoff arrives soon has low duration; an asset whose payoff arrives far away — or never resolves into a final lump — has high duration.

Recall why this matters from the formula: the discount rate `r` is raised to the power `t`, the time until the cash flow arrives. A small change in `r` compounds over many years for a distant cash flow and barely registers for a near one. So the rule is:

**The longer an asset's duration, the more its price falls when real yields rise.** Long duration is the amplifier. Same rate move, bigger damage.

#### Worked example: near dollar versus far dollar, same rate shock

Take the same \$10 payment and the same jump in the real rate from 1% to 3%, but change *when* the money arrives.

- **Arrives in year 1 (very short duration):** at 1%, `\$10 ÷ 1.01 ≈ \$9.90`; at 3%, `\$10 ÷ 1.03 ≈ \$9.71`. The two-point rate jump costs you about `(\$9.90 − \$9.71) ÷ \$9.90 ≈ 1.9%`. Barely a scratch.
- **Arrives in year 10 (medium duration):** as computed above, \$9.05 → \$7.44, a **−18%** hit.
- **Arrives in year 20 (long duration):** at 1%, `\$10 ÷ (1.01)^20 ≈ \$8.20`; at 3%, `\$10 ÷ (1.03)^20 ≈ \$5.54`. That is a `(\$8.20 − \$5.54) ÷ \$8.20 ≈ 32%` hit.

Identical \$10, identical two-point rate move. The near dollar lost 2%, the year-10 dollar lost 18%, the year-20 dollar lost 32%. The damage scales with duration. The intuition: rising real yields do not punish assets randomly — they punish them in strict proportion to how far in the future their value lives. That single fact is why 2022 looked the way it did.

### The duration ladder across the asset classes

Now map this onto real assets. Every asset class can be placed on a ladder from shortest to longest duration, and that ladder *is* the ranking of how badly each gets hurt when real yields surge.

![Duration ladder showing longest-duration assets like gold and growth stocks hit hardest by a real-yield rise, cash unhurt](/imgs/blogs/real-yields-the-variable-that-prices-everything-3.png)

Walk it from the top down:

- **Longest duration — the most exposed.** *Gold* has, in a discounting sense, almost *infinite* duration: it throws off no cash flow at all, so its entire value is a claim on a far-future resale price, with nothing near-term to anchor it. *Profitless growth stocks* are nearly as long: a company that loses money today and promises a fortune in 2035 is a giant pile of year-2035 dollars, and we just saw what a rate move does to those. *30-year Treasury bonds* lock in cash flows for three decades. *Long-lease real estate* — think a building with rents contracted out for decades — is a stream of far-future rent. All of these have huge effective duration, so a rising real yield is a wrecking ball.
- **Medium duration.** The *broad equity index* (the S&P 500) is a blend — some companies pay fat dividends now (short duration), some reinvest for distant growth (long duration) — so it sits in the middle. A *10-year bond* is, by definition, medium. *Established, profitable tech* is longer than the index but shorter than the dreamers.
- **Short duration — relatively protected.** *Value stocks* — mature companies paying out cash today, banks, energy, consumer staples — have most of their value in near-term cash flows, so a higher discount rate stings less. *Short-dated bonds* mature soon and reprice gently. *Floating-rate debt* actually *benefits*, because its coupon rises with rates.
- **Zero duration — the safe harbor (and sometimes the winner).** *Cash and Treasury bills* have essentially no duration: they reprice overnight to the new rate. When real yields surge, cash does not fall — it starts *earning more*. In 2022, while everything long-duration was bleeding, cash quietly returned about **+1.5%**.

This ladder is the single most useful picture in cross-asset investing. Once you internalize it, a rising-real-yield headline stops being abstract: you can immediately rank, in your head, which of your holdings is most exposed. The thing with the most distant cash flows takes the most pain.

It is worth laying the ladder out as a table, because the same logic that ranks the *pain* from rising real yields also ranks the *gain* when real yields fall — duration cuts both ways:

| Asset | Effective duration | When real yields **rise** | When real yields **fall** |
|---|---|---|---|
| Gold | Very long (no cash flow) | Stalls or falls (2022: −0.3%) | Rallies hard (2024: +27%) |
| Profitless growth / crypto | Very long | Crushed (2022 crypto −64%) | Soars (2020-21 melt-up) |
| 30-year Treasury | Very long | Falls hard (2022 worst on record) | Rallies hard |
| Broad equity index | Medium | Moderate hit (2022 S&P −18%) | Solid gains |
| Value stocks | Short | Mild hit, often outperforms | Lags the long-duration rally |
| Short bonds / floating-rate | Short | Mild; floaters *benefit* | Modest |
| Cash / T-bills | Zero | *Earns more* (2022: +1.5%) | Earns less |

The pattern is perfectly symmetric: the assets at the top of the table are the big winners when the dial falls and the big losers when it rises, and the assets at the bottom barely move either way. That symmetry is the whole reason knowing your portfolio's duration is so useful — it tells you, in a single word, which direction of the real yield you are *really* betting on.

#### Worked example: a value stock versus a growth stock, same rate shock

Put two companies side by side and run the same real-yield move through both, using the perpetuity logic from the Foundations section. Both will eventually pay out \$100 per share over time — but the *timing* differs.

- **The value company (short duration).** It is mature and pays \$5 a year *now*, growing slowly at 1%. Model it as a growing perpetuity worth `\$5 ÷ (r − g)`. With a discount rate of `r = 7%` and growth `g = 1%`, it is worth `\$5 ÷ (0.07 − 0.01) = \$5 ÷ 0.06 ≈ \$83`. Now the real yield rises two points, lifting `r` to 9%: `\$5 ÷ (0.09 − 0.01) = \$5 ÷ 0.08 ≈ \$63`. A drop of `(\$83 − \$63) ÷ \$83 ≈ 24%`.
- **The growth company (long duration).** It pays almost nothing now but is expected to grow its payout 5% a year. With `r = 7%` and `g = 5%`, it is worth `\$5 ÷ (0.07 − 0.05) = \$5 ÷ 0.02 = \$250`. Raise `r` to 9%: `\$5 ÷ (0.09 − 0.05) = \$5 ÷ 0.04 = \$125`. A drop of **50%** — *exactly half* — from the same two-point rate move.

The growth name fell twice as hard as the value name on an *identical* rate shock, purely because its cash flows are weighted toward the distant future, which makes `r − g` small and the price violently rate-sensitive. The intuition: when you hear "growth stocks got hit by rates," this is the arithmetic — a small denominator amplifies every move in the discount rate.

### A subtle but important point: it is the *real* yield, not the nominal yield

You might ask: why real yields and not just nominal yields? After all, the headline 10-year Treasury yield also rose sharply in 2022 — from **1.52% at the end of 2021 to 4.05% in October 2022**.

The reason we focus on the real yield is that it isolates the *true* discount-rate shock from the inflation noise. Nominal yields can rise for two very different reasons: because expected inflation rose (which is roughly neutral for *real* assets, since their cash flows tend to rise with inflation too), or because the real cost of money rose (which is a pure headwind for *every* asset's present value). It is the second one — the real-yield component — that does the cross-asset damage. In 2022 the nominal 10-year yield rose about 2.5 points, but the *real* component accounted for the bulk of it: the TIPS yield rose nearly 1.9 points while the inflation-expectations component actually drifted *down* over the year. That is the tell that 2022 was a real-rate shock, not just an inflation scare — and real-rate shocks are the ones that hit long-duration assets across the board.

## How real yields actually behave: what makes them move

If the real yield is the dial, what turns it? Three forces, and it helps to know them because they tell you *why* the dial is moving, which in turn tells you what regime you are in.

### Growth expectations

The real yield is, at its heart, the price of *real* money — and the demand for real money is driven by how much productive use the economy can make of it. When the economy is expected to grow strongly, businesses want to borrow and invest at higher rates because the projects they are funding promise good real returns; the equilibrium real rate rises. When growth expectations collapse — a recession looms, demand for capital evaporates — the real rate falls, sometimes below zero, because there is more saving than there are good real projects to absorb it. So **strong expected growth pushes real yields up; weak growth pushes them down.** This is why deep recessions and the fear of them drive real yields negative, as in 2020.

### Federal Reserve policy

The Fed sets the very short-term nominal interest rate, and through it, exerts heavy influence over short-term *real* rates and, via expectations, the whole curve. When the Fed hikes aggressively to fight inflation — as it did in 2022, taking its policy rate from near zero to over 4% in a single year — it drags real yields up across the maturity spectrum, because the market prices in a higher safe real return available from holding cash. When the Fed cuts, or signals it will, real yields fall. The 2020-2021 era of deeply negative real yields was partly *engineered*: the Fed held rates at zero and bought bonds (a policy called [quantitative easing](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal)) precisely to push real yields down and stimulate the economy. The 2022 surge was that policy slamming into reverse. (For the machinery of how the Fed actually sets rates, the macro series covers it directly; here we only need the direction: tightening lifts real yields, easing lowers them.)

### Term premium

The third driver is subtler. A *term premium* is the extra yield investors demand for locking up money in a long bond instead of rolling short ones — compensation for the risk that rates move against them over a decade. When investors are nervous about the long-run path of rates, inflation, or government debt issuance, they demand a fatter term premium, which lifts long real yields independent of growth or near-term Fed policy. When they are complacent (or when a central bank is buying all the long bonds), the term premium compresses, even turning negative, dragging long real yields down. Part of the 2020-21 negativity and part of the 2023 spike toward **+2.48%** were term-premium stories — the latter driven by worries about heavy Treasury issuance and a less reliable central-bank buyer.

### The anchor underneath: the natural real rate

Beneath these three short-to-medium-term drivers sits a slower-moving anchor that economists call the *natural real rate of interest* — often written `r*` (pronounced "r-star"). It is the real interest rate that would prevail when the economy is running neither too hot nor too cold: the long-run gravitational center that the real yield orbits. `r*` is set by deep structural forces — the economy's productivity growth, its demographics (an aging, saving-heavy population pushes it down), and the global appetite for safe assets. You cannot observe it directly; it is estimated, and the estimates are uncertain. But it matters because it tells you the *level* the real yield tends to return to. Through the 2010s, most estimates of `r*` drifted down toward roughly **0.5%**, which is part of why real yields could sit so low for so long — the whole anchor had sunk. A live debate since 2022 is whether `r*` has structurally *risen* (because of large deficits, deglobalization, and the costs of the energy transition), which would mean the real yield's new home is higher than the 2010s taught a generation of investors to expect. If true, that is not a cyclical headwind for long-duration assets but a *permanent* re-rating to a higher discount rate — a meaningfully different world to allocate in. You do not need to resolve the debate to use the framework; you only need to know that the dial has a center of gravity, and that center may have moved up.

The chart below is the history that matters: the 10-year real yield from the deeply negative 2020-21 era, through the violent 2022 surge, to the elevated plateau it has held since.

![Line chart of the 10-year real yield from about minus one percent in 2021 to plus 2.5 percent in 2023, settling near 2 percent](/imgs/blogs/real-yields-the-variable-that-prices-everything-2.png)

The shape tells the whole story in one picture. For two years the real cost of money was *negative* — an extraordinary, policy-driven anomaly that inflated every long-duration asset. Then, in 2022, growth-and-inflation reality plus a hawkish Fed dragged it from −1% to +1.7% in ten months — the steepest move in the post-2008 era. It pushed on to **+2.48% in October 2023** on term-premium fears, eased back, and has hovered near **+2%** since (the latest reading in our data, **April 2026, is 1.99%**). The investing world of the 2010s — long-duration everything, "there is no alternative" to stocks, free money for growth dreams — was built on the left side of this chart. The world since 2022 is the right side. Almost every cross-asset puzzle of the last few years dissolves once you overlay it on this line.

## The 2022 case: one variable, many victims

We have built all the machinery; now let us watch it operate on the cleanest example in modern markets.

Going into 2022, the 10-year real yield sat near **−1%** (it was −1.04% at the end of December 2021). Then the Fed, staring at 9% inflation, began the fastest hiking cycle in forty years. The real yield broke above zero in the spring (**−0.14% in April**), hit **+0.65% by June**, and reached **+1.74% in October** — a swing of nearly two full points. By our worked-example logic, that is *exactly* the kind of shock that detonates long-duration assets. And it did, all at once.

![Horizontal bar chart of 2022 total returns, Bitcoin minus 64 percent down to cash plus 1.5 percent, ordered by duration](/imgs/blogs/real-yields-the-variable-that-prices-everything-5.png)

Read the bars from top (worst, longest duration) to bottom (best, zero duration):

- **Bitcoin: −64%.** A pure long-duration speculation — no cash flow, all of its value a bet on a distant future. When the discount rate on "distant future" doubled, this was the asset with the most to lose, and it lost the most.
- **US REITs (real estate trusts): −24.9%.** Long-lease, long-duration income streams, often leveraged with borrowed money that got more expensive. A double hit from the rate move.
- **S&P 500: −18.1%**, with the growth-and-tech-heavy slices far worse (the long-duration corners of the index were down 30-40%). Even the broad index, a medium-duration blend, took a serious hit.
- **US bonds (the Aggregate index): −13.0%** — the worst year in the index's history. Bonds *are* duration by definition, and a real-rate shock is exactly what they are most exposed to.
- **High yield: −11.2%.** Corporate debt, hit on both the rate and the credit-risk channels.
- **Gold: −0.3%.** Essentially flat — *stalled*. Gold's infinite duration meant the rising real yield was a powerful headwind; only safe-haven and inflation demand kept it from falling outright. "Gold went nowhere in a crisis year" baffled a lot of people until you remember it competes directly with real yields.
- **Cash: +1.5%.** Zero duration. As the real yield rose, cash simply earned more. The only winner — because it was the only thing with no future to discount.

This is the picture worth burning into memory. Six "different" assets — a cryptocurrency, a property index, a stock index, a bond index, a metal, and a bank deposit — that an investor would normally think of as having nothing to do with each other, all moved in a single, duration-ranked line, because a single variable underneath them moved. The narratives (crypto fraud, tech bubble, Fed error) were real but secondary. The *primary* driver was the real yield, and the ranking of the damage was the ranking of duration. One variable, many victims.

#### Worked example: applying the discount math to a 2022 growth stock

Make it concrete with a stylized but realistic case. Imagine a profitless software company the market expects to earn its first meaningful profit — call it **\$10 per share** — in **year 10**, growing thereafter. (Real growth stocks are valued on a whole stream, but the year-10 dollar captures the duration intuition.)

- **End of 2021, real yield ≈ −1%.** Investors discounted that future profit at a very low rate. Even using a modest 8% total discount rate (a low real yield plus a risk premium), the present value of the year-10 profit was high, and the multiple investors paid was enormous — the market as a whole traded near **22 times earnings**.
- **Through 2022, real yield → +1.7%.** Add the same risk premium and the total discount rate jumps toward 10-11%. Discounting our \$10 at the higher rate: where `\$10 ÷ (1.08)^10 ≈ \$4.63`, now `\$10 ÷ (1.105)^10 ≈ \$3.64` — a roughly **21% cut** to the value of that single far-off profit from the rate move alone. Across a whole stream of distant cash flows, the hit compounds, which is why the most speculative names fell 50-80%.

At the index level you can see the same gravity in the multiple: with the real yield near −1% the S&P traded around **22 times earnings**; as the real yield climbed toward +2%, the multiple compressed toward **16-17 times**. Earnings barely moved — the *price the market would pay* for those earnings is what fell, because the discount rate rose. The intuition: in 2022 nothing fundamental had to break for prices to fall hard; the discount rate did the demolition by itself.

## The gold, real-rates, dollar triangle

One relationship deserves its own section because it confuses so many people and because it is a direct corollary of everything above: the interlocking triangle of **gold, real yields, and the US dollar.**

![Triangle diagram linking real yields, the US dollar, and gold, with rising real yields lifting the dollar and pressing gold down](/imgs/blogs/real-yields-the-variable-that-prices-everything-6.png)

Walk the three edges:

**Gold versus real yields (the core link).** Gold has no yield — it pays no coupon, no dividend, no rent. Its competitor for your money is therefore the safe real return you could earn instead, which is the real yield. When the real yield is *negative*, holding zero-yield gold costs you nothing relative to cash (cash is losing purchasing power too), so gold shines. When the real yield is *high and positive*, holding gold means giving up a real, guaranteed return, so the *opportunity cost* of owning the rock rises and gold tends to stall or fall. This is why gold is, to a first approximation, **the mirror image of the real yield.** (We dig into all of gold's drivers in [the gold deep-dive](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock); here the focus is the real-yield edge.)

![Dual-axis chart of gold price versus the inverted real yield, the two lines tracking closely from 2020 to 2024](/imgs/blogs/real-yields-the-variable-that-prices-everything-4.png)

The chart makes the mirror visible. The real-yield axis on the right is *inverted* — flipped upside down — so that the inverse relationship reads as the two lines tracking together. With real yields near −1% in 2020-21, gold sat near record highs (\$1,898 at the end of 2020, \$1,829 at the end of 2021). When real yields turned sharply positive in 2022, gold *stalled* — it finished 2022 at \$1,824, essentially flat, despite a war and the highest inflation in forty years, because the rising real yield was fighting every reason to own it. Then, as the market began pricing future rate *cuts* into 2023-24, the real-yield headwind eased and gold ripped to **\$2,625 by the end of 2024**. The relationship is not mechanical or perfect — central-bank buying drove much of the late surge — but the real yield is the gravity gold is always fighting.

#### Worked example: gold's opportunity cost in two regimes

You have \$1,800 and are deciding between an ounce of gold and a 10-year TIPS.

- **Regime A — real yield = −1%.** The TIPS *loses* you about 1% a year in purchasing power. Gold loses you nothing (ignoring small storage costs). So gold is the *better* store of value — you give up nothing by holding it, and it might rise if others reach the same conclusion. The opportunity cost of gold is *negative*. Gold tends to rally.
- **Regime B — real yield = +2%.** The TIPS now *gains* you a guaranteed 2% a year in real terms. Holding gold means forgoing that 2% — over ten years, compounding, you give up about `(1.02)^10 − 1 ≈ 22%` of real purchasing power versus the bond. That is a steep price to hold a rock. Gold faces a stiff headwind.

The intuition: gold's price is not about gold; it is about the *return you sacrifice to hold it*, and that return is the real yield. Watch the real yield and you have watched the single biggest input to gold.

**Real yields versus the dollar.** The second edge: the US dollar tends to *rise* with US real yields. Money is global and chases real return. When US real yields climb above those available elsewhere, capital flows toward dollar assets to capture that return, bidding up the dollar (the index that tracks it is the DXY). In 2022, surging US real yields helped drive the dollar to two-decade highs. So real yields and the dollar are usually *positively* linked — the same force that lifts one tends to lift the other.

**The dollar versus gold (closing the triangle).** Gold is priced in dollars worldwide. When the dollar strengthens, an ounce of gold costs *more* in every other currency, which dampens demand and weighs on the dollar price of gold. So a strong dollar is, on its own, a headwind for gold. Notice that this *reinforces* the first edge: rising real yields both directly raise gold's opportunity cost *and* strengthen the dollar, which presses on gold a second way. The three corners lock together. Rising real yields → stronger dollar → weaker gold, all at once; falling real yields → weaker dollar → stronger gold. The triangle is just the discount-rate logic, refracted through gold's zero cash flow and the dollar's role as the world's pricing unit.

## Common misconceptions

A few beliefs about real yields are widespread and wrong. Each is worth correcting with a number, because the corrections *are* the insight.

**"Stocks and bonds are different worlds, so I'm diversified holding both."** This was *the* hard lesson of 2022. For most of the 2000-2021 era, stocks and bonds moved opposite each other — bonds rose when stocks fell, which is the entire premise of the [60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). But when the *real yield* is the dominant driver, both are long-duration assets being discounted at the same rising rate, so they fall *together*. In 2022 the classic 60/40 lost about **16%** — its worst year since 1937 — because both legs had the same hidden exposure. Diversification across asset *labels* is not diversification across *risk factors*; duration is a factor that cuts across labels.

**"The nominal interest rate is what matters for stocks."** It is the *real* yield that does the heavy lifting. You can have nominal yields rise while stocks rise — if the rise is all higher inflation expectations and the real component is flat, corporate cash flows inflate alongside, roughly offsetting. The damage comes specifically when the *real* yield rises. In 2022, the real yield rose nearly 1.9 points while inflation *expectations* fell — and that real-rate component is exactly what crushed equities. Watch the TIPS yield, not just the nominal one.

**"Gold is an inflation hedge, so it must rise when inflation is high."** Gold's relationship is with *real* yields, not headline inflation. 2022 had the highest inflation in forty years — and gold finished **flat (−0.3%)**, because the real yield surged. High inflation only helps gold if it is *not* matched by even higher nominal rates; what gold actually tracks is the real yield, which can rise even as inflation rises. The "inflation hedge" label is a half-truth that fails exactly when people most expect it to work.

**"Rising rates are bad for value stocks too, so it doesn't help to rotate."** Duration is precisely why the rotation works. In 2022, as the real yield surged, value stocks (short duration — banks, energy, staples paying cash now) dramatically outperformed growth (long duration). The S&P value indices were roughly flat-to-down-single-digits while growth and tech fell 30%+. The *same* rate move that hammered long duration only grazed short duration. Rotating from long to short duration is the single most effective move when real yields are rising.

**"If real yields explain so much, the market is easy to predict."** No — the real yield explains a large share of *cross-asset co-movement* (why things move *together*), not the *direction* of the next move. Predicting the real yield itself is as hard as predicting the economy and the Fed. The value of this framework is *understanding and positioning*, not forecasting: when the real yield *does* move, you know in advance which assets are most exposed and why. That is a different, more achievable edge than calling the next turn.

## How it shows up in real markets

Beyond 2022, the real yield's fingerprints are all over modern market history. A tour of named episodes, each one the same mechanism in a different costume.

**2020 — the negative-real-yield melt-up.** When COVID hit, the Fed cut to zero and bought bonds at scale, deliberately crushing real yields below zero (the 10-year TIPS yield bottomed near −1.06% in early 2021). With the safe real return *negative*, every long-duration asset inflated: the Nasdaq doubled off its lows, profitless tech IPOs soared, gold hit records (+25.1% in 2020), and the seeds of the 2021 crypto and meme-stock manias were planted. It looked like a dozen separate bubbles. It was one negative number making every distant cash flow look precious.

**2021 — the everything rally on borrowed time.** Real yields stayed deeply negative all year (−1.04% at year-end), and the long-duration party continued: the S&P returned 28.7%, REITs an extraordinary 41.3%, and speculative growth ran hot. But the fuel was a real yield that could only go one direction from −1%. Investors who understood the discount-rate logic could see the entire edifice rested on real yields staying negative — a bet that became untenable the moment inflation forced the Fed's hand.

**2022 — the repricing.** Covered above in full. The real yield's −1% to +1.7% surge was the cleanest cross-asset event in decades: long bonds, growth stocks, gold, and crypto all fell in a duration-ranked line, the 60/40 had its worst year since 1937, and only cash and short-duration value held up. If you remember one case study from this post, it is this one.

**October 2023 — the term-premium spike.** Long after the Fed's hiking had mostly finished, the 10-year real yield spiked again, to **+2.48%**, its highest since 2008. This time the driver was not growth or near-term policy but the *term premium*: fears about the flood of Treasury issuance needed to fund large deficits, and a Fed that was now *selling* bonds rather than buying. The spike hammered long-duration assets again — utilities, REITs, long bonds, and the most expensive tech wobbled — a reminder that real yields can lurch on supply-and-positioning grounds, not just the economy.

**Late 2023 into 2024 — the pivot rally.** When the Fed signaled it was done hiking and the market began pricing *cuts*, the real yield eased from +2.48% back toward +1.7%, and the long-duration assets that had suffered came roaring back. Tech led a powerful rally, long bonds rallied hard into year-end 2023, and gold — freed from the real-yield headwind and helped by central-bank buying — climbed from \$2,063 to **\$2,625** by the end of 2024. The dial turned back a notch, and the same assets that fell on the way up rose on the way down. Symmetry, exactly as the model predicts.

**Emerging markets — the leveraged version of the same trade.** Real yields do not stop at the US border. Many emerging-market countries and companies borrow in *dollars*, so when US real yields surge, their debt gets more expensive to service *and* the stronger dollar that comes with higher US real yields makes those dollar debts heavier in local-currency terms — a double squeeze. Capital that was reaching for yield in emerging markets when US real yields were negative rushes home when US real yields turn positive and the safe return at home improves. This is why emerging-market stocks and bonds are, in effect, a *leveraged* bet on US real yields falling: they fly when the dial is low (emerging-market equities returned 18.3% in 2020) and get hit hard when it rises (they fell about 20% in 2022). The same variable that prices a US tech stock prices a Brazilian bond — through the dollar and the global hunt for real return.

**The 1970s and 2013, as bookends.** History rhymes. The 1970s stagflation drove *deeply negative* real yields (high inflation, lagging nominal rates), and gold ran from \$35 to \$850 — the opportunity cost of holding the rock was negative for a decade. The 2013 "taper tantrum," when the Fed merely *hinted* at slowing its bond-buying, spiked real yields and instantly knocked gold down nearly 30% on the year and rattled long bonds and emerging markets. Different decades, same physics: real yields fall, long duration flies; real yields rise, long duration falls.

## When to own it: the real-yield regime playbook

Here is the payoff — turning all of this into a decision framework. The real yield is a *dial*, and the playbook is mostly a matter of knowing which way it is turning and tilting your portfolio's duration accordingly.

![Two-column matrix mapping falling versus rising real yields to long versus short duration tilts](/imgs/blogs/real-yields-the-variable-that-prices-everything-7.png)

The matrix has two settings:

### When real yields are *falling* (easing, slowdown): favor long duration

A falling real yield usually means the Fed is cutting or about to, growth fears are rising, and money is reaching for safety and for the distant cash flows that a low discount rate makes precious. In this regime the present value of far-off cash flows is *rising*, so you lean toward **long-duration assets**:

- **Growth and tech stocks** — their distant profits get revalued upward as the discount rate falls.
- **Long-dated bonds** — their prices rise mechanically as yields fall, and the longer the bond, the bigger the gain.
- **Gold** — its real-yield headwind becomes a tailwind; zero-yield gold faces no opportunity cost when the safe real return is shrinking.

The 2019-2020 and late-2023-into-2024 windows were textbook "real yields falling" regimes, and long duration led in both.

### When real yields are *rising* (tightening, reflation): favor short duration

A rising real yield usually means the Fed is hiking, the economy is hot, inflation is forcing real returns up, or term-premium fears are lifting long yields. The present value of far-off cash flows is *falling*, so you rotate toward **short-duration assets**:

- **Value stocks** — mature businesses paying out cash now, whose value sits in the near term and so is less discounted away.
- **Cash and floating-rate debt** — cash *earns more* as real yields rise; floating-rate coupons reset upward. Both are the rare things that benefit.
- **Commodities** — short-duration, real-economy assets that often rise in the same reflationary conditions that push real yields up. Commodities were the *only* major positive asset class in 2022, up 16%, precisely because the same hot economy and inflation that drove real yields up also drove demand for physical goods. A barrel of oil or a bushel of wheat is the ultimate short-duration asset: its value is the *spot* price right now, with no distant cash flow to discount away, so a rising real yield barely touches it directly.

The 2022 and the October-2023 episodes were "real yields rising" regimes, and short duration was the place to hide.

### Sizing, pairing, and what invalidates the case

You do not have to bet the whole portfolio on the direction of one variable — and you should not, because the real yield is hard to forecast. The practical uses are calmer:

- **Know your portfolio's aggregate duration.** Before worrying about the next move, simply ask: *how much of my portfolio is long-duration?* A book that is heavy in growth tech, long bonds, and gold is one big bet on real yields *falling* — even if it looks "diversified" by asset label. That hidden concentration is what blindsided so many in 2022. Awareness alone is most of the value.
- **Tilt at the margin, do not flip wholesale.** A reasonable response to a clear rising-real-yield regime is to *trim* the longest-duration exposures (the most speculative growth, the 30-year bonds, an oversized gold slice) and *add* short-duration ballast (cash, value, floating-rate), not to liquidate and reverse. The dial moves both ways and whipsaws.
- **Pair long and short duration deliberately.** A portfolio that holds *some* long duration (for the falling-yield regimes) and *some* short duration and cash (for the rising-yield regimes) is robust to the dial turning either way. The mistake is owning *only* one end of the ladder.
- **What invalidates the framework.** Real yields explain co-movement, not everything. Idiosyncratic events — a specific company's fraud, a single-country crisis, a commodity supply shock — can dominate for a given asset regardless of the real yield. And the relationships are statistical, not mechanical: gold can rise *despite* rising real yields if central-bank buying is heavy enough, as in 2024. Treat the real yield as the single most important *common* factor, not as the only one.

The deepest takeaway is the one we opened with. When you next see a day where stocks, bonds, gold, and crypto all fall together and the headlines reach for four different explanations, look first at one number: the 10-year real yield. More often than the narratives suggest, the four "separate" stories are a single story — the discount rate under every asset moved, and everything with a future to discount moved with it. Learn to watch that dial, and a huge share of cross-asset behavior stops being mysterious and starts being legible.

## Further reading and cross-links

- [Government Bonds: The Risk-Free Anchor and Duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — the mechanics of duration and why bonds are the purest expression of the discount-rate trade.
- [Gold: Money, Insurance, or Just a Rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — gold's full set of drivers, with the real-yield relationship at the center.
- [Stock-Bond Correlation: The 60/40 Engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — why stocks and bonds sometimes fall together, and how the real yield flips their correlation.
- [Equities: Stocks, Owning a Slice of Growth](/blog/trading/cross-asset/equities-stocks-owning-a-slice-of-growth) — why growth stocks are the longest-duration corner of the equity market.
- [Real vs Nominal: Inflation and the Real-Yield Master Signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the macro companion: how the real yield is constructed from the bond market and read as a signal.

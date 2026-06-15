---
title: "Government Bonds: The Risk-Free Anchor and the Duration Trade"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-deep guide to government bonds: what a bond is, the price-yield seesaw, duration, why Treasuries are the risk-free anchor, and when they hedge a portfolio versus when inflation breaks the hedge."
tags: ["asset-allocation", "cross-asset", "government-bonds", "treasuries", "duration", "yield-curve", "fixed-income", "interest-rates", "recession-hedge", "portfolio-construction"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A government bond is a loan with a fixed schedule of payments; its price moves opposite to interest rates, and *duration* tells you exactly how violently. Treasuries are the portfolio's anchor and its recession hedge — but only when the shock is a growth shock, not an inflation shock.
>
> - A bond's price and yield sit on a seesaw: when market yields rise, the price of bonds you already own falls. Duration is the multiplier — a bond with ~8.7-year duration loses about 8.7% for every 1-point rise in yield.
> - Long bonds are a leveraged bet on rates. In 2022, the 10-year yield jumped from 1.5% to over 4%, and the US Aggregate Bond Index fell **−13.0%** — the worst year in its history.
> - Bonds are ballast *only* when the shock is a growth shock (2008: long Treasuries **+25.9%** while the S&P fell **−37%**). When the shock is inflation, stocks AND bonds fall together — that is why 2022 hurt so much.
> - The one number to remember: **price change ≈ −duration × yield change**. Everything else in fixed income is a footnote to that line.

When a bank pays you almost nothing on your savings but a government bond can lose you 13% in a single year, something about the word "safe" is clearly doing a lot of hidden work. In 2022, the safest, most boring, most government-guaranteed corner of the market — the US Aggregate Bond Index, the benchmark that tracks investment-grade American bonds — had its worst calendar year on record: **−13.0%**. Retirees who had been told for decades that bonds were the cautious choice watched their "safe" money fall almost as hard as stocks. At the same time, just fourteen years earlier in 2008, those very same kinds of bonds had been the one thing that *worked*: while the S&P 500 lost more than a third of its value, long-dated US Treasuries returned **+25.9%**.

Same asset class. Two opposite outcomes. The difference between them is the single most important idea in this entire post, and it is not complicated once you see it: a government bond protects you when the economy is collapsing for *growth* reasons, and it punishes you when the economy is overheating for *inflation* reasons. Understanding why requires understanding what a bond actually is, how its price moves, and the one number — duration — that governs how hard it swings.

The diagram below is the mental model we will keep returning to. A bond's price and its yield sit on opposite ends of a seesaw. You cannot push one up without sending the other down. Everything else — duration, the recession hedge, the 2022 rout — is a consequence of that single mechanical fact.

![Price and yield on a seesaw moving in opposite directions](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-1.png)

By the end of this post you will be able to look at a bond, estimate how much it moves when rates change, know whether it is doing its job in your portfolio, and — the payoff — know which part of the economic cycle is the right time to own a lot of it versus almost none.

## Foundations: a bond is a loan you can sell

Strip away the jargon and a bond is the most familiar financial object there is: **a loan**. The difference from the loan in your head is just the direction. When you buy a bond, *you* are the lender, and the borrower is whoever issued it — in this post, a national government like the United States Treasury.

Here is the deal in plain terms. You hand the government a chunk of money today. In exchange, the government promises two things: to pay you a fixed amount of interest on a regular schedule for the life of the loan, and to hand back your original money in full on a specific future date. That is the entire contract. Everything else is vocabulary.

Let us define the vocabulary now, because the rest of the post leans on it:

- **Face value** (also called *par value* or *principal*) — the amount the government promises to repay you at the end. For most bonds this is a round number; we will use \$1,000 throughout. It is the size of the loan.
- **Coupon** — the fixed interest payment, quoted as an annual percentage of the face value. A 3% coupon on a \$1,000 bond pays \$30 per year. The word "coupon" is a historical leftover: old paper bonds had detachable paper coupons you literally clipped off and mailed in to collect each interest payment. Most bonds pay this in two installments — \$15 every six months — but we will keep it annual at first to stay clean.
- **Maturity** — the date the loan ends and your face value comes back. A "10-year Treasury" matures ten years after it is issued. Maturity is the length of the loan.
- **Yield** — the return you actually earn if you buy the bond at today's market price and hold it to maturity. This is the number that matters, and it is *not* the same as the coupon once the bond starts trading. We will build it up carefully, because the gap between coupon and yield is where all the action is.

A *bond's coupon and face value are fixed and printed on the contract; its price floats freely in the market.* Hold onto that sentence. The coupon never changes. The \$1,000 you get back at the end never changes. But the price someone will pay you for that stream of fixed payments changes every single day — and that floating price is what creates yield, duration, gains, and losses.

The figure below shows the whole life of one bond as a stream of cash flows. You pay once, up front (an outflow, in red). Then you collect a fixed coupon at every interval (inflows, in green). At maturity you collect the final coupon *plus* your principal back (the big inflow, in blue). That picture — one outflow now, a chain of fixed inflows later — *is* a bond. If you can read this timeline, you can price any bond in the world.

![Bond cash flow timeline showing one purchase outflow then coupon and principal inflows](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-2.png)

#### Worked example: the cash flows of a 5-year bond

Suppose you buy a brand-new 5-year US Treasury with a \$1,000 face value and a 3% coupon, paid \$15 every six months. Lay out exactly what you receive:

- Today (t = 0): you pay **−\$1,000**.
- Every six months for five years (ten payments): you receive **+\$15** each. That is 10 × \$15 = **+\$150** of coupons in total.
- At the end of year 5: you receive your final \$15 coupon *and* your **+\$1,000** face value back.

Add it up. You put in \$1,000 and you got back \$150 in coupons plus your \$1,000 principal, for \$1,150 total. Your profit over five years is \$150, which works out to 3% per year on your \$1,000 — exactly the coupon rate. That is not a coincidence: *when you buy a bond at its face value, your yield equals its coupon.* The 3% coupon and the 3% yield are the same number, but only at that one starting price.

The intuition: a bond bought at par is a 3% savings account that returns your deposit at the end — nothing surprising happens *as long as you paid exactly face value.*

The interesting cases are all the times you *don't* pay face value — and that is where price and yield split apart.

### Yield to maturity: the number that actually matters

The coupon tells you the dollar interest, but it does not tell you your *return*, because you rarely pay exactly face value. The number that captures your true return is the **yield to maturity** (YTM) — the single annual rate that makes the present value of all the bond's future payments equal to the price you pay today. In plain English: YTM answers the question "if I buy this bond at this price and hold it to maturity, collecting every coupon and the final principal, what annual percentage return do I earn?" It bakes in three things at once — the coupons you collect, the timing of those coupons, and any gain or loss between the price you paid and the \$1,000 you get back at the end.

This is why a bond can trade at three different "states" relative to its face value:

- **At par** — price equals face value (\$1,000). YTM equals the coupon. The simple case from the worked example above.
- **At a discount** — price below face value (say \$920). You collect the same coupons *and* a \$80 gain when the bond matures at \$1,000, so your YTM is *higher* than the coupon. Bonds trade at a discount when market yields have risen above the bond's coupon.
- **At a premium** — price above face value (say \$1,080). You collect the same coupons but take an \$80 loss when it matures back down at \$1,000, so your YTM is *lower* than the coupon. Bonds trade at a premium when market yields have fallen below the bond's coupon.

Whenever a financial site quotes "the 10-year yield," it means the yield to maturity, not the coupon. From here on, when we say "yield," we mean YTM — the number on the right side of the seesaw.

#### Worked example: a discount bond's true yield

You buy an older 5-year bond with a 3% coupon (\$30/year on \$1,000 face), but because market rates have risen you only pay **\$920** for it. What do you actually earn?

- Coupons: 5 years × \$30 = **+\$150** over the life of the bond.
- Capital gain at maturity: it matures at \$1,000, but you paid \$920, so you pocket an extra **+\$80**.
- Total received above your purchase price: \$150 + \$80 = **+\$230** on a \$920 outlay, over 5 years.

Roughly, that is \$230 / \$920 ≈ 25% total, or about **4.6% per year** — well above the 3% coupon. The discount you bought at *adds* return, because the \$80 pull-to-par is pure gain on top of the coupons. That extra return is exactly why a discount bond's YTM exceeds its coupon.

The intuition: your real return is the coupon *plus* whatever gain or loss is built into the gap between the price you paid and the \$1,000 you get back.

## What drives the price: the price-yield seesaw

Here is the question that unlocks everything: why would the price of an existing bond ever change? The coupon is fixed at \$30. The \$1,000 repayment is fixed. The maturity date is fixed. Nothing in the contract moves. So what makes the price move?

The answer is **competition from new bonds.** When you own a bond, you are holding a fixed stream of payments. But the government keeps issuing new bonds, and the interest rate on those new bonds tracks whatever the central bank and the market demand *right now*. (For the full story of what sets that rate, see [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).) When prevailing interest rates change, your old bond suddenly looks either generous or stingy compared with the new ones — and the market re-prices it until it is competitive again.

Walk through it slowly with the seesaw figure from the top of the post in mind.

**Case 1 — rates rise.** You own a bond paying a \$30 coupon (a 3% yield, bought at \$1,000). Then market interest rates rise, and the government starts issuing brand-new \$1,000 bonds with a \$40 coupon — a 4% yield. Now ask yourself: would anyone pay you \$1,000 for your old bond that pays only \$30, when they could buy a new one for \$1,000 that pays \$40? Of course not. Your bond is now inferior. The only way to sell it is to drop the price until the *yield* a buyer earns on your bond matches the 4% they could get elsewhere. The price has to fall.

**Case 2 — rates fall.** Now the opposite. Market rates drop, and new bonds only pay a \$15 coupon (a 1.5% yield). Suddenly your old bond paying \$30 is a gem — it pays double what new bonds pay. Buyers will compete for it, bidding the price *up* above \$1,000, until the yield they earn buying at that higher price falls to match the 1.5% available elsewhere. The price has to rise.

That is the price-yield seesaw, and it is mechanical, not psychological:

> **When market yields rise, existing bond prices fall. When market yields fall, existing bond prices rise. Always, and in exact proportion to how far the yield moved and how long the bond has left to run.**

The phrase "*how long the bond has left to run*" is the seed of duration, which we get to next. But first, let us make the seesaw quantitative, because "the price falls" is not good enough — we need to know *by how much.*

#### Worked example: pricing a bond when rates move

You own a 10-year bond, \$1,000 face, paying a \$30 annual coupon (3% yield, bought at par). Tomorrow, market yields for 10-year bonds jump to 4%. What is your bond now worth?

A bond's price is just the *present value* of all its future cash flows, discounted at the new market yield. "Present value" means: how much is a future dollar worth today, given that you could otherwise earn the market rate? The math discounts each future \$30 coupon and the final \$1,000 by the new 4% rate. We will not grind through all ten terms by hand, but the result is what matters:

- At a 3% yield, the bond is worth exactly **\$1,000** (you paid par).
- At a 4% yield, the same bond is worth about **\$919**.

So a 1-percentage-point rise in yield knocked roughly **\$81**, or about **−8.1%**, off the price of a 10-year bond. You did nothing wrong. You still get every \$30 coupon and your \$1,000 at the end if you hold to maturity. But if you needed to *sell today*, you would take an 8% loss — purely because rates moved.

The intuition: a bond's price is the present value of fixed cash flows, so raising the rate you discount them at mechanically shrinks what they are worth today.

That ~8% drop for a 1-point yield move is not random. It is almost exactly the bond's **duration** — and duration is the master number we turn to now.

## Duration: the master sensitivity number

If you remember one thing from this entire post, make it this line:

$$\text{price change} \approx -\text{duration} \times \text{yield change}$$

Where **duration** is a number, measured in years, that tells you how sensitive a bond's price is to a change in its yield. The minus sign encodes the seesaw: yields up means price down. Let us unpack every piece.

**What duration means, in plain English.** Duration is best understood as the *weighted-average time you wait to get your money back* from a bond, counting every coupon and the final principal. A bond that pays you back sooner has a short duration; a bond that makes you wait decades has a long one. A zero-coupon 10-year bond — one that pays no coupons and just returns \$1,000 in ten years — has a duration of almost exactly 10 years, because you wait the full ten years for everything. A normal 10-year bond that pays coupons along the way has a *shorter* duration than 10 years (around 8.7 years for a typical 3% coupon), because some of your money comes back early in those coupons.

But the reason duration matters to an investor is not the "average wait" interpretation — it is what that number does as a **price multiplier.** Duration tells you, to a very good first approximation, the percentage by which a bond's price moves for each 1-percentage-point move in its yield:

- Duration 2 → price moves about **2%** for a 1-point yield change.
- Duration 8.7 → price moves about **8.7%** for a 1-point yield change.
- Duration 19 → price moves about **19%** for a 1-point yield change.

The figure below draws this directly: three bonds — a short 2-year note, a medium 10-year note, and a long 30-year bond — and how much each one's price moves as the yield changes. The lines have different slopes, and the slope *is* the duration. The long bond's line is steep: a small move in rates throws its price around violently. The short bond's line is nearly flat.

![Duration sensitivity chart showing price change versus yield change for short medium and long bonds](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-3.png)

This is why people say **long bonds are a leveraged bet on interest rates.** You are not borrowing money, so it is not leverage in the literal sense — but the *effect* is the same: a 30-year bond turns a 1-point move in yields into a roughly 19% move in your wealth, the way 19-to-1 leverage would. When rates fall, that is a windfall. When rates rise, it is a wreck. The bond didn't get riskier; you just chose a long-duration instrument, and duration is the dial that sets how much rate risk you are taking.

#### Worked example: why 2022 destroyed long-bond holders

This is the example the whole post has been building toward, so let us do it carefully with real numbers.

Going into 2022, the US 10-year Treasury yielded about **1.5%** (it was 1.52% in December 2021 — source: FRED DGS10). A 10-year Treasury at that time had a duration of roughly **8.7 years**. Over 2022, as inflation surged and the Federal Reserve hiked aggressively, the 10-year yield climbed to about **4.0%** (it hit 4.05% in October 2022). That is a yield change of:

$$4.0\% - 1.5\% = +2.5 \text{ percentage points}$$

Now apply the master formula:

$$\text{price change} \approx -8.7 \times 2.5\% = -21.75\% \approx -22\%$$

A \$1,000 bond lost about **\$220** of market value. Not because the US government's creditworthiness changed — Treasuries are still considered the safest credit on Earth — but purely because the yield on the seesaw shot up and dragged the price down, multiplied by a duration of nearly nine years.

That is why the broad US Aggregate Bond Index, stuffed with medium-duration bonds, fell **−13.0%** in 2022 — its worst year ever. Long-Treasury funds, with durations near 17 to 18 years, fell over **30%**. The "safe" asset behaved like a crashing stock.

The intuition: 2022 was not a credit event; it was a *duration* event — long bonds are a leveraged bet on rates, and rates went the wrong way.

### Convexity: the small correction that helps you

One honest footnote. The formula `price change ≈ −duration × yield change` is a *straight-line approximation*, but the true price-yield relationship is gently curved. That curvature is called **convexity**, and the good news is that it works in the bondholder's favor: when yields fall, prices rise *a little more* than duration alone predicts, and when yields rise, prices fall *a little less* than predicted.

You can see it in the duration chart above if you look closely — the real relationship bows slightly so that the downside is a touch cushioned and the upside is a touch amplified. For a 1-point move, convexity is a small correction (a fraction of a percent). For a huge move like 2022's, it shaves a bit off the loss — the index fell 13%, not the full straight-line estimate. For everyday allocation decisions, duration is 95% of the story and convexity is the polish. We mention it so that when you see the word, you know it just means "the seesaw is slightly curved, in your favor."

### Single bonds versus bond funds: where duration really bites

There is one more wrinkle that trips up almost every beginner, and it determines whether the 2022 loss was "on paper" or permanent. It comes down to whether you own a *single bond* or a *bond fund*.

If you own a *single* government bond and you hold it all the way to maturity, the price swings in between never become a realized loss. You bought a contract that pays \$30 a year and returns \$1,000 in ten years, and as long as the government doesn't default (it won't), you get exactly that. The price might fall to \$780 in year three when rates spike, but if you simply wait, it climbs back to \$1,000 by maturity. The duration loss was real on a mark-to-market statement, but it reverses itself if you hold on. Your only true cost is *opportunity*: your money was locked at the old low coupon while new bonds paid more.

A bond *fund* — the way most people actually own bonds, through an ETF or mutual fund — is different in a way that matters enormously. A fund holds hundreds of bonds and constantly sells the ones that are getting short and buys new longer ones to keep its target duration roughly constant. The fund *never matures.* So there is no fixed future date when your principal comes back at \$1,000 — there is only the fund's price, which moves with rates every single day, governed by the fund's stated duration. When rates rose in 2022, a fund with a 6-year duration simply fell about 13% and *stayed down* until rates came back, with no maturity date to pull it home.

#### Worked example: the same rate move, single bond versus fund

You put \$10,000 into government bonds in December 2021, when the 10-year yielded 1.5%. Rates then rise 2.5 points over 2022.

- **Single 10-year bond held to maturity.** Its market price drops about 22% in 2022 — your statement shows roughly **\$7,800**. Frightening. But you hold on. Each year the bond pulls back toward \$1,000 of face value, and you collect your coupons the whole time. By the 2031 maturity you receive your full \$10,000 of principal back plus every coupon. Realized loss: **\$0** (you sacrificed opportunity, not capital).
- **Bond fund with ~8-year duration.** It also falls about 18-20% in 2022 → roughly **\$8,100**. But the fund never matures. To get your money back to \$10,000, you now *need rates to fall again* — there is no maturity date doing the work for you. If rates stay high, the loss is effectively permanent unless you wait for new, higher coupons to slowly rebuild the value.

The intuition: duration risk is a temporary, self-healing loss for a held-to-maturity single bond, but a real, rate-dependent loss for a perpetual bond fund — and almost everyone owns the fund.

## Why Treasuries are the "risk-free" reference

We keep calling government bonds — specifically US Treasuries — "risk-free." That phrase is doing something precise, and it is worth being exact, because we have just spent a whole section showing how a Treasury can lose you 22%. How can something that volatile be "risk-free"?

The answer: **"risk-free" refers to credit risk, not price risk.** There are two completely different ways a bond can hurt you:

1. **Default risk (credit risk)** — the borrower fails to pay you back. A company can go bankrupt and stiff its bondholders. This is the risk that [corporate credit](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) compensates you for.
2. **Price risk (interest-rate risk)** — the borrower pays you every cent on schedule, but the market price of your bond falls in the meantime because rates rose. This is the 2022 risk.

A US Treasury is considered **free of the first kind of risk.** The US government issues debt in its own currency, which it can always create to pay back; the probability it simply refuses to pay a maturing Treasury is treated as effectively zero by the entire financial system. So if you buy a Treasury and *hold it to maturity*, you are promised your money back with certainty. That promise — guaranteed nominal repayment — is what "risk-free" means.

But Treasuries are absolutely *not* free of the second kind of risk. Between today and maturity, their price swings with rates, sometimes brutally. "Risk-free" never meant "won't fall in price." It meant "won't default."

This distinction is why the Treasury yield is the bedrock reference rate for the entire financial world. Because Treasuries carry no credit risk, their yield is treated as the **risk-free rate** — the baseline return you can earn for taking no default risk at all. Every other asset on Earth is priced relative to it:

- A corporate bond yields the Treasury rate *plus a spread* to compensate for default risk.
- A stock's expected return is judged against the Treasury yield as the "do nothing risky" alternative.
- Even the price of an option uses the risk-free rate in its formula.

When the Treasury yield moves, it re-prices everything. That is why this asset class sits at the center of the cross-asset map — it is not just one asset among many, it is the *ruler* everything else is measured with. (For where it fits among all the others, see [the map of asset classes](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own).)

#### Worked example: the risk-free rate as the hurdle every asset must clear

Suppose the 10-year Treasury yields **4.5%** (close to its mid-2026 level of 4.48% — source: FRED DGS10). You are considering a corporate bond from a solid company yielding **6.0%**, and a stock you think might return **8%**.

- The Treasury hands you 4.5% with no default risk. That is your floor — your "why bother with anything else" number.
- The corporate bond pays 6.0%. The extra **6.0% − 4.5% = 1.5 percentage points** (150 *basis points* — a basis point is one-hundredth of a percent, so 1.5% = 150 bps) is your pay for taking the company's default risk. Is 1.5% enough to compensate you for the chance the company goes bankrupt? That judgment *is* credit investing.
- The stock's hoped-for 8% must clear the same 4.5% hurdle. The extra **8% − 4.5% = 3.5 points** is the *equity risk premium* you demand for stock-market volatility.

Notice that when the Treasury yield rises from 4.5% to, say, 5.5%, every one of those comparisons gets worse for the risky asset — the safe option just got more attractive, so stocks and corporate bonds have to offer more to compete. This is the mechanical reason "higher rates are bad for risk assets."

The intuition: the risk-free rate is the hurdle every other investment has to clear, so when it rises, the whole market re-prices downward.

## The yield curve and real yields: two ideas you need

Two more concepts complete the foundation, and both connect to posts that go deeper, so we will cover the essentials and link out.

### The yield curve

So far we have talked about "the" Treasury yield, but there isn't one — there is a different yield for every maturity. A 2-year Treasury, a 5-year, a 10-year, and a 30-year all trade at their own yields. Plot those yields against their maturities and you get the **yield curve.**

Normally the curve slopes *upward*: longer bonds yield more than shorter ones, because lenders demand extra compensation for tying up money longer and for the bigger duration risk we just described. A normal upward curve says "the market expects steady growth and normal rates ahead."

Sometimes the curve *inverts* — short-term yields rise above long-term yields. This happens when the central bank has jacked up short rates to fight inflation while the market bets that growth will weaken and rates will eventually fall. An inverted yield curve has preceded nearly every US recession of the past 60 years, which is why it is one of the most-watched signals in all of finance. The full mechanics of how the curve forms and what it forecasts live in [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable); for our purposes, the key takeaway is that *which part of the curve you own changes your duration and therefore your risk.* Owning the 2-year is a mild rate bet; owning the 30-year is an aggressive one.

### Nominal versus real yield

The yield printed on a Treasury — say 4.5% — is the **nominal yield**: the raw percentage you earn in dollars. But dollars themselves lose purchasing power to inflation. What actually matters for your wealth is the **real yield**: the nominal yield minus inflation.

$$\text{real yield} \approx \text{nominal yield} - \text{expected inflation}$$

If your bond yields 4.5% but inflation runs at 3%, your real yield is only about 1.5% — that is the genuine growth in your purchasing power. And if inflation ever runs *above* your nominal yield, your real yield is **negative**: you are guaranteed to lose purchasing power even though the bond "made money" in dollars.

The US issues bonds whose payments adjust with inflation — Treasury Inflation-Protected Securities, or TIPS — and their quoted yield *is* the real yield directly. That number is one of the most important signals in markets. Real yields were deeply **negative** at the end of 2021 (the 10-year TIPS real yield was **−1.04%** in December 2021 — source: FRED DFII10), which is part of why every asset was so expensive: there was no safe real return anywhere, so money fled into risk. By October 2023 the 10-year real yield had rocketed to **+1.74%**, and by December 2024 it was **+2.20%** — suddenly a safe, positive, inflation-beating return existed again, and that gravity pulled money *out* of riskier assets. The whole real-yield story — why it is the master signal — is in [real versus nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). For now: *the swing from −1% to +2% real yields over 2022-2024 is the deepest reason markets convulsed the way they did.*

## How bonds behave: ballast, flight to quality, and the 2022 break

Now we connect the mechanics to behavior — to what bonds actually *do* in a portfolio when the world goes wrong. This is the heart of the allocator's case for owning them, and it has a crucial, often-missed condition.

The standard pitch for government bonds is that they are **ballast**: a stabilizing weight that steadies a portfolio when stocks are crashing. The logic is the *flight to quality*. When investors panic — a recession looms, a bank fails, a pandemic hits — they sell risky things (stocks, corporate bonds) and rush into the safest asset they can find, which is Treasuries. That rush *bids up Treasury prices* (and pushes their yields down), so government bonds *rise* exactly when stocks *fall*. Your bonds make money while your stocks bleed, cushioning the total portfolio. That is the recession hedge in one sentence.

The chart below shows it working perfectly in 2008. As the S&P 500 lost **−37.0%**, the broad US Aggregate Bond Index returned **+5.2%**, and long-dated Treasuries (the highest-duration, most rate-sensitive government bonds) surged **+25.9%**. Investors were so desperate for safety, and rates fell so far, that the longest bonds — leveraged to falling rates — produced a stock-like *gain* in the middle of a stock-market catastrophe.

![2008 flight to quality bars showing stocks down 37 percent while Treasuries rallied](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-6.png)

The same thing happened in the COVID crash of 2020. As stocks fell about **−34%** from their February peak to the March trough, the US Aggregate Bond Index finished the full year **+7.5%**. Treasuries did their job: they rallied as the world ran for cover.

Here is the condition everyone forgets, and it is the most important sentence in this post: **bonds are ballast only when the shock is a *growth* shock, not an *inflation* shock.**

Think about *why* the flight to quality lowers rates. In a recession, the central bank slashes interest rates to revive the economy, and falling rates push bond prices up (the seesaw). A growth scare is *good* for bond prices because it means lower rates. But what happens when the crisis is the opposite — when the problem is *too much* inflation? Then the central bank *raises* rates to fight it, and rising rates *crush* bond prices. There is no flight to quality, because the safe asset is itself the thing being repriced. The hedge doesn't just fail to help — it actively loses money alongside your stocks.

That is exactly what 2022 was. It was not a growth shock; it was an inflation shock. Inflation hit 9%, the Fed hiked at the fastest pace in decades, the 10-year yield went from 1.5% to over 4%, and *both* stocks (−18.1%) and bonds (−13.0%) fell together. The ballast became dead weight.

The chart below puts the whole decade of bond returns in context. Look at the pattern: small positive years, year after year — the quiet "grind out a few percent" character of bonds — and then the single violent red bar of 2022 that broke the pattern and broke a generation's assumption that bonds can't really hurt you.

![US Aggregate Bond Index annual total returns 2014 to 2024 with 2022 highlighted red](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-4.png)

And here is the yield move that produced that red bar — the 10-year Treasury yield climbing from its 0.62% pandemic low all the way to a 4.88% peak in October 2023. Every step up on this chart is a step down in the price of every bond outstanding. This single line explains the worst bond bear market in modern history.

![US 10-year Treasury yield from 2020 to 2026 rising from 0.6 percent to nearly 5 percent](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-5.png)

#### Worked example: the same hedge, two opposite outcomes

Imagine you hold a simple portfolio: \$60 in stocks and \$40 in long government bonds, for \$100 total. Watch how the *same* allocation behaves in the two regimes.

**Growth shock (like 2008).** Stocks fall 37%, so your \$60 becomes \$60 × (1 − 0.37) = **\$37.80**. But rates fall, and your long bonds rise — say +25%, so your \$40 becomes \$40 × 1.25 = **\$50.00**. Total: \$37.80 + \$50.00 = **\$87.80**. You lost about 12% — painful, but the bonds cut your stock loss nearly in half. The hedge worked.

**Inflation shock (like 2022).** Stocks fall 18%, so your \$60 becomes \$60 × (1 − 0.18) = **\$49.20**. This time rates *rise*, and your long bonds fall too — say −25%, so your \$40 becomes \$40 × 0.75 = **\$30.00**. Total: \$49.20 + \$30.00 = **\$79.20**. You lost about 21% — *worse* than if you'd held stocks alone in some respects, because the part you held for safety also sank. The hedge inverted.

The intuition: the identical bond position rescued you in one crisis and amplified the damage in the other — because the *kind* of shock, not the bonds, decides whether they hedge.

## Correlation with equities: the hedge that flips sign

We can now state the cross-asset relationship that this whole post orbits, in its general form. The correlation between stocks and bonds — whether they move together or in opposite directions — is **not a constant.** It flips sign depending on what is driving markets.

A quick definition: *correlation* measures whether two things move together. A correlation of **+1** means they move in lockstep, **−1** means they move exactly opposite, and **0** means no relationship. For a hedge to work, you want your two assets *negatively* correlated — when one falls, the other rises.

For most of the period from 2000 to 2021, stocks and bonds were *negatively* correlated. The dominant force in markets was the growth cycle: when growth scares hit, stocks fell and bonds rallied (rates fell), so bonds reliably hedged stocks. An entire generation of investors — and the famous **60/40 portfolio** (60% stocks, 40% bonds) — was built on that negative correlation. Bonds were the perfect diversifier because they zigged when stocks zagged.

But go back further, to the high-inflation 1970s and 1980s, and the correlation was *positive*: stocks and bonds moved *together*, because inflation and interest rates were the dominant force, and rising rates hurt both at once. And in 2022, the positive correlation came roaring back — stocks and bonds fell together because, once again, inflation and rates were in the driver's seat.

The chart below traces this sign-flip across the decades. The key regions are shaded: the green band (negative correlation) is where bonds hedge stocks, and the red band (positive correlation) is where they don't. The line spends 2000-2021 in the green and then jumps into the red in 2022.

![Stock bond correlation by era flipping from positive to negative and back to positive](/imgs/blogs/government-bonds-the-risk-free-anchor-and-duration-7.png)

The rule that emerges is clean and worth memorizing:

> **When growth fears drive markets, stocks and bonds are negatively correlated and bonds hedge. When inflation and rate fears drive markets, stocks and bonds are positively correlated and bonds do not hedge — they pile on.**

This single insight reframes the entire job of government bonds in a portfolio. They are not an *unconditional* hedge. They are a hedge *against growth shocks specifically*, and you have to know which regime you are in to know whether your ballast will float or sink. The deeper mechanics of how money rotates between regimes are covered in the macro series; the allocator's job is to know which regime is live.

## Common misconceptions

**"Government bonds are safe, so they can't lose money."** They can't *default* — but they can absolutely lose price. The US Aggregate Bond Index fell **−13.0%** in 2022 with zero defaults. "Safe" means the issuer will pay you back at maturity; it says nothing about the price along the way. If you must sell before maturity into rising rates, you lock in a real loss.

**"Bonds and stocks always move in opposite directions."** False, and dangerously so. Their correlation flips sign by regime: negative (a good hedge) when growth fears dominate, positive (a bad hedge) when inflation and rate fears dominate. The rolling stock-bond correlation was around **−0.30** in 2018 and around **+0.55** in 2022. Anyone who assumed permanent negative correlation got blindsided.

**"A higher yield means a better bond."** Not necessarily. A bond's yield can be high because its price has *fallen* — and if you already own it, that fall is your loss, not a gift. And a corporate bond yielding more than a Treasury isn't "better"; the extra yield is compensation for default risk you are now carrying. Yield is a price, and a high price for taking on risk is not the same as a good deal.

**"Long bonds are the safest because they're the most government-guaranteed."** The opposite of how risk works here. Every Treasury carries the same (effectively zero) default risk regardless of maturity — but the *longer* the bond, the *higher* its duration and the more its price swings with rates. A 30-year Treasury is the *most* volatile government bond, not the safest. If you want stability, you want *short* duration, not long.

**"If I hold to maturity, rate moves don't affect me, so duration is irrelevant."** Half true and half a trap. It is true that if you hold a single bond to maturity and it doesn't default, you get exactly the yield you bought at, regardless of price swings in between. But (a) you bear the *opportunity cost* — if rates rose, your money is stuck earning the old low rate while everyone else earns more; (b) you've lost purchasing power if inflation outran your yield; and (c) almost nobody holds single bonds to maturity — most people own bond *funds*, which never mature and so feel the full price hit. For a fund holder, duration is everything.

**"The government can always print money, so a Treasury is risk-free in every sense."** This conflates the two risks again, in a subtle way. It is true the government can always create the dollars to repay you — that removes *default* risk. But printing dollars to pay you can *itself* cause inflation, which silently erodes the real value of those dollars. So the very mechanism that makes a Treasury free of default risk (a government that can always print) is what exposes you to *inflation* risk. A bond can pay you back every promised dollar and still leave you poorer in real terms. "Risk-free" is a statement about getting your dollars back, never about what those dollars will buy.

**"Bonds are for old people; young investors should skip them entirely."** A common oversimplification. It is broadly true that a long horizon argues for more equities, because equities' **+6.5% real** return compounds far above bonds' **+1.7%** over decades. But "skip them entirely" ignores what bonds do beyond return: they are the dry powder that lets you *buy* stocks cheaply in a crash. An investor with some bonds in 2008 could sell those (up +5% to +25%) and rotate into stocks at the bottom — a rebalancing windfall a 100%-equity investor couldn't capture. Bonds aren't only about lowering volatility; they are optionality.

## How it shows up in real markets

**2008 — the textbook flight to quality.** As the financial system seized up, investors dumped everything risky and bought Treasuries with both hands. The 10-year yield collapsed, long-Treasury prices soared **+25.9%**, and the US Aggregate Bond Index returned **+5.2%** while the S&P 500 lost **−37.0%**. This is the canonical example of bonds doing exactly the job allocators hire them for: rising hard when stocks crash. It worked because 2008 was a *growth and credit* shock — the Fed was cutting rates to zero, and falling rates lift bond prices.

**March 2020 — the COVID crash, with a twist.** When the pandemic hit, stocks fell about **−34%** in five weeks, and Treasuries initially rallied as expected — the 10-year yield fell to **0.62%** by mid-year, a record low. The US Aggregate Bond Index finished 2020 **+7.5%**. But there was a scary moment in mid-March 2020 when even Treasuries briefly sold off, as panicked investors sold *everything* — including safe assets — to raise cash. The Federal Reserve had to step in and buy Treasuries to stabilize the market. The lesson: in a true liquidity panic, *for a few days* even the safe asset can wobble — but the central bank backstops it, and the flight to quality reasserts itself.

**2021 — the calm before the rout.** By December 2021, the 10-year yield was just **1.52%** and the 10-year real yield was **−1.04%** — investors were *paying* (in real terms) for the privilege of holding safe bonds. This was the setup for disaster: with yields this low and durations this long, bonds had almost no cushion. There was nowhere to go but down, and a tiny rise in yields would inflict a large price loss. The math was a coiled spring.

**2022 — the regime break.** Inflation surged past 9%, the Fed hiked rates at the fastest pace since the 1980s, and the 10-year yield leapt from 1.5% to over 4%. Apply the duration formula: a roughly +2.5-point yield move on ~8.7-year duration → about **−22%** on a 10-year bond, and over **−30%** on the longest Treasuries. The US Aggregate Bond Index fell **−13.0%**, its worst year ever. And because this was an *inflation* shock, stocks fell too (−18.1%), so the classic 60/40 portfolio lost about **−16%** — its worst year since 1937. The "diversification" everyone relied on evaporated precisely when it was needed.

**2023-2024 — the reset to a higher plateau.** The 10-year yield peaked at **4.88%** in October 2023 before settling into a **3.8%-4.6%** range through 2024 (4.58% at year-end 2024; 4.48% in mid-2026). The real yield went from −1% to **+2.2%**. The pain of the repricing was over, but the meaning was profound: bonds now offered a genuinely positive *real* return again. After fifteen years of near-zero yields, government bonds were finally being paid to do their job — which, paradoxically, is exactly when their hedge is most worth owning again, because there is now room for yields to *fall* in the next recession.

**The 1970s — the original inflation trap.** Before 2022, the cautionary tale was the stagflation of the 1970s, and it tells the same story even more brutally. Inflation ran in the double digits, the Federal Reserve eventually pushed short rates toward 20% to break it, and long bonds were a disaster for over a decade — investors who bought "safe" government bonds in the early 1970s watched their real value erode year after year as yields ratcheted higher and inflation ate the coupons. The lesson the 1970s taught and 2022 re-taught: *sustained inflation is the one environment in which government bonds are not a safe haven but a slow-motion wealth destroyer.* Bonds protect you from deflation and recession; they have no defense against inflation, because inflation is precisely the force that drives their yields up and their prices down. This is also why some allocators pair their bonds with assets that *do* benefit from inflation — commodities and real assets — so that the portfolio has a hedge for the regime bonds can't cover.

**The long-run record — what bonds actually earn.** Zoom all the way out. Over the 124 years from 1900 to 2023, US government bonds returned about **+1.7% per year after inflation** (real), versus **+6.5% real for US equities** and **+0.4% real for cash** (source: UBS/Credit Suisse Global Investment Returns Yearbook 2024). The message is sobering and clarifying at once: *bonds are not where you get rich.* Over a lifetime they barely beat inflation. You do not own them for their return — you own them for what they do to your portfolio's *risk*, and for the regimes in which they are the only thing that goes up. They are insurance, and insurance is not supposed to be your best-returning asset.

Put those three real numbers side by side and the entire reason for a multi-asset portfolio appears. Equities compound at **+6.5% real** but can lose half their value in a year; cash earns **+0.4% real** but never falls; bonds sit in between at **+1.7% real** and — crucially — often *rise* in the exact years equities crash hardest, provided the crash is a growth shock. You don't own all three because each is great on its own. You own them because their *worst years don't line up* — most of the time. The whole craft of allocation is assembling assets whose pain arrives on different schedules, and government bonds earn their place by having their best years (2008, 2020) in the middle of equities' worst. The 2022 lesson is simply that this scheduling has an exception, and the exception has a name: inflation.

## The allocation playbook: when to own duration

Everything above lands here — the decision. When should you own a lot of government bond duration, when should you stay short, and how much belongs in a portfolio? Let us be concrete. (This is educational, not personalized advice — the point is the framework, not a recommendation.)

### When to own duration (go long)

You want *more* government bonds, and specifically *longer* duration, when the conditions favor falling yields, because falling yields mean rising bond prices and a working hedge:

- **Disinflation** — inflation is high but clearly *falling*. As inflation rolls over, the central bank can stop hiking and eventually cut, pulling yields down. Long duration is a leveraged bet on exactly that.
- **Recession risk is rising** — growth is weakening, unemployment ticking up, the yield curve inverted. A recession brings rate cuts and a flight to quality; long bonds rally hard, as they did **+25.9%** in 2008.
- **Peak rates** — the central bank has finished hiking and the policy rate is at its top. From a peak, the asymmetry favors you: yields have more room to fall than rise, so the convexity and the carry are on your side. The reset to ~4.5% yields by 2024 created exactly this setup — positive real yields *and* room to fall.

### When to stay short (avoid duration)

You want *less* duration — short-term bonds or cash — when the conditions favor *rising* yields:

- **Rising inflation** — the single most dangerous environment for bonds, because it forces the central bank to hike, and hiking crushes long bonds. 2022 is the cautionary tale. In a rising-inflation regime, the bond hedge inverts and long duration is the worst place to be.
- **An active hiking cycle** — when the central bank is raising rates and isn't done, every hike pushes yields up and bond prices down. Stay short until the cycle peaks.
- **When you are paid almost nothing for the risk** — if long bonds yield barely more than short bonds (a flat curve) or real yields are deeply negative (as in 2021), you are taking large duration risk for tiny compensation. That is a poor trade; sit in [cash and money markets](/blog/trading/cross-asset/cash-money-markets-the-underrated-asset) instead and wait. When short-term safe instruments yield 5% with no duration risk, the bar for taking duration is high.

### Barbell versus bullet: how to hold it

Two classic ways to structure a bond allocation, worth knowing by name:

- **Bullet** — concentrate your bonds around a single target maturity (say, all 7-to-10-year bonds). Simple, and it pins your duration to one point. Good when you have a clear view on one part of the curve.
- **Barbell** — split between very short bonds (or cash) and very long bonds, with little in the middle. The short end gives you liquidity and low risk; the long end gives you the powerful hedge and the leveraged bet on falling rates. A barbell lets you hold *some* of the explosive long-duration hedge while keeping the rest of your money safe and flexible — you get convexity at the long end and dry powder at the short end. Many allocators prefer a barbell precisely because it separates the two jobs bonds do: ballast (long) and liquidity (short).

#### Worked example: sizing the bond sleeve and feeling the duration

You have a \$100,000 portfolio and decide on a classic **60/40**: \$60,000 in stocks, \$40,000 in government bonds. Now choose your duration, and see what it commits you to.

- **Short-duration sleeve (duration ≈ 2).** If yields rise 1 point, your \$40,000 bond sleeve falls about 2% → −\$800. If yields fall 1 point, it gains about \$800. Stable, boring, low hedging power.
- **Medium-duration sleeve (duration ≈ 7).** A 1-point yield rise costs about 7% → −\$2,800; a 1-point fall gains \$2,800. This is the typical "total bond market" choice — meaningful hedge, moderate volatility.
- **Long-duration sleeve (duration ≈ 18).** A 1-point yield rise costs about 18% → **−\$7,200**; a 1-point fall gains \$7,200. Enormous hedging power in a recession, but you are taking a serious rate bet.

Notice that the *same* \$40,000 allocation behaves like three completely different positions depending on duration. "40% in bonds" is meaningless without stating the duration — that is the number that sets your actual risk. In a recession-hedge regime you might *want* the long-duration sleeve for its explosive upside; in a rising-rate regime you'd want the short sleeve to avoid the −\$7,200 outcome.

The intuition: deciding *how much* in bonds is only half the decision — *which duration* is the other half, and it can swing your risk by nearly 10x.

### How much, in the end

There is no universal number, but the logic is consistent. The more you need stability and the closer you are to spending the money, the more government bonds (and the shorter their duration) you want — they are the asset that protects you from a stock crash *when the crash is a growth shock*. The younger you are and the longer your horizon, the less you need them, because over decades equities' **+6.5% real** return dwarfs bonds' **+1.7% real**, and you can ride out the volatility. And in every case, the *kind* of bond — its duration — should match your read of the regime: long when disinflation and recession risk dominate, short when inflation and hiking dominate.

That is the whole job of government bonds in a portfolio. They are the risk-free anchor — the ruler every other asset is priced against, and the ballast that steadies the ship. Just never forget the one condition stamped on the contract in invisible ink: *they are ballast against growth shocks, and dead weight against inflation shocks.* Know which storm you are in, and you will know whether your anchor is holding you steady or dragging you down.

## Further reading and cross-links

- [The map of asset classes: what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — where government bonds sit among every other asset, and how the pieces fit together.
- [Cash and money markets: the underrated asset](/blog/trading/cross-asset/cash-money-markets-the-underrated-asset) — the zero-duration alternative you hold when duration isn't paying you enough to take it.
- [Corporate credit: investment grade, high yield, and spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) — what you earn by adding the default risk that Treasuries deliberately avoid.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — what actually sets the yields that move every bond, and how the yield curve forecasts the cycle.
- [Real versus nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the real yield, not the headline yield, is the deepest driver of every asset's price.

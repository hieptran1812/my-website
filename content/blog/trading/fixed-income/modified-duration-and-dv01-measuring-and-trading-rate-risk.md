---
title: "Modified duration and DV01: measuring and trading rate risk"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the two numbers bond desks actually live by: modified duration, the percent your price moves per 1% in yield, and DV01, the dollars you make or lose per basis point — and how to turn either into a real hedge."
tags: ["fixed-income", "bonds", "duration", "modified-duration", "dv01", "pv01", "interest-rate-risk", "hedging", "treasury-futures", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — duration tells you *how much* a bond's price moves when rates move; this post turns that sensitivity into the two numbers desks trade on — modified duration (a percent) and DV01 (a dollar) — and shows you how to hedge with them.
> - **Modified duration** is Macaulay duration divided by one-plus-the-per-period-yield, and it reads directly as the **percent price change per 1% move in yield**. Our 5-year 4% bond has a modified duration of about **4.49**.
> - **DV01** (the "dollar value of an 01", also called PV01) is the **dollars you gain or lose for a one-basis-point move** — about **\$0.45 per \$1,000 bond**, or **\$449** on a \$1,000,000 position.
> - To get a dollar P&L, you don't need the percentage: **DV01 × (basis points moved)** is the answer. A 25 bp day on that \$1M position is roughly **\$11,200**.
> - You hedge by **matching DV01, not face value**: against the \$1M 5-year you short about **\$549,000** of the 10-year (or roughly **7 Treasury-note futures**), and the combined book stays within ~\$1,000 of flat across a ±100 bp move.
> - A portfolio's duration is the **market-value-weighted average** of its holdings' durations, and its DV01 is simply the **sum** of the holdings' DV01s — so one number sizes the rate risk of an entire book.

There is a moment every new bond trader has. They have just learned that a bond's price falls when rates rise, and that the size of the fall depends on something called duration. They feel like they understand interest-rate risk. Then a senior trader leans over and asks a deceptively simple question: *"What's your DV01?"* — and suddenly the percentages and the seesaws and the discount factors all collapse into a single demand for one number, in dollars, right now. How much do you make or lose if rates twitch one basis point? That number, and the few that surround it, are the working language of every fixed-income desk on earth.

This post is about turning the *idea* of rate sensitivity — duration — into the *tools* you actually trade with. If the [previous post on duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) gave you the bond's center of gravity, this one bolts that abstraction onto reality: it converts duration into a percent you can multiply by a rate move (**modified duration**), then into a dollar you can put on a risk report (**DV01**), and finally into a **hedge** that cancels your exposure. We will keep one example running the whole way through — a \$1,000,000 position in a plain 5-year, 4%-coupon bond — and by the end you will be able to compute its sensitivity, express it the way a desk does, and build the trade that neutralizes it.

![A before and after diagram showing Macaulay duration of four point five eight years on the left converting into a modified duration of four point four nine on the right by dividing by one plus the per period yield](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-1.png)

The diagram above is the mental model for the first half of this post: one small adjustment — dividing by one-plus-the-per-period-yield — turns *Macaulay duration*, which is a **time** measured in years, into *modified duration*, which is a **sensitivity** measured in percent-of-price per percent-of-yield. That conversion is the bridge from "when do I get my money back, on average?" to "how much does my price move?" Everything that follows — DV01, dollar P&L, hedging — is built on top of that one number. (Everything here is educational, not investment advice; the goal is to understand the mechanics, not to recommend a trade.)

## Foundations: the words you need before we do any math

Let's build the vocabulary from zero. If you have read the earlier posts in this series, treat this as a quick refresher; if not, do not skip it, because every later sentence leans on these definitions.

A **bond** is a tradable loan. You, the buyer, are the lender; the **issuer** is the borrower. The bond promises a fixed stream of cash: a periodic **coupon** (the interest) and the **face value** — also called **par**, almost always \$1,000 per bond — returned at the end. The **maturity** is when that final payment lands. Our running example is a **5-year \$1,000 par bond with a 4% coupon**, paying \$20 every six months (most bonds pay **semiannually**, twice a year) for five years, then \$1,000 back. When the market yield equals the coupon — here, 4% — the bond trades at exactly par, \$1,000.

A few units you will see constantly:

- A **basis point** (abbreviated *bp*, and pronounced "bip") is **one hundredth of a percent**: 0.01%. When a trader says "rates moved 25 bps," they mean 0.25%. The whole reason basis points exist is that bond yields move in tiny increments, and "a quarter of a percent" is clumsier than "25 bps."
- The **yield** — more precisely the **yield to maturity (YTM)** — is the single interest rate that makes the bond's future cash flows, discounted back to today, equal its current price. It is the bond's true return if you buy now and hold to maturity. Price and yield move in opposite directions: this is the [price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).
- **Present value** is what a future dollar is worth today, after *discounting* it back at the yield. A dollar due in five years is worth less than a dollar in hand, because today's dollar can earn interest in the meantime. The whole machinery of [discounting cash flows](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) is the foundation underneath everything here.

Now the two ideas this post is built on. The first is **duration**, which we met in the [previous post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income). There are actually two flavors, and the difference matters enormously here:

- **Macaulay duration** is the *weighted-average time you wait to get your money back*, where each cash flow is weighted by its present value. It is measured in **years**. Our 5-year bond's Macaulay duration is about **4.58 years** — a bit less than its 5-year maturity, because the coupons pull the average wait forward.
- **Modified duration** is Macaulay duration *adjusted* so it reads as a **price sensitivity** rather than a time. It is the headline number of this post, and the next section builds it from scratch.

The second idea is the **price-sensitivity question itself**, which has two equally valid answers — a percent and a dollar. The percent answer is modified duration; the dollar answer is **DV01**. Most of this post is about computing both, seeing how they relate, and using them. The single most important habit to form is this: *desks think in dollars*. A percentage is a description; a dollar is a position. The moment you can say "this position is \$449 per basis point," you have stopped describing risk and started measuring it.

### The one relationship that does all the work

Everything in this post is a consequence of a single approximation: for a small change in yield, the percentage change in a bond's price is (negative) modified duration times the change in yield.

$$\frac{\Delta P}{P} \approx -D_{\text{mod}} \times \Delta y$$

- $\Delta P / P$ is the **percentage change in price** (what you want).
- $D_{\text{mod}}$ is the **modified duration** (the sensitivity, a pure number).
- $\Delta y$ is the **change in yield**, expressed as a decimal (a 1% move is 0.01; one basis point is 0.0001).
- The **minus sign** encodes the seesaw: when yield rises ($\Delta y > 0$), price falls ($\Delta P < 0$).

Read it in plain English: *price moves by duration-many percent for every one-percent move in yield, in the opposite direction.* If $D_{\text{mod}} = 4.49$ and yields rise 1% ($\Delta y = 0.01$), the price falls about $4.49 \times 0.01 = 4.49\%$. That is the whole engine. DV01 is just this same relationship rescaled into dollars per basis point, and a hedge is just two of these sensitivities arranged to cancel. Hold this equation in your head; the rest is bookkeeping.

## From Macaulay duration to modified duration

Macaulay duration is a beautiful idea — the balance point of a bond's cash flows in time — but it answers the wrong question for a trader. A trader does not care *when* the money arrives on average; they care *how much the price moves*. The fix is small and exact. Modified duration is Macaulay duration divided by one-plus-the-per-period-yield:

$$D_{\text{mod}} = \frac{D_{\text{mac}}}{1 + y/k}$$

- $D_{\text{mac}}$ is the **Macaulay duration** in years.
- $y$ is the **annual yield** (here 4%, or 0.04).
- $k$ is the **number of coupon periods per year** (semiannual means $k = 2$).
- $y/k$ is therefore the **per-period yield** (here 0.04 / 2 = 0.02, or 2% per six months).

The division by $(1 + y/k)$ looks like a fussy technicality, but it has a clean meaning. Macaulay duration measures sensitivity to the *log* of (one plus the yield); the price actually responds to the *level* of the yield, and the conversion factor between the two is exactly $1 + y/k$. The practical upshot is that modified duration is always a little *smaller* than Macaulay duration, and it carries the right units: **percent price change per one-percentage-point change in yield**.

#### Worked example: converting our bond's duration

Our 5-year \$1,000 bond at a 4% yield has a Macaulay duration of **4.58 years** (computed in the [duration post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income); it is the present-value-weighted average wait across the ten coupons and the final principal). To get modified duration, divide by one-plus-the-per-period-yield:

> $D_{\text{mod}} = \dfrac{4.58}{1 + 0.04/2} = \dfrac{4.58}{1.02} = \textbf{4.49}.$

So our bond's modified duration is about **4.49**. Read that number as a prediction: a 1% rise in yield should cut the price by roughly 4.49%. Let's check it against a brute-force re-pricing. Price the bond at 4% (it's par, \$1,000.00); re-price it at 5% (it falls to \$956.24). The actual change is **−4.38%**. The modified-duration prediction was −4.49%. The two agree to within about a tenth of a percent — and the small gap is *convexity*, the curvature we will meet shortly.

*The lesson: dividing Macaulay duration by one-plus-the-per-period-yield converts a wait time into a price sensitivity — modified duration 4.49 means "expect about a 4.49% price move per 1% in yield."*

Notice what modified duration buys you. You no longer have to re-price the bond from scratch every time rates move. One multiplication — duration times the rate move — gives you the answer instantly, accurately enough for any small move. That is exactly why it is the number a trader carries in their head. A 7-duration bond moves about 7% for a 1% rate change; a 2-duration bond moves about 2%. Once you know the duration, you know the bond's rate personality.

### Why the approximation is good, and when it isn't

The duration estimate is a *straight line*, and the true price–yield relationship is a *curve*. For small moves they are nearly identical; for large moves they separate.

![A line chart showing the true convex price yield curve in blue and the straight duration tangent line in orange touching at the par point, with the straight line sitting below the true curve at both extremes](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-4.png)

The chart shows the picture. The blue curve is the bond's *true* price at every yield — it bows, because of convexity. The orange dashed line is the *duration estimate*: a straight line tangent to the curve at today's yield (4%, \$1,000). Right at the tangent point, the line and the curve are the same. Step a little to either side and they stay close. Step far — out to 1% or 7% — and the straight line sits *below* the true curve by about \$10 to \$11. The duration estimate always *understates* the price, because the curve bends away from the line in both directions. That systematic undershoot is convexity, and it is good news for the bondholder: you gain a little more than duration predicts when rates fall, and lose a little less than it predicts when rates rise. We give convexity [its own post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story); for now, just know that duration is a first-order approximation that is excellent for the small, everyday moves desks actually trade, and needs a convexity correction only for big shocks.

#### Worked example: predicting a price move from duration alone

Suppose all you know is that a bond has a modified duration of 4.49 and that yields just rose by 30 basis points (0.30%). You can predict the price move without touching the cash flows:

> $\dfrac{\Delta P}{P} \approx -4.49 \times 0.0030 = -0.01347 = \textbf{−1.35\%}.$

On a \$1,000 bond that is about −\$13.47; on a \$100,000 position, about −\$1,347. Re-pricing the bond exactly gives −1.34%, so the duration estimate is off by about a hundredth of a percent — invisible for a move this size.

*The lesson: for everyday moves, modified duration is all you need — multiply it by the rate change and you have the percent price move, no re-pricing required.*

### When the formula breaks: effective duration

Modified duration, as we computed it, assumes the bond's cash flows are *fixed* — \$20 every six months and \$1,000 at the end, come what may. That assumption holds for a plain Treasury or a standard corporate bond. But a huge slice of the bond market has cash flows that *change when rates change*, and for those bonds the tidy $D_{\text{mac}} / (1 + y/k)$ formula gives the wrong answer.

The two big examples are **callable bonds** and **mortgage-backed securities**. A callable bond lets the issuer buy the bond back early at a fixed price; when rates fall, the issuer calls the bond (refinancing cheaply) and your bond's life suddenly shortens. A mortgage-backed security passes through homeowners' mortgage payments, and when rates fall, homeowners refinance, prepaying their loans — again shortening the security's life exactly when you'd want it to lengthen. In both cases the cash-flow schedule itself moves with the yield, so you cannot just discount a fixed stream and divide.

The fix is **effective duration**, computed not from a formula but by *brute force*: bump the yield down a little, re-price the bond (re-running whatever option or prepayment model determines its cash flows); bump the yield up the same amount, re-price again; and measure how much the price moved per unit of yield. In symbols:

$$D_{\text{eff}} = \frac{P_{-} - P_{+}}{2 \times P_0 \times \Delta y}$$

- $P_{-}$ is the **price after a small yield *decrease***.
- $P_{+}$ is the **price after the same yield *increase***.
- $P_0$ is the **starting price**.
- $\Delta y$ is the **size of the yield bump** (say 0.0025, or 25 bps).

This "bump and reprice" method is exactly how every risk system actually computes duration in production — for plain bonds it reproduces modified duration to the decimal, and for option-laden bonds it captures the cash-flow shifts that the closed-form formula cannot.

#### Worked example: effective duration of our plain bond

Let's run the bump-and-reprice on our 5-year 4% bond to confirm it reproduces modified duration. Bump the yield ±25 bps around 4%:

- $P_{-}$ (yield 3.75%) = **\$1,011.30.**
- $P_0$ (yield 4.00%) = **\$1,000.00.**
- $P_{+}$ (yield 4.25%) = **\$988.84.**

> $D_{\text{eff}} = \dfrac{1011.30 - 988.84}{2 \times 1000.00 \times 0.0025} = \dfrac{22.46}{5.00} = \textbf{4.49}.$

It lands on 4.49 — the same modified duration we got from the formula, because this bond's cash flows don't move. The power of the method is that for a callable bond or an MBS, where $P_{-}$ and $P_{+}$ are computed by a model that *changes the cash flows*, the same arithmetic captures a duration the formula would miss entirely. A callable bond near its call price can even show *negative* effective duration over part of the yield range — its price barely rises as rates fall, because the market expects it to be called away — a phenomenon impossible for a fixed-cash-flow bond and central to why [mortgage bonds have negative convexity](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity).

*The lesson: when a bond's cash flows shift with rates, drop the formula and measure duration by bumping the yield and re-pricing — effective duration is what risk systems actually compute, and it's the only honest measure for callables and mortgages.*

## DV01: the dollar value of a basis point

Modified duration is a percent. But a risk manager does not ask "what percent will you lose?" — they ask "how many dollars?" To answer in dollars, we rescale the sensitivity into the unit desks actually quote: **DV01**, the *dollar value of a one-basis-point move*. (You will also hear **PV01**, "present value of an 01," and **BPV**, "basis-point value" — they all mean the same thing.)

DV01 is just modified duration, multiplied by the price to get into dollars, and by one basis point to get the per-bp figure:

$$\text{DV01} = D_{\text{mod}} \times P \times 0.0001$$

- $D_{\text{mod}}$ is **modified duration** (4.49 for our bond).
- $P$ is the **price** in dollars (the full dollar value of the position, not the per-100 quote).
- $0.0001$ is **one basis point** as a decimal.

The logic is direct: modified duration times $P$ gives you the dollar price change for a *full 1%* move; multiply by 0.0001/0.01 = 0.01... — actually, let's be careful. $D_{\text{mod}} \times P$ is the dollar change for a 1.00 (i.e. 100%) move in yield, which is enormous and hypothetical; multiply by 0.0001 to shrink it to a single basis point. The result is a small, concrete dollar figure: what you make or lose per tick.

#### Worked example: the DV01 of one bond and of the position

Take our 5-year \$1,000 bond at par. Its DV01 per bond is:

> $\text{DV01}_{\text{bond}} = 4.49 \times \$1{,}000 \times 0.0001 = \textbf{\$0.4491}.$

So a single basis point moves one bond's price by about **45 cents**. Let's sanity-check it by re-pricing: at a 4.00% yield the bond is \$1,000.00; at 4.01% it is \$999.5509. The drop is \$0.4491 — exactly the formula's answer.

Now scale it to the position. A \$1,000,000 face position in a bond trading at par is **1,000 bonds** (\$1,000,000 ÷ \$1,000). The position DV01 is just the per-bond DV01 times the number of bonds:

> $\text{DV01}_{\text{position}} = \$0.4491 \times 1{,}000 = \textbf{\$449.13 per basis point}.$

That is the number the senior trader was asking for. Your \$1,000,000 5-year position makes or loses about **\$449 for every basis point** the yield moves.

*The lesson: DV01 collapses everything — duration, price, position size — into one dollar figure, the gain or loss per basis point, which is the unit a desk budgets risk in.*

![A grid figure showing how each cash flow's present value moves a sliver for one basis point and how those slivers sum to about forty five cents per bond and four hundred forty nine dollars on the whole position](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-2.png)

The figure traces where that \$449 comes from. Each individual cash flow's present value barely flinches for a single basis point — the six-month coupon's PV moves a hundredth of a cent, the final \$1,020 payment's PV moves about four-tenths of a cent. But summed across all eleven cash flows you get \$0.4491 per bond, and summed across 1,000 bonds you get \$449 per basis point on the position. The right column shows why traders love DV01: it scales linearly. A 25-basis-point day moves the position about \$11,200; a 50-basis-point day, about \$22,500. No re-pricing, no percentages — just DV01 times basis points.

### Turning DV01 into dollar P&L

This is the payoff of working in DV01: dollar profit-and-loss is a single multiplication. For any rate move, your dollar P&L is DV01 times the number of basis points the yield moved (with the sign flipped, because price falls when yield rises):

$$\text{P\&L} \approx -\text{DV01} \times (\Delta y \text{ in basis points})$$

#### Worked example: a 40-basis-point day

Your \$1,000,000 5-year position has a DV01 of \$449. Overnight, a hot inflation report pushes the 5-year yield up 40 basis points. Your loss is:

> $\text{P\&L} \approx -\$449 \times 40 = \textbf{−\$17,960}.$

Re-pricing the bond exactly (4.00% → 4.40%) gives a loss of \$17,800 — so the DV01 estimate is within about \$160 on an \$18,000 move, an error of under 1%. For a desk marking risk in real time, that accuracy is more than enough, and it took one multiplication instead of re-discounting eleven cash flows.

Now flip it. The next day the report is revised and the yield falls back 40 basis points. Your gain is about +\$449 × 40 = **+\$17,960** — except convexity makes the actual gain slightly *larger* (about +\$18,120), because the price–yield curve bows in your favor. Over the round trip you come out a touch ahead of where DV01 alone predicts. That small asymmetry is convexity quietly working for you.

*The lesson: in DV01 terms, your dollar P&L is just DV01 times the basis-point move — the single most useful arithmetic on a bond desk.*

### DV01 grows with maturity

Two bonds with the same face value can have wildly different DV01s, because DV01 inherits duration's sensitivity to maturity. The longer the bond, the more its money sits in the distant future, the larger its duration — and so the larger its DV01.

![A line chart with DV01 per one thousand dollars of face on the vertical axis rising steeply with maturity on the horizontal axis, from about nineteen cents at two years to one dollar seventy four at thirty years](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-5.png)

The chart plots DV01 per \$1,000 of face against maturity, for 4%-coupon bonds priced at par. It climbs steeply: a 2-year bond's DV01 is about **\$0.19**, the 5-year is **\$0.45**, the 10-year is **\$0.82**, the 20-year is **\$1.37**, and the 30-year is **\$1.74**. Same \$1,000 of face, but the 30-year carries roughly **nine times** the per-bp risk of the 2-year. This is the single most important fact for sizing trades: *face value tells you almost nothing about rate risk; DV01 tells you everything.* A million dollars of 2-year notes and a million dollars of 30-year bonds are completely different bets, even though they cost the same.

#### Worked example: same dollars, very different risk

You have \$1,000,000 to put to work and you are deciding between the 2-year and the 30-year, both at par. Compute each position's DV01:

- **\$1M of the 2-year:** 1,000 bonds × \$0.19 = **\$190 per bp**. A 50 bp move swings you about \$9,500.
- **\$1M of the 30-year:** 1,000 bonds × \$1.74 = **\$1,738 per bp**. The same 50 bp move swings you about \$86,900.

Identical cash outlay, but the 30-year position takes more than nine times the rate risk. If you wanted the two positions to carry the *same* risk, you would hold far less of the 30-year — about \$109,000 of it, so its DV01 matched the 2-year's \$190. This is the seed of the whole hedging idea: when you want to match or cancel rate risk, you match DV01, never face value.

*The lesson: DV01 rises steeply with maturity, so equal dollar amounts of different bonds are utterly different rate bets — size by DV01, not by face.*

## Building a hedge: matching DV01, not face value

Here is where measurement becomes trading. Suppose you own the \$1,000,000 5-year position and you are nervous about rates rising. You do not want to sell — maybe you like the yield, or you are a market-maker who must hold inventory — but you want to *neutralize the rate risk*. The solution is a **hedge**: take an offsetting position in a related instrument so that whatever you lose on the bond when rates move, you make back on the hedge. The art is sizing the hedge correctly, and the rule is simple: **make the hedge's DV01 equal the position's DV01**. Match dollar risk, not dollar face.

Why DV01 and not face? Because, as we just saw, equal face does not mean equal risk. If you hedged your \$1M 5-year position by shorting \$1M of the 10-year, you would *over-hedge*: the 10-year has nearly double the DV01, so its P&L would swing almost twice as hard as your bond's, and you would end up with a net *short* rate position instead of a flat one. To cancel exactly, you short whatever *amount* of the 10-year carries the same \$449 of DV01 as your position.

The hedge ratio is the position's DV01 divided by the hedge instrument's DV01:

$$\text{hedge notional} = \frac{\text{DV01}_{\text{position}}}{\text{DV01}_{\text{hedge per unit}}} \times (\text{unit})$$

#### Worked example: hedging the 5-year with the 10-year Treasury

Your position: long \$1,000,000 of the 5-year, DV01 = **\$449**. Your hedge instrument: the 10-year Treasury, whose DV01 is **\$0.8176 per \$1,000 of face**. To neutralize \$449 of DV01, short enough 10-year face that its DV01 equals \$449:

> $\text{hedge face} = \dfrac{\$449.13}{\$0.8176} \times \$1{,}000 = \textbf{\$549,000 (rounded).}$

So you short about **\$549,000 face of the 10-year** — only **0.55×** the face of your position. The hedge ratio in face terms is 0.55, but the hedge ratio in *DV01* terms is exactly 1.00 — which is the whole point. After the trade, your book is **long \$1M 5-year + short \$549k 10-year**, and its net DV01 is \$449 − \$449 ≈ **\$0 per basis point**. You are rate-neutral.

*The lesson: a hedge is sized so the hedge's DV01 equals the position's DV01 — match dollar risk, and the offsetting face value falls out of the arithmetic, usually nothing like a 1-to-1 face match.*

![A two line chart showing the long bond P and L rising as yields fall and the short hedge P and L falling, with their thick green sum staying nearly flat along the zero line across a plus or minus one hundred basis point move in the ten year yield](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-3.png)

This is the influence figure — the proof that DV01-matching actually works. The blue dashed line is the long 5-year's P&L: it gains as yields fall (left) and loses as they rise (right), swinging tens of thousands of dollars across a ±100 bp move. The orange dashed line is the short 10-year hedge: a near-mirror image, gaining exactly when the bond loses. The thick green line is their **sum** — the hedged book's P&L — and it sits essentially flat along zero, never straying more than about \$1,000 from break-even even on a full 100 bp move. That is what a DV01-matched hedge buys you: the rate exposure is gone. The tiny residual dip (the green line bends very slightly negative at the extremes) is *convexity mismatch* — the 10-year is more convex than the 5-year, so the hedge isn't perfect for large moves — but for the everyday ±25 bp world, the book is flat to the dollar.

#### Worked example: the hedge in action on a 50 bp sell-off

Rates sell off hard: the curve shifts up 50 basis points across the board. Watch each leg:

- **Long 5-year:** DV01 \$449 × (−50 bp) = **−\$22,500** (re-pricing exactly: −\$22,166).
- **Short \$549k 10-year:** its DV01 is also \$449, and a short *gains* when prices fall, so +\$449 × 50 = **+\$22,500** (exactly: +\$21,924).
- **Net:** about −\$22,166 + \$21,924 = **−\$242.**

You lost \$242 on a move that would have cost the unhedged position \$22,166. The hedge absorbed over 98% of the loss. The \$242 residual is the convexity mismatch — small, and you could shrink it further with a second hedge instrument, but for the size of move this is a clean kill.

*The lesson: when the two legs carry equal DV01, a rate move that would have cost the position thousands nets out to near zero — the definition of a hedge that works.*

### Hedging with Treasury futures instead of a cash bond

In practice, desks usually hedge with **Treasury futures** rather than by shorting a cash bond, because futures are deeply liquid, cheap to trade, and don't require borrowing the bond to sell it short. A futures contract is a standardized agreement to deliver a Treasury at a future date; for our purposes, what matters is that each contract has its own DV01, and the hedge math is identical — just divide by the per-contract DV01 instead of the per-\$1,000 DV01.

#### Worked example: hedging with 10-year T-Note futures

The 10-year Treasury-note futures contract has a DV01 of roughly **\$65 per contract** (this depends on the contract's "cheapest-to-deliver" bond and changes over time, so treat \$65 as illustrative). To hedge your \$449 of DV01:

> $\text{contracts} = \dfrac{\$449}{\$65} \approx \textbf{7 contracts (short).}$

You sell about 7 contracts. Now your \$1M cash position is hedged with a handful of futures you can trade in seconds, rather than locating and shorting half a million dollars of a specific bond. The principle never changes: **count up the DV01 you want to cancel, divide by the hedge's DV01 per unit, and trade that many units.**

*The lesson: futures hedge the same way as cash bonds — DV01 divided by per-contract DV01 gives the contract count — but with far more liquidity and no short-borrow, which is why desks reach for them first.*

### A hedge is not "set and forget": DV01 drifts

The hedge you just built was correct *at one instant*. But DV01 is not constant — it changes as time passes and as rates move — so a hedge that is perfectly DV01-matched today will be slightly *mismatched* tomorrow. A hedge is a living thing that needs re-checking, and understanding why is what separates someone who can compute a hedge ratio from someone who can actually run a hedged book.

Two forces push the match out of alignment. The first is **time decay of duration**: every day, both your bond and your hedge get one day closer to maturity, so both their durations (and DV01s) fall. They do not fall at the *same rate*, though — the shorter 5-year ages faster, in duration terms, than the longer 10-year, so the ratio between the two DV01s slowly shifts. The second is the **convexity of each leg**: when rates move, each bond's DV01 changes, and because the 10-year is more convex than the 5-year, a big rate move changes the hedge's DV01 by more than it changes the position's. After a large move, the hedge that started matched is no longer matched, and you must re-size it. This re-sizing is called **rebalancing** (or, on an options-style book, *gamma hedging*).

#### Worked example: the hedge drifts after a big move

You set up the hedge at 4% yields: long \$1M 5-year (DV01 \$449), short \$549k 10-year (DV01 \$449), perfectly matched. Now suppose yields gap up 100 basis points to 5% and stay there. Recompute each leg's DV01 at the new, higher yield:

- **5-year position DV01 at 5%:** the bond's price has fallen to ~\$956 and its duration has eased slightly, so its DV01 is now about **\$426** per bp (down from \$449).
- **10-year hedge DV01 at 5%:** the more-convex 10-year's DV01 fell *further*, to about **\$0.74 per \$1,000**; on \$549k of face that is about **\$408** per bp.

The two legs that started perfectly matched at \$449 each have drifted apart to **\$426 vs \$408** — a net mismatch of about **\$18 per basis point** that did not exist when you put the hedge on. You are now slightly *under*-hedged: your long position is more sensitive than your short. To re-flatten, you would short a little more 10-year (about \$24,000 more face, adding \$18 of hedge DV01). This is *rebalancing*, and the lesson is that the drift is not random — the more-convex hedge always loses DV01 faster on a sell-off, so a 5s-vs-10s hedge systematically becomes under-hedged when rates rise and over-hedged when they fall. A desk re-checks the net DV01 of every hedged book at least daily — often continuously — and tops up or trims whenever the residual exceeds its risk limit. The cost of that rebalancing (the bid-ask spread paid each time you adjust) is a real, recurring expense of carrying a hedge, and it is the reason a "free" hedge is never quite free.

*The lesson: a DV01-matched hedge is matched only at a moment — duration decays with time and shifts with rates, so a hedged book must be re-checked and rebalanced, and that rebalancing carries a real transaction cost.*

### A note on hedging across the curve

The hedge above assumes a **parallel shift** — every maturity's yield moving by the same amount. Reality is messier: the yield curve can *steepen* (long rates rise more than short), *flatten*, or *twist*. When you hedge a 5-year with a 10-year, you are exposed to the curve *changing shape*, because the two points can move by different amounts. A DV01-matched 5-year-vs-10-year book is neutral to the *level* of rates but still has a view on the *slope*. Desks that want to be neutral to slope too use two or three hedge instruments at different maturities — a topic the curve-trading post takes up in [steepeners, flatteners, and butterflies](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies). For now, the key idea stands: DV01-matching cancels the first and biggest risk, the parallel move; finer structure needs finer hedges.

## Portfolio duration: one number for the whole book

So far we have one bond and one hedge. Real money managers hold hundreds of positions. The elegant fact that makes this tractable is that rate risk **adds up linearly** in two equivalent ways:

- A portfolio's **modified duration** is the **market-value-weighted average** of its holdings' durations.
- A portfolio's **DV01** is simply the **sum** of its holdings' DV01s.

These are two faces of the same coin. DV01 is additive because it is already in dollars — your total dollar risk is the sum of the dollar risks. Duration is a *weighted average* because it is a percentage, and percentages combine by value-weighting. Both give you a single rate-risk number for an entire book, which is what lets a manager say "the fund is at 6.9 years of duration" and have it mean something precise.

The portfolio duration formula:

$$D_{\text{port}} = \sum_i w_i \, D_i, \qquad w_i = \frac{\text{market value of holding } i}{\text{total portfolio value}}$$

- $D_i$ is the **modified duration** of holding $i$.
- $w_i$ is its **weight** — its market value divided by the total portfolio value.
- The weights sum to 1, so the result is a genuine average, somewhere between the shortest and longest holding's duration.

![A matrix table showing three Treasury holdings of two, ten, and thirty years with their market values, durations, weight times duration contributions, and DV01s, summing to a portfolio duration of about six point nine years and a DV01 near six hundred eighty eight dollars](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-6.png)

The matrix works a concrete book. A \$1,000,000 portfolio holds \$500,000 of the 2-year (duration 1.90), \$300,000 of the 10-year (duration 8.18), and \$200,000 of the 30-year (duration 17.38). Each row's contribution to portfolio duration is its weight times its duration — 0.95, 2.45, and 3.48 — and those sum to a **portfolio duration of 6.88 years**. The same rows' DV01s — \$95, \$245, and \$348 — sum to a **portfolio DV01 of \$688 per basis point**. Two routes, one answer: the book moves about \$688 for every basis point, the same as a single bond with 6.88 years of duration.

#### Worked example: the portfolio's duration and DV01

Let's compute it both ways and confirm they agree.

**Via weighted-average duration:**

> $D_{\text{port}} = (0.50)(1.90) + (0.30)(8.18) + (0.20)(17.38) = 0.95 + 2.45 + 3.48 = \textbf{6.88 years.}$

The portfolio DV01 then follows from the portfolio duration and total value:

> $\text{DV01}_{\text{port}} = 6.88 \times \$1{,}000{,}000 \times 0.0001 = \textbf{\$688 per bp.}$

**Via summed DV01s:**

> $\$95 + \$245 + \$348 = \textbf{\$688 per bp.}$

Both routes land on \$688. Now use it: if the whole curve sells off 30 basis points, the portfolio's loss is about \$688 × 30 = **−\$20,640**. One number sized the rate risk of three positions.

*The lesson: rate risk aggregates — portfolio duration is the value-weighted average of component durations, and portfolio DV01 is just the sum of component DV01s, so a single number captures a whole book's exposure.*

#### Worked example: rebalancing to a target duration

Portfolio duration is also a *control* — you can dial it to a target by shifting weights. Suppose you decide 6.88 years is too long; a recession scare aside, you want to cut the book's rate risk to a target duration of **4.0 years** without selling everything. The cleanest lever is to shift money out of the 30-year (duration 17.38) and into cash or short bills (duration ≈ 0).

How much 30-year do you sell? Each \$100,000 moved from the 30-year to cash removes $0.10 \times 17.38 = 1.74$ years of weighted duration. You need to cut 6.88 − 4.00 = 2.88 years. So you move about $2.88 / 17.38 \times \$1{,}000{,}000 \approx \$166{,}000$ from the 30-year into cash:

> New 30-year weight ≈ (\$200,000 − \$166,000) / \$1,000,000 = 0.034; new cash weight ≈ 0.166.
> New duration ≈ (0.50)(1.90) + (0.30)(8.18) + (0.034)(17.38) + (0.166)(0) ≈ **4.0 years.**

The book's DV01 drops from \$688 to about \$400 per bp — you have cut your rate risk by more than 40% with a single trade in the longest, most sensitive holding.

*The lesson: because duration is a weighted average, you steer a portfolio's rate risk by moving weight along the maturity spectrum — and the most sensitive long bonds give you the most steering per dollar moved.*

### The DV01 hedge table

When a desk hedges a real book, all of this lives in one table: the position, its DV01, the hedge instrument, its DV01, and the resulting hedge size. It is worth seeing the whole calculation laid out the way a risk sheet would show it.

![A matrix table laying out the hedge calculation row by row, the long one million dollar five year position with four hundred forty nine dollars of DV01, the ten year hedge instrument, the hedge ratio, and the resulting near zero net DV01 after shorting five hundred forty nine thousand dollars of the ten year](/imgs/blogs/modified-duration-and-dv01-measuring-and-trading-rate-risk-7.png)

The table is the worked hedge in one view. The top row is your position: long \$1,000,000 of the 5-year, +\$449 of DV01, fully exposed. The second row is the hedge instrument: the 10-year, at \$0.8176 of DV01 per \$1,000 face. The third row solves the hedge ratio — \$449 ÷ \$0.8176 × \$1,000 = \$549,000 of face to short, equivalently about 7 T-note futures. The bottom row is the result: long 5-year plus short \$549k 10-year nets to about \$0 per basis point — a rate-neutral book. This single table is, in miniature, what a fixed-income risk system computes across thousands of positions every night: each line's DV01, the hedges, and the residual exposure the desk is choosing to keep.

### The limit of a single number: when one duration isn't enough

Portfolio duration and total DV01 are wonderful summaries, but they hide an assumption that can bite you: they assume the whole yield curve moves *in parallel* — every maturity's yield shifting by the same number of basis points. A single duration number tells you what happens if 2-year, 10-year, and 30-year yields all rise 25 bps together. It tells you *nothing* about what happens if the 2-year rises 25 bps while the 30-year falls 25 bps, even though the total move "nets to zero." Yet that kind of curve reshaping — a *steepening* or *flattening* — is one of the most common things the bond market does.

The tool that fixes this is **key-rate duration** (also called *partial DV01* or *bucket DV01*). Instead of one duration for the whole portfolio, you compute a separate sensitivity to each *segment* of the curve — a 2-year key-rate DV01, a 5-year key-rate DV01, a 10-year, a 30-year, and so on — by bumping just one maturity at a time and re-pricing. The key-rate DV01s sum back to the total DV01, but they tell you *where on the curve* your risk lives, which the single number cannot.

#### Worked example: two books with the same duration, opposite curve bets

Consider two \$1,000,000 portfolios, each with a total DV01 of about \$688 and a duration near 6.9 years — identical by the single-number summary:

- **Book A (a "bullet"):** all \$1,000,000 in the 10-year. Its DV01 sits entirely in the 10-year bucket: \$818 of 10-year key-rate DV01, roughly zero elsewhere. (To match Book B's \$688 exactly you'd hold slightly less; treat the structure, not the rounding, as the point.)
- **Book B (a "barbell"):** split between the 2-year and the 30-year, sized to the same total DV01 — say \$190 of 2-year key-rate DV01 and \$498 of 30-year key-rate DV01, with nothing in the middle.

Now the curve *steepens*: short yields fall 20 bps and long yields rise 20 bps, with the 10-year roughly unchanged. By the single-duration summary, both books are "the same," so both should be flat. They are not:

> **Book A:** 10-year barely moved → P&L ≈ **\$0.** The bullet is concentrated exactly where nothing happened.
> **Book B:** +\$190 × 20 (2-year rallied) − \$498 × 20 (30-year sold off) ≈ +\$3,800 − \$9,960 = **−\$6,160.** The barbell got hurt by the steepening because its risk sits at the two ends that moved.

Same duration, same total DV01, wildly different outcomes — because the *shape* of the move mattered and a single number couldn't see it. This is exactly why curve traders think in key-rate durations, and why a risk report shows DV01 *by maturity bucket*, not just one grand total. The trades that exploit these shape moves — steepeners, flatteners, butterflies — are the subject of [trading the curve](/blog/trading/fixed-income/trading-the-curve-steepeners-flatteners-and-butterflies).

*The lesson: total duration assumes a parallel move; when the curve changes shape, two books with identical duration can have opposite P&L, which is why desks break risk into key-rate (per-maturity) DV01s.*

## Common misconceptions

**"You hedge a bond by shorting the same dollar amount of another bond."** This is the most common and most expensive hedging mistake. Equal face value does *not* mean equal risk, because DV01 depends on duration, which depends on maturity. Shorting \$1M of the 10-year against a long \$1M of the 5-year nearly *doubles* your hedge's sensitivity relative to your position — you would end up net short rates, not flat. The correct rule is to match **DV01**: short whatever face of the hedge carries the same dollars-per-basis-point as your position (here, \$549k of the 10-year, not \$1M).

**"Modified duration and Macaulay duration are the same thing."** They are close — modified is Macaulay divided by $(1 + y/k)$ — but they answer different questions and carry different units. Macaulay duration is a **time** (years): the average wait for your money. Modified duration is a **sensitivity** (percent per percent): how much the price moves. Confusing them leads to small but real errors, and at low yields the gap between them shrinks while at high yields it widens. Use Macaulay to understand a bond's center of gravity; use modified (and DV01) to size risk and trades.

**"DV01 is a fixed property of a bond."** It is not constant — it drifts as the bond ages and as yields move. As a bond approaches maturity, its duration falls, so its DV01 shrinks (a bond with one year left has far less DV01 than it did at issue). And because the price–yield relationship is curved, DV01 itself changes as the yield changes — it is slightly larger at low yields and smaller at high yields. Desks recompute DV01 continuously; the \$449 figure is a snapshot at today's price and yield, not a permanent label.

**"A hedge with matched DV01 is perfectly riskless."** No — it neutralizes the *parallel* rate move, which is the biggest risk, but it leaves two others. First, **curve risk**: if the 5-year and 10-year move by different amounts (the curve steepens or flattens), a 5s-vs-10s hedge is no longer flat. Second, **convexity mismatch**: the two bonds have different curvature, so the hedge drifts for large moves (the small residual you saw in the P&L chart). DV01-matching kills the first-order risk; the leftover second-order risks are why real desks use multiple hedge points and watch convexity.

**"A higher coupon means a higher DV01."** Counterintuitively, the opposite tends to hold for bonds of the same maturity. A higher coupon hands you more cash early, pulling the bond's center of gravity forward, shortening its duration — and therefore *lowering* its DV01. A zero-coupon bond of a given maturity has the *highest* duration and DV01 of any bond at that maturity, because all of its money sits at the far end. So when sizing risk, never assume the fat-coupon bond is the riskier one; check the duration.

**"You can ignore convexity if you use DV01."** For small, everyday moves, yes — DV01 is accurate to within a fraction of a percent. But DV01 is a *linear* measure, and for large shocks the linearity breaks down: a DV01-based loss estimate will *overstate* your loss when rates rise and *understate* your gain when they fall, because the true price–yield curve bows in the holder's favor. On a 100 bp move in a long bond, the convexity correction can be worth a meaningful fraction of the total move. DV01 is the first term; convexity is the second, and big positions track both.

## How it shows up in real markets

**Dealer desks quote and risk-manage everything in DV01.** Walk onto any government-bond or swaps desk and you will hear risk discussed almost entirely in DV01 (often just "dollars-per-bp" or "the oh-one"). A market-maker who has bought \$50 million of the 7-year from a client doesn't think "I'm long \$50 million of face"; they think "I just picked up about \$33,000 of DV01, and I need to sell roughly \$33,000 of DV01 in futures to flatten before the close." Position limits, end-of-day risk reports, and the firm-wide rate exposure that risk committees watch are all denominated in DV01. It is the lingua franca precisely because it is additive and in dollars — you can sum the DV01 of a bond, a future, a swap, and an option into one firm-wide number, which you cannot do with durations or percentages directly.

**The 2022 bond rout, priced in DV01.** When the Federal Reserve hiked from near zero to over 4% in 2022, the damage to bond portfolios was, mechanically, just DV01 times a very large number of basis points. The Bloomberg US Aggregate index ran a duration of roughly 6.5 years; on a roughly \$1 trillion class of holdings, every basis point of yield was worth on the order of hundreds of millions of dollars of market value, and yields rose by *hundreds* of basis points. A pension fund running, say, \$10 billion at 7 years of duration carried a DV01 near \$7 million — meaning a 200 bp rise in yields implied a market-value loss on the order of \$1.4 billion. The funds that had measured their DV01 honestly saw the risk coming; the ones that thought of their bonds as "safe" because they were government-guaranteed had measured credit risk and ignored rate risk entirely. The cross-asset story of that year — stocks and bonds falling together — is dissected in [the 2022 case study](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

**Silicon Valley Bank: a DV01 nobody hedged.** SVB held a large book of long-dated, low-coupon Treasuries and mortgage bonds bought when rates were near zero — the maximum-duration, maximum-DV01 combination. Critically, the bank had *not* hedged that DV01 (it had largely removed its interest-rate hedges to book more current income). When the Fed hiked through 2022–23, the market value of that book fell by an amount that was, again, just DV01 times the basis-point move — and the resulting unrealized losses eventually exceeded the bank's equity. When depositors fled and SVB had to sell, the paper loss became real and the bank failed in days. The lesson sits exactly on this post's theme: SVB's positions had a large, measurable DV01, and an unhedged DV01 is an unhedged bet on rates, regardless of how creditworthy the bonds are. The full anatomy is in [the SVB and Credit Suisse case study](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

**Treasury-futures hedging in the basis trade.** Hedge funds running the "Treasury cash-futures basis trade" are doing DV01-matching at industrial scale: they buy cash Treasuries and short the corresponding futures, sized so the two legs' DV01s cancel, capturing a tiny price difference with leverage. The trade works only because the DV01 match makes the position nearly rate-neutral — the fund is betting on the small basis converging, not on the direction of rates. When the match breaks down under stress (as in March 2020, when liquidity vanished and the cash and futures legs decoupled), the supposedly neutral trade can blow up, which is why regulators watch the size of this DV01-matched book closely. It is a vivid reminder that DV01-matching neutralizes the *level* of rates but not every risk a position carries.

**Liability-driven investing and the duration of pension promises.** Pension funds and insurers owe future payments that themselves have a duration — a stream of promised pensions decades out behaves like a very long bond, often 15–20 years of duration. Liability-driven investing (LDI) sizes the fund's *asset* DV01 to match the *liability* DV01, so that when rates move, the assets and liabilities move together and the funding gap stays stable. This is portfolio-duration matching turned into a whole investment philosophy. When it goes wrong — as in the UK in 2022, when gilt yields spiked and leveraged LDI hedges faced margin calls they couldn't meet — the unwind is forced selling that pushes yields further, a feedback loop the [duration-matching and immunization post](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge) examines in detail.

**Index funds and the published duration number.** Every bond ETF and mutual fund publishes its portfolio's *effective duration* — that single weighted-average number from this post — precisely because it is the most useful one-line summary of the fund's rate risk. When you read that a total-bond-market fund has a duration of 6.0, you can immediately estimate that a 1% rise in rates would cost it about 6% of its value, and a 25 bp move about 1.5%. That published number is doing exactly what this post built: compressing thousands of holdings into one figure you can multiply by a rate move. The macro lens on what *drives* those rate moves — and therefore what the duration number is exposed to — lives in [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

## When this matters to you, and where to go next

You meet modified duration and DV01 the moment you look at any bond fund. The "average duration" on the fact sheet is the number that tells you how much the fund will move when rates move — multiply it by the expected rate change and you have your estimated gain or loss, no finance degree required. A retiree comparing a short-duration fund (duration 2) to an intermediate one (duration 6) is choosing, whether they know it or not, between a position that moves 2% per 1% in rates and one that moves 6%. The same arithmetic governs the rate sensitivity of your bond holdings, the way your bank manages its balance sheet, and the hedges that keep the world's largest market functioning.

If you take one habit from this post, make it this: **think in DV01.** A percentage describes a bond; a dollar-per-basis-point positions it. Once you can look at any holding and say "that's X dollars per bp," you can size it, compare it, aggregate it, and hedge it — which is the entire job. The natural next step is to correct DV01 for the curvature we kept brushing against: that is [convexity, why duration is not the whole story](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). From there, the duration track turns to the two-sided nature of rate moves in [reinvestment risk and the two faces of yield](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield), and to how institutions hedge entire liability streams in [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge). For the full derivation behind every formula here — the calculus that turns the pricing equation into duration and DV01 — the quantitative companion is [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics). Once DV01 is a reflex, you have stopped guessing at bond risk and started measuring it to the dollar.

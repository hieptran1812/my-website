---
title: "Mental math and arithmetic speed for quant trading interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Market-making firms test raw arithmetic speed under a stopwatch because traders price and update in their heads all day. This deep dive builds the whole mental-math toolkit from zero, drills every technique on real numbers, and shows exactly how the tests and the desk work."
tags:
  [
    "quant-interviews",
    "mental-math",
    "arithmetic-speed",
    "zetamac",
    "optiver-80-in-8",
    "market-making",
    "trading-interview-prep",
    "estimation",
    "basis-points",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — market-making and prop firms put a stopwatch on your arithmetic because a trader prices, hedges, and re-prices in their head all day, and a slow answer in live markets is a wrong answer. A small, learnable toolkit turns a panic test into routine.
>
> - The two famous gates are **Zetamac** (a free two-minute drill of add, subtract, multiply, divide) and **Optiver's 80-in-8** (eighty questions in eight minutes). Both reward a *steady* pace, not a hero sprint.
> - The whole toolkit is small: **add left-to-right**, **subtract via complements**, a handful of **multiplication shortcuts** (×11, ×5/25/50, near-100, difference of squares), **memorized fractions ↔ percents**, **squaring tricks**, and **basis points on a price**.
> - **Speed and accuracy trade off.** On a penalized test, your net score peaks at a pace just inside your error limit — past that, wrong answers erase the gains.
> - The number to remember: a fill of **1,200 shares at $0.05 of edge is $60** of profit, and a trader computes that the instant the trade prints. The same one-line arithmetic is the test *and* the job.
> - Practice the way you'll be tested: timed, mixed, daily. Two weeks of ten-minute sessions moves most people from "panicked" to "fluent."

Here is a number that surprises almost everyone the first time they hear it: a strong candidate at Optiver answers around **80 simple arithmetic questions in 8 minutes** — one every six seconds, for eight straight minutes, with no calculator and no scratch paper. That is not a trick question or a brain-teaser round. It is plain addition, subtraction, multiplication, and division, and the firm cares enough about it to use it as a first filter on people applying to trade hundreds of millions of dollars.

Why would a sophisticated trading firm screen on something a fourth-grader technically *can* do? Because a market maker — a trader whose job is to continuously quote a price to buy and a price to sell — does this exact arithmetic, in their head, hundreds of times an hour, while a market moves under them. The test is not a proxy for intelligence. It is a *direct sample of the job*. The diagram below is the mental model for the whole post: every quote, hedge, and profit-and-loss check a trader makes is one small arithmetic step that has to clear in seconds.

![A market maker takes a price input, runs one mental arithmetic step, and emits a quote, a hedge size, and a P&L update, all under time pressure](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-1.png)

This article builds the entire skill from zero. We will define every term, drill every technique on real numbers, and frame all of it for the interview — including a section of fully worked timed problems you can solve along with me. By the end you should understand not just *how* to compute fast, but *why* this particular skill sits at the front door of firms like Jane Street, Optiver, SIG, IMC, Hudson River Trading, Jump, and Citadel Securities. A quick honesty note before we start: this is educational. Nothing here is financial advice, and arithmetic speed is necessary but nowhere near sufficient to trade well.

## Foundations: why speed matters, and what the tests actually are

Let us start with zero assumptions and build up the vocabulary you need.

### What a market maker does, in one paragraph

A **market maker** is a trader who stands ready to both buy and sell a security at all times, publishing two prices: a **bid** (the price they will buy at) and an **ask** or **offer** (the price they will sell at). The gap between them is the **bid-ask spread** — the small margin the market maker earns for providing liquidity, which is just a fancy word for "being willing to trade when you want to." If a stock is quoted **42.16 bid, 42.20 offered**, the market maker buys at $42.16 and sells at $42.20, and that **4-cent spread** (also written as **4 ticks**, where a *tick* is the smallest price increment — here $0.01) is their gross edge per round trip. The word **edge** means exactly this: the expected profit per unit of trading, the difference between the price you pay and what the thing is really worth.

The market maker's whole day is recomputing those two numbers as information arrives. A buyer lifts the offer, the stock ticks up, an index future jumps, a related option moves — and the fair value in the trader's head updates, so the quote updates. Each update is arithmetic: add a few ticks, take a percentage of a price, multiply a share count by an edge, divide to get a ratio. None of it is hard math. All of it is *fast* math, done while the clock and the market both move.

### Why "fast" is the whole point

Slowness is not a minor inconvenience on a trading desk; it is a direct loss. If your fair value is $42.18 and the true value just moved to $42.25, every second your quote lags is a second someone can buy from you at $42.20 a thing worth $42.25 — they pocket your $0.05 and you eat the loss. This is called being **picked off** or **adversely selected**, and it is the central risk of quoting. The faster and more accurately you reprice, the less you get picked off. So the firm's logic is brutally simple: a candidate who can do the arithmetic in two seconds will get picked off less than one who needs ten. The test measures the thing that the job pays for.

### The two tests you will actually face

There are two standard arithmetic gates in this corner of finance, and you should know both cold.

**Zetamac.** This is a free web drill (the "Arithmetic Game") that became the de facto practice standard because its default settings mirror what firms ask. The default game runs **two minutes** and draws random problems from four operation bands. Your **score** is simply the number you get right in the time. There is no penalty for a wrong answer on Zetamac itself — you just lose the seconds you spent — so the strategy is "go as fast as you can while staying accurate." The default ranges are worth memorizing because they tell you what to practice.

![The default Zetamac game mixes four operation bands across addition, subtraction, multiplication, and division, with rough score bands](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-12.png)

As the table shows, addition and subtraction draw two numbers roughly in the **2 to 100** range; multiplication is a **small factor (2 to 12) times a larger one (2 to 100)**; and division is always *exact* (it is just multiplication run backwards, e.g. "91 ÷ 7"). A rough read on scores: around **40** in two minutes is solid for a beginner, the **mid-50s** is strong, and **70 and up** is very strong. Most desks like to see candidates comfortably in the 60s or higher, though the exact bar varies by firm and role.

**Optiver's 80-in-8.** Optiver, a large market-making firm, runs a famous timed test of **80 questions in 8 minutes**. The format is stricter than Zetamac in two ways. First, the questions can include *negative numbers and decimals*, which Zetamac's defaults usually do not — you might see `-7 × 12` or `6.5 + 3.8`. Second, and more importantly, **wrong answers are penalized**: the test is scored net of mistakes, so blindly sprinting and guessing actively hurts you. The arithmetic is the same family of operations, but the scoring changes the optimal strategy, as we will see in the speed-vs-accuracy section.

The single most useful fact about 80-in-8 is its *pace*. Eighty questions in eight minutes is exactly **ten questions per minute**, which is **six seconds per question** on average. That is the metronome you train to.

It is worth knowing that the exact format and bar differ across firms, even though the underlying skill is identical. Different desks weight the arithmetic gate differently: a pure market-making seat at a firm like Optiver or IMC leans hardest on raw speed because the role *is* fast repricing, while a more research- or model-driven seat at some quant funds treats arithmetic as a baseline screen and puts more weight on the probability and statistics rounds. Some firms administer the test online before any human conversation; others run it live with an interviewer reading questions aloud, which adds the pressure of performing in front of someone. None of this changes what you practice — fast, accurate, mixed arithmetic — but it does mean you should not over-index on hitting one specific magic number. Clear the bar comfortably and move on to the rounds that decide the offer.

![Eighty questions in eight minutes is a constant rate of ten per minute, so each answer must clear in six seconds or less on average](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-2.png)

The pacing line in the figure is the heart of the test. Plot questions completed against minutes elapsed, and a perfect run is the straight diagonal from the origin to "80 done at 8 minutes." Stay above that line and you are ahead of pace; drop below it and you are behind. The trick is that you do not need to be a freak — you need to be *steady*. Banking eleven a minute for the first few minutes buys you a small cushion; the goal is to never fall behind the line, because falling behind tempts you into the rushing that produces errors.

### The units a trader thinks in

Two unit conventions appear constantly and trip up beginners, so define them now.

A **percentage** is parts per hundred: 5% of something is 5 out of every 100, i.e. multiply by 0.05. A **basis point** (abbreviated **bp**, said "bip") is one *hundredth* of a percent — **0.01%**, or one part in **10,000**. Traders use basis points because they are a flat, scale-free unit: a "10 bps move" means the same proportional thing whether the price is $40 or $4,200, so you can compare a small stock and a giant index on one ruler. We will spend a whole section making basis-point arithmetic automatic, because it shows up in spreads, fees, interest rates, and risk limits all day.

The other convention is the old **fractional pricing** of US stocks. Until 2001, American stock prices moved in fractions of a dollar — eighths and sixteenths — so a quote might read **$3⅜** rather than $3.375. Decimalization killed this for stocks, but fractions survive in bond markets (Treasuries trade in 32nds) and in interview questions, and the fluency it demands — converting `3/8` to `0.375` instantly — is exactly the fraction-to-decimal skill the job rewards. So we keep it.

With the vocabulary in place, let us build the toolkit. The organizing principle of everything that follows is the same: **trade hard arithmetic for easy arithmetic by changing how you represent the problem.**

## Addition and subtraction: go left-to-right, and use complements

The way you were taught to add in school — line the numbers up, start at the ones column, carry to the left — is optimized for paper and for getting a guaranteed-correct answer. It is *terrible* for speed, for one reason: the answer arrives last, all at once, after you have finished the entire problem. You cannot start speaking until you are completely done. Worse, you have to *hold* the running carries in your head while you work, which is exactly the kind of memory load that breaks under time pressure.

### Add from the big end first

Fast mental addition runs the other way: **left to right**, biggest place value first. The reason is psychological as much as mathematical. When you add the hundreds first, the most significant part of the answer is locked in immediately, and you can literally start saying it while you compute the rest. The small corrections at the end barely move the number you have already committed to.

![Left-to-right addition computes hundreds, then tens, then ones, so the answer forms on your lips while you are still computing the small digits](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-3.png)

#### Worked example: 468 + 357

Watch the running total form, big to small, exactly as in the figure:

1. **Hundreds first:** 400 + 300 = **700**. Say "seven hundred" to yourself.
2. **Tens next:** 60 + 50 = 110, so 700 + 110 = **810**. Update to "eight ten."
3. **Ones last:** 8 + 7 = 15, so 810 + 15 = **825**.

The answer is **825**. Notice that by the end of step 1 you already knew the answer was "somewhere in the 700s–800s," and by step 2 you were within 15 of the final number. The right-to-left school method gives you *nothing* until the very last carry resolves. Left-to-right gives you a usable estimate immediately and refines it — which is precisely how a trader wants to think, because a fast approximate price now beats a perfect price a moment too late.

The one-sentence intuition: **start from the digits that matter most, so your answer is always "good enough" early and just gets sharper.**

There is a second worked case worth seeing, because it shows how to handle the carry without losing the left-to-right flow. Add **675 + 248**. Hundreds: 600 + 200 = 800. Tens: 70 + 40 = 110, so 800 + 110 = 910 — notice the tens overflowed past 100 and bumped the hundreds, which left-to-right handles naturally because you simply carry into the number you are already holding. Ones: 5 + 8 = 13, so 910 + 13 = **923**. The whole time you were tracking a single running total, not a column of suspended carries. A useful habit: as you add each place, glance whether the next-smaller place will overflow, and you can pre-adjust. With three-digit sums this becomes effortless; with the two-digit sums that dominate Zetamac it is nearly instant.

For adding a *column* of several numbers — which estimation questions often require — the same principle says group into round chunks. To add 18 + 27 + 32 + 23, do not go in order; pair the numbers that complete to round sums: (18 + 32) + (27 + 23) = 50 + 50 = **100**. Spotting complements inside a list is the single biggest speedup for multi-number addition, and it is the same complement skill that powers subtraction.

### Subtract by completing to a round number

Subtraction is where complements earn their keep. A **complement** is the distance from a number to a convenient round number — usually the next power of ten. The complement of 87 to 100 is **13**, because 87 + 13 = 100. Memorizing complements to 100 (and to 10) makes subtraction feel like addition, which most people find easier and faster.

The technique: to subtract, *add the complement and adjust the round number.*

#### Worked example: 1,000 − 376

The school method makes you borrow across three zeros, which is error-prone. Instead, find the complement of 376 to 1,000 directly by completing each digit to make the columns reach 9, 9, 10:

- 3 → 6 (to reach 9), 7 → 2 (to reach 9), 6 → 4 (to reach 10). Complement is **624**.
- So 1,000 − 376 = **624**. Done, no borrowing.

A second flavor handles non-round subtractions. To do **$4,200 − $376** at a desk, round the subtrahend up to $400, subtract the round number, then add back the $24 you over-removed: $4,200 − $400 = $3,800, then + $24 = **$3,824**. The mental load is one clean subtraction plus one tiny addition, instead of a column of borrows.

The intuition: **subtraction is addition wearing a disguise; complete to a round number and the borrows disappear.**

## Multiplication: a handful of shortcuts cover most of the test

Multiplication is where the biggest time savings live, because the school algorithm (multiply every digit pair, shift, and sum) is slow and carry-heavy. The trader's approach is to recognize the *shape* of the problem and apply the matching shortcut. Here are the ones that pay off most often.

### Distribute: break one factor into friendly pieces

The workhorse, and the one you fall back on when no special trick applies, is plain distribution: split a factor into a sum of easy parts and multiply each. This is the distributive law, `a × (b + c) = a×b + a×c`, used deliberately.

To do **47 × 6**, split 47 into 50 − 3: that is (50 × 6) − (3 × 6) = 300 − 18 = **282**. Or split 47 into 40 + 7: (40 × 6) + (7 × 6) = 240 + 42 = **282**. Either way you replaced one awkward multiplication with two trivial ones and an addition. The skill is picking the split that lands on round numbers.

The same idea scales to two-digit-by-two-digit products, which are the genuinely hard ones on these tests. For **37 × 24**, do not reach for the column algorithm. Split the *smaller* or rounder factor and distribute: 37 × 24 = 37 × 25 − 37 × 1 = 925 − 37 = **888** (using ×25 = ×100/4 to get 37 × 25 = 3,700/4 = 925). Or split the other way: 37 × 24 = 37 × 20 + 37 × 4 = 740 + 148 = **888**. The choice between splits is itself a small optimization: you want each piece to be a multiplication you can do without writing anything, so you look for a nearby multiple of 10, 25, or 100. With a few weeks of drilling, you will pick the good split automatically, the way a chess player sees a good move.

A close cousin worth a separate mention is **doubling and halving**, which rebalances a product into a friendlier shape. Because `a × b = (a/2) × (2b)`, you can halve one factor and double the other without changing the answer. **35 × 16** is awkward, but halve-and-double twice: 35 × 16 = 70 × 8 = 140 × 4 = **560**. Each step traded one hard multiplication for an easier one. The rule of thumb: when one factor is even, halving it (and doubling the other) often lands you on a multiplication you already know.

### ×11: drop the digit sum in the middle

Multiplying a two-digit number by 11 has a beautiful shortcut: write the outer two digits, and slot their *sum* between them.

![Times eleven splits the two digits and inserts their sum in the middle, carrying when the sum overflows ten](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-11.png)

#### Worked example: 63 × 11 and 78 × 11

For **63 × 11**: the outer digits are 6 and 3, and their sum is 6 + 3 = 9, so the answer is **6 9 3 = 693**. That is it.

When the middle sum overflows past 9, you carry. For **78 × 11**: the middle is 7 + 8 = 15, so write the 5 and carry the 1 into the left digit: 7 becomes 8, giving **8 5 8 = 858**. Why does this work? Because 11 × n = 10n + n, so every digit of n lands once shifted left and once not, and the overlapping positions add the neighboring digits. The trick is just that addition made visible.

The intuition: **×11 is not multiplication at all — it is adding each digit to its neighbor.**

### ×5, ×25, ×50: multiply and divide by powers of ten

Multiplying by 5 is multiplying by 10 and halving, because 5 = 10/2. So **86 × 5** = 860 / 2 = **430**. Similarly, ×50 is ×100 then halve (**86 × 50** = 8,600 / 2 = 4,300), and ×25 is ×100 then quarter, or equivalently divide by 4 (**86 × 25** = 8,600 / 4 = 2,150). These come up constantly because 25 and 50 hide inside dollar amounts, quarter-points, and half-percentages.

### Near 100 (and near any round base)

When both factors sit close to 100, there is a clean shortcut. Write each as 100 minus a small deficit. For **97 × 96**: the deficits are 3 and 4. The answer's leading part is `100 − (3 + 4) = 93`, and the trailing two digits are the product of the deficits, `3 × 4 = 12`. Stick them together: **9312**. Check: 97 × 96 = 9,312. The same idea works near 1,000 or near 50 with the appropriate base; it is the engine behind the difference-of-squares trick next.

### Difference of squares: the trick worth memorizing

This is the single most elegant multiplication shortcut, and interviewers love watching candidates use it. Any product `a × b` can be rewritten by centering it on the **midpoint** of the two factors. If `m` is the average of a and b and `d` is half their gap, then:

$$a \times b = (m + d)(m - d) = m^2 - d^2$$

In words: the product equals the **midpoint squared minus the half-gap squared**. The reason to love it is that the midpoint is often a round number whose square you know, turning a hard multiplication into one subtraction.

![Difference of squares re-centers a product on the midpoint of its factors, so the product is the midpoint squared minus the half-gap squared](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-4.png)

#### Worked example: 47 × 53

The two factors straddle 50: their average is (47 + 53)/2 = **50**, and they each sit 3 away, so the half-gap is **3**. Then:

$$47 \times 53 = 50^2 - 3^2 = 2500 - 9 = \mathbf{2491}$$

One subtraction. Compare the long way: 47 × 53 = 47 × 50 + 47 × 3 = 2,350 + 141 = 2,491 — correct, but more steps and a carry. The trick shines whenever the factors share a clean midpoint: **18 × 22** = 20² − 2² = 400 − 4 = **396**; **96 × 104** = 100² − 4² = 10,000 − 16 = **9,984**; **35 × 45** = 40² − 5² = 1,600 − 25 = **1,575**.

The intuition: **a product is symmetric around its midpoint, so you can compute it from the midpoint's square and one correction.**

## Division and fluent fractions, percents, and decimals

Division is the operation people fear most, but on these tests it is the *easiest*, because the problems are constructed to come out exactly. A Zetamac division question like "84 ÷ 12" is really the multiplication "12 × ? = 84" run backwards, and if you know your times tables the answer (7) is instant. The skill is **factor recognition**: seeing 84 and immediately thinking "12 × 7." So the fastest way to get good at division is, paradoxically, to drill multiplication until the products are recognized on sight.

For messier division, fall back on the same "make it friendly" instinct. To estimate **3,840 ÷ 16**, notice 16 × 240 = 3,840 exactly, so the answer is **240**; or chip it down: 3,840 ÷ 16 = (3,200 + 640) ÷ 16 = 200 + 40 = 240.

### Memorize the fraction table

The highest-leverage memorization in the whole toolkit is a small table of fractions as percents and decimals. Once these are automatic, any conversion a problem throws at you is recall, not calculation.

![A memorized table of fractions as percents and decimals turns any price fraction into a decimal with no division](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-5.png)

The figure shows the core set. Commit these to memory:

- **1/2 = 50%**, **1/4 = 25%**, **3/4 = 75%** (everyone knows these).
- **1/8 = 12.5%**, and so the eighths step by 12.5: **3/8 = 37.5%**, **5/8 = 62.5%**, **7/8 = 87.5%**.
- **1/16 = 6.25%** (half of 1/8) — the "teenie" of old stock quotes.
- **1/3 = 33.33%**, **2/3 = 66.67%**.
- **1/7 = 14.29%** (the decimal 0.142857 repeats), so **2/7 = 28.57%**, and so on.
- **1/12 = 8.33%** (one month of a year), and **1/6 = 16.67%**.

A bonus pattern: the **ninths** are gorgeous. **1/9 = 11.11%**, and every ninth is a multiple of that: 2/9 = 22.22%, 3/9 = 33.33%, up to 8/9 = 88.89%. So `n/9` is just `n × 11.11%`.

#### Worked example: convert a $3⅜ price to a decimal

An interviewer says "the bond is trading at three and three-eighths." You need the decimal instantly. From the table, 3/8 = 0.375, so $3⅜ = **$3.375**. If they had said three and five-sixteenths, you would recall 1/16 = 0.0625, multiply by 5 to get 0.3125, and answer **$3.3125**. No long division, just recall and a small multiply.

The intuition: **fractions are a foreign language whose vocabulary you memorize once; after that, conversion is reading, not translating.**

### Percentages from anchors

To take a percentage of a price in your head, build it from two anchors you can always find instantly: **10%** (move the decimal point one place left) and **1%** (move it two places left). Then add and halve those anchors to assemble any percent.

![The percentage ladder builds any percent of a price from the 10% and 1% anchors by adding and halving](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-10.png)

#### Worked example: percentages of $1,800

Start with the anchors: 10% of $1,800 = **$180**, and 1% = **$18**. Now everything is reachable:

- **5%** = half of 10% = **$90**.
- **20%** = 10% doubled = **$360**.
- **15%** = 10% + 5% = $180 + $90 = **$270**.
- **25%** = 20% + 5% = $360 + $90 = **$450** (or just a quarter of $1,800).
- **2%** = 1% doubled = **$36**.
- **7%** = 5% + 2% = $90 + $36 = **$126**.

There is one more percentage trick that feels like cheating: **x% of y equals y% of x**, because both are `x × y / 100`. So if someone asks "18% of 50," do not compute it directly — flip it to "50% of 18," which is just half of 18, or **9**. This commutation turns awful percentages into trivial ones surprisingly often.

The intuition: **never compute a percentage from scratch; anchor on 10% and 1% and build the rest by adding and halving.**

## Squaring and rough square roots

Squaring shows up in the difference-of-squares trick, in variance and volatility questions, and as its own interview drill. Two cases have clean shortcuts.

### Numbers ending in 5

Any number ending in 5 squares with a two-step recipe: take the digit(s) before the 5, multiply by the next integer up, and append **25**. For **35²**: the leading digit is 3, times the next integer 4 is 12, append 25 → **1225**. For **85²**: 8 × 9 = 72, append 25 → **7225**. For **115²**: 11 × 12 = 132, append 25 → **13225**. It works because (10n + 5)² = 100·n(n+1) + 25.

### Numbers near 50

Squaring near 50 has its own tidy method, because 50² = 2,500 is a clean anchor.

![Squaring a number near 50 adds the offset to 25 for the front digits and appends the offset squared](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-9.png)

#### Worked example: 56²

Write 56 as 50 + 6, so the offset `d` is **6**. The algebra (50 + d)² = 2,500 + 100d + d² says: the **front** of the answer (the hundreds) is `25 + d`, and the **back** (the last two digits) is `d²`. So:

- Front: 25 + 6 = **31**.
- Back: 6² = **36**.
- Join: **3136**. So 56² = **3,136**.

It works below 50 too, with a negative offset. For **47²**, d = −3: front = 25 − 3 = 22, back = (−3)² = 9, which you pad to two digits as 09 → **2209**. Check: 47² = 2,209. The padding rule matters — the back is always two digits.

The intuition: **anchor squares on a round base you know, and the offset does the rest in two small pieces.**

### Rough square roots by bracketing

You will not be asked for an exact irrational root, but you will be asked to *estimate* one — for instance, to sanity-check a volatility. The method is bracketing: find the two perfect squares your number sits between and interpolate. To estimate √50: it is between 7² = 49 and 8² = 64, and very close to 49, so √50 ≈ **7.07**. To estimate √2,000: it is between 44² = 1,936 and 45² = 2,025, closer to 45, so √2,000 ≈ **44.7**. Knowing that **√2 ≈ 1.414** and **√3 ≈ 1.732** lets you handle a surprising number of these by factoring out a perfect square: √50 = √(25 × 2) = 5 × 1.414 = 7.07.

## Basis points and percentage moves on prices

This section is where the abstract arithmetic meets the trading desk, so it deserves its own careful treatment. A trader converts between basis points, percentages, and dollar amounts on a given price constantly, and the conversions must be reflexive.

Recall the definitions: **1% = 100 bps**, and **1 bp = 0.01% = 1/10,000**. So a basis point *of a price P* is `P × bps / 10,000`. The fastest mental route is to find **1 bp of the price once**, then scale.

![On a $4,200 price, one basis point is $0.42 and fifty basis points is $21, scaling linearly along the ruler](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-6.png)

#### Worked example: basis points on a $4,200 index level

Take a price level of **$4,200** (think of a stock index). To find 1 bp of it, move the decimal point **four places left**: $4,200 → **$0.42**. That single number is your unit. Now everything scales:

- **1 bp** = $0.42.
- **10 bps** = $4.20 (just shift the decimal one place).
- **50 bps** = 50 × $0.42 = **$21.00**.
- **100 bps** = 1% = $42.00.

So if a risk manager says "you are down 50 bps on the index position," you instantly know that is **$21 per unit** of exposure. And the reverse: a **0.5%** move on $4,200 is, by the percentage ladder, $4,200 × 0.005 = **$21** — the same number, because 0.5% *is* 50 bps. Holding both representations and seeing they agree is a good internal check.

#### Worked example: a percentage drop on a stock price

Suppose a stock at **$1,800** falls **3.5%**. Build it from anchors: 3% of $1,800 = $54, and 0.5% = $9, so the total drop is $54 + $9 = **$63**, and the new price is about $1,800 − $63 = **$1,737**. A trader does that in under two seconds and quotes around the new level before the screen even settles. The same machinery prices quote widths: if you quote a stock **two cents wide** around a $50 fair value, your bid is $49.99 and offer $50.01, and that **$0.02 spread on, say, 500 shares is $10** of gross edge if you trade both sides.

The intuition: **find one basis point of the price once, then every bp and percentage question is a multiplication or a decimal shift.**

## Mental P&L: the arithmetic that is literally the job

Profit and loss — **P&L** — is the scoreboard. The most common P&L computation a trader does is the simplest: profit on a trade equals the **number of shares times the edge captured per share**. That is one multiplication, and it is the same `shares × cents` arithmetic the tests drill.

![Profit on a fill is shares times edge per share, computed the instant a trade prints and then netted for fees](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-7.png)

#### Worked example: P&L on a fill

A fill prints: you **bought 1,200 shares** at **$42.15**, and your fair value was **$42.20**. Your edge per share is the difference, $42.20 − $42.15 = **$0.05**. The gross P&L is:

$$1{,}200 \text{ shares} \times \$0.05 = \$60$$

So you made **$60** on that fill, before costs. Now net it down: suppose exchange and clearing fees are about **$4** on this trade. Your net is $60 − $4 = **$56**. A trader runs this instantly because it tells them whether the fill was good and how much risk they just took on. Scale it up and the same one-liner governs a whole day: 200 such fills at $56 net is $11,200, and a single fat-fingered mis-multiplication that makes you think a losing trade is a winner can cost real money. This is why the firms test the multiplication: **shares × edge is the test and the job, the same arithmetic in two costumes.**

A related desk calc is sizing a hedge. If you are long 1,200 shares of a stock and want to neutralize your exposure with a related future that moves about 1.5× as much per dollar, you need roughly 1,200 / 1.5 = **800 units** of the future to offset — a division you do on the fly. Get it wrong and you are over- or under-hedged, carrying risk you did not intend.

## Estimating messy products and quotients fast

Not every number you need is clean, and not every answer needs to be exact. A huge fraction of real trading arithmetic — and many interview questions — only requires a *good estimate, fast*, with a clear sense of how far off you might be. The skill is **rounding to friendly numbers, computing, and tracking the direction of your error** so you know whether the true answer is a little above or below your estimate.

The core move is the same one we used for multiplication: replace each messy factor with a nearby round one. To estimate **312 × 488**, round to 300 × 500 = **150,000**, and note that you rounded one factor *down* (312 → 300) and one *up* (488 → 500), so the errors partly cancel and your estimate is close. The exact value is 152,256, so 150,000 was within about 1.5% — plenty for a sanity check. If you want to tighten it, correct the larger error: 312 is 12 above 300, and 12 × 500 = 6,000, so 312 × 500 = 156,000, then subtract 312 × 12 = 3,744 to get 152,256 exactly. You can stop at whatever precision the question needs.

For **quotients**, round the denominator to something that divides cleanly. To estimate **4,830 ÷ 61**, round 61 to 60: 4,830 ÷ 60 = 80.5, and since you rounded the denominator *down*, the true quotient is slightly *smaller* (a bigger denominator divides into fewer), so the answer is a touch under 80 — about **79**. The exact value is 79.2. Knowing the *direction* of the error is what separates a useful estimate from a guess: a trader who says "about 79, and if anything a hair lower" has bounded the answer, not just thrown out a number.

The most important discipline in estimation is **keeping the magnitude straight** — getting the right number of zeros. Far more estimation errors come from a misplaced decimal point than from a sloppy leading digit. A reliable trick is to estimate in **scientific-notation pieces**: handle the significant figures and the powers of ten separately. For **0.0042 × 1,900**, compute 4.2 × 1.9 ≈ 8 for the digits, then track the powers: 0.0042 is 4.2 × 10⁻³ and 1,900 is 1.9 × 10³, so the powers cancel to 10⁰ = 1, giving ≈ **8**. The exact answer is 7.98. Separating "what are the digits" from "where is the decimal" is how you avoid being off by a factor of ten — the kind of error that, on a desk, turns a $50,000 position into a $500,000 one in your mental model.

#### Worked example: a fast market-sizing estimate

**"Roughly, what's the daily dollar volume of a stock that trades 3.2 million shares a day at about $47?"** Round: 3.2 million × $47 ≈ 3 million × $50 = **$150 million**, and since you rounded shares down and price up, the estimate is close. Tighten if asked: 3.2 × 47 = 3.2 × 47 = 150.4, so **$150.4 million**. You produced a usable number in two seconds and a precise one in five, which is exactly the gradient of precision a trading conversation moves along.

The intuition: **round to friendly numbers, compute, and always track which way you rounded so you know which side of your estimate the truth lies on.**

## Speed versus accuracy: how to actually pace a scored test

Now the strategy layer. On a test that penalizes wrong answers — like 80-in-8 — raw speed is not the objective; **net score** is. And net score is not monotonic in speed. Go too slow and you leave easy points unanswered; go too fast and your error rate climbs until the penalties eat your gains. There is an optimal pace in between, and it sits *just inside* your personal accuracy limit.

![Net score rises with speed then falls as errors cost points, peaking at a pace fast enough to bank questions but slow enough to stay accurate](/imgs/blogs/mental-math-arithmetic-speed-quant-interviews-8.png)

The curve is an inverted U. On the left (too slow) you are accurate but answering too few questions — points left on the table. On the right (too fast) you are attempting many but the penalty for wrong answers drags the net down. The peak is where you are moving as fast as you can *while keeping errors rare*. Practically, that means: find the fastest pace at which you are still about 95%+ accurate, and live there. Pushing past it feels productive but is usually negative expected value.

This connects to a deep idea in trading itself. A market maker's edge per trade is tiny — a few cents — and is only profitable *on average over many trades*. One careless error (a mis-priced quote, a wrong-sized order) can wipe out the profit from dozens of good trades. So the discipline the test rewards — going fast but not so fast that you make catastrophic errors — is the *same* discipline the job rewards. The penalty structure of 80-in-8 is not arbitrary cruelty; it is a faithful simulation of how trading P&L actually works.

A few concrete pacing rules that follow from the curve:

- **Never let a hard question stall you.** If a problem is not yielding in a few seconds, skip it (if the format allows) or take your best fast attempt and move on. On a paced test, time is the scarce resource, not difficulty.
- **Bank a small cushion early.** The pacing line is unforgiving when you fall behind, because falling behind induces panic-rushing. Aim to be slightly ahead of the ten-per-minute line for the first couple of minutes.
- **Accuracy is the governor.** If you find yourself making more than the occasional slip, you are over the peak of the curve — slow down a notch until your error rate drops, then hold there.

## In the interview room: five fully solved drills

Theory is cheap; let us actually do the work. Here are five interview-style problems with full solutions, mixing the techniques above. Time yourself: aim for the six-second-per-question feel on the short ones, and a clear method on the longer ones. Cover the solution, attempt it, then check.

### Drill 1 — A timed Zetamac-style mixed set

Solve these in your head, in order, as fast as you can while staying accurate. This is the texture of a real Zetamac stretch.

1. `47 + 68` → left-to-right: 40 + 60 = 100, then 7 + 8 = 15 → **115**.
2. `91 ÷ 7` → factor recognition: 7 × 13 = 91 → **13**.
3. `8 × 45` → ×45 is hard, so 8 × 45 = 8 × 9 × 5 = 72 × 5 = **360** (or 45 × 8 = 360 directly).
4. `12 × 12` → known square, **144**.
5. `1,000 − 437` → complement to 1,000: 5/6/3 by completing to 9,9,10 → **563**.
6. `35²` → ends in 5: 3 × 4 = 12, append 25 → **1225**.
7. `7 × 88` → distribute: 7 × 80 + 7 × 8 = 560 + 56 → **616**.
8. `96 ÷ 8` → 8 × 12 = 96 → **12**.

If you cleared all eight in about 50 seconds with no errors, you are at a strong Zetamac pace (that is roughly a score in the 60s over two minutes). The lesson: every single one had a *method*, not a grind. Speed comes from recognizing the shape, not from computing faster.

### Drill 2 — The difference-of-squares special

**"What is 48 × 52?"** The interviewer is testing whether you see the structure. The factors straddle 50, each 2 away. So 48 × 52 = 50² − 2² = 2,500 − 4 = **2,496**. Say it in two seconds and you have signaled that you know the trick. If you instead grind 48 × 52 = 48 × 50 + 48 × 2 = 2,400 + 96 = 2,496, you get the right answer but reveal you did not spot the pattern — which is the thing they are actually probing.

Follow-up they often ask: **"And 49 × 51?"** Same midpoint 50, half-gap 1, so 2,500 − 1 = **2,499**. The pattern that products just below the square of the midpoint shrink by 1, 4, 9, 16 as you widen the gap is itself a nice thing to notice aloud.

### Drill 3 — A basis-points-on-price P&L question

**"You're holding a position worth $250,000 and the market moves 8 basis points in your favor. What's your P&L?"** Find 1 bp of $250,000 first: move the decimal four places left → **$25**. Then 8 bps = 8 × $25 = **$200**. So you made **$200**. The interviewer may then flip it: **"And if it had moved against you by 0.25%?"** That is 25 bps, so 25 × $25 = **$625** loss. Notice how the single anchor — $25 per bp — powered both answers. The whole question is "find one bp, then scale," and a trader does it without reaching for the percentage from scratch.

### Drill 4 — A fraction-to-decimal pricing question

**"A Treasury bond is quoted at 101 and 7/32. Express that as a decimal price."** Treasuries trade in 32nds, so you need 7/32 as a decimal. Use the anchor 1/32 = 0.03125 (which is half of 1/16 = 0.0625), so 7/32 = 7 × 0.03125 = **0.21875**. The price is **101.21875**. If you have the eighths and sixteenths memorized, the 32nds are one more halving away, and the conversion is recall plus a small multiply. A candidate who freezes on "7/32" has not done the memorization; a candidate who answers in three seconds has.

### Drill 5 — A pure estimation question under time pressure

**"Roughly, what is 312 × 19?"** They want a fast, close estimate, not the exact value, to see if you can bound a number quickly. Round 19 up to 20: 312 × 20 = 6,240, then subtract one 312 (because 19 = 20 − 1): 6,240 − 312 = **5,928**. That is exact, in fact, and took two clean steps. If they only wanted an estimate, "about 6,000" would have been fine and instant. The meta-skill here — **rounding to a friendly number and correcting** — is the most general tool in the kit, and it is exactly how a trader prices something they have never seen before: anchor on something close and known, then adjust.

### Drill 6 — A quote-width and edge question

**"You're making a market in a stock with a fair value of $50. You quote it $49.97 bid, $50.03 offered. How wide is your market, and what's your gross edge if you trade 800 shares on each side in a day?"** The width is the offer minus the bid: $50.03 − $49.97 = **$0.06**, a six-cent-wide market. Your edge per share *per round trip* (buy at the bid, sell at the offer) is the full width, $0.06, but the cleaner way interviewers want it framed is half-spread per side: you capture $0.03 of edge versus fair on each fill. Trading 800 shares each way means 800 buys at the bid and 800 sells at the offer, so the gross edge is 800 × $0.06 = **$48** — or equivalently 1,600 shares each capturing $0.03, which is also $48. The follow-up they love: **"If a competitor quotes a penny tighter on each side, what happens to your fills?"** A tighter quote gets hit first, so you either match (giving up edge to keep flow) or step back (keeping edge but losing volume) — and now the interview has slid from arithmetic into the actual economics of market making, which is exactly where they want to take you once the arithmetic is clearly handled.

### Drill 7 — Chained percentages

**"A stock rises 10% and then falls 10%. Is it back where it started?"** The instinct says yes; the arithmetic says no. Start at $100. Up 10% → $110. Down 10% of $110 → $110 − $11 = **$99**. You are down $1, or 1%, because the second percentage applied to a *larger* base. The general fact, worth carrying: a gain of `x%` followed by a loss of `x%` leaves you at `(1 + x)(1 − x) = 1 − x²` of where you started — always slightly below par, by exactly the square of the move. For a 10% move, x² = 1%, matching the $99. For a 20% up-then-down, you would end at 1 − 0.04 = 96% → **$96** from $100. This "volatility drag" is a real phenomenon in leveraged products, and a candidate who computes the $99 instantly and explains the 1 − x² structure has shown both the arithmetic and the insight.

### A note on the "estimate a big messy thing" genre

Beyond these five, market-making interviews love **Fermi estimation** — "how many piano tuners are in Chicago," "how many golf balls fit in a 747," "what is the daily dollar volume of this ETF." These are not arithmetic-speed tests per se, but they live on the same skill: decompose into factors you can each estimate, multiply, and keep the running magnitude straight. The arithmetic fluency from this whole post is what lets you carry the multiplication without losing the thread. A worked taste: *daily revenue of a coffee shop* ≈ 200 customers/day × $5 average ticket = **$1,000/day**, so ≈ $30,000/month — you built it from two estimates and one multiplication, exactly the muscle the drills train.

## Common misconceptions

**"It's just arithmetic, so I'll be fine — I'm good at math."** Being good at *mathematics* (proofs, calculus, probability) is almost unrelated to being fast at *arithmetic* under a stopwatch. Plenty of strong mathematicians bomb Zetamac the first time because they have never trained raw speed. It is a separate, trainable motor skill, like sight-reading music. Treat it as one.

**"I should learn dozens of obscure Vedic-math tricks."** No. The toolkit is small on purpose: left-to-right add, complements, distribute, ×11, ×5/25/50, difference of squares, the fraction table, the squaring shortcuts, and basis-point scaling. That is nearly everything the tests reward. Collecting exotic tricks you will not recognize under pressure is procrastination dressed as preparation. Drill the few that matter until they are reflexive.

**"Faster is always better."** Only on an unpenalized test like Zetamac, and even there only up to your accuracy limit. On a penalized test like 80-in-8, and certainly on a real desk, an error can cost more than several correct answers earn. The goal is the *peak of the speed-accuracy curve*, not the maximum speed.

**"You need to memorize times tables up to 30 × 30."** You need the standard tables to about 12 × 12 cold, plus the squares of round-ish numbers (15, 25, and the squares ending in 5), plus the fraction table. Beyond that, the *techniques* (distribute, difference of squares) generate the rest faster than memorization would, and they generalize to numbers no table covers.

**"Mental math is the hard part of the interview."** It is the *gate*, not the summit. Clearing the arithmetic screen gets you into the room; the offer comes from the probability, expected-value, market-making, and risk-thinking rounds. Strong arithmetic is necessary but nowhere near sufficient — see the cross-links below for the conceptual rounds that actually decide the outcome.

## How it shows up on a real trading desk

The interview test is a faithful miniature of the workday. Here are concrete places the same arithmetic surfaces once you are actually trading.

**Pricing in your head, continuously.** A market maker watches a stream of inputs — the underlying price, related futures, the order book — and holds a fair value that updates as they move. When the input ticks, the trader adds or subtracts a few ticks (left-to-right), re-centers the quote, and the new bid and offer are out before a slower competitor reacts. If fair value moves from $42.18 to $42.25, the trader who does that addition in one second tightens or pulls their $42.20 offer before they get picked off for $0.05 a share. Over thousands of quotes a day, those saved nickels are the business.

**Computing P&L on every fill.** When a trade prints, the trader instantly multiplies shares by the edge to know what they just made or lost and how much risk they took on. We worked the canonical case: **1,200 shares × $0.05 = $60** gross, **$56** net of $4 fees. Scaled across a day of hundreds of fills, the running P&L lives in the trader's head as a continuously updated sum, and a multiplication error there is a misread of the scoreboard.

**Quoting options in cents.** Options market makers think in even tighter increments. An option quoted **$1.20 bid, $1.24 offered** is **4 cents wide**, and on a standard equity option contract covering **100 shares**, that 4-cent spread is **$4 per contract** of gross edge. Trade 50 contracts on each side and that is **$200**. The trader also converts the option's price sensitivity (its **delta** — how much the option moves per $1 of stock) into a hedge: an option with 0.40 delta on 50 contracts behaves like 0.40 × 50 × 100 = **2,000 shares** of stock, so they hedge with 2,000 shares. Every number in that chain is a fast multiplication.

**Basis points everywhere in fixed income and risk.** A bond trader quoting a yield move of "5 bps" on a $10,000,000 position needs the dollar figure fast. One bp of $10,000,000 is **$1,000** (decimal four places left), so 5 bps is **$5,000** of value at stake — though for bonds the precise sensitivity uses **duration**, the arithmetic instinct of "find one bp, then scale" is the same. Risk limits, financing rates, and fees are all quoted in bps, so the conversion runs all day.

**The Knight Capital lesson, in spirit.** The most expensive trading-desk errors in history were not arithmetic slips by a human, but they rhyme with the discipline this skill builds. In 2012, a software-deployment error caused Knight Capital to send millions of unintended orders in 45 minutes, costing roughly **$440 million** and nearly destroying the firm. The human-scale version of that lesson is exactly what 80-in-8's penalty teaches: speed without a check on errors is dangerous, and the rare catastrophic mistake matters more than the typical small win. A trader internalizes that the moment they see their net P&L swing on a single fat-fingered order.

**The hiring funnel itself.** Finally, the arithmetic test does real work as a *filter*. A firm might receive thousands of applications for a handful of trading seats. Zetamac and 80-in-8 are cheap, objective, and predictive enough of the on-desk skill that they are an efficient first cut. That is why preparing for them is not gaming the system — it is building the exact muscle the seat requires, which is why the practice transfers directly to day one.

## When this matters to you and where to go next

If you are preparing for trading interviews at market-making or prop firms, this is the most *trainable* part of the whole process, with the clearest practice loop: do timed, mixed sets every day, track your score, and watch it climb. A realistic plan is two weeks of ten-minute daily Zetamac sessions plus a few timed 80-in-8 simulations; most people move from "panicked" to "fluent" in that window, because — unlike the conceptual rounds — raw arithmetic speed responds almost linearly to deliberate practice. The key is to practice the way you are tested: under a clock, mixing all four operations, and pushing your pace to just inside your accuracy limit.

The deeper payoff is that this fluency frees up working memory for the parts of the interview that actually decide the outcome. When the arithmetic is automatic, your mind is free to reason about the *problem* — the probability, the expected value, the market structure — instead of grinding through a multiplication. That is the whole point: the desk does not pay you to multiply; it pays you to think, and fast arithmetic is what clears the runway for the thinking.

### A concrete two-week practice plan

If you want a schedule rather than a vague "practice more," here is one that works for most people. The principle behind it is **specificity** — train under the exact conditions of the test — and **progressive overload**, nudging your target pace up as your accuracy holds.

- **Days 1–3: build the table and the reflexes.** Before timing anything, memorize the fraction table and the complements to 100 cold, and rehearse the named tricks (×11, difference of squares, squaring near 50, the percentage anchors) on a handful of examples each until you do not have to think about the steps. This is the foundation; rushing past it just bakes in slow habits.
- **Days 4–10: daily timed Zetamac.** Run one full two-minute game every day with the default settings, and write down the score. Most people start somewhere in the 30s–40s and, with nothing but daily reps, climb 2–4 points a day for a while before plateauing. When you plateau, do not just grind — diagnose. If you are losing time on division, drill multiplication-recognition; if multi-digit multiplication is slow, drill the distributing split. The score tells you the symptom; your sense of *which* questions felt slow tells you the cause.
- **Days 11–14: simulate the penalized format.** Switch to a trainer that includes negatives, decimals, and negative marking to rehearse 80-in-8 conditions. The goal here is not a higher raw count but finding your peak on the speed-accuracy curve — the pace where you are fast but your error rate stays near zero. Run a few full eight-minute simulations and watch where the wrong answers cluster; that cluster is your "slow down a notch" boundary.

Throughout, keep the sessions *short and frequent*. Ten focused minutes a day beats a single two-hour grind, because arithmetic speed is a motor skill that consolidates with sleep and spaced repetition, exactly like learning an instrument. Two weeks of this is enough for most candidates to clear the gate with margin; a few weeks more pushes you into comfortably-strong territory.

For the conceptual rounds that follow the arithmetic gate, build out from here:

- [Expected value techniques: linearity, indicators, and symmetry](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) — the four tools that crack most expected-value problems in a line, and where most of the "hard" interview questions actually live.
- [Conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) — the disease-test, Monty Hall, and two-children puzzles, built from the same first-principles machinery.
- [Counting and combinatorics for quant interviews](/blog/trading/quantitative-finance/counting-combinatorics-quant-interviews) — why most "hard" probability questions are counting in disguise, with the full toolkit from zero.
- [The classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems) — a technique-organized tour of the canonical problems, each solved in full.

**Further reading and practice tools.** The single best practice tool is the **Zetamac Arithmetic Game** (the free web drill) — set it to the default config and run it daily. For the *why* behind the tricks and a much larger collection of shortcuts, **"Secrets of Mental Math" by Arthur Benjamin and Michael Shermer** is the standard reference, and it goes far past what any interview needs (whole chapters on multi-digit squaring and mental long division). For interview-specific timed practice in the 80-in-8 format, sites like **TraderMath** and similar arithmetic trainers replicate the negative-marking and decimal/negative-number style that Zetamac's defaults omit. Pick one of each — a daily drill and a periodic timed simulation — and the gate takes care of itself.

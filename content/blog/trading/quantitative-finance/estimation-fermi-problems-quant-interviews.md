---
title: "Estimation and Fermi problems for quant trading interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch playbook for the estimation questions that open quant trading interviews -- how to decompose an unknown into factors, anchor each one with a memorized reference number, recombine in orders of magnitude, and bracket the answer, with five fully worked interview problems and the dollar examples a market maker actually computes."
tags: ["fermi-estimation", "quant-interviews", "estimation", "market-sizing", "order-of-magnitude", "mental-math", "trading-interview", "back-of-envelope", "geometric-mean", "quantitative-trading"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** -- a trading firm does not ask "how many piano tuners are in Chicago?" because it cares about pianos. It asks because the job is putting a defensible number on something nobody knows, fast, and then betting on it. Fermi estimation is that skill, made testable.
>
> - **The whole method is one loop**: decompose the unknown into a product of factors you can each guess, estimate each factor from a memorized anchor, multiply them back together, and sanity-check the order of magnitude. That loop solves almost every estimation question you will ever be asked.
> - **You are graded on structure, not the final digit.** An interviewer who hears a clean decomposition, stated assumptions, and a sanity-check will pass you even if your number is off by 3x. Someone who blurts "around 50 million" with no reasoning fails even if they are exactly right.
> - **Think in orders of magnitude.** Being within a factor of 10 of the truth is a *win*. The goal is the right power of ten, not the last digit -- a Fermi answer is graded on a log scale.
> - **Multiply geometric means, not arithmetic averages.** When a factor could be 10 or 1,000, its center is $\sqrt{10 \times 1000} = 100$, not 505. Averaging the exponents is the honest midpoint of an uncertain range.
> - **The one number to remember**: a stock that trades **10 million shares a day at \$50** turns over **\$500 million a day** -- about **\$125 billion a year**. That single back-of-envelope calculation, volume times price, is one a trader runs in their head dozens of times a day.

Here is a question a market-making firm might open an interview with: *how many golf balls fit inside a school bus?* You will be tempted to protest that you have never measured a golf ball or a bus, that this is unknowable, that it has nothing to do with trading. All three protests are wrong, and the interviewer is watching to see whether you know it.

The question is not about golf balls. It is about whether you can take something you genuinely do not know, break it into pieces you *can* put numbers on, multiply those pieces together, and arrive at an answer you are willing to defend -- all out loud, in ninety seconds, without panicking. That is the single most common thing a trader does. A market maker quoting a price on an illiquid bond, a prop trader sizing a position before the data is in, a desk estimating how much of a stock it can buy before it moves the price -- every one of those is the golf-ball question wearing a suit.

![The Fermi loop: an unknown becomes a number by decomposing into factors, estimating each from anchors, recombining by multiplication, and sanity-checking the order of magnitude](/imgs/blogs/estimation-fermi-problems-quant-interviews-1.png)

The diagram above is the mental model for this entire post: **decompose, estimate, recombine, sanity-check.** Master that loop and the golf balls, the piano tuners, the market sizes, and the dollar turnover of a stock all become the same problem. This post builds the loop from zero -- no math beyond multiplication, no finance assumed -- and then puts you in the interview room with five fully worked problems and the dollar calculations a real desk runs every day.

A quick note on what this is and is not. This is educational: it teaches a reasoning skill and the arithmetic behind it. Nothing here is financial advice, and none of the worked numbers are predictions about any real security.

## Foundations: what a Fermi question is and why firms ask it

Let us define our terms before we use a single one of them.

A **Fermi problem** -- also called an **estimation problem**, a **back-of-the-envelope problem**, or a **guesstimate** -- is a question whose exact answer is hard or impossible to look up, but whose *approximate* answer you can build from common-sense pieces. The name comes from the physicist Enrico Fermi, who was famous for estimating quantities to within a factor of a few using almost no data. The classic Fermi question -- "how many piano tuners are there in Chicago?" -- has been asked in physics classrooms for seventy years and now lives in the interview rooms of every major trading firm.

An **order of magnitude** is a factor of ten. Two numbers are "the same order of magnitude" if they are within a factor of ten of each other: 30 and 200 are the same order of magnitude (both "hundreds"); 30 and 30,000 are three orders of magnitude apart. When we say a Fermi answer should be "within an order of magnitude" of the truth, we mean within a factor of ten -- close enough that you have the right number of zeros, even if the leading digit is wrong. This is the unit Fermi estimation works in, so hold onto it.

An **anchor** (or **reference number**) is a quantity you have memorized or can reconstruct, which you use as a starting point. "The US has about 330 million people" is an anchor. "A cup of coffee costs about \$4" is an anchor. The whole game is to express the unknown thing in terms of anchors you already trust.

### Why a trading firm cares

You might reasonably ask why a firm that hires people to trade futures and equities spends interview time on golf balls. There are three real reasons, and they are worth understanding because they tell you exactly what the interviewer is grading.

**First, traders price the unpriced.** A market maker's job is to quote a buy price and a sell price on something *right now*, before anyone has had time to compute a "correct" value. When a customer asks a desk to quote a thinly traded corporate bond, or an options market maker has to put a price on a strike nobody has traded today, there is no screen to read the answer off. The trader has to estimate it from related instruments, typical spreads, and gut-level anchors -- and commit real money to that estimate. The golf-ball question is a low-stakes rehearsal for exactly that.

**Second, the job is reasoning under uncertainty, out loud, calmly.** A trading floor is loud, fast, and full of incomplete information. The interviewer wants to see how you behave when handed a question you cannot fully answer: do you freeze, or do you start breaking it down? Do you state your assumptions so a colleague could check them, or do you mumble a number and hope? An estimation question is a stress test for the temperament the job actually requires.

**Third, it is hard to fake and hard to memorize.** You can drill calculus or grind LeetCode. You cannot memorize the answer to "estimate the annual revenue of all the coffee shops in Manhattan," because the interviewer will just change "coffee shops" to "dry cleaners" or "Manhattan" to "London." The skill transfers; the trivia does not. That makes estimation a clean signal of how you think.

### How it is graded (this is the part candidates miss)

Here is the single most important thing in this entire post, and the thing most candidates get exactly backwards: **you are not graded on the final number. You are graded on the structure and reasoning that produced it.**

![How a Fermi answer is graded: a strong answer decomposes clearly, anchors each guess, recombines tracking magnitude, and sanity-checks; a weak answer jumps to a number with false precision](/imgs/blogs/estimation-fermi-problems-quant-interviews-11.png)

The matrix above is the rubric an interviewer is using whether or not they say so. Look at what earns a "strong" mark in each row. In the **decompose** row, a strong candidate names the factors out loud and writes the product down before computing anything. In the **anchor** row, they state each guess and *why* -- "I'll assume a share price of \$50, because most large US stocks trade between \$20 and \$200." In the **recombine** row, they multiply in powers of ten and keep track of the magnitude. In the **sanity-check** row, they look at the answer and ask whether it feels too big or too small, and say so.

Now look at the "weak" column. The weak candidate jumps to a number with no visible structure. They invent figures with no reference point. They drop a factor of a thousand in the arithmetic and do not notice. And -- the giveaway that they have missed the entire point -- they report **\$487,213,902** when they mean "roughly half a billion dollars." That last one, **false precision**, is the clearest signal to an interviewer that a candidate does not understand what estimation is for. Nobody who actually thinks in orders of magnitude writes nine significant figures for a quantity they guessed.

So the grading is almost inverted from what nervous candidates expect. A clean decomposition with a final answer that is off by 3x beats a lucky exact guess with no reasoning, every single time. Internalize this and half the interview pressure evaporates: you do not have to be *right*, you have to be *clear*.

## The decomposition method: turn one hard guess into a product of easy ones

The engine of every Fermi estimate is decomposition. The principle is simple: **you cannot directly estimate the hard quantity, but you can write it as a product of factors you can each estimate, and a product of okay guesses is a surprisingly good guess.**

Why does multiplying rough guesses work so well? Because errors partly cancel. If you decompose a quantity into four factors and you overestimate two of them while underestimating the other two, the high guesses and low guesses pull against each other in the product. A single direct guess has nowhere to hide its error; a product of four guesses tends to land closer to the truth than any one of its factors deserves. This is not a guarantee, but it is a strong tendency, and it is the entire reason the method works.

The mechanical recipe:

1. **Write the unknown as a product.** Find a chain of factors whose product is the thing you want. "Total tunings demanded per year" equals "number of pianos" times "tunings per piano per year." "Daily dollar turnover" equals "shares traded per day" times "price per share."
2. **Keep decomposing until every factor is something you can anchor.** If "number of pianos" is still hard, break it further: households times fraction of households with a piano. Stop when each leaf is a number you can defend with a sentence.
3. **Estimate each leaf.** Use anchors. State each one.
4. **Multiply back up the tree.** Combine the leaves into the branches, the branches into the answer.
5. **Sanity-check the magnitude.** Does the final number have a plausible number of zeros?

#### Worked example: the daily dollar turnover of a stock

Let us run the loop on the most trading-relevant estimation there is, and the one you should be able to do in your sleep: how many dollars' worth of a given stock change hands in a day?

The quantity is **daily dollar turnover** -- the total dollar value of all the shares that trade in one day. (Turnover here just means "value traded," not profit.) Decompose it into two factors:

$$\text{daily dollar turnover} = (\text{shares traded per day}) \times (\text{price per share})$$

The first factor, "shares traded per day," has a name: **average daily volume**, or **ADV**. It is one of the most-watched numbers on any trading desk. For a large, liquid US stock, ADV is typically somewhere between 1 million and 50 million shares. Let us anchor it at 10 million shares a day.

The second factor is the share price. Most large US stocks trade between \$20 and \$200; let us anchor at \$50.

Multiply:

$$10{,}000{,}000 \text{ shares} \times \$50 = \$500{,}000{,}000 \text{ per day}$$

So this stock turns over about **\$500 million a day**. Annualize it -- there are about 250 trading days in a year (roughly 365 minus weekends and holidays) -- and you get:

$$\$500{,}000{,}000 \times 250 \approx \$125{,}000{,}000{,}000 \text{ per year}$$

About **\$125 billion a year** of this single stock trades hands.

![A stock's daily dollar turnover is average daily volume times price: ten million shares at fifty dollars is five hundred million dollars a day and about one hundred twenty-five billion dollars a year](/imgs/blogs/estimation-fermi-problems-quant-interviews-8.png)

The figure traces the two-factor multiply and the annualization. Notice how little you needed: one anchor for volume, one for price, one for trading days. **The intuition this teaches: a stock's dollar liquidity is just volume times price -- and the whole calculation collapses to multiplying two numbers you can carry in your head.** A trader uses exactly this to judge how big a position they can take without becoming a meaningful fraction of the day's flow.

## Anchoring: the reference numbers that turn estimation into arithmetic

Decomposition tells you *what* to multiply. Anchoring gives you the numbers to multiply *with*. The difference between a candidate who flails and one who flows is almost entirely a stock of memorized anchors -- because once your factors bottom out in numbers you already know, the rest is just multiplication.

![The anchor card: a dozen reference numbers worth memorizing, from the US population of 330 million to a typical share price of twenty to two hundred dollars](/imgs/blogs/estimation-fermi-problems-quant-interviews-2.png)

The card above collects the anchors that come up again and again. You do not need to memorize a phone book -- you need maybe two dozen numbers, and most of them you half-know already. Group them by type:

**Population anchors.** The US has about **330 million** people. The world has about **8 billion**. A large metropolitan area has 1 to 10 million people (New York metro about 20 million, Chicago metro about 9 million, a mid-size city 1 to 2 million). Households are roughly population divided by 2.5, so the US has about **130 million households**. These are the denominators of countless estimates.

**Time anchors.** A year is **365 days**, about **8,760 hours**, about 52 weeks. A working year is about **2,000 hours** (50 weeks times 40 hours) and about **250 working days**. A human lifespan is about **80 years**. These convert "per year" into "per day" and back.

**Money anchors.** US median household income is roughly **\$70,000**; median *personal* income is closer to **\$40,000**. A cup of coffee is **\$3 to \$5**. A restaurant meal is **\$15 to \$40**. A new car is **\$30,000 to \$40,000**. US GDP is about **\$27 trillion**. These let you sanity-check whether a market size is plausible.

**Market anchors (the trader's set).** The S&P 500 index is around **6,000 points** (as of early 2026). A typical large-cap share price is **\$20 to \$200**. Large-cap average daily volume runs **1 million to 50 million shares**. The total US stock market is worth roughly **\$50 trillion**. Total US equity trading is on the order of **\$500 billion a day**. Know these and the dollar-flow questions answer themselves.

The skill is not memorizing every number -- it is knowing *which anchor* a factor reduces to, and being honest that an anchor is a range, not a point. "Big-city population" is "1 to 10 million," and which end you pick depends on the city. Stating the anchor and its range out loud is exactly the "strong" behavior the grading rubric rewards.

#### Worked example: a coffee shop's annual revenue

Anchors turn a vague question into a chain of multiplications. Estimate the annual revenue of a single busy coffee shop.

Decompose: annual revenue equals customers per day, times average spend per customer, times days open per year.

- **Customers per day.** A busy shop might serve a customer every two minutes during a 10-hour day -- that is 30 per hour times 10 hours equals **300 customers a day**. Anchor: a few hundred.
- **Average spend.** A coffee is about \$4; some people add a pastry. Anchor the average ticket at **\$6**.
- **Days open per year.** Open most days, closed a few holidays: **360 days**.

Multiply:

$$300 \times \$6 \times 360 = \$648{,}000 \approx \$650{,}000 \text{ per year}$$

So a busy independent coffee shop pulls in **roughly \$650,000 a year** in revenue. Does that feel right? It is in the ballpark of real small-cafe revenue, which typically runs a few hundred thousand to over a million dollars. **The intuition: a small business's revenue is daily customers times average ticket times days open -- three anchors and a multiply.** If the interviewer pushes back ("a major chain location?"), you scale the factors -- more customers, longer hours -- and the structure carries you.

## Recombining: multiplying ranges and thinking in orders of magnitude

When you multiply your anchors back together, two things matter that beginners get wrong. First, you should think in *powers of ten*, not in exact digits. Second, when each factor is a *range* rather than a point, you need to combine ranges honestly -- and the right center of a range is not its average.

### Multiply the exponents, not the digits

The fastest way to multiply rough numbers is to count zeros. \$50 is $5 \times 10^1$. Ten million is $10^7$. Multiply the powers of ten by adding the exponents ($1 + 7 = 8$) and multiply the leading digits ($5 \times 1 = 5$), giving $5 \times 10^8 = \$500$ million. You almost never need long multiplication; you need to track the order of magnitude and the leading digit. This is why Fermi estimates are fast: the arithmetic is mostly bookkeeping of zeros.

![Thinking in orders of magnitude: on a log scale every tick is a factor of ten, so piano tuners cluster near one hundred, a city near five million, a stock's turnover near five hundred million, and US GDP near twenty-seven trillion](/imgs/blogs/estimation-fermi-problems-quant-interviews-3.png)

The number line above is logarithmic: each tick to the right is a factor of ten, not a fixed addition. On a log scale, "100 piano tuners," "5 million people in a city," "\$500 million daily turnover," and "\$27 trillion of GDP" each occupy a clean position, and the spacing between them is the number of orders of magnitude between them. **Training yourself to see quantities on this log line is the core mental shift of estimation:** you stop asking "what is the exact value?" and start asking "which tick is it nearest?" Being one tick off is a small error; being three ticks off is a disaster you will usually catch with a sanity-check.

### Ranges, and why the geometric mean is the honest center

Real anchors are ranges. "Big-city population" is 1 to 10 million. "Share price" is \$20 to \$200. When you only have a low and a high, what single number should you carry forward as your best guess?

The instinct is to take the average -- the **arithmetic mean**, $(L + H)/2$. For a range that spans a big factor, this is badly wrong, and here is why. Suppose a quantity could be as low as 10 or as high as 1,000. The arithmetic mean is $(10 + 1000)/2 = 505$. But 505 is essentially 1,000 -- it is fifty times bigger than the low end and only twice the high end. The arithmetic mean of a wide range is dominated by the high number and ignores the low one. It is biased high.

The honest center is the **geometric mean**: the square root of the product, $\sqrt{L \times H}$. For our range, $\sqrt{10 \times 1000} = \sqrt{10000} = 100$. And 100 is exactly ten times the low (10) and exactly one-tenth the high (1,000) -- equally far from both *on a log scale*. That is what "the middle" should mean for a quantity you are estimating to within an order of magnitude.

![Why the geometric mean centers a range: on a log axis the geometric mean of ten and one thousand is one hundred, exactly ten times above the low and ten times below the high, while the arithmetic mean of 505 sits biased toward the high end](/imgs/blogs/estimation-fermi-problems-quant-interviews-9.png)

The figure makes the contrast visual. On the log axis, the green geometric mean (100) sits dead center between low and high; the amber arithmetic mean (505) sits jammed up against the high end. The rule, stated cleanly: **the geometric mean of $L$ and $H$ is $\sqrt{L \times H}$, which is the average of their exponents, not their values.** For 10 ($10^1$) and 1,000 ($10^3$), the average exponent is 2, giving $10^2 = 100$. When the range is narrow (say \$45 to \$55), the geometric and arithmetic means are almost identical, so it does not matter. When the range is wide -- which is exactly when you are most uncertain -- the geometric mean is the one that does not lie to you.

### What happens to uncertainty when you multiply factors

Each factor in your decomposition is a range, and multiplying ranges *widens* the uncertainty. This is not a flaw; it is information about how confident you should be.

![Multiplying ranges: each factor is a low-high bracket, the center is the product of geometric means, and because each factor spans four-x the three-factor result spans sixty-four-x from twenty million to one point two-eight billion dollars](/imgs/blogs/estimation-fermi-problems-quant-interviews-4.png)

The figure works a concrete case. Say you are sizing a product's revenue as households times annual buy-rate times price. Each factor is a bracket:

- Households: low **1 million**, high **4 million**, geometric mean **2 million**.
- Annual buy-rate: low **0.1**, high **0.4**, geometric mean **0.2**.
- Price: low **\$200**, high **\$800**, geometric mean **\$400**.

Multiply the geometric means for your center estimate:

$$2{,}000{,}000 \times 0.2 \times \$400 = \$160{,}000{,}000$$

Now multiply the lows together and the highs together for the band:

$$\text{low} = 1{,}000{,}000 \times 0.1 \times \$200 = \$20{,}000{,}000$$
$$\text{high} = 4{,}000{,}000 \times 0.4 \times \$800 = \$1{,}280{,}000{,}000$$

Each factor spanned a factor of 4 (low to high), and three of them multiplied together span $4 \times 4 \times 4 = 64$, so the result band runs from \$20 million to \$1.28 billion -- a 64x spread around a \$160 million center. **The intuition: multiplying uncertain factors compounds their uncertainty, so a three-factor estimate is honestly a wide band -- and the geometric-mean product is your single best point inside it.** That width is not a failure of the method; it is the method telling you the truth about how much you know.

## Sanity-checking and bounding: bracket the answer from both sides

You have a number. Before you say it out loud, you check it -- and the most powerful check is to bound the answer from both directions with bounds you are *certain* about.

A **lower bound** is a number you are sure the true answer is *at least*. An **upper bound** is a number you are sure it is *at most*. If you can find a floor that is obviously too low and a ceiling that is obviously too high, the truth is trapped between them, and your estimate had better live in that bracket.

![Bracketing an answer: a lower bound of one hundred million dollars you are sure is too low and an upper bound of two billion you are sure is too high trap the truth, and the geometric mean of the bounds gives a best guess of four hundred fifty million dollars](/imgs/blogs/estimation-fermi-problems-quant-interviews-7.png)

The figure shows the geometry. The red zones are impossible -- below the floor and above the ceiling. The green zone in the middle is where the answer must live. Say you are estimating a market size and you reason: "It is surely worth at least \$100 million -- I can name customers paying that much already. And it is surely no more than \$2 billion -- that is bigger than the entire adjacent category." Now any estimate outside \$100 million to \$2 billion is wrong by construction, and your best single guess is the geometric mean of the bounds:

$$\sqrt{\$100{,}000{,}000 \times \$2{,}000{,}000{,}000} = \sqrt{2 \times 10^{17}} \approx \$450{,}000{,}000$$

A useful rule of thumb: **if your lower and upper bounds are within a factor of 10 of each other, your estimate is interview-grade.** A bracket of \$100 million to \$2 billion is a factor of 20 -- a bit wide, but the geometric mean still pins a defensible \$450 million. If you can tighten the bracket to within 10x, you are done.

There are three other sanity-checks worth running every time, fast:

**The unit check.** Make sure the units of your factors actually multiply to the units of the answer. Shares times dollars-per-share gives dollars -- good. If you find yourself multiplying "people" by "people" to get a dollar figure, you have decomposed wrong. Units that do not cancel correctly are the single most common silent error.

**The cross-check from a second decomposition.** Estimate the same quantity a different way and see if the two agree to within an order of magnitude. We will do this in the piano-tuner problem below: estimate the number of tuners from the demand side (pianos needing tuning) and confirm it against the supply side (tuners' working capacity). Two independent paths landing on the same order of magnitude is strong evidence you are right.

**The "does it feel insane?" check.** Step back and look at the number with human eyes. "This city needs 100,000 piano tuners" should immediately feel absurd -- that is more tuners than teachers. If your answer fails the gut check, you dropped or added a factor of a thousand somewhere; go find it. Saying this check out loud -- "let me sanity-check that, 100 tuners for a city of 5 million feels about right, one per 50,000 people" -- is itself a strong-candidate behavior.

## Market sizing and flow estimation: the trader's dialects of Fermi

Two specific flavors of estimation come up so often on a trading desk and in interviews that they deserve their own treatment: **market sizing** (how big is the dollar opportunity?) and **flow estimation** (how much trades, and how much of it can I touch?).

### Market sizing: the funnel from everyone to your dollars

Market sizing asks: what is the total dollar value of a market for some product? The clean way to do it is a **funnel** -- start from the whole population and narrow it through a series of filters until you reach the people who actually pay you, then multiply by price.

![The market-sizing funnel: 260 million US adults narrow through filters for smartphone ownership, wanting the product, and choosing you, leaving 2.2 million customers who at fifty dollars a year produce one hundred ten million dollars of revenue](/imgs/blogs/estimation-fermi-problems-quant-interviews-6.png)

The funnel above sizes a subscription app. Read it top to bottom: start with **260 million US adults**. Filter to smartphone owners (about 85%) for a **total addressable market** of **220 million** -- the TAM, everyone who *could* use the product. Filter to the fraction who actually want this kind of product (say 20%) for a **serviceable market** of **44 million** -- the SAM, the realistic universe of buyers. Filter to your share of those (say you win 5%) for **2.2 million** actual customers -- the SOM, what you can realistically capture. Multiply by price (\$50 a year):

$$2{,}200{,}000 \times \$50 = \$110{,}000{,}000 \text{ per year}$$

The three acronyms -- **TAM** (total addressable market, everyone who could buy), **SAM** (serviceable available market, everyone you could realistically sell to), and **SOM** (serviceable obtainable market, the slice you actually win) -- are just three stages of the same funnel, and you should name them when you use them. The discipline is to state each filter percentage and why. An interviewer does not care whether your win-rate is 5% or 8%; they care that you said "I'll assume we win about 5% of a competitive market" and then carried it through cleanly.

### Flow estimation: how much trades, and how much can I touch

Flow estimation is the trader-native version. The questions are: how much of this instrument trades per day (the dollar turnover we computed earlier)? And given that flow, how big a position can I build or unwind without becoming the market myself?

The second question introduces **capacity** -- the most a strategy can trade before its own activity moves prices against it and eats the edge. A common rule of thumb is that you can trade up to some small percentage of average daily volume without excessive **market impact** (the price moving against you because of your own buying or selling). If a stock's ADV is 10 million shares and you keep to 5% of ADV, your capacity is 500,000 shares a day. At \$50 a share, that is \$25 million of **notional** -- the total dollar value of the position, computed as shares times price -- that you can put on per day in that one name.

This is why the dollar-turnover estimate is not academic. A trader who has a signal that makes 2 cents a share has to ask, immediately: how many shares can I trade before I move the price and lose those 2 cents? The answer is a flow estimate, and they make it before sending the order. We will work the full version in the desk section.

## In the interview room: six problems, fully solved

Theory is cheap. Here are six problems of the kind firms actually ask -- five pure estimation questions and one pricing question that uses the identical loop -- each solved the way you should solve it out loud: decomposition first, anchors stated, arithmetic in powers of ten, sanity-check at the end. Read each one as a script for how to *talk*, not just what to compute.

### Worked example 1 -- how many piano tuners are in a large city?

This is the original Fermi problem and still a favorite. The unknown is the number of working piano tuners in a city of, say, 5 million people.

**Decompose from the demand side.** The number of tuners must satisfy the demand for tunings. So:

$$\text{number of tuners} = \frac{\text{tunings demanded per year}}{\text{tunings one tuner supplies per year}}$$

Now decompose each side.

![Piano tuners in a city decompose into demand over supply: a five million city has about two million households, one in twenty owns a piano, each tuned once a year is one hundred thousand tunings, and one tuner doing four a day for two hundred fifty days supplies one thousand, giving about one hundred tuners](/imgs/blogs/estimation-fermi-problems-quant-interviews-5.png)

**Demand side (the numerator).** How many pianos are there? Start from households: 5 million people at about 2.5 per household is **2 million households**. What fraction owns a piano? Pianos are not rare but not common -- anchor at **1 in 20**, giving $2{,}000{,}000 / 20 = 100{,}000$ pianos. How often is each tuned? A casual owner tunes roughly **once a year**. So demand is about **100,000 tunings per year**.

**Supply side (the denominator).** How many tunings can one tuner do in a year? A tuning takes a couple of hours including travel, so call it **4 per day**. Working **250 days** a year gives $4 \times 250 = 1{,}000$ **tunings per tuner per year**.

**Recombine.**

$$\text{number of tuners} = \frac{100{,}000}{1{,}000} = 100$$

About **100 piano tuners** in a city of 5 million.

**Sanity-check.** That is one tuner per 50,000 people, or one per 1,000 pianos. Does that feel right? A tuner servicing 1,000 pianos, each once a year, is exactly the 1,000-tunings-a-year capacity we computed -- the demand and supply sides agree by construction, which is the cross-check working. And 100 specialists in a metro area is a plausible number for a niche trade. The answer is solid. Notice we never needed to know the *true* number of pianos; we needed a defensible chain, and we got one.

### Worked example 2 -- what is the daily dollar turnover of a given stock?

This is the most trading-relevant estimation question, and we solved the core of it earlier -- but in the room you should narrate it as a clean two-factor estimate and then push it further than the basic version, because interviewers reward candidates who extend.

**Decompose.** Daily dollar turnover equals average daily volume times price:

$$\text{turnover} = \text{ADV} \times \text{price}$$

**Anchor.** A liquid large-cap might trade **10 million shares** a day at **\$50**.

**Recombine.**

$$10{,}000{,}000 \times \$50 = \$500{,}000{,}000 \text{ per day}$$

**\$500 million a day.** Now extend it the way a strong candidate would, unprompted:

- **Annualize.** Times 250 trading days is about **\$125 billion a year** of turnover in one name.
- **Relate it to market cap.** If the company has 2 billion shares outstanding at \$50, its market capitalization (shares outstanding times price -- the total value of the company's equity) is \$100 billion. Daily turnover of \$500 million is **0.5% of market cap traded per day**, which means the entire company "turns over" about every 200 trading days, or roughly once a year. That is a typical, healthy turnover ratio, and stating it shows you understand what the number means.
- **Sanity-check against the whole market.** Total US equity turnover is on the order of \$500 billion a day. Our one stock at \$500 million is about 0.1% of all US trading -- plausible for a single large-cap among thousands of listed names.

**The point:** the base calculation is trivial, but extending it into market cap, turnover ratio, and market share is what separates a "fine" answer from a "hire this person" answer. Same loop, pushed two steps further.

### Worked example 3 -- how many golf balls fit inside a school bus?

A pure volume estimation -- no finance, just geometry and the willingness to make assumptions. The unknown is how many golf balls fill the interior of a school bus.

**Decompose.** The count is the usable interior volume of the bus divided by the volume one golf ball effectively occupies (including the wasted space between packed spheres):

$$\text{count} = \frac{\text{bus interior volume}}{\text{effective volume per ball}}$$

**Anchor the bus.** A school bus is about 10 meters long, 2.5 meters wide, and 2 meters tall inside. But seats and the driver area eat space -- call usable volume 70% of the box. So:

$$10 \times 2.5 \times 2 = 50 \text{ cubic meters}, \quad \times 0.7 \approx 35 \text{ cubic meters}$$

**Anchor the ball.** A golf ball is about 4 centimeters across, so it fits in a cube about 4 cm on a side: $0.04^3 = 0.000064$ cubic meters, or $6.4 \times 10^{-5}$ m³. Spheres do not pack perfectly -- random packing wastes roughly a third of the space -- so the *effective* volume per ball is larger; bump it to about $1 \times 10^{-4}$ m³ to account for the gaps. (Using the cube already over-counts the empty space somewhat, so this is a reasonable rough handling.)

**Recombine.**

$$\frac{35}{1 \times 10^{-4}} = 350{,}000 \text{ golf balls}$$

So **on the order of several hundred thousand** golf balls -- call it 300,000 to 500,000.

**Sanity-check.** The answer is dominated by the cube of the ratio of bus-size to ball-size, so small changes in your assumptions move it a lot -- which is exactly why you give a *range* (a few hundred thousand) and not a false-precise "417,332." Bracket it: surely more than 100,000 (the bus is enormous compared to a ball) and surely fewer than 10 million (that would be one ball per cubic centimeter, denser than solid packing allows). Our 350,000 sits comfortably in the bracket. State the range, state the bracket, and move on.

### Worked example 4 -- what is the dollar market size for a product?

A market-sizing problem -- run the funnel. The unknown is the annual US market size for, say, premium wireless earbuds.

**Decompose** via the funnel: market size equals buyers per year times price.

- **Start from the population.** 330 million people, about 260 million adults.
- **Filter to plausible buyers.** Premium earbuds appeal to maybe **40%** of adults (people who own decent phones and care about audio): $260{,}000{,}000 \times 0.4 \approx 100{,}000{,}000$ potential buyers -- this is the TAM.
- **Filter to annual purchasers.** People replace earbuds every few years, so in any given year maybe **1 in 4** of those buyers actually buys a new pair: $100{,}000{,}000 / 4 = 25{,}000{,}000$ buyers this year.
- **Anchor the price.** Premium earbuds run **\$150 to \$250**; anchor at **\$200**.

**Recombine.**

$$25{,}000{,}000 \times \$200 = \$5{,}000{,}000{,}000 \text{ per year}$$

About **\$5 billion a year** for the premium-earbuds market in the US.

**Sanity-check with a bracket.** Lower bound: surely at least \$1 billion -- a single major manufacturer does close to that. Upper bound: surely under \$30 billion -- that would rival the entire US smartphone-accessory category. Our \$5 billion sits inside, and its geometric mean with sensible bounds is the same order. Cross-check from the other direction: global consumer audio is tens of billions a year, and the US premium slice being a few billion is consistent. Two paths, same order of magnitude -- confident answer. **State your filters, state your bracket, name the TAM, and you have given a textbook market-sizing answer.**

### Worked example 5 -- what is a vending machine's expected annual revenue?

A revenue estimation with a probability flavor -- and a great one because it forces you to estimate a *fraction* (the conversion rate of passersby into buyers). The unknown is the dollars one well-placed vending machine brings in per year.

![A vending machine's yearly revenue decomposes into revenue per day times operating days: about ten units a day at two dollars is twenty dollars a day, and across three hundred fifty days that is about seven thousand dollars a year](/imgs/blogs/estimation-fermi-problems-quant-interviews-10.png)

**Decompose** as daily revenue times operating days, and daily revenue as units sold times price:

$$\text{annual revenue} = (\text{units per day}) \times (\text{price}) \times (\text{days per year})$$

**Estimate units per day via a conversion rate.** Suppose the machine sits in a moderately busy hallway with about **500 people** passing each day. What fraction buy something? Most people walk past; anchor the conversion at **1 in 50**, giving $500 / 50 = 10$ **units sold per day**. (This is an *expected value* -- on average 10 sales a day, even though any individual passerby is unlikely to buy. Framing it as a per-person probability times the number of people is exactly the indicator-variable trick from [expected value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews).)

**Anchor the price.** A soda or snack is about **\$2**.

**Anchor the days.** The machine runs nearly every day: **350 days**.

**Recombine.**

$$10 \times \$2 \times 350 = \$7{,}000 \text{ per year}$$

About **\$7,000 a year** in revenue from one machine.

**Sanity-check.** Is that plausible? It is \$20 a day, or about 10 sales -- a believable trickle for a single hallway machine. Bracket it: surely more than \$1,000 (even a slow machine sells a few items a day) and surely under \$50,000 (that would be 70 sales a day, a convenience-store volume that a single machine in a hallway cannot sustain). \$7,000 sits comfortably inside. If the interviewer asks about a machine in a packed train station, you scale foot traffic up by 10x and revenue follows -- the structure holds, only the anchor moves. **The intuition: a revenue estimate built on a conversion rate is just traffic times conversion times price times days -- and the conversion rate is itself an expected-value estimate.**

### Worked example 6 -- what is a fair price to pay to play this game?

Trading interviews love to slip a pricing question in among the estimation ones, because pricing a game is exactly what a market maker does -- and the tool is the same expected-value-then-bracket loop. The game: *I roll a fair six-sided die once and pay you that many dollars. What would you pay to play?*

**Decompose into expected value.** The fair price of a one-shot gamble is its **expected value** -- the probability-weighted average of the payoffs (the long-run average if you played it many times). Each face 1 through 6 is equally likely, with probability $1/6$, and pays its face value in dollars:

$$E = \tfrac{1}{6}(\$1 + \$2 + \$3 + \$4 + \$5 + \$6) = \tfrac{1}{6} \times \$21 = \$3.50$$

So the game is worth **\$3.50**, the average of the six payoffs. If you can play for less than \$3.50 you have positive expected value (a good bet); more than \$3.50 and you are overpaying. As a market maker you would quote a two-sided market *around* \$3.50 -- maybe bid \$3.40, offer \$3.60 -- capturing the spread while staying close to fair value.

**Extend it the way an interviewer wants.** Now suppose you get to *re-roll once* if you do not like the first roll, keeping the second result. What is it worth now? Decompose by the decision rule: you should re-roll whenever the first roll is below the expected value of a fresh roll, which is \$3.50 -- so re-roll on a 1, 2, or 3, and keep a 4, 5, or 6. With probability $1/2$ you keep the first roll (averaging \$5, the mean of 4, 5, 6); with probability $1/2$ you re-roll and get a fresh \$3.50 expected value:

$$E = \tfrac{1}{2} \times \$5 + \tfrac{1}{2} \times \$3.50 = \$4.25$$

The option to re-roll is worth **\$4.25 minus \$3.50 = \$0.75**.

**Sanity-check.** The re-roll value must be positive (an option to improve can only help) and bounded above by \$6 (the most you can ever win), so \$4.25 sits sensibly between the no-re-roll \$3.50 and the ceiling. **The intuition: the fair price of any game is its expected value, and an optional do-over is worth the expected gain from exercising it only when it helps -- the same logic by which an option's value is the expected payoff of using it optimally.** This is where estimation and derivatives pricing touch: a market maker prices a game, an insurance contract, and a call option with one move -- expected value, then a spread around it.

### A note on the related interview categories

Estimation is one pillar of the quant trading interview, and it interlocks with the others. The conversion-rate move in the vending-machine problem is an expected-value calculation in disguise; the probability questions in [classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) and [conditional probability and Bayes](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) share the same "decompose and recombine" spirit; and the fast multiplication of anchors leans on the same [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) that turn a messy count into a sum. Estimation is where they meet: a Fermi problem is a decomposition (combinatorial thinking), populated with expected values (probability), executed at speed (mental math), and reported in orders of magnitude (the estimation discipline itself).

## Common misconceptions

Even strong candidates carry wrong beliefs about estimation into the interview. Here are the ones that cost the most.

**"There is a right answer and I have to find it."** There is not. A Fermi problem has a range of defensible answers, and the interviewer has a range of acceptable ones -- usually anything within an order of magnitude of their own estimate, *with sound reasoning*. Hunting for the "correct" number makes you slow and brittle. Aim for defensible, not correct. The candidate who relaxes into "let me build a reasonable estimate" outperforms the one straining for an exact hit.

**False precision -- reporting more digits than you have.** This is the single most damaging tell. If you guessed every factor to one significant figure, your answer has one significant figure of meaning. Reporting "\$487,213,902" when you mean "about half a billion" signals that you do not understand what estimation *is*. Round aggressively: say "roughly \$500 million" or "a few hundred million." The rounding is not sloppiness -- it is honesty about your uncertainty, and interviewers read it as sophistication.

**Anchoring bias -- letting the first number you hear or say pull your estimate.** "Anchoring" has a second meaning in psychology: the tendency to stick too close to an initial number, even a wrong one. If you blurt an early guess of "a million" and then build your decomposition, you will unconsciously bend the factors to land near a million. Guard against it by building the decomposition *first*, from independent anchors, and only then computing the product -- so the answer surprises you rather than confirming a number you pre-committed to. The order matters: structure before number.

**Not bracketing -- giving a point estimate with no bounds.** A bare number ("\$5 billion") is weaker than a bracketed one ("between \$1 billion and \$30 billion, best guess around \$5 billion"). The bracket proves you have thought about the bounds and demonstrates the sanity-check the rubric rewards. Candidates who skip it leave easy points on the table -- and risk reporting a number that is wildly outside any sensible range without noticing.

**Averaging ranges arithmetically.** As we saw, the arithmetic mean of a wide range is biased toward the high end. A candidate who takes "10 to 1,000" and carries 505 forward has baked a high bias into every downstream factor. Use the geometric mean for any range spanning more than about a factor of 3, and the bias disappears.

**Treating it as a math test instead of a thinking-out-loud test.** The arithmetic in a Fermi problem is deliberately easy -- multiplying round numbers. The interviewer is not testing whether you can multiply; they are testing how you *structure* and *communicate* a messy problem. Silence while you compute is wasted signal. Narrate: "I'll break this into A times B times C, I'll estimate A as roughly this because..., now multiplying..." The talking is the test.

**Over-decomposing into more factors than you can estimate.** More decomposition is not always better. Each factor you introduce is another guess with its own error, and as we saw, errors compound multiplicatively. If you can anchor "pianos in a city" directly at 100,000 with reasonable confidence, do not insist on deriving it through five sub-factors -- each sub-factor adds uncertainty. Decompose only until the leaves are anchorable, then stop.

## How it shows up on a real trading desk

Estimation is not a hazing ritual that ends after the interview. It is a daily, hourly habit on a trading desk, and the loop from the top of this post -- decompose, estimate, recombine, sanity-check -- runs constantly under real money. Here is where it actually bites.

![Estimation on a real trading desk: a trader estimates average daily volume to set capacity and fill probability, sizes an order against signal edge, and checks expected P&L and notional risk before every trade](/imgs/blogs/estimation-fermi-problems-quant-interviews-12.png)

The figure traces a single trade decision, and every box in it is a Fermi estimate made in seconds. Walk through it.

**Estimating average daily volume to size a trade.** Before a trader puts on a position, they estimate how much of the name trades per day -- the ADV from worked example 2. This is not always a clean lookup: for a newly volatile stock, an illiquid bond, or a thin option, the trader estimates ADV from related instruments, recent sessions, and gut feel. The estimate sets the ceiling on how aggressively they can trade.

**Estimating strategy capacity.** Given ADV, the trader estimates **capacity** -- how many shares they can trade before their own flow moves the price and eats their edge. The figure uses the 5%-of-ADV rule of thumb: with ADV of 10 million shares, capacity is about 500,000 shares a day. A real desk refines this with market-impact models, but the first-cut number is a Fermi estimate, and getting it wrong by an order of magnitude is how strategies "blow up on the way in" -- the trader tries to build a position too big for the liquidity and pushes the price away from themselves before they are done.

**Estimating fill probability.** When a trader posts a passive order at the bid (offering to buy at the current best buy price), they estimate the probability it actually gets filled before the market moves. The figure anchors it at about 30% -- a number the trader estimates from how fast the queue is trading and how volatile the stock is. Fill probability times edge-per-share is the expected value of posting the order, and that expected-value framing is the same one from the vending machine.

**Sizing the order and checking expected P&L.** Combining a signal worth about \$0.02 per share of edge, the capacity ceiling, and the fill probability, the trader sizes the order -- say 50,000 shares. The **notional** (shares times price) is $50{,}000 \times \$50 = \$2.5$ million. The expected profit is $50{,}000 \times \$0.02 = \$1{,}000$ if the signal is right and the order fills. That \$1,000 expected P&L on \$2.5 million of notional is a thin edge -- about 4 basis points (a basis point is one hundredth of a percent, 0.01%) -- which is exactly the kind of margin a market maker lives on, repeated thousands of times a day.

**Checking notional against risk limits.** Before the order goes out, the trader sanity-checks the \$2.5 million notional against their position limit and the desk's risk budget. This is the bracketing discipline applied to risk: is this position within the bounds I am allowed to take? An estimate that blows through a limit gets caught here -- or, if the trader skipped the check, it gets caught by the risk desk, which is a worse outcome.

Beyond a single trade, the same estimation muscle shows up across the desk:

**Quoting an illiquid instrument.** A corporate-bond or single-name options market maker is asked to quote something that has not traded today. They estimate fair value by decomposing: a comparable bond's yield plus a spread for the credit difference, or a nearby option's implied volatility adjusted for the strike gap. There is no screen price; there is only an estimate, and they commit a two-sided market to it. The candidates who can do golf balls in the interview are the ones who can quote the unprintable bond on the desk.

**Estimating a strategy's total capacity before launch.** Before a fund commits capital to a new strategy, someone estimates how much money it can run before the edge decays. That is a market-sizing problem: total addressable flow, times the fraction the strategy can capture, times the edge per dollar. Over-estimate it and the strategy disappoints; under-estimate it and you leave money on the table. It is the funnel from the market-sizing section, applied to alpha.

**Pre-mortem sanity-checks on a P&L.** When a desk's daily P&L comes in surprisingly large -- positive or negative -- a good trader immediately Fermi-checks it. "We made \$2 million today; we run about \$200 million of notional; that is a 1% move on the book, which matches the market being up about 1% -- consistent." A P&L that fails the back-of-envelope check is a flag for an error: a fat-fingered trade, a mismarked position, a model bug. The estimate is the smoke detector.

In all of these, the thing being tested in the interview -- can you put a defensible number on the unknown, fast, and know how much to trust it -- is the thing the job *is*. The golf balls were never the point.

## When this matters and where to go next

If you are preparing for a quant trading interview, estimation is among the highest-return topics you can drill, because it is almost pure technique. Unlike a probability brain-teaser that may hinge on spotting one clever trick, every Fermi problem yields to the same loop -- decompose, anchor, recombine, bracket -- and the loop is learnable in an afternoon and sharpenable with reps. Spend an hour memorizing the anchor card until the population, time, and market numbers are reflexive. Then do twenty problems out loud, on a whiteboard, narrating every step, and record yourself: you will hear your own false precision, your skipped brackets, your silent computing, and you will fix them faster by hearing them than by reading about them.

A practice regimen that works: pick a random object or business you can see -- a parking garage, a pizzeria, the building you are in -- and estimate something about its economics or its physical count, out loud, in under two minutes, ending with a bracket. "How much revenue does this parking garage make a year?" "How many bricks are in this building?" "What is the daily dollar turnover of the most-traded stock I can name?" The everyday-ness is the point: a trader estimates the world around them reflexively, and the habit is built by doing it on real things, not just on flashcards.

Beyond estimation itself, the other pillars of the quant trading interview reward the same decompose-and-recombine instinct. The [expected value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) post shows how linearity and indicators turn a messy count into a sum you can add up term by term -- the probability cousin of Fermi decomposition. [Conditional probability and Bayes](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) handles the questions where new information should update your estimate. [Classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) and [counting and combinatorics](/blog/trading/quantitative-finance/counting-combinatorics-quant-interviews) cover the brain-teasers that sit alongside estimation in a typical loop. Work through all of them and you will have the full toolkit a market-making firm is probing for: the ability to take something uncertain, break it into pieces you can reason about, recombine them honestly, and put a number on the result that you are willing to trade on.

That last phrase is the whole job. A trader is someone who is paid to put defensible numbers on uncertain things and bet accordingly. The piano tuners, the golf balls, the \$500-million-a-day stock -- they are all the same question, and now you know the loop that answers it.

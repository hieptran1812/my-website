---
title: "Base, Quote, Pips: How to Actually Read an FX Quote"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Read any currency quote line by line — base vs quote, bid vs ask, pips and pipettes, lot sizes — and compute your profit in dollars per pip per lot."
tags: ["forex", "currencies", "fx-quote", "pips", "bid-ask-spread", "lot-size", "base-currency", "pip-value", "trading-basics", "exchange-rate"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An FX quote is one number: the price of the *base* currency measured in the *quote* currency, shown as two prices (a bid you sell at, an ask you buy at). Learn to read those five things — base, quote, bid, ask, spread — and you can compute your profit in dollars on any trade.
>
> - In `EUR/USD = 1.0853`, the euro is the **base** (the thing being priced) and the dollar is the **quote** (the unit of the price): one euro costs 1.0853 dollars.
> - A **pip** is the standard smallest step in the price — the 4th decimal for most pairs (`0.0001`), the 2nd decimal for yen pairs (`0.01`). A **pipette** is one tenth of that.
> - **Pip value = pip size × position size.** On a standard lot of 100,000 units, one pip of EUR/USD is worth **\$10**. A 50-pip move is **\$500**.
> - **The one fact to remember:** when EUR/USD goes *up*, the euro got *stronger* and the dollar got *weaker* — the base rising always means the quote currency is buying less.

On the morning of 15 January 2015, a trader watching EUR/CHF saw the quote sitting calmly at `1.2010`. The Swiss National Bank had promised for three years to defend a floor at `1.20` — it would print unlimited francs to stop the pair falling below. Then, at 09:30 Zurich time, the central bank abandoned the floor without warning. Within three minutes the quote on the screen went from `1.2005` to `1.0500` to `0.8500`. Anyone who could not read what those numbers *meant* — which way the franc had moved, how many pips, how much money per lot — had no chance of reacting in time. Fortunes were made and lost in the gap between two prices.

You do not need to trade a central-bank shock to need this skill. Every single FX position you will ever take starts the same way: a quote appears on a screen, two prices side by side, four or five decimal places, and you have to know — instantly — what currency you are buying, what you are paying in, how much one tick is worth, and which direction means you are making money. That is the entire job of this post. By the end you will be able to look at any currency quote, name every part of it out loud, and compute your profit or loss in dollars without a calculator.

This is the second post in the series, and it builds the literacy everything else depends on. The first post, [why currencies are different](/blog/trading/forex/why-currencies-are-different-fx-trading-introduction), made the core argument: you never own *a currency* in isolation — every FX position is a **pair**, a relative bet of one money against another. This post takes that idea and makes it concrete, down to the last decimal place. Here is the whole quote, taken apart piece by piece.

![Grid showing the parts of a EUR USD quote — base, quote currency, bid, ask, spread, pip, and what the price reads as](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-1.png)

## Foundations: How an FX quote works

Start from zero. A currency quote is a price, and a price always answers one question: *how much of one thing does it take to buy another thing?* When you see `EUR/USD = 1.0853`, that is the answer to "how many US dollars does it take to buy one euro?" The answer is 1.0853 dollars. That is the whole secret. Everything else is vocabulary for the parts of that sentence.

### The base currency and the quote currency

Every pair has two currencies, written with a slash between them, like `EUR/USD` or `USD/JPY` or `GBP/USD`. The currency **on the left** is the **base currency**. The currency **on the right** is the **quote currency** (also called the *counter* or *terms* currency). The rule that unlocks everything:

> The price tells you how many units of the **quote** currency it takes to buy **one unit** of the **base** currency.

So in `EUR/USD = 1.0853`, the base is the euro, the quote is the dollar, and the price says *one euro = 1.0853 dollars*. The base is always priced in single units of one. The quote currency is the measuring stick.

This is exactly how you already think about ordinary prices, you just do not call it a "quote". If a coffee costs \$4.50, the coffee is the base (one unit of it) and the dollar is the quote currency (the unit you measure the price in). An exchange rate is the same idea, except the "thing" you are pricing is itself a money. That is the one genuinely strange part of FX: the thing being bought and the unit it is priced in are both currencies, so you have to keep straight which is which.

The order is a fixed convention, not a choice. Markets quote `EUR/USD`, never `USD/EUR`, by long-standing agreement. There is a rough hierarchy that decides which currency goes first when two majors meet: the euro outranks everything, then the pound, then the Australian and New Zealand dollars, then the US dollar, then the rest. So you see `EUR/USD` and `GBP/USD` (euro and pound first), but `USD/JPY` and `USD/CHF` (dollar first, because it outranks the yen and the franc). You do not need to memorise the full pecking order — you just need to read the slash and know the left one is the base. The map of which pairs exist and how they are grouped is its own subject, covered in [majors, minors, and exotics](/blog/trading/forex/majors-minors-and-exotics-the-map-of-currency-pairs).

### The bid and the ask

So far we have talked about *the* price, singular. In reality a quote is always **two** prices, shown side by side:

```
EUR/USD   1.0851 / 1.0853
            bid      ask
```

The lower number is the **bid**. The higher number is the **ask** (also called the *offer*). Here is the rule for what they mean — and it is the one beginners get backwards most often, so anchor it from the dealer's point of view:

- The **bid** is the price at which the dealer will **buy** the base currency from you. So the bid is the price at which **you sell**.
- The **ask** is the price at which the dealer will **sell** the base currency to you. So the ask is the price at which **you buy**.

The trick to keeping it straight: the prices are quoted from the *dealer's* side of the trade. The dealer bids (offers to buy) low and asks (offers to sell) high — because the dealer wants to buy cheap and sell dear, just like any shop. You, the customer, are on the opposite side of every one of those: you buy at the dealer's high price (the ask) and sell at the dealer's low price (the bid). A currency exchange booth at the airport works exactly this way — it sells you euros at one rate and buys them back at a worse one, and the gap is its margin.

A second source of confusion is the word "buy" itself, because in FX every buy is also a sell. When you *buy* EUR/USD, you are simultaneously buying euros and selling dollars — there is no way to hold a one-sided position, because a currency only has a price *relative to* another currency. This is the relative-bet idea from the [introduction to FX](/blog/trading/forex/why-currencies-are-different-fx-trading-introduction) made mechanical: "long EUR/USD" means long euros and short dollars at the same time, in one trade. So when you read the ask as "the price you buy at", be precise about what you are buying — the **base** currency — and remember the quote currency is being sold in the same breath. Getting this backwards is the single most common way beginners end up positioned the opposite of what they intended.

### The spread

The gap between the bid and the ask is the **spread**. In our example the bid is `1.0851` and the ask is `1.0853`, so the spread is `1.0853 − 1.0851 = 0.0002`, which we will soon learn to call **2 pips**. The spread is the dealer's compensation for standing ready to trade with you at any moment, and it is a real cost you pay — we will quantify it in dollars later in the post. For now, hold onto the shape of the thing: **bid below, ask above, spread in between, and you always trade on the wrong side of it.**

### The pip and the pipette

A **pip** is the standard unit of price movement in FX — the smallest "normal" increment a quote moves in. The word originally stood for *percentage in point* or *price interest point*, but the etymology does not matter; what matters is the size. For almost every currency pair, **one pip is the fourth decimal place: `0.0001`**. So if EUR/USD moves from `1.0853` to `1.0854`, it has moved one pip. From `1.0853` to `1.0903`, it has moved 50 pips.

There is one big exception. For pairs that involve the **Japanese yen** — `USD/JPY`, `EUR/JPY`, and so on — one pip is the **second decimal place: `0.01`**. This is because the yen is a "small" currency: one dollar is worth roughly 150 yen, not 1.5, so the price has fewer decimals and the meaningful step is two places in. If USD/JPY moves from `150.00` to `150.50`, that is 50 pips, not 5,000.

A **pipette** (sometimes called a *fractional pip* or *point*) is one tenth of a pip — the fifth decimal for most pairs, the third for yen pairs. Modern platforms quote an extra digit for precision, so you will often see EUR/USD shown as `1.08531`, where that last `1` is a single pipette. The pipette lets dealers price more finely and shave spreads, but for almost all of your mental math you can ignore it and work in whole pips.

Look back at the cover figure above: it lays out exactly these parts — base, quote, bid, ask, spread, pip — on a single quote, with what each piece means. Keep it in mind as the skeleton; the rest of the post puts muscle on each bone.

### The lot

When you actually place a trade, you do not buy "some euros" — you buy a specific quantity, and FX measures that quantity in **lots**. A lot is just a standardised position size, counted in units of the **base** currency:

- A **standard lot** is **100,000 units** of the base currency (100,000 euros in EUR/USD).
- A **mini lot** is **10,000 units**.
- A **micro lot** is **1,000 units**.
- A **nano lot** is **100 units** (offered by some brokers for very small accounts).

The whole point of the lot is that it converts a tiny price wiggle — a single pip, four decimal places in — into a sum of money big enough to matter. One pip of EUR/USD is `0.0001` dollars per euro, which is nothing on a single euro. But on a standard lot of 100,000 euros, that one pip is worth `0.0001 × 100,000 = 10` dollars. That is the bridge between the quote and your bank account, and the next section builds it explicitly.

> [!note]
> **Why this matters before anything else.** Carry trades, central-bank interventions, crisis unwinds — every advanced topic in this series is told in the language of pips, spreads, and lots. If you cannot read a quote, the rest of the series is a foreign language. Get this post solid and everything downstream gets easier.

## The anatomy of a quote, read out loud

Let us take a real quote and narrate every part, the way you should be able to in your head within a second of seeing it. Here is the quote again, fully labelled:

```
EUR/USD   1.0851 / 1.0853
```

Reading it out loud: *"Euro-dollar. The base is the euro, the quote currency is the US dollar. The bid is 1.0851, the ask is 1.0853. I can sell one euro for 1.0851 dollars, or buy one euro for 1.0853 dollars. The spread is 2 pips. One euro currently costs me about 1.0853 dollars to buy."* That is the complete reading. Five facts — base, quote, bid, ask, spread — and one plain-English sentence about what a euro costs.

Now flip the pair. Here is `USD/JPY`:

```
USD/JPY   149.85 / 149.88
```

The base is now the **dollar** and the quote currency is the **yen**. The price says *one dollar = roughly 149.86 yen*. The bid `149.85` is where you sell a dollar (for 149.85 yen); the ask `149.88` is where you buy a dollar (for 149.88 yen). The spread is `149.88 − 149.85 = 0.03`, and because this is a yen pair, one pip is `0.01`, so the spread is **3 pips**. Notice how the pip definition quietly changed when the yen entered the quote — the same mechanical reading, but the decimal place that counts as a pip shifted from the fourth to the second.

#### Worked example: reading and pricing a full EUR/USD quote

You see `EUR/USD = 1.0851 / 1.0853` and you want to buy 100,000 euros (one standard lot). What does that actually cost you, and in what currency?

- You buy at the **ask**, `1.0853`. You are buying 100,000 euros.
- Cost in dollars = `100,000 euros × 1.0853 dollars/euro = \$108,530`.
- So opening one standard long EUR/USD costs \$108,530 of notional value. (You will not pay all of that in cash — leverage means you post a small *margin* deposit, a topic for a later post — but the notional, the size your profit and loss is calculated on, is \$108,530.)
- If you immediately wanted out, you would sell at the **bid**, `1.0851`, receiving `100,000 × 1.0851 = \$108,510`. You are down `\$108,530 − \$108,510 = \$20` the instant you trade — that \$20 is the spread, the dealer's cut.

The intuition: buying a lot of euros is just multiplying the number of euros by the ask price to get the dollar cost, and the \$20 you are immediately down is the toll you pay for crossing the spread.

### How quotes actually appear on a screen

In the wild, you rarely see a quote written as neatly as `1.0851 / 1.0853`. Three shorthand conventions trip up beginners, so name them now.

First, dealers often quote only the **last two or three digits** of each side, because the "big figure" (the leading part, `1.08`) does not change moment to moment. So a EUR/USD market quoted "51 / 53" means `1.0851 / 1.0853` — the `1.08` is assumed. On a fast-moving USD/JPY desk you might hear "85 / 88" for `149.85 / 149.88`. The full price is implied; only the moving digits are spoken. If you do not know the big figure already, you are not in the conversation.

Second, the spread is frequently stated as a single number rather than two prices. "EUR/USD is 0.2 wide" or "a 2-pip market" tells you the gap without naming the bid and ask, because for most purposes the spread *is* the information you need about cost. You then place that spread around the mid-price (the average of bid and ask) to reconstruct both sides.

Third, platforms display the extra **pipette** digit, so a retail screen shows `1.08531 / 1.08549` rather than `1.0853 / 1.0855`. The pipette (that fifth decimal) makes the price look more precise and lets the broker quote a spread of, say, 1.8 pips instead of rounding to 2. When you count pips, work from the fourth decimal and treat the fifth as a fractional tail; do not mistake a 1-pip move for a 10-pipette avalanche.

None of this changes the underlying object. It is still base, quote, bid, ask, spread — just compressed for speed. The fluent reader fills in the big figure automatically and hears "51 / 53" as a complete EUR/USD market without a beat of hesitation.

## Pip value: turning a price tick into dollars

This is the most important calculation in retail FX, so we will build it slowly and then drill it. The question is simple: *if the price moves one pip, how many dollars do I gain or lose?* The answer is a single multiplication.

> **Pip value = pip size × position size (in units of the base currency).**

The pip size is `0.0001` for most pairs and `0.01` for yen pairs. The position size is how many units of the base currency you hold — i.e. your lot size in units. Multiply them and you get the value of one pip, in the **quote** currency.

For one standard lot of EUR/USD:

```
pip value = 0.0001 × 100,000 = 10
```

So one pip is worth **\$10** (the quote currency here is the dollar, so the answer is already in dollars). For a mini lot it is `0.0001 × 10,000 = 1` dollar; for a micro lot, ten cents; for a nano lot, one cent. Each step down the lot ladder divides the pip value by ten, because each lot is one tenth the size of the one above it. The chart below makes the whole ladder visible at once.

![Pipeline from one pip to dollars showing pip size times lot size times pips moved equals profit](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-4.png)

The pipeline above is the engine of every FX P&L calculation: a pip becomes a pip *value* when you multiply by your lot size, and that pip value becomes a *profit* when you multiply by how many pips the price moved. There is no other formula. Learn this chain and you can price any move on any major pair in your head.

#### Worked example: a 50-pip move on a standard lot

You are long one standard lot of EUR/USD (100,000 euros), bought at `1.0850`. The price rises to `1.0900`. How much did you make?

- The move is `1.0900 − 1.0850 = 0.0050`, which in pips is `0.0050 ÷ 0.0001 = 50 pips`.
- Pip value for a standard lot = `0.0001 × 100,000 = \$10` per pip.
- Profit = `50 pips × \$10 = \$500`.

You can check it the long way: you bought 100,000 euros for `100,000 × 1.0850 = \$108,500` and sold them for `100,000 × 1.0900 = \$109,000`, a difference of `\$500`. Same answer. The pip-value shortcut just saves you from multiplying six-figure numbers in your head.

The intuition: on a standard lot, **every pip is ten dollars**, so a 50-pip move is \$500, a 100-pip move is \$1,000, and a 10-pip scalp is \$100 — you can read the dollars straight off the pip count.

The chart below shows the pip value at each rung of the lot ladder, with the standard-lot 50-pip move marked. Notice that the dollar value scales by tens — this is why your choice of lot size, not just your view on the price, is what really controls how much money is at stake.

![Bar chart of the dollar value of one pip for standard, mini, micro, and nano lots on a log scale](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-5.png)

### A wrinkle: pip value when the quote currency is not the dollar

There is one honest complication. The pip-value formula gives you the answer **in the quote currency**, not automatically in dollars. For EUR/USD the quote currency *is* the dollar, so we got dollars for free. But consider `EUR/GBP`: the quote currency is the pound, so one pip of a standard lot is `0.0001 × 100,000 = £10`, not \$10. To express that in dollars, you convert the £10 at the current GBP/USD rate. If GBP/USD is `1.27`, then `10 × 1.27` gives \$12.70 per pip.

Yen pairs have their own version of this. For `USD/JPY`, one pip is `0.01`, and a standard lot is 100,000 dollars of base, but the pip value comes out in **yen**: `0.01 × 100,000 = ¥1,000` per pip. To get dollars, divide by the USD/JPY rate (because the rate is yen-per-dollar): at `150.00`, that is `1,000 ÷ 150`, or about \$6.67 per pip. This is why a yen-pair pip is worth a bit less than a EUR/USD pip in dollar terms, even though both are "one standard lot".

#### Worked example: pip value on a USD/JPY standard lot

You hold one standard lot of USD/JPY (100,000 dollars), and the pair moves from `150.00` to `150.50`. How much, in dollars, did you make or lose?

- Pip size for a yen pair is `0.01`. The move is `150.50 − 150.00 = 0.50`, i.e. `0.50 ÷ 0.01 = 50 pips`.
- Pip value in yen = `0.01 × 100,000 = ¥1,000` per pip.
- Profit in yen = `50 pips × ¥1,000 = ¥50,000`.
- Convert to dollars at the new rate `150.50`: `¥50,000 ÷ 150.50 = \$332`.

If you were **long** the dollar (you bought USD/JPY), a rise to `150.50` means the dollar strengthened against the yen, so you gained about \$332. The intuition: a yen-pair pip is worth roughly two-thirds of a EUR/USD pip in dollars right now, purely because you have to convert ¥1,000 back through a ~150 exchange rate.

For the rest of this post we will stay on dollar-quoted majors so the pip value lands directly in dollars and the arithmetic stays clean. Just remember the rule underneath: the formula always pays you in the quote currency, and you convert to dollars only when the quote currency is not already the dollar.

## Lot sizes and why position size is the real risk dial

Beginners obsess over picking the right direction. Professionals obsess over **size**, because size is what turns a correct view into a survivable one. The lot you choose sets your pip value, and your pip value sets how much a given move costs you. Two traders can have the identical view — "EUR/USD goes up" — and one blows up while the other barely notices the same 80-pip dip against them, purely because of lot size.

Make it concrete. Suppose you have a \$5,000 account and the market moves 80 pips against you before turning your way:

- On a **standard lot** (\$10/pip), 80 pips against you is `80 × \$10 = \$800` — that is 16% of your account gone on a single wiggle.
- On a **mini lot** (\$1/pip), the same 80 pips is `80 × \$1 = \$80` — 1.6% of the account. Annoying, survivable.
- On a **micro lot** (\$0.10/pip), it is `80 × \$0.10 = \$8` — a rounding error.

Same view, same market, same 80 pips. The only difference is the lot, and the lot is entirely your choice. This is why "how much should I size?" is a more important question than "which way will it go?" — getting the direction right does you no good if your size means a normal pullback wipes you out first.

#### Worked example: sizing a trade to a fixed dollar risk

You have a \$10,000 account and a rule that you will never risk more than 1% — that is \$100 — on a single trade. Your analysis says you should buy EUR/USD with a stop-loss 40 pips below your entry (if it falls 40 pips, you are wrong and you exit). What lot size keeps your loss at \$100 if the stop is hit?

- Risk per trade = \$100. Distance to stop = 40 pips. So your maximum acceptable pip value = `\$100 ÷ 40 pips = \$2.50 per pip`.
- A standard lot is \$10/pip; a mini lot is \$1/pip. You need \$2.50/pip, which is 2.5 mini lots (25,000 euros), since `2.5 × \$1 = \$2.50`.
- Check: if the trade hits your stop, `40 pips × \$2.50 = \$100` lost — exactly your 1% limit.

The intuition: you do not pick a lot size and hope; you work *backwards* from the dollars you are willing to lose and the distance to your stop, and the pip-value formula hands you the exact size. Size is solved, not guessed.

This is the deepest "so what" of the whole post: the quote tells you the price, but the **lot** tells you the stakes, and you control the lot completely. The numbers in the chart two figures up — \$10, \$1, ten cents, one cent per pip — are not trivia. They are the dial you turn to decide how much of your account rides on every pip.

### Notional, leverage, and why a small account can move a standard lot

A fair question at this point: how does a trader with a \$5,000 account control a standard lot worth \$108,530 of euros? The answer is **leverage**, and understanding it sharpens why pip value matters so much. When you open a standard lot, you do not hand over the full \$108,530 — your broker lets you control that **notional** size by posting a much smaller deposit called **margin**. At 50:1 leverage, the margin on a \$108,530 position is about `108,530 ÷ 50 = \$2,170`; at 100:1 it is about \$1,085. The rest is, in effect, borrowed for the duration of the trade.

The crucial point for reading a quote is this: **your profit and loss are computed on the full notional, not on the margin you posted.** That is why one pip on a standard lot is \$10 regardless of whether you put up \$2,170 or \$1,085 to open it. Leverage does not change the pip value — it changes how little of your own cash is tied up while the full pip value still hits your account. A 50-pip move against a standard lot is \$500 whether your margin was \$2,170 or \$1,085; against the smaller deposit, that \$500 loss is a far bigger fraction of what you committed.

This is the trap that turns sizing from a footnote into a survival skill. Leverage lets a small account take a large notional, which means a large pip value, which means an ordinary price wiggle can be a huge percentage of the account. The trader who thinks "I only put up \$1,085, so my risk is small" has it exactly backwards: the risk is set by the *notional* and the *pips*, and the small margin only disguises how exposed they are. The disciplined trader works the other way — pick the dollar risk first, derive the lot from the stop distance (as in the worked example above), and let the margin be whatever it turns out to be. Read the quote, then read the stakes; the stakes live in the notional, not the deposit.

## Why EUR/USD up means the dollar is weaker

Here is the single most common point of confusion for new FX traders, and once it clicks it never un-clicks. When you see a headline like "EUR/USD rises to 1.10," it is reporting two facts that are the *same fact*: the euro got stronger **and** the dollar got weaker. People stumble because they read "EUR/USD up" and think only about the euro, forgetting that the dollar is sitting right there on the other side of the slash, doing the opposite.

Reason it through from the definition. The price `EUR/USD = 1.10` means *one euro = 1.10 dollars*. If the price rises to `1.15`, then *one euro = 1.15 dollars* — each euro now commands more dollars. That is the textbook definition of the euro being stronger. But flip the sentence around: if one euro buys more dollars, then one dollar buys *fewer euros*. At `1.10`, a dollar bought `1 ÷ 1.10 = 0.909` euros; at `1.15`, a dollar buys only `1 ÷ 1.15 = 0.870` euros. The dollar's purchasing power, measured in euros, fell. **A rising base currency is, by arithmetic, a falling quote currency.** The before-and-after below shows exactly this trade-off.

![Before and after comparison showing a higher EUR USD rate means each euro buys more dollars and the dollar is weaker](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-3.png)

The rule generalises to every pair: **when a pair's price goes up, the base currency strengthened and the quote currency weakened.** So when `USD/JPY` rises from `145` to `150`, the dollar (the base) strengthened and the yen (the quote) weakened — a dollar now buys 150 yen instead of 145. When `GBP/USD` falls from `1.30` to `1.25`, the pound (base) weakened and the dollar (quote) strengthened. The direction always describes the **base**; the quote currency is automatically doing the opposite.

This is why the same dollar move can look "up" or "down" depending on which pair you watch. If the dollar broadly strengthens on a given day, `USD/JPY` goes **up** (dollar is the base, base rises) while `EUR/USD` goes **down** (dollar is the quote, so a stronger dollar pushes the pair the other way). Both charts are telling you "the dollar got stronger" — they just sit on opposite sides of the slash. Beginners who do not internalise this will swear the dollar did two contradictory things in one day. It did one thing; the *quoting convention* made it look like two.

#### Worked example: a strengthening dollar across two pairs

Suppose over one session the dollar strengthens against everything. EUR/USD falls from `1.0900` to `1.0850`, and USD/JPY rises from `150.00` to `150.75`. You hold one standard lot of each. What happens to your P&L if you were positioned *for* a stronger dollar — i.e. short EUR/USD and long USD/JPY?

- EUR/USD fell 50 pips (`1.0900 → 1.0850`). You were **short** (you sold the euro, betting the base would fall), so you profit: `50 pips × \$10 = \$500`.
- USD/JPY rose 75 pips (`150.00 → 150.75`). You were **long** (you bought the dollar as base), so you also profit. Pip value in yen = ¥1,000; profit = `75 × ¥1,000 = ¥75,000`; in dollars at `150.75`, that is `¥75,000 ÷ 150.75 ≈ \$497`.
- Total: about `\$500 + \$497 = \$997` from one coherent view — "the dollar gets stronger" — expressed correctly on both sides of the slash.

The intuition: a single macro view (stronger dollar) means **short** the pairs where the dollar is the quote currency and **long** the pairs where it is the base — and if you read the slash right, both legs pay you for the same idea. *Why* the dollar moves at all — rate differentials, flows, risk sentiment — is the subject of the macro layer; see [what moves exchange rates](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry). Here the point is narrower and prior to all of that: once the dollar moves, the quoting convention decides which way each chart points.

### Reading a quote with no dollar in it: cross-rates

So far every example has had the dollar on one side. But the market quotes plenty of pairs with no dollar at all — `EUR/GBP`, `EUR/JPY`, `GBP/JPY`, `AUD/NZD`, and so on. These are called **crosses** or **cross-rates**, and they read by exactly the same rules: the currency on the left is the base, the one on the right is the quote. `EUR/GBP = 0.8400` means *one euro = 0.84 pounds*; `EUR/JPY = 162.50` means *one euro = 162.50 yen*. Nothing about base, quote, bid, ask, pips, or direction changes when the dollar leaves the quote — a higher `EUR/GBP` still means the euro (base) strengthened and the pound (quote) weakened.

What *is* worth understanding is where the price of a cross comes from. Historically, almost no one held, say, euros and pounds directly against each other in size — they each traded against the dollar, the deepest market by far. So a cross-rate is built, by arithmetic, from the two dollar pairs. The euro trades against the dollar (`EUR/USD`) and the pound trades against the dollar (`GBP/USD`), and the cross `EUR/GBP` is simply the ratio:

```
EUR/GBP = EUR/USD  ÷  GBP/USD
```

The dollar in the numerator and the dollar in the denominator cancel, leaving euros per pound. This is not a metaphor; it is how dealers actually quote crosses, and it is why a cross's spread is usually *wider* than either dollar leg — you are effectively crossing two spreads (buying euros with dollars, then selling those dollars for pounds), so the costs stack. That is exactly why `EUR/GBP` sits a notch wider than `EUR/USD` on the spread chart earlier in this post.

#### Worked example: computing a cross-rate from two dollar quotes

The market shows `EUR/USD = 1.0850` and `GBP/USD = 1.2700`. What is `EUR/GBP`, and what does it mean?

- `EUR/GBP = EUR/USD ÷ GBP/USD = 1.0850 ÷ 1.2700 = 0.8543`.
- Reading it: one euro = 0.8543 pounds — or, flipping it, one pound = `1 ÷ 0.8543 = 1.1706` euros.
- Sanity check via dollars: one euro is worth \$1.0850, and one pound is worth \$1.2700. So a euro is worth `1.0850 ÷ 1.2700 = 0.8543` of a pound, since the pound is the more valuable of the two. Same number, reasoned through the common dollar yardstick.

The intuition: you never need a separate data feed for a cross — given the two dollar quotes, the cross is just their ratio, and the dollar you both divide by quietly cancels out. That is why the dollar is the hub of the entire quoting system even for pairs that do not name it.

This cross-rate arithmetic is also the seed of a deeper idea you will meet later in the series: **triangular arbitrage.** If a dealer's quoted `EUR/GBP` ever drifts away from the `EUR/USD ÷ GBP/USD` ratio by more than the spreads, a trader can loop euros → dollars → pounds → euros and lock in a riskless profit, which is precisely the force that keeps the three prices consistent. For now, the takeaway is just literacy: a cross is read like any pair and *priced* like a ratio of two dollar pairs.

## The spread as a cost: the toll you pay to get in

We have met the spread as the gap between bid and ask. Now let us treat it as what it really is to your account: **a guaranteed, upfront cost on every round trip.** You buy at the ask (high) and you sell at the bid (low), so the moment you open a trade you are already underwater by the spread, and the price has to move in your favour by at least the spread just to get you back to break-even.

![Before and after showing you buy at the ask and sell at the bid so the spread is a round trip cost the dealer keeps](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-6.png)

The figure makes the round trip concrete: open by buying at the ask `1.0853`, and you are instantly 2 pips down because you could only sell back at the bid `1.0851`. On a standard lot, 2 pips is `2 × \$10 = \$20`, and that \$20 goes to the dealer no matter what the price does next. The spread is not a fee charged separately — there is no line item — it is baked silently into the two prices you trade on. That is what makes it easy for beginners to ignore and expensive to ignore.

How big a deal is the spread? It depends entirely on which pair you trade. The most liquid majors — EUR/USD, USD/JPY — have spreads of a fraction of a pip in the interbank market, because so many participants are willing to trade them that dealers can quote razor-thin margins. The least liquid "exotic" pairs have spreads of dozens of pips, because few dealers will touch them and each one demands a fat margin for the risk. The chart below shows the range, and it is enormous — note the log scale.

![Horizontal bar chart of typical dealer spreads in pips by pair tier from EUR USD majors to exotic pairs on a log scale](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-2.png)

The spread on EUR/USD is about `0.2` pips in the interbank market; on USD/TRY (the Turkish lira) it is around `25` pips; on a USD/VND non-deliverable forward it can be `40` pips or more. That is a **200-fold** difference in the cost of admission between the tightest major and a thin exotic. (Retail brokers add their own markup on top of these interbank numbers, so what you actually pay is wider — but the *relative* picture, majors cheap and exotics expensive, holds exactly.)

#### Worked example: the spread cost of a scalping strategy

You have a strategy that opens and closes EUR/USD positions 20 times a day, one standard lot each, and your broker quotes a 1-pip spread (a realistic retail number for EUR/USD). What does the spread alone cost you per day, before you are right or wrong about a single trade?

- Cost per round trip = the spread = 1 pip = `1 × \$10 = \$10` per standard lot.
- Trades per day = 20, so daily spread cost = `20 × \$10 = \$200`.
- Over a 250-day trading year = `250 × \$200 = \$50,000`.

That is \$50,000 a year handed to the dealer purely for the privilege of crossing the spread 20 times a day — and that is on the *cheapest* pair on the board. Run the same strategy on a 25-pip exotic and the spread cost is 25 times larger. The intuition: the more often you trade and the wider the pair, the more the spread quietly eats you, which is exactly why high-frequency FX strategies live and die on EUR/USD-tight spreads and would be hopeless on exotics.

This is the spread's deep lesson and a thread that runs through the whole series: **the cost of trading a currency is inseparable from how liquid it is.** The thinner the market, the more you pay just to enter and exit. We will see this same liquidity gradient drive everything from where the big players trade to how violently exotic currencies move in a crisis. The plumbing that makes the majors so cheap to trade — the dealers, the interbank ladder, the electronic networks — is the subject of [inside the interbank FX market](/blog/trading/forex/the-biggest-market-on-earth-inside-the-interbank-fx-market).

### Spread is not the same as risk

One more thing the spread does *not* tell you: how much the pair will move. A tight spread means the pair is cheap to *trade*, not that it is calm. EUR/USD has the tightest spread on the board and still swings 7% (annualised) in a typical year. The measure of how far a pair tends to move is its **volatility**, not its spread, and the two are different things. The chart below shows implied volatility by pair — the market's estimate of how far each pair will travel.

![Horizontal bar chart of one month implied volatility by currency pair from EUR USD to USD TRY](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-7.png)

EUR/USD has a tiny spread *and* a modest ~7% implied vol; USD/TRY has a huge spread *and* a ~22% implied vol. They happen to line up here because thin, exotic currencies tend to be both expensive to trade and wild — but the two facts are logically separate. A pair can be cheap to trade yet jumpy, and you should never read a tight spread as a promise of a quiet ride. The deep mechanics of implied volatility — what it is, how it is priced — belong to the options world; see [reading the vol surface like a trader](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear). For reading a quote, the takeaway is narrow: the spread is your cost of entry, the volatility is your range of outcomes, and they are not the same number.

## Common misconceptions

Beginners arrive at FX with a few sticky wrong ideas, mostly imported from how stock prices feel. Here are the ones that cost real money, each corrected with a number.

**"EUR/USD up is good for me because the euro went up."** Not unless you are *long* the euro. "EUR/USD up" simply means the euro strengthened and the dollar weakened. If you are short EUR/USD (you sold the euro, betting it would fall), a rise is a loss. If EUR/USD goes from `1.0850` to `1.0950` and you are short one standard lot, you lost `100 pips × \$10 = \$1,000`. Direction is only "good" relative to which side of the pair you are on. The chart is neutral; your position decides whether up is profit or pain.

**"A pip is always the fourth decimal."** True for most pairs, false for yen pairs, where a pip is the second decimal (`0.01`). If you assume the fourth decimal on USD/JPY, you will misprice every move by a factor of 100 — a real 50-pip move (`150.00 → 150.50`) would look like 5,000 "pips" to you, and your dollar math would be wildly wrong. Always check whether the yen is in the pair before you count pips.

**"The spread is a small detail."** On a single trade, maybe. On a strategy, never. We just saw that a 1-pip spread, traded 20 times a day, costs \$50,000 a year on a standard lot. The spread is the most reliably negative number in your trading — you pay it every single time, win or lose — so it deserves more respect than the headline price, not less.

**"A tight spread means a safe, calm pair."** No. The spread is the cost to *trade*; volatility is how far the pair *moves*. EUR/USD has the tightest spread anywhere and still moves ~7% a year. A spread tells you about liquidity, not about risk. Conflating the two leads people to overtrade "cheap" pairs without respecting how far they can run against a position.

**"Bigger lots just mean bigger profits."** Bigger lots mean bigger *outcomes*, in both directions. The same 80-pip move is \$800 on a standard lot and \$8 on a micro lot — and that cuts both ways. Traders who size up to chase profit are, by identical arithmetic, sizing up their losses, which is why most blow-ups are sizing failures, not analysis failures. The pip-value dial does not care which direction the move goes.

**"High leverage means high risk."** Leverage and risk are related but not the same — and conflating them leads to bad decisions in both directions. Leverage is the *ratio* of notional to margin; risk is set by your *notional* and the *pips* the price moves. A trader at 100:1 leverage who opens a single micro lot (\$0.10/pip) is barely exposed, while a trader at 20:1 who opens five standard lots (\$50/pip) is taking enormous risk. The leverage figure on your account is a ceiling on how large a notional you *can* take, not a measure of how much you *are* risking. The number that matters is dollars-per-pip times the pips to your stop — that is your risk, and it is independent of the leverage label. Read the stakes from the notional, not from the leverage ratio your broker advertises.

## How it shows up in real markets

The rules become vivid the moment a real market moves. Here are three episodes where reading the quote correctly — base, pips, direction — was the whole game.

**The Swiss franc, 15 January 2015.** EUR/CHF had been pinned at the SNB's `1.20` floor for over three years. When the SNB walked away, the pair collapsed from `1.2010` toward `0.85` in minutes — a move of roughly 3,500 pips (at `0.0001` per pip). Read it through the rules: EUR/CHF *falling* means the base (euro) crashed against the quote (franc), i.e. the franc rocketed about 30% stronger. On a single standard lot of EUR/CHF, a 3,500-pip move is `3,500 × \$10 = \$35,000` — on one lot. Traders who were short the franc (long EUR/CHF, betting the floor would hold) and over-sized were obliterated; some retail brokers went bankrupt because clients' losses blew through their account balances. The lesson is not exotic: the same pip-value arithmetic from our \$500 example, applied to a 3,500-pip move on a leveraged lot, is a career-ending number.

**The yen, summer 2024.** USD/JPY climbed steadily through 2024, peaking near `161.9` on 3 July as the gap between US and Japanese interest rates stayed wide. Reading the quote: USD/JPY *up* means the dollar (base) strengthened and the yen (quote) weakened — the yen had fallen to multi-decade lows. Then, in early August, the Bank of Japan hiked and the move reversed violently: USD/JPY fell from about `161.9` to `141.7` by 5 August, roughly 2,000 pips, as a crowded "carry" trade unwound. On a standard lot, that 2,000-pip drop is `2,000 × \$10 = \$20,000` per lot — devastating for anyone long the dollar at the top, a windfall for anyone short. The full mechanics of *why* the yen was so weak and why the unwind was so sharp are a rate-and-flow story told in [carry trade unwinds](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks); the point here is that you cannot even *follow* that story without reading "USD/JPY up = weaker yen" instantly.

**The pound, 26 September 2022.** After the UK's "mini-budget" spooked markets, GBP/USD fell to an all-time intraday low around `1.035`. Read it: GBP/USD *down* means the pound (base) weakened against the dollar (quote) — sterling was in free fall. From its pre-crisis level near `1.125` to the `1.035` low is about 900 pips, or `900 × \$10 = \$9,000` per standard lot for anyone holding the pound. A trader who could read the quote saw, in real time, the pound losing ground pip by pip; a trader who could not was staring at a number falling for no reason they understood.

**The Vietnamese dong, a slow-motion version of the same reading.** Not every currency moves in dramatic minutes — some grind. `USD/VND` (the dollar is the base, the dong the quote) sat near `21,340` at the end of 2014 and drifted to about `25,450` by the end of 2024. Read it with the rules: USD/VND *rising* means the dollar (base) strengthened against the dong (quote) — the dong weakened, year after year, in a managed crawl rather than a free float. The magnitude is enormous in absolute terms — over 4,000 "points" — but the *meaning* is identical to the pound's crash: the base went up, so the quote currency lost value. The pips look different (a VND pip is a whole point of dong, since the price has no decimals) and the speed is glacial, yet the literacy is exactly the same: read the base, read the direction, and you know which money is losing ground. The dong's managed crawl, and why an emerging-market central bank steers a currency this way, is a story for the Vietnam track of this series; here it simply proves that the same five-part reading works on a quote that moves over a decade as well as one that moves in three minutes.

#### Worked example: same news, two pairs, opposite charts

The dollar strengthens broadly on a hot US inflation print. You watch two screens: EUR/USD and USD/JPY. EUR/USD drops 60 pips (`1.0850 → 1.0790`); USD/JPY jumps 90 pips (`150.00 → 150.90`). A beginner panics: "Did the dollar go up or down? One chart fell and one rose!"

- EUR/USD fell: the dollar is the **quote** currency here, so a stronger dollar pushes the pair *down*. `60 pips × \$10 = \$600` of move per standard lot.
- USD/JPY rose: the dollar is the **base** currency here, so a stronger dollar pushes the pair *up*. `90 pips × ¥1,000 = ¥90,000`, or about `¥90,000 ÷ 150.90 ≈ \$596` per standard lot.
- Both charts are saying the identical thing — *the dollar strengthened* — they just sit on opposite sides of the slash.

The intuition: when you know which side the dollar is on, two "contradictory" charts collapse into one clean fact, and you stop being fooled by the quoting convention. This is also why the dollar is on roughly 88% of all FX trades and is the master currency to read — a fact worth seeing directly.

![Bar chart of the share of FX trades with each currency on one side showing the US dollar on 88.5 percent](/imgs/blogs/base-quote-pips-and-how-to-read-an-fx-quote-8.png)

The dollar appears on **88.5%** of all FX trades, according to the BIS Triennial Survey — far more than the euro's 30.5% or the yen's 16.7%. (The shares sum to 200% because every trade has two currencies, and each side counts once.) The practical consequence for a beginner: the single most valuable reading skill you can build is fluency in **dollar-quoted pairs**, because the dollar is on one side of nearly everything you will ever trade. Master "is the dollar the base or the quote here, and which way did it just move?" and you can read most of the market. The reason the dollar sits at the centre of the entire system — why it is on the other side of almost every trade — is the dollar-dominance story in [the dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).

## The takeaway: read the quote, then read the stakes

Strip away the vocabulary and an FX quote is one of the simplest objects in finance: a price of one money in another, shown twice (a sell price and a buy price), measured in standardised steps (pips) and standardised sizes (lots). If you can do four things, you can read any quote in the world:

1. **Name the base and the quote.** Left of the slash is the base, the thing being priced; right is the quote, the unit of the price. `EUR/USD = 1.0853` means one euro costs 1.0853 dollars. The price always describes one unit of the base.
2. **Place the bid, the ask, and the spread.** Bid below (you sell there), ask above (you buy there), spread in between (the dealer's cut, and your guaranteed cost on every round trip).
3. **Count pips and convert to dollars.** A pip is the 4th decimal (`0.0001`) for most pairs, the 2nd (`0.01`) for yen pairs. Pip value = pip size × lot size. On a standard lot of a dollar-quoted major, one pip is \$10, so a 50-pip move is \$500. Multiply pips by pip value and you have your P&L.
4. **Read the direction off the base.** Pair up means the base strengthened and the quote weakened. EUR/USD up = stronger euro, weaker dollar. USD/JPY up = stronger dollar, weaker yen. The quote currency always does the opposite.

But the most important habit is the one that separates traders who last from those who do not: **read the quote, then read the stakes.** The quote gives you the price; the *lot* gives you how much money rides on every pip — and the lot is entirely your choice. The exact same 80-pip move is \$800 on a standard lot and \$8 on a micro lot. Picking the right direction means nothing if your size means a normal pullback bankrupts you first. So train yourself to look at a quote and immediately think in two layers: *what is the price doing*, and *how much is one pip worth to me right now*. The first is reading; the second is risk. Both live inside the same little string of numbers on the screen.

There is a deeper point hiding in this literacy, and it is the spine of the whole series. A quote is not a property of one currency — it is a *relationship* between two. The price `EUR/USD = 1.0853` is not "what the euro is worth" in any absolute sense; it is what the euro is worth *measured in dollars right now*, and that same euro reads as `0.85` against the franc, `162` against the yen, `0.92` against the pound. There is no single number that is "the value of the euro", only a web of relative prices, each one a pair. When you read a quote, you are reading one edge of that web. That is why every FX position is a relative bet, why you can never own a currency in isolation, and why the question that drives the whole series is never "what is this currency worth?" but "what is it worth *against what*, and why is that relationship moving?"

That is the literacy the rest of this series runs on. Every carry trade, every intervention, every crisis unwind is just this quote, moving — base against quote, pip by pip, lot by lot. Once you can read the quote, you can read the market.

## Further reading & cross-links

Within this series:

- [Why currencies are different: an introduction to FX trading](/blog/trading/forex/why-currencies-are-different-fx-trading-introduction) — the relative-price thesis: you never own a currency, only a pair. The conceptual ground this post stands on.
- [Majors, minors, and exotics: the map of currency pairs](/blog/trading/forex/majors-minors-and-exotics-the-map-of-currency-pairs) — which pairs exist, how they are grouped, and why the spread and volatility differ so much across tiers.
- [The biggest market on Earth: inside the interbank FX market](/blog/trading/forex/the-biggest-market-on-earth-inside-the-interbank-fx-market) — the dealers, the ladder, and the plumbing that makes the majors so cheap to trade.

Going deeper into what moves the quote:

- [What moves exchange rates: rates, flows, carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — once you can read the quote, this is *why* it moves.
- [The dollar system: why the USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — why the dollar sits on one side of nearly every quote you will read.
- [Carry trade unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the yen episode from this post, told as a full rate-and-leverage story.
- [Reading the vol surface like a trader](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — how the market prices the *range* of a pair, the volatility that a tight spread does not capture.

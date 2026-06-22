---
title: "Valuing a Bank: Price-to-Book, ROE, and the Warranted Multiple"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why banks are priced on book value instead of earnings, how the warranted multiple (ROE − g) / (COE − g) ties price to profitability, and how to tell a genuine bargain from a value trap."
tags: ["banking", "bank-valuation", "price-to-book", "return-on-equity", "cost-of-equity", "warranted-multiple", "residual-income", "dividend-discount-model", "tangible-book-value", "value-trap", "equity-research"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank is worth a multiple of its *book value*, and that multiple is set almost entirely by one comparison: how much the bank earns on its equity (ROE) versus how much its owners demand (the cost of equity, COE).
>
> - Banks are valued on **price-to-book**, not just price-to-earnings, because their assets and liabilities are mostly financial instruments carried near market or recoverable value — so book value is a meaningful starting point, and earnings are too cyclical to anchor on.
> - The **warranted price-to-book** multiple is roughly **(ROE − g) / (COE − g)**. When ROE equals COE the bank is worth exactly **1.0× book**; above it, a premium; below it, a discount.
> - A **\$12** ROE on a **10%** cost of equity (with 3% growth) justifies about **1.29×** book. The same arithmetic, run through residual income or a dividend discount model, gives the *identical* answer — they are three views of one truth.
> - A **low P/B is only cheap if the low ROE is temporary**. If a bank earns less than its cost of equity *forever*, a sub-1× multiple is not a bargain — it is the market correctly pricing value destruction. That is the difference between a deep-value buy and a **value trap**.
> - The one number to remember: a bank earning its cost of equity is worth **1× book**, and every point of ROE above or below that line moves the warranted multiple in a straight, predictable way.

In the first week of March 2023, two banks told you everything you need to know about how banks are valued.

JPMorgan Chase, the largest bank in the United States, was trading at roughly 1.5 times its book value. SVB Financial Group — the parent of Silicon Valley Bank, the 16th-largest bank in the country — was trading at well under one times book, and falling. Within days SVB would be seized by regulators, and its equity would be worth zero. JPMorgan would go on to *absorb* a failed competitor and post record profits. Two banks, two wildly different multiples of the same accounting concept — book value — and the gap between them was not an accident. It was the market pricing the single thing that determines what a bank is worth: whether the bank earns more on its shareholders' money than those shareholders could earn elsewhere.

That comparison has a name and a formula, and once you understand it, bank valuation stops being a black box. You stop asking "is a 12 price-to-earnings cheap?" and start asking "does this bank earn its cost of equity, and will it keep doing so?" Everything else — the residual-income model, the dividend discount model, the tangible-book adjustment, the value-trap test — is just a different lens on that one question.

The diagram below is the mental model for the whole post. Three inputs — how well a bank earns (ROE), what its owners demand (the cost of equity), and how fast it grows — collapse into a single warranted multiple of book value. Hold that picture in your head; everything that follows is an unpacking of it.

![Warranted price-to-book identity from ROE cost of equity and growth](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-1.png)

This post is educational, not investment advice. It explains the mechanics of how banks are valued and where the method breaks; it does not tell you to buy or sell anything.

## Foundations: book value, ROE, the cost of equity, and the warranted multiple

Before we can value a bank we need a shared vocabulary. Each of these terms gets defined from zero, because the whole valuation framework is just these pieces snapped together.

### Book value: what the owners actually own

A bank, like any company, has a balance sheet: **assets** on one side (the loans it has made, the bonds it owns, the cash in its vault) and **liabilities** on the other (the deposits it owes you, the bonds it has issued, the money it has borrowed). The two sides must balance. Whatever is left over after you subtract liabilities from assets belongs to the shareholders. That leftover is **equity**, also called **book value** or **net worth**.

If a bank has \$100 billion of assets and \$92 billion of liabilities, its book value is \$8 billion. That \$8 billion is the accounting measure of what the owners own. Divide it by the number of shares and you get **book value per share** — the per-share slice of the bank that, on paper, belongs to each shareholder.

Here is why book value matters *so much more* for a bank than for, say, a software company or a coffee chain. A coffee chain's real value sits in things the balance sheet barely captures: its brand, its real estate leases, its customer loyalty. Its book value might be \$2 billion while the market values it at \$40 billion, and nobody blinks, because the book number is almost meaningless. A bank is different. A bank's assets are overwhelmingly **financial** — loans, securities, cash — things that are already carried on the books at something close to what they are worth or what they will collect. A bank's liabilities are also financial — deposits and debt, carried at what is owed. So for a bank, book value is not a footnote; it is a real, defensible estimate of the net worth being valued. That single fact is why bank valuation starts with book value, not with earnings.

### Tangible book value: stripping out the fairy dust

There is one adjustment almost every bank analyst makes immediately. Some of a bank's "assets" are not financial at all — they are **goodwill** and other **intangibles**, the accounting residue left over when the bank paid more than book value to acquire another bank. Goodwill cannot absorb a loss, cannot be sold to a depositor demanding cash, and is the first thing regulators deduct when they compute a bank's real capital.

So analysts compute **tangible book value** — book value minus goodwill and intangibles — and the ratio they actually watch is **price-to-tangible-book (P/TBV)**. If a bank reports \$10 billion of book equity but \$2 billion of that is goodwill from past deals, its tangible book is \$8 billion. A bank trading at \$12 billion of market value is at 1.2× book but 1.5× tangible book. Acquisitive banks that have rolled up rivals can carry a lot of goodwill; for them the tangible number is the honest one. We will mostly say "book value" for simplicity, but in practice the tangible version is the one the market prices.

### Price-to-book: the multiple itself

**Price-to-book (P/B)** is simply the bank's market value divided by its book value — what the market will pay for each dollar of the owners' accounting net worth. A P/B of 1.0 means the market values the bank at exactly its book equity. A P/B of 1.5 means investors will pay \$1.50 for each \$1 of book. A P/B of 0.6 means they will only pay 60 cents.

The deep question — the one this entire post answers — is: **what determines whether a bank deserves a P/B above or below 1.0?** The intuitive answer is almost embarrassingly simple. If the bank can take a dollar of your equity and turn it into a stream of profits worth *more* than a dollar, you will pay more than a dollar for it. If it turns your dollar into a stream worth *less* than a dollar, you will pay less. The whole game is whether the bank's profitability beats the return you demand.

### Return on equity: how well the bank earns

**Return on equity (ROE)** is the bank's annual net profit divided by its book equity. If a bank earns \$1 billion on \$8 billion of equity, its ROE is 12.5%. ROE answers: *for every dollar of the owners' money the bank holds, how many cents of profit does it generate per year?*

ROE is the engine of bank valuation. We covered its mechanics in depth in [ROE, ROA, and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) — the short version is that a bank earns a razor-thin return on its *assets* (around 1%) and levers it up about twelvefold into a respectable return on *equity* (around 10–12%). For valuation, ROE is the number that matters, because it tells you how productively the bank uses the very thing you are buying: its book equity.

### Cost of equity: what the owners demand

The **cost of equity (COE)** is the annual return shareholders require to hold the stock, given its risk. It is not a cost the bank pays in cash — it is an *opportunity cost*, the return investors could get on an equally risky alternative. A bank stock is leveraged and exposed to credit and rate risk, so investors typically demand something like **9–11%** to own one. Throughout this post we will use a round **10%** as the base-case cost of equity, which is squarely in the historical range for large banks.

The cost of equity is usually estimated with the **capital asset pricing model (CAPM)**: COE = risk-free rate + beta × equity risk premium. If the risk-free Treasury yield is 4%, the equity risk premium is 5%, and the bank's beta (its sensitivity to the market) is 1.2, then COE = 4% + 1.2 × 5% = 10%. The exact estimate is debatable — beta and the risk premium are slippery — but the *concept* is rock-solid: shareholders have a hurdle rate, and a bank either clears it or it does not. Equity-research practitioners build this up carefully; if you want the full machinery, the sibling post [building a DCF, part 2: cost of capital](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) walks through CAPM in detail.

### Excess return and the warranted multiple

Now we can state the central idea. The bank takes your equity and earns **ROE** on it. You demanded **COE**. The difference — **ROE − COE** — is the **excess return**, sometimes called the *spread* over the cost of equity. This single quantity decides everything:

- If **ROE > COE**, the bank earns more than you demanded. Every dollar of equity is worth more than a dollar. The bank deserves a **premium to book** (P/B > 1).
- If **ROE = COE**, the bank earns exactly what you demanded. A dollar of equity is worth exactly a dollar. The bank is worth **1.0× book**.
- If **ROE < COE**, the bank earns less than you demanded. Every dollar of equity is worth *less* than a dollar. The bank deserves a **discount to book** (P/B < 1).

When you turn this intuition into algebra (we will derive it in a moment), you get the **warranted price-to-book** multiple — the P/B a bank *should* trade at given its fundamentals:

$$\text{Warranted P/B} = \frac{\text{ROE} - g}{\text{COE} - g}$$

where $g$ is the bank's sustainable growth rate of book value. Notice the structure: subtract growth from both the numerator and the denominator, then divide. If ROE = COE, the numerator and denominator are equal and the ratio is exactly 1, regardless of growth. Above COE, the ratio exceeds 1; below, it falls under 1. The formula is just the intuition, made precise.

### Value trap: when cheap is correctly cheap

Finally, the term that separates the analysts from the bargain-hunters. A **value trap** is a stock that *looks* cheap on a multiple — say a bank at 0.6× book — but is cheap for a good reason: its profitability is permanently impaired, so the low multiple is *deserved*, and it will not recover. The opposite of a value trap is a genuine bargain: a stock that is cheap because the market is temporarily too pessimistic, and whose true warranted multiple is higher than where it trades. Telling the two apart is the hardest and most valuable judgment in bank valuation, and it comes down to one question we will return to at the end: *is the low ROE temporary or permanent?*

With the vocabulary in place, let us build the framework piece by piece.

## Why banks are valued on book value, not just earnings

If you have ever read a stock-picking book, you learned to value companies on **price-to-earnings (P/E)** — the price you pay for each dollar of annual profit. For most companies that is the right starting point. For banks, it is a trap, and understanding *why* is the foundation for everything else.

### Three reasons book value works for banks

**First, a bank's book value is real.** As we said in the foundations, a bank's balance sheet is almost entirely financial instruments carried at or near their realizable value. A manufacturer's book value is distorted by decades-old factories depreciated to a fraction of their worth and brands worth billions that appear nowhere. A bank's book value is loans (carried at amortized cost net of expected losses), securities (often marked to market), and deposits (carried at face). It is a defensible estimate of net worth. When book value is meaningful, you can value the company as a multiple of it.

**Second, a bank's equity is the regulated, binding constraint on its entire business.** A bank cannot simply decide to grow. Every dollar of risk-weighted assets it holds must be backed by a regulator-mandated slice of equity capital — this is the heart of [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion). Book equity is therefore not just an accounting figure; it is the *fuel tank* that determines how big the bank can be. The amount of equity literally caps the size of the business, which makes "how much do we pay per dollar of equity" the natural valuation question.

**Third, bank earnings are violently cyclical, and P/E lies at exactly the wrong moments.** This is the subtle one. A bank's profit is net of **loan-loss provisions** — money set aside for loans that will go bad. In a boom, defaults are low, provisions are tiny, and earnings look fat: the P/E looks *low* (cheap) right at the top of the cycle. In a bust, provisions explode, earnings collapse or go negative, and the P/E spikes to absurd levels or becomes meaningless: the stock looks *expensive* right at the bottom, exactly when it is cheap. P/E is pro-cyclical in the most dangerous way. Book value, by contrast, erodes far more gradually — it absorbs losses through retained earnings rather than swinging on a single year's provision — so P/B is a steadier yardstick across the cycle. (The mechanics of why earnings whipsaw like this are the subject of the credit-cycle post in this series.)

#### Worked example: why P/E misleads at the cycle's turn

Take a bank with \$8 billion of equity. In a normal year it earns \$1 billion — an ROE of 12.5% and, if the market values it at \$12 billion, a P/E of 12 and a P/B of 1.5.

Now a recession hits. The bank must provision \$1.5 billion against loans it expects to sour. Its pre-provision profit was still \$1 billion, but after the provision it posts a *loss* of \$0.5 billion. Its P/E is now negative — undefined, useless. Has the bank's franchise been destroyed? No. Its book value fell from \$8 billion to about \$7.5 billion (the loss eroded equity), so on book value it dropped only ~6%. If the market believes the bank will earn ~12% again once the cycle turns, the right yardstick is book value, which barely moved, not earnings, which fell off a cliff. **The lesson: a bank's earnings tell you where you are in the credit cycle; its book value and through-cycle ROE tell you what the franchise is worth.**

This is exactly why the financial industry standardized on P/B for banks. It is not tradition — it is that book value is the one stable, meaningful, regulated anchor a bank offers. P/E remains a useful *cross-check* (we will put it in a comparison matrix shortly), but it is the co-pilot, not the captain.

## The warranted multiple: deriving (ROE − g) / (COE − g)

Now we earn the central formula. The derivation is short and worth seeing once, because it makes clear *why* ROE and COE are the only things that matter.

### Building it from a single retained dollar

Start with a bank that has book value $B$ and earns a steady ROE on it. Each year it generates earnings of $\text{ROE} \times B$. It pays out a fraction of those earnings as dividends and retains the rest to grow. The retained portion is what lets book value compound. If the bank retains a fraction $b$ (the *retention ratio*) of its earnings, then book value grows at:

$$g = \text{ROE} \times b$$

This is the **sustainable growth rate** — a bank can only grow its book value by plowing back profits, and it grows faster the more it retains and the higher its ROE. The payout ratio is $1 - b$, so dividends per year start at $D_1 = \text{ROE} \times B \times (1 - b)$.

A share of stock is worth the present value of all the dividends it will ever pay, discounted at the cost of equity. For a dividend stream growing forever at rate $g$, the **Gordon growth** formula gives the value:

$$V = \frac{D_1}{\text{COE} - g}$$

Substitute $D_1 = \text{ROE} \times B \times (1-b)$ and use $g = \text{ROE} \times b$, so $b = g / \text{ROE}$ and $(1-b) = (\text{ROE} - g)/\text{ROE}$. Then:

$$V = \frac{\text{ROE} \times B \times \frac{\text{ROE}-g}{\text{ROE}}}{\text{COE} - g} = \frac{B \,(\text{ROE} - g)}{\text{COE} - g}$$

Divide both sides by book value $B$, and the ROE terms collapse into the warranted price-to-book:

$$\frac{V}{B} = \frac{\text{ROE} - g}{\text{COE} - g}$$

Every symbol earns its place: ROE (how well the bank earns), COE (what owners demand), and $g$ (how fast book compounds). Nothing else. The bank's size, its name, its sector all wash out — what is left is pure profitability versus required return.

![Warranted price-to-book rises with ROE and crosses one when ROE equals cost of equity](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-2.png)

The chart above plots the formula directly, holding COE at 10% and growth at 3%. It is almost a straight line, and the single most important point on it is where it crosses **1.0× book** — exactly at ROE = 10% = COE. Below that ROE the bank trades at a discount (the red zone); above it, a premium (the green zone). The marked point at 12% ROE lands at 1.29×, which is our running worked example.

#### Worked example: warranted P/B from a 12% ROE

Take a bank that sustainably earns **ROE = 12%**, whose shareholders demand **COE = 10%**, and that grows book value at **g = 3%** a year. Plug into the formula:

$$\text{Warranted P/B} = \frac{0.12 - 0.03}{0.10 - 0.03} = \frac{0.09}{0.07} = 1.286$$

So this bank is worth about **1.29 times** its book value. If its book value per share is \$50, the warranted price is about \$50 × 1.29 = **\$64.30** per share. The 2-percentage-point excess return (12% ROE versus 10% COE) translates into a 29% premium over book. **The intuition: a bank earning a modest 2 points above its cost of equity is worth roughly 30% more than its accounting net worth — small edges in ROE compound into meaningful premiums.**

Notice how sensitive this is. If the same bank's ROE were 14% instead of 12%, the warranted multiple would be (0.14 − 0.03)/(0.10 − 0.03) = 1.57×, a 57% premium. Two extra points of ROE nearly *doubled* the premium. This is why bank analysts obsess over a percentage point of ROE — it moves the warranted multiple far more than it moves the earnings.

There is a second, subtler lever hiding in the formula: growth. Growth is a *double-edged* input, and beginners almost always get its sign wrong. Intuitively you might think faster growth always lifts the multiple — more is better. But look at the algebra. Growth ($g$) appears in *both* the numerator and the denominator, subtracted from each. When ROE is *above* COE, raising growth helps: you are compounding a positive excess return, so the multiple rises. When ROE is *below* COE, raising growth *hurts*: you are compounding value destruction, so faster growth makes the bank *worth less*, not more. A bank earning 7% on a 10% cost of equity should *shrink*, not grow — every dollar it retains and reinvests at 7% is worth less than the dollar it could have paid out. This is why a struggling bank that announces an aggressive growth plan often sees its stock *fall*: the market understands it is about to compound a sub-hurdle return. Growth only creates value when it is profitable growth — ROE above COE. Below that line, growth is the enemy.

#### Worked example: how growth cuts both ways

Take two banks, both growing book at **5%** a year, both facing a **10%** cost of equity. Bank A earns a **13%** ROE; Bank B earns a **7%** ROE. Run the formula:

$$\text{Bank A: } \frac{0.13 - 0.05}{0.10 - 0.05} = \frac{0.08}{0.05} = 1.60\times \qquad \text{Bank B: } \frac{0.07 - 0.05}{0.10 - 0.05} = \frac{0.02}{0.05} = 0.40\times$$

Now slow both banks' growth to **2%** and recompute. Bank A drops to (0.13 − 0.02)/(0.10 − 0.02) = 1.38×, *lower* — slowing a value-creating bank costs it some premium. But Bank B *rises* to (0.07 − 0.02)/(0.10 − 0.02) = 0.63×, *higher* — slowing a value-destroying bank is good news, because it stops compounding losses. **The intuition: growth multiplies whatever excess return you already earn. For a good bank it amplifies the premium; for a bad bank it deepens the discount, so the cheapest thing a sub-hurdle bank can do is stop growing and hand the capital back.**

### The break-even that anchors everything

The most useful single fact in bank valuation: **when ROE equals COE, the warranted P/B is exactly 1.0, no matter what growth is.** Set ROE = COE in the formula and the numerator and denominator are identical; they cancel to 1. This is the pivot point. A bank earning exactly its cost of equity creates no value and destroys none — each retained dollar is worth precisely one dollar — so the market pays book and nothing more. Every bank you look at, you should first ask: is its ROE above or below ~10%? That one comparison tells you, instantly, whether it should trade above or below book.

The break-even line divides the world of banks into two regimes, and the diagram below contrasts them directly. On the left, a bank earning a 7% ROE against a 10% cost of equity falls three points short: each retained dollar is worth less than a dollar, so the bank deserves a discount — a warranted P/B of about 0.6×. On the right, a bank earning a 12% ROE beats the same 10% hurdle by two points: each retained dollar is worth more than a dollar, so it earns a premium of about 1.3×. Same accounting book value, two opposite verdicts, and the only thing that changed was whether ROE cleared the cost-of-equity bar.

![Premium to book when ROE beats cost of equity versus discount when it falls short](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-4.png)

What makes this picture so powerful is that the two columns are not different *kinds* of analysis — they are the *same* formula evaluated on opposite sides of a single line. There is no separate "premium model" and "discount model." There is one relationship, (ROE − g) / (COE − g), and the cost-of-equity hurdle is the fulcrum it pivots on. A reader who internalizes only this diagram can already classify most banks on sight: find the ROE, compare it to a ~10% hurdle, and you know which side of book the stock belongs on before you have opened a single financial statement.

## The residual-income model: where the value above book comes from

The warranted-multiple formula gives you a number, but it can feel like a black box. The **residual-income model** (also called the *excess-return* or *economic-profit* model) gives you the same answer while showing you exactly *where* the value above book comes from. It is the most intuitive way to value a bank, and it makes the value-trap logic crystal clear.

### The idea: charge the bank rent on its equity

Here is the move. Accounting earnings flatter the bank because they treat equity as free. But equity is not free — shareholders demand the cost of equity on it. So we charge the bank "rent" on the equity it uses. The rent is **COE × book value**. Whatever the bank earns *above* that rent is genuine value creation; whatever it earns *below* it is value destruction. That excess is the **residual income** (RI):

$$\text{RI} = \text{Net income} - (\text{COE} \times \text{Book value}) = (\text{ROE} - \text{COE}) \times \text{Book value}$$

Read the right-hand side slowly: residual income is the excess return (ROE − COE) times the equity it is earned on. If ROE beats COE, RI is positive — the bank prints economic profit. If ROE trails COE, RI is negative — the bank is destroying value even while it reports positive accounting earnings.

The bank's intrinsic value is then its book value *plus* the present value of all future residual income:

$$V = B_0 + \sum_{t=1}^{\infty} \frac{\text{RI}_t}{(1 + \text{COE})^t}$$

In words: **you are worth your accounting net worth, plus a bonus for every future year you earn above your cost of equity** (or a penalty for every year you earn below it). This is the cleanest statement of bank valuation there is.

![Residual income builds intrinsic value on top of book value](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-7.png)

The waterfall above shows it for our running bank. Start with \$100 of book value (the blue bar). Layer on the present value of all the future profit it earns *above* its cost of equity (the green bar, about \$29). The total — about \$129 — is the intrinsic value, which is exactly 1.29× book. The premium to book is *nothing more than* the capitalized value of excess returns. No excess return, no premium.

#### Worked example: a residual-income valuation

Our bank has **\$100** of book value, earns **ROE = 12%**, faces **COE = 10%**, and grows book at **g = 3%**.

Year-one net income is 12% × \$100 = \$12. The equity rent is 10% × \$100 = \$10. So year-one residual income is:

$$\text{RI}_1 = \$12 - \$10 = \$2 \quad\text{(equivalently } (0.12 - 0.10) \times \$100 = \$2)$$

Because book value grows at 3%, next year's equity is larger, so residual income also grows at about 3%. A growing perpetuity of residual income, discounted at the 10% cost of equity, is worth:

$$\text{PV of RI} = \frac{\text{RI}_1}{\text{COE} - g} = \frac{\$2}{0.10 - 0.03} = \frac{\$2}{0.07} = \$28.6$$

Add it to book value:

$$V = \$100 + \$28.6 = \$128.6 \quad\Rightarrow\quad \text{P/B} = 1.29\times$$

The residual-income model lands on **exactly** the same 1.29× the warranted-multiple formula gave — as it must, since both are algebraic rearrangements of the same dividend stream. **The intuition: a bank's premium to book is the present value of its excess returns; strip the excess returns away and the bank is worth book, full stop.**

This is also why a *negative* excess return is so dangerous. If RI is negative, the present value of future residual income is *subtracted* from book, and the bank is worth *less* than book. The residual-income model is the formal machinery behind "a bank that earns below its cost of equity deserves to trade below book" — a point we will hammer in the value-trap section.

## The dividend discount model for a capital-constrained bank

The third lens is the **dividend discount model (DDM)**, and for banks it carries a special twist that makes it both essential and treacherous.

### Banks return value primarily through dividends and buybacks

A mature bank cannot endlessly reinvest its earnings, because — as we noted — its growth is capped by how much capital regulators let it deploy. A well-run bank that does not need all its capital to grow returns the surplus to shareholders, as dividends and buybacks. That makes the DDM a natural fit: the value of the stock is the present value of everything it ever returns to you.

For a stable, growing bank, the DDM is the Gordon growth formula we used in the derivation:

$$V = \frac{D_1}{\text{COE} - g}$$

where $D_1$ is next year's dividend per share, COE is the cost of equity, and $g$ is the sustainable growth rate. The payout that funds those dividends is constrained: $g = \text{ROE} \times (1 - \text{payout ratio})$, so a bank that wants to grow faster must pay out less. This is the capital constraint showing up directly in the valuation — a bank cannot have a high payout *and* high growth unless its ROE is high enough to fund both.

#### Worked example: the DDM for a capital-constrained bank

Same bank: **\$100** book, **ROE = 12%**, **COE = 10%**, target **g = 3%**.

To grow book at 3% with a 12% ROE, the bank must retain $b = g / \text{ROE} = 3\% / 12\% = 0.25$, i.e. retain 25% of earnings and pay out the other **75%**. (This is the capital constraint: the faster it wants to grow, the less it can pay out.)

Year-one earnings are 12% × \$100 = \$12. With a 75% payout, the dividend is:

$$D_1 = \$12 \times 0.75 = \$9$$

The intrinsic value via Gordon growth:

$$V = \frac{D_1}{\text{COE} - g} = \frac{\$9}{0.10 - 0.03} = \frac{\$9}{0.07} = \$128.6 \quad\Rightarrow\quad \text{P/B} = 1.29\times$$

Once again, **1.29×** — the third method, the third identical answer. The warranted multiple, residual income, and the DDM are the same equation wearing three different coats. **The intuition: a bank's value is the present value of what it can return to you, and what it can return is governed by its ROE and how much growth it must fund — the same two forces that set the warranted multiple.**

### Why the DDM is treacherous for banks

The DDM has a sharp edge: a bank's dividend is not freely chosen. Regulators run **stress tests** and can block a bank from paying dividends or buying back stock if its capital is too thin. In 2020, the Federal Reserve capped large-bank dividends and suspended buybacks entirely during the pandemic, because it wanted banks to hoard capital against an uncertain credit shock. A DDM that naively extrapolates last year's dividend would have been wildly wrong, because the dividend was about to be administratively cut to preserve capital. **The capital constraint is not a footnote in a bank DDM — it is the binding force.** This is why residual income (which keys off ROE and book, not the discretionary dividend) is often the cleaner model for banks: it sidesteps the question of *when* value is returned and focuses on *whether* it is created.

## How ROE drives the multiple in the real data

We have been working with a stylized 12% ROE. Where does the real industry sit, and how stable is the number that drives every valuation? The answer is reassuringly steady, with two violent exceptions that prove the rule.

![US bank return on equity from 2010 to 2024](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-3.png)

The chart traces the aggregate return on equity for all FDIC-insured US banks from 2010 to 2024. After the 2008 crisis the industry crawled out of a 5.85% ROE in 2010 — far below any reasonable cost of equity, which is exactly why bank stocks traded *below* book for years afterward. By 2018–2019 the industry had climbed to a healthy 11.4–12.0%, comfortably above a ~10% cost of equity, and bank P/B multiples expanded accordingly. Then 2020 hit: pandemic loan-loss provisions slammed ROE down to **6.65%**, briefly pushing the industry's earnings below its cost of equity — and bank stocks sold off hard. As those provisions proved too conservative and were released, ROE snapped back to 12.1% in 2021 and has settled around **10.3–10.4%** in 2023–2024.

That settling point is the whole ballgame. An industry ROE hovering right around 10% — its approximate cost of equity — implies an *average* bank should trade at roughly **1.0× book**. And that is broadly what you see: the typical US bank trades near book, the best franchises trade at a premium, and the weakest trade at a discount. The dispersion *around* that average is where the analysis happens.

#### Worked example: from ROE to a sector P/B

Suppose the aggregate banking sector earns a through-cycle ROE of **10.3%** (the 2024 figure), against an industry cost of equity of **10%**, growing at **3%**. The warranted multiple for the *average* bank is:

$$\text{Warranted P/B} = \frac{0.103 - 0.03}{0.10 - 0.03} = \frac{0.073}{0.07} = 1.04\times$$

So the average bank is worth a whisker above book — about **1.04×**. That is the gravitational center the whole sector orbits. **The intuition: with the industry earning almost exactly its cost of equity, "trades around book" is not a bargain or a warning — it is fair value, and your job is to find the banks meaningfully above or below that line for a reason.**

### Banks line up by ROE, not by book

If the warranted-multiple formula is right, then across a set of banks, P/B should rise more or less linearly with sustainable ROE. That is exactly the pattern you observe in the market.

![Price to book versus return on equity for stylized banks](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-5.png)

The scatter shows five stylized bank archetypes plotted against the warranted line. A troubled bank earning 5% ROE sits near 0.55× book. A value-trap candidate at 7% ROE sits near 0.7×. The average bank at 10% ROE sits right on 1.0×. A solid franchise at 12% earns its 1.35× premium, and an elite franchise compounding at 17% ROE commands well over 2× book. The banks do not scatter randomly — they ride up the warranted line in lockstep with their profitability. When a bank sits *off* the line — trading much cheaper or richer than its ROE justifies — that gap is either an opportunity or a warning, and the rest of your work is figuring out which.

This is the practical payoff of the whole framework: instead of memorizing that "JPMorgan trades at 1.5× and some regional trades at 0.8×," you understand that JPMorgan earns a high-teens ROE and the regional earns single digits, and the multiples simply *follow*. The multiple is the consequence; ROE is the cause.

## Picking the right tool: P/B, P/E, DDM, and residual income

We now have four valuation methods on the table. They are not competitors so much as a toolkit, each answering a slightly different question and each best in different conditions. A disciplined analyst runs more than one and checks that they roughly agree.

![Four bank valuation methods and when each one fits](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-6.png)

The matrix lays out what each method measures, where it shines, and where it breaks:

- **P/B versus ROE** is the workhorse. It measures the price paid per dollar of book against the ROE earned, ties directly to the warranted-multiple formula, and is stable across the cycle. Its weakness: book value can be stale or wrong if the bank's assets are mismarked — for example, bonds held at cost that have actually lost value (the SVB problem, which we will get to).
- **P/E** is a quick cross-check when earnings are stable, but it whipsaws violently through the credit cycle — low at the top, sky-high or negative at the bottom — so it lies at exactly the turning points. Use it to sanity-check, never to anchor.
- **The DDM** suits mature banks with a steady payout policy and is the natural way to value the cash a bank returns. Its weakness, as we saw, is that capital rules cap the payout, so the dividend is not freely chosen and can be cut by regulators overnight.
- **Residual income** is the most illuminating, because it shows *exactly* where value sits above (or below) book by capitalizing the excess return. Its cost is that it needs a credible long-run forecast of ROE and COE — garbage in, garbage out.

The professional move is to triangulate. Use P/B-versus-ROE as the spine, confirm with a residual-income build that decomposes the premium, sanity-check against a normalized P/E, and use the DDM to think about how and when capital comes back. When all four roughly agree, you have a defensible value. When they diverge sharply, the divergence itself is telling you where the uncertainty lives — usually in the through-cycle ROE assumption. The deepest reading of any bank's filings, including these inputs, comes from working through the [analyst's checklist for a bank's annual report](/blog/trading/banking/how-to-read-a-banks-annual-report-the-analysts-checklist).

## When a low P/B is cheap, and when it is a value trap

This is the section that earns the post. Everything so far tells you what a bank *should* trade at. Now we confront the hardest question in practice: you see a bank trading at 0.6× book. Is it a screaming bargain or a slow-motion disaster?

### The decisive question: is the low ROE temporary or permanent?

Work the formula backwards. A bank trading at 0.6× book is being priced *by the market* for a specific level of profitability. Let us back out what ROE that 0.6× implies.

![The value trap test for a cheap bank](/imgs/blogs/valuing-a-bank-price-to-book-roe-and-the-warranted-multiple-8.png)

The decision tree above is the whole test in a single flow. A bank at 0.6× book *looks* cheap. The only question that matters is the one in the amber box: **is the bank's sustainable ROE likely to be above its cost of equity?** If yes — if the market is too gloomy and the bank's ROE will recover above ~10% — then its warranted multiple is above 1, the current 0.6× is too low, and the stock should re-rate upward: a genuine bargain. If no — if the ROE is durably stuck below the cost of equity because the franchise is broken — then the warranted multiple really *is* below 1, the 0.6× is *deserved*, and there is no bargain at all: a value trap. The cheap multiple is the market doing its job, correctly pricing a bank that destroys value.

#### Worked example: backing out the implied ROE from a 0.6× P/B

A bank trades at **P/B = 0.6×**. Its cost of equity is **10%** and analysts think it can grow book at **2%**. What ROE is the market pricing in? Rearrange the warranted-multiple formula to solve for ROE:

$$\text{P/B} = \frac{\text{ROE} - g}{\text{COE} - g} \;\;\Rightarrow\;\; \text{ROE} = \text{P/B} \times (\text{COE} - g) + g$$

$$\text{ROE} = 0.6 \times (0.10 - 0.02) + 0.02 = 0.6 \times 0.08 + 0.02 = 0.048 + 0.02 = 0.068$$

The market is pricing this bank for a sustainable ROE of **about 6.8%** — well below its 10% cost of equity. Now the value-trap test is sharp and answerable: *do you believe this bank can earn more than 6.8% sustainably?*

- If you think the bank's true through-cycle ROE is, say, 10% (the low number is a temporary cyclical dip or a one-off charge), then the market is too pessimistic. At a 10% ROE the warranted multiple is (0.10 − 0.02)/(0.10 − 0.02) = 1.0×, versus 0.6× today — a 67% upside if you are right. **Genuine bargain.**
- If you think 6.8% (or worse) is the durable reality — the bank is structurally unprofitable, over-costed, stuck in low-margin lending, or facing terminal deposit flight — then 0.6× is roughly fair, the market is right, and there is no edge. **Value trap.**

**The intuition: a low P/B is never cheap or expensive on its own — it is cheap or expensive relative to the ROE you believe the bank will actually earn. The whole judgment collapses into one forecast: sustainable ROE versus the cost of equity.**

### The asset-quality trap inside the value trap

There is a second, sneakier way a low P/B fools you, and it goes to the heart of why even book value can mislead. The warranted-multiple math assumes the *stated* book value is real. But book value is only as good as the marks on the assets behind it. If a bank is carrying bad loans at full value (not yet provisioned) or bonds at amortized cost that have quietly lost 20% of their market value as rates rose, then its *stated* book is overstated. The real, marked-to-market book is lower — sometimes dramatically — so a "0.8× stated book" might actually be "1.1× real book." That is not cheap; it is expensive in disguise.

This is precisely what destroyed Silicon Valley Bank in 2023. SVB held a vast portfolio of long-dated Treasuries and mortgage bonds at amortized cost — at *par* on the books — even though rising rates had cratered their market value. Its stated book value looked solid. Its *economic* book value, once you marked the bonds to market, had a hole in it big enough to wipe out its equity. A naive P/B screen would have called SVB reasonably valued days before it failed. The lesson: **before you trust a low P/B as cheap, mark the book yourself — check the unrealized losses on held-to-maturity securities, the adequacy of loan reserves, and the level of nonperforming loans.** A bargain on overstated book is no bargain.

## Common misconceptions

**"A bank below book value is automatically cheap."** No. A bank earning below its cost of equity *deserves* to trade below book — the discount is the market correctly capitalizing value destruction. Our worked example showed a 0.6× bank is pricing in a ~6.8% ROE; if that ROE is real, 0.6× is fair value, not a discount. Cheapness depends entirely on whether the implied ROE is too low relative to what the bank can actually earn. A sub-1× multiple is a *question*, never an answer.

**"P/E works for banks just like any other company."** P/E is the most misleading single number for a bank. Bank earnings are net of loan-loss provisions, which collapse in booms and explode in busts, so the P/E looks cheapest at the top of the cycle (when provisions are artificially low) and most expensive — or negative — at the bottom (when provisions spike). It is pro-cyclical in the exact wrong direction. Use a *normalized*, through-cycle earnings figure if you must use P/E at all, and treat it only as a cross-check on the P/B-versus-ROE spine.

**"Book value is hard accounting truth."** Book value is an *estimate*, and a bank can flatter it. Loans carried at full value that are quietly going bad, held-to-maturity bonds marked at cost while their market value has fallen, goodwill from overpriced acquisitions — all inflate stated book. Always work with *tangible* book (strip goodwill) and sanity-check the marks. SVB's stated book looked fine right up until its unrealized bond losses turned out to exceed its equity.

**"A high ROE always justifies the premium."** Not if the ROE is unsustainable or risk-fuelled. A bank can manufacture a temporarily high ROE by levering up, by under-provisioning (booking too little for future losses), or by chasing high-yield, high-risk lending that will blow up later. The warranted multiple should be built on a *sustainable, through-cycle, appropriately-provisioned* ROE, not last quarter's flattered number. A 20% ROE earned by taking reckless risk is worth less than a steady 14% — it is borrowed from the future.

**"You should pick the one right valuation model."** No model is the truth; each is a lens. The warranted multiple, residual income, and the DDM are algebraically the same equation, so when they disagree it is because your *inputs* differ across them — and that disagreement is information about where your uncertainty lives. P/E adds an independent, earnings-based check. Triangulate; never anchor on a single method.

## How it shows up in real banks

The framework is not academic. It explains the actual spread of bank multiples you can pull up on any screen, and the dramatic re-ratings that happen when ROE expectations shift.

### High-ROE franchises trade at a premium

The banks that consistently earn well above their cost of equity command persistent premiums to book, year after year, and the warranted-multiple formula explains why. A franchise that reliably earns a high-teens ROE — through a sticky, low-cost deposit base, dominant scale, fee businesses that do not consume much capital, or a wealth-management arm that earns fees on someone else's assets — has a warranted multiple well above 1. Run a 17% ROE through our formula at a 10% COE and 3% growth and you get (0.17 − 0.03)/(0.10 − 0.03) = 2.0×, a 100% premium to book. That is not the market being exuberant; it is the market correctly capitalizing seven points of excess return forever. The best US franchises and a handful of standout global banks have traded around or above 2× tangible book for exactly this reason. The premium is *earned*, dollar for dollar, by the excess return.

The harder question — and the one that separates a durable premium from a fleeting one — is *what makes the high ROE last.* A premium multiple is the market betting the excess return persists for years, so the whole bet rests on the *durability* of ROE, not its current level. The most durable sources are structural: a deposit franchise so sticky and cheap that the bank funds itself well below what rivals pay (a low cost of funds is the single most valuable asset a bank can own); a scale advantage that spreads fixed technology and compliance costs over a vast balance sheet; and capital-light fee income — payments, asset management, advisory — that earns ROE without consuming the regulated equity that caps balance-sheet growth. Those edges are slow to erode, so the premium is slow to fade. The fragile sources of high ROE — thin provisioning, aggressive leverage, a hot-money funding base, or a concentrated bet on one cyclical sector — can vanish in a single bad year, which is why the market is right to assign them a *lower* multiple even at the same headline ROE. When you see a premium, the diligence is not "is the ROE high?" but "is it high for a reason that will still be true in five years?"

### Low-ROE banks trade at a discount — sometimes a deserved one

At the other end, banks that chronically earn below their cost of equity trade below book, and many *European* banks spent the entire 2010s decade there. After the eurozone crisis, negative policy rates crushed net interest margins, weak economies kept loan growth and loan quality poor, and structural over-banking kept costs high. Sector ROEs sat stubbornly in the 5–7% range against costs of equity near 10%. The result: many large European banks traded at 0.4× to 0.7× book *for years*. To a naive screener, an entire continent's banks looked like deep-value bargains. To anyone applying the warranted-multiple framework, those discounts were largely *deserved* — they correctly priced banks earning well below their cost of equity, with no clear catalyst to lift ROE. Some were value traps; the few that restructured their cost base or benefited from rising rates and lifted ROE back above their cost of equity *did* re-rate. The framework told you exactly what to look for: a credible path to ROE above COE.

### The 2023 regional-bank sell-off

March 2023 was a live demonstration of every theme in this post, compressed into a few weeks. As we saw in the ROE chart, the industry was earning a healthy ~10% — comfortably at its cost of equity — and regional banks broadly traded around book. Then Silicon Valley Bank failed, and the market suddenly repriced the entire regional-bank sector on two fears that map directly onto our framework.

First, **the marks on stated book were wrong.** Investors realized that many regionals, like SVB, were carrying large unrealized losses on held-to-maturity bond portfolios that their stated book value did not reflect. The market re-marked their *economic* book downward, and prices fell to reflect the lower real book — not the comfortable stated book. Banks that had looked like 1.0× suddenly looked like 1.3× or 1.5× of *honest* book, and that is expensive, not cheap.

Second, **future ROE expectations collapsed.** The same rate shock that hammered bond values forced banks to pay up for deposits to keep them from fleeing, which compressed net interest margins and therefore future ROE. A lower expected ROE means a lower warranted multiple — mechanically, by the formula. The sell-off was the warranted multiple recalculating in real time as both inputs (real book value and sustainable ROE) deteriorated at once.

The cleanest illustration was the contrast at the top: JPMorgan, earning a high-teens ROE on a fortress balance sheet with a sticky deposit base, traded *up* and acquired failed assets, while the weakest regionals — earning thin ROEs on shakier funding with underwater bond books — traded toward zero. Same crisis, opposite multiples, and the warranted-multiple framework predicted the direction of both. (The full anatomy of these runs sits in the cross-asset post on the [SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

#### Worked example: re-rating from a margin shock

Take a regional bank earning a 10% ROE, trading at fair value of ~1.0× book (COE 10%, g 3%). Now a deposit-cost shock cuts its sustainable ROE to **8%**. The warranted multiple drops to:

$$\text{Warranted P/B} = \frac{0.08 - 0.03}{0.10 - 0.03} = \frac{0.05}{0.07} = 0.71\times$$

A 2-point cut in expected ROE — from 10% to 8% — drops the warranted multiple from 1.0× to 0.71×, a **29% fall in fair value**, before any change in stated book at all. Now layer in a downward re-mark of book itself (say underwater bonds knock real book down 10%), and the implied price falls by roughly another tenth on top. **The intuition: bank stocks fall hard in a rate shock not because they are irrationally sold but because both inputs to the warranted multiple — sustainable ROE and real book value — deteriorate at the same time, and the multiple multiplies the damage.**

## The takeaway: how to actually value a bank

If you remember one thing, remember the pivot: **a bank earning its cost of equity is worth exactly one times book, and every point of ROE above or below that line moves the warranted multiple predictably.** That single sentence is the entire framework. Everything else is refinement.

Here is the workflow that falls out of it, in the order a disciplined analyst runs it:

1. **Estimate the sustainable, through-cycle ROE.** Not last quarter's flattered number — the ROE the bank can earn across a full credit cycle, with honest provisioning and no leverage tricks. This is the hardest and most important judgment; everything downstream depends on it. Decompose it with the DuPont logic from [ROE, ROA, and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) to check that it is durable, not borrowed from risk.
2. **Estimate the cost of equity.** A round 9–11% for a typical bank via CAPM. Be honest about whether this bank is riskier than average (thin capital, concentrated funding, volatile markets exposure all push COE up).
3. **Compute the warranted multiple** with (ROE − g) / (COE − g), and cross-check it with a residual-income build (so you can *see* the premium as capitalized excess return) and a normalized P/E.
4. **Mark the book.** Before trusting any P/B, check tangible book (strip goodwill), unrealized securities losses, loan-reserve adequacy, and nonperforming loans. A bargain on overstated book is a trap.
5. **Run the value-trap test on any low multiple.** Back out the implied ROE from the current P/B and ask whether the bank can sustainably beat it. Cheap is only cheap if the low ROE is temporary.

Tie it back to the spine of this series. A bank is a leveraged, confidence-funded maturity-transformation machine: it earns a thin spread on a huge balance sheet, funded mostly by borrowed money and a thin slice of equity. Valuation is just the market's verdict on whether that machine generates enough return on its thin equity to justify the risk the owners bear. When it does — when ROE clears COE — the franchise is worth more than its book, and the premium is the capitalized value of that edge. When it does not, the discount is the market correctly pricing a machine that consumes capital faster than it creates value. The warranted multiple is, in the end, a single honest question wearing a formula's clothes: *does this bank earn its keep?* Get the sustainable ROE right, and the price takes care of itself.

## Further reading & cross-links

- [ROE, ROA, and the leverage identity: how a bank is judged](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) — the engine of the warranted multiple; how a 1% return on assets becomes a 12% return on equity, and how to decompose ROE to test its durability.
- [The income statement of a bank: net interest income, fees, and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — where the earnings that drive ROE actually come from, and why provisions make them so cyclical.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why book equity is the regulated, binding constraint that makes price-to-book the natural valuation lens.
- [How to read a bank's annual report: the analyst's checklist](/blog/trading/banking/how-to-read-a-banks-annual-report-the-analysts-checklist) — where to find the ROE, the marks, the unrealized losses, and the reserve adequacy you need to value a bank honestly.
- [Building a DCF, part 2: cost of capital (WACC and CAPM)](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) — the full machinery for estimating the cost of equity that anchors the warranted multiple.
- [Valuing the hard cases: banks, insurers, REITs, and cyclicals](/blog/trading/equity-research/valuing-the-hard-cases-banks-insurers-reits-cyclicals) — the equity-research view of why financial firms get their own valuation playbook.
- [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the live case study where stated book value and sustainable ROE deteriorated at once, and multiples repriced in days.

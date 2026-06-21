---
title: "The Financing Ladder: From Bootstrap to Public Markets"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How a company climbs from founders' cash to angels, venture capital, private growth rounds, and finally the public market — and what it trades away at every rung."
tags: ["capital-markets", "venture-capital", "ipo", "startup-financing", "dilution", "preferred-stock", "private-equity", "primary-market", "cost-of-capital", "liquidity"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A company raises money in stages, and each stage is a trade: cheaper, more patient capital in exchange for less control and more disclosure, until the firm reaches the public market where capital is widest and the demands are steepest.
>
> - The ladder runs bootstrap → friends/family & angels → seed → venture capital (Series A/B/C) → late-stage private → public equity and debt. Each rung has a different *who* providing the money and a different *what* they demand.
> - The price of money falls as the firm de-risks: a seed investor wants to own 20–30% of a tiny company; a public bond buyer accepts a 5% coupon. Early money is expensive precisely because most early companies fail.
> - Venture money is *preferred* stock with a liquidation preference and a board seat — it gets paid first and votes on big decisions, which is why a founder can sell for \$50M and a 1x-preference investor still walks away whole before common holders split the rest.
> - The number to remember: in 2021, US IPOs raised about **\$142 billion**; in 2022 that collapsed to about **\$8 billion**. The top of the ladder is the most powerful and the most fickle rung of all.

In March 2021, a software company most people had never heard of filed a document with the US Securities and Exchange Commission and, a few weeks later, watched its shares open on the New York Stock Exchange at nearly double the price its bankers had set the night before. The founders, who eight years earlier had been three people writing code on borrowed laptops, were suddenly worth hundreds of millions of dollars on paper. The venture capital fund that wrote the company's first real check — \$2 million when the business was barely an idea — turned that stake into something north of \$400 million.

That morning is the top of a ladder. It is also the part everyone sees: the bell, the confetti, the ticker. What almost nobody outside the company sees is the eight-year climb — the founders maxing out credit cards, the awkward dinner where a parent wrote a \$25,000 check, the seed round done on a two-page document, the brutal Series B term sheet, the growth round led by a fund that had never touched a startup before. Each of those steps was a rung on the **financing ladder**, and at each rung the company swapped a little more control and a little more privacy for a little more money at a slightly lower price.

This post is about that climb. It belongs to the **primary market** — the engine of a capital market that *creates* securities to raise money, as opposed to the **secondary market** that merely trades them afterward. (If those two words are new, start with [what a capital market is and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use).) Our whole series rests on one secret: **secondary-market liquidity is what makes primary issuance possible** — nobody funds a risky young company unless they can eventually sell their claim on it. The financing ladder is the story of how that "eventually" gets shorter and cheaper as a company grows up, until the firm reaches the rung where the claim can be sold any morning the market is open.

![The financing ladder from bootstrap to public markets](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-1.png)

## Foundations: what "raising capital" actually means

Before we climb, let's build the idea from zero, because the whole ladder makes sense only once you see what is actually being bought and sold.

A company needs money for two reasons: to **start** (build the product, hire the first engineers, rent the office) and to **grow** (open new markets, buy inventory, outspend a rival). It can get that money in exactly two ways, and this fork runs through every rung of the ladder. It can **borrow** it — take cash now and promise to pay it back with interest — which we call **debt**. Or it can **sell a piece of itself** — hand over a slice of ownership in exchange for cash that never has to be repaid — which we call **equity**. Most of the early ladder is equity, because a startup with no revenue has nothing to pay interest *with* and no assets a lender could seize; the higher rungs reintroduce debt once there are real cash flows to borrow against. The deep mechanics of that choice live in a sibling post — [debt vs equity, the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — and we will lean on it rather than re-derive it.

When a company sells equity, it is selling **shares**: units of ownership. If a company has 1,000,000 shares and you own 100,000, you own 10% of it — 10% of the votes, 10% of any dividends, 10% of whatever the company sells for one day. A **security** is just a tradeable claim like this — a share of stock, a bond, a convertible note — a legal contract that says "the holder of this is owed something." The primary market is where these securities get *created and sold for the first time*; everything after that is the secondary market.

Three concepts will recur at every rung, so let's nail them now.

**Cost of capital.** This is the price the company pays for money, expressed as a return the provider expects. For debt it is obvious — the interest rate. For equity it is subtler: an investor who buys 20% of your company for \$2 million is implicitly demanding that the company eventually be worth enough that their 20% is a great return on \$2 million. Early-stage equity is the most *expensive* capital in the world — a seed investor might want a 10x return because nine of their ten bets will die — and public debt is among the cheapest. The entire ladder is a slide from expensive money to cheap money.

**Control.** Money rarely comes free of strings. Lenders attach covenants; equity investors attach **board seats**, **veto rights**, and **protective provisions**. The more of your company someone buys, and the earlier and riskier the stage, the more control they tend to demand. A founder who owns 100% on day one will, by the time the company is public, often control well under half the votes.

**Liquidity.** This is how easily a holder can turn their claim back into cash. A founder's shares in a two-person startup are essentially **illiquid** — there is no buyer. A share of a public company can be sold in milliseconds. As the company climbs the ladder, the liquidity available to its owners widens, and that widening is the deep reason the ladder exists at all.

![US IPO proceeds collapse and recover by year](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-2.png)

Look at the chart above before we go further, because it is the punchline of the whole ladder, shown in dollars. The top rung — the public market — is enormously powerful: in 2021, US companies raised roughly **\$142 billion** through traditional initial public offerings (IPOs). But it is also wildly cyclical: the very next year, 2022, that figure collapsed to about **\$8 billion** as rising interest rates slammed the IPO window shut. Companies that had planned to climb the last rung in 2022 were stranded one step down, forced to raise an extra private round or simply wait. The lesson the chart teaches is that *the higher you climb, the more your fate depends on a market you don't control*. Hold that thought; it explains the modern "staying private longer" trend we reach at the end.

### Why early money is so expensive: the power law of failure

It is worth pausing on *why* the cost-of-capital curve slopes the way it does, because the reason is not greed — it is arithmetic. Early-stage investing is governed by a **power law**: in a typical venture portfolio, most companies return little or nothing, a handful return the original money, and one or two return a multiple large enough to carry the entire fund. A study of venture returns famously found that the best-performing investment in a fund often returns more than all the others combined. An investor who knows that nine of every ten early bets will fail *must* price each surviving bet to compensate — which is exactly why a seed investor wants to own 20% of your company for a small check and expects, on the bets that work, something like a 10x to 100x return.

This is the mechanism behind the slide we keep referencing. A pre-revenue startup has, by any honest accounting, a high probability of returning zero. To break even across a portfolio of such bets, the investor needs each *winner* to be enormous, which means demanding a large ownership stake at a low price per dollar of company value. As the company climbs each rung — ships a product, signs paying customers, shows the growth engine repeats, turns profitable — its probability of total failure drops. Lower failure probability means the next investor needs a smaller multiple to break even across their portfolio, which means they will pay a *higher* price for the same slice. The price of money is, at bottom, a bet on survival, and every rung is the company buying down its own failure probability with proof.

So what? "Raising capital" is never just getting money. It is a negotiated exchange of **control and disclosure for cash at a price**, and the terms of that exchange shift predictably as you climb, because the price of money tracks the odds the company survives to repay it. Now let's take the rungs in order.

## Rung 1 — Bootstrapping: the founder is the bank

The first and cheapest source of capital is the one nobody calls "capital": the founders' own money, plus whatever the business itself generates.

**Bootstrapping** means funding the company from internal resources — founder savings, a day-job salary that quietly subsidizes the side project, credit cards, and above all **revenue**: money customers pay you, which you plow back into the business instead of taking home. The classic bootstrapped path is "sell something, use the profit to make more, repeat." Companies like Mailchimp, Zoho, and Spanx grew to enormous size this way without ever taking outside equity until very late, if at all.

**Who provides the capital?** The founders, and the customers. That's it. There is no outside investor at this rung.

**What do they demand?** Nothing — and that is the entire point. Bootstrapped capital comes with zero dilution (the founders still own 100%), zero board seats, zero disclosure obligations beyond paying taxes. The founders answer to no one but their customers.

**What does it cost?** On paper, nothing — there is no interest, no equity given up. But the real cost is **opportunity and speed**. Money you reinvest is money you don't take as salary; growth funded only by profit is slow; and a well-funded competitor can outspend you into oblivion while you wait for revenue to compound. The cost of bootstrapped capital is the deals you can't do and the markets you can't grab fast enough.

**What liquidity is available?** Essentially none. The founders' wealth is entirely locked inside the company. They can't sell a share because there is no buyer and often no formal share structure to sell.

#### Worked example: how far does \$50,000 of bootstrap money go?

Suppose two founders put in \$25,000 each — \$50,000 total — and the business earns a 30% net margin. To fund a single \$120,000 senior engineer hire purely from reinvested profit, the company must generate \$120,000 of *profit*, which at a 30% margin means \$400,000 of revenue: `\$120,000 ÷ 0.30 = \$400,000`. If the company is doing \$50,000 of revenue a month (\$600,000 a year, \$180,000 of profit), it can afford that one hire and a little slack — but not the *five* hires a freshly funded rival just made overnight. The intuition: bootstrapping keeps 100% of the company in the founders' hands, but it rations growth to the speed of the cash register, and in a winner-take-most market that ration can be fatal.

Bootstrapping is not merely the rung you climb off of as fast as possible — for many businesses it is the *right* permanent home. A company in a market that rewards profitability over land-grab speed, or whose product can be sold before it is fully built, can compound on its own cash flow and keep 100% of the upside. The trade only tilts toward outside capital when the market is winner-take-most, when the product needs heavy up-front spending before any revenue, or when a competitor's funding would let them out-build you faster than your cash register can fund a response. The mistake founders make is treating outside capital as a sign of success rather than as a tool with a steep price; raising a round you do not need simply hands away ownership and control for money you could have earned.

So what? Bootstrapping is the *only* rung where the founder gives up nothing — which is exactly why it can't last for a company chasing a big, contested market, and exactly why it can last forever for one that isn't. The decision to step onto rung two is the decision to trade ownership for speed, and it should be made because speed is genuinely needed, not by reflex.

## Rung 2 — Friends, family, and angels: the first outside checks

When the founder's own money runs out but the company isn't yet ready for professional investors, the next rung is the **friends-and-family round** and the **angels**.

A friends-and-family round is exactly what it sounds like: a parent, a former boss, a college roommate writes a check — often \$10,000 to \$100,000 — based largely on trusting the founder rather than analyzing the business. An **angel investor** is a wealthy individual (in the US, usually an "accredited investor" — meaning income or net worth above a regulatory threshold) who invests their *own* money in early startups, typically writing checks of \$25,000 to \$250,000. Angels often have operating experience and bring advice and introductions alongside the cash.

**Who provides the capital?** Individuals, spending their own money, often without a fund or a partnership behind them.

**What do they demand?** Less than professional investors, but more than nothing. Friends and family usually take simple equity or a convertible instrument (more on those in a moment) and ask for little control. Angels may want a small ownership stake, occasional updates, and sometimes an informal advisory role, but rarely a formal board seat at this size. The disclosure burden is light: a pitch deck, a financial model, and honest conversations.

**What does it cost?** Still very expensive in equity terms, because the company is still extremely risky. An angel buying into a pre-revenue startup is implicitly demanding a huge multiple, because most of these companies will return zero. The hidden cost is *relational*: taking your uncle's retirement savings into a business that has a 70% chance of failing is a different kind of risk than taking a fund's money.

**What liquidity is available?** None for years. These investors are buying a lottery ticket they cannot cash until a much later round, an acquisition, or an IPO — often five to ten years away, if ever.

So what? This rung is governed less by spreadsheets than by trust, and it is where the founder first confronts the reality that other people now have a financial stake in their decisions. The dollar amounts are small, but it is the psychological transition from "my company" to "our company."

## Rung 3 — Seed and venture capital: where the machinery begins

Now the ladder turns professional, and the instruments get genuinely interesting. This is the rung where most of the financial machinery of startups lives, so we'll spend the most time here.

**Venture capital (VC)** is money raised by a fund from large investors — pension funds, university endowments, wealthy families, called **limited partners** or **LPs** — and deployed into a portfolio of high-risk, high-growth startups by the fund's managers, the **general partners** or **GPs**. The VC model is brutal arithmetic: a fund expects most of its investments to fail, a few to return its money, and one or two to return the *entire fund several times over*. That is why VCs chase only companies that could plausibly become enormous — a business that can return 2x is useless to a fund that needs a single bet to return 50x.

The early VC rungs have names. The **seed round** is the first institutional money, often \$1–5 million, meant to get a product built and find early customers. **Series A** (typically \$5–20 million) funds the search for a repeatable growth engine. **Series B and beyond** (often \$20–100 million+) pour fuel on an engine that already works.

### Two ways to take seed money: priced rounds and convertibles

There are two fundamentally different ways a startup takes early money, and the difference matters.

A **priced round** sets an explicit value on the company today, called the **valuation**, and the investor buys shares at that price. We say a company raises "\$2 million at an \$8 million pre-money valuation." **Pre-money** is what the company is deemed worth *before* the new cash goes in; **post-money** is pre-money plus the new cash. The investor's ownership is simply their check divided by the post-money valuation.

The alternative is a **convertible instrument** — money that goes in *now* but whose price is set *later*, at the next priced round. The two common forms are the **convertible note** (a loan that converts into shares instead of being repaid) and the **SAFE** (Simple Agreement for Future Equity, a standardized instrument invented by the accelerator Y Combinator that is not even debt — just a right to future shares). Both let a young company take money fast without the founders and investors having to argue about an exact valuation when the company is too early to value sensibly. They usually carry a **valuation cap** (a ceiling on the price at which the money converts, protecting the early investor from being diluted by a sky-high later round) and sometimes a **discount** (the early money converts at, say, 20% below the next round's price, rewarding the investor for coming in early).

The cap and the discount are not decoration — they are the early investor's reward for taking on the most risk, and they bite at the next priced round. The investor converts at whichever gives them *more* shares: the capped price or the discounted price. This is why a founder who raises a lot of money on SAFEs with a low cap can be surprised at how much of the company those SAFEs claim when they finally convert. The convertible feels free at the time — no valuation set, no immediate dilution on the cap table — but the dilution is merely *deferred*, and it lands all at once when the priced round closes.

#### Worked example: how a SAFE converts at the next round

A founder raises \$500,000 on a SAFE with a **\$5,000,000 valuation cap** and a **20% discount**. Eighteen months later the company raises a priced Series A at a **\$20,000,000 pre-money valuation**, with the new shares priced at, say, \$10.00 each. The SAFE converts at the *better* of two prices for the investor. The discount price is `\$10.00 × (1 − 0.20) = \$8.00`. The cap price is the Series A price scaled by the cap over the round's pre-money valuation: `\$10.00 × (\$5,000,000 ÷ \$20,000,000) = \$2.50`. The investor takes the lower price — \$2.50 — because it buys more shares: `\$500,000 ÷ \$2.50 = 200,000` shares, versus only `\$500,000 ÷ \$8.00 = 62,500` shares at the discount. So the cap, not the discount, governs, and the SAFE investor ends up owning shares worth `200,000 × \$10.00 = \$2,000,000` at the Series A price — a 4x paper gain on their \$500,000 for taking early risk. The intuition: a low valuation cap is a powerful reward for early money, and founders who stack several capped SAFEs are quietly promising away a large block of the company that only shows up on the cap table at the next priced round.

![What a priced round does to the cap table](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-4.png)

The figure above shows the mechanical heart of every priced round: new money buys *new* shares, which means the existing owners' percentage shrinks even as the company's total value grows. This shrinkage has a name — **dilution** — and it is the single most important concept on the entire ladder. Let's make it concrete.

#### Worked example: a \$2M seed at an \$8M pre-money valuation

A founder owns 100% of a company. A seed fund offers \$2,000,000 at an \$8,000,000 pre-money valuation. The post-money valuation is `\$8,000,000 + \$2,000,000 = \$10,000,000`. The investor's ownership is their check over the post-money value: `\$2,000,000 ÷ \$10,000,000 = 20%`. So after the round the seed fund owns 20% and the founder owns 80%.

Here is the part founders get wrong: the founder did not "lose" \$2 million. Before the round their 100% was worth \$8 million (the pre-money value). After the round their 80% is worth `0.80 × \$10,000,000 = \$8,000,000` — exactly the same dollar value. Dilution shrank their *percentage* but not their *dollar value*, because the company is worth more by exactly the cash that went in. The intuition: a smaller slice of a bigger pie can be the same size or bigger — dilution only hurts if the new money doesn't grow the pie faster than it shrinks your share.

### The hidden dilution: the option pool and the fully-diluted cap table

There is a subtlety in that clean example that real term sheets exploit. A startup needs to pay employees in equity, so it sets aside an **option pool** — a block of shares reserved for future hires, typically 10–20% of the company. The fight is over *when* that pool is created relative to the round, because it changes who bears the dilution. Investors almost always insist the pool be carved out of the **pre-money** valuation — meaning it dilutes the *founders*, not the new investor. This is the "option pool shuffle," and it can quietly cost a founder several points of ownership.

Watch how it works on the seed example. The investor offers \$2,000,000 at an \$8,000,000 pre-money valuation but also requires a 15% post-money option pool. If the pool is carved out of the pre-money, then of the \$10,000,000 post-money company, the investor takes 20% and the pool takes 15% — `0.15 × \$10,000,000 = \$1,500,000` of value — and *both* come out of the founder's slice. The founder is left with `100% − 20% − 15% = 65%`, not 80%. The \$8,000,000 "pre-money" the founder thought they had was really worth `\$8,000,000 − \$1,500,000 = \$6,500,000` once the pool is stripped out — an "effective pre-money" the founder rarely sees stated plainly.

This is why sophisticated founders track the **fully-diluted cap table** — ownership computed as if every option, warrant, SAFE, and convertible had already turned into shares — rather than the simpler count of issued shares. The fully-diluted view is the only one that tells you what you'll actually own at an exit, because that is the moment everything converts at once. So what? The headline "\$2 million at \$8 million pre" hides at least two further dilutions — the option pool and any converting SAFEs — and a founder who reads only the headline will be surprised by their real ownership every single round.

### What VCs actually buy: preferred stock with teeth

Here is the detail that separates professional venture capital from a friend's check, and it is almost always invisible in the headline. VCs do not buy the same **common stock** the founders hold. They buy **preferred stock** — a senior class of equity with three sharp teeth.

**Tooth one: the liquidation preference.** A **1x liquidation preference** means that if the company is sold or wound down, the preferred investors get their money back *first* — before common shareholders see a cent — and only then does the remainder get split. ("1x" means one times their investment; aggressive rounds sometimes carry 2x or 3x.) This is the mechanism that lets a VC fund a risky company: even a mediocre exit returns their capital, while the founders' common stock only pays out after the preferred is satisfied.

**Tooth two: the board seat.** A meaningful round usually comes with one or more seats on the company's **board of directors** — the body that hires and fires the CEO and approves big decisions. A founder who controlled the board alone now shares it.

**Tooth three: protective provisions.** These are vetoes: the preferred investors must approve certain actions — raising more money, selling the company, changing the share structure — regardless of how few shares they hold. It is control decoupled from ownership percentage.

![The exit waterfall with a 1x liquidation preference paid first](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-6.png)

The figure above shows how the first tooth bites at exit. The investors' preference comes off the top of the proceeds; only the residual flows down to common holders. Let's run the numbers, because this is where founders are most often surprised.

#### Worked example: a 1x liquidation preference in a \$50M exit

A startup raised \$10,000,000 of preferred stock at a 1x liquidation preference, and the investors own 20% of the company on paper. The company sells for **\$50,000,000**. Naively you'd split the proceeds by ownership: investors get 20% = \$10,000,000, founders get the rest. But with a 1x preference, the investors choose the *better* of two outcomes. Option A — take the preference: \$10,000,000 off the top (their 1x). Option B — convert to common and take 20%: `0.20 × \$50,000,000 = \$10,000,000`. Here both happen to equal \$10,000,000, so the founders and other common holders split `\$50,000,000 − \$10,000,000 = \$40,000,000`.

Now watch what happens in a *bad* exit. If the company sells for only \$12,000,000, the investors take their \$10,000,000 preference off the top, leaving just `\$12,000,000 − \$10,000,000 = \$2,000,000` for everyone else — even though the investors "only" own 20%. The intuition: a liquidation preference is downside insurance for the investor paid for by the common holders, and it is why founders can build a company worth tens of millions and still walk away with surprisingly little if the exit is small.

### The cost and liquidity at the VC rung

**Who provides the capital?** Institutional funds (VCs), pooling money from LPs.

**What do they demand?** Preferred stock with a liquidation preference, board seats, protective provisions, **pro-rata rights** (the right to keep their percentage by investing in future rounds), and regular reporting. The disclosure burden jumps sharply: monthly metrics, board decks, audited financials at later stages.

**What does it cost?** In percentage terms, a lot — a seed round commonly costs 15–25% of the company; a Series A another 15–25%. But the *price per dollar* has fallen versus angels, because the company is less risky than it was. The slide down the cost-of-capital curve has begun.

**What liquidity is available?** Still low, but a crack appears: later rounds sometimes include a small **secondary** component where founders or early employees sell a few shares for personal cash. This is the first time the founder can take real money off the table — and it foreshadows the employee-liquidity markets we reach at the top.

So what? The VC rung is where capital stops being a gift of trust and becomes a *contract with explicit power*. The founder gets larger, cheaper money — and accepts that a board can now fire them and that investors get paid first when the music stops.

## Rung 4 — Growth and late-stage private: the big private money

A company that has proven its engine works — real revenue, real growth, a path to profitability — graduates to the **growth** or **late-stage private** rung. Here the checks balloon to \$50 million, \$200 million, sometimes a billion dollars, and a new cast of investors appears.

**Crossover funds** are investors — often mutual funds or hedge funds like those run by Fidelity, T. Rowe Price, or Tiger Global — that traditionally buy *public* stocks but "cross over" into late private rounds to get in before the IPO. **Private equity (PE)** firms buy whole companies or large stakes, often using debt, and focus on more mature businesses. (For how the buy-side firms that provide this capital actually operate, the hedge-fund and PE internals live outside our series — we own their *role as capital providers*, not their plumbing.)

This rung also reintroduces **debt**, now that the company has cash flows to service it. **Venture debt** is a loan to a venture-backed startup, usually from a specialized lender, that supplements an equity round without diluting owners as much — the cost is interest plus, often, **warrants** (rights to buy a sliver of equity cheaply later). The appeal is straightforward: a company that just raised \$30 million of equity might take \$5–10 million of venture debt alongside it to extend its runway by several months *without* selling more of the company, because debt does not dilute. The catch is that debt must be serviced and repaid on a schedule, and a startup whose growth stalls can find a debt maturity arriving before its next equity round — a squeeze that has killed otherwise-viable companies.

**Mezzanine financing** sits between senior debt and equity: it is subordinated debt (paid after senior lenders but before equity) that often carries a high coupon plus an equity kicker, used to fund growth or buyouts. Its place in the capital structure is exactly its name — the mezzanine floor between the ground (senior debt, paid first) and the top (equity, paid last) — and its return reflects that middle risk: higher than a senior loan, lower than equity. The deep mechanics of how each layer of debt is priced and how the claims stack live in the fixed-income and banking series; here we only need the shape: as a company matures, its capital structure grows *layers*, each with a different claim on cash flows and a different price.

**Who provides the capital?** Crossover funds, PE firms, sovereign wealth funds, large family offices, and specialized debt lenders.

**What do they demand?** At this size, investors demand near-public-quality disclosure — audited financials, detailed unit economics, sometimes information rights as strong as a public shareholder's. Control terms persist (board seats, protective provisions), and debt providers attach **covenants** (promises about leverage, cash levels, and so on, with the lender able to act if they're breached).

**What does it cost?** Equity is now relatively cheap — a healthy late-stage company might sell 5–15% to raise a huge round. Venture debt costs a single-digit-to-low-teens interest rate plus warrants. The company has slid well down the cost curve.

**What liquidity is available?** Meaningfully more. Growth rounds frequently include substantial **secondary sales**, letting founders and early employees cash out millions while the company stays private. This is also the rung where **structured secondary markets** for private shares emerge — a partial, second-best version of the daily liquidity the public market provides, and a direct illustration of the series spine: even a *private* company needs *some* way for early holders to sell, or the early money would never have come in.

These private secondary markets are worth understanding because they are how the "staying private longer" trend is even possible. A handful of mechanisms do the work. A **direct secondary** is a one-off negotiated sale of an early holder's shares to an incoming investor, usually requiring the company's blessing (companies control who is on their cap table through transfer restrictions). A **tender offer** is the company organizing a sale at a set price across many employees at once — orderly, priced, and capped in size. And **specialized platforms** match private buyers and sellers, though always at a discount to the last round and subject to the company's approval. The common thread: liquidity at this rung is *rationed and discounted*. You can get some cash out, but not all, and not at the full headline price — which is precisely the gap the public market closes.

#### Worked example: a \$30M Series B at a \$120M pre-money valuation

A company raises \$30,000,000 at a \$120,000,000 pre-money valuation. Post-money is `\$120,000,000 + \$30,000,000 = \$150,000,000`. The new investor's ownership is `\$30,000,000 ÷ \$150,000,000 = 20%`. Suppose the founders owned 60% going into this round (earlier rounds already diluted them from 100%). They are diluted by 20% of their stake: their new ownership is `0.60 × (1 − 0.20) = 0.60 × 0.80 = 48%`. In dollar terms, though, their stake is now worth `0.48 × \$150,000,000 = \$72,000,000`, up from `0.60 × \$120,000,000 = \$72,000,000` — again, *identical* dollar value, because the round added exactly the cash that funded the new shares. The intuition: every priced round dilutes your percentage by `new money ÷ post-money`, so a 20% raise costs every existing holder 20% of their slice — predictable, mechanical, and survivable as long as the company keeps growing the pie.

So what? The late-stage rung is where a company can raise IPO-sized sums *without going public* — which is precisely what makes the modern decision to stay private possible. The capital is abundant and cheap; the only things the public market still offers that this rung doesn't are *permanent* capital and *daily* liquidity.

## Rung 5 — The decision: go public, or stay private longer?

Here the ladder reaches a genuine fork, and it is the most interesting strategic decision a maturing company makes. For most of financial history the answer was obvious: a successful company went public as soon as it could, because the private market simply couldn't supply enough capital. That is no longer true.

![Go public or stay private longer decision](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-7.png)

The figure above lays out the choice. **Going public** — doing an IPO — gets the company three things the private market struggles to match: a large pool of *permanent* capital (public shareholders never ask for their money back; they sell to each other), *daily liquidity* for every shareholder and employee, and a public valuation that becomes a currency for acquisitions and stock-based pay. But it also imposes the heaviest demands on the entire ladder: quarterly financial reporting, SEC oversight, the relentless scrutiny of analysts and short-sellers, the tyranny of the next earnings call, and a share price that anyone can watch fall in real time. The cost of public capital is *transparency and the loss of long-horizon privacy*.

### The "staying private longer" trend

Over the past fifteen years a powerful trend reshaped the top of the ladder: companies stay private far longer than they used to. In the 1990s a tech company might IPO five or six years after founding; today it is common to wait ten years or more. The median age of a US tech company at IPO roughly doubled across that span.

Three forces drive this. First, the late-stage private market got *enormous* — crossover funds and megafunds can now write the \$100M+ checks that once required a public offering, so a company can fund growth privately for years. A company valued above \$1 billion while still private even earned a nickname: a **unicorn**, coined in 2013 when such companies were rare. They are no longer rare. Second, **employee-liquidity secondary markets** matured: platforms and structured **tender offers** (where the company organizes a sale of employee shares to incoming investors) let employees and founders sell shares for cash *without* an IPO, removing one of the historic reasons to go public — giving people a way to realize their wealth. Third, the regulatory and reputational burden of being public rose, while the private capital available to avoid it grew.

#### Worked example: the cost of liquidity in a private tender offer

An employee holds 50,000 shares from joining a startup early. The company runs a tender offer at \$40 per share, but a private secondary sale typically clears at a discount to the "last round" price — say the last round implied \$50 per share, and the tender clears at \$40, a 20% discount: `(\$50 − \$40) ÷ \$50 = 20%`. The employee can sell 20% of their holding — `0.20 × 50,000 = 10,000` shares — for `10,000 × \$40 = \$400,000` of cash today. The other 40,000 shares stay locked, illiquid, riding on whether the company ever IPOs or gets acquired. The intuition: private secondary liquidity is real but *partial and discounted* — you pay roughly a fifth of the per-share value for the privilege of getting cash now instead of waiting for the top rung, which is exactly why the IPO, with its deep daily liquidity at full market price, never lost its appeal.

So what? Staying private longer is rational when private capital is abundant and the IPO window is uncertain — but it concentrates risk. A company that delays too long can find the window slammed shut (recall 2022) or its private valuation marked far above what the public market will pay, forcing a painful "down-round IPO." The fork is real, and timing it is one of the hardest calls in corporate finance.

## Rung 6 — The public markets: the widest, cheapest, most demanding capital

The top of the ladder is the **public capital market**, and it has two doors: **public equity** (selling shares to anyone via an IPO and then trading them on an exchange) and **public debt** (issuing bonds to a broad market of investors). This is where capital is widest and, for the lowest-risk issuers, cheapest — and where the demands are heaviest.

![Number of US IPOs per year](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-3.png)

The chart above counts US IPOs per year. Notice the same boom-and-bust shape as the proceeds chart: 397 traditional IPOs in the 2021 frenzy, then just 71 in 2022 when rates spiked and risk appetite vanished. The public-equity door doesn't open and close gently; it gaps. A company has to be *ready* when the window is open, because readiness plus a closed window equals a year of waiting.

**Who provides the capital?** Everyone — pension funds, mutual funds and ETFs, insurance companies, foreign investors, and retail investors buying through a brokerage app. The capital base is the broadest on the ladder by orders of magnitude.

**What do they demand?** The most disclosure of any rung. Public companies file detailed quarterly (10-Q) and annual (10-K) reports, disclose material events promptly (8-K), and submit to SEC enforcement and the discipline of a daily-traded price. They answer to thousands of shareholders, sell-side analysts, proxy advisors, and activist investors. The *control* a founder gives up is no longer to a handful of board members but to a diffuse public — and to the market's verdict every single day.

**What does it cost?** For a solid public company, equity is now relatively cheap and debt can be very cheap — an investment-grade firm might issue a 10-year bond at a coupon only a couple of percentage points above the US Treasury yield. (Why the Treasury yield is the anchor for *all* borrowing costs is a story we link out to, rather than re-derive: [the yield curve, the most important chart in finance](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance).) The slide down the cost-of-capital curve reaches its bottom here, because the public market prices in the full de-risking the company achieved by climbing every rung below.

**What liquidity is available?** Maximum. Every shareholder — founder, early employee, VC, public buyer — can sell their stake on an exchange in seconds, subject only to lockup periods and insider-trading rules. This is the payoff at the top of the ladder, and it loops straight back to our series' spine.

![Global IPO proceeds by year](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-5.png)

The global picture (above) tells the same story at world scale: roughly \$459 billion raised in global IPOs in 2021, falling to around \$121 billion by 2024. The top rung is the same shape everywhere — vast, powerful, and at the mercy of the macro cycle.

#### Worked example: an IPO raising \$500M at a \$5B valuation

A company IPOs at a **\$5,000,000,000** valuation and raises **\$500,000,000** of fresh **primary** capital (new shares the company sells to fund itself, as opposed to **secondary** shares that existing holders sell to cash out). The new public investors therefore buy `\$500,000,000 ÷ \$5,000,000,000 = 10%` of the company. If a founder owned 25% going in, post-IPO they own `0.25 × (1 − 0.10) = 22.5%`, a stake worth `0.225 × \$5,000,000,000 = \$1,125,000,000` on the first day's valuation. The bankers running the deal earn an underwriting fee — for a deal this size often around 3–4%, say `0.035 × \$500,000,000 = \$17,500,000` — for guaranteeing the raise and placing the shares. The intuition: an IPO converts a small slice of a now-public company into a huge, permanent cash pile *and* turns every remaining share into a liquid, daily-priced asset — which is why, even after the staying-private boom, the IPO remains the ladder's defining last rung.

![The IPO pop, median first-day return by year](/imgs/blogs/the-financing-ladder-from-bootstrap-to-public-markets-8.png)

One last public-market quirk, shown above: the **IPO pop** — the jump from the price the bankers set the night before to where the stock actually opens. The median first-day return was about 42% in 2020 and 32% in 2021. A "pop" looks like a triumph, but it is money *left on the table*: a stock that opens 42% above its IPO price means the company sold its shares 42% too cheap and the difference went to the investors who got allocation, not to the company. Pricing that gap correctly is the central drama of IPO night — and the subject of the very next post. The full mechanics of picking banks, the prospectus, the roadshow, and pricing night live in [the IPO process, end to end, from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade), and the banks' own economics in [inside an investment bank, how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).

So what? The public market is the only rung that offers *permanent* capital and *daily* liquidity to everyone at once — and it charges for that with the heaviest disclosure and the daily verdict of the price. It is the destination the entire ladder is built to reach.

## Common misconceptions

Let's correct the myths that trip people up, each with a number or a mechanism.

**"Dilution means founders lose money."** No — dilution shrinks your *percentage*, not your *dollar value*, as long as the new cash grows the company by at least what it cost. In the seed example, the founder's 80% after the round was worth exactly the same \$8 million their 100% was worth before. Founders lose money only in a **down round**, where the new round is priced *below* the previous one — then the percentage shrinks *and* the per-share value falls.

**"A higher valuation is always better."** Not for the founder. A sky-high private valuation taken with a 2x or 3x liquidation preference can leave founders with little in a mediocre exit, because the preferred gets a multiple of its money off the top first. A \$1 billion "unicorn" valuation built on aggressive preferences can be worth less to common holders than a \$600 million valuation on clean 1x terms.

**"Venture capital and a bank loan are similar — both are just money."** They are opposites. A bank loan is debt: it must be repaid with interest, the lender has no ownership, and the lender wants to be *sure they get paid back* (so they hate risk). VC is preferred equity: it is never repaid, the investor owns a slice and gets a board seat, and they *want* you to take big risks because they need a giant outcome to make their fund work. The two providers pull a company in opposite directions.

**"Going public means the founders cash out."** Mostly false at the moment of IPO. Most IPOs are primarily *primary* raises — fresh capital for the company — and insiders are usually locked up (barred from selling) for ~90–180 days afterward. The IPO is a *financing* event first and a liquidity event second; the real founder cash-out comes over the months and years after lockup expires.

**"Staying private avoids dilution and control loss."** No — late-stage private rounds dilute and impose control terms just like public capital, sometimes more (preferences, vetoes, ratchets). What staying private avoids is *disclosure* and the daily price, not the surrender of ownership and control.

## How it shows up in real markets

Let's ground all of this in real episodes, with real dates and numbers.

**The 2021 boom and 2022 freeze.** The cleanest illustration of the ladder's top rung is the whiplash we charted: US IPO proceeds of roughly \$142 billion across 397 deals in 2021, collapsing to about \$8 billion across 71 deals in 2022. What changed was not company quality but the *macro regime* — the US Federal Reserve began raising interest rates aggressively in 2022, which made the safe return on cash and bonds jump and crushed appetite for risky, unprofitable growth stocks. Companies that had raised giant private rounds at 2021 valuations were suddenly worth far less in public-market eyes, and many simply refused to IPO into that, choosing to raise another private round or cut costs and wait. The ladder's top rung is wired directly to the macro cycle. (For how policy and liquidity drive these regimes, the macro-trading series owns that thread.)

**The "unicorn" pile-up.** By the mid-2020s there were hundreds of private companies valued above \$1 billion that had *not* gone public — a direct consequence of staying private longer plus the 2022 freeze. Some had raised at 2021 valuations that the public market would no longer honor, creating a backlog of companies stuck one rung below the top, their employees holding illiquid paper marked at prices nobody could realize. This is the dark side of the staying-private trend: it works beautifully while private capital is cheap and the IPO window is open, and it traps companies when both reverse.

**Employee liquidity without an IPO.** Companies like Stripe and SpaceX became famous for running large, recurring **tender offers** — organized sales letting employees sell a slice of their vested shares to incoming investors — precisely so they could stay private for a decade-plus without their early employees revolting over locked-up wealth. SpaceX's recurring tenders have let employees sell at steadily rising valuations year after year, substituting for the liquidity an IPO would have provided. This is the employee-liquidity secondary market doing the job the public market used to monopolize.

**The down-round and the cleanup.** When a company that raised at a 2021 peak finally had to raise again in 2023–2024, it often did so at a lower valuation — a **down round** — which triggered **anti-dilution** protections in earlier preferred stock (provisions that issue extra shares to old investors to compensate for the lower price), savaging founder and employee ownership. Some companies instead did a "structured" round with heavy preferences (2x or 3x) to *avoid* officially lowering the headline valuation — preserving the unicorn label while quietly handing investors a much better real claim. The lesson: when the ladder's higher rungs get harder, the *terms* deteriorate even when the *valuation* doesn't, and the damage lands on common holders.

**Why the window opens and shuts.** It is worth being precise about the macro mechanism, because it governs the top two rungs. When central banks hold interest rates low, the safe return on cash and government bonds is meager, so investors are pushed out along the risk curve in search of return — they will pay up for risky growth equity, and the IPO window swings wide open. When rates rise sharply, the safe return jumps, the risk curve steepens, and the same investors retreat to bonds; appetite for unprofitable growth stories evaporates and the window slams. The 2021-to-2022 collapse we charted was this exact mechanism: a low-rate world that lavished \$142 billion on US IPOs flipping to a high-rate world that managed only \$8 billion the next year. A company cannot control the window; it can only be ready to step through it when it is open, which is why so much of late-stage strategy is about *optionality* — keeping enough private runway to wait out a closed window without being forced to raise on terrible terms.

**Vietnam's thinner ladder.** Not every market has all six rungs in full working order. In Vietnam, the early rungs (angels, local VC) and the public market both exist, but the *middle* — deep late-stage private capital — is thinner, so companies often jump more abruptly toward a listing on the Ho Chi Minh exchange (HOSE). The public rung itself has grown dramatically: the number of securities trading accounts climbed from about 2.2 million in 2018 to over 9 million by 2024, deepening the pool of capital available at the top of the ladder. A thinner middle means the ladder's rungs are spaced further apart, and companies feel the jump between them more sharply — a reminder that the ladder we've drawn is the *mature-market* version, and real markets vary.

## The takeaway: the ladder is a slide down the cost of capital

Step back and the whole climb resolves into one elegant idea. **The financing ladder is a machine for de-risking a company in the eyes of capital, one rung at a time — and the price of money falls as the risk falls.**

At the bottom, the founder's own money is the only capital willing to back an unproven idea, and it is "expensive" in the sense that it is scarce and slow. Each rung up, the company has proven a little more — a product, then customers, then a growth engine, then profits — and that proof attracts a wider pool of capital demanding a lower return for the lower risk. By the top, the public market will fund the company at a cost only a few points above the risk-free rate, because the company has demonstrably survived everything below. The expensive seed investor wanted to own a fifth of the company; the cheap public bond buyer accepts a 5% coupon and no ownership at all. That slide — from 10x-seeking venture preferred to single-digit-yielding public debt — *is* the ladder.

#### Worked example: the same dollar at the top and bottom of the ladder

Consider what a single dollar of capital "costs" at the two ends. A seed investor putting \$1.00 into a company at a \$10,000,000 post-money valuation, hoping for a 30x return on the bets that survive, is implicitly demanding that their dollar become \$30.00 — an enormous expected return that compensates for the bets that go to zero. Now take the same company a decade later, public and investment-grade, issuing a 10-year bond. If the risk-free 10-year Treasury yields 4% and the company borrows at a 1.5-point spread, it pays `4% + 1.5% = 5.5%` a year in interest, and the lender expects to get their \$1.00 back at maturity plus those coupons — a low-single-digit return, no ownership, no upside. The same dollar that cost the company a *fifth of its equity* at the seed stage now costs it `5.5 cents` a year and nothing else. The intuition: climbing the ladder is, dollar for dollar, the cheapest thing a successful company ever does — it converts proof of survival directly into a lower price for money.

And every rung is the same trade in different proportions: **more capital, at a lower price, in exchange for more control surrendered and more disclosure accepted.** The founder who starts owning 100% of an invisible private company ends owning perhaps a fifth of a visible public one — but that fifth is liquid, daily-priced, and worth vastly more in dollars. Whether that trade is worth making, and when, is the central judgment of building a company.

This is exactly why our series insists that **secondary-market liquidity is what makes primary issuance possible**. Every investor on every rung — the angel, the VC, the crossover fund, the public buyer — wrote their check only because they could see a path to selling their claim: to the next round, to an acquirer, to a tender offer, and ultimately to the deepest buyer of all, the public secondary market. The whole ladder exists to carry a company up to the rung where that liquidity is widest and the financing is permanent. That last rung has a name and a process all its own — the IPO. We've set it up; now let's climb it.

## Further reading & cross-links

- [What is a capital market: how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) — the series intro: the savings-into-investment machine and the primary/secondary split that frames this whole ladder.
- [Debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — the fundamental fork that runs through every rung, in full detail.
- [The IPO process, end to end: from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — the next post: exactly how the last rung gets climbed, from picking banks to the opening cross.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — who underwrites the public raise, and the fee pools they fight over.
- [The yield curve explained: the most important chart in finance](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — why the Treasury yield anchors the cost of every dollar a public company borrows.

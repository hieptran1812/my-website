---
title: "The Loan-Plus-Options Deal: How Crypto Market Makers Really Get Paid"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A plain-English, worked-arithmetic tour of the crypto-native market-making contract — how a project lends a market maker its own tokens and pays the desk in call options, why that makes the desk profit from the launch it is quoting, and when the same structure flips into a dump on retail."
tags: ["crypto", "market-maker", "market-making", "call-option", "token-loan", "strike-price", "tranches", "crypto-players", "liquidity", "options", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 32
---

> [!important]
> **TL;DR** — The most common way a crypto project pays its market maker is not a fee. The project *lends* the desk a pile of its own tokens to quote with and pays it in **call options** on those tokens — so the desk's paycheck is the token going *up*, on the very launch it is making markets for.
>
> - A **market maker (MM)** is a firm that continuously posts a price to buy (the *bid*) and a price to sell (the *ask*) so you can always trade; it earns the gap between them, the *spread*. A brand-new token needs one or it barely trades at all.
> - Under the **loan-plus-call model**, the project lends the MM tokens — often **1–2% of supply**, on a **12–24 month** term — and instead of cash the MM keeps **call options**: the right to buy those tokens at a fixed *strike* price later. If price rises above the strike, the desk pockets the difference. This is the standard early-stage structure ([Caladan](https://caladan.xyz/retainer-vs-options/); [Flowdesk](https://www.flowdesk.co/insights/crypto-market-making-retainer-vs-loan-call-model)).
> - The alternative is the **retainer model**: the project pays a monthly cash fee (roughly **\$3,000–7,000 per exchange per month**, plus a performance cut) and keeps its tokens and all the upside ([Caladan](https://caladan.xyz/retainer-vs-options/)). Cash out of pocket, but no one is short your token.
> - The structure is genuinely double-edged. The loaned tokens are a **short position** (the desk owes them back); the call is a **long position** above the strike. Where the strike sits versus today's price decides whether the desk wants price *up* (support the book) or *down* (dump the loan and buy it back cheaper).
> - The one thing to remember: when a fresh token has a paid market maker, someone on the other side of the launch may be **structurally rooting against you** — or, worse, may be paid to sell into the crowd. The Movement (MOVE) episode of 2024–25 is the cautionary case, and every number in it is sourced below.

Here is a sentence that should make you stop scrolling: the firm hired to make sure a new token *has* a fair price is very often paid in a bet that the token's price goes *up* — using the project's own coins as the chips. Not a salary. Not a flat fee. A stack of **call options** whose value comes entirely from the launch succeeding. The market maker and the launch are, by contract, on the same side of the trade. Until, sometimes, they are violently not.

Most people picture a market maker as a neutral utility — a kind of plumbing that quietly keeps the pipes full of liquidity. That picture is not wrong, but it hides the deal. Behind almost every freshly listed token is a private contract that says who lends what, who gets paid how, and — the part nobody advertises — which direction the desk secretly needs the price to move to collect. Understanding that contract tells you more about the first few months of a token's life than any candlestick chart ever will.

The diagram below is the whole article in one picture. A token project hands a market maker two things — a loan of its own tokens and some quote-currency cash — and the desk does two jobs with them: it quotes both sides of the order book, and at the end it returns the loan and *keeps the call options* that pay off if the price climbed. Everything that follows is a slow walk around this picture, built from zero, with the arithmetic shown at every step.

![A project lends the market maker its own tokens plus cash; the desk quotes the book, returns the loan, and keeps call options that pay if price rises](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-1.webp)

This post is one stop on a larger map. If you want the whole cast of funds and trading firms first, start with [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers), the hub for this series, and with [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), which builds the spread-and-inventory machine this article assumes. Here we zoom all the way into the *contract* — the single most important mechanic in the series — and answer the question underneath it: how does the desk really get paid, and how does the way it gets paid end up in the price you pay? The step-by-step dollar walkthroughs use round, made-up numbers on purpose. Every real-world figure — deal sizes, fee ranges, the Movement episode — is sourced and dated.

## Foundations: the building blocks

Before we can read the contract, we need a shared vocabulary. Read this section even if a couple of terms feel familiar, because the entire argument later hinges on the *precise* meaning of "strike", "call option", and "token loan". A practitioner can skim; a beginner should not skip.

### What a market maker is, in one paragraph

A **market maker** is a firm that stands in the market all day posting two prices: a **bid** (the price at which it will *buy* from you) and an **ask** (the price at which it will *sell* to you). The tiny gap between them is the **bid-ask spread**, and capturing that spread, over and over, thousands of times a day, is how a market maker earns its keep. The reason a new token needs one is brutal arithmetic: on day one there is almost no natural buyer sitting next to every natural seller, so without a desk continuously quoting, the order book is a ghost town and the price whipsaws on every small trade. The **inventory** is simply the pile of tokens and cash the desk holds to service both sides. That is the whole job; we cover its mechanics in depth in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does). What this article adds is the part that comes *before* the desk quotes anything: how it gets paid to show up.

### What a call option is — built from scratch

You do not need any options background for this, so let's build the idea from zero.

A **call option** is a contract that gives its holder the *right, but not the obligation*, to **buy** something at a fixed price on or before a certain date. Three words carry all the meaning:

- The **strike** (or *strike price*) is the fixed price you're allowed to buy at. Say \$1.50.
- The **premium** is what you pay up front to own the option. Say \$0.10.
- The **expiry** is the deadline. Say twelve months from now.

Now watch how it pays. At expiry, you look at the token's actual market price and ask one question: *is it above my strike?*

- If the price is **below** the strike, your right to buy at \$1.50 is worthless — why buy at \$1.50 when the market sells it cheaper? You let the option expire. You lose only the \$0.10 you paid. That's it. That is the *most* you can lose.
- If the price is **above** the strike, you exercise: you buy at \$1.50 and could instantly sell at the higher market price. Every \$1 the price sits above your strike is \$1 of value in your pocket.

The picture below is that payoff, drawn as the market maker's actual paycheck. The green bars are gains, growing dollar-for-dollar once the price clears the strike; the small red bars are the only thing at risk — the premium — when price stays below the strike.

![A call option's per-token payoff: nothing below the $1.50 strike except the lost premium, then $1 of gain for every $1 of price above it](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-2.webp)

Two quantities fall straight out of that shape and you will use them for the rest of the article:

- **Breakeven** = strike + premium. You have not actually *made* money until the price clears the strike by more than the premium you paid. Here that is \$1.50 + \$0.10 = **\$1.60**. Below \$1.60 you are still climbing out of the hole you dug paying the premium.
- **Maximum loss** = the premium, and nothing more. Owning a call is a capped-downside, open-ended-upside bet. That asymmetry — lose a little, or win a lot — is the entire appeal, and it is exactly why paying a market maker in calls is such a clever move for a project short on cash.

#### Worked example 1: a call's payoff at several strikes

Let's make the shape concrete. Suppose the token — we'll call it **NOVA** — lists at a market price of \$1.00, and the desk holds a call on **one** NOVA token, strike \$1.50, premium \$0.10. Walk the price up:

| Price at expiry | Intrinsic value = max(0, price − \$1.50) | Net P&L = intrinsic − \$0.10 premium |
|---|---|---|
| \$1.00 | \$0.00 | **−\$0.10** (lose the premium) |
| \$1.50 | \$0.00 | **−\$0.10** (at the strike, still underwater) |
| \$1.60 | \$0.10 | **\$0.00** (breakeven) |
| \$2.00 | \$0.50 | **+\$0.40** |
| \$2.50 | \$1.00 | **+\$0.90** |
| \$3.00 | \$1.50 | **+\$1.40** |

Now flip it around, because the *strike itself* is the thing the desk negotiates. Fix the price at expiry at \$2.00 and ask what one call is worth at three different strikes — the exact strikes ($1.20, $1.50, $2.00) a real desk and project haggle over:

| Strike | Payoff per token at \$2.00 | On a 1,000,000-token loan |
|---|---|---|
| \$1.20 | \$0.80 | **\$800,000** |
| \$1.50 | \$0.50 | **\$500,000** |
| \$2.00 | \$0.00 | **\$0** (at the money — nothing yet) |

The lesson in one sentence: **a lower strike is worth far more to the market maker**, which is precisely why a project that doesn't understand option pricing can hand away a fortune by agreeing to a strike that looks harmless. Desks have reportedly negotiated loans of 4–5% of a token's circulating supply at strikes so low the options were deep in the money on day one ([crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)).

### Token loan, notional, and tranches

Three last terms and we can read the whole contract.

- A **token loan** is exactly what it sounds like: the project lends the market maker a quantity of its own tokens for the desk to quote with, on the promise the desk returns the *same number* of tokens later. Crucially, borrowing 1,000,000 tokens and owing 1,000,000 tokens back is economically a **short position** — the desk is on the hook for those coins no matter what the price does in between.
- **Notional** is the total face value a position controls: 1,000,000 tokens at \$1.00 is a \$1,000,000 notional. It's the number you multiply a per-token payoff by to get the real dollars.
- **Tranches** are slices. Instead of one option on the whole loan at one strike, the deal is often split into several tranches with **rising strikes that start at different times** — tranche 1 at a low strike now, tranche 2 at a higher strike in a few months, and so on. We'll see why that protects the project shortly.

That is the entire toolkit. Now let's read the two contracts a project can actually sign.

## 1. The loan model: borrow the tokens, keep the upside

The **loan-plus-call model** (also called the *loan/option* model) is the most common arrangement for early-stage token projects, and once you see it you'll understand why ([Flowdesk](https://www.flowdesk.co/insights/crypto-market-making-retainer-vs-loan-call-model); [Caladan](https://caladan.xyz/retainer-vs-options/)). It solves a real problem: a young project has lots of tokens but very little cash, and it needs liquidity *today*. So it pays in the one currency it has plenty of — its own tokens — structured as options so the desk shares in the upside.

Here is the shape of the deal. The project lends the desk a chunk of tokens (say 1,000,000 NOVA) plus some quote-currency cash (stablecoins like USDT or USDC, so the desk can post bids). The desk uses that inventory to quote both sides of the book for a year or two. At the end, it returns the same number of tokens — and it keeps the call options it was granted as payment. A single legal document usually governs both instruments at once: a *token-loan receivable* and a *call option*, bolted together ([tech accounting](https://blog.techaccountingpro.com/p/accounting-for-market-making-arrangements)).

The diagram below is the whole life of that loan drawn on a timeline. Notice the two things happening in parallel: the **round trip** of the tokens (borrow the same number you return) and the **kept option** (the payoff that survives after the tokens go home).

![The token loan is a round trip — borrow 1,000,000 tokens, quote for 18 months, return 1,000,000 tokens — with a call option kept on top that pays the price upside](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-3.webp)

The elegant trick, and the one that trips up newcomers, is how the loan and the option are usually settled together. At maturity the desk has a choice baked into the contract: **return the tokens, or settle the loan in cash at the strike and keep the tokens.** Those two options are worth exactly the payoff of the call. Let's do the arithmetic.

#### Worked example 2: the loan round trip

The project lends the desk **1,000,000 NOVA** at listing, when NOVA is \$1.00, so the loan is a \$1,000,000 notional. Say that's 2% of a 50,000,000-token supply. The desk quotes for 18 months. Its call options let it "buy" the loaned tokens at a **\$1.50 strike**. Walk two endings:

- **Price ends at \$3.00.** The desk owes 1,000,000 tokens, now worth \$3,000,000. Rather than buy them in the open market, it exercises: it settles the loan at the \$1.50 strike, paying 1,000,000 × \$1.50 = **\$1,500,000**, and keeps tokens worth \$3,000,000. Net to the desk: \$3,000,000 − \$1,500,000 = **\$1,500,000**. That is identical to the call payoff, (\$3.00 − \$1.50) × 1,000,000 = \$1,500,000. The desk got paid \$1.5M — *by the token going up*.
- **Price ends at \$0.50.** The desk owes 1,000,000 tokens, now worth only \$500,000. Paying \$1.5M to keep them would be insane. So it simply **returns the 1,000,000 tokens** it borrowed. The option expires worthless. The desk's pay for the whole engagement was just the bid-ask spread it earned along the way.

One sentence to keep: **the loan is a round trip; the option is the tip — and the tip is paid only if the launch worked.** That is the honest, elegant version of this deal.

### Who deploys the cash, and who takes the risk

A detail people miss: in the loan-plus-call model the **market maker provides its own cash on the buy side.** The project lends tokens (the "sell" inventory) and usually some quote-currency to seed bids, but a serious desk is also risking its own balance sheet to support the book, absorbing sell pressure the project never funded ([Flowdesk](https://www.flowdesk.co/insights/crypto-market-making-retainer-vs-loan-call-model); [Caladan](https://caladan.xyz/retainer-vs-options/)). That is the desk's genuine contribution, and it is why the call is not pure free money — the desk is taking capital risk that the token craters while it holds inventory.

The spread it earns is not free money either, because of a hazard called **adverse selection** — the tendency for the person hitting your quote to know something you don't. When a desk posts a bid and an ask, the trades that fill are disproportionately the ones that go *against* it: informed sellers dump on its bid right before bad news, informed buyers lift its ask right before good news. The desk quotes a spread wide enough to survive being on the wrong side of those informed flows. So even the "boring" spread income is compensation for real risk, not a toll it collects for nothing. Hold that thought — it matters, because the call option changes how much of that risk the desk is willing to eat. A desk that is *long* the token through a valuable call will happily support the book on down days (it wants the price up); a desk whose option is worthless has no such reason to catch falling knives.

Now we have to size the option, because that is where honest turns into something else.

## 2. Sizing the option: strikes and tranches

Everything about how much the market maker really makes — and whether it is aligned with you or against you — lives in two numbers: how big the loan is, and where the strikes sit.

**Loan size.** A typical loan is **1–2% of the token's total value (FDV) or supply**, though it ranges higher; some desks have taken 4–5% of circulating supply ([Caladan](https://caladan.xyz/retainer-vs-options/); [crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)). That percentage matters enormously, because it is simultaneously the desk's quoting ammunition *and* the size of the short it is carrying against your token. A 5% loan on a thin float is a large hammer to leave lying around.

**Strike level.** In a fair deal the strike sits a sensible distance above today's price — reportedly **25–50% above the borrow price**, with market conditions sometimes pushing that to 60–70% ([PANews](https://www.panewslab.com/en/articles/8h8z15d3), reporting on desk practice). Set the strike there and the desk only collects if it genuinely helps the token appreciate. Set it too *low* and you've handed away deep-in-the-money options for free. Set it absurdly *high* — 5× or 10× spot — and the option becomes decoration, which, as we'll see in section 4, quietly turns the desk into a seller. The strike is not a detail; it is the whole incentive.

**Term.** Contracts typically run **12–24 months**, with a rough floor around three months, because it takes time to build real liquidity and nobody benefits from a desk that quotes for six weeks and leaves ([Caladan](https://caladan.xyz/retainer-vs-options/)).

**Tranches.** A well-advised project does not grant one giant option at one strike. It splits the grant into **tranches** — a ladder of rising strikes that switch on at different times. The figure below shows the idea: an early slice at a low strike, then higher strikes that begin only later. If the token rips upward in month one, the project hasn't pre-sold *all* of its upside at a bargain strike; the higher tranches are still out of reach, protecting the treasury.

![Tranches ladder the strike upward and start each slice later, so a fast run-up doesn't let the desk capture all the upside at a single cheap strike](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-4.webp)

#### Worked example 3: how much the desk actually collects

Take the same 1,000,000-NOVA loan, but structure the option as three equal tranches of ~333,333 tokens each, at strikes \$1.20, \$1.50, and \$2.00. Suppose NOVA settles at \$2.50 at expiry. Tranche by tranche:

- Tranche 1, strike \$1.20: (\$2.50 − \$1.20) × 333,333 = \$1.30 × 333,333 ≈ **\$433,000**
- Tranche 2, strike \$1.50: (\$2.50 − \$1.50) × 333,333 = \$1.00 × 333,333 ≈ **\$333,000**
- Tranche 3, strike \$2.00: (\$2.50 − \$2.00) × 333,333 = \$0.50 × 333,333 ≈ **\$167,000**
- **Total to the desk ≈ \$933,000.**

Now compare that to a naive single-tranche deal at one \$1.20 strike on the whole 1,000,000: (\$2.50 − \$1.20) × 1,000,000 = **\$1,300,000**. The tranching saved the project roughly \$370,000 of its own upside for the same market-making service. Same desk, same token, same price path — **\$933,000 versus \$1,300,000**, decided entirely by how the option was sized. That gap is why the contract, not the candlestick, is where the money is made.

## 3. The retainer model, and what each model really costs

There is a second way to pay a market maker, and it has been gaining ground precisely because projects grew wary of the loan model's hidden costs. It is the **retainer model** (sometimes "market-making as a service").

Under a retainer, the project pays the desk a straightforward **monthly cash fee** and keeps everything else. The project still provides the working inventory — tokens *and* the quote-currency cash — but as a plain loan it gets back in full, and it defines the trading strategy alongside the desk. The MM is not granted options and takes no directional bet; it "only makes money off the fee, regardless of the performance of the token" ([Caladan](https://caladan.xyz/retainer-vs-options/); [Flowdesk](https://www.flowdesk.co/insights/crypto-market-making-retainer-vs-loan-call-model)). Reported 2024-era pricing runs around **\$3,000–\$7,000 per exchange per month**, often with a first-exchange rate plus cheaper add-ons, and a performance component of **15–20%** ([Caladan](https://caladan.xyz/retainer-vs-options/)).

The table below lays the two models side by side. Read the colors from the *project's* point of view: green is good for the project, red is a cost the project bears.

![Retainer versus loan-plus-call: the retainer costs predictable monthly cash and keeps all upside; the loan model costs about zero upfront but hands the desk a call on the token's rise](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-5.webp)

The trade is now legible. The retainer is **predictable cash out the door** but keeps your token supply and your upside intact, and — the quiet part — nobody is short your coin. The loan model is **almost free upfront** but sells a slice of your success, and it puts a large short position in the hands of a professional trader. Which is cheaper depends entirely on what the token does next.

#### Worked example 4: retainer versus loan, two price paths

Same launch: 3 exchanges, an 18-month engagement. Compare the two contracts under two futures.

**The retainer.** \$5,000 per exchange per month × 3 exchanges × 18 months = **\$270,000** in cash, plus a modest performance cut. The project keeps 100% of its tokens and 100% of the upside. This number is fixed no matter what NOVA does.

**The loan-plus-call.** \$0 cash upfront. The project lends 1,000,000 NOVA and grants a call at a \$1.50 strike on those tokens.

- **If NOVA stays flat at \$1.00 through expiry:** the call finishes below its \$1.50 strike and is worthless. The loan model cost the project essentially **\$0**, while the retainer cost \$270,000. *Here the loan model wins for the project.*
- **If NOVA triples to \$3.00:** the desk's call pays (\$3.00 − \$1.50) × 1,000,000 = **\$1,500,000**, all of it value transferred out of the project's own token upside. The retainer would still have cost only \$270,000, with the project keeping the full \$3,000,000 of appreciated tokens. *Here the retainer was cheaper by about \$1,230,000.*

The one-sentence intuition: **the loan model is a fee that scales with your own success.** It feels free because you pay nothing when you launch — and you pay the most exactly when things go best. A cash-rich project that believes in its token often prefers the retainer for that reason; a cash-poor project takes the loan because it has no choice. Neither is a scam; they are different bets. The danger is not the loan model itself — it is not understanding which bet you signed.

## 4. Which way does the desk want price to go?

Now the deep part, and the reason this deal deserves a whole article. A market maker in a loan-plus-call deal is holding **two opposite positions at once**, and which one dominates decides whether it is quietly on your side or quietly against you.

- The **token loan** is a **short**: the desk owes 1,000,000 tokens back, so falling prices *help* it (it can buy them back cheaper).
- The **call option** is a **long** above the strike: rising prices *help* it (the option pays).

Whether the desk is net-long or net-short comes down to one thing — where the strike sits relative to today's price. Options people measure this with **delta**: roughly, how many tokens of real exposure the option gives you. A call far out of the money has a delta near zero (it barely moves with price); a call near or in the money has a delta approaching one (it moves almost like the token itself). So the desk's *net* token exposure is approximately **(call delta × loan size) − loan size**. The diagram below shows the two regimes that fall out of that arithmetic.

![Where the strike sits flips the desk: a strike near spot makes it net long and want price up; a strike far out of reach leaves it holding only the loan — a short that wants price down](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-6.webp)

#### Worked example 5: the strike is the switch

Loan of 1,000,000 NOVA, spot \$1.00. Two strikes, two completely different desks.

- **Aligned: strike \$1.50 (near spot).** The call has real value and a meaningful delta. If price climbs above \$1.50, the desk keeps (price − \$1.50) × 1,000,000 — it is *long the upside*, and its whole incentive is to quote tightly, deepen the book, and let the token appreciate. Every \$1 above \$1.50 is +\$1,000,000 to the desk. It **wants price up**, which is the outcome the project and holders want too. For a while, everyone rows the same way.
- **Predatory: strike \$10 (10× spot).** The call needs a moonshot to matter; its delta is near zero, so it is effectively decoration. Strip it away and the desk's real position is just the **short** from the loan. Now the profitable play is ugly: sell the 1,000,000 loaned tokens today near \$1.00 (collecting ~\$1,000,000 in cash), let — or help — the price fall to, say, \$0.50, buy 1,000,000 back for ~\$500,000, return the loan, and pocket roughly **\$500,000**. It **wants price down**, and it is holding exactly the ammunition to push it there.

The one-sentence intuition: **a strike near today's price makes the desk your partner; a strike it can never reach makes the desk your counterparty.** This is why an out-of-the-money strike — which sounds *safer* for a project, since the desk "only wins if we moon" — is often the more dangerous choice. It removes the desk's stake in the upside and leaves it holding a loaded short. When you read that market makers "hedge" a mispriced deal by shorting the token to lock in riskless profit, this is the machinery ([crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)).

### Gamma: the desk trades around the option, too

There is one more layer, and it is where the desk's edge really lives. Holding a call doesn't just give a static payoff at expiry; it gives the desk a position whose *sensitivity* to price changes as price moves — the option's **gamma**. In plain terms: as the token rises toward and through the strike, the desk's effective long exposure grows; as it falls away, that exposure shrinks. A sophisticated desk continuously re-hedges against that shifting exposure — buying a little as price falls, selling a little as it rises — and captures small profits from the churn. Caladan describes the loan/call desk as compensated "through call option and gamma trading of the option," not the option alone ([Caladan](https://caladan.xyz/retainer-vs-options/)). You do not need the calculus. The takeaway is that the call is not a lottery ticket the desk buys and forgets; it is a live position it actively trades, which is exactly why professional desks are so much better at extracting value from these contracts than the product teams who sign them. The information asymmetry is real: market makers are derivatives professionals, and founders are builders ([crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)).

## 5. How it shows up in the price you see

None of this would matter to you if it stayed on paper. It doesn't — it shows up on the tape, and it shows up worst on exactly the days retail is most excited: listing day and the days right after.

The reason is **float**. A new token usually launches with only a small fraction of its supply actually tradable — the *free float* — while the rest is locked in vesting schedules (we unpack this in [the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto)). A market maker holding a loan of 1–5% of *supply* can therefore be holding a huge fraction of the *tradable float*. Selling that into a thin book moves price hard.

This is the single most important number the headlines skip. "The desk borrowed 2% of supply" sounds tiny. But if only 10% of supply is actually circulating on day one, that 2%-of-supply loan is **20% of the float** — a fifth of everything that can trade, in one professional's hands. The mechanics of how a small notional moves a large headline price in a thin book — resting liquidity, slippage, and the reflexive feedback that follows — are the subject of [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move); here we only need the punchline: **loan-as-a-share-of-float, not loan-as-a-share-of-supply, is what determines how much a desk can shove the price around.**

#### Worked example 6: a dump into a thin float

Suppose NOVA launches with a free float of just **5,000,000 tokens** actually circulating, and the desk holds a **1,000,000-token** loan. If the desk decides its best play is the short (a far-out-of-the-money strike, as in the predatory case above), and it sells that whole loan into the market, it is dumping **1,000,000 into a 5,000,000 float — 20% of everything tradable — in a compressed window.** There is nowhere near enough resting bid to absorb that, so the price gaps down through level after level of the order book. Retail, watching a fresh listing crater, either panic-sells into the same hole or catches the falling knife. Then the desk buys its 1,000,000 back at the depressed price, returns the loan, and keeps the difference. The same thin float that made the token look excitingly volatile on the way up is what makes the dump so violent on the way down. Mechanically, this is the launch-day rug in slow motion — and it is legal, contractual, and often invisible until the wallets are traced after the fact.

And the dump does not happen in a vacuum — it is **reflexive**. As the price gaps down, it trips stop-loss orders and liquidates over-leveraged longs, which throws *more* sell orders into an already-thin book, which drops the price further, which trips more stops. A 20%-of-float sale can cascade into a move far larger than the sale alone would suggest, because the market's own plumbing amplifies it. The desk does not need to sell all day; it needs to start the avalanche and let leverage finish the job, then buy the bottom. That reflexive amplification is why launch-day charts so often look like a cliff rather than a slope.

That is the honest cost of the loan model when the incentive is wrong: your token's first impression — the price chart a whole cohort of new holders will anchor to forever — can be set by a desk that is paid to sell it.

## Common misconceptions

**"The market maker is neutral — it just provides liquidity."** In a retainer deal, roughly true: the desk earns a fixed fee and has no directional stake. In a loan-plus-call deal it is *never* neutral. It holds a short (the loan) and a long (the call), and the net of those two is a directional position by construction. Neutral is the exception, not the default.

**"An out-of-the-money strike is safer for the project."** It feels safer — the desk only cashes in if the token soars. But a strike the desk can't reach strips out its upside stake and leaves it holding a naked short in your token. The "safe-sounding" high strike is frequently the one that turns the desk into a seller. A strike a sensible distance above spot, with tranches, keeps the desk rooting for the same outcome you are.

**"Paying in options is free money for the project."** It's *cash-free*, which is not the same as free. You are handing over call options on your own token, and options have real value — sometimes enormous value if the token runs. Worked example 4 showed a "free" loan model quietly costing \$1.5M when the token tripled. You pay the most precisely when you succeed the most.

**"If the desk holds a loan, it's basically an investor and wants us to win."** Only if the option, not the loan, is the dominant leg. A desk whose option is worthless is an investor in your *decline*. Whether "aligned" is true is a question you answer by looking at the strike, not the marketing.

**"Market makers set the fundamental price."** They set the *liquidity* and smooth the microstructure; they do not conjure lasting value. What they can do is dominate the price *action* in the low-float early days — the window when there's little else trading. Over a longer horizon, unlocks, real demand, and the broader market reassert themselves. The desk's power is concentrated exactly where new holders are most impressionable.

**"The desk returns the exact same number of tokens, so the project loses nothing on the loan."** True on the loan leg — and irrelevant, because the cost was never the loan. The cost is the *option* granted alongside it. The project gets its 1,000,000 tokens back, and separately hands the desk the right to profit from the price rise. Counting only the returned tokens and ignoring the option is exactly the accounting mistake that makes the loan model feel free.

**"This is a fringe, shady corner of crypto."** The loan-plus-call structure is the *mainstream* early-stage arrangement, used by reputable desks and sketchy ones alike. The leading market makers commonly named in the space include Wintermute, GSR, Keyrock, Flowdesk, Amber Group, and Jump ([Cryptic](https://crypticweb3.com/best-crypto-market-makers-in-2026/)). The structure is not the problem; the *mispricing and opacity* around it are.

## How it shows up in real markets

### 1. Movement (MOVE): the deal that paid the desk to dump

This is the case that dragged the whole structure into daylight, and every figure here is sourced. In late 2024 the Movement (MOVE) token was preparing to list. A market maker, the Chinese firm **Web3Port**, operating through a middleman entity called **Rentech**, ended up controlling roughly **66 million MOVE — about 5% of supply, and around half of the token's publicly held float** ([CoinDesk](https://www.coindesk.com/tech/2025/04/30/inside-movement-s-token-dump-scandal-secret-contracts-shadow-advisors-and-hidden-middlemen)). The final contract was signed on December 8, 2024; MOVE debuted December 9.

The structure is the tell. According to CoinDesk's reporting on the internal contracts, the deal contained a provision letting the desk liquidate if MOVE's **fully diluted value (FDV) topped \$5 billion**, with a **50-50 profit split**. As one adviser quoted in the investigation put it, that created "incentives basically to manipulate the price to over \$5 billion fully diluted value and then dump on retail for shared profit" ([CoinDesk](https://www.coindesk.com/tech/2025/04/30/inside-movement-s-token-dump-scandal-secret-contracts-shadow-advisors-and-hidden-middlemen)). Pump above a threshold, then split the proceeds of selling into the crowd — the incentive was written into the paper.

The timeline below is what happened next.

![The MOVE episode: a deal that let the desk sell above a $5B FDV, a ~$38M dump the day after launch, and Binance freezing the funds months later](/imgs/blogs/the-loan-plus-options-deal-how-market-makers-get-paid-7.webp)

Within roughly a day of the debut, wallets linked to the market maker unloaded about **\$38 million** of MOVE into the open market, triggering a steep drop ([CoinDesk](https://www.coindesk.com/markets/2025/05/17/movement-labs-and-mantra-scandal-are-shaking-up-crypto-market-making); [The Block](https://www.theblock.co/post/347931/binance-move-market-maker-movement-38-million-usdt-buyback-program)). Binance later identified and **banned the market-making account for misconduct**, froze roughly **\$38 million**, and Movement committed to a buyback program using the recovered funds ([The Block](https://www.theblock.co/post/347931/binance-move-market-maker-movement-38-million-usdt-buyback-program)). Coinbase suspended MOVE from trading on May 15, 2025 ([CoinDesk](https://www.coindesk.com/markets/2025/05/17/movement-labs-and-mantra-scandal-are-shaking-up-crypto-market-making)). Movement Labs placed co-founder **Rushi Manche** on administrative leave and commissioned a third-party investigation ([CoinDesk](https://www.coindesk.com/tech/2025/04/30/inside-movement-s-token-dump-scandal-secret-contracts-shadow-advisors-and-hidden-middlemen)). The lesson is not that market makers are villains; it is that a *mispriced structure with a hidden clause* turns the desk into a seller against the very holders it was supposed to serve — and that the terms live in contracts "negotiated in the shadows," with no standardized disclosure ([crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)).

### 2. The delta-neutral desk that does it right

The mirror image is worth seeing, because most market making is not predatory. Established desks frequently run **delta-neutral**: they hedge the token exposure from their inventory (using perpetual futures and the like) so they are *not* making a directional bet on the token at all, and profit from the spread and flow instead. Wintermute, for instance, has been described as running a delta-neutral approach that hedges out token volatility rather than betting on direction ([PANews](https://www.panewslab.com/en/articles/8h8z15d3)). A desk in a fair loan-plus-call deal with a sensible strike, hedging its residual exposure, genuinely earns its keep by making the book tighter and deeper. The structure works fine; it is the *combination* of a worthless strike and an unhedged short that curdles.

### 3. The transparency backlash

The MOVE episode and others like it (the OM/Mantra collapse the same season) triggered a live debate across the industry about disclosure. Because there are "no industry benchmarks" and "no standardized disclosures" for these agreements, projects and their investors often cannot even verify how active their own market maker is, or whether the strike they agreed to was fair ([crypto.news](https://crypto.news/broken-market-making-deals-derail-promising-projects/)). Some desks have begun publicly calling for more transparency in market-making terms as a competitive differentiator. Whether that becomes the norm is unsettled — but the fact that it is now a *sales pitch* tells you how mispriced the opacity had become.

### 4. The quiet migration toward retainers

The most durable effect of the 2024–25 blowups was not a rule; it was a shift in what informed projects ask for. As founders learned what a mispriced call actually costs, the **retainer model** — cash fee, no options, no one short your token — became a more common request, especially from better-capitalized teams that could afford to keep their upside. The trade-off never went away: a retainer is real cash out of a treasury that often does not have much, which is exactly why cash-poor early projects still reach for the loan model. But the negotiating table changed. A team that walks in knowing the difference between a strike near spot and a strike out of reach, and that asks for tranches instead of one cheap grant, is a far harder counterparty than one that signs whatever the desk drafts. The structure did not get safer; the projects got smarter. That is usually how these things improve in crypto — not by regulation arriving, but by an expensive lesson propagating through the founder network until the naive version of the deal stops getting signed.

## When this matters to you

If you never touch a freshly listed token, this is background knowledge about how the sausage is made. If you *do* — and most crypto participants eventually chase a new listing — it is a direct defense. Here is the practical checklist the structure hands you.

- **Assume a paid market maker exists on any new listing, and ask which model.** You usually can't see the contract, but you can reason about incentives. A loan-plus-call deal means someone is short your token by design.
- **Respect the float.** A token where a large share of supply is locked, leaving a thin tradable float, is a token where a single desk's loan can move price violently. Low float plus a market-maker loan is a recipe for a launch-day air pocket. Cross-reference the [cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) if the project publishes one.
- **Be most skeptical exactly when it's most exciting.** The desk's power over price is concentrated in the low-liquidity first days — the same days a hype-driven crowd is buying. The launch candle you're anchoring to may be manufactured.
- **Watch for the tells of a wrong-way incentive.** A brand-new token that runs hard on day one and then bleeds relentlessly, on volume that doesn't match organic interest, is the visual signature of a desk working a short. It is not proof — but it is the pattern the arithmetic predicts.
- **Understand the deeper conflict.** The market maker sits inside a web of aligned and misaligned incentives that runs through the whole crypto stack — VCs, exchanges, foundations, and you. Map the whole thing with [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) and [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).

The point is not paranoia. Market makers are a genuine, necessary service; without them a new token barely trades. The point is *literacy*: to know that "the market maker" is not neutral plumbing but a paid counterparty with a contract, and that the shape of that contract — a loan, a strike, a tranche schedule — quietly decides whether the professional on the other side of your first trade is rooting for you or feeding on you. This is educational, not financial advice. But knowing which side the strike puts them on is the difference between reading the tape and being the exit liquidity in someone else's payoff diagram.

## Sources & further reading

Primary sources behind the figures and the case study:

- [Crypto Market Making: Retainer vs. Loan/Call Model](https://www.flowdesk.co/insights/crypto-market-making-retainer-vs-loan-call-model) — Flowdesk (market maker), on how each model works and who provides capital.
- [Retainer vs Options Model for Market-Making](https://caladan.xyz/retainer-vs-options/) — Caladan (market maker), for loan sizes (1–2% of FDV), 12–24 month terms, tranche structure, and 2024 retainer pricing.
- [How Market-Making Deals Work In Crypto](https://www.blocmates.com/articles/how-market-making-deals-work-in-crypto) — blocmates, a worked walkthrough of the loan/option structure.
- [Broken market-making deals are derailing promising projects](https://crypto.news/broken-market-making-deals-derail-promising-projects/) — crypto.news, on mispriced strikes, the hedging-into-a-short pattern, and the transparency gap.
- [Analyzing Wintermute's market making method](https://www.panewslab.com/en/articles/8h8z15d3) — PANews, on strike levels (25–50% over borrow) and delta-neutral hedging.
- [Inside Movement's Token-Dump Scandal](https://www.coindesk.com/tech/2025/04/30/inside-movement-s-token-dump-scandal-secret-contracts-shadow-advisors-and-hidden-middlemen) — CoinDesk (Apr 30, 2025), the internal contracts, the \$5B FDV clause, Web3Port/Rentech, and personnel actions.
- [Binance identifies alleged MOVE-dumping market maker; \$38M buyback](https://www.theblock.co/post/347931/binance-move-market-maker-movement-38-million-usdt-buyback-program) — The Block, on Binance's freeze and Movement's buyback.
- [Mantra and Movement token scandals are shaking up crypto market-making](https://www.coindesk.com/markets/2025/05/17/movement-labs-and-mantra-scandal-are-shaking-up-crypto-market-making) — CoinDesk (May 17, 2025), the Coinbase suspension and industry fallout.

Related on this blog:

- [Crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub and the whole cast of players.
- [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — the spread-and-inventory mechanics this article builds on.
- [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — who profits at each step, and why launch-day retail collides with insiders.
- [The crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) — how token funds get paid, the sibling deal to this one.
- [Calls, puts, and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) — the full options-payoff toolkit, if you want to go deeper on the call we built from scratch.

---
title: "Designated vs Principal Market Making: The Two Hats and the Conflict"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A crypto trading firm can be hired to quote your token's order book and, at the same time, trade its own capital against the flow it sees — this is how the two hats work, why the conflict is so sharp, and how TradFi keeps them apart."
tags: ["crypto", "market-maker", "designated-market-maker", "proprietary-trading", "conflict-of-interest", "order-flow", "best-execution", "volcker-rule", "crypto-players", "retail-defense", "market-microstructure"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A crypto trading firm can wear two hats at once: a *designated* market maker paid to quote a project's order book (agency-like), and a *principal* trader betting its own capital for its own profit. The sharp conflict is that one desk can do both — it quotes your book, it sees your flow, and it holds the token's upside — all inside one legal entity. TradFi spent decades building walls to stop exactly this; crypto ships without them.
>
> - **Agency vs principal is the whole game.** An *agent* trades your money in your interest for a fee. A *principal* trades its own money in its own interest. The same firm can legally be both to the same token at the same time.
> - **A designated market maker** is contracted to provide two-sided liquidity on a specific book — often paid a cash retainer plus a token loan and call options (the loan-plus-options deal). That last piece means it profits if the token it quotes goes up.
> - **The information advantage is the live wire.** As the desk quoting your book, it sees resting orders and the order-flow imbalance before anyone else — and it can position its own account ahead of the fill it is about to give you.
> - **Add up every seat and the arithmetic is lopsided.** In a stylized one-token year, the visible agency retainer might be **$240k** while the same firm's principal option upside is **~$10M** — roughly **40x** more, funded by the project's own token supply and the retail buyers on the other side.
> - **TradFi separates the hats by law.** The Volcker Rule (2010) bars banks from proprietary trading; an NYSE designated market maker cannot see orders before they are public; best execution is a legal duty (FINRA Rule 5310). Crypto has almost none of this by default.
> - **Retail defense:** a thin token that fills instantly at a tight spread is not a gift. It is a clue to ask who quotes it, what else they hold, and whether the venue is neutral. Assume it is not.

You place a buy order for a small-cap token. It fills in a blink, at a spread so tight you barely notice the cost. It feels like a frictionless, efficient market — the kind of liquidity a mature exchange is supposed to provide.

Now ask a question almost nobody asks: *who was on the other side of that trade?*

Very often the answer is a single firm that is (a) paid by the project to make that market, (b) trading its own capital on the same book, and (c) holding call options on the very token you just bought. Three roles, one desk, one legal entity. In the US stock market, that combination would be illegal, heavily walled off, or both. In crypto, it is the default setting.

This post is about those two hats — **designated** market making (the agency-like role, where a firm is hired to quote a book) and **principal** trading (the proprietary role, where a firm bets its own money) — and about what happens when one firm wears both at the same time. We will build every term from zero, walk four numeric examples with round dollar figures, and finish with the one thing that actually matters if you are the retail buyer on the other side of the flow.

![One firm, two hats: the order flow a firm sees while quoting your book as a designated market maker feeds the account it trades for its own profit.](/imgs/blogs/designated-versus-principal-market-making-1.webp)

The diagram above is the mental model the rest of the article tours. One legal entity splits into two roles. The designated-market-maker hat provides the two-sided quotes on your order book. The principal hat trades the firm's own capital. And the line running between them — the order flow the quoting desk *sees* — is where the whole conflict lives. Keep that picture in mind; everything below is a zoom into one of its boxes.

## Foundations: agency, principal, and the market maker

Before we can see the conflict, we need five plain-English building blocks: the difference between an agent and a principal, what a market maker actually does, what "designated" and "proprietary" mean, what "order flow" gives you, and what a conflict of interest is. If you already know these cold, skim. If you don't, you cannot follow the rest — so we go slowly.

### Agency vs principal: the one distinction that runs the whole post

There are exactly two ways a financial firm can stand in a trade, and the difference is not a technicality — it decides whose money is at risk and whose interest legally comes first.

An **agent** acts *on behalf of someone else*. When you use a stockbroker in the agency model, the broker takes your order, goes out to the market, and finds someone to trade with — using *your* money, in *your* interest, and getting paid a **fee** or **commission** (a small charge for the service, e.g. a flat ticket charge or a few cents per share). The agent's own balance sheet is not on the line. Its job is to get *you* the best deal it can. That last duty even has a legal name, which we will meet in a moment: *best execution*.

A **principal** acts *for itself*. When a firm trades as a principal — also called a **dealer**, or when it is speculating, a **proprietary** or **prop** desk — it uses *its own* capital and keeps *its own* profit and loss ("P&L", the running tally of what a book has made or lost). It buys from you because it wants the thing you are selling, or it sells to you because it wants your cash. There is nothing shady about this by itself; every market needs principals willing to take the other side. But its interest is, by construction, *its own* — not yours.

![Agency vs principal: an agent trades your money in your interest for a fee; a principal trades its own money in its own interest — and the same firm can do both.](/imgs/blogs/designated-versus-principal-market-making-2.webp)

The table above is the entire foundation of this post in one picture. Read across the two rows. The agent's capital is the client's, the agent's duty runs to the client, and the agent is paid a fee. The principal's capital is its own, its duty runs to its own P&L, and it is paid by trading profit. A firm that is *both* an agent and a principal to the same asset at the same time is holding two duties that point in opposite directions. Hold that thought — it is the whole ballgame.

> An agent is paid to serve you. A principal is paid to beat you. The conflict is what happens when one firm is both, to the same token, on the same day.

### What a market maker actually does (the 60-second version)

A **market maker** (MM) is a firm that stands ready to both *buy and sell* an asset continuously, quoting two prices at once: a **bid** (the price at which it will buy from you) and an **ask** or **offer** (the higher price at which it will sell to you). The gap between them is the **spread**. If the MM will buy at $0.99 and sell at $1.01, the spread is $0.02, or about 2%.

The MM's basic business is to earn that spread many times over: buy a little below "fair," sell a little above, and net the difference across thousands of round-trips. In exchange, it provides **liquidity** — the ability for anyone to trade *right now* without waiting for a matching counterparty to show up. Without a market maker, a thinly traded token might have buyers at $0.80 and sellers at $1.20 and simply no trades in between. The MM fills that gap and gets paid for it. (We cover the mechanics — spread capture, inventory, adverse selection — from zero in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does); here we only need the one-line version.)

Crucially, **a market maker is a principal.** When it buys from you at the bid, it uses its own money and now owns the token — that is a principal trade. So market making is, at its core, a principal activity. The twist this whole post is about is what happens when that principal activity is dressed up in an *agency-like* contract.

### Designated market making: the contracted role

A **designated market maker** is a firm that has been *formally contracted or appointed* to make a market in a specific security or token, usually with obligations attached: it must quote continuously, keep the spread inside some limit, and post a minimum size on both sides. "Designated" just means "officially assigned this book," as opposed to any random trader who happens to be quoting.

There are two flavors, and it is worth separating them:

1. **Exchange-designated.** A trading venue appoints a firm as the official liquidity provider for a listing, sometimes with rebates or fee discounts in return for quoting obligations. The classic example is the New York Stock Exchange's **Designated Market Maker** program (more on the TradFi version later). Some crypto exchanges run similar "designated liquidity provider" or MM-incentive programs.
2. **Project-designated.** A token project *hires* a market-making firm to quote its book. This is the crypto-native arrangement, and its standard structure is the **loan-plus-options deal**: the project *lends* the MM a pile of tokens to quote with, pays a cash retainer, and grants the MM **call options** (the right to buy tokens at a fixed price later) as the real incentive. We break that contract down in [the loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid); the piece that matters here is that it makes the MM *long the token* — it profits if the price rises.

The word "designated" gives the role an agency *flavor*: the firm is "hired to provide a service" to the project or the exchange, the way a broker is hired to work your order. But underneath, it is still trading as a principal. That gap — agency wrapper, principal engine — is the seam the conflict grows out of.

### Proprietary (prop) trading: your own book, your own P&L

**Proprietary trading** is a firm trading its own capital purely to make money for itself — directional bets, arbitrage, relative-value trades, anything. No client, no fee, no service. Just the firm versus the market, keeping every dollar it makes and eating every dollar it loses.

Most large crypto trading firms run a prop desk alongside whatever market-making or OTC business they have. That is not, by itself, a scandal — prop trading is a legitimate, centuries-old activity. The question this post asks is narrower and sharper: *what happens when the same firm's prop desk sits next to a designated-market-making desk that is quoting a book — and can see everything happening on it?*

### Order flow and the information advantage

**Order flow** is the stream of orders arriving at a market: who is trying to buy, who is trying to sell, how much, and at what price. To the firm quoting a book, order flow is *information* — and in a thin market, it is extremely valuable information.

Here is why. Suppose the order book for a token is thin: there are only a few hundred thousand dollars of resting orders near the current price. Now a large buy order arrives. Whoever *sees* that order first knows something the rest of the market does not yet: price is about to move up, because that buy will eat through the thin sell-side and lift the price. The firm quoting the book sees the resting orders, the size of what is arriving, and the **imbalance** (more buyers than sellers, or vice versa) before any of it is reflected in the public price.

That is the **information advantage**: knowing near-term supply and demand a beat before everyone else. In a deep, liquid market it is worth little. In a thin token — which is most tokens — it is worth a great deal, because a single order can move the price and the firm that sees it coming can get in front.

### Conflict of interest, Chinese walls, and best execution

A **conflict of interest** exists when a firm's duty to one party and its incentive for itself point in opposite directions. A designated market maker that is *also* a prop trader on the same book is the textbook case: its designated role gives it a service-like relationship with the project and a front-row view of your flow, while its principal role gives it every reason to use that view for its own P&L.

TradFi's answer to conflicts like this is the **Chinese wall** (also called an **information barrier**): an internal separation — different teams, different systems, restricted data access — meant to stop information from one side of a firm reaching the side that could trade on it. If the market-making desk cannot pass what it sees to the prop desk, the conflict is at least contained.

And the reason all of this is enforced in stock markets is a legal duty called **best execution**: the requirement that a broker seek the most favorable *reasonably available* terms for a client's order. Best execution is why a broker cannot just route your order wherever pays the broker the most; it owes *you* the best fill it can reasonably get. In US equities this duty is codified in **FINRA Rule 5310**. Hold onto best execution — its near-total *absence* in crypto is half of why the two-hats conflict is so much worse there.

That is the toolkit. Now we can build the conflict.

## The two hats, precisely

Let's state the two roles as precisely as we can, because the whole argument turns on the difference.

**The designated (agency-like) hat.** The firm has a contract — with a project, or an appointment from an exchange — to provide two-sided liquidity on a specific book. It is *paid to be there*: a retainer, fee rebates, or the loan-plus-options package. It has *obligations*: keep quoting, keep the spread tight, keep size on both sides. In the language of the trade, it is providing a *service*, and the counterparty to that service (the project, the exchange, and indirectly you) is meant to benefit from the liquidity it supplies. This is the hat that *looks* like agency.

**The principal (prop) hat.** The same firm runs a book for its own account. It takes directional views, hedges, arbitrages, and positions. It keeps its own P&L. Nobody it trades against is its client. This hat is unapologetically self-interested — as prop trading is meant to be.

Why does a firm want both? Because they feed each other, and that is exactly the problem:

- The **designated hat pays the bills and buys the seat.** The retainer and rebates cover costs; the contract gives the firm a reason to be quoting the book at all.
- The **designated hat generates information** — the order flow it sees while quoting — that is directly useful to the **principal hat.**
- The **loan-plus-options** package hands the firm the token's *upside*, so its "agency" role secretly makes it a *principal bettor* on the very asset it is supposed to neutrally quote.

So the two hats are not two separate businesses that happen to live under one roof. They are wired together. The agency wrapper produces the raw material — information and inventory — that the principal engine turns into profit. A firm that could only do one would be far less dangerous to the person on the other side of the trade.

There is a third party that quietly benefits from this arrangement: the exchange. A venue wants its listings to look liquid, because liquid markets attract traders and trading generates fees. A designated market maker that keeps a book tight and active is doing the exchange a favor too — which is one reason venues run market-maker incentive programs and rarely press hard on whether a quoting firm is also trading its own account against the flow. When the exchange is *itself* an investor in the token or the gatekeeper that approved the listing, the incentive to look the other way only compounds. The two-hats firm is not fighting the venue; more often it is doing exactly what the venue's own fee model rewards.

## The information advantage: where the conflict bites

The single sharpest edge of the conflict is informational, so let's walk exactly how it plays out on one order.

![The order lifecycle when one desk wears both hats: because the same firm sees the resting order and runs a principal book, it can position ahead of the fill it is about to give you.](/imgs/blogs/designated-versus-principal-market-making-3.webp)

Trace the pipeline above left to right. You send a buy order. It hits the book the firm is quoting as designated market maker, so the firm's designated desk *sees the resting order* — and in a thin market, sees that it is large enough to move the price. That view is a signal. The same firm's prop book can *position ahead*: buy first, before your order is filled, knowing your buying is about to lift the price. Then your order fills at the desk's quote. And the firm books both the spread *and* the edge from having front-run the move.

In equities, that last step — trading ahead of a customer order you can see — is called **front-running**, and it is illegal. The entire point of the Chinese wall is to make it impossible by keeping the order-seeing desk and the trading desk apart. In crypto, there is usually no wall, no rule, and no client relationship that front-running would violate — the firm has no *legal* duty of best execution to the anonymous trader whose order it just saw.

I want to be careful and precise here, because this is the part where it is easy to slide from "structural temptation" into "accusation." The claim is *not* that any particular named firm does this. The claim is that the **structure** — one entity that both quotes a thin book and trades its own account, with no wall and no best-execution duty — makes the edge available and unpoliced. Whether a given firm takes it is a separate, firm-by-firm question. What we can say for certain is that the incentive and the opportunity both sit inside the same building.

#### Worked example: a prop desk trading the flow it sees

Let's put round numbers on it. This is a *stylized, hypothetical* walkthrough — the figures are illustrative, chosen to be easy to follow, not a description of any real firm's book.

Suppose a token trades at $1.00 and its order book is thin: there is only about $200,000 of sell-side liquidity resting between $1.00 and $1.05. Now a large buyer sends a market order to buy 1,000,000 tokens.

1. The firm quoting the book **sees the incoming order** before it is fully filled. It knows the buy will sweep the thin sell side and push the price from about $1.00 toward $1.05.
2. On its **prop book**, it buys 200,000 tokens at an average of $1.00 *ahead* of the customer's order — using its own capital.
3. The customer's buying then lifts the price. The prop desk sells its 200,000 tokens into that strength at an average of about $1.04.
4. Profit on that one event: 200,000 × ($1.04 − $1.00) = **$8,000.**

Eight thousand dollars is trivial. But this is *one order* on *one token* on *one day*. A firm quoting dozens of thin books, seeing this pattern hundreds of times, compounds a small per-event edge into a real revenue line. The single sentence to remember: **on a thin book, the right to see the flow first is worth money, and the firm quoting the book is the one who has it.**

The "what it costs / when it breaks" note: this edge shrinks toward zero as the book gets deep. On a liquid, heavily traded token where no single order moves the price, seeing the flow first buys you almost nothing. That is why the conflict is most dangerous precisely where retail is most exposed — new, thin, low-float listings — and least dangerous on the blue-chip pairs where you probably didn't need protecting.

## Worked example: designated-MM economics

Now the other hat. Let's see how the designated role actually pays, because the *visible* fee is not where the money is.

![Designated-MM economics over one 12-month deal: the cash retainer is the small, visible fee; the token loan and the call options are where the real money is made.](/imgs/blogs/designated-versus-principal-market-making-4.webp)

The timeline above tracks the cash. Consider a stylized project-designated deal with round numbers:

- **Day 0.** The project hires the firm as its designated market maker. It pays a **cash retainer of $20,000 per month** and *lends* the firm **5,000,000 tokens** to quote with. At the listing price of $1.00, that loan is $5,000,000 of tokens the firm now holds as inventory but does not own outright — it must return them at the end of the deal.
- **Each month.** The firm keeps the retainer. Over a 12-month deal, that is 12 × $20,000 = **$240,000** in cash fees. This is the "agency" leg — the visible, invoice-able service fee.
- **At listing.** As part of the same deal, the firm holds **call options on 5,000,000 tokens at a strike of $1.00** — the right, but not the obligation, to buy those tokens at $1.00 regardless of the market price later. (This is the loan-plus-options structure; the options are the real payment.)
- **If the token 3x's.** Say the token trades at $3.00 by the time the options vest. The firm exercises: it buys 5,000,000 tokens at $1.00 (paying $5,000,000) and they are worth $15,000,000 — an option gain of 5,000,000 × ($3.00 − $1.00) = **$10,000,000.**
- **Deal end.** The firm returns the borrowed tokens and walks away having kept the $240,000 of retainer *plus* the $10,000,000 of option upside — about **$10,240,000** total.

Look at the proportions. The part that looks like a service fee — the retainer — is $240,000. The part that is really a *principal bet on the token* — the options — is $10,000,000, more than **40 times** larger. The designated market maker is not primarily being paid to provide a service. It is being paid, overwhelmingly, to be *long the very token it is supposed to neutrally quote.*

The one-sentence intuition: **a designated market maker on a loan-plus-options deal is a principal bettor wearing an agency costume — the costume is the retainer, the bet is the options.**

The "when it breaks" note: options cut both ways only in theory. If the token *falls*, the options expire worthless — but the firm still keeps the retainer, still returns the borrowed tokens, and its downside is capped. Meanwhile it spent the whole deal quoting a book it was long. The asymmetry — capped downside, uncapped upside, and a front-row seat to the flow — is the point.

## The combined-hat payoff vs the client's cost

Let's draw the payoff of the two hats together, because the shape tells the whole story.

![The combined-hat payoff: the MM's payoff is flat at the retainer below the strike, then climbs with the token it quotes — it is never neutral.](/imgs/blogs/designated-versus-principal-market-making-5.webp)

The chart plots the firm's profit (vertical axis) against the token's price at unlock (horizontal axis). Below the strike of $1.00, the payoff is *flat* at the retainer floor of about **+$0.24M** — even if the token goes to zero, the firm keeps its fee. Above the strike, the payoff *climbs* one-for-one with the token, because the firm is long 5,000,000 call options. At $3.00, the payoff is **+$10.24M**, exactly the number from the timeline. The line is the sum of an agency fee (the flat part) and a principal bet (the rising part) — which is why the firm is *never neutral* about the price of the token it makes markets in.

Now the crucial second half: **what does this cost the client?** "The client" here is the project that hired the firm, and — one hop downstream — the retail buyers on the project's book. Walk the cost:

- The project handed the firm **$240,000 in cash** (the retainer) — a real expense.
- The project granted the firm **call options on 5,000,000 tokens at $1.00.** If the token reaches $3.00, those options are worth $10,000,000 — value that came out of the project's own token supply. Every token the firm buys at $1.00 and the market values at $3.00 is $2.00 of upside the project *gave away*, and $2.00 the eventual holders did not get.
- And who buys those 5,000,000 tokens at $3.00 when the firm eventually sells them? The market — which, at a retail-heavy listing, means retail. The people buying at $3.00 are the **exit liquidity** for the tokens the firm acquired at $1.00.

So the combined-hat payoff is not a chart of value *created*. It is a chart of value *transferred* — from the project's treasury and its late buyers, into the firm's book. The firm's upside and the client's cost are two sides of the same line.

#### Worked example: the client's cost, made explicit

Put the two ledgers side by side at the $3.00 unlock, using the same numbers:

| | The firm receives | The client / market pays |
|---|---|---|
| Retainer | +$240,000 cash | −$240,000 from treasury |
| Option upside | +$10,000,000 | −$10,000,000 of token value given away |
| **Total** | **+$10,240,000** | **−$10,240,000** |

The columns are mirror images because this is a transfer, not a service with a separate output. The intuition to keep: **when you read a market-making deal, the firm's payoff line and the client's cost line are the same line seen from opposite sides.**

## Who captured the edge

Zoom out from a single deal to a token's whole first year, and add up *every* seat the firm sits in. This is where wearing both hats really compounds.

![Who captured the edge across one token's first year: add up every seat and the same firm collects far more from its principal option than from the agency fee everyone sees.](/imgs/blogs/designated-versus-principal-market-making-6.webp)

The ledger above tallies four edge sources:

- **Retainer fees** — the designated (agency) hat — about **+$240k** over the year, funded by the project treasury.
- **Spread capture** — earned across both hats while quoting the book — about **+$250k**, funded by the traders on the book (the ordinary cost of the bid-ask spread, paid a few basis points at a time by everyone who trades).
- **Informed-flow edge** — the principal (prop) hat, using what it saw while quoting — about **+$400k**, funded by whoever's flow it could see (the stylized front-running edge from earlier, compounded over a year).
- **Option upside** — the principal (prop) hat again — about **+$10.0M** if the token 3x'd, funded by the project's token supply flowing to retail.

Add them: roughly **$240k + $250k + $400k + $10.0M ≈ $10.9M** captured by one firm across one token's first year. Notice the shape of it. The one number a casual observer sees — the retainer, the "market-making fee" on the invoice — is $240k, about **2%** of the total. The other **98%** is principal: spread, informed flow, and above all the option. The agency fee is the part you are shown. The principal capture is the part you are not.

#### Worked example: the "who paid for it" reconciliation

Every dollar the firm captured came *from* someone. Reconcile it:

1. **Project treasury** funded the $240k retainer and *granted away* the option value — call it $10.0M of token upside handed to the firm. Total borne by the project and its token supply: about **$10.24M.**
2. **Traders on the book** funded the $250k of spread — split across everyone who round-tripped a trade that year.
3. **Whoever's flow the desk could see** funded the $400k informed-flow edge — a slightly worse fill than they would have gotten on a neutral venue.

The grand total the firm captured — about **$10.9M** — is exactly the grand total everyone else paid. That is always true; edge does not appear from nowhere. The single most useful habit this article can leave you with is to ask, of any tight-spread, well-liquid token: *if the market maker is capturing an edge, which of these seats am I sitting in?*

## TradFi's stricter separation

None of this is a crypto invention. Stock markets ran head-first into the two-hats conflict a century ago and spent decades building walls against it. Seeing those walls makes the crypto default look less like "how markets work" and more like "the walls just aren't there yet."

![TradFi walls vs crypto's merged desk: stock markets separate the agency and principal roles by law; in crypto they usually sit inside one entity by default.](/imgs/blogs/designated-versus-principal-market-making-7.webp)

Read the two columns side by side. On the left, four walls US equity markets built. On the right, their crypto counterparts — mostly absent.

**The Volcker Rule (2010).** Named after former Federal Reserve chair Paul Volcker and enacted as Section 619 of the Dodd-Frank Act on 21 July 2010, the Volcker Rule broadly *bars banks from proprietary trading* — from betting their own capital for their own profit. (The final implementing rule was issued in December 2013 and banks had to conform by July 2015.) The rule is not a blanket ban on market making — it explicitly permits it — but permitted market making must be genuinely serving customer demand rather than a disguised prop bet, and pure proprietary speculation inside a bank is off the table. The whole design is to keep the agency-flavored activity (serving customers) apart from the principal activity (betting the house). In crypto, there is no Volcker Rule: a market-making firm can run an unlimited prop book right next to its quoting desk.

**The NYSE Designated Market Maker cannot see your order first.** Here is the sharpest contrast. The New York Stock Exchange replaced its old "specialist" system with **Designated Market Makers** in 2008. DMMs still have obligations to maintain a fair and orderly market — but the reforms deliberately stripped away the specialists' old informational privileges. In the NYSE's own words to regulators, DMMs *"no longer received copies of orders entered in Exchange systems prior to the orders' publication to all market participants."* The one privilege that mattered most — seeing orders before the public — was taken away by design. A crypto market maker quoting a project's book has exactly the opposite: it sees the full book it quotes and trades on, in real time, first.

**Best execution is a legal duty.** In US equities, a broker owes its customers **best execution** under FINRA Rule 5310 — the duty to seek the most favorable reasonably available terms for their orders. It is why a firm cannot simply route your order to whatever destination pays *the firm* the most, without regard to your fill. There is no equivalent duty in crypto: a market maker owes no best-execution obligation to you or, in most arrangements, even to the project. The anonymous trader on the other side of a crypto MM's quote is not its client, and the law that would protect a stock-market client from a worse fill simply does not attach.

**Information barriers between agency and principal desks.** Multi-service financial firms are required to maintain Chinese walls separating the parts of the business that hold client information from the parts that trade for the firm's own account. The barrier is imperfect — walls get breached, and enforcement exists precisely because they do — but the *default* is separation, and breaching it carries legal consequences. The crypto default is the reverse: agency-like market making, proprietary trading, and the token's options all living inside one legal entity, with no wall required and none assumed.

The honest caveat: TradFi's walls are not perfect, and crypto is not lawless — securities and commodities regulators have brought and won cases against crypto firms, as we will see. But the *baseline* is genuinely different. In stocks, separation is the rule and combination is the exception you must justify. In crypto, combination is the rule and separation is the exception almost nobody bothers with.

## How it shows up in price

All of this plumbing eventually reaches the number on your screen. Here is the causal chain, made concrete.

**Tighter spreads than the book deserves.** A designated market maker that is long the token (via the loan-plus-options deal) *wants* the token to look liquid and healthy, because a healthy-looking chart supports the price its options are struck against. So it may quote tighter spreads and more size than the organic order flow would justify — a subsidized-looking market. Tight spreads feel like a benefit to you, and sometimes they are. But a spread that is tight because the firm quoting it is talking its own book is not the same as a spread that is tight because a hundred independent traders are competing. The first can be withdrawn the instant it stops serving the firm.

**Support that appears and disappears on the firm's schedule.** Because the same firm quotes and trades, it can *provide* liquidity when doing so props the price near an option strike or an unlock, and *withdraw* it when holding the bid no longer pays. To you, the book looks deep — until the moment you actually need it to be, and it thins out. The liquidity was real, but it was conditional on the firm's P&L, not on genuine two-sided demand.

**Your fill is a shade worse than on a neutral venue.** The informed-flow edge is not free money conjured from the air; it is paid, a little at a time, by the people whose flow got front-run. On a thin token, your fill can be a few basis points worse than it would be if the firm quoting the book could not also trade ahead of you. You will almost never see this in a single trade. It shows up as a slow, structural tax across many trades — the difference between a truly neutral market and one where the house can see your cards.

**And when the token unlocks, you are the exit.** The tokens the firm acquired cheaply — via the loan, via the options — eventually get sold. The natural buyer for millions of tokens at an elevated price is the retail crowd that the tight spreads and healthy chart helped attract. The price you see is, in part, an advertisement for the liquidity you will provide.

**The size you can't see moves the price you can.** Not all of the flow runs across the public book. Large trades are often negotiated privately as **over-the-counter (OTC) block trades** — deals struck off the order book, directly between two parties, precisely so they don't move the visible price while they execute. A firm wearing both hats can quote you a tight, calm-looking market on the screen while separately moving real size through its OTC desk, and the screen only catches up to that flow later. So the public price you trade on is not the whole market; it is the part of the market the firms with the most information have chosen to let you see. The tape tells you what happened on the book — not what was arranged beside it.

## Common misconceptions

**"Market making is a neutral service, like plumbing."** It sounds neutral — "providing liquidity" — and sometimes it is close to neutral. But a designated market maker on a loan-plus-options deal is *structurally long the token*, and a firm that also runs a prop desk on the same book has *structural reasons* to use what it sees. Neutral plumbing does not hold call options on the water.

**"If the spread is tight, the market is healthy."** A tight spread means *someone* is willing to quote both sides closely. It does not tell you *why*. It can be tight because many independent traders compete, or because one firm that is long the token wants the chart to look good. The two look identical on screen and behave completely differently in a crisis — the first stays, the second can vanish.

**"Front-running is illegal, so it doesn't happen in crypto."** Front-running a customer order is illegal in *regulated securities markets*, where the broker owes best execution and the order is a client order. In much of crypto there is no best-execution duty and the trader on the other side is not a client, so the specific legal prohibition often does not attach. That does not make trading ahead of flow *right* — but it does mean "it's illegal" is not the protection you think it is.

**"A big, reputable firm wouldn't have these conflicts."** Size and reputation reduce the temptation to do the crudest things, because a large firm has more to lose. But the *structure* is the same regardless of reputation: if one entity both quotes a book and trades its own account with no wall, the conflict exists whether or not the firm acts on it. Reputation is a reason to trust a firm's *conduct*; it is not the same as the conflict not being there.

**"If the firm is delta-neutral, it's neutral about my trade."** A market maker often hedges the token inventory it is forced to hold — for example by selling perpetual futures against its spot position — so it carries no *directional* bet on the price. That state is called being *delta-neutral*, and it is prudent risk management. But delta-neutral on inventory is not the same as *conflict-neutral*. The firm can be perfectly hedged on the tokens it holds and still be long the token through its *options*, still see your order flow before you, and still earn more the more the book trades. It is neutral on its own price risk — not neutral on you.

**"The project hired them, so their interests are aligned."** The project and the market maker share *one* interest — a higher token price — but only up to a point. The firm wants the price high enough to make its options pay and to attract the buyers it will sell into; after it has exited, its interest in the price is gone. "Aligned incentives" marketing rarely survives the arithmetic of who sells to whom, and when. (We map these structural conflicts across the whole stack in [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).)

## How it shows up in real markets

Enough theory. Here are named, sourced episodes where the two-hats structure — or the walls built to prevent it — actually mattered. Where a firm's conduct is contested, it is framed strictly as reported or alleged, with the source; the *structure* is what these cases illustrate.

### 1. Alameda and FTX: every conflict in one collapse

The definitive case of the hats fused into one is Alameda Research and FTX. Alameda was a crypto trading firm — a market maker *and* a proprietary trader — and FTX was the exchange, both controlled by the same people. According to the US Securities and Exchange Commission's complaint of 13 December 2022, Alameda was FTX's primary market maker *and* received undisclosed special treatment on the platform: a "virtually unlimited" line of credit funded by FTX customers' own deposits, and an exemption from the automatic risk checks and liquidation that applied to every other account, implemented in FTX's code with features prosecutors later described as an "allow negative" flag. The Commodity Futures Trading Commission brought parallel charges the same day. Sam Bankman-Fried was convicted on seven counts in November 2023 and sentenced to 25 years in prison in March 2024.

The mechanism is the extreme end of this article: a principal trader (Alameda) that also acted as the exchange's designated liquidity provider, sitting inside a venue it controlled, with privileges no outside counterparty had — including, in effect, the ability to trade against customer flow on infrastructure it built. It is what the two-hats conflict looks like with *no* walls at all, and it is why FTX became the cautionary tale of the entire cycle. (We cover the collapse in depth in the sibling profile [Alameda Research: the cautionary tale] and in [FTX's collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).)

### 2. Wintermute: the dual role as ordinary business

Not every firm that wears both hats is a scandal — most are simply doing normal, disclosed business. Wintermute, founded in 2017, is one of the largest algorithmic crypto market makers, and it openly operates across roles: two-sided quoting on centralized venues, on-chain liquidity provision in DeFi, an over-the-counter (OTC) desk, and proprietary trading, moving billions of dollars of volume daily. In September 2022 it suffered a roughly **$160 million** hack of its DeFi operations (traced to a vulnerability in a "vanity" wallet-address tool); its CEO said the firm remained solvent and its centralized and OTC operations were unaffected.

The lesson here is not wrongdoing — none is alleged — but *ubiquity*. The combined market-maker-plus-prop model is simply how a large crypto trading firm is built. The conflict this article describes is not a rare abuse; it is the standard corporate structure of the industry's biggest liquidity providers, operating in plain sight. That is precisely why understanding the structure matters more than trusting any single firm's conduct.

### 3. DWF Labs: the "venture market maker" model, and a contested case

DWF Labs rose quickly from 2023 as a self-styled **"venture market maker"** — a firm that fuses the two hats by design. Per reporting (including by *The Wall Street Journal* in May 2024), DWF's model was to buy a project's tokens at a discount as a principal *and* make markets in them, benefiting when the price rose — a structure that differs from traditional venture capital and puts the firm's principal book and its market-making squarely in the same asset. The same reporting described allegations, raised by Binance's own market-surveillance investigators, that DWF conducted roughly **$300 million** of wash trading in 2023 and manipulated the prices of certain tokens; the report also said Binance dismissed investigators after the internal report. **DWF Labs has denied the allegations, calling them unfounded, and Binance said it found insufficient evidence of market abuse.** The matter is contested and no finding of wrongdoing is asserted here.

Whatever the truth of the specific allegations, the DWF *model* is the clean illustration of this post's thesis: a firm openly holding a principal position in the very tokens it makes markets in. Buy cheap as a principal, quote the book as the designated MM, and the two roles reinforce each other. (The controversy and detection question are covered in the sibling posts on [wash trading and manufactured volume] and DWF's profile.)

### 4. Jump Crypto and the limits of "support"

Jump Crypto, the digital-asset arm of the Chicago trading firm Jump Trading, is another firm that combined heavy principal exposure with market making. In the collapse of the Terra/UST stablecoin, the SEC later alleged that a market maker (identified in reporting as Jump) had, at an earlier stage, traded to help restore UST's peg while the project publicly credited an "algorithm" — an episode framed strictly as alleged per the Commission's filings, with the firm's role contested. The general shape — a principal-and-market-maker firm whose trading can *support* a price the public believes is holding on its own — is the same conflict seen from the "propping it up" side rather than the "front-running the flow" side. (Terra's mechanics and failure are covered in [the Terra/Luna collapse](/blog/trading/crypto/terra-luna-2022-collapse).)

### 5. The old-guard OTC desks: quieter, but the same structure

Not every combined-hat firm is aggressive about it. Established institutional desks — GSR, Cumberland (the crypto arm of the Chicago trading firm DRW), and B2C2 — run market-making and OTC businesses with a quieter, more institutional posture than the pump-adjacent newcomers. They still sit as principals across the assets they quote; the difference is one of *conduct and culture*, not of *structure*. That distinction — same conflict, different behavior — is exactly why you cannot read safety off the business model alone. You have to read the firm, the token, and the incentives on the specific deal.

### 6. The TradFi contrast: payment for order flow and the $65M reminder

The clearest reminder that these walls are *maintained*, not automatic, comes from equities. In December 2020 the SEC charged the retail broker Robinhood with failing its duty of best execution and misleading customers about **payment for order flow** — the practice of routing customer orders to trading firms in exchange for payment. The SEC found that the arrangement led to *inferior* prices for customers that, in aggregate, cost them tens of millions of dollars even after accounting for commission savings, and Robinhood paid a **$65 million** penalty to settle (without admitting or denying the findings). The episode is a TradFi echo of the crypto conflict: a firm's incentive to route flow where *it* profits can quietly cost the customer a worse fill — and in equities, unlike crypto, there is a best-execution duty and a regulator to enforce it. The wall exists because, left alone, the incentive bends the other way.

## Reading who is on the other side

So what do you actually *do* with this, as the person placing the retail order? You cannot un-merge a firm's desks or write crypto a Volcker Rule. But you can read the situation for what it is.

![Reading who is on the other side: instant fills on a thin token are not a gift; they are a clue about who is quoting it and what else they hold.](/imgs/blogs/designated-versus-principal-market-making-8.webp)

The decision flow above is the whole defense in one picture. It starts with the thing that feels *good* — an instant fill, a tight spread, on a thin token — and treats it as a *question* rather than a gift. Ask who quotes the book; on a small token it is often one named market-making firm. Then check the two things that turn a quoting firm into a conflicted one: does it also hold the token's call options (via a disclosed loan-plus-options deal, if the project published one)? Can it see your order flow first (it can, by virtue of quoting)? If either is true — and on a new listing, both usually are — the venue is not neutral, and the correct assumption is that the firm on the other side *can* trade against you.

A few concrete habits fall out of that:

- **Treat tight spreads on thin tokens as a service you are paying for, not a favor.** Someone is being compensated for that liquidity, and on a loan-plus-options deal that someone is long your token.
- **Read the tokenomics and the market-maker disclosure, if any.** A project that discloses its MM deal terms — retainer, loan size, option strike — is handing you the conflict map. A project that discloses nothing is not handing you *less* conflict; it is handing you less *information*. (We walk the on-chain and cap-table forensics in [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).)
- **Size your expectations to the float, not the price.** The tight, healthy-looking market that draws you in on day one is the same market that needs someone to sell millions of unlocked tokens into later. If the arithmetic says you are the natural buyer at the elevated price, you are the exit liquidity.
- **Do not confuse a firm's reputation with the absence of the conflict.** The best firms behave well; the conflict is still there. Trust conduct where you have earned reasons to, but never assume the structure has been dismantled just because the logo is familiar.

### When this matters to you

This is educational, not advice, and none of it says "don't trade small tokens." It says: *know which seat you are in.* The two-hats structure is most dangerous exactly where it is easiest to feel safe — a new token, a tight spread, a chart that looks liquid and alive. That is the environment the combined-hat firm is best paid to create, and the environment where the informed-flow edge, the option upside, and the exit-liquidity dynamic all point the same way: away from you.

The single durable takeaway is the mental model we started with. One firm can wear two hats. The agency hat is the one you are shown — the "market-making service," the tight spread, the healthy book. The principal hat is the one you are not — the option, the prop book, the flow it can see. When you catch yourself admiring how smoothly a small token trades, let it trigger the question instead of the comfort: *who is on the other side, and which of their hats am I looking at?*

## Sources & further reading

Primary and reported sources behind the real-world figures above (as-of dates noted for moving facts):

- **Volcker Rule / Section 619 of the Dodd-Frank Act** — enacted 21 July 2010; final implementing rule issued 10 December 2013; bank conformance deadline 21 July 2015. See the Federal Reserve's [Volcker Rule page](https://www.federalreserve.gov/supervisionreg/volcker-rule.htm) and the [OCC Volcker Rule implementation](https://www.occ.treas.gov/topics/supervision-and-examination/capital-markets/financial-markets/trading-volcker-rule/volcker-rule-implementation.html) materials.
- **NYSE Designated Market Makers replaced specialists (2008); DMMs no longer receive orders before public publication** — [SEC / NYSE rule filing SR-NYSE-2023-36 (Nov 2023)](https://www.sec.gov/files/rules/sro/nyse/2023/34-98869.pdf) and the [Federal Register notice on the DMM program (Nov 2023)](https://www.federalregister.gov/documents/2023/11/13/2023-24868/self-regulatory-organizations-new-york-stock-exchange-llc-notice-of-filing-of-proposed-enhancements).
- **Best execution duty (FINRA Rule 5310)** — see FINRA's [Best Execution rule](https://www.finra.org/rules-guidance/rulebooks/finra-rules/5310).
- **SEC charges Robinhood over payment for order flow and best execution; $65M settlement (17 December 2020)** — [SEC press release 2020-321](https://www.sec.gov/newsroom/press-releases/2020-321).
- **SEC and CFTC actions on FTX/Alameda (13 December 2022); SBF convicted Nov 2023, sentenced to 25 years March 2024** — [SEC press release 2022-219](https://www.sec.gov/newsroom/press-releases/2022-219) and [CFTC press release 8638-22](https://www.cftc.gov/PressRoom/PressReleases/8638-22).
- **Wintermute ~$160M DeFi hack (20 September 2022); firm reported solvent, OTC/CeFi unaffected** — [CoinDesk report (Sep 2022)](https://www.coindesk.com/business/2022/09/20/crypto-market-maker-wintermute-hacked-for-160m-says-ceo).
- **DWF Labs "venture market maker" model; reported ~$300M wash-trading allegations (WSJ, May 2024); DWF denies, Binance cited insufficient evidence** — [The Block coverage of DWF's denial (May 2024)](https://www.theblock.co/post/293429/dwf-labs-denies-report-that-it-did-300-million-of-wash-trading-on-binance-last-year). Allegations reported and contested; no finding of wrongdoing.

Sibling posts in this series that go deeper on the mechanics referenced above:

- [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — spread capture, inventory, and adverse selection from zero.
- [The loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) — the crypto-native MM contract in full.
- [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — who profits at each layer of the stack.
- [Crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the overview this series expands.

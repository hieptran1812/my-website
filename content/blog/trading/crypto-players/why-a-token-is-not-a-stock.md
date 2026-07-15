---
title: "Why a Token Is Not a Stock: The Structural Gap That Lets Insiders Win"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A share is a legally enforced claim on profits, votes, and disclosure that funnels insider selling through an IPO gate and a lockup; a token usually is none of that, which is why crypto insiders can exit on a public market years before an equity investor ever could."
tags: ["crypto", "tokenomics", "float", "fdv", "vesting", "ipo", "securities", "market-structure", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A share and a token can trade on a screen the same way, but legally they are different objects, and that difference is the quiet engine behind every "insiders win, retail loses" story in crypto.
>
> - A **share** is a legally enforced bundle: a residual claim on profits and assets, a dividend when the board declares one, a vote backed by corporate law, mandated quarterly disclosure, and an exit that must pass an IPO gate plus a lockup. A **token** is usually a governance or utility right with *none* of those.
> - The load-bearing gap is the **exit**: an equity insider cannot sell on any public market until the company goes public, often 7 to 10 years after the seed round, and then only after a ~180-day lockup. A token insider can sell on a public order book roughly a year after seeding, braked only by a vesting schedule the project set for itself.
> - **Float and FDV** are where the gap shows up in a price. A token can post a \$20B "fully diluted valuation" while only ~8% of supply trades, so a \$1.6B real market carries a headline 12.5x larger — a promise that demand must rise 12x just to hold the price flat.
> - The real case study: when **Starknet's STRK** launched on the low-float template that came to define the cycle (2024-02-20), roughly 7% of its 10 billion tokens were circulating, and its fully diluted valuation reached about \$35B on a market cap near \$2.5B (CoinDesk, 2024-02-20). Binance Research found 2024 launches averaged a market-cap-to-FDV ratio of just 12.3% (Binance Research, May 2024).
> - The one habit that protects you: **read the float, not the headline market cap.** The number in the headline is a price multiplied by tokens that mostly cannot be sold yet.

Here is a puzzle that trips up almost everyone new to crypto. You can buy a share of a company and you can buy a project's token, and on a trading app they look identical: a ticker, a price, a green or red candle, a "buy" button. So it feels natural to reason about them the same way — "the market cap is \$2 billion, that's what people think it's worth." But that instinct quietly imports a whole legal machine that exists for shares and mostly does not exist for tokens. The share comes wrapped in centuries of corporate and securities law that says what you own, forces the company to tell you the truth on a schedule, and controls when the people who got in early are allowed to sell. Strip that machine away and keep only the price chart, and you get the token — the same-looking object with a very different set of rules underneath.

The diagram above is the mental model for this whole piece: a share and a token are two different legal objects that happen to share a price chart. Everything that follows is a tour of the specific pieces the token drops — the profit claim, the mandated disclosure, and above all the exit gate — and the exact arithmetic by which those missing pieces let insiders win at the expense of whoever buys at launch.

![A share versus a token: what each one legally grants](/imgs/blogs/why-a-token-is-not-a-stock-1.webp)

This is a companion to the broader map in [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers), which walks the whole cast of funds and trading firms that move token prices. That piece answers *who* the players are. This one answers something more basic and more structural: *why the instrument itself is built so that they can win.* We are going to build every term from zero — share, IPO, underwriter, lockup, dividend, token, SAFT, float, market cap, FDV, dilution — anchor each idea in round-number arithmetic, and then land it on a real, dated launch. No conspiracy, no shilling. Just the legal plumbing and the math it produces.

## Foundations: the building blocks

We need a shared vocabulary before we can compare anything. Read this section even if some terms feel familiar, because the entire argument later turns on the *precise* legal meaning of "share", "lockup", and "float" — and on the fact that a token quietly redefines or discards each one.

### What a share actually is (a legal object, not just a price)

A **share** (or "stock", the words are used interchangeably) is a unit of *ownership* in a company. When you own one share of a company that has issued one million shares, you legally own one-millionth of that company. That ownership is not a vibe; it is a bundle of specific, court-enforceable rights:

- **A residual claim on profits and assets.** "Residual" is the key word. A company pays its suppliers, its employees' wages, the interest on its debt, and its taxes *first*. Whatever is left over — the residual — belongs to the shareholders. You are last in line, which is exactly why you own the *upside*: once everyone ahead of you is paid, the rest is yours. If the company is ever wound down and sold for parts, shareholders split whatever remains after all those senior claims are settled.
- **A dividend when the board declares one.** A **dividend** is a cash payment out of profits, distributed to shareholders in proportion to how many shares they hold. It is not automatic — a company's board of directors *declares* it — but the right to receive it if it is declared is legally yours, and once declared it becomes a debt the company owes you.
- **A vote enforced by corporate law.** One share typically carries one vote on major corporate decisions — electing the board, approving a merger. If management ignores a valid shareholder vote, shareholders can sue, and courts will enforce the outcome. The vote is backed by the legal system, not by a company's goodwill.

The picture below is the cleanest way to hold this in your head. A share is the *bottom* of a payment waterfall — the residual claim — and it owns everything left after the senior claims are paid.

![A share is the residual claim at the bottom of the payment waterfall; a token is not on it](/imgs/blogs/why-a-token-is-not-a-stock-6.webp)

#### Worked example: what the residual claim actually pays

Suppose a company earns \$100 of revenue in a year. Watch the waterfall drain, top to bottom:

```
revenue                         = $100
- suppliers, wages, operations  = $60   -> $40 left
- interest to lenders           = $10   -> $30 left
- corporate taxes (say 20%)     = $6    -> $24 left
= net profit (the residual)     = $24
```

That \$24 belongs to the shareholders. The board might pay out, say, \$10 of it as a dividend and reinvest the other \$14 to grow the business (which, if it works, raises the value of your share). Either way, the \$24 is *yours* as the residual owner. **The intuition: a share is a legal claim on the money that is left after everyone senior is paid — it is a slice of the business itself.** Hold onto this, because the single most important fact about a typical token is that it has *no line at all* in this waterfall.

### How a company reaches the public: IPO, underwriter, and the two markets

A private company's shares do not trade on any public exchange. If you own shares in a startup, you generally cannot sell them to the public at all — there is no order book, no ticker, no buyer waiting. To change that, the company does an **IPO** (Initial Public Offering): the first time it sells shares to the general public and lists them on a stock exchange.

An IPO is not a button the founder presses. It runs through an **underwriter** — typically an investment bank — whose job is to shepherd the company through a legally demanding process:

- The company files a **registration statement** (in the US, a Form S-1) with the securities regulator, a long document disclosing its financials, risks, and ownership. Regulators review it. Lying in it is securities fraud.
- The underwriter does **due diligence** — verifying the company's books — because the underwriter is putting its own reputation and, often, its own capital behind the offering.
- Only after this gate clears do the shares list on an exchange and become publicly tradeable. We cover the venue side of this in [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses).

Two terms fall out of this and matter for the rest of the piece:

- The **primary market** is where a security is *first sold* by the issuer to raise money — the IPO itself, or a private funding round. The company receives the cash.
- The **secondary market** is where investors then trade that security among themselves — the public exchange. The company receives nothing; buyers and sellers just swap.

The distinction is load-bearing. Insiders make their money by buying in the *primary* market cheaply and selling in the *secondary* market, where the public is. The whole question of "when can an insider sell?" is the question of *when they get access to the secondary market.*

### The lockup: the brake on insider selling

Even after an IPO, the earliest investors and employees usually cannot sell immediately. They are bound by a **lockup** (or "lock-up period"): a contractual promise not to sell their shares on the public market for a set window after the IPO. The industry-standard window is **180 days**, though lockups range from roughly 90 to 180 days, and the exact length is negotiated between the company and its underwriters (LegalClarity; StrikeRates, 2025).

Two things about the lockup are worth pinning down precisely, because crypto commentary often garbles them:

- **The lockup is a contract with the underwriter, not a government mandate.** The securities regulator does not *set* a lockup length; the underwriters impose it as a condition of the deal, to prevent a flood of insider selling from crushing the new stock right after listing. The underwriter can even waive it early at its discretion.
- **A separate legal rule also restrains insiders: Rule 144.** In the US, Rule 144 governs when "restricted" or insider-held shares can be resold — for a company that files regular reports, an insider generally must have held the shares at least six months, and large "affiliate" holders face volume limits on how much they can dribble out. Because the lockup usually runs about 180 days, that six-month holding period is typically satisfied right around when the lockup lifts (LegalClarity, 2025).

Stack these up and you get the equity insider's reality: buy in a private round, wait years for the company to be ready to go public, pass through the IPO gate with its mandated disclosure, and *then* sit through a ~180-day lockup before selling a single share to the public. The exit is slow, gated, and disclosed. Hold that image; the token is about to break every part of it.

### What a token usually is

A **token** is a unit of value recorded on a blockchain — a shared, public ledger that many independent computers maintain so no single party controls the record. Some tokens are money-like (a stablecoin pegged to a dollar). The ones this piece is about are **project tokens**, which come in two flavors:

- A **utility token** is meant to be *used* inside a product — to pay fees on a network, to access a service. Its pitch is "you'll need this to do things here."
- A **governance token** grants a vote inside the project's own software — for example, voting on how a protocol's treasury is spent. Crucially, that vote is enforced by *code*, not by corporate law, and it usually governs the protocol's internal parameters, not a claim on any company's profits.

Here is the part that surprises people: **a typical project token gives you no legal claim on the project's profits, no dividend, and no ownership of the entity that built it.** It is not equity. Owning a project's token is closer to owning a ticket that might become more valuable if the thing it accesses becomes popular — a price that moves up and down, and sometimes a vote inside an app. When founders raise money by selling tokens before the network exists, they often use a **SAFT** (Simple Agreement for Future Tokens): a contract where an investor pays now for tokens to be delivered later, at network launch. The SAFT is how a fund gets its cheap, early position — the token equivalent of a private funding round.

One more thing the SAFT makes clear: the token has its own primary and secondary markets, and the split is even starker than in equities. The private rounds and the SAFT are the token's *primary* market — insiders buy newly created tokens directly from the project. The public listing is the *secondary* market, where those insiders (once vested) sell to everyone else. In an IPO, the primary sale to the public and the secondary market open together, under one gated, disclosed process. For a token, the primary sale happened privately, months or years earlier, at prices the public never saw, and the secondary market opens on its own — so by the time you can buy, the entire primary market has already closed above your head. You are not buying a fresh issue on equal footing; you are buying resale supply from people who bought first and cheaper.

So the token keeps the one thing a share has that everyone fixates on — a live, tradeable price — and quietly drops the profit claim, the dividend, the law-backed vote, the mandated disclosure, and the gated exit. That is the whole story in one sentence, and the rest of this piece is the arithmetic of what each dropped piece is worth.

### The supply words: float, market cap, FDV, and dilution

Four more terms, all about *supply*, and all weaponized later:

- **Circulating supply**, or **float**, is the number of tokens currently free to trade. This is the actual, tradeable market.
- **Total supply** (or **max supply**) is every token that will ever exist, including the locked insider tokens that have not been released yet.
- **Market capitalization** ("market cap") is price × *circulating* supply — the value of what actually trades.
- **Fully diluted valuation** (**FDV**) is price × *total* supply — the value of everything, as if every locked token were already trading.
- **Dilution** is what happens when locked tokens are released into circulation: the float grows, so each existing token represents a smaller slice of a bigger pie, and — absent a matching rise in buying — the price falls.

When the float is a small fraction of the total supply, FDV towers over market cap, and the gap between them is the *overhang*: supply that will hit the market on a known schedule. That gap is the single most important number a token buyer can look at, and we will compute it three different ways before we are done.

With the vocabulary pinned down, we can now measure the gap. It has three parts: the *entry* (insiders buy cheaper), the *exit* (insiders sell sooner), and the *supply* (the float hides how much is coming). We take them in order.

## 1. The entry-price gap: insiders buy on a different ladder

Start with entry, because it sets up everything else. Both an equity VC and a crypto VC do the same thing at the front end: they raise a pool of money from outside investors and write early checks into projects, betting that a few big winners carry the fund. (The fund economics — management fees, carried interest, the pressure to return capital — are the same template we unpack in [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20).) The difference is the *price* at which they enter, and it is enormous.

A token typically raises across several private rounds before the public ever sees it, each at a higher price: a seed round at, say, \$0.05 per token, a private round at \$0.20, a strategic round at \$0.50, and then a public launch at \$2.00. Each round is a rung on a ladder, and by the time retail can click "buy", the price has already climbed the entire ladder.

![The funding-round price ladder: retail buys at the very top](/imgs/blogs/why-a-token-is-not-a-stock-7.webp)

The figure traces the rungs. Notice who is standing where. The seed fund's cost basis is \$0.05. The retail buyer's cost basis is \$2.00 — the top of the ladder. They now own the identical token at the identical live price, but their *break-evens* could not be more different, and break-even is what decides who can afford to sell into whom.

#### Worked example: the 40x entry-price gap

Suppose a fund buys 10 million tokens in the seed round at \$0.05:

```
tokens bought    = 10,000,000
price per token  = $0.05
amount invested  = 10,000,000 x $0.05 = $500,000
```

A year later the token launches publicly at \$2.00. On paper the fund's stake is now:

```
tokens held      = 10,000,000
launch price     = $2.00
paper value      = 10,000,000 x $2.00 = $20,000,000
multiple         = $2.00 / $0.05      = 40x
```

A 40x on paper, before a single retail order was ever filled. Now compare the two break-evens. The fund breaks even at \$0.05, so it profits at *any* price above a nickel — it can sell all the way down to \$0.10, \$0.20, \$0.50 and still book a huge gain. The retail buyer breaks even at \$2.00, so the token has to *stay* at \$2.00 just for them to not lose money. **The intuition: the price chart you can see tells you almost nothing about who is winning, because the same token that is a disaster for the \$2.00 retail buyer is still a 10x for the \$0.05 seed fund.** A token down 75% from launch is a wipeout for retail and a 10-bagger for the seed round.

This entry gap exists in equities too — venture investors also buy far cheaper than IPO buyers. What makes the crypto version sharper is what comes next: how *soon*, and how *quietly*, those cheap insiders can sell.

## 2. The exit gate: why a token insider can sell years earlier

This is the heart of the whole piece. The entry gap is large but familiar. The *exit* gap is where the token's missing legal machine changes the game, because it collapses the insider's timeline from a decade to about a year.

Recall the equity insider's path from the Foundations section: seed round, then years of the company staying private (there is simply no public market to sell into), then the IPO gate with its mandated disclosure, then a ~180-day lockup. Only after all of that can an early investor sell shares to the public. Historically the whole journey from seed to a liquid public exit takes on the order of 7 to 10 years.

The token insider's path is radically shorter. There is no IPO — the token simply lists on an exchange, often within a year or so of the private rounds. There is no underwriter gate and no mandated registration statement. And there is no lockup imposed by an exchange or a regulator. The only brake on insider selling is a **vesting schedule** — a cliff and a release curve — that *the project wrote for itself*, usually enforced by a smart contract. The two roads look like this:

![IPO lockup versus token unlock: two roads to a public market](/imgs/blogs/why-a-token-is-not-a-stock-2.webp)

Both insiders start at the same place — a cheap seed round. The equity insider's first opportunity to sell to the public arrives around year 7 to 10. The token insider's arrives around year 1. That six-to-nine-year head start, on a *public* market, is the structural gap this entire series is named for. It is not that token insiders are greedier than equity insiders; it is that the instrument removed the gates that used to make them wait.

### The exit gate itself: what a token skips

It helps to see exactly which gates disappear. Picture everything that stands between an early insider and a sale on a public order book, on each path.

![What stands between an insider and a public sale, for equity versus a token](/imgs/blogs/why-a-token-is-not-a-stock-5.webp)

On the equity path, insider selling funnels through two hard gates: the **IPO gate** (an S-1 filing plus underwriter due diligence — a public audit of the company before anyone can sell) and then the **lockup plus Rule 144** (a ~180-day contractual freeze, followed by resale limits on large holders). On the token path, there is one soft gate: a **vesting schedule the project set for itself**, whose cliff typically ends in *months*, not years. Both paths dump into the same place — a public order book where retail is buying — but one path passed through a regulator and an underwriter to get there, and the other passed through a smart contract the founders configured.

#### Worked example: the IPO lockup timeline versus the token unlock timeline

Put real months on it. Take an equity seed investor and a token seed investor who both write their check on the same day, call it month 0.

The **equity** investor's earliest public sale:

```
seed round                         month 0
company stays private (build)      months 0 to ~96
IPO lists on the exchange          ~month 96   (8 years, a typical figure)
180-day lockup after IPO           +6 months
earliest public sale               ~month 102  (~8.5 years after seeding)
```

The **token** investor's earliest public sale:

```
seed round                         month 0
token generation event + listing   ~month 12
self-imposed cliff (say 12 months) ...but many cliffs are shorter,
                                    and monthly unlocks then begin
earliest public sale               ~month 12 to 24 after seeding
```

The equity investor waits roughly **8.5 years** to touch a public market. The token investor waits roughly **1 to 2 years**. That is a gap of six to nine years of public-market access (about seven in this example) — and every one of those extra early sales is filled by someone, usually retail buying at launch. **The intuition: the token did not just make insiders richer at entry; it moved their exit forward by most of a decade, onto a public market, with retail on the other side of the trade.**

### Why no exchange makes a token wait

You might expect the *exchange* to fill the gap the underwriter leaves — to demand a lockup before it lists a token, the way a stock exchange sits at the end of a gated IPO. It generally does not, and the reason is structural. A crypto exchange makes its money from trading fees, so its incentive is to list tokens that will trade *actively* from day one, not to slow their supply down. In practice, getting a token listed has historically involved the opposite of a lockup: a requirement that the project arrive with a **market maker** already lined up to quote a deep, liquid order book, often funded by a loan of the project's own tokens. The exchange wants the book to look busy; a self-imposed insider vesting schedule is the project's business, not a listing condition the venue enforces.

That flips the equity logic on its head. In equities, the venue is a gatekeeper that helps *restrain* early selling (the exchange lists the stock at the end of the underwriter's process, and the lockup keeps insiders out for ~180 days). In crypto, the venue is a *counterparty and beneficiary* of the trading it hosts — and several large exchanges also run venture arms that hold cheap early stakes in the very tokens they list, stacking one more interested party on the insider side of the table. The mechanics of that listing-and-liquidity layer are the subject of [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) and [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase); the point here is narrower. The equity market has an institution whose job includes making insiders wait. The token market does not. The only thing making a token insider wait is the schedule the insiders wrote — which, as Starknet will show us, they can rewrite.

### The disclosure gap

There is a second missing gate that is easy to overlook because it is invisible: **mandated disclosure.**

A US public company lives under the Securities Exchange Act of 1934, which forces it to tell the public the truth on a fixed schedule (Securities Exchange Act of 1934). Once a company is public (above roughly \$10 million in assets and a threshold number of holders), it must file:

- a **Form 10-K** every year — a comprehensive report with **audited** financial statements;
- a **Form 10-Q** every quarter — a lighter update, generally filed within 45 days of quarter-end;
- a **Form 8-K** within about four business days of any *material* event — a CEO departure, a bankruptcy, a major deal.

All of it goes onto a public system (EDGAR) that anyone can read for free. Miss a filing, or lie in one, and you are in front of the regulator. The point of this machine is that an equity investor buys into a company whose books are audited and whose bad news must be disclosed on a clock.

A typical token project has **no equivalent mandated disclosure regime.** There is no required audited annual report on the project's finances, no mandated quarterly update, no legal obligation to disclose a material event on a four-day clock. Good projects publish transparency reports and put their token unlock schedules on-chain voluntarily; many do not, and none are *compelled* to in the way a public company is. So the token buyer is not only entering at a worse price and facing an earlier insider exit — they are doing it with far less mandated information about what they are buying. The float and the unlock schedule are often knowable from on-chain data (we will use exactly that below), but the project's actual finances, runway, and internal decisions usually are not.

> A share forces the company to tell you the truth on a schedule; a token asks you to trust that it will.

The unresolved legal question hanging over all of this is whether a given token is, in fact, a *security* — in which case much of the equity machine (registration, disclosure, insider-sale rules) would apply after all. That fight is ongoing and is beyond this piece; what matters here is that, in practice, most project tokens have operated without that machine, and the structure described above is the result.

### The vote that is not the same vote

Token marketing leans hard on "governance", and it is easy to hear "governance token" and think "voting share". They are not the same instrument. A shareholder vote is backed by corporate law: it elects the people who run the company and can approve or block a merger, and if management defies a valid vote, shareholders can go to court. It is a claim on *control of the entity that owns the assets and earns the profits.*

A governance token's vote is enforced by *code*, and it usually governs the *protocol's own parameters* — a fee level, how a community treasury is allocated, which software upgrade ships — not the finances of the company that built the protocol. Two consequences follow. First, the thing you are voting on is often narrower than "who controls the business": the founding company, its equity, and its revenue can sit entirely outside the token's reach. Second, because votes are usually weighted by tokens held, and insiders hold the largest allocations, on-chain "governance" can be dominated by the same funds and teams whose selling the token holder is exposed to — the vote and the overhang are held by the same hands. A governance vote can be real and useful, but calling it the equivalent of a shareholder's legal control over a company quietly overstates what the token grants. It is one more place where the token keeps the *word* that a share uses ("vote") while dropping the *legal object* underneath it.

## 3. Float versus FDV: the number that hides the overhang

Now we get to how the gap shows up in an actual price — and to the single most misread number in crypto. We defined it in Foundations: market cap is price × *circulating* supply, while FDV is price × *total* supply. When only a sliver of supply is circulating, these two numbers diverge wildly, and the headline you see is almost always the bigger, emptier one.

![Float versus FDV: the headline valuation rests on a thin real market](/imgs/blogs/why-a-token-is-not-a-stock-3.webp)

The figure is the iceberg. The tall gray block is the locked supply — tokens that exist, are counted in the FDV, but cannot be sold yet. The thin green sliver at the waterline is the float — the tokens that actually trade. The headline "\$20 billion token" is the whole iceberg; the real market is the sliver.

#### Worked example: \$20B FDV on an 8% float

A token launches with a total supply of 10 billion tokens. At launch only 8% of them — 800 million — are circulating; the other 92% are locked under investor, team, and treasury vesting. The opening price is \$2.00.

The **market cap** (what actually trades):

```
circulating float = 800,000,000
price             = $2.00
market cap        = 800,000,000 x $2.00 = $1,600,000,000   ($1.6B)
```

The **fully diluted valuation** (what the headline counts):

```
total supply      = 10,000,000,000
price             = $2.00
FDV               = 10,000,000,000 x $2.00 = $20,000,000,000  ($20B)
```

The ratio of FDV to market cap:

```
FDV / market cap  = $20B / $1.6B = 12.5x
```

Headlines will call this a \$20 billion token. But the actual money supporting the \$2.00 price is the buying against 800 million circulating tokens — a real market of \$1.6 billion. The other 9.2 billion tokens, worth \$18.4 billion at today's price, are locked and scheduled to arrive later. To keep the price at \$2.00 as all of that unlocks, the market would have to absorb up to \$18.4 billion of *new* selling with fresh buying demand — a roughly 12x increase in the dollars in the market, just to stand still. **The intuition: a giant FDV on a tiny float is not a measure of value; it is a promise that demand will multiply by roughly the FDV-to-market-cap ratio, or the price falls. Read the float, not the headline.**

There is a second reason the FDV matters, and it is about incentives, not just optics. The FDV is the number *insiders* use to mark the value of their own locked tokens. A fund that bought 100 million tokens at \$0.05 and now sees a \$2.00 price gets to tell its own investors that its stake is "worth" \$200 million — priced off the same thin float that a retail buyer is holding. The bigger the FDV, the better the insiders' bags look on paper, which is precisely why a launch is often engineered for a high FDV on a low float: it flatters every number the insiders report while pushing the *real* price discovery onto a sliver of supply the public trades against. The headline valuation is not a neutral fact about the market; it is a figure that serves the people who set it.

This is why "read the float, not the market cap" is the first reflex of anyone who has been burned. The headline number is engineered to look big; the float tells you how thin the ice under it really is.

## 4. Dilution and the supply cliff: how it shows up in price

The float being small is only half the danger. The other half is that it does not *stay* small — it grows on a published schedule, and each step is a **supply cliff** that pushes the price down. This is dilution in action, and it is the most predictable price event in a token's life.

Why does an unlock pressure the price at all? Because the people whose tokens just unlocked are frequently sellers. A fund sitting on a 40x wants to lock in some of that gain; an employee paid in tokens wants to pay rent. None of them *has* to sell, but at the margin an unlock pours new sellable supply into a market whose buying demand has not grown to match. More sellers, same buyers, lower clearing price. And because unlock dates are public, sophisticated traders often position *ahead* of them, shorting into the unlock — so the price can sag before a single insider token is even sold.

![How a supply cliff shows up in price](/imgs/blogs/why-a-token-is-not-a-stock-4.webp)

The figure walks three snapshots of the same token. Total supply is fixed at 10 billion the whole time; what changes is how much of it is *circulating*. At launch, 8% floats. Twelve months in, unlocks have pushed it to 25%. Twenty-four months in, to 50%. As the tradeable float triples and then grows six-fold, the same pool of buying demand has to clear a much larger supply — so the clearing price steps down.

#### Worked example: the dilution an unlock creates

Make the mechanism concrete. A token trades at \$2.00 with 800 million tokens circulating, so the dollars actually in the market are:

```
circulating float   = 800,000,000
price               = $2.00
money in the market = 800,000,000 x $2.00 = $1,600,000,000
```

Now a scheduled unlock and the following months of unlocks lift the circulating float to 2 billion tokens — a bit more than a 2x increase — while genuine buying demand, the real dollars people bring, grows more slowly, say from \$1.6B to \$2.4B (a 1.5x increase). The price that clears is roughly the dollars in the market divided by the tokens that must be held:

```
new money in market ~ $2,400,000,000
new circulating      = 2,000,000,000
implied price ~ $2,400,000,000 / 2,000,000,000 = $1.20
```

The price falls from \$2.00 to \$1.20 — a 40% drop — even though buying demand actually *rose* by 50%. Demand grew; supply grew faster. Push it one more cliff, to 5 billion circulating against \$3.5B of demand, and the implied price is \$0.70. **The intuition: in tokens the supply curve is a calendar, and the biggest events on that calendar are insider unlocks. A holder can be right that the project is growing — more users, more demand — and still lose money because the float grew faster than the demand did.** This is the mechanism the price chart hides and the unlock schedule reveals.

Notice what just happened across sections 1 through 4. The insider entered at \$0.05 (section 1), can exit onto a public market within a year (section 2), sells into a thin float propping a headline FDV 12x its real size (section 3), and the very act of that scheduled selling dilutes the price the retail buyer is holding (section 4). Each piece is legal. Together they are a machine, and the fuel it runs on is the retail buyer at the top of the ladder.

## Common misconceptions

**"A token is basically a stock for a crypto company."** It is not. A stock is a residual legal claim on a company's profits and assets, with a dividend right, a law-backed vote, and mandated disclosure. A typical project token is a utility or governance right with none of those — no claim on profits, no dividend, no audited books it is forced to publish. They share a price chart and almost nothing else legally. Reasoning about a token as if it were equity imports protections that were never there.

**"If the market cap is \$1.6 billion, that's what the market thinks it's worth."** Market cap is just price × circulating float, and float is often a single-digit percentage of total supply. The headline you usually see is the FDV, which multiplies the same price by *all* the supply, including the 90%+ that is locked. Both numbers are "a price times a quantity", and the price is only real to the extent someone would actually pay it for the *whole* quantity — which, for the locked 90%, no one has.

**"There's no lockup in crypto, so it's a free-for-all."** There is a brake — the vesting schedule — but it is a brake the *project set for itself*, enforced by code, not a ~180-day freeze imposed by an underwriter as a condition of listing, and not backed by a resale rule like Rule 144. A self-imposed cliff can be, and sometimes is, changed by the same insiders it constrains (as we will see with Starknet). "Self-imposed" and "exchange-enforced" are not the same kind of promise.

**"Unlocks are already priced in, so they don't matter."** Sometimes partly, but the empirical pattern is that scheduled supply reliably pressures prices, because an unlock changes the *supply* side of the market on a known date while doing nothing for the *demand* side. Even when traders anticipate it by shorting ahead, that anticipation *is* the price falling — it just falls before the date instead of on it. "Priced in" often means "already fell", not "won't fall".

**"A high FDV means the project is important."** FDV is the most manufacturable number in crypto: list with a tiny float and a huge total supply, and you can print a multi-billion-dollar FDV out of a thin real market. It is the number insiders use to mark their locked bags at an impressive figure. A high FDV on a low float is closer to a warning label — a large overhang scheduled to arrive — than a badge of importance.

**"Because it trades publicly, insiders and I are on a level playing field."** You share a *current price*, not a *cost basis*. The seed fund's break-even might be \$0.05 while yours is \$2.00, and the fund knows the unlock schedule cold while most retail buyers do not even know it exists. A shared price with unshared cost bases and unshared information is not a level field; it is the same field with the insiders standing on a hill.

## How it shows up in real markets

These are named, dated episodes. Figures are as-of the events described and sourced below.

### 1. Starknet (STRK): the template in one launch

Starknet, an Ethereum scaling network built by StarkWare, launched its STRK token on 2024-02-20 with what was then billed as one of the year's largest airdrops — roughly **728 million tokens distributed to about 1.3 million wallets** (The Block; CoinDesk, 2024-02-20). The total supply was **10 billion**, so the airdropped float was on the order of **7% of supply** at the debut — a textbook low float. STRK traded as high as about **\$5** on listing day (its all-time high of \$5.30 was set that day) and settled around **\$3.50** in a volatile open (CoinDesk, 2024-02-20). Pre-launch futures on Aevo had implied an inception price near **\$1.65**, which would have meant roughly a **\$16 billion FDV** before the token even traded (CoinDesk, 2024-02-14).

Run the float-versus-FDV arithmetic on the actual open. At about \$3.50 with ~728 million circulating, the *market cap* was on the order of \$2.5 billion, while the *fully diluted valuation* — \$3.50 × 10 billion — was about **\$35 billion**, and it reached as high as roughly \$50 billion intraday at the peak (CoinDesk, 2024-02-20). That is an FDV-to-market-cap ratio in the neighborhood of 14x, produced entirely by the thin launch float. On the insider side, StarkWare's allocation put roughly **20% of supply to early contributors and about 18% to investors** — close to 38% of all tokens to insiders — with more to the company itself (per the published tokenomics). This is the entry gap, the exit gate, and the float-versus-FDV mechanism from sections 1 through 4, all visible in a single dated launch.

### 2. Starknet's unlock, and what "self-imposed" really means

Starknet is also the cleanest illustration of the "self-imposed vesting" point. The original schedule would have unlocked a large slug of early-contributor and investor tokens beginning **2024-04-15** — reportedly on the order of **1.34 billion tokens (about 13.4% of supply)** in the early window. After community backlash, StarkWare *revised its own schedule*: the 2024-04-15 unlock was cut to just **64 million tokens (0.64% of supply)**, with 64 million releasing monthly until early 2025 and then 127 million monthly for the following two years, into 2027 (Blockworks; The Block, 2024-02). STRK jumped about **10%** on the announcement that the unlock had been delayed (CoinDesk, 2024-02-22). The lesson cuts both ways: the vesting cliff *is* a real brake — moving it visibly moved the price — but it is a brake the insiders control and can reset, which is exactly what "self-imposed, not exchange-enforced" means. An equity underwriter's 180-day lockup is not something the founders can renegotiate on a Thursday.

### 3. The 2024 low-float, high-FDV wave

Starknet was not an outlier; it was the template. In May 2024, Binance Research published a study, *Low Float & High FDV: How Did We Get Here?*, examining the year's token launches. Its findings quantify everything above: tokens launched in 2024 carried an average **market-cap-to-FDV ratio of just 12.3%**, meaning on average only about an eighth of the "valuation" was actually circulating; in the sample, **no token launched with more than 20% of supply circulating, and some floated as little as ~6%** (Binance Research, May 2024). The report estimated that roughly **\$155 billion** of token supply was scheduled to unlock between 2024 and 2030, and that holding prices flat against 2024's cohort alone would require on the order of **\$80 billion** of fresh demand. That is the overhang from section 3, measured across an entire year of launches rather than one token.

### 4. The equity mirror: Facebook's lockup expiry

The equity world shows the same supply-pressure mechanic in its one narrow window where insider supply hits the market on a known date — the **lockup expiry** — and Facebook's 2012 IPO is the textbook case. Facebook went public on 2012-05-18 at \$38 per share on Nasdaq. Over the following months its lockups expired in tranches, freeing early investors and insiders — Accel Partners, Peter Thiel, and others — to sell. When the first big tranche of roughly 271 million shares came unlocked around 2012-08-16, the stock fell to a then-low near \$19.69, roughly *half* the IPO price (TIME; Yahoo Finance, 2012-08). Same mechanism as a token unlock: a known date, a wave of previously restricted supply, and a price that sags to absorb it. Facebook, of course, recovered and went on to be one of the great compounders — the lockup was a supply event, not a verdict on the business.

The instructive contrast is *scale and frequency*. Facebook passed through its lockup cliffs once, within its first year; after that its share count was relatively stable and any new issuance was disclosed and slow. A token faces a lockup-expiry-like event *repeatedly*, month after month, as its vesting curve releases supply for years — Starknet's revised schedule alone runs monthly unlocks into 2027. The equity market has one supply cliff and then flat ground; the token has a staircase of them, printed on the tokenomics page from day one. And where Facebook's insiders were bound by a lockup they could not move, a token project can, and sometimes does, rewrite its own.

### 5. Why the ETF bridge does not change the token's nature

It is worth heading off a hopeful misreading. The arrival of spot crypto ETFs — covered in [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge) — wraps certain crypto assets in a regulated, disclosed, exchange-listed product. But an ETF wrapper around Bitcoin does not turn *project tokens* into shares. The thousands of governance and utility tokens that launch each cycle still have no profit claim, no mandated disclosure, and a self-imposed vesting schedule. The bridge changed how large institutions can *hold* a few blue-chip crypto assets; it did not change the legal anatomy of the long tail of tokens where the insider-versus-retail structure lives.

### 6. When the structure is taken to the extreme

The clearest demonstrations that this structure is real are the cases where it was pushed until it broke. The [FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) turned on FTX's own FTT token being marked as billions in "collateral" against a thin float the firm itself supported — a headline number multiplied by a quantity no one could actually sell into. And the [3AC and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) showed how positions marked at FDV-style valuations evaporate when the real, thin market is finally tested. In both, the fatal error was the same arithmetic mistake this piece keeps returning to: treating a price that holds for a small float as if it held for the whole stack.

### 7. The information you can still see

There is a hopeful flip side to the missing disclosure machine. A token has no mandated 10-K, but much of what you need is *on-chain* — public by construction. The circulating supply and the total supply are queryable. The allocation to investors, team, and treasury is usually published in a tokenomics document. The vesting schedule and the exact dates on which locked tokens release are often encoded in smart contracts and tracked by unlock calendars anyone can read. So while a token buyer gets far less *mandated* information than a stock buyer, the two facts that matter most for the structure in this piece — the float and the unlock schedule — are frequently more transparent than a private equity investor could ever get. The catch is that almost no retail buyer looks. The Starknet unlock that moved the price 10% was public days in advance; the low float behind every 2024 launch was on-chain the moment the token listed. The defense is not access to secret information — it is the discipline to read the public information that is already there. Tracing who owns what, wallet by wallet, is its own skill, and it is the subject of a dedicated piece on reading a token's cap table in this series.

## When this matters to you

You do not need to trade a single token for this to be useful — it is a lens that makes any token chart legible. If you ever do look at one, the structure in this piece turns into a short, concrete checklist, and almost all of it is answerable from public, on-chain data even without the mandated disclosure a stock would carry:

- **What is the float versus the total supply, and what is the FDV?** If a small single-digit-percent float sits under a multi-billion FDV, the price you see is propped on a thin slice of supply with a lot more scheduled to arrive. This is the first thing to check and often the most revealing.
- **What is the unlock schedule?** Unlock dates are usually public. They are the biggest predictable selling events in a token's life. A cliff in the next few months is a calendar of supply you will be competing against.
- **Who funded it, and at what price?** If well-known funds entered in seed and private rounds far below the launch price, their cost basis is far below yours — which means they can profit selling into prices where you lose.
- **Is the vesting self-imposed, and can it change?** A token's lockup is a promise the project made to itself, enforceable by code but adjustable by the same insiders. That is a weaker guarantee than an underwriter-enforced equity lockup, and it is worth knowing which one you are relying on.
- **Read the float, not the headline.** The single most protective habit is to treat every "size" number — market cap, FDV, TVL, volume — as a price multiplied by a quantity, and to ask how much of that quantity could *actually* be sold at that price. For a low-float token, the answer is: not much.

There is a quick mental rule that ties these together. The float percentage tells you, roughly, how much total demand has to *multiply* for the price to survive full dilution. If 10% of supply is circulating, then over the token's life the market has to find close to 10x the current dollars just to hold the price flat as the rest unlocks — because you are effectively pricing 100% of the supply on 10% of the buying. At 5% float, it is closer to 20x. Ask yourself whether that much fresh demand is plausible for this particular project, on the timeline of its unlock schedule. Usually the honest answer is "no", and that answer is the whole low-float trade in one line: the price today is borrowing against demand that has to show up later, and mostly does not. This is not a formula to trade on — real prices are messier, demand is not constant, and a great project can outrun its unlocks — but as a first filter it separates a headline you can trust from one you cannot.

None of this says tokens are frauds or that the structure is illegitimate. Plenty of token projects are real, venture funding builds real things, and a low float is not by itself evidence of bad intent. The point is narrower and more durable: a token is *not* a stock, and the legal machine that a stock carries — the profit claim, the mandated disclosure, the IPO gate, the exchange-enforced lockup — is exactly the machine that slows insiders down and keeps them honest with the public. A token keeps the price chart and drops the machine. Once you can see which pieces are missing, the headline market cap stops being a fact about value and becomes what it usually is: a number engineered to look bigger than the market beneath it. This is educational, not investment advice — but it is the difference between reading a price as a signal and reading it as someone else's exit.

For the wider cast of players who operate inside this structure — the funds that seed the ladder and the market makers that supply the launch liquidity — start with [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers), and see how the listing-and-custody layer works in [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).

## Sources & further reading

- **IPO lockups and Rule 144** — [Lock-Up Period: Restrictions on Post-IPO Share Sales](https://legalclarity.org/lock-up-period-restrictions-on-post-ipo-share-sales/) (LegalClarity) and [IPO Lock-Up Periods: The 180-Day Selling Restriction](https://www.strikerates.com/learn/ipo-lockup) (StrikeRates, 2025): 180 days is the industry-default lockup (range ~90–180), negotiated with underwriters, not mandated by the SEC; Rule 144 imposes a ~6-month holding and volume limits on affiliates.
- **Mandated disclosure** — [Securities Exchange Act of 1934](https://www.toppanmerrill.com/glossary/securities-exchange-act-of-1934/) (Toppan Merrill glossary) and the SEC's EDGAR filings: Form 10-K (annual, audited), Form 10-Q (quarterly, within 45 days), Form 8-K (material events, within ~4 business days).
- **Starknet (STRK) launch figures** — [Starknet Token STRK Begins Trading at \$5 After Mammoth Airdrop](https://www.coindesk.com/business/2024/02/20/starknet-token-strk-begins-trading-at-5-after-mammoth-airdrop) and [Starknet's STRK Could Debut With Market Cap of Over \$1B](https://www.coindesk.com/markets/2024/02/14/starknets-strk-could-debut-with-market-cap-of-over-1b-aevos-pre-launch-futures-suggest) (CoinDesk, 2024-02): ~728M airdropped to ~1.3M wallets, 10B total supply, ~\$5 high / ~\$3.50 settle, ~\$35B FDV.
- **Starknet unlock revision** — [Starknet's STRK Jumps After Developer StarkWare Agrees to Delay Token Unlocks](https://www.coindesk.com/tech/2024/02/22/starknets-strk-jumps-after-developer-starkware-agrees-to-delay-token-unlocks) (CoinDesk) and [StarkWare revises STRK token lockup schedule after criticism](https://blockworks.co/news/starkware-airdrop-token-lockup) (Blockworks, 2024-02): the 2024-04-15 unlock cut from ~1.34B to 64M tokens after backlash.
- **The low-float / high-FDV pattern** — [Low Float & High FDV: How Did We Get Here?](https://www.binance.com/research/analysis/low-float-and-high-fdv-how-did-we-get-here) (Binance Research, May 2024): 2024 launches averaged a 12.3% market-cap-to-FDV ratio; no sampled token floated above 20%; ~\$155B of unlocks scheduled 2024–2030.
- **Equity lockup expiry (Facebook)** — [Facebook Hits Record Low as Insider Stock Sale Lock-Up Period Ends](http://business.time.com/2012/08/17/facebook-hits-record-low-as-insider-stock-sale-lock-up-period-ends/) (TIME, 2012-08-17) and [Facebook hits new low as IPO lock-up ends](https://finance.yahoo.com/news/facebook-hits-low-ipo-lock-143200785.html) (Yahoo Finance, 2012-08): IPO at \$38 on 2012-05-18, first tranche of ~271M shares unlocked ~2012-08-16, stock to a ~\$19.69 low.
- **On this blog** — [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) (the players who operate this structure), [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge), [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase), [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried), [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses), and [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20).

---
title: "Cui Bono: The Incentive Map of Crypto and Why Retail Is the Exit"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A first-principles map of who profits at each layer of a token launch — VC, market maker, exchange, foundation, team — and why their payoffs are structurally aimed at the launch-day retail buyer."
tags: ["crypto", "tokenomics", "incentives", "conflict-of-interest", "exit-liquidity", "venture-capital", "market-makers", "principal-agent", "retail-defense", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — In a token launch, every layer above the launch-day retail buyer — the VC, the team, the market maker, the exchange, the foundation — enters at a lower cost and can exit earlier, so their profit is *structurally* the money retail brings in. "Aligned incentives" is a marketing phrase; the cap table and the unlock calendar are the real incentive map.
>
> - The core idea is *cui bono* — "who benefits?" Ask it of every layer of a launch and you find the same answer: whoever entered cheaper and earlier than you.
> - We reuse one hypothetical token launch for the whole post. A seed fund's cost is \$0.02, a strategic fund's is \$0.10, the team's is ~\$0 — retail's is \$1.00 to \$2.00. Same token, wildly different break-even.
> - The market maker's loan-plus-call deal makes it a *stakeholder* in the token it quotes: heads it makes millions on the option, tails it walks away flat keeping the spread. It is never on retail's side of that trade.
> - When the token trades at \$0.40 a year after a \$2.00 peak — an 80% fall from the top, a 73% loss for a launch buyer at \$1.50 — every insider layer is *still* in profit. Retail is the only red on the board. That is not a bug; it is the design.
> - Binance's own research called newly launched low-float tokens "exit liquidity for insiders" (Binance Research, *Low Float & High FDV*, May 2024). The industry says the quiet part in its own reports.

Here is a question worth asking before you buy any token, and almost nobody does: *cui bono?* It is Latin — "to whose benefit?" — the question a detective asks at a crime scene and a question a trader should ask at every price. When a token launches at \$2 and a thousand people on social media are cheering, somebody sold them those tokens. Who? At what price did *that* person get in? And when the chart later falls 70%, is everyone losing — or is the loss landing on exactly one group while another quietly booked the win years ago?

This post is the capstone of a series that maps the players who actually move crypto prices. The earlier posts each took one layer apart: [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock), [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock), [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move), and [how to read a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table). This one ties them together with a single, unifying idea: the whole stack shares one structural conflict of interest, and once you see it, the price chart stops being a mystery and becomes a map of who is paying whom.

![A matrix showing each layer's entry price, when it cashes out, and who is on the other side of its trade — retail buyers, in every row.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. Read the rows top to bottom: the seed fund, the strategic fund, the team, the market maker, the exchange — and at the bottom, retail. Each row has three facts: what it paid to get in, when it can get out, and who is on the other side of its trade. For every layer *except the last*, the "other side" is the same phrase — *retail buyers*. That is the entire thesis in one picture. This is not a claim that crypto is fraud, or that these players are villains. It is a claim about arithmetic and timing. The people who funded the project and supplied its liquidity entered earlier, cheaper, and with better information, and the structure of a token launch turns the launch-day buyer into the natural counterparty for their eventual sell. Everything below builds that claim from zero and then walks the dollars.

## Foundations: the building blocks of an incentive map

Before we can trace who profits, we need a shared vocabulary. Crypto reuses words from finance and law and then bends them, so the whole argument turns to mush if any of these terms is fuzzy. Read this section even if the words look familiar — the later conflicts hinge on their *precise* meaning. None of this assumes any finance background.

### What an incentive actually is

An **incentive** is simply the answer to "what makes this person better off?" It is not what someone says they want; it is what the structure *pays them* to do. If a salesperson earns commission on volume, their incentive is volume, whatever they tell you about caring for your needs. Incentives are the gravity of markets: over enough time and enough people, behavior bends toward whatever the money rewards, regardless of anyone's intentions. When we "map incentives," we are ignoring the marketing and asking, mechanically, *what does each actor get paid to do?*

A **conflict of interest** exists when one party's incentive runs against another's, especially when the first party is supposed to be serving the second. Your financial advisor has a conflict of interest if they earn a bigger fee steering you into fund A than fund B; the advice they *should* give and the advice they are *paid* to give point in different directions. The conflict is not the same as wrongdoing — a good advisor resists it — but the conflict is *structural*, and it stays latent only as long as someone chooses not to exploit it.

### The principal-agent problem

There is a formal name for the deepest version of this, and it is worth knowing because it recurs at every layer of crypto. The **principal-agent problem** describes what happens when a *principal* (the person whose money is at risk) hires an *agent* (the person who makes the decisions) and the two have different incentives and different information. The classic 1976 paper by Michael Jensen and William Meckling framed corporate finance around it: shareholders (principals) hire managers (agents) who may enrich themselves at the shareholders' expense because they know more and are paid differently. The agent is closer to the information, controls the timing, and is compensated on a schedule that may not match the principal's actual outcome.

Hold onto that shape — *better information, control of timing, compensation decoupled from the principal's result* — because it describes the crypto venture capitalist versus their investors, the venture capitalist versus retail, and the market maker versus the project all at once. Every layer we meet is an agent to someone else's principal, and at the very bottom of the chain, holding the risk with the least information and no control of timing, sits the retail buyer.

### Exit liquidity

**Liquidity** is how easily you can sell something without moving its price. A liquid market has many buyers waiting close to the current price, so you can sell a large amount and barely nudge it. An illiquid market is thin: try to sell size and the price caves in, because there simply is not enough demand on the other side. Crucially, to sell, you need someone to *buy*. Your exit requires someone else's entry.

**Exit liquidity** is the term for *the buyers whose purchases let an earlier holder sell out*. If a fund owns 100 million tokens it wants to convert to cash, it needs 100 million tokens' worth of buying to appear. Those buyers are the fund's exit liquidity — the demand it sells *into*. The phrase is neutral engineering when you say it that way, and brutal when you notice who usually provides it. When a token launch is engineered so that insiders hold most of the supply and retail holds most of the fresh demand, retail is, definitionally, the exit liquidity. This is not a slur invented by cynics; as we will see, Binance's own research desk used the exact phrase in a 2024 report about the token launches on its own exchange.

### "Aligned incentives"

You will hear this phrase constantly in crypto marketing: the token "aligns incentives" between the team, the investors, and the community. The *idea* is real and good — if everyone holds the same token, everyone wants its price to rise, so everyone rows in the same direction. The problem is that alignment on *direction* (everyone prefers the price to go up) hides a total misalignment on *timing and cost*. A seed fund at \$0.02 and a retail buyer at \$1.50 both "want the price to go up," yes — but the fund is already up 75x and can sell profitably at any price above \$0.02, while the retail buyer needs the price to *hold above \$1.50* just to break even. They are aligned the way a homeowner and a squatter are aligned: both would prefer the house not burn down. That is not the same as being on the same side.

### Carry and fees: how the money managers get paid

Two more terms, because they explain *why* funds behave the way they do. A professional investment fund — a venture fund or hedge fund — is run by **general partners** (GPs), the small team who make decisions, using money raised from **limited partners** (LPs), the pensions, endowments, and wealthy individuals who put up almost all the capital but make no decisions. The GPs are compensated with the industry-standard template, often summarized as "2 and 20": a **management fee** of around 2% of the fund's size *every year* regardless of performance, and **carried interest** — "carry" — of around 20% of the *profits*. We unpack this model in detail in [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20); the number to remember is that carry is a slice of gains, so anything that makes the gains look bigger *sooner* pays the GPs more.

### The BD/listing deal

**BD** stands for "business development," and in crypto a **BD deal** or **listing deal** is the arrangement by which a token gets onto a major exchange. Historically this has involved some combination of a cash listing fee, a marketing commitment, and — the contested part — a slice of the token's own supply handed to the exchange. The details are famously opaque and disputed (we will look at a real public fight over the numbers later), but the shape matters: getting listed is a *negotiation with a gatekeeper who often ends up holding the token it is about to make tradeable*. A listing is therefore not a neutral technical event. It is a price event arranged between interested parties.

### The market-maker call-option deal

Last one, and it is the single most important mechanic in the whole series, so we will only sketch it here and go deep in [the loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid). A **market maker** is a firm that continuously posts both a buy price and a sell price for a token, so anyone who wants to trade can. A new token needs one or its order book is a desert. But the project usually cannot or will not post millions in inventory itself, so the standard crypto arrangement is a **loan-plus-call deal**: the project *lends* the market maker a batch of tokens to quote with, and in exchange grants the market maker **call options** — the right, but not the obligation, to buy those tokens at a fixed "strike" price later. If the price rises above the strike, the market maker exercises and keeps the gain. If it falls, the options expire worthless and the market maker just returns the borrowed tokens. Public write-ups of these deals (from market-making firms like Flowdesk and Caladan and industry explainers) describe exactly this structure: the loan sized as a percentage of supply, a strike often set to a time-weighted average price, a term of a year or more, and the market maker compensated through the option. The point for our incentive map: the firm that makes the market *also owns upside in the token it makes the market for*. It is not a neutral utility. It is a stakeholder.

With those eight ideas — incentive, conflict of interest, principal-agent, exit liquidity, aligned incentives, carry/fees, the BD deal, and the market-maker option deal — we have everything we need. Now let us put them on one token and count.

## The running example: one launch, counted every way

For the rest of this post we will use a single, deliberately round, hypothetical token launch. Call it **NOVA**. Every number below is illustrative arithmetic — I am not claiming NOVA is real — but the *structure* and the *ratios* are typical of 2021–2024 launches, and reusing one example is the whole point: it lets us compute each layer's profit and loss on the *same event* and see where the money actually goes.

Here is NOVA's setup, fixed for the whole article:

| Fact | Value |
|---|---|
| Total supply | 1,000,000,000 tokens (1 billion) |
| Launch ("TGE") price | \$1.00 |
| Fully diluted valuation at launch | \$1.00 × 1B = \$1 billion |
| Circulating float at launch | 100,000,000 (10% of supply) |
| Launch market cap | \$100 million |
| Thin-float peak (month 1–2) | \$2.00 |
| Price 12 months later, post-unlocks | \$0.40 |
| Retail's blended entry price | \$1.50 |

TGE means "token generation event" — the moment the token first exists and trades publicly. **Fully diluted valuation (FDV)** is the price times the *total* supply, as if every token were already trading; **market cap** is price times only the *circulating* supply. Notice NOVA's FDV (\$1 billion) is ten times its real market cap (\$100 million), because 90% of the supply is locked. That gap is where the whole story lives.

And here is NOVA's cap table — who owns the billion tokens and what they paid:

| Layer | Allocation | Tokens | Cost per token | Cash in |
|---|---|---|---|---|
| Seed VC | 15% | 150,000,000 | \$0.02 | \$3,000,000 |
| Strategic VC | 10% | 100,000,000 | \$0.10 | \$10,000,000 |
| Team & advisors | 20% | 200,000,000 | ~\$0 | ~\$0 |
| Foundation / treasury | 20% | 200,000,000 | \$0 | \$0 |
| Ecosystem / rewards | 20% | 200,000,000 | \$0 | \$0 |
| Community float (public) | 10% | 100,000,000 | trades at launch | — |
| Market-maker loan | 5% | 50,000,000 | borrowed | strike \$1.20 |

Allocations vary widely in the real world, but this shape — roughly 80% to insiders and treasury, a thin slice of genuine public float, and a market-maker loan carved out to seed liquidity — is squarely in the range that drew heavy criticism across the 2024 launch cycle. Keep this table open in your mind. Every worked example below draws its numbers from it.

## 1. The cost-basis ladder: why the chart lies about who is winning

Start with the simplest, most powerful fact in the whole post: the price you see is a single number, but the people trading against it have completely different **cost bases** — the price *they* paid. And cost basis is what determines who can afford to sell into whom.

![A ladder of the layers by cost basis: retail at the top having paid the most, the team and foundation at the bottom at near-zero cost.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-2.webp)

The ladder above sorts NOVA's players by what they paid. Read it as a literal ladder: the higher you sit, the more you paid to get on, and the further you can fall. The team and foundation are at the bottom rung at ~\$0 — the entire price is upside for them. The seed fund is one rung up at \$0.02, the strategic fund at \$0.10. And retail is at the very top, having paid \$1.00 to \$2.00. When a chart is green, everyone feels like a winner. When it is red, everyone feels like a loser. Both feelings are wrong, because a shared *current* price is not a shared *cost basis*.

#### Worked example: the same token, five different break-evens

Let us make the ladder concrete. NOVA launches at \$1.00. Here is what each layer's stake is worth on paper the instant it lists, and what its multiple is:

```
Seed VC:      150,000,000 tokens x $1.00 = $150,000,000
              cost $3,000,000  ->  50x on paper, break-even $0.02

Strategic VC: 100,000,000 tokens x $1.00 = $100,000,000
              cost $10,000,000 ->  10x on paper, break-even $0.10

Team:         200,000,000 tokens x $1.00 = $200,000,000
              cost ~$0         ->  ~infinite, break-even ~$0

Retail:       buys at $1.00-$2.00 (call it $1.50 blended)
              cost = the price  ->  1x, break-even $1.50
```

The word "on paper" is doing real work — nobody can dump their whole stake at once without collapsing the price, and insider tokens are locked and vesting. But the *break-evens* are already carved in stone, and they are what matter. The seed fund breaks even at \$0.02. Retail breaks even at \$1.50. That is a 75-fold difference in where "profit" begins for two people looking at the *same* token on the *same* screen.

**The intuition:** the price chart tells you nothing about who is winning, because winning is measured from cost basis, and everyone above retail bought at a fraction of what retail paid. A token down 60% from launch is a catastrophe for the \$1.50 buyer and still a 20x for the \$0.02 seed fund.

### What this costs, and when it breaks

This asymmetry is not itself dishonest — early risk *should* earn a lower price; the seed fund put money into a project that might have shipped nothing. The problem is what the asymmetry does to *behavior* once the token is liquid. An investor whose break-even is \$0.02 can sell profitably into almost any price retail is willing to pay, and their rational move — take some risk off a 50x — is to sell. An investor whose break-even is \$1.50 needs the price to *stay up* and so keeps buying and holding. The cheap money is structurally a seller; the expensive money is structurally a buyer. Put those two facts next to each other and you have already derived who the exit liquidity is, before we have discussed a single named firm.

## 2. The timing conflict: everyone above you got out before you got in

Cost basis is *half* the incentive map. The other half is *timing* — and timing is where "aligned incentives" quietly dies. Two people can want the same direction and still be mortal enemies if one gets to act first.

![A timeline showing seed and private rounds entering years before launch, retail buying at TGE, and insiders selling after the one-year cliff.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-3.webp)

The timeline traces NOVA through time. The seed fund entered three years before the public could buy, at \$0.02. The strategic fund entered a year before, at \$0.10. Then comes day zero — the TGE and listing — which is the *first* moment retail can participate, and by definition the *last* and highest price in the chain. Then, twelve months later, the one-year **cliff**: the date before which insider tokens are frozen and after which they begin to **vest**, becoming sellable. And in the months after that, as the locked supply drips into the market, insiders distribute — they sell — and the price grinds toward \$0.40.

Look at the shape. The lowest-cost buyers are also the *earliest* buyers, and after the cliff they are the *first sellers*. Retail's entire window — from the first click to being fully invested — sits in the middle, exactly where it becomes the demand that the earlier, cheaper money sells into. The unlock schedule is public; anyone can read it. But retail almost never does, and even when it does, it rarely prices in that the biggest scheduled events in a token's life are its insiders becoming free to sell.

#### Worked example: the unlock as a scheduled transfer

Return to NOVA. At launch, only the 10% community float — 100 million tokens — is circulating. The one-year cliff releases a wave of previously locked VC and team tokens into that thin market. Suppose over the twelve months after the cliff, insiders whose tokens have vested sell 60 million tokens into the market at an average realized price of \$1.20 (some into the \$2.00 pump, more on the slow way down):

```
tokens insiders sell   = 60,000,000
average selling price   = $1.20
insider proceeds        = 60,000,000 x $1.20 = $72,000,000
their cost on those     ~ $1,000,000   (seed/strategic/team blend, near zero)
insider realized profit ~ $71,000,000
```

That \$72 million did not fall from the sky. Every dollar of it came from a buyer — from retail. And here is the same 60 million tokens valued at the year-end price:

```
60,000,000 tokens x $0.40 = $24,000,000
```

Retail paid \$72 million for tokens now worth \$24 million. The \$48 million difference is retail's loss, and it is *also*, almost exactly, the insiders' realized gain. The unlock is not a market accident. It is a scheduled transfer of wealth, printed on a public calendar months in advance.

**The intuition:** in tokens, the supply curve is a calendar, and the biggest events on that calendar are the dates your counterparties become legally free to sell to you.

### Why "aligned incentives" cannot survive this

Now we can be precise about the marketing phrase. "Aligned incentives" claims the team, investors, and community are on the same side. Test it against the arithmetic.

![A before-and-after figure contrasting the marketing claims with the cap-table reality: 80% insider supply, a cost-basis gap, a month-12 cliff, and backing that is exit liquidity.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-5.webp)

On the left, the pitch: "community-owned," "aligned incentives," "fair launch, we're all early," "backed by top-tier VCs." On the right, the arithmetic that the cap table and the unlock calendar actually describe: 80% of supply is insider-allocated, the insider cost basis is \$0.02 against your \$1.50, the unlock cliff hits in month 12, and — the sharpest one — *their backing is their exit liquidity*. That last line reframes the proudest marketing claim of all. "Backed by a top-tier VC" sounds like a stamp of quality, and sometimes it is. But mechanically, a VC's backing means a large, cheap, time-locked block of supply sitting above the market, waiting for a schedule to let it sell. The backing *is* the overhang. The two are the same fact described by opposite words.

None of this requires anyone to break a rule or even to be cynical. The team can genuinely love its project; the VC can genuinely believe in the technology. But love and belief do not change a break-even, and belief does not stop a ten-year fund from needing to return capital to its LPs on a schedule. The incentives are set by the structure, and the structure says: the earliest, cheapest money sells, on a calendar, to the latest, most expensive money. That is what "aligned" cannot paper over.

## 3. The venture capitalist: an agent playing with someone else's clock

Now walk the layers one at a time and ask *cui bono* of each. Start with the VC, because the VC sits at the top of the chain and often decides which projects reach retail at all.

The crypto venture fund raises a pool from LPs and buys tokens (or rights to future tokens) at private-round prices — NOVA's seed at \$0.02, its strategic round at \$0.10 — long before the public can. So far this is ordinary venture capital. The twist that reshapes every downstream incentive is the *exit*. An equity VC that backs a startup cannot sell its shares until an IPO or acquisition, historically seven to ten years out, and getting there forces years of building and a public audit of the books. A token VC can sell on a *public order book* within a year of launch, the moment its tokens vest. The exit collapses from a decade to a single year, and — this is the crucial part — it no longer requires the project to have built anything durable. It requires only a liquid market at a price above \$0.02. We go deep on this business model in [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model); here we only need its incentive.

Recall the principal-agent problem. The VC's GPs are *agents* to two different principals. To their LPs, they are agents paid on carry — a slice of gains they are rewarded for realizing fast and marking high. To retail, they are not formally agents at all, but they are the informed, early, timing-controlling party against whom retail trades. The GP has better information (they know the unlock schedule and the real state of the project), controls the timing (they decide when to sell within the vesting rules), and is compensated on a schedule decoupled from retail's outcome (carry on realized gains). That is the principal-agent problem twice over, with retail holding the risk at the bottom.

#### Worked example: carry pays the GP whether or not you recover

Suppose NOVA's seed investment sits inside a \$100 million seed fund charging the standard "2 and 20." First, the fee that flows regardless of any outcome:

```
fund size            = $100,000,000
management fee (2%)  = $100,000,000 x 0.02 = $2,000,000 per year
```

\$2 million a year, every year, just for managing the pool — before a single investment pays off. Now the carry. The fund's NOVA position cost \$3 million for 150 million tokens. Say the fund realizes \$30 million from NOVA by selling into strength over the vesting period (it cannot realize the full \$150 million paper value — dumping would crush the price — but \$30 million on a \$3 million cost is a clean 10x on the cash actually returned):

```
proceeds realized    = $30,000,000
cost                 = $3,000,000
profit               = $27,000,000
GP carry (20%)       = $27,000,000 x 0.20 = $5,400,000
LPs receive          = $21,600,000  (before fees)
```

The GPs keep \$5.4 million of carry on this one position, plus their \$2 million annual fee, *whether or not the remaining locked tokens ever recover and whether or not any retail buyer ever gets back to break-even*. The GP's biggest payday is computed on gains it is incentivized to realize fast and mark high — which is exactly why the liquid-token exit pushes crypto VCs toward early launches and lofty FDVs.

**The intuition:** the VC is an agent paid on realized and marked gains, so its incentive is to get a token liquid, mark it at a big FDV, and realize early — and none of those incentives point at whether the launch-day buyer is made whole.

### What this costs, and when it breaks

The VC layer also has a second, quieter power: it decides *what launches at all*. A founder needs the largest checks, which come from a small set of funds; those funds then open the doors to the market makers and exchanges that a launch needs. So the same names recur across launch after launch — not because of a formal cartel, but because capital, liquidity, and listings are all concentrated in whoever already has all three. The system breaks — turns from ordinary venture capital into pure extraction — when the marking gets circular, when a fund's paper returns and a project's headline FDV are propped by the same thin float and the same friendly market maker. We will see the extreme version of that when we reach the real-world blow-ups.

## 4. The market maker: paid to be a stakeholder, not a referee

The second insider class is the trading firm. Where the VC funds the project, the market maker makes it *tradeable* — and gets paid in a way that quietly puts it on the insiders' side of the book.

Recall the loan-plus-call deal from the foundations. The project lends the market maker tokens to quote with, and grants it call options at a strike. The market maker earns two ways: the **spread** it captures on every trade (the small gap between its buy and sell quotes, multiplied by volume), and the **option** upside if the price rises above the strike. NOVA's deal: a 50-million-token loan (5% of supply) and call options on 20 million tokens at a \$1.20 strike. Let us draw the payoff, because it is the clearest single picture of why the market maker is never on retail's side.

![The market maker's payoff diagram: a flat +$2M spread floor below the $1.20 strike, kinking upward to +$16M as the price runs to $2.00.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-4.webp)

The horizontal axis is the token price; the vertical axis is the market maker's profit and loss. Look at the shape. Below the \$1.20 strike, the payoff is *flat and positive* — the calls expire worthless, but the market maker returns the borrowed tokens and keeps whatever spread it earned quoting the book, on the order of \$2 million in NOVA's first quarter. Above the strike, the payoff *rises*: every dollar the price climbs is 20 million tokens' worth of in-the-money option. There is no region where the line goes meaningfully negative. Heads it wins big; tails it walks away flat. Compare that to retail's payoff, which is a straight diagonal — a dollar of gain for every dollar up, a dollar of loss for every dollar down, with a break-even at \$1.50 marked on the axis. The two payoffs could not be more different.

#### Worked example: the market maker's two outcomes on NOVA

Compute both ends. First, the spread. Suppose the market maker quotes NOVA with a 0.3% spread and trades roughly \$15 million of volume a day in the first quarter, capturing on average half the spread:

```
daily volume         = $15,000,000
captured spread       = 0.15% (half of 0.3%)
daily capture         = $15,000,000 x 0.0015 = $22,500
over ~90 days         ~ $2,000,000
```

Roughly \$2 million banked in the first quarter alone, regardless of price direction. Now the option. If the price runs to the \$2.00 peak, the market maker exercises its calls:

```
strike               = $1.20
market price          = $2.00
profit per token      = $2.00 - $1.20 = $0.80
tokens under option   = 20,000,000
option profit         = 20,000,000 x $0.80 = $16,000,000
```

A \$16 million gain on the option, on top of the \$2 million of spread. And if instead the price ends at \$0.40, the calls (strike \$1.20, well above \$0.40) expire worthless, the market maker returns the borrowed tokens, and it keeps the ~\$2 million of spread. Its downside on the option leg is *zero*.

**The intuition:** a loan-plus-call deal hands the market maker a free option on the token's success, so it is structurally long the very token it quotes — a stakeholder, not a neutral referee, and never the party on the other side of retail's loss.

### What this costs, and when it breaks

The legitimate version of this is genuinely useful: new tokens need liquidity, someone must risk capital to provide it, and the option compensates the market maker for holding a volatile, maybe-worthless new asset. The dark version — dumping the borrowed tokens to crash the price and buying back cheap, or painting a misleadingly active market — is a real temptation *precisely because* the firm holds the inventory and the options. That temptation has surfaced in public allegations, which we will treat carefully as reported claims in the real-markets section. For the incentive map, the point stands either way: the market maker's payoff has no red region, and retail's is all diagonal. When you buy a freshly listed token, the deepest liquidity you are trading against belongs to a firm that profits if the price rises and walks away flat if it falls. That firm is not your counterparty by accident; it is your counterparty by contract.

## 5. The exchange and the foundation: gatekeepers who also hold the token

Two more layers, quicker, because they rhyme with the ones above.

The **exchange** is supposed to be a neutral venue — a place where buyers and sellers meet. But major exchanges are also investors (through venture arms like Coinbase Ventures and the former Binance Labs), listing gatekeepers, and sometimes counterparties. A listing is a price event because a token on a top exchange gains legitimacy and a wave of new buyers; and the exchange, historically, has extracted value from granting it. The exact terms are fiercely disputed — we will look at a real public fight over the numbers — but the *shape* is a gatekeeper that often ends up holding the token it is about to make tradeable, which stacks one more interested party on the insider side.

#### Worked example: an exchange's listing deal on NOVA

Model a modest, illustrative BD deal: the exchange takes 1% of supply — 10 million tokens — as part of listing NOVA, plus it earns trading fees on the volume it hosts.

```
BD allocation        = 1% of supply = 10,000,000 tokens
value at $1.00 launch = 10,000,000 x $1.00 = $10,000,000
exchange cost basis   = $0
trading fees          = $15M/day volume x ~0.1% ~ $15,000/day = millions/year
```

Like every layer above retail, the exchange's cost basis is ~\$0, and it earns whether the token rises or falls. (The *contested* real-world upper bound is far larger — we will come to a public claim of up to 15% of supply — but even the modest version puts the exchange on the insider side of the table.)

The **foundation and treasury** are the least-discussed large holders. NOVA's foundation and ecosystem allocations total 40% of supply — 400 million tokens — held to fund grants, development, and operations. Foundations sell tokens to pay for those things, and a foundation selling into the market is exactly the same supply pressure as a VC selling, just wearing a mission statement. Token foundations and DAO treasuries can move markets by selling, granting, or staking, and their cost basis, like everyone else's up here, is zero. They are on-chain central banks with enormous balances and very little obligation to the launch-day buyer.

## 6. Exit liquidity, drawn: where retail's money actually goes

We now have every layer's incentive. Let us assemble them into one flow and follow the dollars. This is the synthesis figure of the whole post.

![The exit-liquidity fan: $72M of retail money flowing out to the seed VC, strategic VC, team, foundation, market maker, and exchange.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-6.webp)

Retail buyers put \$72 million into NOVA over its first year — the exit-liquidity example from section 2. The fan shows where that money went. It did not fund a product; it flowed *up the stack* to the layers selling into the bid: an illustrative split of roughly \$26 million to the seed VC, \$16 million to the strategic VC, \$14 million to the team, \$6 million to the foundation, \$6 million of spread to the market maker, and \$4 million of fees and BD value to the exchange. (The split is illustrative — the exact routing depends on who sold what and when — but it sums to the \$72 million retail actually spent.) The arrows all point one way: from the bottom of the ladder to the top. Retail's purchase is not an investment in a shared future; it is the liquidity event for everyone who got in earlier and cheaper.

This is what the phrase "retail is the exit" means, drawn as plumbing. And it is not a fringe critique. In May 2024, Binance's own research desk published a report titled *Low Float & High FDV: How Did We Get Here?* which stated plainly that most newly launched tokens "function as 'exit liquidity' for insiders capitalizing on the lack of retail access because of the low initial circulating supply." The report estimated roughly \$155 billion of tokens would unlock between 2024 and 2030 — an ocean of scheduled supply that, without a matching wave of new demand, has only one place to go. When the largest exchange in the world describes its own listings this way in a public report, the incentive map has stopped being a theory.

## 7. The whole board: same launch, one loser

Here is the payoff of the entire argument, on one picture. Take NOVA a year after launch, priced at \$0.40 — an 80% fall from the \$2.00 peak, and a 73% loss for a launch buyer who paid \$1.50 — the kind of chart that looks like a disaster for everyone. Now compute each layer's actual profit and loss on the same event.

![A profit-and-loss board: six green insider cells all in profit, and one wide red cell for retail down $48M — the only loss on the board.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-7.webp)

Every insider layer is green. The seed VC is up ~\$26 million, the strategic VC ~\$16 million, the team ~\$14 million, the foundation ~\$6 million, the market maker ~\$6 million of spread, the exchange ~\$4 million. Retail is the one red cell on the board, down ~\$48 million. *An 80% fall from the peak, and every single layer above retail still booked a profit.* That is not a market where "everyone lost." It is a market that worked exactly as its structure designed it to: the cheap, early, informed money extracted a gain, and the expensive, late, uninformed money provided it.

#### Worked example: reconciling the board

Tie the numbers together so nothing is hand-waved. Retail bought 60 million tokens' worth of insider selling at an average \$1.20, paying \$72 million. Those tokens are worth \$24 million at \$0.40, so retail's mark-to-market loss on them is \$48 million — the red cell. The insiders' cost on the tokens they sold was near zero, so their ~\$72 million of proceeds is almost all profit, distributed across the layers as the fan showed. The board and the fan are the *same event* seen two ways: the fan shows the money moving, the board shows the resulting P&L. Both say the same thing. The launch did not destroy \$48 million of value; it *moved* it, from the bottom of the ladder to the top, on a public schedule.

**The intuition:** when a token falls 73% and every layer above retail is still in profit, the loss did not come from the market being hard — it came from the structure putting retail on the paying end of every other layer's trade.

## Common misconceptions

**"If everyone holds the same token, incentives are aligned."** Alignment on *direction* (everyone prefers up) hides total misalignment on *cost and timing*. A \$0.02 holder and a \$1.50 holder both want the price up, but the first profits selling into almost any price and the second needs the price to hold. Same direction, opposite trade. Aligned the way a landlord and a tenant both prefer the building not collapse — which is not the same as being on the same side.

**"A big-name VC backing a token is a quality stamp, so it's safer for me."** The backing is also the overhang. A top VC's stake is a large, cheap, time-locked block of supply sitting above the market on a known unlock schedule. It can be a signal of quality *and* the single largest source of future selling pressure at the same time. The proudest line on the pitch deck and the biggest risk on the cap table are the same fact.

**"The market maker is a neutral utility that just provides liquidity."** On a loan-plus-call deal the market maker holds call options on the token — it profits if the price rises and it controls a large share of the visible liquidity it can quote tightly or pull. Its payoff has no loss region. A neutral utility does not hold options on the thing it is a utility for.

**"A high FDV means the project is valuable."** FDV is price times *total* supply, including all the locked tokens that are not trading. NOVA's \$1 billion FDV sits on a \$100 million real market with 90% of supply locked. FDV is the number insiders use to mark their bags; it is not money anyone could actually realize. On a low float it is closer to a warning label than a valuation.

**"When a token crashes, everyone loses — the insiders got wrecked too."** Not on the same cost basis they didn't. NOVA fell 73% and *every* layer above retail stayed in profit, because they entered at a fraction of retail's price and sold on the way down. The chart being red tells you nothing about whether the early money lost; usually it didn't.

**"This is all illegal and someone will be arrested."** Mostly it is not illegal, and that is the uncomfortable part. No one has to break a rule for the same handful of firms to sit at every table and for the cheapest money to sell to the most expensive on a public calendar. The conflicts are *structural*, latent until exploited, and the legal system is still arguing over whether a token is even a security. The defense is not "the police will handle it." The defense is seeing the structure yourself.

## How it shows up in real markets

These are named, public episodes. Where a claim is contested, it is framed as reported or alleged, with the source. Figures are approximate and as of the events described.

### 1. a16z crypto and the fee-and-carry machine at scale

Andreessen Horowitz built the largest dedicated crypto venture practice, scaling from a \$350 million first fund in 2018 to a \$2.2 billion fund in 2021 and roughly \$4.5 billion in 2022 — the largest dedicated crypto venture vehicle to that point (a16z announcements, reported by outlets including *The Block* and *Fortune*). Run the "2 and 20" logic on a \$4.5 billion fund and the management fee alone is on the order of \$90 million a year, before any investment pays off, with carry of a fifth of every dollar of gain on top. This is the principal-agent machine at institutional scale: the fund earns whether or not retail does, and its biggest payday, carry, is computed on gains it is incentivized to realize fast and mark high. It is also a reminder that even the biggest, smartest funds took heavy paper markdowns when the 2022 cycle broke — the structure advantages insiders over retail, but it does not make them immune to the cycle.

### 2. Binance's own research: "exit liquidity for insiders"

The most striking source in this whole post is not a critic — it is Binance. In May 2024 Binance Research published *Low Float & High FDV: How Did We Get Here?*, which described the mechanism examined here in the exchange's own words: newly launched tokens "function as 'exit liquidity' for insiders" because of low initial circulating supply, and roughly \$155 billion of tokens were scheduled to unlock between 2024 and 2030, posing selling pressure without a matching rise in demand. When the largest venue in crypto publishes, in a formal report, that its listings are exit liquidity for insiders, the retail-facing marketing and the internal analysis are pointing in opposite directions — which is itself the whole thesis of this post.

### 3. The listing-fee fight: reported 15% of supply, denied

In November 2024, a public dispute broke out over what major exchanges charge to list a token. Simon Dedic of Moonrock Capital claimed that a project had been asked for 15% of its token supply — worth up to around \$100 million — in exchange for a Binance listing (as reported across crypto media). Binance co-founder He Yi (Yi He) denied there are fixed listing fees, stating that a project failing Binance's screening would not be listed regardless of money or tokens offered, and dismissing the claim as FUD. The numbers are contested and I am not asserting the 15% figure as fact — the point is the *shape* the public fight reveals: listing is a negotiation with a gatekeeper over how much of the token's own supply changes hands, which is exactly the exchange-as-stakeholder conflict on the incentive map. (For contrast, Justin Sun and Andre Cronje publicly argued Binance charges little while Coinbase demands large fees — the industry cannot even agree on the numbers, which tells you how opaque the listing layer is.)

### 4. DWF Labs and the market-maker-as-stakeholder allegations

In May 2024, *The Wall Street Journal* reported that Binance's own investigators had alleged DWF Labs manipulated the price of several tokens through roughly \$300 million of wash trades in 2023, and that Binance deemed the evidence insufficient and later removed the head of the surveillance team. DWF Labs denied the allegations, calling them unfounded and "competitor-driven FUD," and Binance said it found insufficient evidence of market abuse. These are reported and disputed claims, not established facts, and I frame them as such. But the *business model* the reporting describes — a market maker that "generally bought millions of dollars of a project's token at a discount and benefited when the price rose" — is precisely the loan-plus-call, stakeholder-not-referee structure on the incentive map. The controversy exists because the structure creates the temptation; whether any given firm crossed the line is a separate, contested question.

### 5. Jump and Terra: the market maker propping its own stake

Terra's UST was an algorithmic stablecoin meant to hold \$1 through a mechanism with its sister token LUNA. To bootstrap the peg, the system leaned on market makers. It later emerged through litigation that Jump's crypto arm had stepped in to buy UST and restore an earlier de-peg while, per the allegations, holding agreements to buy LUNA at a steep discount — an arrangement worth potentially hundreds of millions if confidence held. The point is the shape: the market maker propping the peg was a *stakeholder* in the system whose survival it was propping. When UST de-pegged for good in May 2022, roughly \$40 billion of combined UST and LUNA value evaporated. We cover the collapse in full in [the Terra/Luna collapse](/blog/trading/crypto/terra-luna-2022-collapse); on the incentive map it is the market-maker conflict scaled until it detonated.

### 6. Alameda and FTX: the conflict with every wall removed

The definitive blow-up took the structure to its logical extreme. FTX was an exchange; Alameda Research was a trading firm and market maker; both were controlled by the same person. In a healthy structure an exchange holding customer funds and a trading firm taking market risk are walled off. FTX and Alameda erased the wall: Alameda borrowed against FTX customer deposits, collateralized substantially by FTX's own FTT token — a token whose price FTX itself supported. When confidence cracked in November 2022, FTT collapsed and the roughly \$8 billion hole in customer funds could not be filled. We dissect it in [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried); on the incentive map it is what happens when a single party is the VC, the market maker, the exchange, and the counterparty all at once — every conflict in this post concentrated into one entity, with every separation removed. The related contagion took down over-levered funds and lenders too, which we trace in [the Three Arrows Capital collapse](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion).

## When this matters to you

You do not need to trade a single token for this to be useful — but if you ever look at one, the incentive map is the lens that makes the chart legible. The point is not "never buy a token." The point is that the price is not neutral information; it is the last and highest number in a chain, and your job is to figure out who is on the other side of it before you take the trade. The questions are concrete and mostly answerable from public data.

![The cui bono checklist: six questions to ask before buying a token, each paired with the red flag that reveals you may be the exit.](/imgs/blogs/cui-bono-the-incentive-map-of-crypto-8.webp)

Run the checklist on any token:

- **Cap table — who owns it, and at what cost?** If insiders entered below \$0.10 while you would be paying \$1 or more, their break-even is a fraction of yours, and they can profit selling into prices where you lose.
- **Float versus FDV — how thin is the float under the headline valuation?** A single-digit-percent float supporting a billion-dollar FDV means the price you see is propped on a sliver of supply, with a mountain of it scheduled to arrive.
- **Unlock cliffs — when does the locked supply become sellable?** Unlock dates are public and are the biggest predictable selling events in a token's life. A cliff in the next few months is a calendar of supply you will be competing against.
- **Market maker — who quotes it, and do they hold options?** A loan-plus-call market maker is a stakeholder with a no-loss payoff, not a neutral utility, and it controls liquidity it can pull.
- **Who's selling — whose cost basis sits below the current price?** If every potential seller entered far below you, you are structurally the buyer they sell to.
- **Narrative — who seeded the story you heard?** Paid influencers, undisclosed allocations, and coordinated "calls" are how demand is manufactured; if you cannot find who benefits from the story, assume it is not you.

If several answers come back red, you may be the exit. That is not a prediction that the token will fall — plenty of tokens with ugly cap tables have risen, and this is educational, not investment advice. It is a statement about *who you are trading against* and *how the structure is tilted*, which is the one thing the price itself will never tell you.

There is a final habit worth building, and it generalizes past crypto: read every "size" number — FDV, market cap, TVL, daily volume — as a claim about liquidity that may not exist. Each is a price multiplied by a quantity, and the price is only real to the extent that someone would actually pay it for the *whole* quantity. The recurring failure in this post, from the low-float FDV to the exit-liquidity fan, is the same arithmetic mistake made on purpose: treating a price that holds for a thin slice as if it held for the entire stack. Once you see that, the headline numbers stop being facts and become marketing you can choose to discount — and the question *cui bono* stops being rhetorical and becomes the most practical thing you can ask.

## Sources & further reading

Primary sources behind the real-world figures:

- Binance Research, *Low Float & High FDV: How Did We Get Here?* (May 2024) — the "exit liquidity for insiders" framing and the ~\$155 billion 2024–2030 unlock estimate. [Report PDF](https://public.bnbstatic.com/static/files/research/low-float-and-high-fdv-how-did-we-get-here.pdf).
- a16z crypto fund sizes (\$350M 2018, \$2.2B 2021, ~\$4.5B 2022) — a16z announcements, reported by *The Block* and *Fortune* (as of 2022).
- Binance listing-fee dispute (November 2024) — Simon Dedic / Moonrock Capital's claim of a reported 15%-of-supply request and He Yi's denial, as reported by CoinDesk, The Block, and DailyHodl. Contested; framed as reported.
- DWF Labs wash-trading allegations — *The Wall Street Journal* (May 2024), reported via CoinDesk and The Block; DWF Labs' denial ("competitor-driven FUD"). Contested; framed as reported.
- Crypto market-maker loan-plus-call deal structure — industry explainers from Flowdesk, Caladan, and blocmates (deal sized as % of supply, strike as a TWAP, ~1-year term, compensation via the call option).
- Terra/LUNA (~\$40B, May 2022) and FTX/Alameda (~\$8B hole, November 2022) — magnitudes as widely reported; see the linked deep-dives below.

Further reading on this blog:

- [Crypto VC and market makers: the real power structure](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub that this post synthesizes.
- [Why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) and [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) — the structural gap and the pipeline behind the cost-basis ladder.
- [The loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) and [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) — the two insider payoffs, in full.
- [Zero-sum, positive-sum, and the house](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from), [the greater fool and rational bubbles](/blog/trading/game-theory/the-greater-fool-and-rational-bubbles-the-musical-chairs-game), and [information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd) — the game theory of why retail keeps providing the exit.
- [The Terra/Luna collapse](/blog/trading/crypto/terra-luna-2022-collapse) and [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — the incentive map taken to its catastrophic extreme.

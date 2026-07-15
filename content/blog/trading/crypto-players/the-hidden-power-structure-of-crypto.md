---
title: "The Hidden Power Structure of Crypto: Who Actually Moves the Price You See"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A from-zero map of the layered machine behind a crypto token's price, from venture funds and market makers down to the retail buyer, and how each layer's action reaches the number on your screen."
tags: ["crypto", "market-structure", "tokenomics", "venture-capital", "market-makers", "float-vs-fdv", "crypto-exchanges", "retail-investing", "power-structure", "case-study"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — The price you see on a crypto exchange is the *last* link in a chain that started years earlier, in private rooms, at prices you never got. A small stack of players — venture funds, market makers, exchanges, foundations, whales, and paid influencers — sits above retail, and each one hands cheaper supply and more influence down to the next.
>
> - A token is not a stock. Insiders buy it years early and far cheaper, then sell into a public market with no IPO gate, no exchange-enforced lockup, and no quarterly disclosure.
> - A venture fund buying at \$0.02 is up **100×** on paper the day a token lists at \$2.00 — before retail can click "buy". Someone in the private round at \$0.10 is up 20×. You, buying at \$2.00, are up 1×.
> - A token can flash a **\$2 billion** fully diluted valuation while only ~5% of supply actually trades, so a **\$100 million** pool of real liquidity props up the giant headline number.
> - Because that float is thin, a single market order can walk the price 15% in one trade. The price is not "the market's fair value" — it is wherever the next order lands.
> - The famous blow-ups (FTX/Alameda, Jump/Terra) were not bugs in this structure. They were the structure pushed to its limit.

Here is a question worth sitting with: when you open an exchange and a token says \$2.00, who decided that number?

The instinct is to say "the market" — millions of buyers and sellers meeting at a fair price. That instinct is almost entirely wrong for a freshly launched crypto token. The \$2.00 is the *output* of a machine most buyers never see: a fund that bought the token at two cents three years ago, a trading firm that was handed a pile of the tokens to quote with, an exchange that decided today was listing day, and a set of paid voices who spent the week telling you it was going to fly. The number on your screen is real. What is hidden is everything that produced it.

The diagram above — and the one just below — is the mental model this entire piece tours. It is a layered stack. At the top sit the players with the cheapest supply and the most influence; at the bottom sits retail, buying last, at the finish-line price. Influence and cheap supply flow *down* the stack, layer by layer, and what lands at the bottom is the quote you trade against.

![A layered map of crypto's power structure: venture funds, market makers, exchanges, foundations, whales, and influencers sit above retail, with influence and cheap supply flowing down into the price on your screen.](/imgs/blogs/the-hidden-power-structure-of-crypto-1.webp)

This is not a conspiracy, and it is mostly not illegal. It is a *plumbing* story. Traditional equity markets have venture capitalists and market makers too. What makes crypto different is that the same plumbing runs faster, with far weaker disclosure, and — crucially — the earliest buyers can exit on a public market *years* sooner than an equity investor ever could. That speed and that opacity are what turn ordinary financial roles into sharp conflicts of interest.

This is the master map for a whole series. Later posts drill into each player — [why a token is not a stock](/blog/trading/crypto/crypto-vc-and-market-makers), how a market maker's loan-and-option deal really works, how unlock cliffs are priced. This one builds the structure from zero, defines every term, walks the arithmetic with real dollar figures on one running example, grounds it in named, sourced cases, and ends with the questions to ask before you put money into any token. No predictions, no advice, no shilling. Just the machinery.

## Foundations: the words this power structure turns on

Before we name a single fund, we need a shared vocabulary. Crypto borrows half its words from finance and invents the other half, and the whole story collapses into fog if any of these is fuzzy. Read this section even if you think you know it — the conflicts later hinge on the *precise* meaning of "token", "float", and "FDV".

### The players, in one breath

- **A token** is a unit of value recorded on a blockchain (a shared, public ledger many computers keep in agreement). Some tokens are money-like (a stablecoin pegged to a dollar); most that we discuss here are "project" tokens whose price floats up and down.
- **A venture fund (VC)** is a pool of investors' money that buys into projects very early, hoping a few winners pay for all the losers. Crypto VCs buy *tokens* (often before the token even trades), not just company shares.
- **A market maker (MM)** is a trading firm that continuously offers to both buy and sell a token, so that anyone else can trade at any moment. It earns the tiny gap between its buy and sell prices — the *spread*.
- **An exchange** is the venue where tokens trade — Binance, Coinbase, and dozens of others. It matches buyers with sellers and decides which tokens get *listed* (allowed to trade there).
- **A token foundation or treasury** is the entity that holds a project's own reserve of tokens — often an enormous fraction of the total supply — to fund development, pay grants, or reward stakers.
- **A whale** is any single holder large enough to move the price by trading. An **OG** ("original gangster") is an early, influential figure or fund from a previous cycle.
- **A KOL** ("key opinion leader") is an influencer — a person with an audience who posts about tokens. Many are quietly paid, or given cheap tokens, to do so.
- **Retail** is you: the general public, buying on a public exchange with your own money.

That is the cast. The rest of this post is about *how money flows between them* and *how that flow becomes a price*.

### A token is not a stock (the one-paragraph version)

This gap deserves its own future post, but you need the short version now, because it is the crack that makes everything else possible. When you buy a *share* of a company, you own a legal slice of that company: a claim on its profits, a vote protected by company law, and — if it is public — the protection of an IPO process, enforced lockups that stop insiders dumping on day one, and quarterly financial disclosure. A **token** usually gives you none of that. No profit claim, no legally enforced lockup, no mandatory disclosure. It gives you a price that moves, and sometimes a vote inside the project's own software. So the person who bought early can sell to you, on a public market, with far fewer rules in the way. Hold that thought — it is why the "price ladder" below is so steep.

The differences are worth laying side by side, because each row is a rule that protects a stock buyer and is usually *absent* for a token buyer:

| Protection | Public stock | Typical project token |
|---|---|---|
| Legal claim on profits | Yes — you own equity | Usually none |
| Vetting before public sale | IPO process, regulator review | Often just a listing decision |
| Insider lockup | Enforced by exchange/law (often 90–180 days) | A *promise* in a vesting schedule, rarely enforced by law |
| Mandatory disclosure | Quarterly financials, insider-sale filings | Usually none required |
| Who can buy early and cheap | Restricted, disclosed | Private rounds, lightly disclosed |
| Time from insider entry to public exit | Years (lockups, IPO timelines) | Can be months |

Read that last row twice. In equities, the machinery deliberately *slows down* the distance between an insider's cheap entry and their public exit, and forces them to tell you when they sell. In crypto, that distance is short and mostly invisible. None of this makes tokens a scam — it makes them a market with fewer speed bumps, where the burden of seeing the structure falls on you instead of on a regulator. That is the entire reason this post exists.

### Primary market vs secondary market

This is the single most important distinction in the whole post, so we give it a picture.

![Primary vs secondary market: seed at $0.02, private at $0.10, then TGE and a public listing at $2.00 where retail finally buys, last in line.](/imgs/blogs/the-hidden-power-structure-of-crypto-2.webp)

- The **primary market** is where a token is sold *privately*, before it trades publicly: the seed round, the private round, the strategic round. These are gated — you need to be a fund, a connected individual, or an invited party to get in. Prices here are low, because the token has no public market yet and the buyers are taking early risk.
- **TGE** ("token generation event") is the moment the token is created and first becomes tradeable. Around TGE the token gets **listed** on an exchange, and its **first public price** is set.
- The **secondary market** is the public exchange — the order book where anyone can buy or sell. This is the *only* market retail can reach. By the time you get here, every primary-market buyer already owns their tokens, bought cheaper.

Keep the shape in your head: three private rounds, then a public opening. The word "market" hides the fact that there were really several markets, and you were only allowed into the last one.

### Float, FDV, and circulating supply

Three numbers decide how a token's price behaves. Define them once and carefully:

- **Total supply** — every token that will ever exist (or a stated maximum). Say a token, which we will call **NOVA**, has a total supply of **1,000,000,000** (one billion) tokens.
- **Circulating supply (the "float")** — the tokens that are *actually in public hands and free to trade right now*. The rest are locked: held by the foundation, or still vesting for the team and the VCs. If only 50,000,000 of NOVA's billion are unlocked, the float is **5%**.
- **Market capitalization ("market cap")** — the price times the *circulating* supply. This is the value of what actually trades.
- **Fully diluted valuation (FDV)** — the price times the *total* supply, as if every locked token were already out. This is the giant headline number.

The whole "VC coin" critique lives in the gap between market cap and FDV, and we will compute it in full below.

### The order book and liquidity (the short version)

A public price does not come from nowhere. It comes from an **order book**: a live list of everyone's resting offers to buy (**bids**) and to sell (**asks**). The best bid and best ask sit closest to each other, and the midpoint between them is roughly "the price". **Liquidity** is how much size is resting in that book near the current price. A **liquid** book has large orders stacked at every price, so a big trade barely moves it. A **thin** book has almost nothing resting, so even a small trade jumps the price. We will see exactly how far a thin book can be pushed in Section 3.

### A first worked example: the price ladder

Now the payoff for all that vocabulary. Let's watch one number — the *paper multiple* — climb as we move up the layers of NOVA's cap table.

![The price ladder on one token: a seed buyer at $0.02 is up 100x at the $2.00 launch, a private buyer at $0.10 is up 20x, and retail buying at $2.00 is up 1x.](/imgs/blogs/the-hidden-power-structure-of-crypto-3.webp)

#### Worked example 1: what \$1,000 becomes at each layer

Suppose NOVA launches publicly at **\$2.00**. Walk \$1,000 of buying power up the ladder:

- **Seed VC (entry \$0.02).** \$1,000 buys 1,000 ÷ 0.02 = **50,000 NOVA**. At the \$2.00 launch those are worth 50,000 × \$2.00 = **\$100,000**. That is a **100×** paper gain.
- **Private round (entry \$0.10).** \$1,000 buys 10,000 NOVA. At \$2.00 they are worth **\$20,000** — a **20×** paper gain.
- **Retail (entry \$2.00).** \$1,000 buys 500 NOVA, worth exactly **\$1,000** at launch. A **1×** "gain": you paid full price. Every discount above you is already gone.

The one-sentence intuition: *on launch day, the same token is a 100× win for the person at the top of the ladder and a coin-flip for the person at the bottom — and the only difference is who got in first and cheapest.*

But notice the word *paper*. The seed VC's 100× is a mark-to-market number, not cash. They hold 50,000,000 NOVA (from our \$1,000 slice, scaled up: a fund that put in \$1,000,000 at seed holds 50,000,000 tokens worth \$100,000,000 on paper). To turn that into real money they have to *sell* — and, as Section 3 will show, selling that much into a thin public book would crater the price long before they got \$100 million out. This single fact — that a huge paper gain can only be realized by selling into someone else's buying — is the engine of everything below. It is why insiders need a deep pool of retail demand to exit into, why they care so much about narrative and listings and timing, and why the unlock calendar matters. The whole power structure exists to convert paper multiples at the top into realized cash, and the only place that cash can come from is the buyers at the bottom.

#### Why the early price is lower (the honest version)

It would be unfair — and it would make you a worse analyst — to read the ladder as pure theft. The early price is lower for real reasons, and a good defense starts by respecting them. At the seed round, the token does not trade, the product may not exist, and most such bets go to zero; the fund is buying a lottery ticket on a team and a promise, and it is compensated for that risk with a low price and a large allocation. The venture layer earns its multiple the same way an angel investor in a startup does: by being right early, on the minority of bets that survive. The problem this post is about is not that insiders pay less — that is how early-stage risk capital works everywhere. The problem is *the specific crypto twist*: unlike a startup investor who waits years for an IPO with lockups and disclosure, the token investor can often sell to the public within months, into a thin market, with no obligation to tell anyone. Same principle (early risk, lower price); radically fewer speed bumps between the insider's gain and your loss. Keep both halves in your head — it keeps you from being either naive or a conspiracy theorist.

## 1. The seven layers, top to bottom

Now we walk the whole stack. Each layer answers the same three questions: **who are they**, **how do they make money**, and **what is their one lever on the price you finally see?** The table below is the reference; the sections after it add the color.

![A matrix of the seven layers: for venture funds, market makers, exchanges, foundations, whales, KOLs, and retail, it lists who they are, how they get paid, and their lever on the price you see.](/imgs/blogs/the-hidden-power-structure-of-crypto-4.webp)

**Layer 1 — Venture funds (the kingmakers).** They buy tokens years early at \$0.02–\$0.10 (see the ladder above). Their money isn't just capital; a well-known fund's backing is itself a *signal* that pulls other buyers in. Their lever: they help choose which projects launch at all, and they hold cheap supply that they will eventually sell into public demand. A named example we will return to: **a16z crypto**, which announced a **\$4.5 billion** fourth crypto fund in May 2022, bringing its total crypto funds raised to more than **\$7.6 billion** at the time ([a16z](https://a16zcrypto.com/posts/article/crypto-fund-four/)). A fund that size can seed an entire sector's worth of tokens.

**Layer 2 — Market makers (the liquidity engine).** A token literally cannot trade smoothly without someone quoting both sides of the book. MMs provide that, earning the spread. But the crypto-native version of the deal is where the conflict lives: projects often *lend* the market maker tokens to quote with and hand it *call options* that pay off if the price rises — so the MM profits from the very launch it is smoothing. We work the arithmetic of that deal in Section 5. Named examples: **Wintermute**, **Jump**, and the late **Alameda Research**.

**Layer 3 — Exchanges (the gatekeepers).** Exchanges are not neutral pipes. They decide what lists and when, they run their own venture arms that invest in tokens they may later list, and their sheer size moves markets. When the largest venue lists a token, that listing is itself a price event. **Binance** was the largest spot exchange by volume in 2024, handling on the order of \$7 trillion in spot trades and roughly 40% of centralized-exchange spot volume, though its share slipped over the year ([TokenInsight via ChainCatcher](https://www.chaincatcher.com/en/article/2162313)). A yes-or-no listing decision from a venue that large is worth a fortune.

**Layer 4 — Foundations and treasuries (the on-chain central banks).** A project's foundation often holds a huge fraction of total supply — the locked 95% in our NOVA example lives here. It can sell to fund itself, grant tokens to partners, or stake them. Its lever is the least discussed and possibly the largest: it releases new supply onto the market on a *known schedule* (the unlock calendar), and it is the single biggest holder in the room.

**Layer 5 — OGs and whales.** Early holders with outsized wallets. Because the tradeable float is thin, one whale can drain an order book — or, more quietly, move size through an OTC ("over-the-counter") desk that arranges the trade off the public book so it doesn't visibly crash the price. Their lever: raw size against thin liquidity.

**Layer 6 — KOLs and influencers.** The narrative layer. Projects and funds allocate tokens or pay for promotion; the influencer's audience becomes the demand that absorbs the supply the layers above want to sell. Their lever: manufacturing attention, on a schedule that tends to peak right when insiders can sell.

**Layer 7 — Retail (you).** The public buying on the screen, at \$2.00 or more, after every discount above has been taken. In the mechanics of the stack, retail plays one structural role: it is the **exit liquidity** — the buyer that lets everyone above finally turn a paper gain into cash. That sounds harsh. It is simply what the arithmetic says, and seeing it clearly is the whole point of the defensive section at the end.

Notice the pattern down the layers: **each layer's profit is the layer below's cost.** That is not a moral claim; it is an accounting identity in a market where insiders sell and the public buys.

### Do the layers "collude"? Mostly no — and that is the point

It is tempting to picture a smoke-filled room where all seven layers agree to fleece retail. That picture is wrong and it makes you *less* prepared, not more. Most of the time there is no meeting and no plot. The venture fund simply wants its portfolio token to be liquid so it can eventually sell. The market maker simply wants the spread and its options to pay off. The exchange simply wants trading fees and a hot listing. The influencer simply wants the promo fee. Each actor is following its own local incentive — and the incentives happen to *stack*, because they all point the same way: get the token listed, get a narrative going, get retail buying, so that supply can flow down and cash can flow up.

That is why "aligned incentives" marketing rarely survives the arithmetic. The layers do not need to conspire; they are already aligned by structure, the way water does not need to organize itself to flow downhill. The danger to retail is not a secret cabal — it is a set of separately rational players whose separate exits all route through the same order book, at roughly the same moments (listing, narrative peak, unlock). When you buy a hyped token at launch, you are not necessarily the victim of a scheme. You are, more often, the last independent actor in a system whose earlier actors were all structurally incentivized to sell to you. The fix is not outrage; it is the checklist at the end.

## 2. Float vs FDV: how a tiny float invents a giant number

Return to those two numbers — market cap and FDV — because the gap between them is where crypto's most misleading headline lives.

![Float vs FDV shown as a column: 95% of supply is locked and worth $1.9 billion on paper, while only the 5% float, worth $100 million, actually trades and sets the price.](/imgs/blogs/the-hidden-power-structure-of-crypto-5.webp)

#### Worked example 2: a \$100 million token that reads as \$2 billion

NOVA again: total supply 1,000,000,000, launch price \$2.00, and only **5%** (50,000,000 tokens) circulating.

- **Fully diluted valuation** = price × total supply = \$2.00 × 1,000,000,000 = **\$2,000,000,000** (\$2 billion). This is the number that lands in headlines and "top 50 by valuation" lists.
- **Market cap** = price × circulating supply = \$2.00 × 50,000,000 = **\$100,000,000** (\$100 million). This is the value of everything that can actually be bought or sold today.

So the token *reads* as a \$2 billion project while only \$100 million of it truly trades. The other \$1.9 billion is locked supply — held by the foundation, the team, and the VCs — waiting to be released on the unlock schedule. That is not a rounding detail. It means the price is being set by a pool 20 times smaller than the headline suggests, and it means there is \$1.9 billion of supply that *will* arrive later, at prices the current buyers help set.

This is not a fringe pattern. Binance Research studied tokens launched in 2024 and found a median market-cap-to-FDV ratio of about **12%** — meaning only ~12% of the fully diluted supply was in circulation at launch — and estimated that roughly **\$155 billion** of tokens were scheduled to unlock between 2024 and 2030 ([Binance Research, May 2024](https://www.binance.com/research/analysis/low-float-and-high-fdv-how-did-we-get-here)). Our clean 5% is on the aggressive end, but the shape is the real shape of the market.

The one-sentence intuition: *a low float is a magnifying glass — it makes the price (and the FDV) look big on a small amount of real money, and it guarantees a wave of future supply the buyers can already see coming.*

## 3. The order book: how a small player moves a big number

We said the price is "wherever the next order lands". Here is what that means, mechanically, when the float is thin.

![A thin order book: sparse red asks above the $2.00 mid and green bids below, with a 250,000-token market buy walking the price up to $2.30, a 15% jump in one trade.](/imgs/blogs/the-hidden-power-structure-of-crypto-6.webp)

A market **buy** order does not get "the price". It climbs the ladder of resting **asks** (sell orders), filling the cheapest first, then the next, then the next — paying more at each rung — until it is filled. The last rung it reaches becomes the new price. When the book is thin, that last rung can be far away.

#### Worked example 3: 15% slippage from one order

Suppose NOVA's book, just above the \$2.00 mid, has only these resting asks:

| Ask price | Size (tokens) | Cost to clear this level |
|---|---|---|
| \$2.01 | 20,000 | \$40,200 |
| \$2.05 | 35,000 | \$71,750 |
| \$2.10 | 50,000 | \$105,000 |
| \$2.20 | 70,000 | \$154,000 |
| \$2.30 | 75,000 | \$172,500 |

That is 250,000 tokens resting in total. Now a buyer sends a **market buy for 250,000 NOVA**. It sweeps every level: 20k at \$2.01, then 35k at \$2.05, and so on, up to the final 75k at \$2.30. Add the costs and the buyer pays 40,200 + 71,750 + 105,000 + 154,000 + 172,500 = **\$543,450** for the 250,000 tokens — an average fill of about **\$2.17**. But the *last print*, the new "price" everyone now sees, is **\$2.30**. From a \$2.00 mid, that is a **+15%** move, caused by a single order of a few hundred thousand dollars.

The difference between the \$2.00 you expected and the higher average you actually paid is called **slippage**. In a deep, liquid book, \$543,000 would move the price a fraction of a percent. In a thin book, it moves it 15%. Nothing about the project changed. The only variable was how little was resting in the book.

The one-sentence intuition: *"the price went up 15%" and "someone spent half a million dollars" can be the exact same event — thin float turns small money into big candles, in both directions.* This is why a low-float token can look explosively bullish on the way up and equally violent on the way down. (The mechanics of slippage, resting liquidity, and reflexivity get their own dedicated post; here we only need the fact that thin books move easily.)

The mirror image is just as important. A market **sell** order walks *down* the bids, filling the highest bid first, then the next lower, until it is done — so a large sell prints a lower and lower price. In our book, the bids were \$1.99 (20,000), \$1.98 (30,000), \$1.95 (45,000), and \$1.90 (60,000). A sale of 155,000 tokens clears all of them and drops the last print to \$1.90, a −5% move, for a total received of about \$296,000 (an average of ~\$1.91). Now recall Section 2: the seed VC could be holding tens of millions of tokens. Their sell order does not walk down four rungs — it falls off a cliff. That asymmetry, huge holders against a shallow book, is the whole risk retail is exposed to when the float is small.

There is one more actor to name here: the market maker *controls* the book you just watched. It is the firm placing most of those resting bids and asks. In calm times it quotes a tight spread and makes the token feel liquid. But it can widen its spread or pull its orders whenever it wants — during a sell-off, or right before an unlock, or when it senses informed flow on the other side. So the "liquidity" you see is not a fixed feature of the token; it is a service that can be dialed down exactly when you most need it. A thin book plus a market maker who can step back is how a token that felt perfectly tradeable at noon becomes untradeable by dinner.

### The escape hatch: OTC desks

If a large holder cannot sell into a thin book without cratering the price, how do they ever exit? The answer is the **OTC** ("over-the-counter") desk — a private market that sits beside the public order book. Instead of dumping 10,000,000 tokens into a book that can only absorb a few hundred thousand, a whale calls an OTC desk, which finds a counterparty (another fund, a treasury, a family office) willing to take the whole block at a single negotiated price. The trade happens *off* the public book, so it never shows up as a wall of sell orders and never prints the crash it would have caused. This is why you can see a token grind sideways for weeks while, invisibly, hundreds of millions of dollars of it change hands.

For retail, the lesson is uncomfortable but useful: the absence of a visible dump does not mean insiders are not selling. Much of the largest flow is routed precisely so that you cannot see it on the chart. The supply still arrives eventually — the OTC buyer now holds tokens they may later sell into the public market — but the timing and the size are hidden from the tape you are watching. OTC desks get their own deep dive later in the series; here, just add them to your mental map as the reason "the whales aren't selling, look at the chart" is a weak argument.

## 4. Unlocks: the supply the calendar already promised

Section 2 left \$1.9 billion of NOVA locked. That supply does not vanish — it arrives on a schedule the whole market can read in advance, called the **vesting schedule** or **unlock calendar**. A **cliff** is a date on which a large tranche unlocks all at once; after a cliff, tokens typically **vest** (release gradually) over months.

Unlocks are the cleanest example of the whole power structure, because they are *scheduled*. Everyone can see the supply coming; the only question is what price it arrives at.

#### Worked example 4: an unlock that dwarfs the float

Say NOVA's seed VCs and team hit a cliff and **100,000,000** tokens (10% of total supply) unlock. Compare that to the tradeable float:

- Circulating float before the unlock: **50,000,000** tokens.
- Newly unlocked supply: **100,000,000** tokens — **twice the entire float**.

The seed VC's cost basis is \$0.02. At \$2.00, even after a brutal 50% price drop to \$1.00, they are still up 50×. So they can afford to sell aggressively and still make a fortune — while a retail buyer at \$2.00 is underwater the moment the price cracks. If even 10% of the unlocked tranche (10,000,000 tokens) hits the market, recall from Section 3 that the *entire* resting ask book in our example was only 250,000 tokens. Ten million tokens into a book that thin is not a sale; it is an avalanche.

The one-sentence intuition: *an unlock is a pre-announced transfer of supply from people with a near-zero cost basis to whoever is buying — and the calendar tells you the date in advance.* This is why experienced traders watch unlock calendars the way equity traders watch earnings dates.

## 5. The market maker's loan-and-option deal

We keep saying market makers "profit from the token they smooth". Here is the specific contract, because it is the most elegant conflict in the whole structure — important enough that a later post is devoted entirely to it.

When a new token lists, it needs a market maker to quote both sides so it is tradeable from minute one. But a brand-new project has no cash to pay a trading firm. So the common crypto deal works like this: the **project lends the market maker a pile of its own tokens** to quote with, and in exchange the MM gets **call options** — the right, but not the obligation, to buy a set number of tokens at a fixed **strike price** later. If the token rises above the strike, those options are worth money; if it falls, the MM simply doesn't exercise them and hands the borrowed tokens back.

#### Worked example 5: the MM that wins if the token flies

Suppose NOVA's team lends a market maker 20,000,000 tokens to quote with, and grants it call options on **10,000,000** tokens at a **\$2.00** strike.

- If NOVA rises to **\$4.00**, the options are worth their intrinsic value: (\$4.00 − \$2.00) × 10,000,000 = **\$20,000,000**. The MM buys at \$2 and can sell at \$4.
- If NOVA falls to **\$1.00**, the options are worthless — the MM won't pay \$2 for a \$1 token — so it just returns the borrowed tokens. Its downside is capped; its upside is huge.

Now ask the obvious question: which outcome does the market maker prefer? The one where the token it is quoting goes *up*. A call option is a bet on the price rising, so the firm setting your spreads and shaping your order book has a direct financial interest in the token climbing. That is not necessarily illegal — options are a normal way to align a service provider with a project — but it means the "neutral liquidity provider" is quietly long the token, by design. (This is the single most load-bearing mechanic in how tokens are launched, and it gets a full worked treatment in its own post.)

The one-sentence intuition: *the crypto market maker is often not a neutral referee but a paid participant whose payout rises with the price — so its incentives point the same way the insiders' do.*

## How it all shows up in the price on your screen

Now assemble the pieces. No single hand sets a token's price. Instead, several upstream levers push at once, and they all converge on two things: the **order book** and **retail demand**. Where those two meet is your quote.

![A flow diagram: market makers, exchanges, and foundations push on the thin order book; venture funds and KOLs push on retail demand; the order book and demand meet at the price on your screen.](/imgs/blogs/the-hidden-power-structure-of-crypto-7.webp)

Trace the arrows:

- The **market maker** sets the spread and how much rests in the book. A thin, MM-controlled book (Section 3) means small orders move price a lot.
- The **exchange** decides when trading opens at all; the listing itself is a demand spike, as buyers who couldn't reach the token before rush in.
- The **foundation** releases supply on the unlock calendar (Section 4), adding sell pressure at known dates.
- The **venture fund's** backing acts as a *demand signal* — "smart money is in this" — and the fund's cheap supply is the eventual sell side.
- The **KOLs** manufacture the narrative that turns attention into buy orders.

Two mechanisms, five upstream players, one output. When people say "the market moved", what usually moved was one of these levers — a listing, an unlock, a fund's public endorsement, a coordinated narrative push — landing on a book too thin to absorb it. The price you see is not a vote of a million independent minds. It is the meeting point of a handful of upstream decisions and the demand they engineered.

There is a feedback loop that makes this even sharper, and it has a name: **reflexivity**. A rising price is itself a marketing tool. When NOVA jumps 15% on a thin book, that green candle becomes the KOL's next post, which pulls in more buyers, which pushes the price higher, which justifies the next post. Price and narrative feed each other on the way up. The trouble is that the loop runs in reverse just as well: the first real wave of unlock supply breaks the uptrend, the narrative curdles, the buyers who arrived late try to exit into the same thin book, and the same reflexivity that lifted the price now accelerates its fall. The upstream players understand this loop and time their supply to it — selling into strength, into the demand the narrative created — which is why the exit so often happens right when the story is loudest. If you only remember one sentence from this section, make it this: *in a thin-float token, the price is the marketing, and the people who control the supply know exactly when the marketing has done its job.*

## Common misconceptions

**"The price is the market's fair estimate of value."** For a thin-float token, the price is whatever the last order did to a nearly empty book. Section 3 showed a single order moving it 15% with no change in the project at all. Price is a real-time record of order flow, not a considered valuation.

**"A high fully diluted valuation means the project is huge."** FDV multiplies today's price by *all* future supply, most of which is locked and not for sale. A \$2 billion FDV can sit on \$100 million of real, tradeable value (Worked example 2). FDV is a projection, not a market's verdict.

**"If big-name VCs are in, retail and insiders are aligned."** They are aligned on wanting the token to exist and trade. They are *not* aligned on price: the VC's cost basis might be \$0.02 and yours \$2.00. When the VC's tokens unlock, their profitable exit is your falling price. Shared enthusiasm is not shared incentives.

**"Market makers just provide neutral liquidity."** Often they hold call options that pay off if the price rises (Section 5), so they are structurally long the token they quote. "Neutral" describes the service, not the incentive.

**"Insider selling is illegal, like in stocks."** In equities, insiders face lockups and disclosure rules enforced by law and the exchange. Most tokens have neither — the lockup is a *promise in a schedule*, not a legally enforced restriction, and there is usually no requirement to disclose sales. That is exactly the gap between a token and a stock.

**"Low float is bullish because supply is scarce."** Low float cuts both ways. It makes the price easy to push *up* on small buying, which feels bullish — and just as easy to push *down* when locked supply unlocks (Section 4). Scarcity today is guaranteed abundance tomorrow, on a schedule.

**"High trading volume means the market is deep and safe."** Volume and depth are different things. Volume is how much *traded* over a period; depth is how much is *resting in the book right now*. A token can show large 24-hour volume — sometimes inflated, sometimes real churn among a few players — while the book at this instant is thin enough that your order moves it 15%. Always check the live order-book depth, not just the volume headline.

**"The whales are all selling, so I should sell too (or they're buying, so I should buy)."** Wallet-watching tools now make large holders' moves visible, which spawns a reflexive game: people trade on what whales appear to do, and sophisticated whales know they are being watched and can move funds to mislead. A visible transfer to an exchange might be a sale, or collateral, or a decoy. Following on-chain "smart money" is a real signal, but it is a noisy one that the smart money can deliberately corrupt.

**"Being early means clicking fast on launch day."** The people who were genuinely early were in the seed round years ago, at \$0.02. On a public launch you are, by definition, in the secondary market — the last market in the chain. Speed on launch day is not earliness; it is just being first among the people who were all late.

## How it shows up in real markets

The structure above is not theoretical. Here are named episodes where each layer's behavior is on the public record. Every figure below is sourced and dated; contested claims are framed as *alleged* or *per the settlement*, never asserted as proven fact.

### 1. FTX and Alameda — the structure at its absolute limit

The clearest proof that this power structure exists is what happens when one person owns several layers at once. Sam Bankman-Fried owned both the exchange **FTX** and the trading firm **Alameda Research** — a venue *and* a market maker/prop desk trading on it. According to the U.S. Department of Justice, he misappropriated as much as **\$8 billion** of FTX customer funds, much of it funneled to Alameda; in March 2024 he was sentenced to **25 years** in prison and ordered to forfeit **\$11 billion** ([DOJ](https://www.justice.gov/archives/opa/pr/samuel-bankman-fried-sentenced-25-years-his-orchestration-multiple-fraudulent-schemes)).

Every layer in this post appeared in that one collapse. There was a self-issued token, **FTT**, that FTX created and used as collateral — a foundation/treasury holding its own supply, marked at a value the market never truly tested. There was a market maker (Alameda) quoting and trading the very tokens its sister exchange listed. There was an exchange choosing what to list and how customer funds were held. And there was retail, whose deposits were the real money underneath the paper. FTX wasn't a strange exception to the structure — it was the structure with every conflict-of-interest wall removed, so exchange, market maker, treasury, and insider were the same wallet. When a rival announced it would sell its FTT, the token cracked, the collateral evaporated, and customers rushed to withdraw money that had already been spent. We cover it in depth in [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).

### 2. Jump and Terra — a market maker propping a price

When the algorithmic stablecoin **TerraUSD (UST)** wobbled off its \$1 peg in May 2021, a market maker stepped in to buy it — and the market read the recovery as the algorithm working. In December 2024, **Tai Mo Shan**, a subsidiary of **Jump Crypto**, agreed to pay **\$123 million** to settle SEC charges that it negligently misled investors about UST's stability; the SEC said it bought more than **\$20 million** of UST in a way that made the market believe the peg mechanism was self-correcting when the price was in part being held up by those purchases. Tai Mo Shan settled *without admitting or denying* the findings ([SEC, Dec 2024](https://www.sec.gov/newsroom/press-releases/2024-212)). It is a textbook case of a big player's buying being mistaken by the public for organic price action. The wider Terra collapse is covered in [the Terra-Luna 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse).

### 3. Wintermute — how large the MM footprint really is

You rarely see market makers, so it is easy to underestimate their size. In September 2022, **Wintermute**, one of the largest algorithmic crypto market makers, lost about **\$160 million** in a hack of its decentralized-finance operations, caused by a flawed "vanity address" tool ([TechCrunch, Sep 2022](https://techcrunch.com/2022/09/20/crypto-market-maker-wintermute-loses-160-million-in-defi-hack/)). The revealing part was the firm's response: its CEO said Wintermute remained solvent with roughly twice that amount left in equity. A single trading firm you have probably never traded with directly can carry hundreds of millions in inventory across the tokens you *do* trade — that is the scale of the liquidity layer.

### 4. a16z — when backing itself is a price signal

The venture layer's lever is not only cheap supply; it is legitimacy. When **a16z crypto** announced its **\$4.5 billion** fourth fund in May 2022 — the largest crypto venture fund raised to that point, lifting its total crypto funds above **\$7.6 billion** ([a16z](https://a16zcrypto.com/posts/article/crypto-fund-four/)) — the news itself moved sentiment across the sector. When a fund of that size leads a round, other investors follow, exchanges take the listing more seriously, and retail reads the logo as a green light. The backing becomes part of the price before the token even trades.

### 5. A Binance listing as a price event

Because the largest exchange is also a gatekeeper, its listing decision is worth a fortune to a project. Binance handled roughly 40% of centralized-exchange spot volume in 2024 ([TokenInsight via ChainCatcher](https://www.chaincatcher.com/en/article/2162313)); a token that gets listed there gains access to the deepest pool of buyers in the market overnight. The "Binance effect" — a sharp price jump on listing announcement — is well known enough that traders position for it, which is itself a reminder that a listing is a *decision made by a player*, not a neutral fact of nature. The exchange-as-player is covered in [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).

### 6. The 2024 low-float, high-FDV wave

Finally, the float-vs-FDV pattern from Section 2 became the defining launch style of 2024. Binance Research documented a wave of tokens listing with a median circulating supply near **12%** of total and FDVs in the billions, with an estimated **\$155 billion** of unlocks scheduled through 2030 and roughly **\$80 billion** of fresh demand needed just to hold prices flat against that supply ([Binance Research, May 2024](https://www.binance.com/research/analysis/low-float-and-high-fdv-how-did-we-get-here)). Many of those tokens launched at eye-watering valuations and then drifted down for months as unlocks arrived — the exact dynamic Worked examples 2 and 4 predict, playing out across an entire cohort.

## The retail-defense takeaway: who is on the other side of your trade?

You cannot out-fund a16z, out-quote Wintermute, or get the seed price. That is fine — the goal here is not to beat the insiders but to *see* them, and to stop being surprised. The whole post compresses into one question you ask before buying any token: **who is on the other side of this trade, and what is their cost basis?** The checklist below turns that question into things you can actually look up.

![A retail-defense checklist grid: six questions, each with where to look (cap table, float vs FDV, MM terms, unlock calendar, KOL disclosures) and the red flag that means step back.](/imgs/blogs/the-hidden-power-structure-of-crypto-8.webp)

1. **Who funded this, and at what price?** Look up the token's cap table and seed price (later posts in this series go deep on reading these on-chain). Red flag: insiders got in below ~5% of today's price, so they are wildly profitable no matter where it goes next.
2. **How much supply actually trades?** Compare market cap to FDV (the float). Red flag: float under ~15% with an FDV in the billions — a small pool of real money holding up a huge headline, with lots of locked supply coming.
3. **Who makes the market, and how are they paid?** Look for the market maker's deal terms. Red flag: a loan-and-options structure that pays the MM more when the price rises, so the "neutral" party is long.
4. **When do tokens unlock?** Find the vesting schedule and unlock calendar. Red flag: a large cliff within the next few months — pre-announced sell pressure aimed at the current price.
5. **Who is telling me to buy?** Check whether the influencers pushing it disclose paid promotion or token allocations. Red flag: paid or allocated promoters with no disclosure — manufactured demand, arriving right when insiders can sell.
6. **How thin is the book?** Glance at the order-book depth on the exchange. Red flag: a thin book where your own modest order would move the price several percent — you are the liquidity, not a passenger on it.

None of these red flags means "this token will go down". Plenty of low-float, VC-backed, MM-supported tokens have gone up a great deal. They mean something narrower and more useful: *you now know what game you are in and who is sitting across the table.* That knowledge is the entire edge retail can realistically have.

## When this matters to you

This is educational, not advice, and nothing here is a recommendation to buy or avoid any token. But the structure changes how you should read a few everyday moments:

- When a token "pumps" on launch, you now know that can be a thin book plus a listing plus a seeded narrative — not a verdict on the project.
- When you see a giant FDV, you know to ask what the *float* is before you are impressed by the headline.
- When a fund or an influencer is loudly bullish, you know to ask what their cost basis is and when their tokens unlock, because their exit is your price.
- When a chart looks violently strong, you know a thin float moves just as violently down, and the same people who could push it up can let it fall.

It also reframes what "doing your research" means. Reading a project's whitepaper and following its official account tells you what the top of the ladder *wants* you to think. The research that actually protects you is structural: the cap table, the float, the unlock calendar, the market-maker terms, the disclosure (or lack of it) behind the voices promoting it. Those are harder to find and less exciting to read than a roadmap, which is exactly why they are where the edge is. A token can pass every structural check and still fall; a token can fail every check and still moon for a while. The checklist does not predict the price. It tells you the odds you are being offered and who set them — and that is the only honest thing anyone can hand a retail buyer.

The deepest takeaway is the calmest one: in crypto, being early is not a matter of clicking fast on launch day. The people who were genuinely early were in the private rooms years ago, at prices you will never see, and the public price is where their gains get realized. Understanding that doesn't make you cynical — it makes you a reader of the machine instead of a passenger in it. Every other post in this series is a closer look at one of these layers; start with [the crypto VC and market-maker overview](/blog/trading/crypto/crypto-vc-and-market-makers), and read the [Three Arrows Capital contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) for what happens when the leverage above your head finally breaks.

## Sources & further reading

Primary sources behind the headline figures in this post:

- a16z crypto, "Announcing our fourth crypto fund" — the \$4.5 billion Fund IV and \$7.6 billion cumulative figure ([a16zcrypto.com, May 2022](https://a16zcrypto.com/posts/article/crypto-fund-four/)).
- U.S. Securities and Exchange Commission, "Tai Mo Shan to Pay \$123 Million…" — the Jump Crypto subsidiary's TerraUSD settlement ([SEC press release, Dec 2024](https://www.sec.gov/newsroom/press-releases/2024-212)).
- U.S. Department of Justice, "Samuel Bankman-Fried Sentenced to 25 Years…" — the ~\$8 billion customer-fund figure, 25-year sentence, and \$11 billion forfeiture ([justice.gov, Mar 2024](https://www.justice.gov/archives/opa/pr/samuel-bankman-fried-sentenced-25-years-his-orchestration-multiple-fraudulent-schemes)).
- TechCrunch, "Crypto market maker Wintermute loses \$160 million in DeFi hack" ([techcrunch.com, Sep 2022](https://techcrunch.com/2022/09/20/crypto-market-maker-wintermute-loses-160-million-in-defi-hack/)).
- Binance Research, "Low Float & High FDV: How Did We Get Here?" — the ~12% median float and \$155 billion unlock estimates ([binance.com, May 2024](https://www.binance.com/research/analysis/low-float-and-high-fdv-how-did-we-get-here)).
- TokenInsight 2024 exchange data on Binance's spot-volume share ([via ChainCatcher](https://www.chaincatcher.com/en/article/2162313)).

Related posts on this blog:

- [Crypto VC and Market Makers: The Real Power Structure Behind the Tokens](/blog/trading/crypto/crypto-vc-and-market-makers) — the overview this series expands.
- [The FTX Collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — the structure with every wall removed.
- [Centralized Crypto Exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — the exchange-as-player.
- [Three Arrows Capital and Crypto-Lender Contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) — what happens when the leverage above you breaks.
- [The Terra-Luna 2022 Collapse](/blog/trading/crypto/terra-luna-2022-collapse) — the peg that a market maker was paid to defend.

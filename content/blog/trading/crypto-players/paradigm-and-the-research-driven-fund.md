---
title: "Paradigm: The Research-Driven Fund That Shapes the Tech It Invests In"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A from-scratch profile of Paradigm, the crypto fund that publishes research and gives away developer tooling — and how that technical credibility, not just its checkbook, decides which designs and tokens win."
tags: ["crypto", "venture-capital", "paradigm", "crypto-players", "uniswap", "mev", "open-source", "defi", "tokenomics", "soft-power"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 32
---

> [!important]
> **TL;DR** — Paradigm is a crypto venture fund whose real weapon is not its money but its *technical credibility*: it publishes influential research and gives away the software most of the ecosystem builds on, and that reputation lets it win the best deals on the best terms.
>
> - Paradigm was founded in **June 2018** by **Fred Ehrsam** (a co-founder of Coinbase) and **Matt Huang** (a former Sequoia Capital partner). It reportedly managed about **\$12.7 billion** as of April 2025.
> - Its funds are large and concentrated: a roughly **\$750M** first fund (2018), a **\$2.5B** vehicle in 2021 that was the largest crypto fund ever at the time, and an **\$850M** early-stage fund in June 2024.
> - The fund builds and maintains **free, open-source tooling** — most importantly **Foundry** (how a huge share of Ethereum developers now build and test contracts) and **Reth** (a from-scratch Ethereum node) — and publishes technical research read across the whole industry.
> - The mechanism to understand: giving away tools and research buys **credibility**, credibility buys **deal flow** and **cheaper, larger allocations**, and setting a standard the market adopts means Paradigm sees and shapes the ecosystem *before it writes a check*.
> - It shows up in price as a **signal** — "Paradigm led the round" pulls in co-investors, listings, and users — and the same soft power can leave a supply **overhang** for whoever buys last.
> - This is educational, not financial advice. The point is to see the machinery, not to chase or fear a logo on a cap table.

Here is a question that should bother you more than it does. A venture fund's entire job is to make money for its investors. So why would one of the most successful crypto funds on earth pay a team of world-class engineers, for years, to build software — and then give that software away, for free, to everyone including its competitors and the founders it hasn't invested in yet?

That is not a hypothetical. Paradigm, the fund we are going to take apart in this post, employs an engineering team that builds and maintains some of the most widely used free tools in the entire Ethereum ecosystem. It publishes research that its rivals read the morning it comes out. On paper this looks like charity, or vanity. It is neither. It is one of the most effective business strategies in the industry, and once you see how it works you will never read a crypto funding announcement the same way again.

The short version is that Paradigm figured out something most funds never do: in a market where money is abundant and undifferentiated, the scarce thing is *credibility*, and the way you manufacture credibility is by doing visible, verifiable, useful technical work. The research and the tooling are not the product. They are the marketing, the sales team, the early-warning radar, and the moat, all at once. The diagram below is the mental model for the whole post; everything after it is a tour of one arrow at a time.

![The flywheel: free research and open-source tooling buy technical credibility, which converts into inbound deal flow, standard-setting, cheaper and larger allocations, and outsized returns that fund the next round of research.](/imgs/blogs/paradigm-and-the-research-driven-fund-1.webp)

Read the picture left to right. Free research and open-source tooling earn **technical credibility**. Credibility does two things at once: it makes founders *come to Paradigm* (inbound deal flow), and it lets Paradigm *set the standards* the rest of the ecosystem builds on. Deal flow converts into a larger stake at a lower price; standard-setting converts into real usage and safer protocols. Both feed **outsized portfolio returns**, which pay for more research and more engineers — and the wheel turns again. This post builds that picture from absolute zero. We will define every term — venture fund, limited partner, carry, open source, technical standard, MEV, automated market maker, soft power — before we lean on it, and ground each idea in a worked example with round numbers you can check in your head.

## Foundations: the building blocks

If you have never looked closely at how a fund works, this section is the whole game. Read it slowly. If you already know what a limited partner and a carry are, skim to the part about "research-driven."

### What a venture fund actually is

A **venture fund** is a pot of money that professional investors raise from other people and use to buy stakes in young, risky companies (and, in crypto, in tokens and protocols). Three roles matter:

- **Limited partners (LPs)** — the people whose money it actually is. For a big fund these are institutions: university endowments, pension funds, foundations, wealthy families. Wikipedia's account of Paradigm's first raise names **Harvard, Yale, and Stanford** endowments among the backers. LPs are "limited" because their liability and their day-to-day involvement are both limited: they hand over money and wait.
- **General partners (GPs)** — the fund's managers, the people who actually pick the investments and sit across from founders. At Paradigm the co-founders, Fred Ehrsam and Matt Huang, are the marquee GPs.
- **The fund** — a legal vehicle with a fixed size (say, \$850 million) and, usually, a fixed life. The GPs *deploy* the fund's capital into a portfolio of bets over a few years, then spend several more years trying to turn those bets into cash.

The GPs get paid in two ways, and you must understand both to understand every incentive in this post:

- A **management fee** — typically around 2% of the fund per year — that pays salaries and keeps the lights on. On a \$850M fund, 2% is \$17M a year. That is real money, but it is not where the fortune is.
- **Carried interest**, or **carry** — usually 20% of the fund's *profits*. This is the jackpot. If a fund turns \$850M into \$3.4B, that is \$2.55B of profit, and 20% of it — \$510M — goes to the GPs. Carry is why a fund's whole existence bends toward producing a small number of enormous winners.

One more piece of vocabulary you will see constantly: **MOIC**, or *multiple on invested capital*. It is just "how many times your money you got back." Put in \$1M, get back \$32M, and your MOIC is 32x. A related word, **exit**, means the moment an investment turns back into cash or liquid tokens you can sell — a company gets acquired or goes public, or a token lists on exchanges and unlocks.

#### Worked example: the simplest possible fund

Suppose you raise a \$100M fund. You charge LPs 2% a year (\$2M) to run it, and you keep 20% of profits. You make ten bets of \$10M each. Nine go to zero. One returns 25x — your \$10M becomes \$250M. The fund returns \$250M on \$100M, a 2.5x MOIC. Profit is \$150M; your carry is 20% of that, or \$30M. Notice what just happened: **nine out of ten bets failed and you still made a fortune**, because the single winner paid for everything. Hold that thought — it is the entire economic logic of the model, and we will make it concrete with Paradigm's real fund sizes in a moment.

### What "research-driven" means

Most funds compete on money and network: they write a check, take a board seat, make introductions, and push marketing. A **research-driven fund** adds a different kind of value on top of the check. Three things make Paradigm research-driven:

- **Published research.** Paradigm's team writes and openly publishes technical essays and papers — on market design, on the economics of blockchains, on security — at a level that shapes how the whole field thinks. When you produce the analysis everyone else cites, you are no longer just another bidder; you are an authority.
- **Open-source tooling.** *Open source* means software whose code is published for anyone to read, use, and modify for free, usually under a permissive license (Paradigm's carry the **MIT** and **Apache** licenses, which basically say "do what you like, no strings"). Paradigm funds a full engineering team that ships free tools the ecosystem depends on.
- **Technical standards.** A **standard** is a shared way of doing something that everyone agrees to follow, so that independently built pieces fit together. On Ethereum, many standards are proposed as **EIPs** — *Ethereum Improvement Proposals*, the public documents through which changes to the network and conventions for tokens are debated and adopted. A fund whose people help write the standards, or whose tools *become* the de facto standard, gets to influence the shape of the thing before anyone else.

The contrast is the reason the fund is interesting at all. Here is the difference in one picture.

![A check-writing fund brings capital, a board seat, intros, and marketing, and competes mainly on price; a research-driven fund brings all of that plus in-house engineering, open-source standards, and published research that reads as credibility.](/imgs/blogs/paradigm-and-the-research-driven-fund-2.webp)

A generic fund's offerings (left) are things any fund can provide, so founders choose between funds mostly on price and reputation. A research-driven fund (right) brings the same capital *plus* things that are genuinely hard to copy: engineers who can help you ship, the free tools your team already uses, and a public research record that makes its backing a credential. That extra column is the whole thesis of this post.

### Three crypto terms you need, briefly

Because Paradigm's fame is tied to specific pieces of crypto, three quick definitions so the case studies land:

- **Automated market maker (AMM) / Uniswap.** Normally, to trade you need someone on the other side — a buyer for every seller. An **AMM** replaces the human counterparty with a pool of two tokens and a formula: you trade against the pool, and the formula moves the price as the pool's balance shifts. **Uniswap** is the most important AMM on Ethereum — a *decentralized exchange* (a place to swap tokens with no company in the middle). Paradigm was an early backer, and we will look at that bet closely. For the wider DeFi picture, see [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).
- **MEV.** Short for *maximal (originally "miner") extractable value*. On a public blockchain, transactions sit briefly in a waiting area (the "mempool") before they are finalized, and whoever gets to *order* the transactions in a block can profit — by inserting their own trade ahead of yours, for instance. MEV is the money that can be squeezed out of controlling that ordering. Paradigm's researchers helped the whole industry understand it; we will see how.
- **Soft power.** Borrowed from geopolitics: **hard power** is force or money — you make someone do something by paying or compelling them. **Soft power** is influence through attraction and credibility — you get your way because people *want* to follow you, trust your judgment, and adopt your ideas. Paradigm's edge is overwhelmingly soft power. Almost everything below is a mechanism for turning technical work into it.

With the vocabulary in place, we can look at the machine itself — starting with the unglamorous part underneath the research: the fund is, first, a business that lives or dies on a handful of outlier bets.

## 1. The business underneath: a concentrated fund living on outliers

Before we get to the clever soft-power stuff, anchor on the money. Paradigm is a large, concentrated fund, and the sizes are public.

According to Wikipedia's account and contemporary press, Paradigm's **first fund**, formed in October 2018, raised on the order of **\$750 million**, structured as an open-ended vehicle (no fixed deadline to return capital) with endowment LPs. In **November 2021**, at the top of the last bull market, it raised **\$2.5 billion** for a vehicle reported at the time as the **largest crypto venture fund in history** (Decrypt and Bloomberg both covered the raise). Then in **June 2024**, in a much colder market, it raised **\$850 million** for an early-stage fund (Bloomberg, June 13, 2024). Reported assets under management were around **\$12.7 billion as of April 2025**. Those are the load-bearing numbers; treat the AUM figure as a dated snapshot, not a constant.

Why does the size matter? Because it forces a style. A \$2.5B fund cannot make a hundred tiny \$1M bets — that would need a hundred partners and would still leave most of the money undeployed. Big funds write bigger checks into fewer companies, which makes them *concentrated*. And concentrated early-stage investing obeys a brutal statistical law.

![The power law of a concentrated fund: of 30 early-stage bets at \$20M each, twenty go to zero, eight return a solid 3x, and two 30x home runs return \$1.2B — twice the \$600M deployed — on their own.](/imgs/blogs/paradigm-and-the-research-driven-fund-3.webp)

#### Worked example: why two bets can carry the whole fund

Suppose a fund deploys **\$600M** across **30 early-stage bets of \$20M each**. Early-stage crypto is savage, so model a realistic spread of outcomes:

- **20 bets go to zero.** That is \$400M of invested capital returning **\$0**. Most young protocols and tokens simply die.
- **8 bets return 3x.** \$160M invested comes back as **\$480M**. Respectable, unremarkable.
- **2 bets return 30x.** \$40M invested comes back as **\$1,200M** — \$1.2 billion.

Add it up. Total returned = \$0 + \$480M + \$1,200M = **\$1,680M** on \$600M deployed, a **2.8x** gross MOIC. Now look at where it came from. The two home runs, which absorbed just **\$40M** — one-fifteenth of the capital — produced **\$1.2B**, which is *twice the entire fund*. The twenty write-offs, the ones that will feel like disasters at the time, barely move the final number.

This is the **power law**: in a portfolio like this, returns are not spread evenly; they are dominated by a tiny number of extreme winners. The one-sentence intuition: **a concentrated fund is not really trying to be right often — it is trying to be enormously right occasionally, and to make sure it is *in* the deals that become the outliers.**

That last clause is the hinge of the whole post. If your fortune depends on being in the two deals out of thirty that go 30x, then the single most valuable capability you can have is not picking well after the fact — everyone claims that — but *getting into the best deals in the first place, and getting a big piece of them cheaply*. Money alone cannot buy that, because every good fund has money. Credibility can. And that is what the research and the tooling are for. (For the general shape of this business, see the sibling profile [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model); for the map of where funds sit in the whole ecosystem, [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto).)

## 2. The Uniswap bet: what one early check can do

The cleanest illustration of the power law in Paradigm's own history is Uniswap. Paradigm was an early backer of the decentralized exchange; contemporary reporting (Fortune, relayed via Yahoo Finance) describes Paradigm as a lead investor in Uniswap's seed round with a check reported in the low seven figures, and later reporting has put the **return at roughly 32x** the firm's money. Both the check size and the multiple are *reported* figures, not disclosed audited numbers, so treat them as attributed estimates — but the order of magnitude is the point.

![The Uniswap position, using reported figures: a small seed check goes out in 2018-19, Uniswap V2 launches in May 2020, the UNI token airdrops and lists in September 2020, volume surges through 2021, and the stake is reported to have returned roughly 32x.](/imgs/blogs/paradigm-and-the-research-driven-fund-4.webp)

#### Worked example: what "32x" actually means

MOIC is deliberately simple, so let's make the 32x concrete. Imagine — using round, illustrative numbers — that the seed check was **\$5M**. A 32x MOIC means the stake was later worth about **\$5M × 32 = \$160M**. On the \$750M first fund, a single \$160M outcome is more than a **fifth of the entire fund returned by one deal** — and that is *before* the biggest winners in the same fund. If instead the check had been \$10M, 32x would be \$320M, nearly *half* the fund from one position.

Now walk the second-order point, because it is where the story stops being about luck. Uniswap was not a lottery ticket Paradigm happened to hold. Paradigm's researchers went on to publish detailed technical analysis of automated market makers and of Uniswap's design; the fund's engineers built tooling that Uniswap-style teams use. In other words, Paradigm did not just *own* a piece of the most important AMM — it became a recognized authority *on* AMMs. That authority is what gets you invited into the *next* Uniswap before it is obvious, which is the only way to keep hitting the power-law jackpot on purpose. The one-sentence intuition: **one early check into winning infrastructure can pay for a whole fund — and the way you find the next one is by being the person the next founder most wants on their cap table.**

A caution worth stating plainly, since this is a named-firm profile: not every Paradigm bet is a Uniswap. The same firm invested a reported **\$278 million** in the exchange FTX and wrote it down to **zero** when FTX collapsed into bankruptcy in November 2022 (per Wikipedia's account and widely reported at the time). The power law cuts both ways: the model *expects* total losses, even large and embarrassing ones. We will come back to FTX in the real-markets section, because it is also a case study in the limits of soft power.

## 3. Research as a moat: publishing the map of the territory

Now the distinctive part. Why does a fund publish research at all? A cynic's first guess is "content marketing," and that is not wrong — but it undersells how much leverage good research creates in this specific market.

The founding example is an essay from **August 2020** titled **"Ethereum is a Dark Forest,"** by Paradigm's **Dan Robinson** (a General Partner and the fund's Head of Research) and **Georgios Konstantopoulos** (the fund's CTO). The piece told a true story: the authors tried to rescue funds stranded in a vulnerable smart contract, and discovered that the public mempool — the waiting area where pending transactions are visible — was patrolled by predatory bots that detect any profitable transaction and copy or front-run it within milliseconds. They borrowed the "dark forest" metaphor from science fiction: the open blockchain is a place where revealing your intentions gets you hunted.

That essay, and the work around it, did something valuable: it gave the whole industry a shared, vivid vocabulary for **MEV** — the value extractable by whoever orders transactions. Naming a problem clearly is a form of power. Once "the dark forest" and "MEV" were common language, the people who had explained it were the natural authorities on it. Several members of the Paradigm orbit — including Konstantopoulos and the pseudonymous security researcher **samczsun** — were among the crowd that helped spin up **Flashbots**, the research-and-development organization built to study and tame MEV, which Paradigm also helped fund. (For the mechanics of MEV itself, see the sibling series entry on [MEV as the invisible tax on every trade](/blog/trading/crypto/crypto-mining-staking-and-mev).)

Sit with the compounding here. Paradigm identified a structural problem, published the clearest explanation of it, helped stand up the organization that addresses it, and holds stakes across the businesses that grow up around the solution. None of those steps required out-bidding anyone. They required being *early and correct in public*.

> Research is how a fund converts being right into being trusted — and in a market where everyone has money, trust is the only thing that is actually scarce.

There is a subtler benefit, too. Publishing high-quality analysis is a filter that runs in both directions. It attracts the technical founders who care about that kind of rigor (they read the essays and want that fund on their cap table), and it signals to LPs and co-investors that this fund understands the territory at a level a spreadsheet cannot fake. The research does not have to *directly* generate a deal to be worth its cost; it raises the fund's standing on every deal at once.

## 4. Tooling as territory: Foundry and Reth

Research earns respect. Tooling does something even more concrete: it puts the fund *underneath* the thing everyone builds on.

Paradigm funds an engineering team whose output is a stack of free, open-source software for building on Ethereum. Two pieces matter most.

- **Foundry**, quietly released around December 2021, is a toolkit — written in the Rust programming language — for building and testing smart contracts written in Solidity (Ethereum's main contract language). Its pitch is speed and a "test in the same language you build in" workflow, and it spread fast: teams behind major protocols moved their development and testing onto it, and Paradigm shipped a **1.0** release in February 2025. For a large share of serious Ethereum developers, Foundry is simply *how you work now*.
- **Reth** (short for "Rust Ethereum"), announced in December 2022, is a from-scratch implementation of an Ethereum node — the software that actually runs the network — built for modularity and performance. It reached a production **1.0** in June 2024, driven by a small Paradigm-funded core team plus dozens of outside contributors, and a **2.0** followed in April 2026. Alongside them sit lower-level libraries like **revm** and **Alloy**.

The strategic picture is a stack, and Paradigm quietly owns the bottom of it.

![Owning the base layer of the developer stack: apps and protocols like Uniswap, Optimism, and MakerDAO build on Foundry; Foundry, Reth, revm, and Alloy are all Paradigm-funded open source under free MIT / Apache licenses.](/imgs/blogs/paradigm-and-the-research-driven-fund-6.webp)

Applications sit on top. Beneath them is the developer framework (Foundry). Beneath that, the node software (Reth) and the low-level building blocks (revm, Alloy). And beneath all of it, quietly, is *Paradigm* — the fund that pays for the free tools the whole tower rests on. Now ask the cynic's question again, seriously: why give this away?

#### Worked example: the economics of a free standard

Let's put rough numbers on it. Suppose the open-source engineering effort costs Paradigm on the order of **8 to 30 engineers**. Fully loaded (salary, benefits, overhead), a top blockchain engineer might cost around **\$400,000 a year**, so the tooling org plausibly costs somewhere in the range of **\$3M to \$12M a year**. That is a real expense — and against a fund earning 2% on billions, it is also a rounding error. Call it **\$8M a year** for the sake of arithmetic.

Now the benefit. Because most serious new Ethereum teams build and test on your free tools, you get four things money cannot easily buy:

1. **An early-warning radar.** You see which teams are shipping fast, which designs are gaining traction, and which developers are exceptional — often *before* they raise a round.
2. **Distribution for your standards.** If your tool is how contracts get written and tested, your conventions become the ecosystem's conventions.
3. **A credential.** "The fund that builds Foundry" is a sentence that opens every door.
4. **Portfolio support that costs nothing extra.** Your investments ship faster and safer on tools your own team maintains.

Now weigh it. Recall from Section 1 that being in *one* extra outlier is worth hundreds of millions of dollars, and from Section 2 that a single Uniswap-scale hit can return a fifth of a fund. If the tooling helps Paradigm win or find **even one additional outlier per fund cycle**, the \$8M-a-year radar has paid for itself *hundreds of times over*. The one-sentence intuition: **a free standard is not charity — it is the cheapest customer-acquisition and market-intelligence system in the industry, disguised as a public good.**

There is a genuine public benefit here, and it should be acknowledged rather than sneered at: the tools really are free, really are good, and really do make Ethereum development better for everyone, including people who will never take Paradigm's money. The strategy works *because* the gift is real. But "the gift is real" and "the gift is strategic" are both true at once, and a clear-eyed reader holds both.

## 5. How it converts: credibility into allocation

We have the two engines — research and tooling — and we know they produce credibility. Now trace exactly how credibility turns into money, because this is the step people wave their hands at.

![How research output becomes allocation: publishing research and shipping Foundry builds founder trust, which turns into being invited to lead the round, which turns into a bigger stake at a lower entry price, which turns into a higher return per dollar.](/imgs/blogs/paradigm-and-the-research-driven-fund-5.webp)

Read it as a chain. **Publish research and ship free tooling** → **founders adopt the tools and trust the fund** → Paradigm is **invited to lead the round** rather than merely allowed to bid → because the founder *wants* them, Paradigm negotiates a **bigger stake at a lower entry price** → which mechanically produces a **higher return per dollar** at exit. The soft power is not vague; it lands on specific line items of a term sheet.

To see how much this is worth, compare two funds bidding for the same hot round. Everything about the company is identical; the only difference is which fund the founder prefers.

![The credibility premium on a single deal: a generic fund wins a \$5M stake at a \$50M entry valuation with no engineering support, worth \$50M at a 10x round; Paradigm wins a \$10M stake at a \$40M entry with in-house engineering, worth \$125M.](/imgs/blogs/paradigm-and-the-research-driven-fund-7.webp)

#### Worked example: the credibility premium

Two funds want into the same round. The company will, let's say, grow 10x from here in value.

- **The generic fund** is one bidder among many, so it takes what it can get: a **\$5M** stake at a **\$50M** entry valuation. It owns 10% of the round's value. If the company's value rises 10x — the round's valuation goes from \$50M to \$500M — the fund's \$5M position is worth **\$50M** at that mark. A clean 10x. Fine.
- **Paradigm** is the fund the founder actively wants — because the team already uses Foundry, respects the research, and values the engineering help. That preference is leverage. Paradigm negotiates a **\$10M** stake at a **\$40M** entry (it gets in *bigger* and *cheaper*). When the company's value reaches that same \$500M mark, Paradigm entered at \$40M, so it rode a **12.5x** on a **\$10M** position — worth about **\$125M**.

Same company, same market, same exit. The generic fund makes **\$50M**; Paradigm makes **\$125M** — two and a half times as much money — purely because credibility bought it a bigger, cheaper allocation. Stack that edge across dozens of deals and compound it over years, and you have explained most of the gap between a good fund and a great one. The one-sentence intuition: **the fund founders most want to work with does not just win more deals — it wins them on terms that multiply its returns, and technical credibility is what earns that preference.**

This is also the honest answer to "isn't this just being famous?" No. Being famous gets you into the room. Being *technically credible to the specific person deciding* — because they use your tools and read your research every week — is what changes the numbers on the page once you are in it. (For how the seat on the cap table then behaves over a token's life, see [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) and [reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).)

## 6. How it shows up in price

Everything so far has been about how Paradigm makes money. This section is about how *you* feel it — how a research-driven fund's soft power shows up in the price of a token or the terms of a round you might touch.

![How the influence shows up in price: Paradigm leading a round is itself a signal that pulls in co-investors, easier exchange listings, and real users through tooling adoption; demand rises, valuation rises, and the same backing can leave an insider overhang and narrative risk later.](/imgs/blogs/paradigm-and-the-research-driven-fund-8.webp)

Trace the arrows. **Paradigm leads a round.** That fact alone does work in three directions:

- **Co-investors follow the signal.** Other funds treat "Paradigm led" as due diligence they don't have to repeat. This is rational herding — and it means the round fills faster and at a higher valuation than it otherwise would. (On why crowds move together in crypto, the game-theory lens is useful; the fund's imprimatur is a coordination device.)
- **Listings get easier.** Exchanges want tokens that will trade and that carry a credible story. A top-tier backer is part of that story, so a Paradigm-backed token often has a smoother path onto major venues — and a listing is itself a price event.
- **Real users arrive via the tooling and research halo.** A protocol built by a team Paradigm backs, on tools Paradigm maintains, with a design Paradigm's research validates, has an easier time attracting the developers and users whose activity is the fundamental value underneath the price.

All three raise **demand** — for the round, and later for the token — which pushes **valuation and price up**. So far this reads as good news. But the last arrow is the one retail readers must internalize: **the same soft power becomes a risk later.** A fund that got in early and cheap holds a large, low-cost position. When its tokens unlock on a schedule (see [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) for why crypto insiders can exit years before equity investors could), that position is a supply **overhang** — coins that can be sold into the market the fund's own credibility helped inflate. And a narrative that a credible fund *created* can reverse when that fund's attention moves on. The influence that lifted the price is the same influence that can weigh on it. Being on the other side of a Paradigm entry, years later and at a much higher price, is a structurally different trade than the one Paradigm made. For the general mechanics of how these forces reach the order book, see [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) and the overview hub, [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers).

## Common misconceptions

**"Paradigm gives away tools because it's altruistic / for the community."** The tools are genuinely useful and genuinely free, and that matters. But the strategy is not charity; it is the cheapest radar, distribution channel, and credential in the industry (Section 4). Both things are true at once. Treating the gift as *only* altruism, or *only* cynical, misreads it.

**"If a top fund backed it, it must be safe."** No. The power-law model *expects* most bets to fail, including big, public ones. Paradigm invested a reported \$278M in FTX and lost all of it. A prestigious logo on the cap table tells you a smart investor liked the risk/reward *for themselves*, entering early and cheap — it does not tell you the token is a good buy for you, entering late and expensive.

**"Research funds don't care about returns — they're academic."** The research exists *because* of returns, not instead of them. Every essay and every tool is, ultimately, in service of getting into the two-out-of-thirty deals that carry the fund. Mistaking the marketing for the mission gets the causality backwards.

**"A bigger fund is a better fund."** Size cuts both ways. A \$2.5B fund can dominate rounds and fund a huge engineering team — but it also *must* deploy billions, which forces bigger checks, later stages, and higher entry prices, and it makes the power-law math harder (you need bigger absolute winners to move a bigger fund). Paradigm's own drop from a \$2.5B fund in 2021 to an \$850M early-stage fund in 2024 reflects that discipline, not just a colder market.

**"Setting a standard means controlling the network."** Owning the dominant developer tool is influence, not control. Ethereum's rules are set by a messy, public, multi-party process (the EIP process and client diversity are designed precisely so no single party dominates). What tooling buys is *soft* power — being early, being trusted, being underneath — not a veto. Overstating it tips into conspiracy; understating it misses the real edge.

**"The fund's returns come from being a better stock-picker."** Partly, but the durable edge is *access and terms*, not just selection (Section 5). Two funds can pick the same winner and make very different amounts of money depending on the size and price of the stake they were able to negotiate — and credibility is what moves those terms.

## How it shows up in real markets

Named episodes, with dated, sourced figures where they exist. Contested or reported items are flagged as such.

### 1. The Uniswap seed — the power law, realized

Paradigm was an early, lead-level backer of Uniswap, the decentralized exchange, with a seed check reported in the low seven figures (Fortune, via Yahoo Finance). Later reporting put the return at **roughly 32x**. Uniswap V2 launched in May 2020; the UNI governance token airdropped and listed in September 2020; DeFi trading volume surged through 2021. Whatever the exact numbers, this is the textbook case of the model working: a small, early check into foundational infrastructure that returns a meaningful fraction of an entire fund — and cements the fund's authority on the very design (AMMs) it profited from. Treat the specific multiple as *reported*, not audited.

### 2. Foundry's takeover of Ethereum development

Foundry, released around December 2021 and shipped as 1.0 in February 2025, moved from novelty to default in a few years, with major protocol teams adopting it for building and testing. This is standard-setting in its purest form: no acquisition, no exclusivity, just a free tool so good that the ecosystem reorganized around it. The payoff is not a line in a return report — it is the radar, distribution, and credibility that make *every* deal a little easier for the fund that pays for it (Section 4).

### 3. "Ethereum is a Dark Forest" and the MEV vocabulary

The August 2020 essay by Dan Robinson and Georgios Konstantopoulos gave the industry its shared language for MEV, and Paradigm's orbit helped seed Flashbots, the R&D group formed to study and mitigate it. This is research-as-moat: identify a structural problem in public, explain it best, help build the response, and hold positions across the businesses that grow around the solution — none of which required outbidding a competitor. It is influence bought with insight.

### 4. The FTX write-off — the limit of soft power

Paradigm invested a reported **\$278 million** in the exchange FTX and marked it to **zero** when FTX collapsed in November 2022 (widely reported; per Wikipedia's account, Paradigm later said it had misgivings and felt misled). A February 2023 class action named Paradigm among investors accused of lending credibility to FTX. The lesson is double: first, the power law means even elite funds take total losses, so a backer's presence is never a safety guarantee; second, the same credibility that is an asset when a bet works becomes a *liability* — legal and reputational — when a bet blows up, precisely because that credibility was part of what drew others in. For the full anatomy of that collapse, see [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).

### 5. The 2023 "research-driven technology" reframe

In May 2023, Paradigm reworked its public positioning, foregrounding "research-driven technology" (including artificial intelligence) rather than leading with "crypto/Web3," while stating it had not pivoted away from crypto (per Wikipedia's account, some LPs and founders were frustrated by the lack of a heads-up). Whatever the internal reasoning, it is a revealing data point: the fund's own chosen identity is *research and technology first*. The label the firm reaches for when it describes itself is not "we write big checks" — it is "we do the work." That is the brand this entire post is about.

### 6. The Citadel Securities co-investment — credibility travels

In January 2021, Paradigm participated (alongside Sequoia Capital) in a reported **\$1.15 billion** investment into Citadel Securities, a giant traditional-finance market maker (per Wikipedia's account). It is a useful reminder that a research-driven crypto fund's credibility is portable: the same reputation that wins crypto deals lets it sit at the table for large, mainstream financial-infrastructure bets. Soft power built in one arena spends in another.

### 7. The Uniswap lawsuit — the flip side of being named

When Uniswap was sued in April 2022 over allegations tied to tokens traded on the protocol, Paradigm was named among the defendants (with a16z and Union Square Ventures), on the theory that prominent backers bore responsibility. A federal judge dismissed the suit in Uniswap's favor in August 2024. For our purposes the outcome matters less than the pattern: being a *known*, *credible* backer means you are also a known target. The visibility that is the whole strategy has a legal edge to it.

## When this matters to you

You are not going to co-invest alongside Paradigm. So why does any of this touch your money? Because the *output* of this machine — a token, a round's narrative, a "backed by Paradigm" headline — will eventually reach the public market where you can buy it, and by then you are on the far side of every advantage described above.

A few honest, practical takeaways — educational, not advice:

- **A famous backer is information about the *entry*, not about your *entry*.** Paradigm's edge is being early, cheap, and credible. When you read "backed by Paradigm," what you have learned is that a sophisticated investor liked the risk at a valuation you will almost never see. That is a reason to study the deal harder, not a reason to skip the study.
- **Find the entry price and the unlock schedule before the logo impresses you.** The gap between an insider's cost basis and today's price *is* the overhang. The tools to check it are public; the sibling posts on [reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) and [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) walk through exactly how.
- **Separate the real from the promotional.** Foundry being excellent is real. A token's price narrative being "endorsed by the fund that builds Foundry" is promotional. Credibility earned by building tools is legitimately informative about a *team's technical quality*; it is not informative about whether a token is cheap.
- **Respect the asymmetry.** Paradigm can be wrong twenty-eight times out of thirty and still win. You cannot. Your loss function is not a fund's loss function, and strategies that make sense for a diversified power-law portfolio can be ruinous for a concentrated personal one.

The deepest point is the one the whole post has been circling: in crypto, the scarce resource is not capital — it is *credibility*, and credibility can be manufactured by doing visible, verifiable, useful technical work. Paradigm's genius was to see that early and build the machine for it. Understanding the machine does not tell you what to buy. It tells you who is likely on the other side of the trade, how they got there, and why the ground under a shiny new token may not be as solid as the logo suggests. For the wider map of who moves crypto prices, start at the series hub, [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto), and the neighboring giant profile, [a16z crypto, the institutional giant](/blog/trading/crypto-players/a16z-crypto-the-institutional-giant).

## Sources & further reading

Primary and reputable sources behind the headline figures (with as-of dates where the number moves):

- **Paradigm — company overview, funds, AUM, portfolio, events.** *Paradigm (venture capital firm)*, Wikipedia — founding (June 2018), founders, ~\$750M first fund (Oct 2018), \$2.5B fund (Nov 2021), ~\$12.7B AUM (as of April 2025), ~59 employees (2024), the \$278M FTX write-off (Nov 2022), the \$1.15B Citadel Securities co-investment (Jan 2021), the Uniswap lawsuit (filed Apr 2022, dismissed Aug 2024), and the May 2023 positioning reframe.
- **The \$850M 2024 fund.** *Paradigm Raises \$850 Million for Early-Stage Crypto Venture Fund*, Bloomberg, June 13, 2024; also covered by Decrypt.
- **The \$2.5B 2021 fund.** *Paradigm Co-Founder on Record \$2.5B Fund*, Decrypt, 2021.
- **Uniswap seed and reported ~32x return.** Fortune reporting relayed via Yahoo Finance, *Paradigm backs decentralized exchange protocol Uniswap*; return multiple reported in later coverage. Treat the specific multiple and check size as reported estimates.
- **Foundry, Reth, and open source.** Paradigm's own writing and code — *Introducing Reth* (Dec 2022), *Releasing Reth 1.0* (June 2024), *Announcing Foundry v1.0* (Feb 2025), and the Paradigm open-source page (getfoundry.sh; github.com/paradigmxyz).
- **MEV research and Flashbots.** *Ethereum is a Dark Forest*, Dan Robinson and Georgios Konstantopoulos, Paradigm, August 2020; *MEV and me*, Paradigm, February 2021; Flashbots background.

On this blog, to go deeper:

- [Crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the overview hub for the players who move crypto.
- [The crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) — how a crypto fund makes money, in detail.
- [a16z crypto: the institutional giant](/blog/trading/crypto-players/a16z-crypto-the-institutional-giant) — the neighboring profile, and a contrasting playbook.
- [Why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) and [reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) — the structural reasons insiders exit on terms retail never sees.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) and [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — the specific protocols and the specific failure discussed above.

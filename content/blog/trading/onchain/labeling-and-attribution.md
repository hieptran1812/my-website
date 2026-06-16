---
title: "Labeling and Attribution: How an Address Gets a Name"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How a meaningless hex address becomes 'Binance 14' or 'Wintermute' — the attribution sources behind Arkham and Nansen labels, why those labels are probabilistic, and how to sanity-check one before you trade or accuse."
tags: ["onchain", "crypto", "attribution", "labeling", "clustering", "arkham", "nansen", "smart-money", "address-poisoning", "ethereum"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The raw chain is a sea of meaningless hex; the entire value of a platform like Arkham or Nansen is the **label** it puts on top — "Binance 14", "Jump Trading", "Wintermute", "Tornado Cash Router". This post is about where those names actually come from, and why you must never trust one blindly.
>
> - A **cluster** is a set of addresses that heuristics say share an owner; a **label** is a human-readable identity attached to that cluster. Clustering links addresses; labeling names the link.
> - Labels are built from four sources: **ground-truth seeds** (deposit to a known exchange address, then watch the cluster), **self-disclosure** (ENS, project docs, verified contracts), **clustering propagation**, and **off-chain intel** (KYC leaks, court filings, OSINT). The first two are verifiable; the last two are inference.
> - **What you do with it**: read the *basis and confidence* of a label, not just the name. Act on **verified**, size down and double-check **inferred**, and never trade or accuse on a **guessed** label alone.
> - The one rule to remember: **a label is a probability, not a fact.** Acting on a bad label — copying a "whale" that is really an exchange omnibus, or paying a poisoned look-alike — is a real, recurring way people lose money.

On 21 February 2025, roughly **\$1.46 billion** of Ether walked out of Bybit's cold storage and into a web of fresh addresses controlled by the Lazarus Group. Within an hour, blockchain investigators were publicly naming wallets: *this* cluster is the attacker, *that* one is a victim exchange, *this* contract is a swap router the thieves used to convert the loot. None of those addresses came with a name printed on the chain. The chain only ever shows hexadecimal strings like `0x47666...` — public, permanent, and utterly anonymous-looking. Every name in that investigation was a **label**: a human-readable identity that an analyst, or a platform's labeling pipeline, *attributed* to a string of hex.

That gap — between a raw address and a name you can act on — is where almost all the commercial value of on-chain analysis lives. Anyone can run a node and see the bytes. What you pay Arkham, Nansen, Chainalysis, or Elliptic for is the **attribution**: the claim that `0x28C6...` is "Binance 14", that a Solana address is "a smart-money wallet up 40× this cycle", that a specific cluster is "the Lazarus Group". Get the label right and the chain reads like a confession. Get it wrong — and labels *are* wrong, more often than the slick green dashboards admit — and you copy the wrong wallet into a loss, or you accuse an innocent address of being a thief.

This post is the missing manual for that gap. We start from zero: what a cluster is versus what a label is, what a "seed" address means, how a single confirmed seed propagates a name across an entire cluster. Then we go deep on the four real attribution sources, on how Arkham and Nansen actually build their entity graphs and "smart money" tags, and on the false-label traps — mislabeled whales, address poisoning, recycled wallets — that turn a confident name into a costly mistake. Throughout, every claim is grounded in money, because a label only matters in the end if it changes what you do with your capital.

![Pipeline from a raw hex address through a cluster and a ground-truth seed to a named entity and an action](/imgs/blogs/labeling-and-attribution-1.png)

## Foundations: cluster, label, seed, and the two kinds of signal

Four ideas have to be rock-solid before any of the attribution machinery makes sense. None of them is exotic — they are the difference between *linking* addresses and *naming* the link, and the difference between evidence that lives on the chain and evidence that lives off it. We will define each from the ground up, with no assumed crypto background.

### An address is a pen name, not a person

Start with the thing being labeled. An **address** on a blockchain is a short public string — `0x` followed by 40 hex characters on Ethereum and other EVM chains, a `bc1…` or `1…` string on Bitcoin, a long Base58 string on Solana, a `T…` string on Tron. It is derived one-way from a secret key, and it carries **no name, no country, no email**. (If you want the full derivation, the sibling post [Addresses, Wallets, and Contracts](/blog/trading/onchain/addresses-wallets-and-contracts) builds it from the private key up.)

The single most important property for this entire post is that an address is **pseudonymous, not anonymous**. Anonymity would mean no one can *ever* connect the string to a real entity. Pseudonymity means the string is a *pen name* — and pen names get unmasked. Every transaction the address ever signs is recorded forever, in public, for everyone. The moment its behavior, funding source, timing, or counterparties point back to a known entity, the pen name starts to dissolve. Attribution is the craft of dissolving it.

So the chain gives you a permanent, public record attached to a name *you don't know yet*. Labeling is the act of writing the name in.

Why does the pen name decay at all? Because addresses don't exist in isolation — they *transact*, and every transaction leaks information. Four leaks do most of the work. **Funding source**: where did this address get its first coins? If they came from an exchange withdrawal, that exchange knows who you are, and the address now inherits a thread back to a real identity. **Counterparties**: who does it send to and receive from? An address that pays rent to a known landlord wallet, or repeatedly funds a labeled gambling contract, tells you something about its owner. **Timing**: an address that only ever transacts during Asian business hours, or that springs to life minutes after a US exchange opens, narrows the owner's timezone and habits. **Reuse**: the same address used across a forum tip jar, a DeFi position, and a withdrawal from a KYC'd exchange stitches three separate contexts into one identity. None of these is a name on its own; together, accumulated over months, they make pseudonymity a slowly-dissolving disguise rather than a permanent mask. Attribution is the engineering discipline of harvesting those leaks at scale.

This is also why the chain is asymmetric in a way that matters for traders and defenders alike: **information only accumulates.** An address can never *un-transact*. Every leak is permanent and retroactive — a deanonymizing link discovered in 2026 unmasks transactions from 2020 just as well. That is why investigators can trace a years-old hack and why a wallet you thought was private can be deanonymized long after the fact. For the labeler, time is always on the side of more attribution, never less.

### A cluster links addresses; a label names the link

Here is the distinction that trips up almost every beginner, so we draw a hard line around it.

A **cluster** is a set of addresses that one or more *heuristics* say are controlled by the same entity. A heuristic is a rule of thumb about behavior — for example, on Bitcoin the **common-input-ownership heuristic**: if a single transaction spends several inputs at once, whoever signed it almost certainly controlled all those inputs, so the input addresses probably share an owner. On account-chains like Ethereum, the analogous signals are things like one address repeatedly funding gas for a set of others, addresses that always co-spend within seconds, or many deposit addresses that all sweep into the same hot wallet. The output of clustering is purely structural: *these N addresses move together, so they probably belong to one wallet/entity.* It says nothing about **who** that entity is. The mechanics of these rules are the subject of the sibling post [Address Clustering and Heuristics](/blog/trading/onchain/address-clustering-and-heuristics); here we treat a cluster as a given input.

A **label** is a human-readable identity — "Binance 14", "Jump Trading", "Wintermute", "Tornado Cash Router" — attached to a cluster (or to a single address). The label is the *name*; the cluster is the *thing being named*. Two failure modes follow immediately from keeping these separate:

- **A correct cluster with the wrong label.** The heuristics correctly grouped 5,000 addresses that genuinely share an owner — but the name on top ("Wintermute") is wrong. The cluster is real; the identity is a guess.
- **A correct label on a too-large cluster.** The name "Binance" is correct for the seed address, but the clustering over-reached and swept in some addresses Binance doesn't actually control. Now you are attributing Binance's name to a stranger's wallet.

Both are real, both happen, and both cost money. The cover figure above shows the canonical pipeline — hex → cluster → seed → label → action — and the whole rest of this post is about each arrow being fallible.

### A seed is a ground-truth address you already know

A cluster on its own is anonymous. To turn it into a *named* entity you need at least one address in it whose identity you already know for certain. That known address is a **seed** (or "ground-truth" address).

Where do seeds come from? The cleanest is one you generate yourself: you sign up at an exchange, it gives you a **deposit address**, you send a tiny amount, and now you have a confirmed link — *this address belongs to that exchange, because the exchange told me to send there and credited my account.* Other seeds are publicly disclosed: a project announces its treasury multisig in its docs, a foundation publishes its donation address, a verified contract on Etherscan names the protocol it belongs to. A seed is the anchor that converts "these addresses move together" into "these addresses belong to **Binance**".

The figure below makes the canonical seeding move concrete: you deposit to a known exchange address, watch it sweep your funds into a consolidation wallet, and that sweep destination becomes a *confirmed* exchange hot wallet — a brand-new seed you discovered for free.

![Pipeline showing a deposit to a known exchange address sweeping into a confirmed hot wallet that seeds the cluster](/imgs/blogs/labeling-and-attribution-2.png)

#### Worked example: depositing \$500 to seed a \$2B exchange cluster

You want to confirm which on-chain wallet is "Exchange X — Hot Wallet 3". You open an account, and the exchange gives you a deposit address, call it `0xDep0…7a1`. You send it **\$500** of USDC. The deposit address is, by construction, ground truth: the exchange told you to use it, and minutes later your account shows **+\$500**.

Now you watch that deposit address on an explorer. Exchanges don't leave funds sitting in thousands of per-user deposit addresses; they **sweep** them periodically into a few consolidation wallets to manage gas and security. Within an hour, your \$500 (batched with hundreds of other users' deposits) moves in a single sweep transaction into `0xHot3…c92`. That destination just earned a label: it is a confirmed Exchange X hot wallet, seeded by *your* \$500.

From that one seed, clustering spreads outward. The hot wallet co-spends with other hot wallets; those connect to cold wallets holding the bulk of reserves. Chain those links and you can attribute the exchange's full on-chain footprint — say **\$2 billion** in visible reserves across the cluster — all anchored to a deposit that cost you half a thousand dollars and an hour of watching. *A single \$500 ground-truth deposit can anchor the identity of a multi-billion-dollar cluster — which is exactly how analysts map exchange reserves.*

### On-chain signals versus off-chain signals

The last foundational split is **where the evidence lives**.

- **On-chain signals** are everything visible in the public ledger: which address sent what to whom, when, in what amount, with what gas, calling which contract. Clustering heuristics run entirely on these. They are verifiable by anyone — you can re-derive them yourself — but they are *behavioral*: they tell you addresses move together, not who the human is.
- **Off-chain signals** are everything outside the ledger: a KYC database leak, a court filing that names a wallet, an exchange's published cold-wallet list, a founder's tweet, an ENS name, an OSINT match between an address posted on a forum and a real person. These can deliver the actual *human* identity that on-chain signals never can — but most of them are **unverifiable by you**, can be stale, and can be planted.

The most reliable attributions fuse the two: an off-chain seed (the exchange told you this address is theirs) anchors an on-chain cluster (these 5,000 addresses provably move with it). The least reliable lean entirely on one weak off-chain breadcrumb with no on-chain confirmation. Keep this axis in your head for the rest of the post — it is the difference between a label you can bet on and a label you can only treat as a lead.

## The four attribution sources

Every label on every platform traces back to one or more of exactly four sources. Knowing which source produced a given label is how you know how far to trust it. The matrix below lays them out by strength; we then take each in turn.

![Matrix of four attribution sources by what it is, proof type, confidence, and failure mode](/imgs/blogs/labeling-and-attribution-3.png)

### Source 1 — Ground-truth seeds (the gold standard)

A ground-truth seed is an address whose identity you established directly, not inferred. The strongest version is the **deposit-and-watch** move from the worked example above: you transact with the entity, so *you generated the link yourself*. Variants:

- **Deposit-and-withdraw.** Deposit to a known exchange, then withdraw, and you learn both a deposit address (yours, attributed to the exchange) and a hot wallet (the source of your withdrawal). Each is ground truth because you were the counterparty.
- **Published institutional disclosures.** When an exchange voluntarily publishes a "proof of reserves" cold-wallet list, every address on it is a seed — the entity itself asserted ownership. (Trust here is only as good as the entity's honesty, but it is a far stronger basis than a guess.)
- **Court-ordered or subpoenaed records.** Investigators get a real-name-to-address mapping from a regulated exchange's KYC records. This is ground truth for law enforcement and the forensic firms that work with them, though not something a retail analyst can independently verify.

Confidence: **verified**. Failure mode: rare, but real — the seed itself can be wrong if, say, an exchange's published list is out of date, or you mis-recorded which deposit address you used. Ground truth is strong precisely because the link was *generated*, not *guessed* — but it is only as good as the act that generated it.

### Source 2 — Self-disclosure (the entity tells you)

Sometimes the entity simply *announces* its addresses, on-chain or off. This is enormously common and underrated:

- **ENS and naming services.** An address that resolves to `vitalik.eth` is self-disclosing — someone set that ENS record, which costs money and is publicly readable. Strong signal, but note the gap: ENS proves *whoever controls this address claimed this name*, not that the name is true. Anyone can register `binance-official.eth` and point it at a scam wallet.
- **Verified contracts.** On Etherscan, a contract whose source code is "verified" and self-identifies (e.g. the official Uniswap V3 router) is effectively self-disclosing its function and often its owner. The DeFi protocols in [DeFi: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) are labeled almost entirely this way.
- **Project documentation and multisig signers.** A DAO that publishes "our treasury is `0xTreas…`" in its docs is self-disclosing. A Gnosis Safe whose signer set is publicly known ties the multisig to named individuals or roles.
- **Verified social posts.** A founder tweeting "my wallet is `0x…`" from a verified account is self-disclosure — weaker than a signed message, stronger than an anonymous forum post.

Confidence: **verified*** — with an asterisk. Self-disclosure proves the *claim*, not the *truth*. The two failure modes are that the entity is **lying** (a scam project "self-discloses" a clean-looking treasury that is actually a drain wallet) or that it **got hacked** and the attacker now controls the "official" address. Treat self-disclosure as strong but cross-check it against behavior: does the wallet labeled "treasury" actually behave like a treasury?

The ENS case deserves a closer look because it's the self-disclosure people trust most and verify least. An ENS name like `treasury.someproject.eth` is *forward*-resolved — the name points at an address — and that's the direction wallets and dashboards display. But the safer check is the *reverse* record: does the address itself claim the name back? Forward-only resolution means anyone can register `coinbase-hot.eth` and aim it at a wallet they control; a UI that shows that name next to the wallet has just rendered an impersonator's self-disclosure as if it were Coinbase's. The same trap recurs with token names (a scam token can call itself "USDC"), contract names, and copied "verified" badges. The defensive read of any self-disclosed label is to ask *who set this record, and can they prove control of the entity they're naming* — not just *what does the record say*. A name is only a self-disclosure if the *real* entity set it; otherwise it's an impersonation wearing the costume of one.

### Source 3 — Clustering propagation (inference at scale)

This is where labels *multiply*. You have one seed; clustering heuristics link it to thousands of other addresses; the seed's label **propagates** to the whole cluster. This is how a single confirmed "Binance" address becomes the "Binance" label on a 50,000-address cluster.

![Before and after view of a single seed label propagating across four linked addresses in a cluster](/imgs/blogs/labeling-and-attribution-4.png)

The figure shows the move at small scale. Before: one address is the confirmed "Binance 14" seed; three neighbors are unlabeled hex. After clustering links them — one co-spends with the seed, one is a sweep destination, one is a deposit address feeding the same hot wallet — the label propagates, and all four now read "Binance 14".

The power and the danger are the same property: **propagation is statistical, and it compounds.** Every heuristic has a false-positive rate. If a clustering rule is 99% accurate but your cluster has 50,000 addresses, you expect ~500 wrongly-included addresses, each now wearing the "Binance" label. Worse, real-world events deliberately break the heuristics:

- **CoinJoin / mixers.** On Bitcoin, a CoinJoin deliberately co-spends inputs from *many unrelated owners* in one transaction — exactly the pattern the common-input heuristic reads as "same owner". Naively clustered, it merges strangers into one bogus mega-cluster. (Mixers are covered defensively in [Tornado Cash and Sanctioning Code](/blog/trading/crypto/tornado-cash-and-sanctioning-code).)
- **Shared custody and omnibus wallets.** A custodian holds funds for thousands of clients in pooled wallets. Cluster it as "one entity" and you've attributed thousands of separate beneficial owners to a single name.
- **Smart-contract wallets and shared relayers.** Funds routed through shared relayer or paymaster infrastructure (the plumbing behind smart-contract wallets) can co-occur without sharing an owner.

Confidence: **inferred** — usually right, occasionally and silently wrong. The propagation rule of thumb: *a label is only as trustworthy as the weakest heuristic between it and the nearest ground-truth seed.* An address one co-spend away from a confirmed exchange wallet is strong; an address seven peel-chain hops away through a mixer is barely a guess.

One subtlety the dashboards hide: **propagation behaves differently on UTXO chains than on account chains**, so the same label carries different confidence depending on where it lives. On Bitcoin (a UTXO model, covered in [How Blockchains Store Data: UTXO vs Account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account)), the strongest heuristic is **common-input ownership** — co-spent inputs almost certainly share a signer — plus the **change-address heuristic** that follows the "change" output of a spend back to the same wallet. These are powerful but brittle: CoinJoin breaks the first by design, and modern wallets that avoid address reuse blunt the second. On Ethereum and other account chains, addresses are *reused* heavily (one address holds a running balance and transacts repeatedly), which makes a different set of signals dominant: shared gas funding, deposit-address sweeps into common hot wallets, and near-simultaneous co-spending. The practical consequence for reading a label: a "same entity" claim on Bitcoin rests on transaction *structure* (who co-spent), while on Ethereum it rests more on *behavioral* patterns (who funds and sweeps whom). When you grade a cluster's confidence, ask which model you're on and which heuristic actually did the linking — they don't fail in the same places.

A second subtlety is **directionality**. Not every link is symmetric. If wallet A funds wallet B's gas, that suggests A controls B (you pay gas for wallets you operate) far more than the reverse. A deposit *into* an exchange tells you the depositor is a customer, not that the depositor *is* the exchange. Good attribution engines weight links by direction and type; naive ones treat every edge as "these two are the same", which is how a customer's personal wallet sometimes ends up wearing the exchange's label. When a surprising label appears, checking the *direction* of the link that produced it often dissolves the surprise.

### Source 4 — Off-chain intel (the human identity)

The first three sources can tell you an address belongs to *an entity* and which on-chain wallets that entity controls. Only off-chain intel can tell you the entity is **a specific named human or company**. Sources:

- **KYC data and leaks.** Regulated exchanges hold real-name records tied to deposit/withdrawal addresses. Lawful access (subpoena) is how investigators deanonymize. Leaked KYC databases are the dark-market version — used by criminals and, controversially, sometimes scraped into commercial attribution.
- **Court filings and indictments.** US DOJ indictments, SEC complaints, and Treasury/OFAC sanctions listings frequently name specific addresses. The OFAC sanctioning of the Tornado Cash contracts on **2022-08-08** is a public, citable address attribution. These are gold-standard off-chain seeds when they exist.
- **OSINT and social.** An address posted in a Discord, a wallet linked from a personal website, a Twitter handle tied to an ENS — open-source intelligence that an analyst correlates by hand.
- **Industry sharing.** Forensic firms and exchanges share known-bad address lists; a "high-risk" tag often originates here.

Confidence: ranges from **verified** (a court filing) down to **guessed** (a single forum post). The failure modes are the worst of any source: off-chain intel can be **stale** (the address changed hands), **biased** (a competitor's smear), **unverifiable** (you can't re-derive a private KYC record), or outright **fabricated**. The lower-right cell of the matrix — off-chain intel with no on-chain confirmation — is where the most dangerous labels live, because they wear the authority of a "real identity" while resting on the flimsiest evidence.

### Why the sources are stronger together than apart

The four sources aren't a menu you pick one from — the best attributions **stack** them, and the stack is what separates a fact from a guess. The reason is that each source covers the others' blind spot. On-chain sources (ground-truth seeds, clustering) are perfectly verifiable but tell you only *which wallets move together*, never the human. Off-chain sources (KYC, filings, OSINT) can name the human but are unverifiable and decay. Self-disclosure sits in between — it can name the entity, but only as a *claim*. Fuse them and the weaknesses cancel: an off-chain seed names the human, an on-chain cluster proves the wallet set, and self-disclosure or a court filing pins the name with citable authority.

That is exactly why the **strongest possible label is a verified seed plus a one-hop on-chain link**, and the **weakest is an off-chain name with no on-chain confirmation**. A useful mental discipline when you read any label: try to reconstruct which of the four sources produced it, and how many you can find. One source, off-chain only? That's a lead. Two sources that agree, one of them on-chain and re-derivable? Now you can act. The number of independent, agreeing sources behind a label is a better confidence gauge than any colored badge the platform paints on it.

There's a corollary that bites in practice: **the same label can be strong on one address and weak on another within the same entity.** "Coinbase" attached to a deposit address you personally withdrew from is verified; "Coinbase" attached to an address eleven hops down a propagation chain is barely inferred — even though the entity page shows the same name and the same logo. Confidence is a property of the *specific address's link to ground truth*, not of the entity's reputation. This is the single most common way careful-looking people get a label wrong: they trust the brand on the page instead of grading the address in front of them.

## How Arkham and Nansen actually build their labels

Now the practical part: when you open an Arkham entity page or see a Nansen "smart money" tag, *what is under it?* Both platforms are, at heart, **attribution engines** that fuse the four sources above into an entity graph. The walkthrough below is how to read what they show you — and how to find the basis behind a label before you bet on it. (For where these tools sit in the wider stack, see [The On-Chain Tooling Landscape](/blog/trading/onchain/the-onchain-tooling-landscape).)

### Arkham: entity pages and the attribution graph

Arkham's product is an **entity graph**: it groups addresses into named entities ("Binance", "Jump Trading", "Wintermute", "Alameda Research") and shows the flows between them. Inside, the engine combines three layers:

1. **A large seed set** — exchange deposit/hot/cold wallets harvested by deposit-and-watch at scale, plus published disclosures, plus off-chain intel.
2. **Clustering** to propagate those seeds across the addresses that move with them.
3. **An "Intel-to-Earn" bounty program** where users submit attributions (with evidence) for rewards — crowdsourced off-chain intel.

What to actually do on an entity page:

- **Read the entity, then drill into the specific address.** The page-level name ("Wintermute") is the cluster's label. Click into the individual address you care about and ask: is *this* address a confirmed seed, or is it wearing the label by propagation? A seed is high-confidence; a propagated member is inference.
- **Look for the basis.** Arkham often distinguishes addresses it has high confidence in from algorithmically-linked ones. Treat the algorithmic ones as inferred.
- **Sanity-check behavior against the name.** If an address is labeled "Binance cold wallet" but it's making 200 tiny DeFi swaps a day, the label is suspect — cold wallets don't day-trade.

### Nansen: "Smart Money" and what a tag really means

Nansen's signature is the **Smart Money** label — wallets it has tagged as belonging to skilled or notable participants (funds, profitable traders, known builders). It's seductive: "smart money is buying X" feels like an edge. But you must know what the tag *is*: Nansen built a library of millions of labeled wallets from the same four sources, and "Smart Money" is a **curated subset** — wallets selected by criteria like historical profitability, being a known fund, or early participation in successful projects.

The traps, which the post [What Is Smart Money On-Chain](/blog/trading/onchain/what-is-smart-money-onchain) covers in depth:

- **Survivorship bias.** "Profitable wallets" are selected *after* the fact. A wallet that 100×'d once and got tagged "smart money" may have been lucky; the thousands of similar wallets that went to zero were never tagged.
- **Label lag.** A wallet earns "smart money" status from past trades, then changes behavior or hands off — the tag persists, the edge doesn't.
- **Reflexivity.** Once everyone watches the same "smart money" wallets, those wallets' moves get front-run and copied, degrading the very edge the label promised.

To check a Smart Money tag's basis: open the wallet, look at *why* it's tagged (fund? profitable trader? early LP?), check whether its recent activity matches the thesis, and never confuse "this wallet was right before" with "this wallet is right now".

#### Worked example: copying a "whale" that is really a \$40M exchange omnibus

You see a wallet labeled "Whale" on a dashboard. It holds **\$40 million** in a single mid-cap token and just sent **\$2 million** of it to an address. The naive read: *a whale is selling; I should sell too.* You dump your **\$10,000** position and eat slippage.

Here's what you missed by trusting the label instead of its basis. The "whale" is actually an **exchange omnibus wallet** — a pooled hot wallet holding that token on behalf of thousands of the exchange's customers. The \$40M isn't one person's conviction bet; it's the sum of thousands of small balances. And the \$2M "sale" wasn't a sale at all — it was an internal sweep from the omnibus wallet to a cold wallet, a routine custody operation with **zero** market impact. There was no whale and no sell. You sold a real \$10,000 position into the noise of someone else's plumbing, and if the token drifted up 5% afterward, your bad-label trade cost you **\$500** plus fees. *A wallet's dollar size tells you nothing until you know whether it's one owner or an omnibus pooling thousands — the label "whale" hid exactly that distinction.*

This is the central practical danger of attribution: the label "whale" is doing a lot of work it can't support. The number (\$40M) is real and on-chain; the *interpretation* (one conviction holder) is an unstated, wrong attribution. The fix is to check the basis — is this address a confirmed individual, or is it inside an exchange's cluster? — before you let it move your capital. Exchange flows specifically are the subject of [Exchange Flows: Inflows and Outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows), and the distinction there between an omnibus and a personal wallet is exactly this one.

## The false-label trap

If a label is a probability, then acting on a *wrong* label is a quantifiable risk, and adversaries actively manufacture wrong labels to exploit you. This is the defender's half of attribution: knowing how labels get faked so a bad one doesn't cost you money or cause you to accuse the wrong party.

![Graph of an address-poisoning attack where verifying the seed avoids a loss but trusting the fake label costs money](/imgs/blogs/labeling-and-attribution-5.png)

The figure traces the canonical attack and the escape. An attacker plants a look-alike address in your transaction history so a fake "trusted" label appears; you, the reader, hit a fork — *trust the name* (and pay the attacker) or *verify the seed* (find no real basis, and abort). The whole defense is that single fork.

### Address poisoning: planting a fake label

**Address poisoning** weaponizes a UI habit. You often pay an address by copying it from your own past transaction history rather than retyping 42 characters. The attacker exploits this:

1. They grind a **vanity look-alike** address whose first and last few characters match an address you really use — say your exchange deposit address `0xDEAD…beef`. (Generating an address with a chosen prefix/suffix is cheap; matching the *whole* 42 chars is impossible, but you only ever *glance* at the ends.)
2. They send you a **dust transaction** (a tiny or zero-value transfer, sometimes a fake token with a real token's name) *from* that look-alike. Now the poisoned address sits in your history, visually adjacent to the real one, often wearing a spoofed label like a fake token name or a copied "verified" tag.
3. Next time you pay "your" address by copying from history, you grab the look-alike. Funds go to the attacker. Irreversibly.

Note what the attacker manufactured: a **fake label**. The dust transfer's whole purpose is to make a malicious address *look* like a known-good one in your interface — to plant an attribution you'll trust.

#### Worked example: a poisoned look-alike baits a \$25,000 transfer

You regularly top up an exchange from cold storage, sending to deposit address `0x9F2c…A41b`. An attacker grinds a look-alike, `0x9F2c…A41b`-shaped but different in the middle, call it `0x9F2c…AA1b`, and on a quiet day sends you a **\$0** transfer of a token they named to mimic USDC. It lands in your history one row above your last real deposit.

A week later you go to send your monthly **\$25,000** top-up. You open your wallet's history, copy "the exchange address" — and your eye, checking only `0x9F2c…` at the front and `…A1b` at the back, grabs the poisoned `0x9F2c…AA1b`. You sign. The **\$25,000** is now in the attacker's wallet, and there is no reversal, no support line, no chargeback. The fake label — "this is your exchange address" — cost you the full transfer.

The defense is to **verify the seed, not the look**: never copy from history; paste the address from the exchange's own deposit page (the ground-truth source) and verify the *full* string or a checksummed scan, not just the ends. *A label you didn't generate yourself — even one sitting in your own transaction history — is an unverified claim, and confirming the ground-truth source is a \$25,000 habit.* Address poisoning is covered as a hazard of address reuse in [Addresses, Wallets, and Contracts](/blog/trading/onchain/addresses-wallets-and-contracts); here it's the cleanest example of an attacker *manufacturing* a false attribution.

### Recycled and re-funded addresses

A subtler false-label source needs no attacker at all: **addresses change hands.** A wallet that was genuinely "Exchange Y hot wallet" in 2021 might be drained, abandoned, and later re-funded by a completely different entity in 2024. A label minted from the 2021 behavior, never refreshed, now points at the wrong owner. Forensic suites version their labels with dates for exactly this reason; consumer dashboards often don't, so a stale "smart money" or "exchange" tag can be attributing a name to whoever holds the keys *today*, which may be nobody you've heard of. When a label and the wallet's *recent* behavior disagree, trust the behavior — the chain is current; the label might be three years old.

The recycling problem is worse on account chains than people expect, precisely because address *reuse* is the norm there. A smart-contract wallet can have its signer set rotated; a multisig can swap owners; an EOA's seed phrase can be sold, leaked, or handed to a new team after an acquisition. None of these events announces itself with a banner — on-chain, the address simply keeps transacting, and only the *pattern* shifts. So a label that was worth acting on becomes a quiet liability. The practical guard is a freshness check: before you act on a label, glance at when the wallet's behavior last matched the label's thesis. A "fund" wallet that hasn't made a fund-like trade in eighteen months and now only interacts with a single gambling contract is wearing a name it has outgrown. Treating an old label as current is the unforced error here, and it costs exactly as much as the bad trade or false accusation it licenses — which, as the worked examples above show, runs from a few hundred to tens of thousands of dollars per mistake.

### The wrong-accusation risk

The false-label trap isn't only about losing money — it's about *making a false claim*. During fast-moving hacks, crowds race to label addresses, and innocent parties get caught: a routing contract gets tagged "the hacker", an exchange's own recovery wallet gets called a "co-conspirator", a victim who interacted with a drained protocol gets flagged "high-risk". Acting on that — publicly accusing, freezing a counterparty, refusing a legitimate withdrawal — is a real harm built on a bad label. The defensive discipline is identical to the trading discipline: **establish the basis and confidence before you act**, and weight an accusation by the strength of its weakest link to ground truth.

## A dated case: attributing the Bybit theft in real time

Return to the **\$1.46 billion** Bybit drain of 21 February 2025, because it is the clearest public demonstration of attribution done well — and of every confidence band appearing at once. The instant the funds moved, the chain showed a flurry of fresh addresses receiving Ether. Nobody had labeled them; they were minutes old. Attributing them correctly, fast, was the whole game, and it ran exactly along the lines this post has laid out.

The **victim** side was the easy part: Bybit's drained cold wallet was a known seed (Bybit had effectively self-disclosed it through prior activity and public proof-of-reserves), so the *source* was **verified** within minutes. The **destination** side was the hard part. The first hop addresses had no history — pure inference. Investigators clustered them by the only signals available: the funds all originated from the same drain, the addresses co-spent and fanned out in coordinated patterns, and the laundering choreography (rapid splitting, swapping ETH for other assets, routing toward bridges and mixers) matched a known playbook. That playbook was the off-chain key: the **Lazarus Group's** tradecraft is extensively documented in prior DPRK-attributed hacks — Ronin (\$625M, March 2022), WazirX (\$230M, July 2024), DMM (\$305M, May 2024) — so the *behavioral* fingerprint, fused with forensic firms' and the FBI's existing Lazarus seed sets, produced the attribution: this is North Korea.

Notice the confidence ladder inside one investigation. The Bybit source wallet: **verified** (self-disclosed seed). The immediate attacker cluster: **inferred** (clustering off a single drain, hardening as the addresses co-spent). The "Lazarus / DPRK" identity: it *starts* as a strong **guess** from behavioral pattern-matching and graduates to **verified** only when official attribution (FBI, OFAC) cites the addresses. A responsible analyst publishing during the first hour would have said "funds from the Bybit exploit, attacker cluster forming, **consistent with** DPRK tradecraft" — hedged to the confidence available — not "North Korea did it" as established fact. The ones who over-claimed early occasionally fingered the wrong intermediary: a routing contract or an exchange's *own* recovery wallet briefly got tagged "the hacker" before the basis was checked.

#### Worked example: why a one-hour-old attacker cluster is still only inferred

Picture the labeler's problem an hour into the Bybit incident. You can see **\$1.46 billion** leaving a verified Bybit wallet and landing across, say, 40 fresh addresses, each holding tens of millions. You're confident the *source* is Bybit. How confident are you that address #37, holding **\$30 million**, is the *attacker* and not, say, an exchange that unknowingly received a slice the thieves are trying to launder through it?

You can't be certain yet, and the dollars make the stakes concrete. If you mislabel address #37 as "attacker" and it's actually a legitimate exchange's deposit address, you might wrongly pressure that exchange to freeze a **\$30 million** balance that includes innocent customers' funds — a real harm built on an inferred label. The link from the verified source to #37 is one hop, which is strong, but "received stolen funds" is not the same claim as "is the thief": stolen money routinely flows *through* innocent intermediaries. The honest label at hour one is "received funds from the Bybit exploit (verified), role unconfirmed (inferred)". Only after #37 co-spends with other attacker addresses, or routes onward in the Lazarus pattern, does the *role* harden. *In a live trace, the link to ground truth can be rock-solid while the role attribution is still a guess — and conflating the two is how investigators wrongly tag a \$30M innocent intermediary as a thief.*

The case is the whole post in miniature: a verified source, an inferred cluster, a guessed-then-verified identity, and a discipline of hedging each claim to the basis actually in hand. The chain made the theft visible; attribution made it *legible*; and getting the confidence levels right was the difference between a correct forensic account and a libelous one.

## Confidence levels: verified, inferred, guessed

Everything above collapses into one practical habit: before you act on a label, classify its **confidence**. Mature attribution products attach a confidence to every label for precisely this reason; when a platform doesn't show one, you must assign it yourself. Three bands.

![Layered stack of confidence levels from verified through inferred to guessed with an action rule](/imgs/blogs/labeling-and-attribution-6.png)

- **Verified.** You or the entity proved the link: a seed deposit you made, an ENS you control, a verified contract, a signed message, a court filing. This is a fact you can re-derive or cite. *Action: you may act on it directly.*
- **Inferred.** Clustering heuristics linked it to a seed: co-spend, deposit reuse, a short peel chain. Probably right, occasionally wrong, and the error is silent. *Action: act, but size down and seek a second confirmation — especially if more than a couple of hops from ground truth.*
- **Guessed.** One off-chain breadcrumb with no on-chain proof: a forum claim, a stale KYC leak, a name dropped in a thread. *Action: treat it as a lead to investigate, never as a basis to trade or accuse.*

The action rule is the payoff of the whole post: **verified → act; inferred → size down and confirm; guessed → never trade or accuse on it alone.** A surprising label — "this random wallet is Jump Trading", "this fresh address is smart money" — should *raise* your confidence bar, not lower it. The more a label would change your action, the harder you check its basis.

#### Worked example: a market-maker move that a wrong label reads as a "sell"

A wallet labeled "Wintermute" (a real, prominent market maker) sends **\$5 million** of a token to a centralized exchange. A dashboard headline screams "Wintermute deposits \$5M to Binance — are they dumping?" You're holding the token; do you sell?

Walk the confidence ladder before you do. First, *is this even Wintermute?* If the label is **inferred** (the address is seven hops from any seed), the whole premise might be wrong. Suppose it checks out as **verified** — it's genuinely a Wintermute wallet. Now the harder question: a market maker's job is to provide liquidity on *both* sides. A \$5M deposit to an exchange is, for an MM, completely routine **inventory management** — they move tokens to the venue where they're quoting so they can *make markets*, which includes being ready to *buy* your sells. Reading that \$5M inflow as "Wintermute is dumping" is attributing a directional *intent* the label can't support. If you panic-sell your **\$20,000** position into that headline and the MM was actually rebalancing to provide bid-side liquidity, you sold the bottom of a wick into the firm that was about to buy it — a clean **\$1,000+** unforced error on a 5% snap-back. *A correct label still doesn't carry intent: a market maker's exchange deposit is plumbing, not a forecast, and reading direction into it is a second, unjustified attribution on top of the first.*

This is the deepest lesson of attribution: even a *verified* label tells you only **who**, never **why**. The identity is one attribution; the intent ("they're selling") is a second one that the chain rarely supports. The discipline is to stop at the layer you can actually verify.

## How to read it: sanity-checking a surprising label before you trade

Here is the end-to-end walkthrough — the checklist you run when a label would change what you do with money. Suppose Nansen flags a fresh wallet as "Smart Money" accumulating a small-cap token, and you're tempted to follow it with **\$15,000**.

1. **Identify the source.** Click into the wallet and find *why* it's labeled. Is "smart money" because it's a known fund (off-chain intel + seed), or because it was profitable on past trades (survivorship-prone inference)? The matrix earlier tells you which bucket you're in.

2. **Find the nearest ground-truth seed.** Trace from the labeled address back toward any confirmed seed. One co-spend from a published fund wallet is strong; a long chain through bridges and mixers is weak. The label is only as good as that path.

3. **Classify the confidence.** Verified, inferred, or guessed? Be honest. A fresh wallet with a "smart money" tag and no funding history is usually **inferred at best** — the platform is extrapolating from a similar pattern, not citing a seed.

4. **Cross-check a second platform — and read the *basis*, not the name.** Open the same address on Arkham (or a block explorer). Do they agree? If they disagree, the next section tells you how to resolve it. If only one platform has any label at all, that's a yellow flag, not a confirmation.

5. **Sanity-check behavior against the label.** Does a "smart money" wallet behave like one — measured entries, real position sizes, a track record — or like a wash-trading bot cycling the same token to manufacture a signal? On-chain behavior is the one thing you can always re-derive yourself.

6. **Size to the confidence, not the excitement.** Verified and behavior-consistent: a normal position. Inferred: a fraction, with a stop. Guessed: a watchlist entry, **zero** capital. The label sets the *ceiling* on your size, never the floor.

The whole walkthrough is one principle applied six times: **read the basis, not the name.** A label is a compressed claim; your job before risking \$15,000 is to decompress it back into the evidence and grade that evidence.

## Why two platforms disagree on the same address

You will constantly find that Arkham labels an address one thing and Nansen labels it another — or one has a label and the other has none. This is not (usually) one of them being broken. It's the inevitable result of two attribution engines using **different seed sets and different heuristics** on the same public data.

![Matrix resolving three cases where Arkham and Nansen disagree by reading each platform's basis](/imgs/blogs/labeling-and-attribution-7.png)

The disagreement comes from concrete differences:

- **Different seeds.** Arkham may have deposit-and-watched an exchange that Nansen never seeded, so Arkham has a confident "Exchange X" label where Nansen has nothing.
- **Different clustering thresholds.** One platform's heuristics are more aggressive (bigger clusters, more propagation, more reach but more false positives); the other is conservative (smaller, safer clusters). The same address can fall inside one's cluster and outside the other's.
- **Different recency.** One refreshed its labels last week; the other's "exchange" tag is from 2022 and the wallet has since been re-funded by someone else.
- **Different off-chain intel.** One bought or crowdsourced a KYC-derived attribution the other doesn't have.

There's also an *incentive* layer underneath the technical one. Attribution is a competitive product, and the way a platform makes money shapes which way it errs. A platform that sells "smart money" feeds to traders is rewarded for **coverage and confidence** — more labels, bolder tags, fresher "alpha" — which biases it toward aggressive clustering and toward keeping exciting labels live even when the basis has gone stale. A forensic firm that sells to law enforcement and compliance teams is rewarded for **defensibility** — a wrong label there can sink a court case or wrongly freeze a customer — which biases it toward conservative, dated, evidence-backed attribution. Neither bias is wrong; they serve different customers. But it means a consumer dashboard and a forensic suite can look at the identical address and reasonably land in different places, and *knowing which incentive you're reading* is part of grading the label. When a flashy tag and a cautious one disagree, the cautious one is usually closer to the verifiable truth — and the flashy one is usually closer to what someone wanted to sell you.

The resolution rule is shown in the matrix: **don't pick the platform you like; pick the basis that's stronger.** Case A — one platform cites a ground-truth seed and the other shows a vague undated tag: trust the seed. Case B — one label is stale (a 2021 KYC leak) and the other reflects fresh on-chain re-funding: trust the current read. Case C — *both* are merely inferred with no hard seed: the honest resolution is **do not trade or accuse on it**, because two guesses don't make a fact. When the bases are equal and weak, the correct action is *no action*.

## Common misconceptions

**"If Arkham labels it 'Binance', it's Binance."** No — it's Binance *with a confidence*. The page-level name is the cluster's label; the specific address you're looking at may be a confirmed seed (strong) or a propagated member (inference). A 99%-accurate heuristic over a 50,000-address cluster still mislabels ~500 addresses. Always drill from the entity to the individual address and ask for the basis.

**"A bigger wallet is a more important signal."** Dollar size is not conviction. The \$40M "whale" in the worked example was an exchange omnibus pooling thousands of customers; its \$2M "sale" was an internal sweep with zero market impact. Size tells you how much is *there*, never *whose* it is or *why* it moved.

**"Smart money labels are an edge."** They're survivorship-biased (winners are tagged *after* winning, losers never tagged), they lag (the tag outlives the behavior), and they're reflexive (once watched, they're front-run). A smart-money tag is a *starting point for research*, not a buy signal — see [What Is Smart Money On-Chain](/blog/trading/onchain/what-is-smart-money-onchain).

**"A label tells me what they'll do next."** A label tells you *who*, at best — never *why* or *what next*. A verified Wintermute deposit to an exchange is routine market-making plumbing, not a sell forecast. Reading intent into an identity is a second, unjustified attribution stacked on the first.

**"An address in my own history is safe to reuse."** Address poisoning plants look-alikes *in your history* precisely to weaponize that assumption. The poisoned `0x9F2c…AA1b` cost a \$25,000 transfer in the example. Verify the ground-truth source (the exchange's deposit page), never copy from history.

## The playbook: what to do with a label

The signal is a label on an address. Here is the if-then checklist for turning it into an action you won't regret.

**Before you act on any label, run the basis check.**
- **Signal:** A platform names an address ("Binance 14", "smart money", "the hacker").
- **Read:** Drill from the entity to the *specific address*; find the nearest ground-truth seed; classify confidence as verified / inferred / guessed.
- **Action:** Verified → act. Inferred → size down and seek a second confirmation. Guessed → watchlist only, zero capital, no public accusation.
- **Invalidation / false positive:** The label and the wallet's *recent behavior* disagree (a "cold wallet" that day-trades, an "exchange" that went quiet two years ago) → distrust the label, trust the behavior, the wallet may be recycled.

**If a label would move your capital:**
- **Signal:** "Smart money is buying", "a whale is selling", "an entity just deposited \$5M to an exchange".
- **Read:** Separate the *identity* attribution (who) from the *intent* attribution (why). The chain supports the first far better than the second. Check whether the "whale" is one owner or an omnibus, and whether the "sell" is a real market order or an internal sweep.
- **Action:** Size to the *who* you can verify, never to the *why* you're inferring. A market maker moving inventory is not a directional signal.
- **Invalidation:** You can't separate intent from identity → trade your own thesis, not the wallet's headline.

**If you're transferring funds based on a labeled address:**
- **Signal:** "This is my exchange / counterparty address."
- **Read:** Did *you* generate this link (ground truth), or are you trusting a label in a UI?
- **Action:** Paste from the authoritative source (the exchange's deposit page); verify the full string or a checksummed scan; never copy from transaction history.
- **Invalidation:** The address arrived via a dust transfer or sits in your history without your having put it there → assume address poisoning, do not send.

**If you're about to publicly attribute (accuse) an address:**
- **Signal:** "This wallet is the hacker / a scammer / a co-conspirator."
- **Read:** What's the weakest link between your claim and a ground-truth seed? Is the address maybe a routing contract, a victim, or a recovery wallet?
- **Action:** Weight the accusation by its weakest link; if any link is merely *guessed*, don't publish.
- **Invalidation:** Independent re-derivation fails, or a second platform with a stronger basis disagrees → retract the claim before it does harm.

The unifying rule across all four: **a label is a probability with a basis attached. Read the basis, grade the confidence, and let the confidence — not the name — set how much you act.** The raw chain is honest but mute; the labels make it speak, and the whole skill is knowing when the labels are telling the truth.

## Further reading & cross-links

- [Address Clustering and Heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the mechanics of how addresses get linked into a cluster (common-input, co-spend, peel chains) that this post's labels sit on top of.
- [How to Trace a Transaction Flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — following value hop-by-hop, the skill that turns a single seed into a mapped cluster.
- [Addresses, Wallets, and Contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — what an address really is, why a wallet spans many addresses, and address poisoning as a reuse hazard.
- [What Is Smart Money On-Chain](/blog/trading/onchain/what-is-smart-money-onchain) — the survivorship, lag, and reflexivity traps behind "smart money" labels in depth.
- [The On-Chain Tooling Landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where Arkham, Nansen, Chainalysis, and the explorers sit in the wider stack.
- [Exchange Flows: Inflows and Outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — reading inflows and outflows correctly, including the omnibus-vs-personal-wallet distinction that breaks naive "whale" reads.
- [Centralized Crypto Exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — how the exchanges whose clusters you're labeling actually custody and move funds.
- [Tornado Cash and Sanctioning Code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — how mixers deliberately break the clustering heuristics that labels depend on.

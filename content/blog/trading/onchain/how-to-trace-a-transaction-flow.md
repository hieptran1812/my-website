---
title: "How to Trace a Transaction Flow: Following the Money Hop by Hop"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The fundamental on-chain investigative skill — given a starting address or transaction, follow value forward and backward, hop by hop, building a flow graph until the trail hits an exchange, a mixer, or a bridge."
tags: ["onchain", "crypto", "transaction-tracing", "flow-graph", "arkham", "breadcrumbs", "metasleuth", "etherscan", "aml", "blockchain-forensics", "peel-chain", "fan-out"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Tracing a transaction flow is the core investigative skill of on-chain analysis: you pick a starting address, follow value **hop by hop** along the transfers, and build a **flow graph** (nodes = addresses, edges = transfers) until the money lands somewhere you can name.
>
> - **What it is:** following value across a chain of transfers — **forward** ("where did it go?") and **backward** ("where did it come from?") — drawing a directed graph of addresses and the amounts that moved between them.
> - **How to read it:** at each hop, look at the outputs, weight them by **value and timing**, and follow the largest meaningful flow. Recognize **fan-out** (one address splits to many) and **fan-in** (many consolidate to one). Use a block explorer for the manual hop, or a flow-graph tool (Arkham, Breadcrumbs, MetaSleuth) to see the whole tree.
> - **What you do with it:** trace stolen or suspicious funds to a **terminal node** — an exchange (subpoena territory, where crypto becomes cash), a mixer (the link is obfuscated), or a bridge (the trail continues on another chain) — and decide whether to report, freeze, or keep digging.
> - **The one rule to remember:** **value is conserved, identity is not.** Money cannot vanish between hops — it splits, swaps form, or crosses a chain — so when a trail seems to "disappear," you have missed a hop, not hit a dead end.

On 21 February 2025, attackers tied to North Korea's Lazarus Group drained roughly **\$1.46 billion** of ETH out of a Bybit cold wallet — the largest theft in the history of money. Within minutes, the funds were in a fresh wallet with no history. Within an hour, that wallet had split the haul across dozens of new addresses. Within a day, the pieces were swapping ETH for other assets, hopping bridges, and trickling toward exchange deposit addresses around the world. And the entire time, anyone — investigators at Elliptic and Chainalysis, journalists, a curious teenager — could watch it happen in real time, because every single hop was written to a public ledger the instant it settled.

That is the strange power of on-chain analysis: the getaway is filmed in 4K, from every angle, forever. The thief cannot un-broadcast a transaction. But "filmed" is not the same as "solved." The Bybit funds did not move in one clean line you could read down a page. They **fanned out** into a tree of hundreds of addresses, **changed form** through swaps, **crossed chains** through bridges, and aimed at the one place the public trail finally goes dark for retail investigators: an exchange's front door, where crypto turns into cash and the rest of the story lives behind a subpoena.

This post teaches the skill that turns that wall of transactions into a map: **tracing a flow**. It is the most fundamental investigative move in on-chain analysis, the thing every fancier technique — clustering, attribution, exchange-flow monitoring, hack forensics — is built on top of. By the end you will be able to take a single address or transaction hash, follow the value forward and backward through the chain of transfers, recognize the patterns funds make as they move (peel chains, fan-out splits, consolidation), handle the moments when value changes form, and know exactly what it means when the trail forks, mixes, or hits an exchange. We use **illustrative placeholder wallets** (Wallet A, B, C; `0xA11ce…`; "Exchange"; "Mixer") for the worked traces — never real addresses dressed up as specific ones — so the pattern is what you learn, not a single case you memorize.

![A flow graph that starts at one labeled wallet, follows hops through a fan-out, and ends at an exchange off-ramp](/imgs/blogs/how-to-trace-a-transaction-flow-1.png)

## Foundations: what "following the money" means on a public ledger

Before we trace anything, we need a clean mental model of what we are even looking at. Strip away the jargon and a blockchain is a **public list of transfers**. Every transfer records the same handful of facts: who sent it (an address), who received it (another address), how much moved, which asset, and when. There is no central bank you have to ask for permission, no "show me account #4471's statement" form to fill out. The entire history of every address is already sitting on the ledger, free to read, the moment you know where to look.

A few terms we will lean on constantly, defined from zero:

- An **address** is a pseudonymous account identifier — a long string like `0xA11ce…` on Ethereum or a `bc1…` string on Bitcoin. It is **not a name.** Nobody's passport is attached to it on the chain. But an address is sticky: every transfer it ever makes or receives is permanently linked to it, and that behavior is what slowly **de-anonymizes** it. Addresses are pseudonymous, not anonymous — a distinction that is the whole reason tracing works.
- A **transfer** (or **transaction**) is one movement of value from a sender to a receiver. It has an amount, a timestamp, a fee, and a unique transaction hash (`0x9f3c…`) that identifies it forever. If you want the field-by-field anatomy of one transaction — the gas, the logs, the internal transfers — the sibling post on [the anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) walks one end to end; here we treat each transaction as a single arrow and zoom out to the *chain of arrows*.
- A **hop** is one step along the trail — one transfer from one address to the next. "Three hops out" means the value has moved through three transfers since your starting point. Tracing is, very literally, the act of counting hops.
- A **flow graph** is the picture you build: **nodes are addresses, edges are transfers.** An arrow from Wallet A to Wallet B with "\$2.0M" on it says "A sent B two million dollars of value." String enough of these together and you have a directed graph — a map of where the money went.

The reason this works at all is a property worth stating plainly, because it governs everything that follows: **value is conserved.** On a ledger, money does not evaporate between hops. If \$2.0M leaves Wallet A, then \$2.0M (minus a tiny network fee) arrives somewhere — maybe at one address, maybe split across thirty, maybe converted into a different token, maybe locked in a bridge to reappear on another chain. But it *arrives*. The single most important habit of a tracer is to refuse to believe money vanished. When a trail seems to die, you have not hit a dead end; you have **missed a hop** — usually a swap that changed the asset, or a bridge that changed the chain. Almost every "I lost the trail" moment is really "I forgot value can change form."

### Forward vs backward tracing

There are exactly two directions you can walk a flow, and choosing the right one is the first decision in any investigation.

**Forward tracing** asks: *where did this money go?* You start at an address (say, the wallet a hacker drained funds into) and follow its **outputs** — the transfers leaving it — hop after hop, toward wherever the funds end up. This is the direction you use to chase stolen money toward a cash-out, or to see where a whale's coins are heading.

**Backward tracing** asks: *where did this money come from?* You start at an address and follow its **inputs** — the transfers that funded it — backward in time, toward the source. This is the direction you use to answer "who funded this scam wallet?" or "what was the original source of the money that just hit this exchange deposit?" Backward tracing is how investigators connect a brand-new, history-less address to a known entity: you walk back until you hit something labeled.

In practice you do both. You start at a known point in the middle and trace **backward to find the source** and **forward to find the destination**, and the address you started from sits between the two. The figure below is the picture to keep in your head.

![One address in the middle, with backward arrows to its funding sources on the left and forward arrows to its destinations on the right](/imgs/blogs/how-to-trace-a-transaction-flow-2.png)

### Fan-out, fan-in, and the off-ramp

Money rarely moves in a tidy single line. As it travels it takes two characteristic shapes, and learning to recognize them on sight is half the skill.

**Fan-out** (also called *splitting* or *spreading*) is one address sending to many. A wallet holding \$1.8M sends \$0.45M to each of four fresh wallets. Fan-out is what funds do when someone wants to **hide the trail in width** — instead of one obvious river, you now have four streams to follow, and if each of those fans out again you have sixteen, then sixty-four. The math gets intimidating fast, which is exactly the point. The defense, as we will see, is that you do not have to follow all of them equally: you weight by value and follow the meaningful ones.

**Fan-in** (also called *consolidation* or *merging*) is the opposite: many addresses sending to one. This is what funds do when it is time to **use** them — you cannot deposit \$1.8M onto an exchange from sixty wallets of \$30k each conveniently, so eventually the streams re-gather into one or a few consolidation wallets before the off-ramp. Fan-in is a gift to the tracer: it pulls a scattered flow back into a single fat edge you can follow cleanly, and the consolidation wallet itself becomes a new, high-value node to investigate.

Finally, the **off-ramp** is where the trail tends to end for a retail investigator: the point at which crypto becomes cash, which on today's chains almost always means a **centralized exchange (CEX) deposit address**. An exchange is a regulated business that knows its customers (KYC — "Know Your Customer"); the moment funds hit a CEX deposit, the question stops being "which address next?" and becomes "who owns this account?", which is a question only the exchange (and a subpoena) can answer. Exchanges, along with **mixers** (which deliberately break the link between deposits and withdrawals) and **bridges** (which move value to another chain), are the **terminal nodes** of a public-ledger trace — the places where the manual, retail trail stops or transforms. We will spend a whole section on each.

If it helps to ground the whole exercise: a great many real traces begin at a documented theft, and they tend to be large enough that the world is watching from the first hop. The chart below shows the biggest on-chain hacks of recent years — the kind of event that kicks off a flow trace that investigators, exchanges, and journalists all run in parallel.

![Horizontal bar chart of the biggest on-chain thefts by US dollars stolen, led by Bybit at 1.46 billion dollars](/imgs/blogs/how-to-trace-a-transaction-flow-8.png)

## The trace workflow, step by step

Tracing has a rhythm. Once you internalize it, every investigation — a \$2M hack, a \$500 scam refund, a whale you want to front-run — follows the same loop. Here is the workflow.

**Step 1 — Pick a starting point and a direction.** You need an anchor: a transaction hash, or an address. For a hack, the anchor is usually the address the stolen funds first landed in. For due diligence, it might be a token's deployer wallet. Decide whether you are tracing **forward** (where is it going?) or **backward** (where did it come from?). Most investigations start forward from the theft and, separately, backward from any suspicious endpoint to meet in the middle.

**Step 2 — List the outputs of the current node.** Open the address on a block explorer ([etherscan.io](https://etherscan.io) for Ethereum, [tronscan.org](https://tronscan.org) for Tron, [mempool.space](https://mempool.space) for Bitcoin, [solscan.io](https://solscan.io) for Solana) and look at the transfers leaving it after the funds arrived. On Ethereum you check both the "Transactions" tab (ETH moves) and the "Token Transfers (ERC-20)" tab (stablecoins and tokens) — money leaves in both lanes and a beginner who only watches ETH misses half the flow.

**Step 3 — Weight the outputs by value and timing.** This is the judgment call that separates a tracer from a tourist. You do **not** chase every output equally. You rank them by **how much value left** and **how soon after the funds arrived**. The large, prompt transfer that moves most of the balance is the trail; a tiny transfer to a known DEX router is probably a gas-fee swap, not the getaway; a transfer that happens three weeks later from a wallet that received deposits from many sources is probably unrelated. **Follow the largest meaningful output**, and note the others to revisit.

**Step 4 — Take the hop. Repeat.** Move to the address that received the largest output. It is now your current node. Go back to Step 2. Each repetition is one hop deeper. You are building the flow graph as you go — node, edge, node, edge.

**Step 5 — Recognize the shape.** Every few hops, ask: is this fanning out (one to many)? Fanning in (many to one)? Is the asset changing (a swap)? Is the chain changing (a bridge)? The shape tells you what tool or technique you need next.

**Step 6 — Stop at a terminal node.** When the trail reaches a known exchange deposit, a mixer, or a bridge, you have hit a terminal node. The trace doesn't necessarily *end* — at a bridge it continues on another chain; at a mixer you switch to correlation — but the nature of the work changes. For a retail analyst, a confirmed CEX deposit is usually the practical finish line: you have followed the money as far as the public ledger lets you, and the rest is a matter for the exchange's compliance team.

That is the whole loop. Everything else in this post is detail on the hard parts of Steps 3, 5, and 6 — how to weight outputs, how to recognize each shape, and what to do at each kind of terminal node.

#### Worked example: tracing \$2M across five hops

A hacker drains a DeFi protocol and the stolen funds — **\$2.0M** in ETH — land in Wallet A (`0xA11ce…`), a fresh address with no prior history. You trace forward. Hop by hop:

- **Hop 1 (A → B):** Wallet A makes two transfers within ten minutes: **\$1.8M** to Wallet B and **\$0.2M** to Wallet B2. You weight by value: \$1.8M is 90% of the flow and moved promptly, so **B is the trail**; you note B2 as a peeled side-branch to revisit. You follow B.
- **Hop 2 (B → C):** Wallet B forwards **\$1.8M** to Wallet C an hour later. One output, almost the full balance — a clean single hop. Follow C.
- **Hop 3 (C → D, E, F):** Wallet C **fans out**, sending **\$0.6M** to D, **\$0.7M** to E, and **\$0.5M** to F. Now you have three streams. They sum to \$1.8M — value is conserved, no money vanished, it just split.
- **Hop 4 (D, E, F → Exchange):** Within a day, all three of D, E, and F send their balances to deposit addresses at the **same centralized exchange**. The streams **fan back in** at the off-ramp.

You started at \$2.0M and, four hops later, accounted for **\$1.8M** of it arriving at one exchange's deposit addresses (plus \$0.2M peeled at Hop 1, still to chase). The trace is "complete" in the sense that the main flow has reached a terminal node: \$1.8M is now subpoena territory, and you would report those deposit addresses to the exchange. **The lesson: by weighting each hop by value, you turned a scary-looking fan-out into one clean story — \$1.8M into one exchange — instead of getting lost chasing the \$0.2M crumb.**

### Tracing backward: who funded this?

The forward trace above answered "where did it go?" Just as often you need the other direction. You arrive at an interesting address — a scam wallet that just rug-pulled, a fresh deposit that hit an exchange, a token deployer you are vetting — and you want to know **who paid for it.** Backward tracing follows the *inputs* of an address back in time toward their source, and it is the technique that connects a brand-new, history-less wallet to a known entity.

The mechanics mirror forward tracing with the arrows reversed. You list the **incoming** transfers to the address, weight them by value and timing, and step backward to the address that *sent* the largest meaningful one. Then you repeat. On a UTXO chain like Bitcoin the inputs are explicit — a transaction literally names the prior outputs it spends — which makes backward tracing especially clean; the [UTXO-vs-account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) post explains why the Bitcoin model hands you the funding sources on a plate. On an account-based chain like Ethereum you read the address's receive history instead.

Backward tracing answers a different and often more powerful question than forward tracing. Forward tells you where the money is heading (useful for freezing it before cash-out). Backward tells you **who is behind an address** (useful for attribution). The classic move is to backward-trace a fresh suspect wallet until you hit its **funding source** — and very often that source is an exchange *withdrawal*. A fresh scam wallet that has never interacted with anything was, at some point, funded with the gas and seed capital to operate, and that funding frequently traces back one or two hops to a KYC'd exchange account. The first dollar in is often the operator's least-careful moment.

#### Worked example: backward-tracing a scam wallet to its funding source

A token rug-pulls and the deployer wallet, holding the stolen **\$400k** of liquidity, is a fresh address with almost no history. You trace **backward** to learn who funded it. The deployer's incoming transfers are: a **\$390k** liquidity-pull from the token's own pool (the theft itself, the forward story), and — crucially — a tiny **\$300** transfer that arrived *before* any of the scam activity, the very first deposit the wallet ever received. That \$300 is the seed gas, and it came from another fresh wallet. You step back one more hop: that wallet was funded by a **\$5,000** withdrawal from a major exchange's hot wallet two days earlier. The backward trail has reached a **KYC'd off-ramp in reverse** — an exchange that knows who withdrew that \$5,000. You cannot unmask the operator yourself, but you have built the on-chain bridge from "anonymous scam deployer" to "this exchange's customer," which is exactly the package an investigator hands to that exchange. **The lesson: backward tracing the *first, smallest* funding transfer of a fresh wallet — not the big theft, but the seed gas — is often what links a pseudonymous actor to a real, identifiable account.**

## Reading value through a chain of transfers

The arithmetic of a trace is mostly bookkeeping, but it is bookkeeping you must do, because the numbers are your proof that you are still on the right money. At every hop, you track **how much value entered the node and how much left, and to where.** When they match (net of fees and the occasional swap), you know you have accounted for the whole flow. When they don't match, you have missed an output — go back and find it.

This is why "follow the largest output" is a rule of thumb, not a law. The largest output is the **most likely** continuation of the main flow, and following it keeps you on the bulk of the value with the least effort. But the disciplined version of the rule is "follow the outputs **in order of value** until you have accounted for the flow you care about." If \$1.8M arrives and the largest output is \$0.9M, you are only tracking half the money — you must also follow the next-largest until your running total covers the amount you are chasing. A tracer who follows only the single biggest edge at every hop will, on a flow that keeps splitting in half, lose track of more money than they keep.

Timing is the other dimension. A transfer that leaves a node **minutes** after value arrived is almost certainly the same money moving on. A transfer that leaves **weeks** later, especially from an address that has received funds from many unrelated sources in the meantime, may be a different dollar entirely — value is fungible once it pools, and you cannot always say *which* incoming dollar funded a given outgoing one. This is the **"commingling" problem**, and it is why prompt, large, single-source transfers are the cleanest links and why mixers (which deliberately pool and delay) are so effective at breaking trails. When in doubt, weight prompt-and-large over late-and-small.

A useful mental check at every node is the **balance equation**: value in = value out + value still sitting in the wallet. If \$1.8M arrived, \$1.5M left in transfers you can see, and the wallet's current balance is \$0.3M, the books balance and you have found every output. If \$1.8M arrived, \$1.5M left, and the wallet now holds nothing, then \$0.3M left in a transfer you have not yet found — go back and look harder, because that missing \$0.3M is, by the logic of conservation, exactly the kind of slice a peel chain hides. Doing this arithmetic at every hop is tedious, and it is also the single habit that most reliably catches the outputs a launderer is counting on you to miss. The ledger cannot lie about totals; a wallet that received \$1.8M either still holds it or sent it somewhere, and "somewhere" is always a transfer you can find.

#### Worked example: weighting the \$1.8M largest output

Wallet C receives **\$1.8M** and, over the next hour, makes four outgoing transfers: **\$0.9M** to D, **\$0.5M** to E, **\$0.3M** to F, and **\$0.1M** to G. A naive tracer follows only D (the largest) and reports "the money went to D." But \$0.9M is only **half** the flow — they have lost the other \$0.9M. The disciplined trace follows outputs in value order and keeps a running total: D (\$0.9M, total \$0.9M), then E (\$1.4M), then F (\$1.7M), then G (\$1.8M) — now the running total equals the \$1.8M that came in, so the flow is fully accounted for and you can stop adding branches. You would then prioritize chasing D and E (which together hold \$1.4M, or 78% of the value) and treat F and G as lower-priority tails. **The lesson: "follow the largest output" means follow outputs in descending value until your running total covers the money you care about — not stop at the single biggest one.**

### The weighting heuristics, in order

Step 3 of the workflow — "weight the outputs" — is where the craft lives, so it is worth spelling out the priority order an experienced tracer uses when staring at a node's outgoing transfers. These are heuristics, not laws, but they are the right defaults:

- **Value first.** The transfer that moves most of the balance is, by default, the main flow. Account for outputs in descending value until your running total covers the money you care about.
- **Timing second.** Among outputs of similar size, the one that left **soonest** after the funds arrived is the most likely continuation. Money that sits for weeks before moving may have commingled with other funds, weakening the link.
- **Freshness of the destination.** A transfer to a brand-new, empty wallet is a stronger "this is the trail" signal than a transfer to an old, busy wallet that receives from hundreds of sources — the busy wallet is more likely a service or an exchange, and the value you sent is now one drop in a pool.
- **Round numbers and repeated amounts.** Launderers and bots often move in round figures (\$50k, \$100k) or repeat an identical amount across many hops. A string of identical transfers is a behavioral fingerprint that ties hops together even when the addresses look unrelated.
- **Known-service penalty.** A transfer to a labeled DEX router, a bridge contract, or a token contract is usually *not* the destination — it is a service the value passes *through*. Trace what comes back out (a swap) or what appears on the other side (a bridge), not the contract itself.

Applied together, these turn a node with ten outgoing transfers into a ranked shortlist of two or three you actually chase. The discipline is to write down *why* you followed the edge you did — "largest, prompt, to a fresh wallet" — so that if the trail later proves wrong, you know which assumption to revisit.

### Building and recording the flow graph

A trace of more than a few hops will outrun your short-term memory, so the working tracer keeps an explicit record of the graph as they build it. This is not bureaucracy — it is the difference between a result you can defend and a vague sense that "the money went somewhere over there."

The minimal record is a list of **edges**: for each hop, write down the source address (abbreviated, e.g. `0xA11ce…`), the destination, the amount, the asset, and the timestamp. Five columns. From that list you can reconstruct the entire flow graph, compute your running totals, and spot where value went missing (a peel) or changed form (a swap). Flow-graph tools like Breadcrumbs and MetaSleuth maintain this record for you and let you export it, but even with a tool you should keep your own annotations on the **load-bearing hops** — the ones your conclusion rests on — because the tool records *what* moved, and you need to record *why you believe* a given edge is the main flow.

Two habits make the record trustworthy. First, **mark every probabilistic link explicitly.** A direct, same-asset, same-chain transfer is *certain* — the ledger proves it. A mixer correlation, a "probably the same actor" merge, or an inferred entity label is a *probability*. Colour-code them, tag them, do whatever it takes, but never let a 70% guess sit in your graph looking like a 100% fact. Second, **record the invalidation for each probabilistic link** — the observation that would prove it wrong. "I think this withdrawal is the same money as that deposit because the amounts match within fees and the timing fits; I would abandon this link if the amounts diverged by more than the plausible fee, or if the timing were reversed." A trace whose every uncertain edge carries its own kill-switch is evidence; a trace that asserts everything with equal confidence is a story.

## Fan-out and the peel chain

When funds want to disappear into width, they fan out — and the most disciplined version of fan-out is the **peel chain**, a pattern worth knowing by name because it is the signature of someone deliberately laundering, and because it is built precisely to exhaust a lazy tracer.

In a peel chain, the bulk of the value walks straight down a single line of wallets — A → B → C → D — while at each hop a **small, fixed slice "peels off"** to a side wallet that heads toward its own cash-out. Hop one forwards \$0.95M and peels \$50k; hop two forwards \$0.90M and peels \$50k; and so on. The main "chain" stays fat and obvious — which is the trick. A tracer who only follows the largest output at every hop stays glued to the main chain and **never notices the steady drip** peeling off the side, which is where the launderer is quietly cashing out in amounts small enough to slip under exchange monitoring thresholds.

![A peel chain where most value forwards down the main line while a fixed slice peels to a side wallet at each hop, all pooling at a cash-out](/imgs/blogs/how-to-trace-a-transaction-flow-4.png)

The defense against a peel chain is the discipline from the last section: **account for the full value at every hop.** If \$1.00M enters Wallet A and only \$0.95M leaves to the next main-chain wallet, the missing \$50k is a flashing light — there is another output, and *that* output is the one the investigation actually cares about. The peel chain hides the meaningful flow in the *small* edges, inverting the usual "follow the biggest" instinct, which is exactly why it works on tourists and fails against anyone tracking the totals.

#### Worked example: a peel chain siphoning \$50k per hop

Stolen funds of **\$1.00M** enter a peel chain. At each of four hops, the main wallet forwards most of the balance and peels **\$50k** to a fresh side wallet: hop 1 forwards \$0.95M (peel \$50k), hop 2 forwards \$0.90M (peel \$50k), hop 3 forwards \$0.85M (peel \$50k), hop 4 forwards \$0.80M (peel \$50k). After four hops, the main chain still visibly holds **\$0.80M** — but **\$200k** (4 × \$50k) has quietly peeled off into four small wallets, each of which deposits its \$50k into a different exchange in amounts small enough to avoid tripping automated review. A tracer who followed only the fat main edge would report "\$0.80M still moving" and completely miss the \$200k that already cashed out. **The lesson: when the value leaving a node is slightly less than the value that entered, the small missing slice is the peel — and on a peel chain, the small edge is the whole point.**

## Fan-in and consolidation

The mirror image of fan-out is fan-in, and it is the moment a trace gets *easier*. Scattered funds — sitting across dozens of fresh wallets after a fan-out — eventually have to be **used**, and using them (depositing to an exchange, funding a big purchase, bridging in size) is clumsy from sixty wallets of \$30k each. So the streams re-gather: many wallets send to one or a few **consolidation wallets**, and the flow you had to split your attention across collapses back into a single fat edge.

The figure below shows the full arc — one source fanning out to four wallets, those merging through two intermediaries, then consolidating into a single wallet that makes the exchange deposit. Width on the left, a single edge on the right.

![One source fanning out to four wallets, merging through two intermediaries, consolidating into one wallet, then an exchange off-ramp](/imgs/blogs/how-to-trace-a-transaction-flow-3.png)

Consolidation is a gift for three reasons. First, it **re-concentrates the value** so you are back to following one big edge instead of many small ones. Second, the consolidation wallet is a **high-value node** in its own right: an address that pulls in funds from many others is exactly the kind of behavior that links them all together (this is the basis of the "common-input-ownership" and co-spend heuristics covered in [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics)). Third, consolidation usually happens **right before the off-ramp**, so when you see fan-in, you are often one hop from the exchange — the end of the retail trail.

The one trap with fan-in: not every merge is the actor consolidating their own funds. A consolidation wallet might be an **exchange's own hot wallet** sweeping many customer deposits, or a service batching transactions. Before you treat a merge wallet as "the launderer's stash," check whether it is a labeled service. If many unrelated flows from across the chain all pour into one address that also constantly pays out, you are probably looking at an exchange or a payment processor, not your suspect.

#### Worked example: a flow that fans out to 30 wallets then re-consolidates

A wallet holding **\$1.8M** fans out to **30** fresh addresses of roughly **\$60k** each (30 × \$60k = \$1.8M — value conserved). To a tourist, the trail just exploded into thirty branches. But a week later, **28** of those 30 wallets send their \$60k each into **two** consolidation wallets, which now hold about **\$840k** each, or **\$1.68M** combined — 93% of the original \$1.8M back in just two nodes. The two consolidation wallets then each make a single deposit to the same exchange. The fan-out looked like it multiplied your work thirtyfold, but the fan-in collapsed it back to two clean edges, and the missing \$120k (the two wallets that didn't consolidate) is a small, low-priority tail to chase later. **The lesson: a fan-out is rarely the end of the story — funds that split must eventually re-merge to be used, so a patient tracer waits for the consolidation rather than drowning in the split.**

## When value changes form: swaps and bridges mid-trace

This is the part that fools almost every beginner, and the reason the "value is conserved" mantra matters so much. Money moving down a trace does not always stay the same asset on the same chain. Two events change its form, and if you trace only "the same token on the same chain," your trail will appear to **dead-end** exactly where the value is actually still moving.

**A swap changes the asset.** Wallet A holds 400 ETH (worth **\$1.2M**) and sends it to a decentralized exchange (a DEX like Uniswap). Out the other side comes **1.2M USDC** — the same value, a different token. If you were tracing ETH, the ETH trail ends at the DEX router (which is not the destination — it is a service everyone uses). The flow continues, but now in USDC. To follow it, you read the swap: the address deposited 400 ETH and received 1.2M USDC in the **same transaction**, and that USDC is now the money you trace. The [tokens, on-chain transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) post covers how to read those two transfer legs inside one swap; for tracing, the key habit is: **when a trail hits a DEX, look for what came back out, not where the deposited asset went.**

**A bridge changes the chain.** Wallet A locks **1.2M USDC** in a bridge contract on Ethereum, and a moment later **1.2M USDC** is minted to a new address on Arbitrum (an Ethereum Layer 2). The value didn't move "to" anywhere on Ethereum — it was **locked here and re-issued there.** If you trace only on Ethereum, the trail ends at the bridge contract. The flow continues on Arbitrum. To follow it, you read the bridge event (which records the destination chain and often the destination address), switch your explorer to the destination chain, and pick the trail back up. Cross-chain tracing is its own discipline — the [cross-chain tracing, bridges, and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) post goes deep on how bridges record their handoffs and why the USDT rails on Tron are the dominant cash-out path — but the core move is simple: **a bridge is not a dead end; it is a doorway to another chain.**

![A trace where 400 ETH swaps to USDC at a DEX, then bridges from Ethereum to Arbitrum, with value conserved at each step](/imgs/blogs/how-to-trace-a-transaction-flow-5.png)

The unifying principle: **value is conserved across form changes.** A swap conserves dollar value while changing the token; a bridge conserves it while changing the chain. So when a trail "ends," ask the conservation question — *the value didn't vanish, so what form did it take?* The answer is almost always "a different token (look for the swap output)" or "a different chain (look for the bridge event)."

#### Worked example: a swap turning 400 ETH into USDC mid-trace

You are tracing **400 ETH** (worth **\$1.2M** at \$3,000/ETH) forward from a drained wallet. At hop 3, the ETH trail ends: Wallet C sends its 400 ETH to a Uniswap router and nothing in ETH comes back to C. A beginner concludes "the money is gone." But value is conserved — you read the swap transaction and see that the **same transaction** that took in 400 ETH paid out **1,200,000 USDC** back to Wallet C. The flow didn't end; it changed form. You now trace the **1.2M USDC** forward instead of the ETH, and it leads on to a bridge and ultimately an exchange. Had you not switched assets, you would have closed the case \$1.2M short. **The lesson: a DEX is a service, not a destination — when the asset you are tracing enters a swap, the trail continues in whatever asset came back out, at the same dollar value.**

## Where the trail ends: exchanges, mixers, and bridges

Every retail trace eventually reaches a **terminal node** — a point where the simple "follow the next transfer" loop stops working. There are exactly three kinds, and each ends the trail in a different way and demands a different response.

![Three terminal nodes — an exchange needing a subpoena, a mixer that breaks the link, and a bridge that continues on another chain](/imgs/blogs/how-to-trace-a-transaction-flow-6.png)

**The exchange (the off-ramp).** When funds hit a deposit address belonging to a known centralized exchange, the on-chain trail effectively ends — not because the money stopped, but because the **next hop is off-chain.** Inside the exchange, your funds become a number in a database, KYC'd to a real human, and the public ledger has nothing more to show you. This is the most common and most desirable place for a trace to end, because it is **not a dead end — it is a door with a name on it.** An exchange is a regulated business; a law-enforcement request or a subpoena can unmask the account holder and, often, freeze the funds. For a retail analyst, "the funds reached Binance/Coinbase/Kraken deposit address X at time T" is a complete, actionable result: you report it. The trap here is mistaking a **DEX router or a token contract** for an exchange — a router is a service everyone uses, not a person's account — so always confirm the deposit address is a labeled CEX before you stop.

**The mixer (the link-breaker).** A mixer (like the sanctioned Tornado Cash on Ethereum, or CoinJoin implementations on Bitcoin) deliberately **pools many users' funds together and pays them out to fresh addresses**, breaking the on-chain link between who deposited and who withdrew. When a trail enters a mixer, you genuinely cannot read a clean next hop — that is the mixer's entire function. The trail does not *continue* so much as **fork into a probability cloud.** What you do instead is **correlation**: match the amount and timing of a deposit against later withdrawals of similar size, look for round-number tells, and treat any match as a *lead*, not a proof. We cover the detection-and-defense side of mixers in depth in [mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation) — framed, as the whole series is, from the investigator's perspective of recognizing and bridging the gap, not operating one. The honest truth is that a well-used mixer can defeat retail tracing; the funds are not invisible, but the link is now statistical rather than certain.

**The bridge (the chain-changer).** As we saw, a bridge is a terminal node only in the sense that it ends the trail **on the current chain.** Read the bridge event, jump to the destination chain, and the trace continues. The only real friction is that bridge fees and slippage slightly change the amount, so when you re-acquire the flow on the other side, you match on **"about the same value at about the same time,"** not an exact figure.

Knowing which terminal you have hit is what tells you whether to **report** (exchange), **switch to correlation** (mixer), or **change chains** (bridge). It is the single most consequential read in a trace.

### Why everything ends at the exchange

It is worth pausing on *why* the exchange is the structural endpoint of nearly every real trace, because it explains the shape of the whole launder route. Crypto is only useful to a thief once it becomes something they can spend — cash, a bank balance, a house. With rare exceptions, the bridge from crypto to the traditional financial system runs through a centralized exchange: deposit coins, sell them, withdraw dollars. That choke point is not an accident; it is where the regulated, KYC'd financial world touches the pseudonymous on-chain world. Every other move in a launder route — the fan-outs, the peel chains, the swaps, the mixers, the bridge-hops — exists to **obscure the path between the theft and that one unavoidable exchange deposit.** The launderer's whole problem is that they must, eventually, walk through a door with a camera over it.

This is also why **exchange-flow monitoring** is such a powerful complementary signal: the deposits a trace is chasing are the same deposits an exchange-flow dashboard sees as inflow. The [exchange flows, inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) post looks at the aggregate version of the same phenomenon — millions of coins moving toward exchanges as a market-wide sell signal — but the per-flow tracer and the macro flow-watcher are reading the same fundamental event from opposite ends. For the tracer, the implication is practical: the closer your trail gets to an exchange, the closer you are to a result you can act on, because the exchange is the one node in the entire graph that can put a name to an address.

The economic reality also bounds how much obfuscation is worth it. Every hop costs a network fee, every swap costs slippage, every mixer costs a service fee, and every fresh wallet costs a little gas to seed. A launderer cannot fan out and mix \$10,000 across a hundred wallets — the fees would eat it. Obfuscation scales with the size of the haul, which is why the biggest hacks produce the most elaborate flow graphs and the small scams produce a lazy one-or-two-hop path straight to an exchange. As a tracer, calibrate your effort to the money: a \$2M flow earns a careful multi-hop trace through swaps and bridges; a \$2,000 scam refund almost certainly went straight to a deposit address you can find in three clicks.

## How to read it: tooling — the explorer vs. the flow graph

You can trace a flow two ways, and a real investigator uses both. The first is **manual hopping on a block explorer**; the second is a **purpose-built flow-graph tool**. Understanding the trade-off is the difference between a trace that takes an afternoon and one that takes a minute.

**Manual hopping (the block explorer).** This is the method we have been describing: open an address on [etherscan.io](https://etherscan.io) (or the right explorer for the chain), read its outgoing transfers, pick the largest meaningful one, click into the receiving address, and repeat. The explorer shows you the ground truth — every field of every transaction, exactly as it is on-chain — and it is **free and universal.** The cost is that *you* are the graph engine: you hold the flow graph in your head or on a notepad, you do the value bookkeeping by hand, and a fan-out into thirty wallets means thirty tabs. Manual hopping is essential for **verifying** a specific hop and for chains or situations where the fancy tools are weak, but it does not scale to a sprawling launder route. It is also the method to learn *first*, because it builds the intuition the tools then automate.

**Flow-graph tools (Arkham, Breadcrumbs, MetaSleuth).** These platforms do the hopping for you and **draw the flow graph automatically.** You paste a starting address, and the tool renders the tree of inflows and outflows as a visual graph — nodes you can expand, edges labeled with amounts, **known entities pre-labeled** (so an exchange deposit address shows up as "Binance" rather than a raw `0x…`). [Arkham](https://arkhamintelligence.com) leans toward entity attribution and a slick "follow the money" visualizer; [Breadcrumbs](https://breadcrumbs.app) and MetaSleuth (by MetaTrust/BlockSec) are built specifically for investigative flow-graphing, letting you trace, annotate, and export a graph for a report. The huge advantage is **labels and scale**: the tool instantly tells you which terminal nodes are exchanges, follows fan-outs across many hops in seconds, and gives you a picture you can hand to someone else. The [on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) post compares these and the rest of the stack in detail.

The catch — and it is the series' recurring theme — is that **the tools' labels are an inference, not gospel.** A flow-graph tool that labels an address "Exchange X" is showing you someone's attribution, which can be stale, wrong, or gamed. The discipline is to use the tool for **speed and breadth** (let it draw the whole tree and flag the terminals) but to **verify the load-bearing hops manually** on the explorer — especially the final one, the deposit address you are about to report. Trust the tool to find the exchange; confirm on Etherscan that it really is one. How those labels get made — and how reliable they are — is the subject of [labeling and attribution](/blog/trading/onchain/labeling-and-attribution).

So the practical division of labor: **use a flow-graph tool to see the forest** (the overall shape, the fan-outs, the candidate terminals), and **drop to the explorer to verify the trees** (the specific high-value hops and the final off-ramp). Beginners who only ever use the visual tool develop a dangerous habit of trusting auto-labels; beginners who only ever hop manually drown in the first big fan-out. The skill is knowing when to zoom out and when to zoom in.

A realistic session combines them in a tight loop. You paste the starting address into a flow-graph tool and let it render two or three hops in every direction — instantly you see whether the money fans out, where the fat edges go, and which terminals are pre-labeled as exchanges. You use that map to pick the **one or two paths that carry the bulk of the value**, and you switch to the explorer to walk *those* paths hop by hop, doing the balance arithmetic and confirming each transfer is what the tool claimed. When you reach a swap, you read the two transfer legs on the explorer and tell the tool the new asset to follow. When you reach a bridge, you read the destination chain off the event and re-anchor the tool there. When you reach what looks like an exchange deposit, you do **not** take the tool's "Exchange X" label at face value for the final, report-worthy node — you open that exact address on the explorer, check that it behaves like a deposit address of that exchange, and only then write it down. The tool gives you reach; the explorer gives you certainty; the report rests on the certainty.

#### Worked example: time-and-amount correlation to bridge a gap

A trail of **\$300k** in USDT enters a mixer at 14:00, and you lose the clean on-chain link. Switching to correlation, you scan the mixer's withdrawals over the next two hours and find a withdrawal of **\$298k** (the \$2k difference being fees) to a fresh address at 14:40 — a close amount match, the right direction, and a plausible delay. That is a **lead worth a 70% mental probability**, not a certainty: you would keep tracing the \$298k address forward as a hypothesis while flagging that the link is statistical. If that \$298k address then bridges to Tron and lands at a known exchange deposit, the *combination* of the amount match, the timing, and the consistent downstream behavior raises your confidence — but you label the whole segment "correlated, not proven" in your report. **The lesson: when a mixer breaks the on-chain link, amount-and-time correlation can bridge the gap with a probability, and you weight a near-exact value match within a tight time window far higher than a loose one.**

## Common misconceptions

**"If the money goes through a mixer or a fresh wallet, the trail is dead."** Not necessarily. A fresh wallet has no history but every *future* move it makes is public, so a fresh wallet only resets the *backward* trail, not the forward one. And a mixer breaks the *certain* link, not all information — amount-and-time correlation, behavioral tells, and downstream mistakes routinely re-establish the trail with high probability. Plenty of "mixed" funds have been traced to cash-outs because the launderer slipped up after the mix. The trail goes *fuzzy*, not dead.

**"You have to follow every branch of a fan-out."** No — that way lies madness, and it is exactly what the fan-out is designed to make you do. You **weight by value** and follow the meaningful flows. A fan-out into thirty wallets of \$60k each is followed by chasing the ones that re-consolidate into size, not by opening thirty tabs forever. The exception is the peel chain, where the *small* edge is the point — which is why you track totals, not just the biggest edge.

**"A DEX or a bridge in the path means the money is gone."** This is the single most common beginner error. A DEX is a service where value **changes asset** (trace what came back out); a bridge is a doorway where value **changes chain** (trace on the destination chain). Neither is a destination. Value is conserved — when the same-asset, same-chain trail ends, the money changed *form*, and your job is to find which form.

**"The exchange is a dead end."** For the *public ledger* trail, yes — but an exchange is the most *useful* place to end up, because it is a door with a name on it. "The funds reached this exchange's deposit address" is an actionable result that can lead to identification and a freeze. A mixer is a true obfuscation point; an exchange is the opposite — it is where pseudonymity ends and KYC begins.

**"On-chain tracing identifies the criminal."** Tracing identifies **addresses and flows**, not people. It tells you *where* the money went, and it tells you when it reached an entity (an exchange) that *can* connect an address to a person. The final step from address to human almost always requires off-chain information — KYC records, a subpoena, an OSINT slip-up. Tracing builds the airtight on-chain half of the case; it rarely closes the human half by itself.

## The playbook: what to do with it

Here is the if-then checklist that turns the whole post into action. Each row is **a pattern you see → the read → the next move → the false-positive to rule out.** The figure summarizes it; the prose below is the operating procedure.

![A six-row decision matrix mapping each on-chain pattern to its meaning, the next move, and a false-positive trap](/imgs/blogs/how-to-trace-a-transaction-flow-7.png)

1. **Anchor and direction.** Start from a transaction hash or address. Decide forward (where did it go?) or backward (where did it come from?). For a theft, trace **forward** from the drain wallet and **backward** from any suspicious endpoint to meet in the middle.

2. **At every hop, account for the full value.** List the outputs, sort by value, and follow them in descending order until your running total covers the money you care about. If less value leaves a node than entered, **find the missing slice** — it is a peel, and it may be the whole point.

3. **Follow the largest meaningful output, weighted by timing.** Prompt + large + single-source = the cleanest link. Late + small + commingled = probably noise. Don't chase a tiny gas-swap to a router as if it were the getaway.

4. **Name the shape at each turn.** Fan-out → weight by value, chase the big children, wait for the re-consolidation. Fan-in → treat the merge wallet as a new high-value start node (but check it isn't an exchange's own sweep). Swap → trace the asset that came **out**. Bridge → jump to the destination chain.

5. **Identify the terminal node and act accordingly.** Exchange deposit → **report it** (and confirm it's a real CEX, not a router). Mixer → **switch to amount-and-time correlation**, label any link "correlated, not proven." Bridge → **change chains** and continue.

6. **Use the right tool for the zoom level.** Flow-graph tool (Arkham / Breadcrumbs / MetaSleuth) to **see the whole tree** and flag terminals fast; block explorer to **verify the load-bearing hops** — above all, the final off-ramp address you are about to put in a report.

7. **Write down the graph as you go, and the invalidation.** Keep the node-and-edge list (or export it from the tool). For every probabilistic link (a mixer correlation, a "probably the same actor" merge), record *what would prove you wrong* — a contradicting amount, a timing that doesn't fit, a label that turns out stale. A trace you can't invalidate is a story, not evidence.

The invalidation discipline is what separates a trace you can stand behind from a just-so story. The chain gives you certainty on the **direct, same-asset, same-chain** hops and only probability across mixers, large commingled pools, and inferred labels. Be loud about which is which. The most expensive mistakes in on-chain investigation come from treating a 70%-probable correlation as a 100% fact and putting the wrong name in a report.

Trace enough flows and the patterns become reflex: the fat edge you follow, the small missing slice you hunt down, the DEX you trace *through*, the bridge you jump, the exchange where you stop and report. The ledger filmed everything. Tracing is just learning to watch the footage.

## Further reading & cross-links

- [Anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) — the field-by-field read of a single transaction, the atom each hop in a trace is made of.
- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — how to tell when many addresses belong to one actor (the co-spend and consolidation logic behind fan-in).
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — where the "Binance / Mixer / Hacker" labels in a flow-graph tool come from, and how much to trust them.
- [Mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation) — the detection-and-defense view of the link-breaker terminal node.
- [Cross-chain tracing, bridges, and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) — following value across the bridge doorway onto another chain.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — Arkham vs. Breadcrumbs vs. MetaSleuth vs. the explorer, and where each fits.
- [Tokens, on-chain transfers, and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) — reading the two transfer legs inside a swap, which is how value changes form mid-trace.
- [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — the most famous mixer, the legal fault line, and why a terminal node became a sanctions question.

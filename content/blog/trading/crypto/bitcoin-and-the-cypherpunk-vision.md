---
title: "Bitcoin and the Cypherpunk Vision: Money Without a Central Authority"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles tour of why Bitcoin exists and how it works: a fixed-supply, censorship-resistant digital ledger maintained by a decentralized network, born from the cypherpunk movement and a deep distrust of central banks."
tags: ["bitcoin", "cryptocurrency", "blockchain", "proof-of-work", "cypherpunk", "decentralization", "sound-money", "self-custody", "mining", "monetary-policy"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Bitcoin was engineered to be money that no government or bank can control: a fixed-supply, censorship-resistant ledger maintained by a decentralized network of computers, all enforcing the same rules and trusting no single keeper.
>
> - It was born from the cypherpunk movement and a distrust of central banks that the 2008 financial crisis crystallized; the very first block carries a newspaper headline about bank bailouts.
> - Its central technical breakthrough is solving the "double-spend problem" — stopping someone from spending the same digital coin twice — without any trusted middleman, using a public ledger and a costly puzzle called proof-of-work.
> - Supply is capped at 21 million coins, issued on a fixed, ever-slowing schedule (the "halving" cuts new issuance in half roughly every four years), which is the source of its "digital gold" pitch.
> - The flip side of having no central authority is that there is no help desk: if you control the keys you control the coins, and if you lose them, no one can recover them. "Not your keys, not your coins."
> - It is genuinely useful as a scarce, hard-to-seize store of value, but it is volatile, slow as everyday cash, and energy-hungry — real trade-offs, not marketing.

Here is a question that sounds simple and turns out to be one of the hardest problems in computer science: how do you send someone a *digital* dollar? Not a claim on a bank balance — an actual unit of money, as final and unforgeable as handing over a paper note, but over the internet, to someone you have never met, with no bank, no Visa, no PayPal sitting in the middle to vouch that the money is real and was not already spent somewhere else. For decades, the honest answer was: you cannot. Digital files are trivially copyable. A song, a photo, a spreadsheet — you can duplicate it a million times at zero cost. If money were just a file, you could pay your rent and then pay it again with the same coin, and again, forever. That is the **double-spend problem**, and it is the reason every digital payment you have ever made routed through a trusted institution that keeps the one authoritative ledger.

Bitcoin is the first system that solved that problem without a trusted institution. That single sentence is the whole story, and the rest of this post unpacks what it means, why anyone wanted it, and what it costs.

The diagram above is the mental model for the entire piece: a bitcoin payment does not get "approved" by a company. Your wallet signs it with a secret key, broadcasts it to a network of independent computers, it waits in a queue, gets bundled into a block by a miner, and then becomes progressively more permanent as more blocks pile on top. There is no point where a bank says yes. The network's rules say yes. If you internalize that flow — and the fact that *no one is in charge of it* — almost everything else about Bitcoin, from its fixed supply to its energy use to the slogan "not your keys, not your coins," falls into place.

![A bitcoin transaction flowing from wallet through the mempool into a mined block and confirmations](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-1.png)

This post is written for a curious reader with no crypto or finance background, but it does not stop at the surface. We will build every concept from zero — what money is, what a blockchain is, what a hash is, what mining actually computes — and then push to a depth a practitioner respects: how proof-of-work really defeats double-spending, why the 21-million cap is credible, what a 51% attack would cost, and where Bitcoin genuinely fits (and genuinely fails) against gold and the dollar. Throughout, dollar amounts and prices are illustrative and approximate as of mid-2026; Bitcoin's price moves violently, so treat every figure as "roughly, around then," never as a quote. Nothing here is investment advice, and at every upside I will name the matching risk.

## Foundations: money, ledgers, and the problem Bitcoin solves

Before we can say what makes Bitcoin special, we need to agree on what money is and why digital money was hard. Let us define every term we will lean on, one at a time.

### The three functions of money

Economists do not define money by what it is made of — gold, paper, shells, or bytes — but by what it *does*. Money does three jobs.

**Medium of exchange.** It is the thing you hand over to get goods, so you do not have to barter. Without money, a baker who wants shoes must find a cobbler who happens to want bread — the awkward "double coincidence of wants." Money dissolves that friction: sell bread to anyone for money, buy shoes from anyone with money.

**Unit of account.** It is the ruler you measure value with. A coffee is "\$4," a car "\$30,000," a salary "\$70,000 a year." Prices, debts, and contracts are denominated in this unit. Crucially, the unit is usually set by a government, because taxes are owed in it.

**Store of value.** It lets you move purchasing power across time — earn today, spend next year. It is always imperfect (inflation nibbles at it), but a \$20 bill in a drawer is worth roughly \$20 next month. Fresh fish, by contrast, is a terrible store of value.

Anything that does all three reasonably well is money. Notice the definition says nothing about who issues it. That opening is exactly what Bitcoin tries to walk through: it argues that a scarce digital token, accepted widely enough, can do these jobs without a state behind it. We will score Bitcoin against all three later, and the figure for that section shows the verdict in advance — strong on store of value, weak on the other two.

### What a ledger is, and why someone has to keep it

Strip money down and what you find underneath is a **ledger** — a record of who owns what. Your bank balance is not a pile of cash with your name on it in a vault; it is a *number in the bank's ledger* saying the bank owes you that much. When you pay a friend \$50, no physical thing moves. The bank decreases your number by 50 and increases theirs by 50. Money, in the modern economy, is mostly just authoritative bookkeeping.

The catch is the word *authoritative*. For the ledger to mean anything, everyone has to agree on one version of it, and that version must be hard to forge. Traditionally, that job falls to a trusted keeper: your bank keeps your balance, Visa keeps the record of your card transactions, the central bank keeps the record of what banks owe each other. We trust them to not double-count, not erase entries, not invent money for themselves. (For how that trusted hierarchy actually creates money, see [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

The whole intellectual leap of Bitcoin is to ask: *what if the ledger had no keeper?* What if thousands of strangers each held an identical copy, and a set of rules — not a company — decided which new entries were valid? Then no one could be censored, no one could secretly print extra money, and there would be no single office to subpoena, hack, or pressure. That is the prize. The difficulty is making thousands of mutually distrustful copies agree, which brings us to the double-spend problem.

### The double-spend problem

Digital information is perfectly copyable, and that is fatal for digital cash. If a coin is just a file — say a string of numbers that means "1 coin" — then nothing physically stops you from sending the same file to two different people. Both would believe they were paid. The coin would have been spent twice. In the physical world this cannot happen: handing a \$20 bill to one person means you no longer have it to hand to another. Physical scarcity enforces single-spend automatically. Digital scarcity does not exist by default; it has to be manufactured.

Pre-Bitcoin digital cash projects (DigiCash in the 1990s, e-gold, various proposals) all "solved" double-spend the obvious way: a central server kept the master ledger and checked every payment against it. That works, but it reintroduces exactly the trusted keeper we were trying to escape — a company that can fail, be shut down, freeze your funds, or be ordered to. DigiCash went bankrupt; e-gold was prosecuted and shuttered. The unsolved problem was double-spend resistance *without* a central server. That is the specific gap Bitcoin filled.

### A blockchain, a block, and a hash

A **blockchain** is the data structure that holds Bitcoin's ledger. Picture a notebook where you can only ever *append* new pages, never edit or tear out old ones. Each page is a **block**: a bundle of recent transactions plus a small header. Blocks are produced roughly every ten minutes, and each new block contains a fingerprint of the block before it, chaining them in strict order — hence "block-chain." Because every block points back to its predecessor, you cannot quietly alter an old block without breaking every block that came after it. The chain is tamper-evident by construction.

The "fingerprint" is a **cryptographic hash**. A hash function takes any input — a word, a file, an entire block — and produces a fixed-length string of characters that acts as a unique digital fingerprint. Bitcoin uses one called SHA-256, which always outputs 256 bits (a 64-character hexadecimal string). Three properties make hashes the workhorse of the system:

- **Deterministic.** The same input always yields the same hash.
- **Avalanche.** Change the input by a single character and the output changes completely and unpredictably. There is no "close."
- **One-way.** Given a hash, you cannot work backward to the input; the only way to find an input that produces a target hash is to try inputs and check, one by one.

That last property is the engine of everything. Because you cannot reverse a hash, the only way to find a special one is brute-force guessing — and brute-force guessing is *work*. Hold that thought; it becomes proof-of-work in a moment.

### Public keys, private keys, and wallets

Owning bitcoin is not like having a balance at a bank that knows your name. It is about controlling a secret. Bitcoin uses **public-key cryptography**, which gives every user a mathematically linked pair: a **private key** (a secret number you must never reveal) and a **public key** (derived from the private key, safe to share). From the public key you derive an **address** — a shorter string, like an account number, that others send coins to.

The magic is this: you can *prove* you own the private key without revealing it, by producing a **digital signature**. When you spend coins, your wallet signs the transaction with your private key. Anyone on the network can verify, using your public key, that the signature is valid — that the rightful owner authorized this — without ever learning the private key itself. It is like a wax seal only you can stamp but anyone can recognize.

A **wallet** is simply the software that manages your keys and addresses. It does not "hold" coins the way a leather wallet holds bills; the coins live on the blockchain. The wallet holds the *keys* that authorize moving them. Most modern wallets generate everything from a single **seed phrase** — typically 12 or 24 ordinary words — from which all your private keys, public keys, and addresses are mathematically derived. Whoever has the seed phrase controls all the coins, full stop. There is no password reset. We will see the exact hierarchy — seed to private key to public key to addresses — in a dedicated figure later, and it is worth staring at, because misunderstanding it is how people lose money.

### Nodes and what "decentralized" really means

A **node** is a computer running Bitcoin software that keeps a full copy of the blockchain and independently checks every rule. There are tens of thousands of them, run by individuals, businesses, and enthusiasts around the world. When a new block arrives, every node verifies it: are all the signatures valid? Does anyone try to spend coins they do not have or spend the same coins twice? Does the block obey the issuance schedule? If a block breaks any rule, honest nodes simply reject it — no vote, no committee, just each node enforcing the same rulebook.

This is what **decentralized** actually means here, and it is worth being precise because the word is abused. It does not mean "spread across many servers" (a bank could do that). It means *no single party can change the rules or the records*. There is no CEO of Bitcoin, no headquarters, no server you can seize to stop it. The rules live in software that thousands of people independently choose to run, and changing them requires convincing a critical mass of those people — which, for the core monetary rules like the 21-million cap, has proven effectively impossible. Decentralization is not a feature bolted on; it is the entire point. Everything Bitcoin sacrifices — speed, convenience, energy efficiency — is sacrificed to preserve it.

It is worth separating two roles that beginners often conflate: nodes *verify and store* the ledger, while miners *extend* it. Anyone can run a node on a cheap computer; it costs almost nothing and grants no reward, but it lets you check the entire chain yourself and refuse any block that breaks the rules. Mining, by contrast, is the expensive, competitive race to produce the next block. The two roles keep each other honest: miners cannot sneak invalid blocks past the thousands of watchful nodes, and nodes cannot produce blocks without doing the work. A user who runs their own node is, in a real sense, trusting no one — not even the miners — because their software independently validates every rule before accepting anything. That is the deepest form of the "trust no keeper" principle, and it is available to anyone willing to run free software.

### Confirmations, the mempool, and transaction fees

Two more terms close the loop on figure 1. When you broadcast a transaction, it does not go straight into a block; it sits in the **mempool**, the network-wide waiting room of unconfirmed transactions. Miners pick which ones to include, and because block space is limited, they rationally pick the ones offering the highest **fees** (paid by the sender, denominated per unit of transaction size). When the network is busy, fees rise — you bid for scarce space — and low-fee transactions may wait hours. When it is quiet, fees fall to pennies. This fee market is also Bitcoin's long-term security plan: as the block reward keeps halving toward zero, fees are meant to become the dominant payment that keeps miners working.

Once your transaction is in a block, it has one **confirmation**. Each subsequent block mined on top adds another. Why wait for several? Because, very rarely, two miners find a block at almost the same instant and the chain briefly splits; the network resolves it within a block or two by following the longest chain, and the "losing" block's transactions return to the mempool. Waiting for about six confirmations (roughly an hour) makes the probability of such a reversal vanishingly small — which is why exchanges and large payments wait, while a cup of coffee might accept zero or one. "Finality" on Bitcoin is therefore not a yes/no switch but a probability that climbs toward certainty as blocks pile on.

### Proof-of-work and mining, step by step

Now we can assemble the centerpiece. The problem: thousands of nodes with no boss must agree on *one* ordering of transactions, and must do so even though some participants might be malicious. The solution Bitcoin's pseudonymous creator proposed is **proof-of-work**, and the participants who perform it are **miners**.

Here is the mechanism. To add the next block, a miner must find a special number (called a nonce) such that when the block's contents are hashed, the resulting fingerprint falls below a target — in practice, a hash that starts with a long run of zeros. Because hashes are one-way and unpredictable, there is no clever shortcut; miners must guess trillions upon trillions of nonces per second until one happens to produce a qualifying hash. That relentless guessing burns electricity. When a miner finally finds a valid hash, they broadcast the new block; every other node can verify it in a millisecond (just hash it once and check), even though finding it took the whole network enormous effort. This asymmetry — *hard to produce, trivial to verify* — is the heart of it.

Why bother? Because the work makes rewriting history expensive. To change a past transaction, an attacker would have to redo the proof-of-work for that block *and every block after it*, faster than the entire honest network is extending the real chain. The deeper a transaction is buried, the more astronomically costly it becomes to reverse. Money is created from energy and locked in place by energy. The miner who finds a block is rewarded with newly issued bitcoin (the "block reward") plus the transaction fees in that block — which is both how new coins enter circulation and why anyone pays the electricity bill to secure the network. We will price out exactly how brutal a rewrite would be when we get to the 51% attack.

## The 2008 genesis: a headline carved into the first block

Bitcoin did not arrive in a vacuum. It arrived in the wreckage of the 2008 financial crisis, and it wears that origin on its sleeve.

In October 2008, an anonymous author (or group) using the name **Satoshi Nakamoto** published a nine-page paper titled "Bitcoin: A Peer-to-Peer Electronic Cash System." It described, in spare technical prose, a system for electronic cash that needed no trusted third party. On January 3, 2009, Satoshi mined the very first block — the **genesis block** — and embedded a short piece of text in it: *"The Times 03/Jan/2009 Chancellor on brink of second bailout for banks."* It was the front-page headline of the London *Times* that day. That message is not decoration. It is a timestamp (proving the block was not made before that date) and a manifesto in one line: this thing was built because the existing money system was, in its creator's view, broken — central banks printing money, governments bailing out the very banks whose risk-taking caused the crisis, and ordinary savers footing the bill through inflation and taxes.

That sentiment did not come from nowhere. It came from the **cypherpunks** — a loose movement of cryptographers and programmers active since the early 1990s who believed that privacy and freedom in the digital age would have to be defended with cryptography, not laws. Their mailing list incubated ideas that fed directly into Bitcoin: Adam Back's "Hashcash" (proof-of-work, originally an anti-spam tool), Wei Dai's "b-money," Nick Szabo's "bit gold." Their ethos was distilled in Eric Hughes' 1993 "A Cypherpunk's Manifesto": *"Cypherpunks write code."* The conviction was that a system you could be *forced* to trust was a system that would eventually be abused, so you should remove the need for trust wherever possible. Bitcoin is that conviction shipped as working software. (For the broader machine Bitcoin was reacting against — central banks, the dollar, the global plumbing — see [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system).)

Who is Satoshi Nakamoto? Nobody knows. After a couple of years of public posts and code, Satoshi handed off the project to other developers in 2010 and vanished, leaving roughly a million early-mined bitcoins that have never moved. The fact that the founder is anonymous and gone is, paradoxically, a feature: there is no leader to arrest, bribe, or coerce, and no one whose change of heart could redirect the project. Bitcoin had to be able to run without its creator, and it has, for over fifteen years.

### How proof-of-work solves double-spending without a third party

Let us connect genesis to mechanism. Recall the double-spend problem: stop someone from spending the same coin twice, with no central server to check. Bitcoin's answer is to make the *ordering* of transactions objective and expensive to fake.

Every ten minutes, one miner wins the proof-of-work lottery and gets to append the next block, fixing the order of the transactions inside it. If you try to double-spend — broadcasting two conflicting transactions that both spend the same coin — only one can be included in a block; the moment it is, every honest node treats the coin as spent and rejects the other. To get the *second* transaction accepted, you would have to build an alternative chain where your first transaction never happened and overtake the real chain in length. But overtaking means out-mining everyone else combined, which, as we will quantify, costs more than any plausible double-spend is worth. So double-spending is not made *impossible* by a referee; it is made *uneconomical* by physics and arithmetic. The before-after figure later in the post lays the two worlds side by side: the trusted keeper who *could* reverse you, versus the trustless ledger where reversal means out-spending the planet.

## Mining, difficulty, and the fixed-supply schedule

We have said miners "guess until they find a qualifying hash." But guessing speed varies wildly — more miners join, hardware gets faster — so how does a block still appear every ten minutes? Through **difficulty adjustment**.

Roughly every two weeks (every 2,016 blocks), the network looks at how fast the last batch of blocks was found and recalibrates the target. If blocks came too fast, it demands more leading zeros (harder); if too slow, fewer (easier). The aim is a steady ten-minute average regardless of how much computing power — **hashrate** — is pointed at the network. This is elegant: it means you cannot speed up issuance by adding miners. Throw ten times more machines at Bitcoin and you do not get coins ten times faster; you just make the puzzle ten times harder and keep the same ten-minute drip. The schedule is defended against its own success.

### The 21-million cap and the halving

The supply rule is hard-coded and brutally simple. New bitcoin enters circulation only as the block reward paid to miners. That reward started at **50 BTC per block** in 2009 and is cut in half every 210,000 blocks — about every four years — in an event called the **halving**. So:

- 2009–2012: 50 BTC per block
- 2012–2016: 25 BTC per block
- 2016–2020: 12.5 BTC per block
- 2020–2024: 6.25 BTC per block
- 2024–2028: 3.125 BTC per block

and so on, halving and halving until, sometime around the year 2140, the reward rounds to zero and no new bitcoin is ever created. The total that will ever exist is just under **21 million coins**. Because the reward keeps halving, the issuance curve is front-loaded: more than 19.7 million of the 21 million already exist by mid-2026; the remaining ~1.3 million will trickle out over the next century-plus. The timeline figure below marks the whitepaper, the genesis block, the first three halvings, and the 2024 spot ETF, so you can see the issuance steps and the institutional arrival on one line.

![Timeline from the 2008 whitepaper through the halvings to the 2024 spot ETF](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-2.png)

This is the source of the "digital gold" pitch. Gold is valued partly because its supply is hard to expand — you have to find and dig it. Bitcoin makes scarcity *mathematical and known in advance*: anyone can compute exactly how many coins will exist on any future date. No central bank can decide to "print more" in a downturn, because there is no central bank and the rule is enforced by every node. That predictability is the feature. The cost of the feature is that Bitcoin cannot expand its supply to absorb a shock or fund a stimulus the way a flexible fiat currency can — a property defenders call discipline and critics call rigidity.

#### Worked example: the 21-million cap and the halving schedule

Let us actually count, to show why the cap is not a vibe but arithmetic. Each "halving era" lasts 210,000 blocks. Multiply the era's block reward by 210,000 to get all the coins issued in that era.

- Era 1 (50 BTC): 210,000 x 50 = 10,500,000 BTC
- Era 2 (25 BTC): 210,000 x 25 = 5,250,000 BTC
- Era 3 (12.5 BTC): 210,000 x 12.5 = 2,625,000 BTC
- Era 4 (6.25 BTC): 210,000 x 6.25 = 1,312,500 BTC
- Era 5 (3.125 BTC): 210,000 x 3.125 = 656,250 BTC

Notice each era issues exactly half the coins of the one before. Adding up an infinite series where each term halves gives a finite total — the same kind of sum where 10 + 5 + 2.5 + 1.25 + ... approaches but never exceeds 20. Here the eras sum to about 21,000,000 BTC. The first era alone minted half of all bitcoin that will ever exist; by the end of era 4 (2024), roughly 19.69 million — about 94% — were already out. So the "scarcity" is overwhelmingly already baked in; the drama of future halvings is mostly about the shrinking *flow* of new supply, not the *stock*.

**The takeaway:** the 21-million cap is not a promise that can be revised — it is a convergent sum hard-coded into the issuance rule and checked by every node, with about 94% of all coins already mined.

#### Worked example: a miner's economics, electricity versus the block reward

Mining only happens because it pays. Let us trace one miner's rough economics in the 3.125-BTC era, with illustrative numbers (BTC ~\$60,000, approximate as of mid-2026; real prices and electricity rates vary enormously).

Suppose a miner runs a modern machine that does 100 trillion hashes per second (100 TH/s) and draws about 3,000 watts. The network as a whole, say, runs at 600 quintillion hashes per second (600 EH/s). The miner's share of the total is therefore 100 TH/s ÷ 600,000,000 TH/s, which is about 0.0000167% of the network.

The network produces 144 blocks a day (one every ten minutes), each paying 3.125 BTC plus fees. Daily new issuance is 144 x 3.125 = 450 BTC. The miner's expected share is 450 BTC x 0.0000167% which is about 0.000075 BTC per day, worth roughly 0.000075 x \$60,000 = \$4.50 per day in rewards.

Now the cost. At 3,000 watts running 24 hours, the machine uses 72 kilowatt-hours per day. At an electricity price of \$0.05 per kWh (cheap industrial power), that is 72 x \$0.05 = \$3.60 per day. So this miner nets roughly \$4.50 − \$3.60 = \$0.90 per day before hardware depreciation. At \$0.10 per kWh, the bill is \$7.20 and the miner *loses* about \$2.70 a day — they would switch off.

**The takeaway:** mining is a thin-margin commodity business gated almost entirely by the price of electricity, which is why miners cluster around the cheapest power on earth and why a falling Bitcoin price or rising power price flushes the least efficient miners out almost immediately.

That last point matters for security: difficulty adjustment means that when unprofitable miners switch off, the puzzle gets easier for those who remain, so the network keeps humming even as the cast of miners churns.

## Keys, wallets, and self-custody: "not your keys, not your coins"

We defined keys and wallets in the foundations; now let us make the ownership model concrete, because it is where the cypherpunk ideology meets the user's reality — sometimes painfully.

When you "own" bitcoin, what you actually own is the ability to produce a valid signature spending certain coins on the ledger. That ability lives entirely in your private key (and the seed phrase it comes from). This leads to two radically different ways to hold bitcoin:

- **Self-custody.** You hold your own keys — on a phone app, a desktop wallet, or a dedicated hardware device. You alone can move the coins. No company can freeze, reverse, or seize them, and no company can lose them on your behalf. This is the cypherpunk ideal: you are your own bank.
- **Custodial.** You leave your coins with an exchange or service (Coinbase, Binance, etc.), which holds the keys for you. Convenient — they handle security, offer password resets, and let you trade easily — but now you are trusting a third party again, the very thing Bitcoin was built to avoid. If they get hacked, go bankrupt, or freeze your account, your coins are at their mercy.

The slogan **"not your keys, not your coins"** compresses the whole lesson: if you do not control the private keys, you do not really own the bitcoin — you own a *claim* on a company that holds bitcoin, which is exactly the kind of trusted-third-party promise that has failed spectacularly (Mt. Gox, FTX, and others, which we will revisit). The figure below shows the key hierarchy that self-custody rests on: one seed phrase derives your private key, which derives a public key, which derives the addresses people pay you at, all managed by the wallet app.

![Tree from seed phrase to private key to public key to receiving addresses managed by a wallet](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-7.png)

The trade-off is stark and worth dwelling on. Self-custody gives you sovereignty and removes counterparty risk, but it transfers *all* the responsibility to you. Lose the seed phrase and the coins are gone forever — there is no recovery, because there is no one with the authority to recover them. Get phished into revealing it and a thief drains you instantly and irreversibly. Estimates suggest several million bitcoin — possibly 3 to 4 million of the ~19.7 million mined — are permanently lost to forgotten keys, discarded hard drives, and deaths without inheritance plans. The same property that makes the coins unseizable by a government makes them unrecoverable by you. There is no upside here without this exact downside.

#### Worked example: buying \$100 of bitcoin and self-custodying it

Let us walk the full path a beginner actually takes, with the trade-offs labeled.

1. **Buy.** You open an account at an exchange, pass identity verification, and buy \$100 of bitcoin. At an illustrative \$60,000 per coin (approximate, mid-2026), \$100 buys 0.001667 BTC. Note the unit: bitcoin is divisible to 8 decimal places, and the smallest unit (0.00000001 BTC) is called a *satoshi*, so your \$100 is about 166,700 satoshis. You do not need a "whole coin."
2. **Fees.** The exchange charges, say, a 1% trading fee, so \$1 goes to them; you net ~0.00165 BTC. At this point the coins are *custodial* — the exchange holds the keys. Your coins are exposed to the exchange's solvency and security.
3. **Self-custody.** You install a wallet, which generates a 24-word seed phrase. You write it on paper (never a photo, never the cloud) and store it safely. The wallet shows you a receiving address.
4. **Withdraw.** You send the bitcoin from the exchange to your wallet's address. This is an on-chain transaction, so you pay a network fee — say \$2 worth — to the miners (not the exchange). After a few confirmations, the coins are yours alone.

Now you truly own ~0.00163 BTC. No one can freeze it. But if you lose that paper, the \$98-ish of value is gone with it, and Anthropic, Coinbase, the FBI, and Satoshi himself could not get it back.

**The takeaway:** moving from custodial to self-custody is the moment you trade convenience and a safety net for genuine ownership and total responsibility — the cypherpunk bargain in miniature.

## Volatility and the store-of-value debate

Bitcoin is pitched as "digital gold," a store of value. The awkward fact is that over short horizons it is one of the most volatile assets on earth. Both things can be true, and understanding why is central to using the word "store of value" honestly.

**Volatility** is how much a price swings around. Bitcoin has had multiple drawdowns of more than 50% — and several of more than 80% — within its history: roughly \$20,000 to \$3,000 in 2018, roughly \$69,000 to \$16,000 in 2021–2022, with many lesser plunges in between, and equally dramatic rises. A "store of value" that can halve in months is, for someone who might need the money next quarter, a poor one. Over multi-year horizons, holders who bought and waited have generally come out far ahead, but that is survivorship-flavored hindsight and says nothing about any individual's timing. The honest framing: Bitcoin has behaved like a *high-volatility, long-horizon* store of value — potentially excellent if you can stomach the swings and hold for years, potentially ruinous if you are forced to sell into a crash.

#### Worked example: a 50% drawdown on a \$10,000 position

Volatility is abstract until it is your money, so let us make it concrete. Suppose you put \$10,000 into bitcoin at \$60,000 per coin. You own 0.1667 BTC.

A 50% drawdown — entirely ordinary for Bitcoin — takes the price to \$30,000. Your 0.1667 BTC is now worth 0.1667 x \$30,000 = \$5,000. You have lost \$5,000 on paper. The psychological trap is the asymmetry of recovery: to get back from \$5,000 to your original \$10,000, the price does not need to rise 50% — it needs to rise *100%*, from \$30,000 back to \$60,000, because you are now growing a smaller base. A 50% fall requires a 100% gain to undo. If the drawdown were 80% (price to \$12,000, position to \$2,000), you would need a 400% gain just to break even.

**The takeaway:** drawdowns compound against you — the deeper the fall, the disproportionately larger the rise needed to recover — which is exactly why position sizing and time horizon, not price predictions, are what separate a survivable bet from a wipeout.

A genuine store of value also has to *survive*, not just appreciate. Bitcoin's survival case rests on its fixed supply and decentralization; its survival risks include catastrophic bugs, a coordinated state crackdown, a better successor technology, or simply society deciding it does not want it. None of these has happened, but none can be ruled out. Name the upside, name the risk.

## The energy debate

The same proof-of-work that secures Bitcoin against rewriting consumes a great deal of electricity — by various estimates, on the order of a mid-sized country's annual usage (often cited in the range of roughly 100 to 150 terawatt-hours a year, approximate and rising or falling with price and hashrate). This is Bitcoin's most contested external cost, and both the criticism and the defense deserve a fair hearing.

**The criticism.** A payment network that uses as much power as a nation, while settling far fewer transactions than Visa, looks grotesquely inefficient and carbon-intensive, especially where miners burn coal. If you believe Bitcoin's social value is small, its energy use is close to pure waste.

**The defense.** The energy is not spent "per transaction" in any meaningful sense — it secures the entire ledger and its whole stored value, not individual payments, and that cost barely changes whether the network processes a thousand transactions or a million. Miners chase the cheapest power on the planet, which is disproportionately stranded or surplus energy: hydro that would otherwise be spilled, flared natural gas that would otherwise be burned for nothing, curtailed solar and wind. A growing share of mining uses renewables or waste energy, and miners act as a flexible, interruptible load that can stabilize grids and monetize energy that has no other buyer. Defenders also note that the legacy financial system — bank branches, data centers, cash logistics, gold mining — consumes vast energy too, just less visibly.

The honest position is that the energy use is real and large, its carbon intensity depends heavily on the local power mix, and whether it is "worth it" depends entirely on how much value you assign to having censorship-resistant, fixed-supply money. The technology does not let you keep the security and skip the energy; the cost *is* the security. The graph figure below makes the link explicit and shows why that energy is the thing standing between an attacker and the ledger.

![Graph of how mining secures the chain and what a 51 percent attack would require](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-5.png)

#### Worked example: what a 51% attack would cost

The figure above claims rewriting history means out-mining the network. Let us price it, roughly. To pull off a **51% attack** — controlling more than half the hashrate so you can build a longer chain and reverse recent transactions — an attacker needs more machines than everyone else combined.

Say the network runs at 600 EH/s. To exceed it, you need to add at least ~600 EH/s of your own. A top-tier miner does ~0.2 EH/s (200 PH/s) and costs, say, \$5,000. To reach 600 EH/s you would need about 600 ÷ 0.2 = 3,000 such machines... no — 600 EH/s ÷ 0.0002 EH/s per machine = 3,000,000 machines. At \$5,000 each, that is 3,000,000 x \$5,000 = \$15 billion in hardware alone, assuming you could even buy that many (you could not; you would bid the price to the moon). Then add the electricity: 3 million machines at 3,000 watts is 9 gigawatts of continuous draw — comparable to several large power plants — costing millions of dollars *per day* to run.

And the payoff? Even if you succeeded, you could only reverse your own recent transactions (double-spend coins you already moved); you could not steal others' coins or mint new ones, because the signature rules and issuance cap are checked by every node regardless of who mines. The moment the market noticed an attack, the price — and thus the value of your loot and your \$15 billion of now-useless rigs — would collapse. You would have spent a fortune to vandalize the very thing whose value you depend on.

**The takeaway:** Bitcoin's security is economic, not magical — it is safe because attacking it costs far more than it could ever yield, and the attacker's own incentives point at protecting the chain rather than breaking it.

## Bitcoin versus gold versus fiat

It helps to put Bitcoin beside the two monetary systems it is most often compared to. The matrix figure below scores all three on the properties that matter; the prose here explains the scores.

![Matrix comparing Bitcoin, gold, and the fiat dollar across supply, transfer, control, and track record](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-4.png)

**Fiat money** (the dollar, euro, yen) is money by government decree, backed by the issuing state's authority and the requirement that taxes be paid in it. Its supply is *flexible*: central banks can expand or contract it to manage the economy, which is powerful in a crisis (you can flood the system with liquidity in 2008 or 2020) but means the supply is, ultimately, a policy choice that can erode your savings through inflation. Fiat transfers easily through the banking system but is fully controllable — accounts can be frozen, transactions blocked, currencies devalued overnight. (For how that flexibility is actually exercised, see [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system).)

**Gold** is the ancient hard money. Its supply grows only as fast as miners can dig (~1.5% a year), so it cannot be debased by decree, and it has a multi-millennia track record as a store of value. But gold is physically heavy, expensive to store and guard, slow and costly to transport, and impractical for digital commerce. You cannot email a gold bar.

**Bitcoin** tries to copy gold's hard scarcity — fixing supply at 21 million, growing slower than gold's after each halving — while adding what gold lacks: instant, borderless, digital transfer, divisible to eight decimals, verifiable by anyone. The price it pays is the opposite of gold's: gold has millennia of trust and almost no volatility from monetary debate; Bitcoin has barely sixteen years, wild volatility, and depends on electricity and the internet continuing to function. It is, in a sense, an experiment in whether you can manufacture gold's monetary properties in software — with all the fragility and promise of a sixteen-year-old experiment.

The scorecard, then: Bitcoin wins on supply discipline and transferability, ties gold on having no issuer to debase it, and loses badly on track record. Which column you prefer depends on what you fear most — a state that can print and freeze (favor Bitcoin or gold) or a young, volatile, energy-hungry network that could still fail (favor fiat or gold).

#### Worked example: fixed supply versus inflating fiat over ten years

The store-of-value argument lives or dies on purchasing power over time, so let us model it crudely. Imagine two people in year 0, each setting aside savings worth \$10,000 of groceries.

- **The fiat saver** holds \$10,000 in cash (or a low-interest account). Suppose inflation averages 3% a year — historically ordinary. After ten years, prices have risen by a factor of 1.03^10, which is about 1.34. The same groceries that cost \$10,000 now cost ~\$13,440. The \$10,000 in cash still buys only what \$10,000 buys, so its *purchasing power* has fallen to 10,000 ÷ 13,440, about \$7,440 of year-0 groceries. The saver quietly lost roughly a quarter of their real wealth to inflation — money that, mechanically, was diluted as new money was created.
- **The fixed-supply holder** owns an asset whose quantity cannot be inflated. If that asset merely *kept pace* with the goods (rose 34% over the decade), they would preserve their \$10,000 of grocery-buying power. The catch — and it is enormous — is that Bitcoin does not move smoothly with inflation; it gyrates by tens of percent in a month. So the fixed supply protects against *debasement* (no one can dilute your share of 21 million) but does nothing to protect against *volatility* (the market price can still halve). You have swapped a slow, predictable erosion for a fast, unpredictable swing.

**The takeaway:** a fixed supply defends against the silent tax of inflation, but it is not the same as price stability — Bitcoin removes the debasement risk of fiat only by adding a volatility risk fiat does not have, and a careful person weighs which they can better tolerate.

## How the three functions actually score

We promised a verdict on money's three jobs. Here it is, illustrated in the stack figure below.

![Stack of the three functions of money scored for bitcoin](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-3.png)

**Store of value: strongest.** The fixed supply and unforgeable scarcity make Bitcoin a credible (if volatile) store of value, which is where most of its actual use and investment thesis sits today.

**Medium of exchange: weak in practice.** A network that settles a payment in ten minutes to an hour, charges variable fees that spike when it is busy, and whose price can swing 5% while you order coffee is poorly suited to everyday spending. Layer-2 systems like the Lightning Network make small, fast, cheap payments possible by settling off-chain, but adoption for daily purchases remains thin. Most people who hold Bitcoin do not spend it; they hold it.

**Unit of account: weakest.** Almost nothing is *priced* in bitcoin. Even merchants who accept it quote prices in dollars and convert at checkout, precisely because the value is too unstable to denominate a menu in. Until prices are stable enough to quote in BTC, it cannot truly be a unit of account.

The verdict matters because it dissolves a lot of noise: Bitcoin does not have to be good at all three jobs to be significant. It is mainly competing to be a *store of value* — digital gold — and its weakness as everyday cash, far from being a fatal flaw, is largely beside the point of that specific ambition.

## Common misconceptions

A handful of myths recur constantly. Clearing them up sharpens the real picture.

**"Bitcoin is anonymous."** It is *pseudonymous*, which is different and weaker. Every transaction is permanently public on the blockchain, tied to addresses rather than names. But addresses can often be linked to real identities — through exchanges that collect ID, through analysis of spending patterns, through reused addresses. Far from being a criminal's perfect tool, Bitcoin's permanent, public ledger has helped law enforcement trace and recover funds in numerous cases. Privacy on Bitcoin takes deliberate effort and is never guaranteed.

**"Bitcoin has no value because it is not backed by anything."** Neither is the dollar, since 1971 — fiat is "backed" only by trust and the requirement to pay taxes in it. Value comes from people accepting and demanding a thing, not from physical backing. Whether Bitcoin's value is *durable* is a fair debate; whether unbacked things can have value is not — all modern money is unbacked.

**"It is too late / I need to buy a whole coin."** Bitcoin divides into 100 million satoshis. You can own \$5 worth. The "whole coin" framing is a psychological artifact, not a technical limit. (This is not a nudge to buy — it is a correction of a factual error; price direction is unknowable and not the point here.)

**"Quantum computers / a hack will break it overnight."** Bitcoin's cryptography is strong against today's computers, and the system can upgrade its algorithms through coordinated changes if a real threat emerges, as it has navigated technical challenges before. A sudden, total break is possible in principle but is not the imminent certainty headlines imply. The likelier risks are mundane: lost keys, exchange failures, and human error.

**"Bitcoin and blockchain are the same as the thousands of other cryptocurrencies."** Bitcoin is one specific network with one specific, conservative, supply-capped design and the longest track record. The broader crypto universe — thousands of tokens, many with unlimited supply, central teams, or outright scams — is a different and far riskier landscape. Conflating "Bitcoin" with "crypto" is like conflating "the internet" with "every website." (For a related but very different design — a programmable blockchain — see [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money).)

**"Mining is just wasteful number-crunching for nothing."** The computation looks pointless in isolation, but it is doing real work: it is what makes rewriting the ledger economically impossible. The energy *is* the security budget. You can argue the price is too high; you cannot argue the work does nothing.

## How it shows up in real markets

Abstract mechanisms become vivid in real episodes. Here are five that each teach a specific lesson.

### 2010: the \$41 pizza that became a parable

On May 22, 2010, a programmer named Laszlo Hanyecz paid **10,000 BTC for two pizzas** — the first known real-world purchase with bitcoin. At the time, those coins were worth about \$41. The pizzas, at a later \$60,000 per coin, would represent roughly \$600 million. "Bitcoin Pizza Day" is now an annual joke, but it carries a real lesson: it was the moment Bitcoin first crossed from an abstract experiment into a thing that could buy a physical good — proof that the double-spend solution actually worked in the wild. It also illustrates, painfully, the store-of-value-versus-medium-of-exchange tension: spending an appreciating asset on pizza is the cautionary flip side of holding it.

### 2014: Mt. Gox and the custody lesson

Mt. Gox, based in Tokyo, once handled the majority of all Bitcoin trades. In February 2014 it abruptly halted withdrawals and filed for bankruptcy, having "lost" around **850,000 BTC** belonging to customers — worth hundreds of millions then, and tens of billions at later prices — to a mix of theft and mismanagement. Note the crucial distinction: *Bitcoin the protocol was never hacked.* The exchange holding customers' keys was. Every customer who left coins on Mt. Gox learned "not your keys, not your coins" the hard way. Mt. Gox is the founding case study in custodial risk, and its lesson was tragically re-taught in 2022 when the exchange FTX collapsed, vaporizing customer funds in a similar betrayal of trust.

### 2021: El Salvador makes it legal tender

In September 2021, El Salvador became the first country to make Bitcoin **legal tender** alongside the US dollar, rolling out a government wallet ("Chivo") and even buying bitcoin for its treasury. The experiment has been mixed: adoption for everyday payments stayed low, the IMF objected, and the treasury's holdings swung wildly with the price. In early 2025, under an IMF agreement, the country walked back the mandatory-acceptance aspects. It is the clearest real-world test of Bitcoin-as-national-money to date, and its ambiguous results show both the appeal (a small, dollarized economy seeking monetary independence) and the limits (volatility and thin everyday use) of the medium-of-exchange ambition.

### 2024: the US spot ETF and Wall Street's arrival

In January 2024, US regulators approved the first **spot Bitcoin ETFs** — exchange-traded funds that hold actual bitcoin and trade like a stock. This let ordinary investors and big institutions get Bitcoin exposure through a regular brokerage account, without touching keys or exchanges, and the funds absorbed tens of billions of dollars within months. It was a watershed: the asset that began as a rebellion against the financial establishment got formally adopted *by* the establishment. That is either vindication or co-option, depending on your view — and it introduces a new tension, since ETF holders gain convenience by giving up self-custody, re-creating the trusted-third-party model Bitcoin was built to avoid. (Wall Street's embrace of crypto more broadly is its own saga; for the dollar-shaped corner of crypto, see [stablecoins](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).)

### 2021: China bans mining, and the network shrugs

In mid-2021, China — then home to an estimated half or more of all Bitcoin mining — **banned the activity outright**. Hashrate dropped sharply almost overnight, and many predicted the network was crippled. Instead, miners packed up and relocated (heavily to the United States, Kazakhstan, and elsewhere), difficulty adjusted downward to keep blocks coming, and within about a year the global hashrate had fully recovered and surpassed its pre-ban peak. It was a real-world stress test of decentralization: the single largest concentration of mining was outlawed by a major power, and the network routed around the damage exactly as designed. No central point existed for China to actually shut down — only miners, who moved.

## When this matters to you / further reading

If you take one thing from this post, take the mental model in figure 1 and the before-after in your head: Bitcoin is an attempt to build *money with no keeper*, and every strange thing about it — the energy use, the ten-minute blocks, the irreversibility, the lost coins, the volatility — is a consequence of refusing to put anyone in charge.

![Before and after comparison of a trusted third party versus a trustless ledger](/imgs/blogs/bitcoin-and-the-cypherpunk-vision-6.png)

The before-after figure above is the cleanest summary of the whole bargain. On the left is the world we know: a trusted third party (your bank) keeps the ledger, which is convenient and recoverable but can freeze you, reverse you, debase the currency, or fail. On the right is Bitcoin's world: thousands of nodes keep identical copies, rules rather than a firm decide what is valid, and there is no off switch and no reversals — which is liberating if you value sovereignty and terrifying if you value a safety net. Neither side is strictly better; they are different trade-offs for different fears.

When does this matter to *you*, concretely? It matters if you live under a currency that is being inflated away or a regime that freezes dissidents' accounts — then censorship-resistant, fixed-supply money is not a toy but a lifeline, with the caveat that you must master self-custody or you have gained nothing. It matters if you are tempted to treat Bitcoin as a quick way to get rich — then the volatility worked examples are the warning: a 50% drawdown needs a 100% recovery, and the asset owes you nothing. And it matters as a piece of financial literacy regardless, because Bitcoin forced a generation to ask where money comes from, who controls it, and what we actually trust when we trust a dollar.

Where to go next. To understand the system Bitcoin was reacting against, read [how money is created by banks and central banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) and [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) — the two posts that explain the printable, controllable fiat world Bitcoin set out to escape. To see how the same blockchain idea was extended from money into programmable contracts, read [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money). And to see the strange hybrid where crypto rails carry actual dollars, read [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar). Each opens up one box in the picture you now hold. The picture is the thing to keep; the rest is detail you can fill in once you know where it goes.

A closing note on tone, because it is easy to slip into either cheerleading or dismissal. Bitcoin is neither a miracle nor a scam. It is a genuinely novel solution to a genuinely hard problem — money without a keeper — that comes bundled with genuine costs: volatility, energy, irreversibility, and the perpetual risk that an experiment this young still fails. Understanding it on those terms, mechanism by mechanism and trade-off by trade-off, is worth doing whether or not you ever own a single satoshi. And whatever you decide, decide it with the risks named beside the upsides — which is the only honest way to think about money that nobody is in charge of.

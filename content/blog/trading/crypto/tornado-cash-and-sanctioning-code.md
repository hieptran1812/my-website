---
title: "Tornado Cash and the Sanctioning of Code: When a Government Blacklists a Smart Contract"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How an unstoppable privacy tool on Ethereum, used by ordinary people and North Korean hackers alike, became the first piece of code the US government ever sanctioned, and why that collided privacy, free speech, and the limits of law."
tags: ["tornado-cash", "ofac", "sanctions", "ethereum", "privacy", "money-laundering", "smart-contracts", "lazarus-group", "free-speech", "crypto", "case-study"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Tornado Cash was immutable software that gave anyone on Ethereum financial privacy; it was used by ordinary people protecting their salaries and by North Korea's state hackers laundering hundreds of millions, and when the US sanctioned the code itself in 2022, it forced a first-of-its-kind collision between privacy, free speech, and the limits of enforcing law against software no one controls.
>
> - Every transaction on Ethereum is public forever. Anyone can see every dollar you ever sent or received. Tornado Cash existed to break that link and give you privacy back.
> - It worked as a "mixer": you deposit a fixed amount into a shared pool with thousands of other deposits, wait, then withdraw to a brand-new wallet. There is no on-chain link from your deposit to your withdrawal.
> - That privacy is dual-use. A dissident, a company protecting payroll, and North Korea's Lazarus Group hacking crews all used the exact same tool. Lazarus laundered hundreds of millions of stolen dollars through it.
> - In August 2022, the US Treasury's OFAC added the Tornado Cash smart-contract addresses themselves to the sanctions list. It was the first time a government had sanctioned code rather than a person or a company.
> - Developer Alexey Pertsev was arrested in the Netherlands and Roman Storm was charged in the US, igniting a fierce backlash: can you sanction math? Can you jail someone for publishing open-source software?
> - In 2024 a US appeals court ruled the immutable contracts were not "property" OFAC could lawfully designate, partly overturning the sanction and leaving a chilling question hanging over privacy developers everywhere.

The diagram above is the mental model for everything that follows: money goes into one big shared pool, and money comes out of that pool to a fresh address, and the two events have no traceable thread between them. That single missing thread is the whole story. It is what gave a journalist in an authoritarian country the ability to receive donations without exposing every supporter, and it is the same missing thread that let a North Korean hacking unit wash three-quarters of a billion dollars of stolen crypto. The US government decided the thread mattered enough to do something no government had done before: it did not sanction a person, or a bank, or a company. It sanctioned the software.

![How a crypto mixer breaks the link between a deposit and a withdrawal](/imgs/blogs/tornado-cash-and-sanctioning-code-1.png)

This is a case study about that decision and its fallout. It is interesting precisely because there are no clean heroes. The privacy tool was genuinely useful and genuinely abused. The government had a genuine national-security problem and reached for a genuinely novel and contested legal tool to solve it. The developers had genuinely written code that helped genuine criminals, and they had also genuinely not controlled, profited from, or operated that code once it was live. Almost every strong claim you will hear about Tornado Cash is partly true and partly misleading. The goal here is to build the whole thing from zero, so that by the end you can hold all the contradictions at once and reason about them yourself. We will not give you a verdict. We will give you the machinery to form one.

## First principles: the basics you need before any of this makes sense

Before we can talk about sanctioning code, we have to define every term the story turns on. If you have never touched crypto, do not skip this section. It is the foundation, and the rest is incomprehensible without it. I will gloss each term inline the first time it appears.

A **blockchain** is a shared public ledger: a database that nobody owns and everybody can read. Instead of a bank keeping your balance on a private server, thousands of computers around the world keep identical copies of the same ledger and agree, through a voting process called *consensus*, on which transactions are valid. **Ethereum** is the most widely used such ledger for general-purpose programs; **Bitcoin** is the original, narrower one for moving coins. (If you want the full ground-up tour of Ethereum, it has its own deep dive in [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money).)

The single most important fact for this entire story is this: **on a public blockchain, every transaction is visible to everyone, forever.** When you send 1 ETH (ether, Ethereum's native coin) from your wallet to mine, that transfer is written into the public ledger with both addresses, the amount, and the exact timestamp, and it stays there permanently for anyone on earth to read. There is no "private" mode. People often assume crypto is anonymous; the truth is closer to the opposite. It is *pseudonymous*. Your identity is not your name, it is your **address** (a long string like `0x3aF1...c9b2`), but every address's complete financial history is laid bare. The moment anyone links your real identity to one of your addresses, say because you withdrew to it from an exchange that knows your passport, your entire past and future activity on that address becomes an open book.

A **wallet** is the software that holds your **private keys**, the secret codes that authorize moving your coins. "Self-custody" means *you* hold those keys and nobody can freeze or seize your coins. An **address** (also called a public key, loosely) is the destination others send to. You can make as many addresses as you like, for free, instantly. That sounds like it should give you privacy: just use a fresh address each time. But it does not, because blockchain analysis firms (companies like Chainalysis and Elliptic that, for a fee, untangle the ledger for governments and businesses) follow the *flow*. If address A sends to address B, and B sends to C, the chain of custody is right there in the ledger. Fresh addresses do not hide you if the money visibly walked from your old one to your new one.

A **smart contract** is the piece that makes Ethereum special, and the central character of this whole story. It is a program that lives *on* the blockchain itself and runs automatically when someone calls it. It can hold coins, enforce rules, and pay people out with no human in the loop. Crucially, a smart contract has an address just like a person's wallet does, and it can be written to be **immutable**, meaning that once it is deployed, its code can never be changed, paused, or stopped by anyone, including its own author. We will return to immutability again and again, because it is the hinge on which the legal drama turns. Hold this thought: Tornado Cash was a set of immutable smart contracts. After a certain point, *no one on earth, including its creators, could turn it off, change it, or take money out of it.* It just ran, exactly as written, accepting deposits and honoring withdrawals, like a vending machine welded permanently to the wall.

### Fungibility: why your money's history can hurt you

Here is a subtle idea that turns out to be the whole reason mixers exist. **Fungibility** means one unit of money is interchangeable with any other unit. A \$20 bill in your pocket is worth exactly the same as any other \$20 bill; nobody asks where it has been. Cash is fungible. That property is so basic we never think about it.

Crypto, because of the public ledger, is *not perfectly fungible.* Each coin carries a visible, permanent history. If a particular batch of ETH passed through a wallet that once held stolen funds, that ETH is now "tainted." Exchanges and businesses, under pressure from regulators, may refuse to accept it, freeze it, or report it. So 1 ETH with a clean history can be worth more, in practical usability, than 1 ETH with a dirty one, even though the ledger says they are the same amount. This non-fungibility is a real problem: a person who innocently receives tainted coins can find their funds frozen through no fault of their own. Privacy tools restore fungibility by stripping the history. That is the legitimate case for them, and it is not a fig leaf; it is the same reason cash is fungible.

### Money laundering and the rules built to stop it

**Money laundering** is the process of taking money obtained illegally, such as the proceeds of theft, fraud, or sanctions evasion, and disguising its origin so it can be spent without raising alarms. The classic three stages are *placement* (getting the dirty money into the financial system), *layering* (moving it through many transactions to obscure the trail), and *integration* (bringing it back as seemingly clean funds). A crypto mixer is, mechanically, a layering tool. It takes coins with a known dirty origin and lets them come out the other side with no visible link to that origin.

To fight this, the traditional financial system runs on **AML** (anti-money-laundering) and **KYC** (know-your-customer) rules. Banks must verify who you are, monitor your transactions, and report suspicious activity. The entire system is built on the principle that a regulated intermediary, a bank, an exchange, a payment processor, stands between you and the money and can be compelled to police it. The deep problem a tool like Tornado Cash poses is that *there is no intermediary to compel.* The smart contract is not a company. It has no compliance department, no CEO to subpoena, no servers to seize. It is just code on a ledger that thousands of computers worldwide are running. That structural fact is what made the government reach for such an unusual response.

### OFAC, the SDN list, and what a sanction actually is

The **Office of Foreign Assets Control (OFAC)** is the arm of the US Treasury that administers economic sanctions. Its primary weapon is the **Specially Designated Nationals (SDN) list**, a public roster of people, companies, and entities that US persons are forbidden from transacting with. When OFAC "designates" someone by adding them to the SDN list, it becomes a federal crime for any US person or business to send them money, receive money from them, or provide them services. Banks must freeze any of their assets that touch the US system. The SDN list is how the US blocks terrorists, drug cartels, and hostile regimes from the dollar economy. (Sanctions as a tool of statecraft, and how the dollar's plumbing makes them bite, are the subject of [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments).)

For our entire history, every name on that list has been a *person* or an *entity*: a human, a corporation, a vessel, a government agency. The premise was always that there was *someone* on the other end, someone who could comply or be punished. The Tornado Cash designation broke that premise. OFAC added smart-contract *addresses*, lines of autonomous code, to the SDN list as if they were a person. That is the legal novelty at the heart of everything that follows, and we will return to exactly why it is so contested.

One last distinction to plant now, because it does enormous work later: **sanctioning a person versus sanctioning an address.** When you sanction a person, the target is a will, an actor who can choose to stop. When you sanction an address that belongs to immutable code, the "target" cannot choose anything. It cannot stop. It cannot comply. It will keep accepting deposits and honoring withdrawals forever, indifferent to the law, because that is the only thing it knows how to do. You have not disabled the tool. You have only made it a crime for Americans to touch it, while it keeps running for everyone else.

## What Tornado Cash actually was

Now we can describe the thing itself. Tornado Cash, launched in 2019, was a set of smart contracts on Ethereum that acted as a **mixer** (also called a *tumbler*): a service that takes in coins from many people, jumbles them together in a shared pool, and lets each person withdraw the same amount they put in, to a fresh address, with no on-chain link between the deposit and the withdrawal. The figure above traces that flow end to end. Its purpose was singular and stated plainly by its creators: to give Ethereum users financial privacy on a chain where, by default, they had none.

The genius and the danger were the same property. Let us build the mechanism precisely.

### Fixed denominations and the shared pool

A naive mixer that let you deposit any amount would be trivial to de-anonymize. If you put in 7.3194 ETH and someone later withdraws exactly 7.3194 ETH, the amounts match and the link is obvious. So Tornado Cash used **fixed denominations.** You could only deposit in standard sizes, for ETH the pools were 0.1, 1, 10, and 100 ETH, and there were pools for other tokens too. Everyone in the 10-ETH pool deposited exactly 10 ETH, so when someone withdraws exactly 10 ETH, it could have come from *any* of the thousands of deposits in that pool. The uniformity is the camouflage.

When you deposited, your wallet generated a secret random value and submitted only a cryptographic *commitment* to it, a one-way fingerprint, into the pool. You saved the secret (the "note") privately. The pool now held your 10 ETH alongside everyone else's, indistinguishable.

### The zero-knowledge proof that makes withdrawal anonymous

Here is the clever cryptographic core. To withdraw, you needed to prove to the contract two things at once: that you had deposited (so you are entitled to take 10 ETH out), and *without revealing which deposit was yours.* If you had to point at your specific deposit, the link would be re-established and the privacy lost.

This is solved with a **zero-knowledge proof (zk-proof)**, specifically a flavor called a zk-SNARK. A zero-knowledge proof is a piece of cryptography that lets you prove a statement is true while revealing nothing beyond the fact that it is true. Here, your wallet uses your saved secret note to generate a proof that effectively says: "I know a secret that matches one of the commitments in this pool, and that secret has not been used to withdraw before." The contract can verify this proof is valid, and therefore release 10 ETH to whatever fresh address you name, *without ever learning which deposit you are referring to.* The proof reveals only "yes, this person is entitled," never "this withdrawal corresponds to that deposit."

That is the missing thread. On the public ledger, an observer sees: 10 ETH went into the pool from address A at one time, and 10 ETH came out of the pool to address Z at another time. There is no recorded connection between A and Z. The connection exists only inside your private secret note, which never touches the chain.

#### Worked example: a mix that breaks the trail

Let us walk concrete numbers. Suppose you receive \$30,000 worth of ETH, at a price of \$3,000 per ETH that is 10 ETH, into address A, which an exchange knows belongs to you. You want to pay a contractor without the contractor (or anyone watching) being able to trace the payment back to your identity-linked address A.

Step 1: You deposit 10 ETH from address A into the Tornado Cash 10-ETH pool. The ledger publicly records "A deposited 10 ETH into the pool." Your wallet saves a secret note. Cost so far is just the network fee, say \$5 in **gas** (the fee Ethereum charges to process a transaction).

Step 2: You wait. Crucially, while you wait, *other people are depositing and withdrawing 10 ETH from the same pool.* Suppose over the next week, 200 other deposits and 180 withdrawals flow through the 10-ETH pool.

Step 3: You generate a zero-knowledge proof from your secret note and withdraw 10 ETH to a brand-new address Z that has no prior history and no link to you. The ledger publicly records "the pool paid 10 ETH to Z." It does *not* record that Z's withdrawal corresponds to A's deposit.

Step 4: You pay the contractor from Z. Anyone tracing the contractor's incoming \$30,000 sees it came from Z, a fresh address whose only prior event was a withdrawal from the mixer. To link it back to you, they would have to guess which of the ~200 deposits in the pool was yours.

The intuition: you put \$30,000 in and took \$30,000 out, and on the public ledger those two events float free of each other, hidden in a crowd.

There is one more wrinkle in step 3 that turns out to matter enormously, both for how the tool actually worked and for how widely the legal blast radius spread later. Your brand-new withdrawal address Z has, by design, never held a single coin. But on Ethereum every transaction, including the withdrawal itself, costs gas, and gas must be paid in ETH from the address making the call. So how does an empty address pay to withdraw without first being funded, which would re-create exactly the link you are trying to destroy? If you sent address Z a little gas money from your own wallet first, an observer would see "A's wallet funded Z, then Z withdrew from the pool," and the privacy would collapse before it began. Tornado Cash solved this with **relayers**, third parties who submit the withdrawal transaction for you and pay the gas, deducting their fee and the gas cost from the amount that comes out of the pool. We will return to relayers in detail when we trace who the sanction swept up, because that small piece of plumbing put a whole class of helpers into legal jeopardy.

## Privacy is dual-use: the ordinary and the criminal, side by side

The deposit-and-withdraw mechanism does not know or care who you are or why you want privacy. This is the crux of the entire controversy, so it deserves its own treatment.

![Matrix of legitimate versus illicit uses of a privacy tool](/imgs/blogs/tornado-cash-and-sanctioning-code-4.png)

The matrix above lays out the dual-use problem plainly. On the legitimate side: a human-rights worker receiving donations in a hostile country, who cannot afford to expose every donor's address to a regime that would punish them. A company that does not want competitors reading its entire payroll and supplier payments off the public ledger. An ordinary person who simply does not want every merchant, employer, and stranger to see their complete net worth and spending history the instant they learn one address. The Ethereum co-founder Vitalik Buterin publicly stated he had used Tornado Cash himself to anonymously donate to Ukraine, precisely so that the donations could not be misread as support for the Russian side in a war. These are not edge cases; they are the mainstream reason financial privacy exists, and they are the same reason you would not want your bank statement published.

On the illicit side: exactly the same mechanism is a money launderer's dream. And the heaviest illicit user, by a wide margin, was a state actor.

### The Lazarus Group

The **Lazarus Group** is a hacking organization widely attributed by US authorities to the government of North Korea, specifically its intelligence apparatus. North Korea, cut off from the world economy by heavy sanctions, has turned cybercrime into a national revenue stream, and stealing cryptocurrency is one of its largest lines of business. Lazarus has been blamed for some of the biggest crypto thefts in history: the \$625 million Ronin Bridge hack in March 2022 (the bridge behind the game Axie Infinity), the \$100 million Harmony Horizon Bridge hack, and many others. The US Treasury has stated these stolen funds help finance North Korea's weapons programs.

Stolen crypto is useless to a thief until it can be laundered into a usable form. The coins are tainted, traceable, and frozen by exchanges the moment they are recognized. So Lazarus needed a layering tool to break the trail, and Tornado Cash was the industrial-scale option. When OFAC designated Tornado Cash in August 2022, it stated the mixer had been used to launder more than \$7 billion in total since 2019, and that this included more than \$455 million stolen by the Lazarus Group, the proceeds of the Ronin hack among them.

It is worth dwelling on why a *state* was the heaviest abuser, because it sharpens the policy problem. An ordinary criminal launders to spend; a sanctioned state launders to *survive*. North Korea is largely walled off from the dollar banking system, so converting stolen crypto into spendable value is one of the few channels it has to fund its government and weapons programs. That makes it both highly motivated and highly sophisticated: Lazarus does not just dump funds into a mixer and hope, it chains together multiple mixers, cross-chain bridges (services that move value between different blockchains), and over-the-counter brokers in lax jurisdictions, layering the trail many times over. Tornado Cash was one stage in a longer pipeline. And here is the uncomfortable implication for the sanction: a state actor that needs to launder hundreds of millions of dollars to keep its regime running is exactly the actor *least* deterred by a US sanction it can simply ignore, because it was already operating entirely outside the reach of US law. The people for whom a sanction is a binding constraint are the ones inside the system, the US developers, the US relayers, the US users, not Pyongyang.

#### Worked example: Lazarus laundering stolen funds through the mixer

Let us make the layering concrete with the Ronin hack. Lazarus drained roughly \$625 million from the Ronin Bridge in March 2022, in the form of about 173,600 ETH and 25.5 million USDC (a dollar-pegged stablecoin). Every one of those coins sat in attacker-controlled addresses that were now publicly known and flagged worldwide.

Step 1: The attackers first swapped the USDC into ETH, because a mixer pool works on a single uniform asset; you cannot mix a stablecoin in an ETH pool. Suppose they converted the \$25.5 million of USDC into roughly 8,500 ETH at a price around \$3,000.

Step 2: Now holding on the order of 182,000 ETH, all tainted, they fed it into Tornado Cash in the fixed 100-ETH denomination. To launder, say, 100,000 ETH that way requires 1,000 separate 100-ETH deposits, each generating its own secret note.

Step 3: Over weeks, they withdrew to thousands of fresh addresses, each withdrawal an untraceable 100 ETH. From those fresh addresses, the now-delinked ETH could be moved to exchanges and cashed out, ideally exchanges with weak controls in friendly jurisdictions.

Of the \$625 million, US authorities and Chainalysis attributed at least \$455 million as having flowed through Tornado Cash. The intuition: the same fixed-denomination pool that hides a \$30,000 contractor payment hides a \$455 million theft just as well, because the pool does not, and cannot, ask why.

## The privacy set: why the size of the crowd is the anonymity

We need one more concept to reason precisely about mixers, because it explains both how strong the privacy is and how it can fail. The **privacy set** (or *anonymity set*) is the number of indistinguishable candidates your transaction could have been. The bigger the crowd you hide in, the harder it is to single you out.

![Layers showing anonymity rising with the size of the privacy set](/imgs/blogs/tornado-cash-and-sanctioning-code-6.png)

The stack above shows the relationship directly. The mechanism here is just counting. If only one other person has ever deposited into a pool when you withdraw, then your withdrawal must correspond to one of two deposits, so a guesser has a 50% chance of fingering your source on a coin flip. That is hardly privacy at all. If a hundred deposits sit in the pool, a blind guess is right only 1% of the time. If thousands are in the pool, the odds of correctly linking any given withdrawal to its deposit by chance fall below a tenth of a percent.

#### Worked example: how the privacy-set size determines anonymity

Suppose you deposit 1 ETH (worth, say, \$3,000) into a pool, and you want to know how well hidden your eventual withdrawal will be.

Case A, a near-empty pool: only 2 deposits have ever entered, yours and one stranger's, and 2 withdrawals occur. An analyst trying to link your withdrawal to your deposit has just two possibilities to choose between. Probability of a correct blind guess: 1 in 2, or 50%. Your \$3,000 is barely hidden.

Case B, a modest pool: 100 deposits and 100 withdrawals. A blind guess is right 1 in 100 times, or 1%.

Case C, a busy pool: 1,000 deposits and 1,000 withdrawals. A blind guess is right 1 in 1,000, or 0.1%.

But here is the catch that makes real-world mixing harder than the math suggests. The privacy set is only as large as the deposits that are *plausibly yours.* If you deposit and then withdraw five minutes later when almost no one else has used the pool in that window, timing alone shrinks your effective privacy set toward 1, no matter how many historical deposits exist. And if you withdraw an unusual *pattern*, say you deposited exactly 13 times and later 13 fresh addresses each receive exactly one withdrawal and then forward to a single address, the pattern itself re-links you even though no single transaction did. The intuition: the pool gives you a crowd to hide in, but you can still give yourself away by stepping out of the crowd at a conspicuous moment or in a conspicuous shape. This is exactly how blockchain-analysis firms partially de-anonymized even mixed Lazarus funds, by tracing patterns and timing rather than any single broken link.

## The mechanism dissected: a traceable transaction versus a mixed one

Step back and compare the two worlds directly, because the contrast is the entire point of the tool.

![A traceable direct transfer beside a mixed transfer with no on-chain link](/imgs/blogs/tornado-cash-and-sanctioning-code-2.png)

The before-and-after above puts them side by side. In the normal world, on the left, money walks a visible path: your known wallet sends to an intermediate wallet, which sends to the destination, and every hop is a public record. A blockchain-analysis firm can follow that path the way you would trace a line on a map, from source to destination, with certainty. This is why crypto is, for law enforcement, often *easier* to trace than cash: the ledger never forgets.

In the mixed world, on the right, the path is severed at the pool. Your wallet deposits into the pool; the destination receives from the pool; but the pool mingled thousands of deposits and the zero-knowledge proof revealed no link, so there is no recorded line connecting your deposit to your withdrawal. The map simply has a gap where the line should be. The analyst can see money entered the pool and money left the pool, but not which exit corresponds to which entrance.

### Why the code cannot be "shut down"

This is the fact that made the sanction so strange, and it follows directly from immutability. A normal company you can shut down: seize its servers, freeze its bank accounts, arrest its officers, revoke its licenses. Tornado Cash's core pool contracts were deployed to Ethereum and then made immutable, which on Ethereum is accomplished by *renouncing control*, the developers deliberately gave up the administrative keys that could have changed or paused the contracts. There was a front-end website to make the tool easy to use, and that website could be (and was) taken down, its domain seized and its hosting on services like GitHub removed. But the website was just a convenient skin. The actual contracts lived on the blockchain and could still be called directly by anyone with the technical know-how, or through copies of the interface that others immediately re-hosted.

To "shut down" the contracts, you would have to shut down Ethereum itself, which means convincing thousands of independent computers worldwide, most outside US jurisdiction, to stop running the network or to censor specific transactions. No one can do that. So the contracts kept running after the sanction, are running as you read this, and will run as long as Ethereum exists. The sanction did not, and could not, stop the code. It changed only what was *legal* for Americans to do near it.

## The event: when the US sanctioned the code

Now the central event, step by step.

![Timeline from the 2019 launch through the 2022 sanction to the 2024 court rulings](/imgs/blogs/tornado-cash-and-sanctioning-code-5.png)

The timeline above is the spine of the case. In **2019**, Tornado Cash launched. By **2020**, the core pool contracts were made immutable; control was renounced. For roughly three years, the tool ran openly, used by privacy-seekers and, increasingly, by criminals, as its anonymity set grew large enough to be genuinely useful for laundering serious money.

On **August 8, 2022**, OFAC added Tornado Cash to the SDN list. The crucial detail: the designation listed not just a website or an organization, but a set of *Ethereum smart-contract addresses* directly, the on-chain pool contracts themselves. OFAC's stated rationale was that Tornado Cash had laundered over \$7 billion since 2019, including the \$455 million-plus stolen by North Korea's Lazarus Group, and that despite public pressure the service had "repeatedly failed" to put in place controls to stop illicit use. By designating the addresses, OFAC made it a sanctions violation for any US person to send funds to, or receive funds from, those contracts.

### The arrests

Two days later, on **August 10, 2022**, Dutch authorities arrested **Alexey Pertsev**, one of the developers associated with Tornado Cash, in the Netherlands, charging him under Dutch money-laundering law for his role in building the tool. He was held for months in pretrial detention, and in **May 2024** a Dutch court convicted him of money laundering and sentenced him to over five years in prison, finding that he had created a tool that habitually laundered criminal proceeds and had done nothing to prevent it. He has appealed.

In the United States, prosecutors charged **Roman Storm** and **Roman Semenov**, co-founders, in **2023**, with conspiracy to commit money laundering, to operate an unlicensed money-transmitting business, and to violate US sanctions. Semenov was also added to the SDN list as an individual and remained at large; Storm was arrested in the US and faced trial. The charges treated the developers as having *operated* a money-transmitting business, a characterization that hinges on whether writing and deploying autonomous software counts as running a service.

### The backlash: can you sanction math?

The reaction from the crypto and civil-liberties world was immediate and fierce, and it crystallized into a few sharp questions.

The first was a free-speech argument. In the United States, courts have long held that **computer code is a form of expression** protected by the First Amendment, a principle established in the 1990s "Crypto Wars" (we will come back to those). If publishing code is protected speech, the argument went, then sanctioning a published, immutable program, and prosecuting people for writing it, looks like punishing speech. The advocacy group **Coin Center** and the **Electronic Frontier Foundation (EFF)** argued exactly this, warning that the precedent could chill anyone who writes privacy or security software.

The second was the "can you sanction math?" framing. A zero-knowledge proof is, at bottom, mathematics. The pool contract is a published algorithm. Sanctioning an autonomous, ownerless tool, critics argued, is categorically different from sanctioning a person who can comply or a company that can be shut down. You are not blocking a bad actor; you are declaring a piece of math illegal to touch, and in doing so you sweep up every innocent user, including people whose only crime was wanting their own salary kept private.

The third was the *collateral-damage* argument, which became vivid almost immediately. Because anyone can send ETH to any address on Ethereum, pranksters began "dusting" prominent people, sending tiny amounts of mixed ETH from Tornado Cash to the public wallets of celebrities and executives. The recipients had not asked for it and could not refuse it, yet by a literal reading of the sanction, receiving funds from a sanctioned address is a violation. The episode showed how poorly a tool designed for sanctioning a *person* fit a target that was *open infrastructure anyone could push funds through.*

## The reach of the sanction: who got pulled in

The novelty of sanctioning an address rather than a person is exactly that the blast radius is not contained to a single bad actor. It spreads outward to everyone who built, used, or merely touched the code.

![Graph of how sanctioning the code reaches developers, users, relayers, and protocols](/imgs/blogs/tornado-cash-and-sanctioning-code-3.png)

The graph above maps the reach. At the center is the sanctioned code. From it, exposure radiates to the *developers* who wrote it (charged criminally), to ordinary *users* whose funds, even legitimate ones, sitting in the pool were now legally radioactive to touch, to *relayers* (a class of helpers we will explain in a moment), and to *protocols* and businesses across DeFi that might receive mixed ETH and now had to decide whether accepting it was a compliance violation.

A **relayer** deserves a word, because it shows how deep the tangle goes. To withdraw from Tornado Cash to a brand-new address, that fresh address needs a little ETH to pay the gas fee, but if you fund the fresh address yourself, you create a link that defeats the whole purpose. Relayers solved this: a relayer would submit your withdrawal transaction on your behalf, pay the gas, and take a small fee out of the withdrawn amount, so your fresh address never needed any pre-existing funds. After the sanction, relayers were arguably providing a service to a sanctioned entity, and their legal status became fraught, even though many of them were just anonymous operators running a script for a few dollars of fees.

#### Worked example: the relayer's economics and exposure

Suppose you are withdrawing 10 ETH (worth \$30,000 at \$3,000 each) and you use a relayer rather than self-funding your fresh address.

Step 1: The relayer submits your zero-knowledge proof to the pool contract and pays the gas out of their own pocket. On a busy day, the gas for a Tornado Cash withdrawal might run \$15 to \$50 worth of ETH, depending on network congestion. Say it is \$30 today.

Step 2: The relayer charges a fee, commonly a fraction of a percent of the withdrawal plus the gas. At, say, 0.3% of 10 ETH that is 0.03 ETH, or \$90, on top of recovering the \$30 of gas.

Step 3: The contract pays out 10 ETH minus \$120 worth, so you receive roughly 9.96 ETH at your fresh address, and the \$120 covers the relayer's gas and fee. Your fresh address Z never had to be pre-funded, so no link to you was ever created.

Step 4: Now the exposure. The relayer netted about \$90 in profit on this transaction. After August 2022, US authorities took the position that relaying for a sanctioned contract could itself be a sanctions violation, carrying potential penalties in the millions. The intuition: a service whose entire margin was tens of dollars per transaction was suddenly, at least in theory, staring at a multimillion-dollar legal downside, an asymmetry so extreme that any rational US-based relayer simply shut down, which is precisely what happened.

#### Worked example: the compliance dilemma for a protocol touching sanctioned funds

Put yourself in the shoes of a US-based DeFi lending protocol, or really any business that holds crypto. Someone deposits 50 ETH (worth \$150,000 at \$3,000) into your protocol. Your compliance tooling flags that some of that ETH passed through Tornado Cash at some point in its history.

Step 1: You must now decide. The sanction forbids US persons from transacting with the Tornado Cash contracts. Does *receiving* ETH that *once touched* the mixer, possibly several hops and several owners ago, count as transacting with a sanctioned address? The law was not written with this fact pattern in mind, and the guidance was murky.

Step 2: Quantify the cost of getting it wrong. OFAC penalties for sanctions violations can run into the millions of dollars per violation, and in egregious cases far more, plus the reputational and banking fallout of being seen as a sanctions risk. Against that, the value at stake in this single deposit is \$150,000. The asymmetry is brutal: the downside of wrongly accepting tainted funds vastly outweighs the \$150,000.

Step 3: The rational, risk-averse move is to over-comply, to *screen and reject* any funds with mixer exposure, even funds many hops removed, even funds whose current owner is entirely innocent. So a person who legitimately used Tornado Cash years ago to protect their privacy, broke no law, and now wants to use a mainstream service can find themselves quietly debanked from DeFi, their clean money treated as radioactive because of where it once sat.

The intuition: when the cost of a false positive is a multimillion-dollar fine and the benefit of accepting any one user is small, every rational business over-blocks, and the burden falls on innocent users who can least afford it. This is the same dynamic that, in traditional banking, produces "de-risking," where banks drop entire categories of lawful customers rather than carry the compliance cost. (You can see the same overcautious screening logic across DeFi lending and trading venues, explored in [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).)

## A developer's legal exposure, quantified

The hardest question in the whole affair is the developers'. Did Pertsev and Storm commit a crime by writing and releasing software, or were they builders of a neutral tool that others misused? The legal theory against them treats them as operators of an unlicensed money-transmitting business and as conspirators in laundering. The defense theory treats them as authors of published, immutable code over which they had relinquished control. Let us reason through the exposure with concrete numbers, because the stakes were a human being's freedom.

#### Worked example: a developer's legal exposure

Consider a developer, call her Dev, who in 2020 wrote and deployed an immutable mixer contract and then renounced control. By 2022, criminals have run \$455 million of stolen funds through it. Prosecutors charge her.

Step 1: The money-transmitting charge. US law requires money-transmitting *businesses* to register and follow AML rules. The prosecution argues Dev ran such a business. But a money transmitter classically takes custody of funds and moves them on a customer's behalf. Dev's contract is immutable and ownerless; she never held the funds, could not move them, could not stop them, and earned no fee on the pool itself. Whether "deploying autonomous code" equals "operating a business" is the unsettled core of the case.

Step 2: The laundering charge. Money laundering generally requires intent to conceal the proceeds of a *specific* crime. Dev built a general-purpose privacy tool; she did not direct Lazarus to it, did not know which deposits were stolen, and could not have screened them even if she wanted to. The prosecution must show she knowingly facilitated laundering, not merely that her tool was used for it. A car maker is not a getaway-driver accomplice; the question is whether code is more like the car or more like the driver.

Step 3: The sanctions charge. Did Dev violate sanctions by maintaining a tool the government later designated? The designation came *after* she had renounced control, so by the time the contracts were sanctioned, there was nothing she could do to comply, no off switch to flip.

Step 4: Tally the exposure. Conspiracy to launder money carries up to 20 years in US federal prison; operating an unlicensed money-transmitting business carries up to 5; sanctions violations carry their own steep penalties. Stack the charges and Dev is facing decades. The Dutch court, applying its own law, gave Pertsev over 5 years. The intuition: the entire weight of a money-laundering and sanctions case can land on a person whose only act was publishing software they no longer controlled, and reasonable people, and courts in different countries, disagree on whether that is justice or overreach.

## The aftermath: chilling effects and the 2024 rulings

Two things happened after 2022 that matter most.

### The chilling effect

The prosecutions sent a tremor through open-source privacy development. If writing and publishing a privacy tool can land you in prison for what strangers later do with it, the rational developer thinks twice, or moves abroad, or stays anonymous, or simply does not build the tool. Privacy researchers warned that this would push privacy development underground or out of the US entirely, and would discourage exactly the kind of careful, accountable, daylight engineering that might build *better* tools, ones with optional compliance features, while doing nothing to deter the criminals who will use whatever exists. The deterrent, critics argued, lands on the law-abiding builder and misses the lawless user, because North Korea does not care about a US indictment.

The chilling effect is not abstract; it shows up as concrete choices. After the designation, several US-based privacy projects paused work, relocated, or removed code from public hosting. Developers who had contributed to Tornado Cash even tangentially, by, say, submitting a minor code improvement years earlier, worried about whether they too could be swept into a conspiracy theory. The open-source norm of building in the open, under your real name, signing your commits, became a liability rather than a badge of accountability. There is a bitter irony here: the engineers most likely to respond to a prosecution by adding compliance features, or by designing privacy tools that let honest users selectively prove their funds are clean, are exactly the law-abiding, jurisdiction-bound builders the prosecutions deterred. The ones who will keep shipping unrestricted mixers are anonymous developers in jurisdictions beyond US reach, who answer to no court and add no safeguards. A policy meant to reduce harmful privacy tooling may have shifted the supply toward the least accountable corner of it.

### The 2024 court rulings

Then the legal tide partly turned. In **November 2024**, the US Court of Appeals for the Fifth Circuit ruled in *Van Loon v. Department of the Treasury* that OFAC had **exceeded its statutory authority** in sanctioning the immutable Tornado Cash smart contracts. The reasoning was narrow but consequential: the sanctions statute lets OFAC block the "property" of a foreign person, and the court held that the **immutable smart contracts were not "property"** in the sense the statute means, because no one owns or controls them. An immutable contract cannot be owned, transferred, or excluded from, the way property can; it is just running code that anyone can use and no one can stop. Therefore, the court held, OFAC could not lawfully designate the contracts themselves.

In **March 2025**, following that ruling, the Treasury formally **removed Tornado Cash from the SDN list.** The criminal cases against the individual developers, which rest on different legal theories than the contract designation, continued on their own tracks; the appellate ruling about whether code is "property" did not by itself resolve whether a developer ran an unlicensed money-transmitting business. The result is a legal landscape that is clarified in one corner and unsettled in others: you probably cannot sanction the *immutable code itself* as property, but the question of a developer's personal liability for building it remains live.

## Common misconceptions

**"Crypto is anonymous, so a mixer just makes it more anonymous."** Backwards. Crypto is *pseudonymous and fully public*; by default it is one of the most traceable forms of value transfer that exists, because the ledger is permanent and open. A mixer is not adding a little extra anonymity to an already-anonymous system; it is the one tool that actually breaks the default total traceability. That is why it is both genuinely valuable and genuinely dangerous.

**"Sanctioning Tornado Cash shut it down."** No. The core contracts were immutable and ownerless; they kept running through the sanction and run today. What was taken down were conveniences, the website, the hosted front-end, and what was changed was the *legality* of US persons touching the contracts. You cannot turn off code that no one controls without turning off the entire blockchain it lives on.

**"Only criminals use mixers, so banning them costs nothing."** The same mechanism that launders stolen funds also restores the basic fungibility and privacy that cash gives everyone. Dissidents, donors, companies protecting payroll, and ordinary people who simply do not want their entire financial life public all used it. The dual-use nature is the whole dilemma; pretending the legitimate side does not exist makes the policy question look easier than it is.

**"The developers operated the service and took the money, so they are obviously guilty."** The contracts charged no fee to the developers and, once control was renounced, could not be operated, paused, or drained by anyone. The legal fight is precisely over whether *writing and deploying autonomous code* is the same as *running a business*, and a US appeals court has already held that the immutable code is not even "property" the government can sanction. "Obviously guilty" assumes away the hardest question in the case.

**"Receiving mixed funds means you broke the law."** Because anyone can push ETH to any address, people received unsolicited "dust" from the sanctioned contracts they never asked for and could not refuse. A sanctions regime built for blocking deliberate transactions with a *person* fits badly onto open infrastructure that anyone can shove value through. Intent and control, not mere contact, are what a fair rule has to turn on, and that mismatch was one of the loudest criticisms.

**"The 2024 ruling means privacy tools are now safe to build in the US."** Only partly. The Fifth Circuit ruled narrowly that *immutable smart contracts are not property OFAC can designate.* It did not bless mixers, did not resolve the developers' criminal liability under money-transmitting and laundering law, and did not address tools that retain an admin key (which arguably *are* controllable property). The chilling effect on developers, who still face the prospect of personal prosecution, was eased but not erased.

## How it echoes in other markets and fights

Tornado Cash is the newest chapter in a fight as old as cryptography itself: the tension between an individual's ability to keep something private and a state's demand to be able to see and control it.

**The 1990s "Crypto Wars."** In the early 1990s, the US government classified strong encryption as a *munition* and tried to restrict its export, the idea being that uncrackable codes in civilian hands were a national-security threat. When the programmer Phil Zimmermann released **PGP** (Pretty Good Privacy), free email encryption for ordinary people, he was placed under federal investigation for "exporting munitions." Activists fought back partly by *printing the source code in a book* and arguing that banning its export was banning a book, and that **code is protected speech.** Courts ultimately agreed, and the government backed down; strong encryption became standard, and it is why your messages and bank logins are secure today. The Tornado Cash free-speech defense draws a direct line from that precedent: if code is speech, sanctioning code is censoring speech. The cypherpunk movement that drove those battles, and that later gave rise to Bitcoin, is its own story in [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision).

**The Apple-FBI fight (2016).** After a mass shooting in San Bernardino, the FBI demanded Apple write custom software to unlock the shooter's iPhone. Apple refused, arguing that building a tool to break its own encryption would endanger every user and that the government could not *compel* it to write code. The standoff ended when the FBI found another way in, but it left the same question unresolved: can the state force the creation, or forbid the existence, of software that frustrates surveillance? Tornado Cash is the same question wearing a financial costume.

**Sci-Hub and the limits of takedowns.** Sci-Hub, a site offering pirated access to paywalled academic papers, has been sued, blocked, and ordered shut in many jurisdictions, yet it persists because it is mirrored and distributed beyond any single authority's reach. It is a non-crypto illustration of the same hard truth the Tornado Cash sanction ran into: *you cannot turn off information or code that has escaped central control,* you can only make it illegal in your jurisdiction, which changes who uses it and how, but not whether it exists.

**The broader privacy-versus-surveillance debate.** Every few years the same battle reappears in a new form: end-to-end encryption versus law-enforcement "lawful access," cash versus a fully traceable central-bank digital currency, anonymous browsing versus mandatory identity. The structure is always the same. Privacy tools protect the innocent and shelter the guilty with the same mechanism, and any rule strong enough to stop the guilty also strips the innocent. Tornado Cash is simply the sharpest, most literal version yet: a government writing a line of code, an Ethereum address, onto a list meant for terrorists and cartels, and the courts having to decide whether a piece of math can be an outlaw.

## On-chain privacy after Tornado Cash

The demand for financial privacy did not vanish with the sanction; it cannot, because it is rooted in the permanent publicness of the ledger. What changed is the landscape of tools.

![Tree of on-chain privacy tools: mixers, privacy coins, and zero-knowledge systems](/imgs/blogs/tornado-cash-and-sanctioning-code-7.png)

The tree above sketches the family. Tornado Cash sits in the **mixer** branch, pooling identical deposits to break links; other mixers like Wasabi worked similarly on Bitcoin. A second branch is **privacy coins** such as Monero and Zcash, where privacy is built into the chain's own design, Monero hides senders, amounts, and recipients by default using ring signatures and stealth addresses, rather than bolted on top of a transparent chain like Ethereum. A third, fast-growing branch is **zero-knowledge systems** like Aztec and Railgun, which use the same zk-proof cryptography Tornado Cash relied on but bake in optional features, such as the ability for a user to *selectively prove* their funds are clean to a counterparty, or to comply with screening, an attempt to thread the needle between privacy and accountability that pure mixers never tried. The post-Tornado design conversation is largely about whether privacy tools can be built so that the law-abiding can prove their innocence without surrendering everyone's privacy, the very compliance affordance the original tool lacked.

## When this matters to you, and further reading

You may never use a mixer. But the Tornado Cash case matters to anyone who cares about three things, and at least one of them is probably yours.

If you care about **financial privacy**, this is the clearest test yet of whether you can have any on a public ledger without becoming a criminal suspect. The default of total transparency is not neutral; it means your employer, your landlord, your exchange, and any stranger who learns one of your addresses can read your complete financial life. The tools that fix that are now legally fraught, and where the line falls affects whether ordinary privacy is a right or a red flag.

If you care about **free speech and open-source software**, the prosecutions ask whether you can be punished for publishing code based on what strangers later do with it. That principle reaches far beyond crypto: to security researchers, encryption authors, and anyone who builds tools that can be used for good or ill, which is to say nearly all powerful software.

If you care about **how states fight crime in a borderless digital world**, Tornado Cash shows both the reach and the limits of the old tools. Sanctions are devastatingly effective against people and companies that need the dollar system. They are nearly powerless against autonomous, ownerless code that does not need anything from anyone. North Korea kept stealing and laundering; the people deterred were law-abiding developers. Whether that is a price worth paying, or a sign the tool was wrong for the target, is exactly the judgment the case forces.

For the adjacent pieces of this world: the chain it all ran on is in [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money); the broader system of on-chain finance the sanction rippled through is in [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao); the statecraft of cutting actors out of the financial system is in [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments); and the movement that birthed both crypto and the conviction that code is speech is in [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision).

The lasting image is the one we started with: money flowing into a pool no one controls, and money flowing out the other side with the thread between them gone. A government decided that missing thread was dangerous enough to write a line of code onto a list built for cartels and terror financiers. A court decided you cannot do that to code that nobody owns. And the code, indifferent to both, keeps running. That is the strange, unresolved frontier where privacy, free speech, and the law now meet, and it is not going to stop being contested any time soon.

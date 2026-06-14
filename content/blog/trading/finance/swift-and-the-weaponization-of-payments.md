---
title: "SWIFT and the Weaponization of Payments: How a Messaging Network Became a Weapon"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into how cross-border payments really work, why SWIFT moves no money yet gates almost everything, and how controlling dollar clearing in New York became a tool of statecraft."
tags: ["swift", "payments", "correspondent-banking", "sanctions", "dollar-clearing", "geopolitics", "cips", "macro", "finance"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — SWIFT does not move money; it only sends the secure messages that tell banks to move money, yet because almost every cross-border payment rides on it and on dollar clearing in New York, controlling that plumbing has become one of the most powerful tools of modern statecraft.
>
> - A cross-border payment is not one wire from you to the recipient. It is a relay of debits and credits across a chain of correspondent banks, and SWIFT is just the messaging layer that coordinates the relay.
> - The real chokepoint is not SWIFT at all. It is dollar clearing: almost every dollar leg of every international payment eventually passes through a bank account in New York, where US law reaches.
> - "De-SWIFTing" a bank makes it hard to reach. Cutting its access to dollar clearing makes it financially dead. The 2022 sanctions on Russia did both, and additionally froze roughly \$300 billion of its central bank's reserves.
> - The weapon works because of network effects, not because anyone designed it as a weapon. SWIFT is a member-owned cooperative in Belgium; its power comes from the fact that nearly everyone uses it.
> - Rivals exist (China's CIPS, Russia's SPFS, stablecoins), but none of them solves the core problem: they still cannot give a sanctioned party clean access to the dollar.
> - The cost is fragmentation. Every time the dollar rails are used as a weapon, more countries build escape hatches, and the long-run risk is a splintered payment world.

Here is a number that should not make sense. In 2022, the United States and its allies froze roughly **\$300 billion** of another country's central bank reserves — money that physically sat as electronic balances and bonds inside Western financial institutions — and within days a G20 economy with a \$1.8 trillion GDP found a large slice of its rainy-day fund simply unusable. No tanks crossed a border to seize it. No vault was cracked. The freeze happened through accounting entries and messages. How can sending the right messages to the right banks be as decisive as a blockade?

The diagram above is the mental model: a "cross-border payment" is not a single pipe from a sender to a receiver. It is a relay, and at each handoff a bank debits one account and credits another while a message coordinates the whole chain. The messaging is SWIFT. The money lives in correspondent accounts, and the dollar ones overwhelmingly clear through New York. Control the messages and you can isolate a bank; control the dollar clearing and you can strangle it. This post builds that picture from zero, then shows how it became a weapon.

![A million-dollar cross-border payment relayed through correspondent banks and New York clearing](/imgs/blogs/swift-and-the-weaponization-of-payments-1.png)

## How a cross-border payment actually works

Start with the thing everyone gets wrong. When you send \$1,000,000 from a bank in Tokyo to a supplier in Sao Paulo, no money flies across the Pacific. Banks do not stuff cash into a tube. What actually happens is a sequence of bookkeeping entries at several banks, glued together by instructions.

Let me define the vocabulary as we go, because the whole subject turns on three or four plain words.

A **bank account** is just a promise: when your balance reads \$1,000,000, your bank owes you that amount. The number is a liability on the bank's books and an asset on yours. Moving money "between accounts at the same bank" is trivial — the bank lowers one number and raises another, and nothing leaves the building. The hard part is moving money *between different banks*, especially banks in different countries that have never dealt with each other.

For two banks in the same country, a shared utility solves it. In the United States that utility is the **Federal Reserve** (the Fed), the country's central bank. Every American bank holds an account at the Fed, called a **reserve account**, holding **central-bank money** — the most final, riskless form of dollars there is. When Bank X pays Bank Y, the Fed lowers Bank X's reserve balance and raises Bank Y's. That transfer is **settlement**: the moment a payment becomes final and irreversible because real central-bank money has changed hands.

Across borders there is no shared central bank. Brazil's central bank does not give Japanese banks accounts, and the Fed does not give Brazilian banks accounts directly. So banks improvise a private version of the same trick, and that improvisation is the entire foundation of international finance.

### Correspondent banking

When Bank A in Japan needs to make dollar payments but has no account at the Fed, it opens an account at a US bank that *does* — say a large American or international bank in New York. That US bank becomes Bank A's **correspondent bank**: a bank that holds an account for another bank and makes and receives payments on its behalf.

Now two pieces of jargon that sound exotic but mean one ordinary thing seen from two sides.

A **nostro account** (from Latin *nostro*, "ours") is the account *I, the foreign bank, hold at you, the correspondent.* From Bank A's point of view, the dollar account it keeps in New York is its nostro account — "our money, parked over there."

A **vostro account** (from *vostro*, "yours") is the same account seen from the correspondent's side — "your money, sitting here with us." One account, two names depending on who is talking. Banks track these meticulously, because the whole system is a web of "you owe me, I owe you" balances held across borders.

So when the Tokyo payer sends \$1,000,000 to Sao Paulo, the relay looks like this. Bank A in Japan debits the payer's yen-or-dollar account. Bank A then needs dollars to land in Brazil, so it instructs its New York correspondent to move \$1,000,000 from its nostro account to the correspondent bank that serves Bank B in Brazil. The dollars clear in New York between the two correspondents. Bank B in Brazil, seeing its own New York nostro account credited, then pays out the local equivalent to the supplier. Money never left the US dollar system; it just hopped between accounts inside it, and the parties in Tokyo and Sao Paulo experienced "an international payment."

#### Worked example: a \$1,000,000 payment, hop by hop

Walk the \$1,000,000 through every leg and watch the fees and the balances move. The figure at the top of this post is exactly this chain.

1. **Tokyo, t=0.** The payer instructs Bank A to send \$1,000,000 to the Brazilian supplier. Bank A debits the payer \$1,000,000 plus a wire fee — say \$30. The payer's account drops by \$1,000,030.
2. **Bank A to its New York correspondent.** Bank A holds, say, \$50,000,000 in its nostro account at Correspondent A in New York. It sends an instruction (the SWIFT message) telling Correspondent A to pay \$1,000,000 to Correspondent B, the bank that serves Bank B in Brazil. Correspondent A debits Bank A's nostro account: it now reads \$49,000,000.
3. **Dollar clearing in New York.** Correspondent A pays Correspondent B \$1,000,000. If both are direct members of the New York clearing system, the \$1,000,000 moves between them there. Correspondent A might deduct a small clearing/handling fee — say \$15 — so \$999,985 lands.
4. **Bank B's nostro is credited.** Correspondent B credits Bank B's nostro account by \$999,985. Bank B in Brazil now "has" the dollars in New York.
5. **Sao Paulo, payout.** Bank B credits the supplier the local-currency equivalent of \$999,985, minus its own \$10 handling fee, so the supplier effectively receives value for about **\$999,975** of the original \$1,000,000.

The \$25 of handling that evaporated and the \$30 fee in Tokyo are the friction of the relay — small here, but they balloon when more intermediaries sit in the chain. The intuition: an "international payment" is a chain of domestic bookkeeping entries, and the dollars themselves never leave the New York plumbing.

### The dollar's gravity

Notice what happened. A payment between Japan and Brazil — two countries that are not the United States — settled in **US dollars**, inside the United States. This is not an accident of my example; it is the default state of the world. The dollar is the dominant **reserve currency** (the currency central banks and banks hold for international use), the dominant **invoicing currency** (the currency in which cross-border trade, especially commodities, is priced), and the dominant settlement currency. A large majority of global trade finance and a very large share of cross-border payments touch the dollar somewhere.

That gravity is the deep reason payments can be weaponized, and it connects to a longer story about why the dollar sits at the center of everything — see [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) and the [petrodollar system](/blog/trading/finance/petrodollar-and-dollar-dominance). For now, just hold the fact: if you must touch the dollar, you must eventually touch New York, and New York is inside US legal jurisdiction.

### Why the relay has so many hops

It is worth pausing on *why* a payment passes through several banks rather than going straight from Bank A to Bank B. The answer is that no single bank has accounts with every other bank on Earth. There are tens of thousands of banks; if each had to maintain a nostro account at every other one, the number of relationships would explode into the hundreds of millions, and each account would have to be funded and monitored. That is impossible, so the system organizes itself into a *hierarchy*. A handful of very large global banks act as hubs: they hold accounts for thousands of smaller "respondent" banks, and they hold accounts with each other. A small bank in one country reaches a small bank in another by routing through one or two of these hubs.

This hub structure is why dollar payments concentrate in New York. The hubs that hold the largest dollar correspondent networks are the giant US and international banks that clear directly through CHIPS and hold reserves at the Fed. A small bank almost anywhere reaches the dollar system by maintaining a nostro account at one of perhaps a dozen of these hubs. So the relay has many hops *and* funnels through a narrow waist — the same property that makes the system efficient also makes it a chokepoint, because to cut a target you do not have to reach every bank; you only have to reach the handful of hubs, and they are all in US jurisdiction.

#### Worked example: the cost of an extra hop

Count what each additional intermediary costs. Suppose a small bank in a frontier market cannot reach a New York hub directly and must route through a regional bank first.

- **Two-hop route (direct to a hub):** the small bank holds a nostro account at a New York hub; a \$1,000,000 payment clears with one correspondent fee of about \$25 and a 1-day settlement.
- **Three-hop route (through a regional intermediary):** the small bank routes through a regional bank, which routes to the New York hub. Now there are two correspondents in the chain. Each takes a fee — say \$25 plus \$20 — and each adds a settlement step, so \$45 of fees and 2 days. On a \$1,000,000 payment that is small in percentage terms, but the regional bank also takes a foreign-exchange spread if a currency conversion is involved, easily another 0.3% (\$3,000).
- **Four-or-more hops (the de-risked frontier):** when global hubs withdraw from a risky region (de-risking, below), the remaining chain can stretch to four or five intermediaries, each adding fees, spreads, and a day of delay. A \$1,000,000 payment that cost \$25 to a major economy can cost several thousand dollars and a week to a marginalized one.

The intuition: every hop in the relay adds a fee, a spread, and a delay, which is why losing a direct connection to the dollar hubs — voluntarily through de-risking or forcibly through sanctions — quietly taxes everything that flows through the longer chain.

## The three layers of a payment rail

People say "SWIFT" the way they say "Google" — as a stand-in for a whole activity. To see where the power actually lives, separate a payment rail into its three distinct layers, because SWIFT occupies only one of them.

![The three stacked layers of a payment rail: messaging on top, clearing in the middle, settlement at the bottom](/imgs/blogs/swift-and-the-weaponization-of-payments-2.png)

**Messaging** is the top layer: the instructions. "Pay \$1,000,000 from this account to that account, reference invoice 4471." A message moves no money. It is a structured note that tells banks what to do, like an email that says "please transfer the funds," except standardized, authenticated, and trusted.

**Clearing** is the middle layer: the process of taking many such instructions, matching them up, and netting them. If Correspondent A owes Correspondent B \$1,000,000 from your payment, and B owes A \$700,000 from other payments the same day, clearing nets the two so that only \$300,000 needs to actually move. Clearing computes *who owes whom, net*.

**Settlement** is the bottom layer: the irreversible transfer of real money to discharge the netted obligation. Only here does value actually change hands, and for dollars it happens in central-bank money at the Fed.

SWIFT lives entirely in the top layer. It is messaging. It is, almost literally, the postal service for banks — and like a postal service, it carries the letters but never the gold the letters are about. The clearing and settlement layers, for dollars, are American institutions. This separation is the single most important idea in the whole subject, so let me make it concrete.

#### Worked example: separating messaging from money

Suppose Bank A sends a SWIFT message instructing a \$1,000,000 payment, but its New York nostro account holds only \$200,000. What happens?

The message is delivered perfectly — SWIFT did its job; it transmitted a valid, authenticated instruction. But the payment fails, because at the clearing/settlement layer there is not enough money. Correspondent A will reject or hold the instruction: you cannot pay \$1,000,000 out of a \$200,000 balance.

Flip it. Suppose Bank A has \$50,000,000 in its nostro account and plenty of dollars, but it is removed from SWIFT. Can it still pay? In principle yes — the money exists in the clearing layer. But it now has no standard, automated, authenticated way to *instruct* the payment. It must fall back to telephone, telex, email, or another messaging network, each slower and more error-prone, and each correspondent must be willing to act on those non-standard instructions.

The intuition: messaging and money are separable, and the weapon's bite depends entirely on which layer you attack. Cutting the message is an inconvenience; cutting the money is a kill.

### What "SWIFT" is, precisely

SWIFT stands for the **Society for Worldwide Interbank Financial Telecommunication**. The name is a perfect description: it is a *society* — a member-owned cooperative — that provides *telecommunication* for banks. It was founded in 1973 by a group of banks frustrated with the previous standard, telex, which was slow and prone to error and fraud. SWIFT went live in 1977 and replaced the chaos of free-text telex with standardized, numbered message types.

Today SWIFT connects more than **11,000** financial institutions across more than 200 countries and carries on the order of tens of millions of messages a day. The classic format is the **MT** family (Message Type): an MT103 is a customer payment instruction, an MT202 is a bank-to-bank transfer, an MT799 is a free-format message, and so on. The industry is migrating to a richer XML-based standard called **ISO 20022**, which carries far more structured data per message, but the principle is unchanged — these are instructions, not money.

Crucially, **SWIFT is headquartered in Belgium**, is incorporated under Belgian law, and is overseen by a committee of central banks led by the National Bank of Belgium with the major central banks (including the Fed and the European Central Bank) participating. It is governed by a board drawn from its member banks. It is, on paper, a neutral European utility. This matters for the weaponization story: the United States cannot simply order SWIFT to cut someone off, because SWIFT is not American. The 2012 and 2022 disconnections happened through *European* legal action, with the US applying pressure. Hold that thread; we will pull it later.

### What a SWIFT message contains, and why it is trusted

A SWIFT message is not magic; it is a structured, authenticated text record. An MT103 — the workhorse customer-payment instruction — carries fields with fixed tags: the ordering customer, the beneficiary, the amount and currency, the value date, the chain of correspondents to use, fee instructions, and a free-text remittance reference. Because the fields are standardized and machine-readable, a receiving bank's software can parse and act on the instruction automatically, which is what makes payments fast and cheap. The earlier telex system was free text, so a human had to read and interpret every payment, and mistakes and fraud were rampant.

The reason banks *trust* a SWIFT message — act on an instruction worth millions without phoning to confirm — is authentication. Historically banks exchanged secret keys so each could verify that a message genuinely came from the claimed sender; today the network uses public-key infrastructure and a closed, secured network. SWIFT's value is not the wires; it is the shared standard plus the trust that the message is authentic and the sender is who it claims to be. That trust is also what makes removal from SWIFT painful: a de-SWIFTed bank can send an email or a telex, but the receiving bank has no standardized, trusted way to verify and auto-process it, so every payment reverts to slow, manual, suspicious handling.

The migration to **ISO 20022** matters here too. ISO 20022 is a richer XML data standard that carries far more structured information per message — full names and addresses, structured purpose codes, granular remittance data. For payments this improves automation and, not incidentally, compliance: more structured data makes it easier to screen every payment against sanctions lists automatically. The richer the data, the more precisely the rails can be policed, which quietly strengthens the weapon even as it improves ordinary service.

### The galaxy of message types

The MT family is large, and a few types recur in this story. An MT103 is a single customer credit transfer — the everyday "pay this person" instruction. An MT202 is a general financial-institution transfer, used by banks to move funds between themselves, including the cover payments that fund the dollar leg of an MT103. An MT202 COV is a variant introduced after regulators worried that "cover payments" were obscuring who the real originator and beneficiary were — a reform driven precisely by anti-money-laundering and sanctions-screening concerns. Even the message formats, in other words, have been shaped by the tension between frictionless payment and the desire to police it.

## The dollar-clearing chokepoint

Now to the part that journalists usually miss. The headline weapon is "cutting a country off from SWIFT." The real weapon is cutting it off from **dollar clearing** — and you can do the second without touching SWIFT at all.

![A taxonomy tree of the global payment infrastructure split into messaging, clearing, and settlement layers](/imgs/blogs/swift-and-the-weaponization-of-payments-7.png)

For dollars, clearing happens primarily through two systems. **Fedwire** is the Fed's own real-time settlement system, where the largest banks move central-bank money to each other instantly and finally; it settles trillions of dollars a day. **CHIPS** — the Clearing House Interbank Payments System — is a privately owned network of a few dozen large banks in New York that nets the day's dollar payments among themselves and settles the net through Fedwire. The vast majority of large-value cross-border dollar payments clear through CHIPS, which handles on the order of **\$1.8 trillion** a day. Both live in New York; both are reachable by US law.

Here is the chokepoint stated plainly: **almost every dollar in the world ultimately sits as a balance at a US bank or at the Fed.** When Bank A in Japan "has dollars," what it really has is a credit balance in a nostro account at a New York correspondent, which in turn holds reserves at the Fed. The dollar is not a physical thing Bank A can spirit away to Tokyo; it is a claim that only has meaning inside the US banking system. To use those dollars, Bank A must instruct a US institution to act, and that institution is bound by US law.

So if the US Treasury designates a bank as sanctioned, US banks are legally forbidden from dealing with it. Every US correspondent must close that bank's nostro account or freeze it. The bank's dollars, which only existed as a balance in New York, become unusable. It does not matter whether the bank is still on SWIFT — it can send all the messages it likes; no US institution will act on them, because acting would be a crime.

This is why the dollar-clearing cutoff is the real weapon and de-SWIFTing is the loud, visible *symbol* of it. Removing a bank from SWIFT is a coordination announcement; cutting its dollar clearing is the financial death sentence.

### How CHIPS netting actually works

To feel the chokepoint, it helps to see what CHIPS does mechanically, because "clearing" sounds abstract until you watch the numbers. CHIPS is a *netting* system. Throughout the day, its few dozen member banks send each other thousands of dollar payment instructions. Rather than settling each one with a separate transfer of central-bank money — which would require enormous intraday liquidity — CHIPS continuously matches payments and offsets them.

Suppose over a morning Bank P owes Bank Q a total of \$5,000,000,000 across many payments, and Bank Q owes Bank P \$4,800,000,000 across others. Settling each payment individually would require \$9,800,000,000 of money to move. Netting collapses that to a single \$200,000,000 obligation from P to Q. CHIPS does this across all members simultaneously, so a day with, say, \$1,800,000,000,000 of gross payments might settle with only a small fraction of that in actual central-bank money moving through Fedwire at the end. This is why CHIPS is so efficient — and why it is so concentrated. Netting only works among a tight club of banks that trust each other and clear directly; the smaller the club, the easier it is to police, and every member is a US-regulated entity. The efficiency that makes dollar clearing cheap is the same property that makes it a narrow, controllable waist.

#### Worked example: netting compresses \$10 billion into \$200 million

Make the netting concrete and see why it concentrates power. Two CHIPS members exchange payments for clients all day.

- Gross payments Bank P to Bank Q: \$5,000,000,000.
- Gross payments Bank Q to Bank P: \$4,800,000,000.
- Without netting: \$9,800,000,000 of central-bank money would have to move, an impossible liquidity demand for an intraday window.
- With netting: only \$200,000,000 net moves from P to Q, settled finally through Fedwire. The system processed nearly \$10 billion of value while moving only \$200 million of actual money — a 98% compression.

The intuition: dollar clearing works by letting a small, trusted, US-regulated club net their mutual claims, which is wildly efficient and, precisely because the club is small and American, exactly where a government can reach in to cut a target off.

#### Worked example: clearing \$X through New York

Trace what it means for a bank to "clear dollars" and what cutting it does. Suppose Bank S processes \$4 billion of dollar payments a month for its clients — importers, exporters, travelers. Every one of those payments needs a dollar leg that clears in New York.

- In a normal month, Bank S's New York correspondent processes that \$4 billion in and out, charging perhaps \$0.0001 per dollar in spreads and fees — call it \$400,000 of revenue for the correspondent and a service Bank S resells to its clients.
- Now the US Treasury sanctions Bank S. On day one, the correspondent must freeze Bank S's nostro account. Say it holds \$120,000,000 at that moment. That \$120 million is now stranded: Bank S cannot withdraw it (there is nowhere for dollars to "go" outside the US system) and cannot spend it (no US bank may act on its instructions).
- The \$4 billion of monthly client flow simply stops. Importers who needed dollars to pay foreign suppliers cannot get them through Bank S. They must find an unsanctioned bank, pay higher fees, or fail to pay at all.
- Bank S can still send SWIFT messages. They are now worthless for dollar payments, because the instruction has no one willing to execute it.

The intuition: "access to dollar clearing" is access to the only place dollars are real, and revoking it freezes the balances and stops the flow regardless of any messaging network.

## Weaponizing the network: how a cutoff isolates a bank

Put the layers together and you can see the full mechanism of isolating a bank as a stepwise procedure. The figure shows both prongs converging on the same outcome.

![A graph showing a SWIFT removal and a dollar-clearing cutoff converging to isolate a sanctioned bank](/imgs/blogs/swift-and-the-weaponization-of-payments-3.png)

The way this works is two reinforcing moves. **Prong one** removes the bank from SWIFT, so its messages stop flowing through the standard channel — it becomes hard to reach, forced onto phone and telex and bilateral workarounds. **Prong two**, the lethal one, forces its New York correspondents to close or freeze its accounts, cutting dollar clearing. Its dollar balances strand; its dollar flow stops. The bank ends up isolated from the dollar system regardless of which prong you emphasize, but it is prong two that does the killing.

Notice the asymmetry. A bank can survive losing SWIFT — it can route around, slowly and expensively. A bank cannot survive losing dollar clearing, because there is no "around": the dollar only exists in one place. That is why the United States, even though it does not own SWIFT, holds the decisive lever. Its power is not over the messaging layer (that is European); its power is over the settlement layer (that is American, by the simple fact that the dollar is American).

### What a sanction actually does

Let me define **sanction** properly, because it gets used loosely. A financial sanction is a legal prohibition, imposed by a government, on its own people and institutions dealing with a designated target. In the US, the agency is the Treasury's **Office of Foreign Assets Control** (OFAC). When OFAC adds an entity to the **SDN list** (Specially Designated Nationals), every US person and US bank is forbidden from transacting with it, and any of its assets within US jurisdiction must be blocked — frozen in place, not seized, but unusable.

The genius and the danger of this lever is its reach. Because the dollar is everywhere, "US jurisdiction" stretches far beyond US soil. A payment between two non-US banks in two non-US countries still falls under US jurisdiction the moment it clears a dollar leg in New York — and almost all of them do. That is the doctrine behind **secondary sanctions**, which we will get to: the US can threaten to cut off *third* parties — foreign banks that are not themselves the target — from the dollar system if they keep dealing with the sanctioned entity. The threat is credible precisely because no major bank can survive losing dollar access.

#### Worked example: the cost of being de-SWIFTed

Quantify the pain for an exporter when its bank loses the rails. Suppose a company exports \$50,000,000 of goods a year, all invoiced in dollars, and its bank has been de-SWIFTed and cut from dollar clearing.

- **Before:** payments arrive by standard wire. Cost per \$1,000,000 received: maybe \$25 in correspondent fees, settled in 1-2 days. Annual friction on \$50,000,000: about \$1,250.
- **After:** the company must route receipts through a chain of unsanctioned intermediaries in third countries, each taking a cut. A realistic workaround stack might cost 3-10% all-in, between extra intermediaries, currency conversions into and out of a non-dollar bridge currency, and a risk premium charged by anyone willing to touch the flow. Take 5%: that is **\$2,500,000** a year in pure friction, plus delays of weeks and constant uncertainty about whether each payment will arrive.
- Some buyers simply refuse to deal at all, fearing secondary sanctions on themselves. So part of the \$50,000,000 of revenue does not just get more expensive — it disappears.

The intuition: being cut off does not merely add a fee; it converts a frictionless rail into a slow, leaky, partly impassable maze, and the cost is measured in percent of every transaction, not in flat fees.

## How a connected bank differs from a cut-off bank

Set the two states side by side. The contrast is the whole reason sanctions bite.

![A before-and-after comparison of a connected bank versus a de-SWIFTed bank cut from dollar clearing](/imgs/blogs/swift-and-the-weaponization-of-payments-5.png)

A **connected bank** sends an MT103 in seconds, sees its dollar leg clear in New York the same day, and pays roughly \$25 per \$1,000,000 wire. Its clients experience international payments as fast and cheap. A **de-SWIFTed bank** cut from clearing has no automated messaging, finds its dollar balances frozen in New York, and pays 3-10% extra to cobble together workarounds — when the workarounds function at all. The left state is the ordinary plumbing of globalization; the right state is financial isolation. Sanctions are the act of moving a bank, or a whole country's banks, from left to right.

This is also why the *threat* of sanctions is often as powerful as the act. A bank does not need to be on the SDN list to behave as if it is; the mere risk that dealing with a target could get *it* cut off from the dollar makes banks "de-risk" — quietly refusing business that might draw scrutiny. The chilling effect extends the weapon's reach far beyond the formally designated names.

## How it shows up in real markets

The mechanism is not theoretical. It has been used, escalated, and tested against alternatives over the last fifteen years. Here is the timeline, then the episodes.

![A timeline of payment-weaponization milestones from Iran 2012 to the Russian reserve freeze in 2022](/imgs/blogs/swift-and-the-weaponization-of-payments-6.png)

### Iran, 2012

In 2012, amid the standoff over Iran's nuclear program, the European Union — under heavy US pressure — passed regulations that required SWIFT to disconnect designated Iranian banks. Because SWIFT is a Belgian entity bound by EU law, the EU could compel it; the US could not have done so directly. Roughly 30 Iranian banks, including the central bank, were cut off from SWIFT.

But the deeper bite came from US sanctions on dollar clearing and, critically, **secondary sanctions** that threatened any foreign bank dealing with sanctioned Iranian entities with loss of its own US access. Iran's oil revenue — invoiced and settled in dollars — became extraordinarily hard to repatriate. Buyers of Iranian oil had to resort to barter, gold, and payment in local currencies parked in accounts Iran could only spend domestically. Iran's accessible foreign reserves and its ability to convert oil into usable money collapsed, and this financial pressure was a major reason it came to the negotiating table.

In 2015-2016, under the nuclear deal (the JCPOA), Iranian banks were reconnected to SWIFT and some clearing access was restored. The lights came back on, briefly.

### Iran, 2018

In 2018 the US withdrew from the JCPOA and reimposed sanctions. It explicitly pressured SWIFT to cut Iranian banks again, and SWIFT complied to protect its own institutions from US secondary sanctions — a striking moment, because it showed that even a neutral European utility would bow to the dollar lever rather than risk its members being cut off. Iran's brief reintegration ended; the workaround economy of barter, front companies, and non-dollar trade returned.

The Iran case established the template: the formal SWIFT cutoff is the visible symbol, but the real engine is dollar-clearing denial plus secondary sanctions that turn every third-party bank into an enforcer.

### Russia, 2022

The 2022 sanctions on Russia after its invasion of Ukraine were the largest and fastest deployment of the payment weapon in history, and they had two distinct components that the public often blurs together.

First, the **de-SWIFTing**: starting in March 2022, a set of major Russian banks (initially seven, expanded later) were disconnected from SWIFT by EU action. This grabbed headlines, but on its own it was the less consequential half — Russia had been building a domestic alternative since 2014 (more on that below), and de-SWIFTing mainly raised friction.

Second, and far more powerful, the **central-bank reserve freeze**: the US, EU, UK, Japan, and others blocked the assets of the **Central Bank of Russia** held in their jurisdictions. Russia had built a war chest of roughly **\$630 billion** in foreign reserves, but a large share of it — on the order of **\$300 billion** — was held as deposits and bonds inside Western financial institutions. With one coordinated action, that \$300 billion became frozen: still "owned" by Russia on paper, but completely unusable. This is the number from the opening of this post. A central bank's reserves are supposed to be the ultimate safety net; 2022 demonstrated that if those reserves sit inside the system you are at war with, they are hostage, not insurance.

#### Worked example: a sanctioned bank cut off from dollar clearing

Make the 2022 mechanism concrete for a single bank. Suppose a sanctioned Russian bank held, across its Western correspondents, dollar and euro balances worth \$4,000,000,000 and ran \$2,000,000,000 a month of cross-border flow for clients.

- **Day zero:** EU action removes it from SWIFT; US and EU action forces its Western correspondents to freeze its accounts. The \$4,000,000,000 is now blocked — not confiscated, but frozen. The bank's balance sheet still lists \$4 billion of "assets," but it cannot spend a cent of it abroad.
- **The \$2,000,000,000 monthly flow** for clients in hard currency stops almost entirely. Importers cannot pay foreign suppliers; exporters cannot easily receive Western-currency payment.
- The bank pivots to **non-dollar rails**: routing flow through China's CIPS in yuan, through Russia's domestic SPFS for ruble messaging, and through banks in countries that have not joined the sanctions. Each pivot works partially, at higher cost, in currencies that are less freely usable.
- The frozen \$4 billion never thaws while sanctions hold. Multiply across dozens of banks and you get the macro picture: a financial system that still functions domestically but is severed from the West's hard-currency plumbing.

The intuition: sanctions did not destroy Russia's money on its own soil; they froze the slice that lived abroad and choked the pipes that connected it to Western currencies.

### The seize-or-freeze debate

The roughly \$300 billion of frozen Russian reserves became the center of a long legal and political argument that illustrates the difference between freezing and seizing. Most of the assets — held largely in Europe, with a concentration at a single large securities depository — sat blocked: Russia could not touch them, but the holders could not simply hand the principal to Ukraine either, because that would be expropriation of a sovereign's property, raising hard questions of international law and of precedent. If a Western institution can seize one central bank's reserves, every central bank has a reason to hold fewer reserves in the West.

The compromise that emerged focused on the *income* the frozen assets generated. The blocked bonds and deposits kept earning interest and coupons — windfall profits the holders had not expected. Allies moved to channel those windfall profits toward Ukraine while leaving the principal frozen, and later structured loans to Ukraine backed by the future stream of those profits. The episode is the cleanest illustration of the distinction this post insists on: freezing is reversible blocking, and it happens fast through accounting entries; seizing is taking title, and it is slow, contested, and precedent-setting. The weapon's first blow is the freeze; outright confiscation is a separate, far heavier decision.

### De-risking: the weapon's quiet shadow

The most underappreciated effect of weaponized payments is not the formal cutoffs at all — it is **de-risking**: banks pre-emptively refusing or exiting relationships they judge too risky to be worth the compliance cost or sanctions exposure. A global hub bank looks at a small respondent bank in a high-risk jurisdiction and reasons: the fees from this relationship are modest, but if one payment slips through that violates sanctions, the fine could be hundreds of millions of dollars and the reputational damage severe. So it simply closes the relationship.

The numbers behind this are stark. US and EU regulators have levied multi-billion-dollar penalties on banks that processed payments for sanctioned parties — in one notable case around 2014 a major European bank paid roughly \$8.9 billion to settle US charges of evading sanctions. After fines of that magnitude, every compliance department over-corrects. The result is that entire regions — parts of the Caribbean, the Pacific, Africa, the Middle East — have seen correspondent relationships withdrawn, lengthening their payment chains and raising costs for completely legitimate trade and remittances. De-risking is the weapon firing without anyone pulling the trigger: the mere existence and credible enforcement of the sanctions regime makes banks isolate targets, and bystanders, on their own initiative.

#### Worked example: a fine that reshapes a bank's behavior

See why one fine changes a thousand decisions. Suppose a bank earns \$2,000,000 a year in fees from correspondent relationships across a high-risk region.

- The upside is \$2,000,000 a year of revenue.
- The downside, if a single sanctioned payment slips through and is detected, has historically run to settlements measured in the **billions** — call it \$1,000,000,000 in a serious case, plus monitorships, restrictions, and reputational harm.
- A rational compliance officer compares \$2,000,000 of annual upside against a tail risk of \$1,000,000,000. The expected-value math is brutal: even a 0.5% annual chance of a billion-dollar event is an expected \$5,000,000 cost, more than twice the revenue. So the bank exits the relationships.

The intuition: once enforcement is credible and fines are enormous, banks rationally abandon legitimate business near any sanctions risk, which is why the weapon's reach extends far past the names actually on the list.

### SPFS, Mir, and the limits of building your own

Russia's response since 2014 is instructive about how far a determined state can route around the rails — and where it hits a wall. After Western sanctions following the 2014 annexation of Crimea, Russia built two domestic systems: **SPFS**, a messaging alternative to SWIFT, and **Mir**, a domestic card-payment network to replace Visa and Mastercard inside Russia. Both worked for their purpose: by 2022 most domestic Russian payments and card transactions ran on home-grown rails, so the SWIFT cutoff did far less domestic damage than it would have a decade earlier.

But notice what neither solved. SPFS is messaging — it has only a few hundred participants, almost all inside Russia or in closely aligned states, and it does nothing about dollar or euro access. Mir is domestic — Russians could keep paying each other, but a Mir card is little use abroad, and foreign banks grew wary of processing Mir transactions for fear of secondary sanctions. Russia could rebuild the messaging and the domestic card layers because those are within its own control; it could not rebuild the hard-currency settlement layer, because that lives in the West. The lesson generalizes to every would-be escapee: you can build your own postal service and your own domestic card scheme, but you cannot mint the dollar, and that is the layer that matters.

## The alternatives, and why the chokepoint holds

If the dollar rails can be weaponized, why doesn't everyone just build their own? They are trying. The trouble is that each alternative replaces only part of what SWIFT-plus-dollar-clearing provides, and crucially none of them solves the one problem that matters: clean access to the dollar.

![A comparison matrix of SWIFT versus CIPS, SPFS, and stablecoins across members, dollar access, volume, and sanctions resistance](/imgs/blogs/swift-and-the-weaponization-of-payments-4.png)

**CIPS** — China's Cross-Border Interbank Payment System — launched in 2015 to clear and settle payments in **yuan** (renminbi). It is the most serious alternative, with roughly 1,500 direct and indirect participants and growing volume (tens of billions of dollars equivalent per day). But notice the catch: CIPS clears *yuan*, not dollars. To use it as an escape from dollar sanctions, the world's trade would have to actually want to hold and settle in yuan — and the yuan is not freely convertible; China maintains capital controls; few exporters want to be paid in a currency they cannot freely spend. CIPS is real and growing, but it is a yuan rail, not a dollar-replacement rail. It also still relies partly on SWIFT for messaging in many corridors.

**SPFS** — Russia's System for Transfer of Financial Messages — is a pure *messaging* alternative built after 2014 when sanctions first loomed. It replaces SWIFT's letters, not the money. It works fine inside Russia and with a handful of friendly banks, but it has only a few hundred participants and almost no reach outside Russia's orbit. It solves the "hard to reach" problem in prong one, but does nothing about the "no dollar access" problem in prong two.

**Stablecoins** — crypto tokens pegged to the dollar, like Tether (USDT) and USD Coin (USDC) — look like an escape because they move "dollars" peer-to-peer on a blockchain without touching a correspondent bank. And to a degree they do route around the messaging layer. But they have two fatal weaknesses as sanctions-evasion tools. First, they are not actually dollar-independent: a stablecoin is only worth a dollar because its issuer holds real dollars (in US Treasuries and US bank accounts) and promises to redeem; the moment you want to turn the token back into real money, you hit the regulated banking system — and the issuer can and does freeze sanctioned wallets. Second, public blockchains are radically *transparent*: every transfer is permanently visible, so chain-analysis firms and regulators can trace flows far better than they can trace cash. For the deeper mechanics of how a "shadow dollar" actually works, see the deep dive on [stablecoins](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar). Stablecoins move value around the messaging layer, but they do not escape the dollar's gravity — they are *made of* dollars.

The matrix tells the story: each rival matches SWIFT on one axis (CIPS on volume-in-its-own-currency, SPFS on domestic messaging, stablecoins on peer-to-peer reach) but none of them gives a sanctioned party what it actually needs, which is clean, deniable, frictionless access to the dollar.

#### Worked example: a CIPS-or-crypto workaround and its friction

Price out an actual escape attempt. Suppose a sanctioned importer needs to pay a foreign supplier the equivalent of \$1,000,000 for goods, and the normal dollar rail is closed.

- **CIPS / yuan route.** The importer converts local currency to yuan (paying a spread, say 1%, since the currency pair is thin: \$10,000). It pays the supplier in yuan via CIPS. But the supplier did not want yuan — it wanted dollars to pay *its* suppliers. So the supplier must convert yuan back toward usable money, paying another 1-3% spread and possibly holding a currency it cannot freely move (\$10,000-\$30,000). All-in friction: 2-4%, i.e. \$20,000-\$40,000 on the \$1,000,000, plus the supplier's reluctance.
- **Stablecoin route.** The importer buys \$1,000,000 of a dollar stablecoin. Buying it cleanly requires a fiat on-ramp — an exchange with banking — which a sanctioned party struggles to access, so it buys at a premium through an over-the-counter desk: say 2-5% (\$20,000-\$50,000). It transfers the token to the supplier on-chain (cheap, fast). But the supplier must now cash out to real dollars through *its* bank, where compliance may freeze a flow traced to a sanctioned source, and the issuer itself can blacklist the wallet. Effective friction: 3-7% if it works at all, plus seizure risk.
- Either way, the \$1,000,000 payment costs **\$20,000-\$70,000** more than the \$25 it would have cost on the normal rail, takes longer, and carries the constant risk that some link in the chain gets frozen.

The intuition: workarounds exist, but they trade a near-free, instant, reliable dollar rail for a 2-7% tax, delay, and the ever-present chance of seizure — friction is the whole point of the weapon.

## Fragmentation: the blowback

Every time the dollar rails are used as a weapon, the targets — and even some bystanders — get a stronger incentive to build alternatives. This is the second-order cost, and it is the genuine long-run risk to the United States' position.

The logic is straightforward. A reserve currency's power rests on universal usability: everyone holds dollars and uses dollar rails *because everyone else does*, a network effect identical to why everyone uses the same messaging network. Sanctions exploit that network effect — but they also advertise its danger. Each use of the weapon tells every non-aligned government the same thing: *the dollars and rails you depend on can be switched off at someone else's discretion.* That is a powerful reason to diversify.

So you get, slowly, a fragmenting world: more bilateral trade settled in local currencies; more central-bank reserves shifted from dollars toward gold and other assets; more participants in CIPS and other non-dollar systems; more experimentation with central-bank digital currencies and cross-border digital-currency bridges that deliberately route around Western banks. None of these is close to displacing the dollar today — the dollar's share of reserves and trade settlement remains dominant — but the *direction* is unmistakable, and it is partly a consequence of the weapon being used. The connection between money creation, reserve currencies, and this fragmentation is explored further in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).

There is a genuine tension here for policymakers. The weapon is enormously effective in the short run precisely because the rails are universal. But using it erodes the universality that makes it effective. A weapon that, each time it fires, makes itself slightly weaker for next time — that is the strategic dilemma at the heart of weaponized payments.

### CBDCs and digital-currency bridges

The frontier of the fragmentation story is the **central-bank digital currency** (CBDC) — a digital form of central-bank money issued directly by a central bank, rather than the commercial-bank deposits we use today. A CBDC is interesting here for one reason: if two central banks issue digital currencies and link them directly through a shared bridge, two countries could in principle settle a cross-border payment *without* routing the dollar leg through New York at all. Projects exploring exactly this — multi-CBDC platforms where several central banks settle directly with one another on a common ledger — are the most credible long-run challenge to the dollar-clearing chokepoint, because they attack the settlement layer rather than just the messaging layer.

But the obstacles are large. A CBDC bridge only helps if the currencies involved are ones counterparties actually want to hold, which loops back to the same problem CIPS faces: most exporters still want dollars, not a basket of smaller currencies. And linking central banks across hostile blocs requires exactly the political trust that sanctions destroy. So CBDC bridges are real, are being built, and could over decades carve out non-dollar settlement corridors — but they are a slow erosion at the edges, not a switch that flips. The dollar's dominance is a network effect built over eighty years, and network effects unwind gradually, if at all.

#### Worked example: a non-dollar bridge versus the dollar default

Compare the friction of a future direct bridge against today's dollar rail to see why the dollar's gravity persists. Suppose two friendly countries want to settle \$1,000,000 of trade without dollars.

- **Today's dollar default:** convert local currency to dollars (a deep, cheap market, spread perhaps 0.05%, i.e. \$500), clear in New York for about \$25, convert dollars to the other local currency (another \$500). Total friction: roughly \$1,025, plus exposure to a possible cutoff.
- **A direct CBDC bridge in local currencies:** no New York leg, so no cutoff exposure. But the local-currency pair is thin, so the conversion spreads are wider — say 0.5% each side, i.e. \$5,000 plus \$5,000. Total friction: roughly \$10,000, and one side ends up holding a currency it may not want.
- The bridge wins only on *sanctions resistance*; the dollar default wins decisively on *cost and convenience* because its markets are deep.

The intuition: until a non-dollar pair is as deep and liquid as the dollar, escaping the chokepoint costs far more per payment, so countries default back to the dollar unless they specifically fear being cut off — which is exactly why each use of the weapon nudges a few more of them toward the costlier bridge.

## Common misconceptions

**"SWIFT moves money."** It does not. SWIFT is a messaging cooperative; it transmits instructions, not value. Money moves through clearing and settlement systems — for dollars, through CHIPS and Fedwire in New York. Conflating the messenger with the money is the single most common error, and it makes the next misconceptions follow.

**"Being cut off from SWIFT is the death blow."** It is the visible symbol, not the lethal act. A bank removed from SWIFT can route messages another way, slowly and expensively. What actually kills it is losing dollar clearing — the freezing of its New York correspondent accounts — which can be done with or without a SWIFT cutoff. The dollar-clearing chokepoint is the real weapon.

**"The US controls SWIFT, so it just orders the cutoffs."** Legally, no. SWIFT is a Belgian cooperative governed by its members and overseen by European central banks. The 2012 and 2022 SWIFT disconnections were carried out by *EU* law. The US lever is different and arguably stronger: it controls dollar clearing and wields secondary sanctions, which is why even a neutral European utility complies — its member banks cannot risk losing dollar access.

**"Crypto and stablecoins let sanctioned parties escape the dollar."** Mostly not. A dollar stablecoin is *made of* dollars — backed by real dollar assets held in the regulated US system — and its issuer can and does freeze sanctioned wallets. Public blockchains are also more traceable than cash, not less. Crypto can route around the *messaging* layer, but it does not escape the dollar's gravity or the settlement layer where the dollar is real.

**"Frozen reserves are confiscated."** Freezing and seizing are different. Frozen assets are blocked — the owner cannot use them, but legal title does not automatically transfer. Confiscation (actually taking ownership) is a far bigger legal and political step, which is why the roughly \$300 billion of Russian reserves frozen in 2022 sat in limbo for years, generating intense debate over whether and how to seize them, rather than being instantly handed over.

**"Sanctions are surgical."** They are blunt. Because the dollar touches almost everything, cutting a target tends to cut legitimate trade, humanitarian flows, and ordinary citizens too, and the secondary-sanctions chilling effect makes banks over-comply ("de-risking"), refusing far more business than the law strictly requires. The collateral reach is a feature of the network, not a bug that can be easily filed off.

## When this matters to you

You are not going to be de-SWIFTed. But this plumbing shapes the world you invest, work, and live in, and understanding it changes how you read the news.

It explains why the dollar is "exorbitantly privileged" and why that privilege is sticky: the network effect that makes the dollar universal is the same one that makes it weaponizable, and both are hard to dislodge. It explains why sanctions can be so consequential without a shot fired, and why their effects ripple into commodity prices, currency moves, and the strategies of every multinational. It explains the slow, real, but easily over-hyped drift toward fragmentation — why central banks are buying gold, why "de-dollarization" headlines keep appearing, and why the dollar nonetheless remains dominant for now. And it explains why crypto's promise of "borderless money" runs into a wall the moment it has to touch a real bank.

The one mental model to keep: **a cross-border payment is a relay of bookkeeping entries coordinated by messages, and the dollar leg of that relay almost always passes through New York.** Once you see that, the weapon is obvious — it is not a special new power, just the quiet leverage that comes from owning the place where the world's money is actually real.

### Further reading

- [Who controls the world's money: the global financial system](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) — the wider map of central banks, reserve currencies, and the institutions this post lives inside.
- [The petrodollar and dollar dominance](/blog/trading/finance/petrodollar-and-dollar-dominance) — why the dollar became the invoicing currency for oil and trade, the gravity that makes payment weaponization possible.
- [How money is created: banks, central banks, and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — what "central-bank money" and "reserves" actually are, the settlement layer this post sits on top of.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — why a dollar stablecoin is made of dollars, and why that limits it as a sanctions escape.

*All live figures — payment volumes, reserve sizes, member counts, the roughly \$300 billion of frozen Russian reserves and roughly \$1.8 trillion of daily CHIPS volume — are approximate and as of the mid-2020s; they move, and the precise numbers depend on the source and the date.*

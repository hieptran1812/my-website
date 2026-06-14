---
title: "Central Bank Digital Currencies: What Happens When Money Itself Goes Digital and State-Issued"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles tour of central bank digital currencies: what they are, why central banks are building them, how they threaten to drain banks of deposits and hand the state new power over money, and where the world's pilots actually stand."
tags: ["cbdc", "central-banks", "digital-currency", "monetary-policy", "payments", "e-cny", "digital-euro", "stablecoins", "financial-system", "money"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A central bank digital currency (CBDC) would, for the first time, let ordinary people hold a digital claim directly on the central bank — digital cash — instead of relying on a private bank's promise; that single change promises cheaper, faster payments but threatens to hollow out the banks and hand the state an unprecedented window into, and lever over, how money is spent.
>
> - Today the public can hold central-bank money in only one form: physical cash. Everything else you call "money" — your bank balance — is actually a private IOU from a commercial bank. A retail CBDC adds a second public option: digital, default-free, state-issued money.
> - Central banks are exploring CBDCs because cash use is collapsing, private stablecoins and Big Tech threaten to privatise the payment system, and existing electronic payments are slow and expensive.
> - The deepest risk is *disintermediation*: if people park savings in safe CBDC instead of bank deposits, banks lose the cheap funding they lend from — so most designs cap how much CBDC you can hold.
> - "Programmable money" — money that can be told what it may be spent on or when it expires — is the feature that excites technocrats and terrifies civil libertarians.
> - The world is split: China's e-CNY has processed over \$1 trillion in pilots, the ECB is preparing a digital euro, and the United States banned a retail CBDC outright in 2025.

Here is a fact that almost no one stops to examine: the "money" in your bank account is not money in the same sense as the cash in your wallet. The notes are a claim on the central bank — the state itself owes you. The bank balance is a claim on a private company, a promise from a commercial bank that it will give you cash if you ask. Most of the time the difference is invisible, because banks are usually good for it. But in 2008, and again in March 2023 when Silicon Valley Bank collapsed in 48 hours, depositors were sharply reminded that a bank balance is only as safe as the bank. A central bank digital currency is the proposal to give everyone, for the first time, a digital version of the safe kind of money — and that small-sounding change ripples through the entire architecture of finance.

The diagram above is the mental model for this whole post: on the left is the world we live in, where the public can touch central-bank money only as physical cash and everything digital is a bank's IOU; on the right is a CBDC world, where the public holds a digital claim on the central bank directly. Hold the difference between those two pictures in your head, because nearly every promise and every fear attached to CBDCs — cheaper payments, financial inclusion, bank runs, surveillance, programmable stimulus — flows from moving that one box.

![Two-tier money today versus a CBDC world](/imgs/blogs/central-bank-digital-currencies-cbdc-1.png)

CBDCs are not a fringe idea or a thought experiment. As of 2025, more than 130 countries representing over 98% of global GDP are researching or piloting one, according to the Atlantic Council's CBDC tracker. Three have launched live retail CBDCs; China runs the largest pilot in history. The questions are no longer "will this happen?" but "what kind, run by whom, with what limits, and who gets to watch?" This post builds the answer from the ground up — defining every term, grounding each step in arithmetic you can check, and ending with where the real-world fight stands. If you want the deeper plumbing of how ordinary money is created in the first place, that is covered in [how money is actually created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier); here we take that as given and ask what happens when the state issues a digital version of it.

## Foundations: the two tiers of money

Before we can say what a CBDC *is*, we have to be precise about what money is today, because the whole story turns on a distinction that everyday life papers over. There are two fundamentally different kinds of money in a modern economy, issued by two different kinds of institution, and they sit on two different tiers.

**Central-bank money** is money issued by the central bank itself — the Federal Reserve in the US, the European Central Bank (ECB) in the eurozone, the Bank of England in the UK. It is the most trustworthy money in the system because the issuer can never run out of its own currency; a central bank cannot "go bust" in the money it itself prints. Central-bank money comes in exactly two forms today:

- **Physical cash** — the notes and coins in your wallet. This is the *only* form of central-bank money that the public can hold directly. A \$20 bill is a direct liability of the central bank to you, the bearer.
- **Reserves** — electronic balances that commercial banks hold in accounts *at the central bank*. You and I cannot open an account at the Fed; only banks can. Reserves are the banks' own money, used to settle payments with each other at the end of the day. In the US, reserves total roughly \$3 trillion as of 2025.

**Commercial-bank money** is the other tier, and it is the one you actually use. When your banking app shows a balance of \$3,000, that number is not cash sitting in a vault with your name on it. It is a **deposit** — a promise from your bank to pay you \$3,000 in central-bank money (cash) on demand. It is the bank's liability to you, an IOU. The overwhelming majority of the money in the economy is this kind: across advanced economies, **commercial-bank deposits make up roughly 90 to 97% of the broad money supply**, while physical cash is only a few percent. Almost all the "money" in the world is private IOUs.

This two-tier arrangement is so normal we forget it is a design choice. You hold an IOU from a bank (tier two); the bank holds reserves at the central bank (tier one); and the only direct line you have to the safest, top-tier money is by pulling out paper cash. A CBDC proposes to add a new line: a way for the public to hold tier-one money digitally.

To see why the two tiers really are different in kind and not just in branding, follow what happens when you pay a friend who banks elsewhere. Your bank debits your deposit; their bank credits theirs. But the two banks must also settle the *real* money behind the promises, and they do that by moving **reserves** between their accounts at the central bank. Deposits and reserves never become each other directly — they live on separate ledgers. A deposit is the bank's promise to *you*; a reserve is the central bank's promise to the *bank*. When you withdraw cash, your deposit (the bank's IOU to you) shrinks, and the bank hands you a banknote (the central bank's IOU to the bearer), funding it by drawing down its reserves. So cash is the one point where the public reaches across from tier two to tier one. A CBDC simply adds a second, digital bridge across that same gap — and because the digital bridge is far more convenient than walking to an ATM, it is a far bigger deal than it sounds.

One more term the post turns on: **liability**. Every form of money is somebody's liability — an obligation they owe. Cash and reserves are the central bank's liabilities; deposits are a commercial bank's liability; a stablecoin is its issuer's liability. The question "whose liability is this?" is the question "who can fail and leave you holding nothing?" That is why a CBDC, as a liability of the central bank, is treated as risk-free: the issuer is the one institution that cannot run out of the currency it itself creates.

### What a CBDC actually is

A **central bank digital currency** is a digital form of a country's official money that is a direct liability of the central bank, available to the public. Strip away the jargon and it is simply this: *digital cash issued by the state*. Three words in the definition each do real work:

- *Digital* — it lives as electronic balances or tokens, spendable by phone or card, not as paper.
- *Central bank* — it is issued by the monetary authority and is its liability, so it carries no credit risk. A CBDC dollar is as safe as a paper dollar, which is to say, it is the safest dollar there is.
- *Currency* — it is denominated in and worth exactly one unit of the national money. One digital euro equals one euro equals one euro coin. It is not a new currency; it is the same currency in a new wrapper.

That last point separates a CBDC from cryptocurrencies like Bitcoin. Bitcoin is its own unit with a floating price; a CBDC is the national currency, fixed at par, just in digital form. It also separates a CBDC from a stablecoin, which is a *private* token that promises to be worth one dollar but is backed by a private firm's reserves rather than issued by the state — a distinction we will return to, and one explored in depth in [the stablecoins deep dive](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).

### Retail vs wholesale CBDC

The single most important fork in the road is *who gets to use it*. The diagram below lays out the design choices; the very first branch is reach.

![The big CBDC design choices](/imgs/blogs/central-bank-digital-currencies-cbdc-7.png)

A **retail CBDC** is meant for the general public — you, your neighbour, the corner shop. It is the kind that makes headlines because it would change daily life: a state-run digital wallet you tap to pay for coffee. Almost every fear and promise in this post concerns retail CBDCs.

A **wholesale CBDC**, by contrast, is restricted to banks and large financial institutions for settling big transactions among themselves. Here is the subtle point: banks *already* hold a digital form of central-bank money — reserves. So a wholesale CBDC is, in many ways, an upgrade to the existing reserve system rather than something genuinely new. It is far less controversial because it does not touch the public and does not threaten to drain consumer deposits. When a US official says they are "open to wholesale but opposed to retail," they mean exactly this distinction. Most of the political heat is about retail; most of the quiet, technical progress is in wholesale.

### Account-based vs token-based

The second design fork is *how the money knows it is yours*. Money has to solve one problem above all: stopping you from spending the same unit twice (the "double-spending problem"). There are two ways a CBDC can do it.

In an **account-based** system, the CBDC works like a bank account: there is a ledger of who owns what, and to pay someone you prove your identity and the system moves the balance from your account to theirs. Verification is of *identity* — the system must know who you are. This is how most existing digital money works, and it makes anti-money-laundering checks easy, but it ties every transaction to a named person.

In a **token-based** system, the CBDC works more like cash: value is held in digital "tokens" or "notes," and to pay someone you hand over a valid token, the way you hand over a banknote. Verification is of the *token's validity*, not your identity. A well-designed token system can offer cash-like privacy for small amounts. The price is that, like losing a wallet, losing your device or keys can mean losing the money, and large anonymous transfers raise money-laundering concerns. Most real designs blend the two: token-like privacy for small everyday sums, account-like identity checks above a threshold.

### The intermediated, two-tier design

A naive worry about CBDCs is that everyone would suddenly bank directly with the central bank, and the Fed would have to run 330 million checking accounts and a customer-service line. No serious proposal works that way. The consensus design is the **two-tier** or **intermediated** model, and it deliberately preserves the existing banking layer. The diagram below traces a single payment through it.

![An intermediated CBDC payment for one hundred dollars](/imgs/blogs/central-bank-digital-currencies-cbdc-3.png)

In the intermediated model, the central bank issues the CBDC and runs the core ledger (or the settlement layer), but private intermediaries — banks and licensed fintechs — handle everything that touches the customer: opening wallets, identity checks, apps, support, fraud monitoring. You would download a wallet from your bank or a payment app, and it would hold CBDC, but the value in it is a claim on the central bank, not on the bank. If your bank failed, your CBDC would be untouched, because it was never the bank's liability. The bank is a distributor and a service layer, not the issuer. This design keeps the central bank out of retail customer service and keeps banks in business as the public's interface to money — a compromise meant to defuse the bank-disintermediation fear we turn to next.

### Disintermediation risk — the core danger

Here is the structural problem that haunts every retail CBDC. Banks fund their lending largely with deposits, which are a cheap and stable source of money. A CBDC is *safer* than a deposit — it is a claim on the state, not on a bank that can fail. So in normal times some people would shift money from deposits into CBDC for safety and convenience, and in a panic, *everyone* might try to. If deposits drain out of banks and into CBDC, banks lose their cheapest funding, must replace it with more expensive borrowing, and lend less — or the whole thing tips into a digital bank run that happens at the speed of a smartphone. This hollowing-out of the banking system is called **disintermediation**, and it is the single biggest reason central banks tread carefully. We will dissect the mechanics, with numbers, shortly. The standard defence is a **holding cap**: a legal limit on how much CBDC any one person may hold (the ECB has floated figures around 3,000 euros), so a CBDC works for payments but not as a giant safe-deposit box that empties the banks.

### Programmable money

The last foundation is the feature that most divides opinion. Because a CBDC is software, the rules of how it can be spent can in principle be written into the money itself. **Programmable money** means money that carries conditions: it could be set to expire on a date, to be spendable only on certain categories of goods, to flow only to approved recipients, or to automatically pay tax at the point of sale. To a policymaker that sounds like a powerful tool — stimulus that *must* be spent, subsidies that cannot be diverted. To a civil libertarian it sounds like the state deciding what your money is allowed to do. Whether a CBDC is programmable, and who holds the keys, is arguably the most consequential design choice of all, and we will give it a full worked example below.

### Privacy

Cash has a property we rarely name until it is threatened: it is anonymous. When you pay cash, no third party need ever know. Every digital payment today is *seen* — by your bank, the card network, the payment app. A retail CBDC, if account-based and run by the state, could in principle give the central bank (and the government behind it) a complete, real-time record of every transaction in the economy. That is a surveillance capability no government has ever had. Designers know this is the make-or-break issue for public trust, and most Western proposals promise privacy safeguards, data firewalls between intermediaries and the state, and cash-like anonymity for small payments. Whether those promises would survive a future crisis or a different government is exactly what critics doubt.

## Where a CBDC sits in the money hierarchy

It helps to place a CBDC in the layered structure of money. The stack below shows the hierarchy with the new tier added.

![The money hierarchy with a CBDC layer](/imgs/blogs/central-bank-digital-currencies-cbdc-5.png)

At the base sit **central-bank reserves**, the top-tier money that only banks can hold — roughly \$3 trillion in the US. Above that is **physical cash**, the public's existing slice of central-bank money, around \$2.3 trillion in US currency in circulation and falling as a share of payments. A retail **CBDC** would be a new public, digital, default-free layer — central-bank money the public can hold electronically for the first time. And resting on top, dwarfing all of it, are **bank deposits**, the 90-to-97% of broad money that is private IOUs created when banks lend. The reason the disintermediation fear is real is visible right here: the CBDC layer would compete directly with the giant deposit layer above it, because for the first time the public would have a digital alternative to a bank IOU that is actually *safer* than the IOU.

Notice what the CBDC does *not* replace. It does not abolish bank deposits or cash; it adds an option. The fight is over how big that option is allowed to get.

## Why central banks are exploring CBDCs

No central bank wakes up wanting to rebuild the payment system for fun. The CBDC wave is driven by a handful of concrete pressures, and understanding them explains why even reluctant institutions feel they cannot ignore it.

**The decline of cash.** Cash is the public's only existing access to central-bank money, and it is vanishing. In Sweden, cash fell to under 10% of point-of-sale transactions; the Riksbank worried aloud that a society with *no* public access to state money — where every payment runs through private firms — is a society that has quietly privatised its monetary base. A retail CBDC is, in this framing, simply digital cash to replace the paper cash people are abandoning, preserving public access to central-bank money in a cashless age.

**The stablecoin and Big Tech threat.** This is the sharper, more recent driver. Privately issued stablecoins — dollar-pegged tokens like Tether (USDT) and USD Coin (USDC) — already move enormous volumes; the stablecoin market exceeded \$200 billion in 2025. When Facebook (Meta) proposed its Libra/Diem global stablecoin in 2019, central banks panicked: a single private company with billions of users issuing its own near-money could, in effect, privatise the dollar and put a corporation between the state and its currency. CBDCs are partly a defensive move — a way for central banks to offer a public digital option before private money fills the vacuum. As one BIS official put it, the choice may be between a public CBDC and a future dominated by private digital money the state does not control.

**Payment efficiency.** Existing electronic payments are slower and costlier than the marketing suggests. A card payment can take days to fully settle behind the scenes, and merchants pay interchange fees of roughly 1.5 to 3% per transaction. Cross-border payments are worse: slow, opaque, and expensive, often costing 5 to 7% in remittance fees. A well-built CBDC could settle instantly, at near-zero marginal cost, around the clock. We will put numbers on this saving below.

**Financial inclusion.** In many countries a large share of adults have no bank account but do have a phone. The World Bank has estimated well over a billion adults globally are "unbanked." For them, a bank account is often uneconomic — banks make little money on a low-balance customer, so they do not chase them. A CBDC accessible by a basic mobile device, issued by the state and not dependent on a bank finding the customer profitable, could in principle bring those people into the digital economy and let them receive wages, benefits, and remittances electronically. This was Nigeria's stated motivation for the eNaira, and it is a genuine promise. The catch, as we will see, is that "could in principle" is doing heavy lifting: where people already have working mobile money (as in much of Africa), a CBDC adds little, and where they do not, the barrier is often trust or smartphones, which a CBDC alone does not fix.

**Monetary control and a new policy tool.** A CBDC could give central banks more direct levers. In a deep recession with interest rates already at zero, a programmable CBDC could in theory implement negative rates that cash cannot escape, or push stimulus directly into citizens' wallets without going through banks. These are double-edged — the same tools that are powerful for policy are alarming for liberty.

There is a sixth driver that gets less press but motivates a lot of quiet work: **geopolitical and cross-border ambition.** The dollar's dominance of global trade rests partly on the network of correspondent banks and messaging systems (like SWIFT) that route international payments — infrastructure the US can see into and, when it chooses, cut countries off from through sanctions. A network of linked CBDCs could route cross-border value without passing through that dollar-based plumbing at all. China is explicit that the e-CNY is, in the long run, part of a strategy to make the renminbi more usable in international trade and to build payment rails it controls. Smaller economies see linked CBDCs as a way to cut the cost and the gatekeeping of cross-border payments. This is less about your morning coffee and more about who owns the pipes of global money — which is exactly why the question of *whose* cross-border CBDC platform becomes the standard is a strategic contest, not just a technical one.

Notice that these drivers pull in different directions. Financial inclusion argues for a CBDC anyone can hold freely; bank stability argues for tight caps that limit it. Payment efficiency argues for a frictionless, always-on system; anti-money-laundering rules argue for identity checks and friction. The reason CBDC design is so contested is that a single instrument is being asked to serve goals that partly conflict, and every country resolves the conflict differently.

## Design tradeoffs

The reason no two CBDCs look alike is that every design choice trades one good thing against another. There is no neutral, "obvious" CBDC; each is a bundle of political decisions dressed as engineering.

- **Privacy vs control.** More privacy (token-based, anonymous small payments) means less ability to police money laundering and tax evasion. More control (account-based, full identity) means more surveillance. You cannot have both at the extremes.
- **Bank stability vs usefulness.** A CBDC with no holding limit and that pays interest would be wildly useful and a giant deposit magnet that destabilises banks. A tightly capped, non-interest-bearing CBDC is bank-safe but not much more than digital cash. Designers tune the cap and the interest to sit between these poles.
- **Resilience vs efficiency.** A single central ledger run by the central bank is efficient but a single point of failure and a single point of surveillance: if it goes down or is compromised, the whole payment system stops, and whoever controls it sees everything. A distributed design spreads the data and the risk across many nodes, so no single outage or breach is fatal, but it is slower, more complex, and harder to govern. There is also an **offline** dimension: cash works in a blackout, but a digital currency that needs a live network does not — so resilient designs (the digital euro included) invest heavily in an offline mode where two devices can exchange value directly, which in turn reopens the double-spending and privacy questions the online system had solved.
- **Inclusion vs gatekeeping.** Letting anyone with a phone hold CBDC maximises inclusion but complicates identity checks; requiring full bank-grade onboarding undercuts the inclusion goal.

One tradeoff deserves singling out because it quietly decides almost everything else: **should the CBDC pay interest?** Cash pays nothing, so the simplest CBDC pays nothing either and is purely a payment instrument. But the moment a CBDC pays a competitive interest rate, it becomes a savings vehicle — a risk-free, state-backed account that out-competes every bank deposit, because no bank can promise both a return *and* zero default risk. That would supercharge disintermediation. So the dominant design choice is a *non-remunerated* CBDC, often with **tiered remuneration**: the first slice (up to the holding cap) earns zero, and any amount above is either blocked or charged a penalty rate, deliberately making the CBDC useless as a place to park large savings. This is the lever that lets a central bank tune the CBDC to be "good enough for payments, bad enough that it doesn't empty the banks." It is also, not coincidentally, the lever that a negative-rate policy would pull in the other direction — which is why monetary-policy hawks and bank lobbyists watch the interest design more closely than any other parameter.

The disintermediation tradeoff deserves its own treatment because it is where the engineering meets the existential risk to banks.

## The risks: banks, surveillance, and programmability

### Banks losing deposits

We met disintermediation in the foundations; now we trace the mechanism. The graph below shows the two paths a dollar of deposits can take.

![How a CBDC can drain bank deposits](/imgs/blogs/central-bank-digital-currencies-cbdc-4.png)

Start with \$100 sitting in a bank deposit. That \$100 is funding for the bank — it is money the bank can lend out (banks must hold capital and meet liquidity rules, but deposits are the cheap raw material of lending). Now the depositor, either for everyday safety or in a moment of stress, moves the \$100 into a risk-free CBDC. If there is *no holding cap*, the bank simply loses \$100 of cheap, stable funding. To keep lending the same amount, it must replace that \$100 with more expensive wholesale borrowing, which raises its costs and squeezes the credit it can extend — so lending tightens across the economy. Multiply by millions of depositors and a bank can find its deposit base shrinking structurally, and in a panic, catastrophically fast. If there *is* a holding cap (the right branch of the graph), the drain is bounded: people can move only a few thousand dollars each into CBDC, so banks keep most deposits and the system stays stable. The cap is the load-bearing safety device of every serious retail design.

The scarier version is the **digital bank run**. In March 2023, Silicon Valley Bank lost \$42 billion of deposits in a single day as customers fled by phone. The fear is that a CBDC makes the safe haven one tap away: at the first sign of trouble, depositors could move their money to the absolutely-safe CBDC instantly, turning a wobble into a collapse faster than any regulator could respond. A holding cap blunts this too, but does not eliminate the dynamic. The deeper problem is that in a *system-wide* panic, the flight is not from one shaky bank to a sound one — it is from the entire commercial-banking tier to the central bank. Today that flight is throttled by the friction of moving money and the limits of deposit insurance; a CBDC could remove the friction entirely. This is why some economists argue a retail CBDC is, in effect, structurally pro-cyclical: convenient in good times, dangerously frictionless in bad ones. Designers respond with caps, with tiered remuneration (paying zero or negative interest above a threshold so the CBDC is unattractive as a savings vehicle), and with the ability to slow conversions in a crisis — but each of these chips away at the "it's just digital cash" promise.

### Surveillance

The privacy risk is not hypothetical. An account-based retail CBDC run by the state is, by construction, a database of every payment everyone makes. Even if the current government promises restraint, the *capability* persists for whatever government comes next. Critics point to China, where the e-CNY is integrated with a state that already runs extensive surveillance, as a preview of what a CBDC can become in the wrong hands. Western central banks insist on data firewalls — where the central bank sees only anonymised settlement data and the intermediary sees the customer, with no single party seeing everything — but the trust required is enormous, and once the rails exist, the temptation to use them grows.

### Programmability and expiry controls

The most dystopian fear, and the one most worth understanding precisely, is programmable control. Because CBDC is code, money could carry rules: spend by Friday or it disappears; usable only for food, not alcohol; cannot be sent abroad; frozen for a named person. Defenders note that programmability can be *good* — automatic tax remittance, escrow that releases on delivery, welfare that reaches the intended person — and that programmability can live in the wallet (your choice) rather than the money (the state's choice). Critics counter that whoever can program the money can control behaviour, and a government in a crisis or an authoritarian turn could weaponise it. This is not a settled debate; it is the central political question of CBDC design.

The crucial distinction is *where the programmability lives* and *who holds the key*. "Programmable payments" — automation you opt into, like a standing order, a smart contract that releases an escrow when goods arrive, or a parent restricting a child's allowance card — are uncontroversial; they are just convenience, and versions of them already exist. "Programmable money" — rules baked into the currency itself by the issuer, that travel with every unit and that the holder cannot remove — is the dangerous category, because it shifts control from the spender to the state. The same expiry feature that makes a stimulus more effective could, under a different government, become a tool to coerce spending, penalise saving, or punish dissent by freezing an individual's funds at the source. Most Western central banks have gone out of their way to disavow issuer-side programmability of the dangerous kind; the ECB has stated the digital euro would *not* be programmable money in that sense. But disavowals are policy choices, not technical guarantees — the capability exists the moment the money is software, and what a future legislature does with it is precisely what worries the skeptics. The honest summary is that the technology creates an enormously powerful lever, and the only thing standing between "useful automation" and "behavioural control" is law, governance, and political will.

## The global state of play

Theory aside, the world is already running live experiments, and they point in sharply different directions. The timeline below marks the milestones.

![CBDC milestones from 2020 to 2025](/imgs/blogs/central-bank-digital-currencies-cbdc-6.png)

**China's e-CNY** is the giant. The People's Bank of China began public pilots of the digital yuan (e-CNY, or digital renminbi) in April 2021, and by mid-2024 cumulative transaction volume had surpassed the equivalent of \$1 trillion across pilot cities, used for everything from subway fares to salaries for some civil servants. It is the largest CBDC experiment in history by orders of magnitude. China's motivations are debated: payment modernisation, reducing the dominance of private giants Alipay and WeChat Pay, and — geopolitically — a long-run hope of reducing reliance on the dollar-based system for cross-border trade. Adoption inside China has nonetheless been slower than the volume headline suggests, because Alipay and WeChat Pay already work so well that citizens see little reason to switch. The e-CNY uses a centralised, account-based design with a tiered-wallet structure: small wallets need little identity verification (cash-like for tiny sums), while larger wallets require full identification — a deliberate "managed anonymity" that the PBOC frames as privacy-with-traceability and that critics frame as surveillance by design. Whichever framing you accept, the e-CNY is the clearest existence proof that a sovereign retail CBDC can run at the scale of hundreds of millions of users, and it is the reference architecture every other central bank studies.

**The ECB's digital euro** is the most-watched Western project. The ECB completed a two-year investigation phase and in October 2023 moved into a "preparation phase," building toward a possible launch decision later in the decade. The design emphasises a holding cap (figures around 3,000 euros have been discussed) to protect banks, an offline mode that works without internet, and privacy safeguards including the promise that the ECB itself would not see individuals' transaction data. It is explicitly framed as a defensive measure to keep European payments sovereign rather than dependent on US card networks and dollar stablecoins — a point Europe feels acutely, since a huge share of its card payments run through American networks (Visa, Mastercard) over which it has no control. The politics are fierce: European banks have lobbied hard for the lowest possible holding cap to protect their deposit base, consumer and privacy groups scrutinise every line of the data architecture, and the enabling legislation is still grinding through the European Parliament and member states. No final issuance decision has been made, and the digital euro could still be delayed or watered down; it is the live test of whether a major democracy can build a CBDC the public actually trusts.

**The Fed's reluctance — and the US ban.** The United States has been the most cautious major economy. The Fed studied the question but never committed, citing risks to bank intermediation, privacy, and the question of whether a CBDC was even necessary given a strong private payment sector. The debate then turned political: critics on the right framed a retail CBDC as a tool of state surveillance and control. In January 2025, a US executive order explicitly prohibited the establishment of a retail central bank digital currency, effectively taking a US retail CBDC off the table for the foreseeable future while leaving wholesale and private stablecoins as the favoured path. The contrast with China could hardly be sharper.

**The Bahamas Sand Dollar** holds a historic distinction: launched in October 2020, it was the world's first fully deployed retail CBDC. The motivation was practical — the Bahamas is an archipelago where serving remote islands with bank branches and cash logistics is expensive, so a digital currency improves access and resilience after hurricanes. Adoption has been modest in absolute terms (the Sand Dollar is a tiny fraction of Bahamian money), but it proved the concept works, and other small economies (Nigeria, the Eastern Caribbean) followed.

**Nigeria's eNaira** is the cautionary tale. Launched in October 2021 with grand financial-inclusion goals, it saw dismal uptake — by some measures fewer than 1% of Nigerians used it actively a year in, and wallet adoption stayed in the low single-digit millions against a population over 200 million. The lessons were brutal: people already had popular mobile money and bank apps, the eNaira offered them little extra, trust in government monetary management was low, and a clumsy attempt in early 2023 to force adoption by sharply restricting cash withdrawals backfired into ATM queues, street protests, and a Supreme Court rebuke. The eNaira shows that building a CBDC is the easy part; getting people to *want* it is the hard part. A CBDC that solves a problem citizens do not feel — and that arrives without the trust to overcome the surveillance worry — simply sits unused, no matter how elegant the technology. Adoption, not engineering, is the binding constraint.

## CBDC vs stablecoin vs crypto vs cash

It is worth pinning down precisely how a CBDC differs from the alternatives it is often confused with, because the differences are exactly what the policy fight is about. The matrix below compares the four.

![Four kinds of money compared](/imgs/blogs/central-bank-digital-currencies-cbdc-2.png)

**Cash** is issued by the central bank, carries no default risk, is physical, anonymous, and pays no interest. It is the benchmark a CBDC tries to digitise.

**A bank deposit** is issued by a commercial bank, carries the bank's default risk (mitigated by deposit insurance up to a limit), is digital, visible to the bank, and may pay interest. It is what most "money" actually is today.

**A CBDC** is issued by the central bank, carries no default risk, is digital, *potentially* visible to the state (a design choice), and may or may not pay interest (a design choice). It is, in effect, cash's safety with a deposit's convenience — which is exactly why it is both attractive and dangerous to the banks.

**A stablecoin** is issued by a private firm, carries that issuer's credit and reserve risk (it can "break the peg" or suffer a reserve run), is a digital token, visible to the issuer, and rarely pays the holder interest. It is privately issued near-money — a competitor to the CBDC, not a version of it.

And **cryptocurrency** like Bitcoin is the odd one out: no issuer at all, a floating price rather than a fixed one-to-one peg, and therefore not "money" in the unit-of-account sense for most purposes — it is a volatile asset, not a stable medium of exchange. A CBDC and Bitcoin are almost opposites: one is the most centralised money imaginable, the other the most decentralised. The crypto vision and its origins are traced in [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision); a CBDC is, in a sense, the establishment's answer to it.

## Worked examples

Numbers make these mechanisms concrete. Each example below shows the arithmetic step by step, with every dollar figure spelled out so you can check the logic yourself.

#### Worked example: a CBDC payment vs a card payment

Imagine a coffee shop sells a \$100 order and the customer pays by card. The merchant does not receive \$100. Card networks charge an *interchange fee* plus processing, which in the US runs around 2% for a typical card transaction. So the breakdown is:

```
Sale price:            $100.00
Interchange + fees:    $2.00  (about 2%)
Merchant receives:     $98.00
Settlement time:       1 to 3 days
```

Now the same \$100 paid with a CBDC. The central bank's ledger simply moves \$100 of CBDC from the customer's wallet to the merchant's wallet. There is no card network taking a cut; the marginal cost of the ledger update is effectively zero (the central bank might charge a tiny flat fee, but it is not a 2% slice of the sale). So:

```
Sale price:            $100.00
CBDC transfer fee:     ~$0.00
Merchant receives:     $100.00
Settlement time:       instant, final
```

The merchant keeps the full \$2 that the card network would have taken, on every \$100 of sales. For a shop doing \$500,000 a year in card sales, that 2% is \$10,000 a year — real money. Scaled across an economy, the interchange that flows to card networks runs into the tens of billions annually. The intuition: a CBDC threatens the card networks' fee revenue precisely because it lets value move at the cost of a database write rather than the cost of a private toll.

#### Worked example: disintermediation shrinking a bank's lending

Take a mid-sized bank funded with \$10 billion of customer deposits. Suppose a retail CBDC launches with no holding cap, and over a year 20% of deposits migrate to the safer, more convenient CBDC. Step through it:

```
Deposits before:           $10,000,000,000
Migrate to CBDC (20%):     -$2,000,000,000
Deposits after:            $8,000,000,000
```

The bank has lost \$2 billion of its cheapest funding. To keep its loan book steady it must either shrink lending or replace that \$2 billion with wholesale borrowing. Say wholesale funding costs 2 percentage points more than deposits. The extra annual cost is:

```
Replacement funding:       $2,000,000,000
Extra cost (2%):           $40,000,000 per year
```

That \$40 million either comes out of the bank's profits, gets passed to borrowers as higher loan rates, or causes the bank to simply lend \$2 billion less. The intuition: even a moderate, non-panicked shift of deposits into CBDC quietly raises the cost of credit across the economy — which is exactly why the holding cap exists. With a \$3,000-per-person cap, a bank's retail deposits above that threshold are protected and the \$2 billion drain cannot happen.

#### Worked example: programmable stimulus that expires

Suppose a government wants to stimulate the economy in a recession and sends every adult \$1,000. With ordinary money, people might save much of it — in past stimulus rounds, a large share was saved or used to pay down debt rather than spent, blunting the stimulus. With a *programmable* CBDC, the government could attach a rule: the \$1,000 expires in 90 days if unspent. Walk through one household:

```
CBDC stimulus received:        $1,000
Rule attached:                 expires in 90 days
Household's choice:            spend it or lose it
Amount spent within 90 days:   $1,000 (forced)
Amount saved:                  $0
```

Compare a normal stimulus where the same household might spend only \$400 and save \$600. The programmable version pushes a full \$1,000 into circulation, increasing the spending "multiplier" the government wanted. That is the policymaker's dream — and the libertarian's nightmare, because the same mechanism that says "spend this by Friday" could say "this money cannot be spent on category X" or "this money is frozen." The intuition: programmability turns money from a neutral store of value into a controllable instrument of policy, and whoever writes the rules holds real power over behaviour.

#### Worked example: the seigniorage the state captures

When the public holds central-bank money instead of a bank deposit, the central bank earns the investment income on the assets backing it — a profit called **seigniorage**. Suppose a CBDC succeeds well enough that the public holds \$500 billion of it instead of in bank deposits. The central bank issues that \$500 billion as a (non-interest-bearing) liability and holds \$500 billion of interest-earning government bonds against it. At a 4% bond yield:

```
CBDC outstanding (CB liability):   $500,000,000,000
Backing assets (govt bonds):       $500,000,000,000
Yield on backing:                  4%
Annual seigniorage:                $20,000,000,000
```

The central bank earns roughly \$20 billion a year, which it typically remits to the treasury — public revenue. Today, much of that income stream is effectively captured by commercial banks (who earn the spread on deposits) and by private stablecoin issuers (Tether reported multi-billion-dollar annual profits, almost entirely from the interest on the US Treasuries backing its tokens). When you hold a stablecoin, the issuer — not you and not the public — pockets the yield on the reserves; when you hold a CBDC, that yield accrues to the central bank and flows back to taxpayers. The intuition: a CBDC is partly a fight over *who* gets to earn the interest on the money supply — banks, private stablecoin firms, or the state on behalf of taxpayers. The sums are large enough that the question is genuinely a matter of public finance, not a rounding error.

#### Worked example: a cross-border CBDC settlement

Consider a worker in the US sending \$1,000 home to family abroad through a traditional remittance service. The fees and exchange-rate markup are brutal:

```
Amount sent:               $1,000
Remittance fee (~6%):      -$60
Received by family:        $940
Time:                      1 to 4 days
```

The global average cost of remittances has hovered around 6% for years. Now suppose two central banks link their CBDCs through a shared cross-border platform (the BIS has run pilots like Project mBridge connecting several countries' wholesale CBDCs). The two CBDCs settle directly against each other, cutting out the chain of correspondent banks that each take a slice:

```
Amount sent:               $1,000
Platform fee (~0.5%):      -$5
Received by family:        $995
Time:                      seconds
```

The family receives \$995 instead of \$940 — \$55 more — and within seconds rather than days. The saving comes from collapsing a long chain: today the money hops through a correspondent bank in the sending country, an intermediary or two, and a receiving bank, each adding a fee and a delay; a linked CBDC platform settles the two central-bank monies directly against each other in one step. Across the roughly \$800+ billion in annual global remittances, shaving even a few percentage points returns tens of billions of dollars to some of the world's poorest households. The intuition: cross-border CBDC links attack the most expensive, slowest corner of the payment system, which is precisely why the geopolitics of *whose* platform becomes the standard — explored more in [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) — matters enormously. Whoever owns the rails sets the rules, the fees, and who can be cut off.

## Common misconceptions

**"A CBDC means the Fed will run everyone's bank account."** No. Every serious design is two-tier: the central bank issues and settles, but private banks and fintechs run the wallets, onboarding, and support. You would never call the central bank's helpline. The central bank deliberately stays out of retail banking.

**"A CBDC is just a cryptocurrency / it uses blockchain."** Not necessarily either. A CBDC is the national currency in digital form, fixed at one-to-one, issued by the state — the opposite of a decentralised, floating-price cryptocurrency. And while some CBDCs use distributed-ledger technology, many do not; the central design (China's e-CNY included) often uses a conventional centralised database, not a public blockchain. Blockchain is an implementation option, not a defining feature.

**"A CBDC is the same as the digital money I already use."** No — and this is the crux. Your bank balance is a *private IOU* from a commercial bank that can fail. A CBDC is a *direct claim on the central bank* that carries no default risk. They feel identical when you tap to pay, but they are fundamentally different kinds of promise from fundamentally different issuers. The whole disintermediation problem exists *because* a CBDC is safer than the deposit it competes with.

**"A CBDC will abolish cash and force everyone digital."** A CBDC need not replace cash; most proposals explicitly keep cash alongside it. Cash is disappearing on its own as people choose cards and apps; a CBDC is often pitched as *preserving* public access to central-bank money in a world that is going cashless regardless. Whether a future government would later phase out cash is a separate political decision, not an automatic consequence of issuing a CBDC.

**"Programmable money means the government will automatically control your spending."** Programmability is a capability, not a destiny, and it cuts both ways. It can live in the wallet (rules you choose, like a child's allowance card) or in the money (rules the state imposes). The risk is real and worth fighting over, but "CBDC equals automatic state control of your spending" conflates the most dangerous possible design with the technology itself. The design choices — and the laws around them — determine the outcome.

**"CBDCs are inevitable and coming everywhere soon."** The reality is far messier. The US has *banned* a retail CBDC; Nigeria's flopped; the digital euro has no firm launch date; only China runs one at scale, and even there everyday adoption trails the headline volume. CBDCs are a live experiment with very different outcomes by country, not a uniform global rollout.

**"A CBDC will replace stablecoins, or stablecoins will replace the need for a CBDC."** They are competitors, but the contest is unresolved and may not produce a single winner. In jurisdictions that ban or delay a retail CBDC (notably the US after 2025), regulated private stablecoins are being positioned as the digital-dollar substitute, leaving the "public option" off the table. In jurisdictions wary of letting private firms run the monetary base (the eurozone, China), a CBDC is the answer and private stablecoins are kept on a tight leash. The likely future is plural: cash, deposits, CBDCs, and stablecoins coexisting, with the balance set by each country's politics rather than by any technical superiority of one form over another.

## How it shows up in real markets

**China's e-CNY at scale.** The clearest demonstration that a retail CBDC can run at national scale: over \$1 trillion in cumulative pilot volume by 2024, integrated into transit, retail, and some government payments. It also shows the limits — entrenched private apps (Alipay, WeChat Pay) mean a technically successful CBDC still struggles to win daily habit. It is the reference case for both believers and skeptics.

**The digital euro debate.** The ECB's preparation phase has triggered a real fight: banks lobbying hard for low holding caps to protect their deposits, privacy groups scrutinising the data design, and politicians weighing monetary sovereignty against the disruption. It is a live case study in how every design parameter (the cap, interest, privacy) is contested by the interests it affects.

**The US political pushback.** The 2025 executive order banning a retail CBDC turned a technocratic question into a partisan symbol of surveillance and government overreach. It is a striking example of how the *politics* of money can override the economics — and of a major economy choosing private stablecoins and wholesale rails over a public retail currency.

**Nigeria's eNaira struggles.** The eNaira's near-total failure to gain traction — under 1% active use — is the field's most-studied flop. It proved that issuing a CBDC is trivial next to the problem of adoption, trust, and offering people something they actually lack. Forcing it by squeezing cash backfired badly.

**The Sand Dollar precedent.** The Bahamas' 2020 launch is the live proof-of-concept the whole field cites: a real, sovereign retail CBDC operating for years, useful for resilience in a disaster-prone archipelago, even if small. It moved CBDCs from PowerPoint to production.

**A bank-run-to-CBDC scenario.** Though not yet observed (because no large economy has a live retail CBDC), the SVB collapse of 2023 — \$42 billion fleeing in a day, all of it digital — is the template for the regulators' nightmare: a CBDC that makes the risk-free haven one tap away could turn a wobble into a run at smartphone speed. It is why every serious design ships with a holding cap, and why the Fed cited financial-stability risk as a core reason for caution. The deeper mechanics of how central banks manage such stress sit alongside [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

## When this matters to you / further reading

For most people, the practical takeaway is simple: watch the *design parameters*, not the headlines. Whether a CBDC helps or harms you comes down to a few specific choices — the holding cap (does it protect banks and limit your exposure, or is it unlimited?), the privacy architecture (does the state see your payments, or only anonymised settlement data?), the programmability rules (can the money be told what to do, and by whom?), the interest design (is it deliberately unattractive as a savings vehicle, or a deposit-magnet?), and whether cash survives alongside it. A "good" CBDC is digital cash: private for small payments, capped so it does not break the banks, non-interest-bearing so it does not drain deposits, and non-programmable from the state's side. A "bad" one is a surveillance-and-control instrument wearing the costume of convenience. The technology is the same; the politics decide which you get.

So when a CBDC is announced in your country, the questions worth asking are concrete, not philosophical. What is the holding cap, and does it apply per person? Who can see my transactions, and is that guaranteed in law or merely promised in a press release? Can the money be frozen, expired, or restricted at the issuer's discretion? Does it pay interest, and how is the rate set? Is cash protected by law as a parallel option? The answers to those five questions tell you far more than any slogan about "innovation" or "the future of money." A CBDC is not inherently good or bad; it is a tool whose character is set entirely by the rules wrapped around it — and those rules are written by people you can, in a democracy, pressure. The single most important thing an ordinary citizen can do is treat the design debate as the civic matter it is, rather than a technical detail to be left to engineers and central bankers.

If you found this useful, the natural next reads are: [how money is actually created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) for the foundation of where ordinary money comes from; [the stablecoins deep dive](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) for the private competitor that CBDCs are partly a response to; [who controls the world's money](/blog/trading/finance/who-controls-the-worlds-money-global-financial-system) for the geopolitics of monetary power; and [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) for the monetary-policy machinery a CBDC would plug into. The single sentence to carry away: a CBDC gives the public a direct, digital claim on the state for the first time — and everything good and everything dangerous about that flows from the same fact.

*All figures (transaction volumes, market caps, reserve totals, fee percentages) are approximate and as of 2025 unless stated otherwise, and are illustrative of magnitudes rather than precise current values. Nothing here is investment advice.*

---
title: "Operational Risk: Fraud, Cyber, and the Loss Events"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why the risk that has nothing to do with lending or trading positions can still kill a bank, how the seven Basel event types and the loss-distribution approach put a capital number on it, and what the great rogue-trader, cyber, and conduct disasters teach about controls."
tags: ["banking", "operational-risk", "fraud", "cyber-risk", "rogue-trader", "loss-distribution-approach", "key-risk-indicators", "three-lines-of-defense", "barings", "wells-fargo"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Operational risk is the loss a bank takes from its own broken machinery: failed processes, dishonest or careless people, crashed systems, and outside events like fraud and cyber attacks. It is not about whether a borrower repays or a position moves; it is about whether the bank can be trusted to run itself.
>
> - Basel sorts every operational loss into **seven event types** — from internal fraud (a rogue trader) to failed execution (a settlement error) — and banks must hold capital against the whole distribution of them.
> - The losses are **fat-tailed**: thousands of cheap, frequent errors plus a tiny number of catastrophic ones. The capital you hold is sized for the tail, not the average.
> - The defenses are **controls, not buffers**: segregation of duties, limits, key risk indicators, and the three lines of defense. A single missing control is how one trader at Barings turned an \$827 million loss into the death of a 233-year-old bank.
> - The one number to remember: Barings collapsed on a **\$1.3 billion** loss it could not absorb, while Wells Fargo paid roughly **\$4.9 billion** in fines for fake accounts and survived. The difference between a fine and a funeral is whether the loss outruns the equity.

In February 1995, the oldest merchant bank in Britain — the bank that had financed the Louisiana Purchase, that counted the Queen among its clients — was sold to a Dutch group for one pound. Barings had survived 233 years, two world wars, and the Napoleonic period. What finally killed it was not a recession, not a bad loan, not a market crash that everyone saw coming. It was one trader in a Singapore office, a hidden account, and the simple fact that the same man who placed the trades also signed off on the paperwork that was supposed to check them.

That is operational risk. It is the strangest of the four risks a bank runs because it has nothing to do with the bank's actual business decisions. A credit loss means the bank lent to someone who could not pay — at least it was *trying* to make money. A market loss means a position moved against the bank — again, a bet that went wrong. Operational risk is different: it is the loss that arrives because the bank's own internal machinery broke. A clerk fat-fingered an order. A hacker got into the payment system. A team mis-sold a product. A computer migration locked millions of customers out of their accounts. None of these are bets. They are failures of the plumbing.

The diagram below is the mental model for the whole post: every operational loss a bank can suffer falls into one of seven buckets, and each bucket has a real, named disaster attached to it. Keep it in mind as we go, because the entire discipline of operational-risk management is the work of stopping those buckets from filling.

![The seven Basel operational-risk event types with examples](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-1.png)

## Foundations: what operational risk actually is

Let's build this from zero, because the term "operational risk" is one of those phrases that sounds precise and means almost nothing until you unpack it.

### The plain-English definition

Take a bank and compare it to a corner shop that also lends out the cash in its till. The shop makes money two ways that carry obvious risk: it lends money (and the borrower might not repay — that's *credit risk*) and it might hold goods whose value swings (that's *market risk*). But there's a third way the shop can lose money that has nothing to do with either: the till could be robbed, the shopkeeper could pocket cash, the card machine could break on the busiest day of the year, or the shop could accidentally sell expired food and get sued. None of those are about lending or prices. They are about the shop failing to *operate* properly.

That third category is operational risk. The formal Basel definition — the one regulators use — is worth reading slowly:

> Operational risk is the risk of loss resulting from inadequate or failed internal processes, people, and systems, or from external events.

Four sources, then: **processes** (the workflow breaks), **people** (someone is dishonest, careless, or simply makes a mistake), **systems** (the technology fails), and **external events** (something outside the bank — a fraudster, a flood, a hacker — causes the loss). The definition deliberately includes *legal risk* (lawsuits, fines, regulatory penalties) but deliberately *excludes* strategic risk (choosing the wrong business to be in) and reputational risk (losing customers because people stop trusting you) — though as we'll see, a big operational loss almost always drags reputation down with it.

A *basis point*, while we're defining terms, is one hundredth of a percent (0.01%), and you'll see it again later. *Notional* means the headline size of a trade or exposure. *Counterparty* means the other party to a deal. We'll gloss the rest as they come.

### Why it's the hardest of the four risks to measure

Here is the thing that makes operational risk genuinely difficult, and it's worth sitting with. Credit risk and market risk both have a natural unit of measurement. For credit risk you can look at a borrower and estimate a *probability of default*. For market risk you can look at a position and ask how much it moves when prices move. Both are, in a rough way, *predictable* — they have a price, a position size, a history.

Operational risk has none of that. There is no "position" in fraud. There is no notional amount of "settlement error" the bank chose to take on. The losses are not the result of a deliberate bet, so they don't scale with anything obvious. A tiny bank can suffer a giant cyber loss; a giant bank can go years without a major incident and then lose billions in a single rogue-trading event. The losses are **rare, lumpy, and driven by things the bank specifically tried to prevent**. You are, in effect, trying to put a capital number on the failures of your own controls — which is like trying to budget for the burglaries that get past your locks.

This is why operational risk is, for most large banks, one of the *largest* single components of their required capital — often a fifth to a quarter of the total. It is not a footnote. The bank holds real equity against the possibility that its own machinery breaks.

### The seven event types

Basel's great contribution was to refuse to treat "operational risk" as one undifferentiated blob. Instead it carved the universe of operational losses into seven mutually exclusive *event types*, so that a bank could collect data on each, see where its losses actually came from, and hold capital accordingly. They are:

1. **Internal fraud** — losses from deliberate dishonesty *inside* the bank: a rogue trader hiding positions, an employee stealing, books being falsified. Rare, but this bucket holds the biggest single bank-killers.
2. **External fraud** — deliberate dishonesty from *outside*: cheque forgery, card fraud, hacking, cyber heists, account takeover. High volume, growing fast.
3. **Clients, products, and business practices** — harming customers or breaching conduct rules: mis-selling, market manipulation, money-laundering failures, breaches of fiduciary duty. This is the *conduct* bucket, and in the last fifteen years it has produced the largest fines.
4. **Employment practices and workplace safety** — discrimination, harassment, unsafe conditions, wrongful-termination suits. Usually smaller in money terms, but real.
5. **Damage to physical assets** — fire, flood, earthquake, terrorism, vandalism. The 9/11 attacks sit here; so do natural-disaster losses.
6. **Business disruption and system failures** — IT outages, power loss, telecoms failure, a botched system migration. Not fraud, just things breaking.
7. **Execution, delivery, and process management** — the workhorse bucket: failed settlements, data-entry errors, fat-finger trades, missed deadlines, mis-booked transactions, documentation failures. Individually cheap, collectively enormous in *frequency*.

A useful way to remember the shape of the data: **buckets 1 and 3 (internal fraud, conduct) produce the biggest single losses; buckets 2 and 7 (external fraud, execution errors) produce the most frequent ones.** The capital math, which we'll get to, has to respect both shapes at once.

This is the whole spine of the series in miniature. A bank is a leveraged, confidence-funded machine: it survives only as long as depositors trust it and its thin equity absorbs losses faster than they arrive. Operational risk attacks *both* pillars at once — a big loss eats the equity directly, and the *kind* of loss (fraud, mis-selling) eats the trust that funds the bank. That double hit is why an operational disaster can kill a bank that a pure market loss of the same size would not.

## The shape of operational losses: why the tail is everything

Before any capital math, you have to understand the *shape* of operational losses, because that shape is unlike anything in the rest of banking.

### Many tiny, a few enormous

Picture a year of operational losses at a large bank, sorted by size. At the small end there are thousands of them: a few hundred dollars here for a mis-keyed payment, a thousand there for a duplicated transaction, a refund for a billing error. These happen every single day. They are annoying, they are budgeted for, and individually nobody loses sleep over them. This is the **body** of the distribution — high frequency, low severity.

Then, far out to the right, there is the **tail**: the once-in-a-decade rogue-trading loss, the cyber heist, the multi-billion conduct settlement. Vanishingly rare — maybe one in a thousand events, maybe one in a million — but each one is large enough to threaten the bank's existence. High severity, low frequency.

The chart below shows this shape with illustrative numbers. Notice the y-axis is a *log scale*: each step down the bars is roughly a tenfold drop in how often the loss happens. Thousands of sub-\$10,000 events; a single event above \$100 million.

![Operational losses are fat-tailed with rare huge events](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-3.png)

A distribution shaped like this — overwhelmingly small events with a long, thin, heavy tail of catastrophes — is called **fat-tailed** or **heavy-tailed**. And it changes everything about how you manage the risk. With a normal, bell-shaped distribution (like daily stock returns over a calm year), the average tells you almost everything: most outcomes cluster near the middle, and extremes are genuinely negligible. With a fat-tailed distribution, the average is a *liar*. The bank could go nine years averaging \$20 million of operational losses a year, look completely under control, and then lose \$2 billion in year ten. The mean of those ten years is dominated by the one event that the first nine gave no hint of.

### Expected loss versus tail loss

This forces a distinction that runs through the whole post. We split operational loss into two pieces:

- **Expected loss (EL)** — the average annual loss you can confidently predict will happen. The body of the distribution. The cost of doing business. You don't hold *capital* against expected loss; you *price* it into products and *provision* for it as an operating cost, the same way a retailer budgets for shoplifting.
- **Unexpected loss (UL)** — the gap between a *bad* year and an average year. The tail. This is what *capital* exists for. Equity is the buffer that lets the bank absorb a year far worse than average without going insolvent.

The whole game of operational-risk capital is: estimate the *full distribution* of possible annual losses, then hold enough equity to survive a year out at the far tail — conventionally the 99.9th percentile, the one-in-a-thousand-years bad year. Let's make that concrete.

#### Worked example: expected versus tail operational loss

Suppose a mid-sized bank studies five years of its own loss data and models its annual operational losses. It finds:

- **Expected loss** — about \$30 million a year, on average. This is the body: routine errors, small frauds, the steady drip.
- The **99.9th percentile** of the annual-loss distribution — the level a truly terrible year could reach once in a thousand years — comes out at \$430 million.

The bank does not hold capital against the \$30 million; that's just absorbed in the cost base, the way you'd treat any predictable expense. What it must hold capital against is the *unexpected* loss — the distance from the average year to the catastrophic year:

$$\text{Operational risk capital} = \text{UL} = \$430\text{m} - \$30\text{m} = \$400\text{ million}$$

So this bank earmarks \$400 million of its precious equity purely to absorb a one-in-a-thousand-years operational catastrophe — a rogue trader, a cyber heist, a giant conduct fine. Note what that \$400 million is *not* doing: it is not earning a lending spread, not funding a trade. It is sitting idle as a shock absorber against the bank's own machinery breaking. The intuition: **operational-risk capital is the price of the tail you hope never to see, and for a large bank it is one of the biggest single line items of capital it holds.**

### The loss-distribution approach

The method that produces a number like that \$400 million has a name: the **loss-distribution approach (LDA)**. It is the most rigorous way banks model operational risk, and although the precise regulatory formula has changed over the years, the LDA logic is how risk teams actually *think*, so it's worth understanding plainly.

The LDA builds the annual loss distribution from two separate ingredients, estimated for each event type:

1. **A frequency distribution** — how *many* loss events of this type happen in a year. For high-volume buckets like execution errors, this might be thousands; for internal fraud, it might be a fraction of one. Frequency is usually modeled with a Poisson distribution, which is just the standard way to count rare, independent events ("on average λ events per year").
2. **A severity distribution** — given that a loss happens, how *big* is it? This is the fat-tailed piece: most are small, but the distribution has a long right tail (often a lognormal body with a heavier-tailed extension for the extremes).

You then combine them — frequency × severity, simulated many thousands of times — to get the **aggregate annual loss distribution**: the full view of "what could a whole year of operational losses look like?" From that simulated distribution you read off the average (expected loss) and the 99.9th percentile (the tail), and the capital is the gap between them, exactly as in the worked example.

#### Worked example: building an annual loss number from frequency and severity

Let's do the simplest possible version by hand, to demystify it. Take one event type — say, external card fraud — at a small bank.

- **Frequency:** on average, the bank suffers 200 card-fraud events a year.
- **Severity:** each event costs, on average, \$2,000, but the costs are fat-tailed — most are a few hundred dollars, a handful are tens of thousands.

The *expected* annual loss from this bucket is simply frequency × average severity:

$$\text{EL} = 200 \times \$2{,}000 = \$400{,}000 \text{ per year}$$

That \$400,000 is the body — the bank prices it into card fees and treats it as a cost. But the bank also wants to know the *bad* year. Because both the count and the size of frauds vary, a simulation might show that one year in twenty (the 95th percentile), the total reaches \$900,000 — more than double the average — because the bank happened to suffer both more frauds *and* a couple of unusually large ones in the same year. The capital-relevant number is the *unexpected* part: \$900,000 − \$400,000 = \$500,000 set aside to survive a bad fraud year. The intuition: **even in a "well-behaved" bucket, the bad year is double the average, and capital is sized for the bad year, not the typical one.**

### The simpler regulatory formula

Modeling every bank's bespoke loss distribution turned out to be fragile — banks got wildly different capital numbers for similar risk, and regulators couldn't compare them. So under the latest Basel rules, supervisors largely *replaced* bespoke internal models with a **standardized approach** that ties operational-risk capital to two things: the size of the bank's *business* (a proxy called the Business Indicator, built from income and fees — bigger banks have more to go wrong) and the bank's *own historical losses* (a multiplier called the Loss Component, so a bank with a worse track record holds more).

The deep idea survives the simplification: **your operational-risk capital scales with how big you are and how badly you've failed before.** A bank that keeps having incidents pays for it in capital, year after year, which is the regulator's way of making good controls financially worth it.

### Scenario analysis: the data you wish you didn't have

There is a deep problem hiding inside everything we've said so far. To model a fat-tailed distribution, you need data about the tail — but the tail is, by definition, the part you have almost no data on. A bank might have collected a hundred thousand small loss events over a decade and *zero* events above \$500 million, simply because it has been lucky. Fitting a distribution to "lots of small, none enormous" will badly *underestimate* the tail, because the model has never seen the catastrophe it's supposed to protect against. This is the central tension of operational-risk measurement: the events that matter most are the ones you have the least data on.

The answer is **scenario analysis** — a structured way of inventing tail data that the bank's own history hasn't yet supplied. Risk teams sit down with the business and ask, deliberately and uncomfortably: *what is the worst plausible operational event in each category, and how big could it be?* What if our top trader turned out to be a fraud? What if our core payment system was down for a week? What if a regulator found systematic mis-selling across a whole product line? Each scenario gets an estimated frequency ("once in fifty years") and a severity range, and those expert estimates are blended into the loss distribution to fatten its tail to something realistic.

Two industry tools make this less arbitrary than it sounds. First, banks pool their loss data through consortia (the largest is called ORX), so a bank can see the tail events that happened to its *peers* even if they haven't happened to it yet — borrowing other people's catastrophes to calibrate its own model. Second, the public record of disasters — Barings, Société Générale, the cyber heists — gives every bank a library of "this is how big a rogue trader can get" that no single firm's internal data would ever contain. The discipline is to treat someone else's loss event as your own near-miss, and size your tail accordingly.

#### Worked example: scenario analysis fattening the tail

Suppose a bank's *internal* data alone, fitted naively, suggests its 99.9th-percentile annual loss is \$200 million — because the worst single event it has ever personally suffered is \$40 million. Its risk team runs a scenario workshop and asks the rogue-trader question. Looking at the public record — Barings at \$1.3bn, UBS's Adoboli at \$2.3bn, Société Générale's Kerviel at \$4.9bn — they conclude that for a bank of their size, a once-in-thirty-years rogue-trading event could plausibly reach **\$600 million**.

Feeding that single scenario into the model triples the tail: the 99.9th percentile jumps from \$200m to roughly \$600m, and the required capital rises with it. The bank now holds capital for a loss far larger than anything in its own history.

$$\text{Internal-data tail} = \$200\text{m}; \quad \text{With scenario} \approx \$600\text{m}; \quad \text{capital} \uparrow 3\times$$

The intuition: **for operational risk, the absence of a catastrophe in your own history is not evidence you're safe; it's a gap in your data that scenario analysis exists to fill with other people's disasters.**

### How operational risk compounds with the other three risks

A subtle point that separates people who understand banks from people who've only read the textbook: operational risk rarely arrives alone. It is the risk that *amplifies* the other three, and the worst banking disasters are almost always an operational failure layered on top of a credit, market, or liquidity problem.

Consider how it compounds. A market loss is survivable if the bank's risk systems *see* it and the bank can react — but if an operational failure (a rogue trader, a broken risk model, a system that under-reports positions) hides the market loss, it grows unbounded before anyone can act. That's exactly the Barings and Société Générale pattern: the *market* moved against the trader, but it was the *operational* failure of concealment that turned a bad bet into a fatal one. The market loss was the spark; the control failure was the accelerant.

The same compounding shows up in credit and liquidity. Sloppy loan documentation — an operational failure — turns a recoverable defaulted loan into an unrecoverable one because the bank can't enforce its claim on the collateral. A cyber attack or a system outage during a stressed market — operational risk meeting liquidity risk — can stop a bank from moving cash exactly when it most needs to, turning an awkward day into a run. The lesson is that you cannot rank the four risks in isolation: operational risk is the multiplier that decides whether a problem in one of the other three stays contained or spirals. A bank with weak operational controls is more fragile to *every* other risk, not just to fraud.

## The controls: how you actually stop the losses

Capital absorbs the loss *after* it happens. But the real work of operational-risk management is *preventing* the loss, and prevention is about controls. This is where the discipline gets concrete and, frankly, where most of the value is.

### Segregation of duties: the one control that matters most

If you remember one control from this entire post, make it this one: **segregation of duties** — also called the separation of front office and back office, or the maker-checker principle.

The idea is almost embarrassingly simple. No single person should be able to *both* do a thing *and* check that it was done correctly. The person who places a trade should not be the person who confirms and settles it. The person who approves a payment should not be the person who creates it. The person who writes the code should not be the only person who reviews it. The moment one individual controls the whole chain, there is no independent check — and where there's no check, an error or a lie has nothing to stop it.

The chart below tells the whole story of why this control is load-bearing, using the most famous operational failure in banking history.

![One missing control leads to two different outcomes at a bank](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-5.png)

At Barings, Nick Leeson ran the trading desk in Singapore *and* the back office that was supposed to settle and check those trades. Because he controlled both, he could book his mounting losses into a hidden error account — the now-infamous account number 88888 — and no independent clerk ever tried to confirm those trades with the actual counterparties. Had the two roles been split, the very first check would have caught it: a clerk trying to confirm a position that did not exist would have flagged the discrepancy the next morning. The fraud could not have survived a single day of genuine segregation. Instead it survived nearly three years and grew to \$1.3 billion — enough to bankrupt the bank.

#### Worked example: the cost of a control failure

Let's price the missing control at Barings, because it's the cleanest possible case study of operational risk.

The control that was absent — splitting the trader from the settlement clerk — would have cost Barings roughly the salary of one additional back-office supervisor in Singapore. Call it generously \$100,000 a year.

The loss when the control was absent was **£827 million**, about **\$1.3 billion** at the time. The bank's entire capital was around £350 million — *less than half the loss.* So the loss didn't just dent the equity; it exceeded it nearly two-to-one. There was no buffer large enough. The bank was insolvent the moment the true number surfaced.

$$\text{Loss} = \$1.3\text{bn}; \quad \text{Bank's capital} \approx \$540\text{m}; \quad \text{Loss} \approx 2.4 \times \text{capital}$$

The return on that \$100,000 control would have been, in pure avoided-loss terms, on the order of **13,000-to-1** — and that's before counting the 233-year-old franchise that vanished with it. The intuition: **operational controls look like pure cost on a calm day, which is exactly why they get cut; their value is the catastrophe they silently prevent, and you only ever see that value when one is missing.**

### Limits and four-eyes

Beyond segregation, two more everyday controls do enormous work:

- **Limits** — hard caps on what any individual or desk can do without escalation: a maximum position size, a maximum payment a single approver can release, a maximum loss before a trader must be flagged. Leeson's positions blew through any reasonable limit; the limits either didn't exist for his book or weren't enforced because he also controlled the reporting.
- **Four-eyes (dual authorization)** — for anything material, two people must independently approve. The second pair of eyes is cheap and astonishingly effective: most fat-finger errors and a lot of internal fraud die at the second signature.

### Key risk indicators: watching the gauges

Here is the deepest idea in operational-risk *prevention*, and it deserves its own section.

You cannot watch for the loss itself — by the time the loss event happens, it's too late. So instead you watch for the *conditions* that tend to precede losses. A **key risk indicator (KRI)** is a number you can measure every day or every week that tends to *rise before the loss does*. It's the smoke detector, not the fire.

Think about what precedes operational disasters and you can almost design the KRIs yourself. Before a settlement-driven loss, the rate of *failed settlements* usually creeps up. Before a fraud, *unreviewed access rights* and *manual overrides* pile up. Before a cyber breach, the *phishing click rate* in staff tests is high. Before a rogue-trading blowup, *limit breaches* become routine and *overdue confirmations* accumulate. Each of these is measurable today, long before the loss.

The trick that makes a KRI actionable is the **threshold**: a green band where the number is normal and nobody needs to do anything, an amber band where someone needs to watch and ask questions, and a red band that forces escalation to senior management. The matrix below shows a realistic set.

![Key risk indicators with green amber and red thresholds](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-8.png)

#### Worked example: a KRI threshold in action

Take the "failed settlements" KRI from the matrix. A bank settles 50,000 trades a day. Its threshold bands are: green below 1% failed, amber 1–3%, red above 3%.

On a normal day, 200 trades fail to settle on time — that's 200 / 50,000 = **0.4%**, comfortably green. Nobody acts; this is the body of the distribution, the routine noise.

Then, over two weeks, the failure rate drifts: 0.4%, 0.7%, 1.1%, 1.6%, 2.4%. The bank has crossed from green into amber and is climbing toward red. At 2.4% — that's 1,200 failed settlements a day — the operations head is required to investigate. The investigation finds a new counterparty system that's silently rejecting a class of trades, creating a growing pile of unsettled positions and the operational and funding risk that comes with them. Caught at 2.4%, it's a fixable process problem costing maybe a few hundred thousand dollars in funding and remediation. Left until it hit, say, 8%, it could have meant a multi-day backlog, a regulatory breach, and a loss an order of magnitude larger.

$$\text{Daily fails at 0.4\%} = 200; \quad \text{at 2.4\%} = 1{,}200; \quad \text{the KRI tripled before any loss crystallized}$$

The intuition: **a KRI converts a future loss into a present, watchable number, so you act on the smoke instead of the fire — which is the only way to manage a risk whose actual losses are too rare to learn from one at a time.**

### Cyber risk: the fastest-growing tail

Cyber deserves its own treatment because it is the one operational-risk category that is *growing* rather than merely recurring, and because it bends the usual fat-tail logic in a dangerous direction.

Most operational risks are, in a rough sense, *self-limiting*. A rogue trader is constrained by how much margin he can hide; a settlement error is bounded by the size of the trades that flow through one desk; a mis-selling scandal is capped by the number of customers in a product line. Cyber risk has no such natural ceiling. A single intrusion can, in principle, touch every account, every payment, and every record the bank holds, simultaneously and at machine speed. The attacker doesn't need to be at the bank, doesn't need physical access, and can hit thousands of banks with the same tool. That makes the cyber severity distribution potentially *fatter-tailed* than any other operational category — the worst plausible outcome is not "a desk loses money" but "the bank cannot operate at all."

Cyber losses come in three broad shapes, and it helps to keep them distinct. The first is **direct theft** — money moved out, as in the Bangladesh Bank heist. The second is **disruption** — ransomware or a denial-of-service attack that takes the bank offline, where the loss is lost business, remediation, and the run risk of customers who can't reach their money. The third, and often the largest, is **data breach** — the theft of customer data, where the loss is regulatory fines, lawsuits, notification costs, and years of reputational damage. A single sophisticated attack can trigger all three at once.

What makes cyber a genuine *operational*-risk discipline rather than an IT one is that the controls which actually limit the loss are the same operational controls we've already met, applied to the digital plumbing: segregation of duties (no one person can both initiate and release a large payment), four-eyes on anomalous transactions, limits (caps on how much can move before human escalation), and KRIs (the phishing click-rate, the count of overdue access reviews, the number of privileged accounts). The firewall is the lock on the door; the operational controls are what stop the loss once someone gets through the door — and someone, eventually, always does.

#### Worked example: sizing a cyber scenario

A bank runs a scenario for a ransomware attack that takes its retail systems down for three days. It estimates: lost transaction revenue of \$15 million, remediation and forensic costs of \$40 million, regulatory fines and customer compensation of \$70 million, and a harder-to-quantify customer-attrition cost it conservatively pegs at \$25 million. The total scenario loss:

$$\$15\text{m} + \$40\text{m} + \$70\text{m} + \$25\text{m} = \$150 \text{ million}$$

The bank then asks the survival question: against \$30 billion of equity, a \$150 million cyber event is a 0.5% dent — painful but absorbable. But the *run risk* is the real tail: if three days offline triggered even a modest deposit flight, the operational loss could ignite a liquidity crisis far larger than the \$150 million direct cost. The intuition: **cyber risk is dangerous less for its direct dollar loss than for its power to break the confidence and the plumbing that the whole bank runs on — which is why it sits at the top of every major bank's operational-risk register.**

### The three lines of defense

Stack all of this together and you get the standard governance model for operational risk: the **three lines of defense**. It's the organizational answer to "who is responsible for stopping operational losses?" — and the answer is, deliberately, *three independent groups*, so that any single failure has to slip past all three.

![The three lines of defense against operational risk](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-4.png)

- **Line 1 — the business.** The people doing the actual work *own* the risk. The trading desk, the payments team, the lending officers. They run the controls day to day because they're closest to where things break. The first principle of operational risk is that the business that *takes* the risk is responsible for *managing* it — not some distant risk department.
- **Line 2 — risk and compliance.** An *independent* function that sets the limits and policies, challenges the business, monitors the KRIs, and has the authority to say no. They don't do the trades; they make sure line 1 is running its controls properly. Crucially, they report separately, so they can't be silenced by the people they oversee.
- **Line 3 — internal audit.** A second independent function whose only job is to check that lines 1 and 2 are actually working — not just on paper. Audit reports to the board, not to management, so it can tell uncomfortable truths.

The whole point is *independence stacked three deep*. At Barings, all three lines collapsed into one man: Leeson was line 1 (he traded), he was effectively line 2 (he controlled the local reporting), and the London oversight that should have been line 3 didn't understand the Singapore operation well enough to challenge it. With the defenses merged, there was nothing to slip past. The model only works when the layers are genuinely separate — which is exactly the lesson banks paid £827 million to learn.

### Risk appetite, transfer, and what you can't insure away

Two more ideas complete the practitioner's toolkit, and they're the ones that turn operational risk from a measurement exercise into a set of decisions.

The first is **operational-risk appetite**. A bank cannot eliminate operational risk — to make it zero you'd have to stop operating. So the board sets an explicit *appetite*: how much operational loss the bank is willing to tolerate, expressed as limits and KRI thresholds across the seven event types. Appetite is the bridge between the abstract loss distribution and daily decisions, because it tells a business unit, in advance, when a risk has grown past what the bank will accept and must be escalated, mitigated, or stopped. A bank with no stated appetite manages operational risk by reaction; a bank with a clear one manages it by design.

The second is **risk transfer** — the insurance question. Banks do buy insurance against some operational losses: fidelity bonds against employee theft, cyber-insurance policies, professional-indemnity cover, directors-and-officers cover. Insurance is genuinely useful for the *frequent, mid-sized* losses — it smooths the body of the distribution and converts a lumpy cost into a steady premium. But it runs into a hard wall at the tail, and that wall is the deepest lesson of the whole field: **you cannot insure away the catastrophe that kills you.** Insurers cap their payouts, exclude the very largest events, dispute claims when the loss is enormous, and would themselves be threatened if they had to pay out on a true tail event. No insurer was going to write Barings a policy that paid out \$1.3 billion the day it became insolvent. The catastrophic tail is, by its nature, *self-insured* — which is precisely why the bank must hold *capital* against it and, far more importantly, must run the *controls* that stop it from happening at all.

#### Worked example: insurance versus the tail

A bank pays \$8 million a year for an operational-risk insurance program with a \$50 million cap per event. In a normal year, the insurance covers \$6 million of mid-sized losses — fidelity claims, a small cyber incident, a settlement error — so the program is roughly break-even and smooths the bank's loss experience. Useful.

Then a true tail event hits: a \$900 million rogue-trading loss. The insurer pays its \$50 million cap and not a cent more. The bank absorbs \$850 million from its own capital.

$$\text{Insurance recovery} = \$50\text{m (the cap)}; \quad \text{Self-insured by the bank} = \$900\text{m} - \$50\text{m} = \$850 \text{ million}$$

The insurance covered 5.6% of the loss that actually mattered. The intuition: **insurance is a tool for the body of the distribution, never the tail; the tail is covered by capital and prevented by controls, and any bank that thinks it has "insured away" its operational risk has misunderstood where the danger lives.**

## Common misconceptions

A handful of beliefs about operational risk are not just wrong but *dangerously* wrong, because they lead banks to under-invest in exactly the controls that matter.

**"Operational risk is a back-office, low-stakes problem."** It is the opposite. Look at the loss league table below: the biggest operational losses dwarf most credit and market losses, and they're the ones that actually *kill* banks outright. Barings died from operational risk. The largest fines in banking history — LIBOR rigging at roughly \$9 billion across the industry, Wells Fargo at \$4.9 billion, 1MDB at around \$5 billion for Goldman — are all operational (specifically conduct) losses, not credit or market ones. Treating it as a back-office concern is how you end up as the next case study.

![The big operational-risk loss events in billions of dollars](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-2.png)

**"If our average operational losses are low, we're safe."** This is the fat-tail trap, and it's the single most expensive misconception in the field. As we saw, the average is meaningless for a heavy-tailed distribution. A bank can average \$20 million a year for nine years and lose \$2 billion in year ten. The nine quiet years are not evidence of safety; they're just the part of the distribution before the tail event. Low average losses can even be a *warning* sign — if a bank reports almost no operational losses, it more likely has weak *detection* than strong controls.

**"More capital is the answer to operational risk."** Capital absorbs the loss; it does not prevent it. And you cannot hold enough capital to make yourself indifferent to a fraud — the tail is too long. The answer to operational risk is overwhelmingly *controls*, not capital. A bank that responds to a fraud by holding more equity and not fixing the broken control has learned nothing; it will simply suffer the same fraud again, now with a bigger buffer to bleed through.

**"It won't happen to a well-run bank."** Every bank on the loss league table thought it was well-run. Barings was 233 years old and blue-blooded. Wells Fargo was, before the scandal, considered one of the best-managed banks in America, admired for its sales culture — which turned out to be the very thing that drove the fraud. Operational risk is not a sign of a bad bank; it's a sign of a *complex* one, and all big banks are complex. The difference between the survivors and the casualties is not the absence of incidents but the presence of controls that catch them small.

**"Cyber is just an IT problem."** Cyber risk lives in operational risk precisely because the loss is operational, not technical. When the Bangladesh central bank lost \$81 million to hackers in 2016, the technical breach was the entry point, but the *loss* happened because of broken *processes*: weak payment-authorization controls, a printer that conveniently failed so the fraudulent confirmations didn't print, and SWIFT messages that no human independently checked. Treating cyber as the IT team's problem misses that the controls which actually stop the loss — segregation, four-eyes, anomaly monitoring — are operational controls, not firewalls.

## How it shows up in real banks

Operational risk is best understood through the disasters it has caused. Each of these is a different event type from the seven, and each teaches a specific lesson about which control was missing.

### Barings, 1995 — internal fraud and the rogue trader

We've used Barings as the running example, but it's worth telling straight because it is the *archetype* of internal fraud. Nick Leeson, a 28-year-old trader in Barings' Singapore office, was supposed to be running a low-risk arbitrage business — buying a futures contract on one exchange and selling the same contract slightly more expensively on another, pocketing tiny risk-free differences. Boring, safe, profitable in small amounts.

Instead, Leeson started taking large directional bets on the Japanese stock market. When they lost, he hid the losses in error account 88888, a number he chose because his fortune teller called eight a lucky number. Because he *also* controlled the back office, he could falsify the reports that went to London, fabricate documents, and even request more funding from head office to meet the margin calls on positions London didn't know existed. The Kobe earthquake in January 1995 sent the Japanese market down hard, his bet imploded, and the hidden loss exploded to **£827 million** — roughly **\$1.3 billion** — against a bank with about £350 million of capital. There was no recovering from a loss more than double your equity. Barings was declared insolvent on 26 February 1995 and sold for £1. The event type: internal fraud. The missing control: segregation of duties.

### Wells Fargo, 2016–2022 — conduct, and the price of bad incentives

Wells Fargo is the canonical *conduct* failure — Basel event type three, clients and business practices — and it shows that operational risk doesn't need a rogue genius; it can be manufactured at industrial scale by an incentive system.

Wells Fargo set its retail staff aggressive cross-selling targets: each customer should hold a high number of products — checking, savings, credit cards, and more. The targets were unrealistic, and missing them threatened people's jobs. So employees, under pressure, did the rational-but-fraudulent thing: they opened accounts customers never asked for. Around **3.5 million** fake or unauthorized accounts were created. Customers were charged fees on accounts they didn't know existed.

The penalties came in waves: an initial **\$185 million** settlement in 2016, a **\$3 billion** Department of Justice and SEC settlement in 2020, and a further **\$1.7 billion** Consumer Financial Protection Bureau fine in 2022 — roughly **\$4.9 billion** in total, plus an unprecedented regulatory cap on the bank's *growth* that constrained it for years. The chart below puts the Wells Fargo penalties beside the Barings loss to make the central point.

![Wells Fargo fines compared with the Barings loss](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-6.png)

The contrast is the lesson. Wells Fargo paid more in fines than Barings lost in total — yet Wells Fargo survived and Barings died. Why? Because Wells Fargo is a giant: \$4.9 billion of fines, painful as it was, fit inside its enormous equity base. Barings' \$1.3 billion loss did not fit inside its \$540 million of capital. The same dollar loss is survivable or fatal depending entirely on whether it outruns the equity cushion — which is the spine of this whole series. The event type: conduct. The missing control: an incentive system that didn't pressure-test for fraud, and a line-2 function that didn't challenge it.

#### Worked example: a fraud loss and whether the bank survives it

Let's make the survival arithmetic explicit, because it's the most important number in operational risk.

Imagine two banks each suffer an identical operational loss of \$1.3 billion — Barings' number.

- **Small bank:** equity of \$540 million (Barings' actual figure). The loss is \$1.3bn. After the loss, equity = \$540m − \$1,300m = **−\$760 million**. Negative equity means insolvent: liabilities exceed assets, depositors and creditors cannot all be paid. The bank is dead.
- **Large bank:** equity of \$180 billion (roughly a large US bank today). The same \$1.3bn loss takes equity to \$180bn − \$1.3bn = **\$178.7 billion** — a 0.7% dent. It's a bad headline and a bad quarter, but the bank is completely intact.

$$\text{Small: } \frac{\$1.3\text{bn loss}}{\$0.54\text{bn equity}} = 2.4\times \text{ (fatal)}; \qquad \text{Large: } \frac{\$1.3\text{bn}}{\$180\text{bn}} = 0.7\% \text{ (survivable)}$$

The intuition: **operational risk doesn't kill banks in absolute dollars; it kills them relative to equity. The same fraud is a rounding error at one bank and an extinction event at another, which is exactly why thinly capitalized firms must run the tightest controls.**

### Cyber heists — external fraud at machine speed

The Bangladesh Bank heist of February 2016 is the modern textbook case of external fraud (event type two). Attackers, having compromised the bank's systems, sent fraudulent SWIFT payment instructions to move nearly \$1 billion out of the Bangladesh central bank's account at the New York Federal Reserve. Most transfers were blocked — partly by a lucky spelling error in one instruction that flagged a recipient — but **\$81 million** got through to casinos in the Philippines and was largely never recovered.

What makes it an *operational* loss and not merely a hacking story is *how* the \$81 million escaped: the controls that should have caught it were operational ones that failed. Payment instructions weren't independently verified by a second human (no four-eyes). A printer that automatically logged transaction confirmations had been disabled, so staff didn't see the fraudulent payments for days. The timing exploited a weekend across multiple time zones. Cyber risk almost always works like this — the breach is technical, but the *loss* flows through broken processes, and so the defenses that matter are operational controls layered behind the firewall.

### Rogue traders after Barings — the lesson that didn't stick

Barings was supposed to be the end of rogue trading. It wasn't. In 2008, Société Générale disclosed a loss of roughly **€4.9 billion** from a single trader, Jérôme Kerviel, who built enormous unauthorized positions and hid them with fake offsetting trades — a near-replay of Leeson's mechanism, thirteen years later, at a far more sophisticated bank. In 2011, UBS lost about **\$2.3 billion** to trader Kweku Adoboli in London, again through unauthorized positions concealed in the system. In 2012, JPMorgan's "London Whale" lost around **\$6.2 billion** — not fraud exactly, but a derivatives book that grew far past its risk limits while internal controls and model checks failed to rein it in.

The recurring pattern across all of them: a trader who understood the bank's controls well enough to exploit their gaps, a back-office or risk function that didn't independently verify positions, and limits that were either absent or ignored. Each loss was an order of magnitude larger than Barings' — the trades got bigger as banks got bigger — and each one taught the industry, again, that the controls are only as good as their independence. The lesson Barings paid £827 million for keeps needing to be repaid.

### System failures — when the machinery simply stops

Not every operational loss is a fraud. In 2018, the UK bank TSB attempted to migrate its customers onto a new IT platform over a single weekend. The migration failed spectacularly: roughly 1.9 million customers were locked out of their accounts, some saw other people's account details, and the chaos lasted weeks. TSB ended up paying out and writing off well over **£330 million** in compensation, remediation, lost income, and fraud that exploited the confusion, and its chief executive resigned. No fraud, no rogue trader — just a botched process and a system change that wasn't adequately tested. Event type six, business disruption and systems failure. The missing control: a tested, reversible migration plan with the ability to roll back.

### The mechanism that links them all

Step back and every one of these disasters follows the same path: a small control gap that goes unnoticed, then unchallenged, then unbounded. The pipeline below traces it.

![How a small control gap becomes a catastrophic loss](/imgs/blogs/operational-risk-fraud-cyber-and-the-loss-events-7.png)

It always starts with a *gap* — a missing segregation, an unenforced limit, an untested system, a perverse incentive. The gap goes *unnoticed* because no KRI is watching it. Then *someone or something exploits it* — a trader who profits from hiding losses, a hacker who finds the weak process, a sales force pushed past honesty. The problem *compounds* — the hidden number grows daily because nothing is checking it. And then comes *discovery*, usually all at once and at the worst possible moment, when the loss has grown past anything the bank can absorb. The whole discipline of operational-risk management is about intervening early in that chain — closing the gap (controls), or at least spotting it while it's small (KRIs), so the loss never reaches the right-hand end.

#### Worked example: an operational-loss capital number for a real bank profile

Let's tie the capital math to a realistic bank to close the loop. Take a large bank with about \$2 trillion of assets. Suppose its operational-risk-weighted assets — the regulatory measure of its operational risk under the standardized approach — come to **\$250 billion** (operational risk is often a fifth or so of total risk-weighted assets at a large universal bank).

The bank must hold common equity (CET1) of at least, say, 11% against its risk-weighted assets once buffers are included. So the equity the bank holds *specifically to cover operational risk* is:

$$\text{Op-risk capital} = 11\% \times \$250\text{bn} = \$27.5 \text{ billion}$$

That \$27.5 billion is real equity, contributed by shareholders, earning no spread, sitting against the possibility of a rogue trader, a cyber heist, or a conduct fine. It is one of the largest single chunks of capital the bank holds — frequently more than it holds against market risk. The intuition: **operational risk is not a soft, qualitative concern; at a large bank it is a tens-of-billions-of-dollars line of hard capital, which is why the controls that reduce it pay for themselves many times over.**

## The takeaway: how to use this

Here is the thing to carry away, and it reframes how you should read any bank — as an analyst, an investor, a customer, or someone running one.

Operational risk is the purest expression of the series' central truth: **a bank is a leveraged, confidence-funded machine, and operational risk attacks both the leverage and the confidence at the same time.** A big operational loss eats the thin equity cushion directly — and because the losses are fat-tailed, a single tail event can eat more equity than years of profit built up. But operational risk does something credit and market risk usually don't: the *nature* of the loss — fraud, mis-selling, a cyber breach, a system collapse — directly damages the *trust* that funds the bank. Depositors forgive a bank that lost money on a bad loan; they are far less forgiving of a bank that lied to them or couldn't keep their money safe. That double hit, to capital and to confidence, is why operational disasters punch so far above their dollar weight.

So when you assess a bank, don't stop at its loan book and its trading positions. Ask the operational-risk questions, because they're the ones that produce the sudden, unbet-on, bank-killing losses:

- **Are duties genuinely segregated?** Can any one person both do a thing and check it? The single most predictive control.
- **What does the loss history actually show?** Not the average — the tail. Has this bank had near-misses? A pattern of small incidents is the body of a distribution whose tail hasn't arrived yet.
- **What are the incentives rewarding?** Wells Fargo's fraud was an incentive system working exactly as designed. If the targets can only be hit dishonestly, they will be.
- **Is the capital base large enough to absorb a tail event?** The same \$1.3 billion loss is a rounding error at a giant and an extinction event at a small bank. Operational risk kills *relative to equity*, never in absolute dollars.
- **Are the three lines of defense actually independent?** Or have they quietly collapsed into the people they're meant to oversee, the way they did at Barings?

The deepest insight is the one the Barings worked example made arithmetically: the controls that prevent operational losses look like pure cost on every single calm day — a back-office salary, a slower process, a target that's harder to hit honestly. That's precisely why they're the first things cut when a bank chases efficiency or growth. But their value is the catastrophe they silently prevent, and you only ever see that value when one is missing — by which point you're the case study. A bank that understands operational risk spends a little, visibly and continuously, to avoid losing everything, rarely and catastrophically. A bank that doesn't, eventually joins the league table.

*This is educational, not investment advice. The figures cited are drawn from public disclosures, regulatory orders, and press reports as of mid-2026; operational-loss totals and fines are often updated as legal proceedings conclude.*

## Further reading & cross-links

- [Bank Capital and Leverage: Why Equity Is the Thin Cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the equity buffer that absorbs operational losses, and why the same dollar loss is fatal or survivable depending on the cushion behind it.
- [The Trading Book: Market-Making, Flow vs Prop, and the Volcker Rule](/blog/trading/banking/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule) — where rogue trading lives, and the limit framework that's supposed to contain it.
- [Collateral, Security, and Loan-Loss Provisioning: IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) — the expected-versus-unexpected-loss logic, applied to credit, that mirrors the operational-loss capital split.
- [BIS and Basel: How Banks Are Regulated](/blog/trading/finance/bis-and-basel-bank-regulation) — the Basel framework that defines the seven event types and sets operational-risk capital.
- [SVB and Credit Suisse, 2023: The Anatomy of Two Bank Runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — how a loss of confidence, the second thing operational disasters destroy, becomes a run.

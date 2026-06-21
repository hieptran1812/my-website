---
title: "Credit Suisse 2023: The Slow Death of Trust and the AT1 Wipeout"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a 167-year-old bank died not overnight but over years, as scandal after scandal drained depositor trust until a forced UBS takeover wiped out 16 billion francs of bondholders while shareholders walked away with something."
tags: ["banking", "credit-suisse", "at1", "coco-bonds", "bank-failure", "ubs", "creditor-hierarchy", "deposit-run", "financial-crisis", "swiss-banking", "trust"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — Credit Suisse did not die from one bad bet; it died from a slow, years-long leak of trust that finally became a run nobody could stop, and its rescue rewrote the rules of who loses first.
>
> - The bank lost CHF 110 billion of deposits in a single quarter (Q4 2022) — roughly a third of its deposit base — long before the final weekend, because a decade of scandals (Archegos, Greensill, spying, fines) had drained confidence.
> - A CHF 100 billion emergency liquidity line from the Swiss National Bank did **not** stop the bleeding; once trust is gone, cheap central-bank cash is a bridge to nowhere.
> - On 19 March 2023 UBS was pushed to buy Credit Suisse for CHF 3 billion, and the regulator wrote CHF 16 billion of AT1 (CoCo) bonds to **zero** while equity holders still got value — inverting the usual order in which losses are supposed to fall.
> - The one number to remember: AT1 holders lost more than **five times** what UBS paid for the entire bank, a result that repriced the global CoCo market overnight.

In the early hours of Sunday, 19 March 2023, a small group of officials from the Swiss government, the Swiss National Bank, the regulator FINMA, and the two largest banks in Switzerland sat in a room in Bern and agreed to end the independent life of Credit Suisse. The bank was 167 years old. It had financed the railways that stitched Switzerland together in the 1850s. It had survived two world wars, the Great Depression, the 2008 crisis, and a hundred smaller storms. By Monday morning it would no longer exist as an independent firm.

What is strange about this story is that nothing actually *happened* that weekend. There was no rogue trader confessing a hidden loss. There was no single asset that suddenly turned worthless. The bank still had hundreds of billions of francs of assets, a fortress-sounding capital ratio on paper, and an emergency credit line from the central bank that could have covered a normal-sized panic ten times over. And yet by every measure that matters, it was already dead. The corpse had simply not been declared.

This is the defining feature of the Credit Suisse failure, and the reason it is worth a long, careful look. Silicon Valley Bank, which collapsed the same week, died the way a textbook says a bank dies: a clean, fast, mechanical death from a mismatch between the value of its bonds and the demands of its depositors, all over in about 36 hours. Credit Suisse died the other way — slowly, almost gently, by a thousand cuts, as the one thing a bank cannot survive without quietly drained away. That thing is trust. And once you understand how trust dies in a bank, and what happened to the people who thought they were *senior* in line when the music stopped, you understand something fundamental about what a bank actually is.

![Timeline of the Credit Suisse decline from 2021 to 2023 showing Archegos, Greensill, deposit flight, the SNB line, and the UBS deal](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-1.png)

The diagram above is the mental model for this whole post. Read it left to right and notice that there is no single tall spike — no one moment where the bank fell off a cliff. There is instead a staircase down: a loss here, a frozen fund there, a rumour, a quarter of deposit flight, an emergency line that buys a few days, and then the end. That staircase is what "loss of confidence" actually looks like in numbers. Our job in this post is to walk down every step.

## Foundations: what a bank really is, and why trust is the load-bearing wall

Before we can understand how Credit Suisse died, we need to be precise about what a bank *is*, because the popular view of it is wrong in a way that hides the whole drama.

Most people assume a bank is a kind of vault — a safe place where your money sits, waiting for you. It is not. A bank takes the money you deposit and lends almost all of it out, or invests it. The cash that is actually sitting in the bank at any moment is a small fraction of what it owes its depositors. This is not a scandal; it is the entire point. A bank's job is **maturity transformation**: it borrows short (your deposit, which you can take out any day) and lends long (a 30-year mortgage, a 5-year corporate loan), and it earns the gap between the low rate it pays you and the higher rate it charges the borrower. That gap is called the *spread*, and it is the bank's reason to exist. (We unpack this in detail in [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).)

Here is the consequence that matters for our story. Because the bank has lent out most of the cash, it **cannot** give every depositor their money back at once. It does not have it. It is relying on the fact that, on any normal day, only a few percent of depositors want their cash, and those are easily covered by the cash on hand plus new deposits coming in. The whole machine runs on a single, fragile assumption: that depositors believe they can get their money whenever they want, and therefore most of them never ask for it all at once.

### Franchise and confidence: the words to define first

Two words will recur throughout this post, so let us pin them down.

The **franchise** of a bank is its ongoing, going-forward business: the relationships, the deposit base, the trading clients, the wealth-management accounts, the reputation that makes a company want to do its currency hedging with *this* bank rather than another. The franchise is worth a great deal — far more than the bricks and the bond portfolio — precisely because it generates future profit. But the franchise has a peculiar property: it is made entirely of other people's willingness to keep dealing with you. It is not on the balance sheet as a number, but it is the most valuable thing the bank owns, and it can evaporate.

**Confidence** is the belief, held by depositors, lenders, trading counterparties, and clients, that the bank is safe to deal with. Confidence is the fuel the franchise runs on. A bank with confidence can borrow cheaply, keep its deposits, and price its services well. A bank that loses confidence finds its funding costs rising, its deposits leaving, and its clients quietly moving their business elsewhere — and *that* makes the bank weaker, which erodes confidence further. It is a feedback loop, and it can run in either direction. For most of a bank's life it runs in the good direction, invisibly. When it flips, it can run terrifyingly fast.

#### Worked example: how thin the cushion really is

Let us make "the cushion is thin" concrete with friendly numbers. Suppose a bank funds itself like a typical large lender: about 71% from deposits, 10% from wholesale and repo borrowing, 7% from long-term bonds, 4% from other sources, and about **8% from equity** (the shareholders' own money). That 8% equity is the cushion that absorbs losses before anyone who *lent* the bank money loses a cent.

Take a bank with \$100 billion of assets. Then it has roughly \$8 billion of equity and \$92 billion of money it owes to others. The ratio of assets to equity is \$100 billion ÷ \$8 billion = **12.5×** — this is the bank's *leverage*, meaning every \$1 of its own money supports \$12.50 of assets. Now suppose the assets lose just 8% of their value — \$8 billion. The entire equity cushion is gone. The bank is, on paper, insolvent: it owes \$92 billion and its assets are now worth \$92 billion, with nothing left over for the owners.

The intuition: a bank does not need to lose *most* of its money to fail. It needs to lose only the thin slice that is its own. An 8% loss wipes out a 12.5×-levered bank. (For the full mechanics of why equity is the cushion and how leverage cuts both ways, see [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).) This is why confidence is not a soft, fuzzy thing for a bank — it is the structural reason a bank can stand at all. The whole edifice is balanced on a sliver of equity and a much larger volume of trust.

### AT1, CoCos, and the capital stack

Now for the instrument at the centre of the controversy: the **AT1 bond**, also called a **CoCo** (short for "contingent convertible").

To explain it, we need the idea of a *capital stack* — the ordered layers of money that fund a bank, from the most protected at the bottom to the first-to-lose at the top. It works like a building where losses flood in from the top floor down. The order, from first-to-lose to last, is roughly:

1. **Equity (CET1, "Common Equity Tier 1")** — the shareholders' money. This takes the first loss. It can go all the way to zero.
2. **Additional Tier 1 (AT1 / CoCo bonds)** — a special kind of bond designed to absorb losses *after* equity, while the bank is still alive.
3. **Tier 2** — subordinated debt, more protected than AT1.
4. **Senior debt and depositors** — the most protected, supposed to lose last (and deposits are often insured up to a limit on top of that).

The AT1 bond was invented after the 2008 crisis as a clever fix to a real problem. In 2008, when banks ran out of equity, governments had to inject taxpayer money to keep them alive, because there was no other layer that could absorb losses *while the bank kept operating*. Regulators wanted a layer that would do that job automatically — a buffer that converts into fresh equity, or gets written off, the moment the bank gets into trouble, without a government cheque. That layer is AT1.

An AT1 bond pays a juicy coupon (often 7–10%) to compensate investors for a specific, written-into-the-contract risk: if the bank's capital falls below a *trigger* level, or if the regulator declares the bank unviable, the bond either **converts into shares** or is **written down to zero**. The investor knew this when they bought it. The high coupon was the price of bearing that risk. (The full evolution of these capital rules — Basel I, II, and III, and where AT1 fits — is laid out in [Basel I, II, III and the capital rules](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank).)

The crucial design promise — the thing the whole market believed — was about *order*. AT1 sits **above** equity in the stack, meaning equity should lose first and AT1 only after. As we will see, that promise is exactly what the Credit Suisse resolution appeared to break.

### Going-concern versus gone-concern capital

One more pair of terms, because the entire AT1 argument turns on them.

**Going-concern capital** is capital that absorbs losses while the bank is still *alive and operating* — a "going concern", in accountant's language. CET1 equity is the prime example. The point of going-concern capital is to soak up losses so the bank can keep running without ever reaching the point of failure.

**Gone-concern capital** is capital that absorbs losses once the bank has *failed* — once it is a "gone concern" being wound down or resolved. Its job is to take losses so that depositors and the wider system are protected during the wind-down.

Where does AT1 sit? This is the heart of the dispute. AT1 was sold to most investors as primarily *going-concern* capital — a buffer that would convert into equity to help the bank survive, with equity taking losses first. But the fine print of many AT1 contracts, including the Swiss ones, also allowed a full write-down at the *gone-concern* point, when the regulator declares the bank unviable. The two readings imply very different outcomes, and Credit Suisse is where they collided.

![Two-column comparison of the normal creditor loss order versus the inverted Credit Suisse outcome](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-3.png)

The figure above shows the two orders side by side. On the left is what every textbook, every investor presentation, and most investors' mental models said: equity loses first, then AT1, then Tier 2 and senior, with depositors most protected of all. On the right is what actually happened at Credit Suisse: AT1 written to zero while equity holders still received CHF 3 billion of UBS shares. Hold that contrast in your mind; the rest of the post explains how we got there and why it set off a legal firestorm.

## The scandals: a decade of paying for broken trust

If you want to understand why depositors eventually ran, you have to understand why they stopped trusting the bank in the first place. And that is not a 2023 story. It is a story that ran for years, scandal after scandal, each one chipping at the franchise.

A bank's most valuable asset is its reputation for being a safe, competent place to keep your money and do your business. Every scandal spends some of that reputation. A bank can survive one. It can probably survive two. What it cannot survive is a *pattern* — the slow accumulation of evidence that the institution cannot control its own risks. By the time 2022 arrived, Credit Suisse had built exactly such a pattern.

![Horizontal bar chart of Credit Suisse losses and settlements including Greensill, Archegos, and earlier fines](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-4.png)

### Archegos: the risk-management failure

In March 2021, a US family office called Archegos Capital Management collapsed. Archegos had taken enormous, concentrated, leveraged bets on a handful of stocks using derivatives that let it control huge positions while posting relatively little collateral. When those stocks fell, Archegos could not meet its margin calls, and the banks that had financed it had to dump the positions.

Most banks that dealt with Archegos got out fast and limited the damage. Credit Suisse did not. It was slow to react, it had under-margined the client, and it ended up taking a loss of roughly **CHF 5.5 billion** — by far the largest hit of any bank. An independent report commissioned by Credit Suisse itself later described a litany of risk-management failures: warnings ignored, limits not enforced, a culture that prioritised revenue over control.

The dollar figure was painful, but the *meaning* of the loss was worse. A loss of \$5.5 billion is recoverable. A demonstrated inability to manage the risk of a single client is corrosive, because it tells every other counterparty: *this bank does not have its hands on its own controls.* That is a franchise wound, not just a P&L wound.

It is worth dwelling on *why* a risk-management failure damages a bank more than the raw loss suggests. A bank's entire business is the pricing and management of risk — that is the product it sells. When a corporate treasurer chooses a bank to handle a currency hedge, or a pension fund picks a custodian, or another bank decides whether to lend overnight, they are making a single judgement: *can this institution be trusted to manage risk competently?* Archegos answered that question in the negative, in public, with a nine-figure loss. Every prime-brokerage client of Credit Suisse — the hedge funds whose trades the bank financed — now had to wonder whether their own positions were being managed by the same people who missed Archegos. Some did not wait to find out. The loss of those relationships, compounding quietly over the following months, was arguably costlier than the CHF 5.5 billion itself.

### Greensill: the asset-management failure

Almost simultaneously, in March 2021, a separate disaster unfolded. Credit Suisse ran a set of supply-chain finance funds — investment funds that bought short-term invoices and were sold to clients as safe, money-market-like products. The funds were heavily linked to a single firm, Greensill Capital, which packaged those invoices. When Greensill collapsed, Credit Suisse had to freeze the funds, which held about **\$10 billion** of client money.

This was not a loss on the bank's own balance sheet in the way Archegos was. But it was arguably worse for trust, because the people who lost money here were *clients* — the very wealthy individuals and institutions whose business is the heart of the franchise. They had been told these funds were safe. They were not. Recovering their money turned into a multi-year legal slog. Every client who lost money, and every client who heard about it, now had a reason to wonder what else they had been told that was not quite true.

### The spying scandal and the drip of fines

Layered on top were other episodes that, individually, a bank might shrug off, but collectively showed an institution out of control. In 2019, it emerged that Credit Suisse had hired private investigators to physically *spy* on a departing senior executive — a bizarre, almost farcical affair that nonetheless ended careers and made headlines worldwide. There was a long tail of regulatory penalties: a settlement over mis-sold US mortgage-backed securities from the pre-2008 era (around \$5.3 billion, agreed in 2017); a settlement over corruption-tainted "tuna bond" loans to Mozambique (around \$0.5 billion in 2021); and, in 2022, a Swiss criminal conviction for failing to prevent money laundering linked to a Bulgarian cocaine ring — the first such criminal conviction of a major Swiss bank.

None of these was, by itself, fatal. But step back and look at the chart of losses and settlements above. The point of the figure is not any single bar; it is the *accumulation*. Each settlement was a withdrawal from the trust account. By 2022 the account was nearly empty.

#### Worked example: scandals as a drain on the equity cushion

Let us put the scandals into the language of the capital cushion, because that is how a depositor or a rating agency reads them.

Recall our model bank with \$100 billion of assets and \$8 billion of equity. Now imagine it absorbs, over a couple of years, a \$5.5 billion trading loss (Archegos) plus, say, \$2 billion of fines and legal provisions. That is \$7.5 billion of losses against an \$8 billion cushion — it eats roughly **94%** of the bank's loss-absorbing equity if it all landed at once. In reality the losses were spread over time and partly offset by ongoing profits, so the bank was not literally wiped out. But the rating agencies and the market do not just look at the surviving equity; they look at the *trajectory*. A bank that keeps generating losses near the scale of its entire cushion is a bank whose cushion is being asked to do an impossible job.

The intuition: a single big loss is survivable if it is a one-off, but a *stream* of losses each comparable to a large share of your equity tells the market that the cushion is structurally too thin for the risks this bank takes. That perception is what raises funding costs and starts the deposits walking — long before any single loss actually breaks the bank.

## The bleed: when clients vote with their feet

Trust does not announce its departure. It leaves quietly, account by account, and then all at once. The numbers tell the story.

For most of 2021 and 2022, Credit Suisse leaked deposits and client assets steadily as wealthy clients, spooked by the scandals, moved money to rivals. Then, in October 2022, something modern and dangerous happened: a wave of social-media speculation about the bank's health went viral. A journalist's tweet about an unnamed major bank "on the brink" was widely assumed to be about Credit Suisse. Whether or not the rumours were accurate almost did not matter — the *fear* was real, and fear is contagious.

The result showed up in the fourth-quarter results. In Q4 2022, Credit Suisse suffered net asset outflows of around **CHF 110 billion**. Deposits specifically fell sharply, with the group's deposit base dropping from roughly CHF 234 billion to well below CHF 160 billion over the quarter. This was not a 36-hour run. It was a slow, grinding exodus over weeks — but at a scale that no bank can sustain.

![Bar chart of the Credit Suisse deposit base before Q4 2022 versus the CHF 110 billion that left and what remained](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-2.png)

The chart above puts the outflow against the base it came from. The blue bar is the deposit base going into the quarter; the red bar is what left; the amber bar is what remained. The red bar is roughly a *third* of the blue one. Think about what that means operationally: the bank had to find CHF 110 billion of cash to hand back, in a single quarter, while its assets — loans, bonds, illiquid positions — could not be turned into cash that fast. This is the maturity-transformation trap in slow motion. The bank borrowed short and lent long; when the short borrowing (deposits) walked out the door, the long assets could not be sold fast enough to replace it. (The mechanics of how a deposit base funds a bank, and why "cheap money is the franchise", are covered in [retail deposits and the funding base](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).)

#### Worked example: the outflow as a share of the base

Let us do the arithmetic the bank's treasurer would have done, sweating, every morning.

The deposit base entering Q4 2022 was about CHF 234 billion. The outflows were about CHF 110 billion. As a share of the base:

CHF 110 billion ÷ CHF 234 billion = **0.47**, or roughly **47%** of the deposit base attempting to leave over the quarter (the headline CHF 110 billion was total net asset outflows across the group; deposits alone fell on the order of CHF 70–80 billion, still about a third of the deposit base).

Compare that to what a bank is *built* to handle. Liquidity rules (the Liquidity Coverage Ratio, or LCR) require a bank to hold enough easily-sellable assets to cover **30 days** of a stressed outflow, where "stressed" assumes only a *fraction* of deposits leave — often modelled at 5–25% over the month depending on the deposit type. A real outflow of a third to a half of the deposit base over a *quarter* is several times worse than the stress scenario the rules were designed around. (For how the LCR and the liquidity buffer are supposed to work — and why they were not enough here — see [liquidity management, LCR and NSFR](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)

The intuition: a bank's liquidity buffer is sized for a *bad day*, not for a *crisis of confidence*. When the whole deposit base decides to leave, no buffer built to regulatory specification is big enough, because the regulation never imagined the franchise itself dissolving. Credit Suisse actually breached some of its own internal liquidity limits during this period even while remaining above the regulatory minimums — a sign that the rules were measuring the wrong thing for this kind of death.

### Why a slow bleed is, in some ways, worse

There is a cruel irony here. SVB's run was fast — 36 hours — which meant the authorities had to act over a single weekend but also meant the damage was contained and the cause was crystal clear. Credit Suisse's bleed was slow, which sounds gentler, but a slow bleed is corrosive in a different way: it gives the market time to *watch the patient die*. Every quarter of outflows confirmed the thesis that the bank was finished, which caused more outflows. The slowness did not buy a cure; it just stretched out the agony and made the eventual collapse feel inevitable rather than shocking.

![Pipeline showing how scandals erode trust, trust loss drives outflows, outflows cause unviability, and unviability forces a merger](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-5.png)

The pipeline above is the causal chain in five steps. Scandals stacked up; trust eroded; deposits fled; the bank became unviable; and an unviable bank gets resolved over a weekend. Notice that the chain is *one-directional* and self-reinforcing. There is no natural place where it stops on its own, because each step makes the next more likely. The only thing that can interrupt it is a decisive intervention that *restores confidence* — and as we will see next, the standard intervention tool failed to do that.

## The SNB line: why CHF 100 billion of central-bank cash did not save the bank

When a fundamentally sound bank faces a temporary cash crunch — a *liquidity* problem, not a *solvency* problem — the textbook fix is for the central bank to lend it cash against good collateral. This is the central bank's oldest job, described in the 1870s by Walter Bagehot: in a panic, lend freely, against good collateral, at a penalty rate. The idea is that a run is a self-fulfilling panic; if the central bank stands behind the bank, depositors stop running, the panic ends, and the loan gets repaid. (We cover this lender-of-last-resort role in [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).)

On 15 March 2023, after a particularly brutal trading day in which the share price fell sharply and the bank's largest shareholder — the Saudi National Bank — said publicly that it would *not* put in any more money, the Swiss National Bank stepped in. It announced that Credit Suisse could borrow up to **CHF 100 billion** in emergency liquidity. On paper, this was overwhelming firepower — more than enough to cover any plausible cash demand for a long time.

And it did not work.

### Liquidity versus solvency, and the third thing: viability

To see why, we have to separate three ideas that are easy to blur.

A **liquidity** problem is when a bank is fundamentally sound but cannot lay its hands on cash quickly enough — its assets are good but locked up in long-term loans. The cure is borrowing against those assets. The SNB line was the cure for *this*.

A **solvency** problem is when a bank's assets are actually worth less than what it owes — the equity cushion has been wiped out by real losses. No amount of borrowing fixes solvency; lending more to an insolvent bank just makes the eventual loss bigger. Credit Suisse was not clearly insolvent in this strict sense — its capital ratios were still above the regulatory minimums.

But there is a third condition that the textbook two-way split misses, and it is the one that killed Credit Suisse: **viability**. A bank can be technically liquid (thanks to the central bank) and technically solvent (its capital ratio is fine) and still be *unviable* — meaning no rational depositor, lender, or client will keep dealing with it, so it cannot fund itself going forward at any sustainable cost. A bank that has lost its franchise has lost its future earnings, and a bank with no future cannot be saved by a loan, because the loan only covers *today's* cash, not *tomorrow's* business.

This is the deep lesson of the SNB line. The central bank can solve a liquidity panic where the underlying franchise is intact. It cannot solve a *confidence collapse* where the franchise itself has rotted away. The CHF 100 billion bought a few days. It did not buy back trust, because trust is not something a central bank can lend.

#### Worked example: a CoCo's trigger math (and why being above the trigger did not help)

Let us work through the trigger math on an AT1 bond, because it shows exactly how a bank can be "fine" on the metric the rules watch while dying on the metric that matters.

A typical AT1 bond has a *going-concern* trigger written as: convert (or write down) if the bank's CET1 capital ratio falls below, say, **7%** of risk-weighted assets. (Risk-weighted assets, or RWA, are the bank's assets adjusted for how risky each one is — a government bond gets a low weight, a risky loan a high one; see [risk-weighted assets and capital ratios](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work).)

Suppose our bank has CHF 250 billion of RWA and CHF 35 billion of CET1 capital. Its CET1 ratio is:

CHF 35 billion ÷ CHF 250 billion = **14%**.

That is double the 7% trigger. By the contractual logic of the AT1 bond, the bank is nowhere near the going-concern conversion point. An AT1 holder reading only the capital ratio would feel completely safe — the trigger is miles away.

Now ask: does a 14% CET1 ratio stop depositors leaving? No. Depositors do not read the CET1 ratio; they read the headlines, the share price, and what their friends are doing. The bank could sit at 14% CET1, comfortably above every trigger, and still watch a third of its deposits walk out the door — which is exactly what happened. The capital ratio measures whether *losses* have eaten the cushion. It says nothing about whether *trust* has eaten the franchise.

The intuition: the AT1 trigger is a solvency tripwire, but Credit Suisse died of a viability disease. The going-concern trigger never fired, because the bank's capital never fell below it. Instead, the *gone-concern* clause was activated by regulatory decree when the bank was declared unviable — a completely different switch, and the one almost no AT1 investor was truly pricing.

## The weekend: from emergency line to forced merger in 96 hours

The SNB line was announced Wednesday. By Thursday it was clear it had not stemmed the outflows. By Friday the share price was in free fall, counterparties were pulling back, and the Swiss authorities had concluded that Credit Suisse would not survive to the Monday open without a decisive, structural solution. There was no time to find a foreign buyer, no time for an orderly capital raise, no time for anything but the bluntest tool available: force the only other big Swiss bank, UBS, to take it over.

The negotiations ran through the weekend in Bern, under intense pressure and with the threat that a Monday morning with Credit Suisse still independent could trigger a global panic. The authorities had three goals, in order: protect depositors, protect the financial system, and avoid using taxpayer money if at all possible. The deal they struck was extraordinary in several ways, and to make it happen the Swiss government passed *emergency law* over the weekend to override the normal shareholder-approval process — UBS shareholders never got to vote on the largest acquisition in the bank's history.

The headline terms, announced Sunday 19 March 2023:

- UBS would buy Credit Suisse for **CHF 3 billion**, paid in UBS shares (about 0.76 CHF per Credit Suisse share, a fraction of where the stock had traded even days earlier).
- The Swiss National Bank would provide enormous liquidity support to the merged entity.
- The government would backstop a slice of potential losses on a portfolio of harder-to-value Credit Suisse assets.
- And — the part that detonated — FINMA ordered that approximately **CHF 16 billion** of Credit Suisse AT1 bonds be written down to **zero**.

![Bar chart comparing the AT1 written to zero, the UBS purchase price, the SNB liquidity line, and the Q4 2022 outflows, all in CHF billions](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-6.png)

The figure above puts the four defining numbers of the rescue on one chart. Look at the relationship between the first two bars. The AT1 bondholders lost CHF 16 billion. UBS paid CHF 3 billion for the whole bank. The bondholders lost *more than five times what the entire firm sold for*. And the third bar — the SNB line — reminds you that even CHF 100 billion of central-bank cash, more than thirty times the purchase price, could not prevent this outcome, because the problem was never really cash.

#### Worked example: the UBS price versus book value

The CHF 3 billion price tag deserves its own arithmetic, because it shows what the market thought the franchise was worth versus what the accountants said.

On its books, Credit Suisse reported shareholders' equity of roughly **CHF 45 billion** at the end of 2022 — that is the accounting value of what the owners supposedly held. UBS paid **CHF 3 billion** for the whole thing. The ratio of price to book value is:

CHF 3 billion ÷ CHF 45 billion = **0.067**, or about **7 cents on the franc**.

Put simply, the market and UBS valued the bank at roughly *one-fifteenth* of its stated book equity. Why would equity worth CHF 45 billion on paper sell for CHF 3 billion? Because book value records the historical cost of the bank's *assets*, but it does not record the collapse of the *franchise* — the future earnings that had just evaporated, the litigation tail, the restructuring costs UBS would have to swallow, and the deep uncertainty about what some of the harder-to-value positions were really worth (which is why the government had to backstop a slice of them).

The intuition: book value is what the bank cost to build; the takeover price is what its future is worth, and a dead franchise has almost no future. The 93% discount to book is the clearest single measure of how much value "loss of trust" actually destroyed — far more than any individual scandal's headline loss.

## The AT1 wipeout and the legal firestorm

Now we reach the controversy that made Credit Suisse a case study taught in every finance course since.

Recall the capital stack from Foundations. The promise — the entire reason AT1 bonds could be sold to investors — was that they sat *above* equity in the loss order. Equity loses first; AT1 only after equity is gone. That ordering is the deal. It is why AT1 pays a high coupon but is supposed to be safer than the stock.

In the Credit Suisse resolution, the order appeared to invert. AT1 holders were written to zero — total loss — while equity holders received CHF 3 billion of UBS shares. The people who thought they were *more senior* got *less* than the people below them. To AT1 investors around the world, this looked like a fundamental breach of the rules, an act of expropriation, a rewriting of the contract after the fact.

### How FINMA justified it

The Swiss authorities' defence rested on the fine print of the specific Credit Suisse AT1 contracts and on Swiss law. Swiss AT1 instruments contained a clause permitting a full, permanent write-down at a *viability event* — specifically, if the bank received extraordinary government support or was declared unviable by the regulator. Crucially, this write-down was, in the Swiss reading, *not* conditioned on equity being wiped out first. The emergency legislation passed over the weekend reinforced the regulator's authority to impose the write-down.

In other words: the going-concern trigger (the capital-ratio tripwire) never fired. The bank's CET1 ratio was still comfortably above the trigger. Instead, the *gone-concern* clause — write-down on a viability event — was activated. From FINMA's point of view, this was the contract working exactly as written. The AT1 holders had been paid a high coupon precisely to bear this risk, and the risk materialised.

### Why the market exploded anyway

Legally defensible or not, the decision was a thunderclap, for three reasons.

First, it violated the *mental model* almost every investor held. Even if the contract technically allowed it, the universal assumption — reinforced by years of regulator statements and by the EU's own resolution framework, where authorities quickly clarified that *in their jurisdiction* equity would absorb losses before AT1 — was that AT1 ranked above equity. Switzerland had quietly written its contracts differently, and almost nobody had priced that difference.

Second, it broke a principle that the entire bank-resolution architecture was built to honour: that losses follow the hierarchy, predictably, so investors know what they are buying. If a regulator can flip the order in a crisis, then the "seniority" of every junior bank instrument becomes uncertain — and uncertainty must be paid for with a higher yield.

Third, it was a *Swiss* bank, in a country whose entire financial brand was built on legal stability and predictability. If it could happen there, investors reasoned, it could happen anywhere.

#### Worked example: the inverted hierarchy in dollars

Let us make the inversion concrete with the actual numbers, because the arithmetic is the whole story.

Imagine you held CHF 1 million face value of Credit Suisse AT1 bonds, and a friend held CHF 1 million of Credit Suisse equity (the stock), bought at the same time. You believed, reasonably, that you were *senior* to your friend — that if the bank ran into trouble, your friend would lose first and more.

Here is how it actually played out:

- **Your AT1 (the "senior" claim):** written to zero. Recovery = CHF 0. You lost **100%**, all CHF 1 million.
- **Your friend's equity (the "junior" claim):** converted into UBS shares as part of the CHF 3 billion deal. The total equity received roughly CHF 0.76 of UBS stock per Credit Suisse share. It was a brutal loss from where the stock once traded — but it was *not zero*. Your friend recovered *something*, perhaps a few percent of the original investment.

So the supposedly *junior* holder recovered something while the supposedly *senior* holder recovered nothing. On the scale of the whole bank: CHF 16 billion of AT1 went to zero, while CHF 3 billion of value flowed to equity holders.

The intuition: in a normal wind-down, the loss order is equity → AT1 → senior → depositors, and you can compute your expected recovery from where you sit in that stack. Credit Suisse showed that for *one specific instrument under one specific jurisdiction's contracts*, the order was not a law of nature but a clause in a document — and the clause put AT1 below equity at the viability point. The lesson for investors was not "the system is broken" so much as "read the actual contract, not the marketing, because seniority is whatever the document says it is."

![Graph of an AT1 bond at the trigger: a going-concern trigger converting to equity versus a gone-concern trigger writing it off, with the Credit Suisse path highlighted](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-8.png)

The figure above maps the two fates of an AT1 bond. Follow the top branch: a going-concern trigger (capital falls below the threshold) converts the bond into equity, the bank rebuilds its capital, and it keeps operating — the bondholder becomes a shareholder but the bank survives. Now follow the bottom branch: a gone-concern trigger (the regulator declares the bank unviable) writes the bond off to zero. The Credit Suisse path is the bottom branch, in red. The bank never tripped the going-concern conversion; it was pushed straight to the gone-concern write-off. Understanding which branch your specific bond is exposed to — and which trigger governs it — is the difference between a bondholder who becomes a part-owner of a surviving bank and one who loses everything.

### The aftermath in the AT1 market

The immediate reaction was carnage. In the days after the announcement, the global AT1 market — a roughly \$275 billion market across European and other banks — sold off hard, with prices on many bonds falling 10–20% as investors repriced the risk that *their* AT1 might also rank below equity. New AT1 issuance froze; banks could not sell the instruments at any reasonable price for weeks.

Regulators scrambled to contain the contagion. Within a day, the European Central Bank, the Bank of England, and the EU's resolution authority issued a joint statement essentially saying: *in our jurisdictions, equity absorbs losses before AT1, full stop.* This was a remarkable public correction — the authorities outside Switzerland racing to reassure markets that the Swiss outcome was a quirk of Swiss contracts and law, not a new global norm. It mostly worked: the AT1 market stabilised over the following months and issuance resumed, but at higher yields, with much closer attention to the precise write-down language in each contract. The price of that lesson was paid by every bank that issues AT1: the instrument now carries a permanent "Credit Suisse premium" for the demonstrated tail risk.

Litigation followed and continues. AT1 holders launched claims against Switzerland and FINMA, arguing the write-down was unlawful and amounted to an uncompensated taking. The legal merits turn on the specific contract language and Swiss administrative law, and the cases will take years. Whatever the courts decide, the market has already rendered its verdict in the form of a higher yield demanded on every CoCo bond ever since.

There is also a deeper structural lesson that regulators themselves absorbed. The whole point of post-2008 "bail-in" capital was to make bank failures *predictable*: investors would buy AT1 and other junior instruments knowing precisely where they stood in the loss order, so that when a bank failed, losses fell on private capital in a known sequence rather than on taxpayers in a chaotic bailout. Credit Suisse exposed a flaw in that design: predictability depends on every jurisdiction writing the *same* hierarchy into its contracts and its resolution law, and they had not. Switzerland's AT1 contracts and its emergency-law powers allowed an outcome that the EU framework would have forbidden. The instrument was global, but the rules were national — and in a crisis, the national rules win. After Credit Suisse, the homework for every investor became not just "where does my bond sit in the stack?" but "which country's law governs the stack, and what can that country's regulator do to it in an emergency?"

## Common misconceptions

Let us clear up the beliefs that obscure what actually happened.

**"Credit Suisse collapsed because of a bank run, like SVB."** Not quite. SVB suffered a classic, acute run — about \$42 billion of withdrawals attempted in a single day. Credit Suisse suffered a *chronic* bleed — CHF 110 billion of outflows over an entire quarter, the culmination of years of leakage. The final weekend looked run-like, but the disease was years of lost trust, not a sudden duration shock. The distinction matters because the *cure* is different: a true liquidity run can be stopped by a credible lender of last resort, while a confidence collapse cannot, which is exactly why the CHF 100 billion SNB line failed.

**"The bank was insolvent — its assets were worth less than its debts."** On the strict accounting measure, no. Credit Suisse's capital ratios were above the regulatory minimums right up to the end. It was not clearly insolvent; it was *unviable*. The market had decided it had no future, and a bank with no future cannot fund itself, regardless of what its balance sheet says today. This is the gap between solvency (a snapshot) and viability (a verdict on the franchise).

**"AT1 holders were robbed; the rules were broken."** This is the emotionally satisfying version, but it is too simple. The Swiss AT1 contracts explicitly permitted a full write-down at a viability event, and that write-down was not contractually conditioned on equity being wiped out first. Investors who read only the marketing — "AT1 is senior to equity" — were genuinely surprised. Investors who read the actual Swiss contract terms were not. The "robbery" was really a collision between a widely-held mental model and the specific fine print, in a jurisdiction whose contracts differed from the EU's. The lesson is brutal but precise: seniority is defined by the document, not by the category name.

**"The SNB line should have saved the bank — CHF 100 billion is enormous."** Cash was never the binding constraint. The line could cover any plausible *outflow*, but it could not restore the *inflow* — the willingness of clients to keep their money and their business at the bank. A credit line solves a liquidity gap; it cannot manufacture confidence. Once the franchise was dead, more cash just delayed the funeral.

**"This was a uniquely Swiss problem that can't happen elsewhere."** Partly true, partly dangerous. The *specific* inverted AT1 outcome was a function of Swiss contract terms and emergency law, and other jurisdictions moved fast to clarify their own rules differ. But the *deeper* pattern — a systemically important bank dying of accumulated reputational damage rather than a single loss — is universal. Any bank that spends its trust faster than it earns it is on the Credit Suisse path, regardless of country.

## How it shows up in real banks

The Credit Suisse story is not a one-off curiosity. It is a template, and several real episodes show the same forces — and one important contrast.

### The contrast that defines it: SVB versus Credit Suisse

The cleanest way to understand Credit Suisse is to set it against Silicon Valley Bank, which failed the same week in March 2023.

![Comparison matrix of SVB versus Credit Suisse across root cause, speed, trigger, and resolution](/imgs/blogs/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout-7.png)

The matrix above lays the two side by side. SVB died of a **duration trap**: it had parked a huge share of its deposits in long-dated bonds, and when interest rates rose sharply in 2022, those bonds lost market value. When SVB was forced to sell some at a loss to meet withdrawals, its depositors — overwhelmingly uninsured tech companies who all banked together and all talked to each other — fled in a coordinated, digital, **36-hour** run. The trigger was a single forced bond sale that revealed the loss. The resolution was an FDIC seizure with the state ultimately backstopping deposits. (The full SVB mechanism is dissected in the companion one-pager [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

Credit Suisse died the opposite way. Its root cause was not a single asset-price shock but a **years-long loss of trust**. Its speed was not 36 hours but roughly **two years** of bleeding. Its trigger was not a forced bond sale but a confidence break — capped off when its largest shareholder publicly ruled out more support. Its resolution was not a clean government seizure but a *forced merger* with a competitor plus the AT1 wipeout.

The deep point: these are the two archetypes of bank death. SVB is the **fast, mechanical** death — a quantifiable mismatch between asset duration and deposit demands, the kind of thing a stress test can in principle catch. Credit Suisse is the **slow, qualitative** death — an erosion of the unmeasurable franchise that no capital ratio captures until it is far too late. One you can model; the other you can only watch.

### Lehman Brothers, 2008: the franchise that vanished

Lehman Brothers is the older, larger cousin of the Credit Suisse story. Lehman was not killed by a single trade either; it was killed by the market's collapse of confidence in its mortgage assets and its 30-plus-times leverage. When confidence went, its repo lenders — the firms that funded it overnight — refused to roll the funding, and a bank that funds itself overnight cannot survive a single morning when the lenders say no. Like Credit Suisse, Lehman was arguably not insolvent on the morning it failed by some measures; it was *unfundable*, which in a wholesale-funded bank is the same as dead. The shared lesson across fifteen years: the cause of death on the certificate is almost always "loss of funding", and the cause behind the cause is almost always "loss of trust".

### Northern Rock, 2007: the run you can see coming

Northern Rock, a British mortgage lender, failed in 2007 with the first visible bank-run queues in Britain in over a century — images of depositors lined up around the block. It had funded long-term mortgages with short-term wholesale market borrowing; when that market froze in the early credit crisis, it could not refinance. The Bank of England provided emergency support — the equivalent of the SNB line — and, as at Credit Suisse, the emergency support *itself* signalled distress and accelerated the run rather than calming it. The parallel is exact: a lender-of-last-resort facility is supposed to reassure, but when the public reads it as confirmation that the bank is in trouble, it pours fuel on the fire.

### The AT1 market repricing: the lasting financial scar

The most measurable real-world consequence of Credit Suisse is the permanent change in the AT1 market. Before March 2023, AT1 buyers largely assumed a uniform, EU-style hierarchy where equity always lost first. After Credit Suisse, every AT1 prospectus is read line by line for its exact write-down trigger and its treatment relative to equity. The roughly \$275 billion global AT1 market did not disappear — banks still need this capital, and after a few frozen weeks issuance resumed — but it now demands a higher yield to compensate for the demonstrated risk that the hierarchy can invert under the right contract and the right jurisdiction. Every bank that issues a CoCo today pays a little more, forever, because of one weekend in Bern. That is what it looks like when a market *learns*.

### Why regulators changed their tune so fast

The speed of the ECB, Bank of England, and EU resolution authority statement — within roughly a day of the Credit Suisse announcement — is itself a case study in how fragile confidence is in junior bank capital. The authorities understood that if AT1 investors *globally* concluded their bonds now ranked below equity, the entire post-2008 architecture of "bail-in" capital — the system designed to make bank failures cost taxpayers nothing by imposing losses on private investors instead — could unravel, because nobody would buy the instruments. Their rapid clarification was an act of confidence repair, the same medicine Credit Suisse needed and never got, applied this time to a *market* rather than a single bank. It worked because the market still trusted those regulators. Trust, once again, was the active ingredient.

## The takeaway: how to read a bank for the disease that actually kills it

If you take one thing from the death of Credit Suisse, let it be this: **a bank's capital ratio measures whether losses have eaten its cushion, but its franchise measures whether trust has eaten its future, and the second kills more often than the first.** Every metric on the regulatory dashboard — CET1, LCR, leverage ratio — was flashing acceptable while Credit Suisse died. The number that was screaming was the one no regulator publishes: the rate at which clients were quietly deciding the bank was no longer safe to deal with.

So when you read about a bank in trouble, train yourself to ask the questions the capital ratio cannot answer. Is this a *fast* death or a *slow* one — a duration shock that a credible backstop can stop, or a confidence collapse that no backstop can cure? Is the bank *insolvent* (assets worth less than debts), or merely *unviable* (sound on paper but unable to fund itself because nobody trusts it)? Is the central-bank line a bridge to recovery, or a flare that signals the end? And — the question Credit Suisse burned into every fixed-income investor — does the instrument you hold actually rank where its category name implies, or does the fine print put it somewhere else when it matters most?

For anyone who owns or might own bank debt, the practical discipline is to read the *contract*, not the marketing. "AT1 is senior to equity" was true in spirit, false in the specific Swiss fine print, and the gap between those two cost CHF 16 billion. Seniority is a clause, not a law of nature; the only way to know where you stand is to read the document and to know the jurisdiction's resolution law. (None of this is investment advice — it is a map of how the mechanism works so you can read the next failure with clear eyes.)

And for understanding banks in general, Credit Suisse is the purest demonstration of the spine running through this whole series: a bank is a leveraged, confidence-funded maturity-transformation machine. It borrows short, lends long, earns the spread, and survives only as long as depositors trust it. Credit Suisse had everything except the last ingredient. It had capital, it had a 167-year history, it had an emergency credit line bigger than most countries' reserves. What it did not have, by March 2023, was trust — and a bank without trust is not a bank at all. It is a balance sheet waiting for someone to declare the obvious.

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the spine of the whole series: why a bank is a confidence-funded machine.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — how the 8% equity sliver absorbs losses and why a small loss can be fatal.
- [Basel I, II, III and the capital rules that govern every bank](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) — where AT1 and CoCos fit in the capital stack and what the triggers mean.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — why a regulatory liquidity buffer is built for a bad day, not a confidence collapse.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — how the deposit base funds a bank and what its loss means.
- [Deposit insurance, the lender of last resort and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — why a central-bank line can fail to stop a run.
- [Risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) — the RWA denominator behind the CET1 trigger math.
- [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the companion overview contrasting the two failures.

---
title: "The Savings and Loan Crisis: Interest-Rate Mismatch and a Thousand Failures"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How thrifts that borrowed short and lent long for 30 years were destroyed by Volcker's rates, then gambled the cleanup into a 124-billion-dollar taxpayer bill — the systemic version of SVB's duration trap."
tags: ["banking", "savings-and-loan", "duration-risk", "interest-rate-risk", "moral-hazard", "bank-failure", "deregulation", "financial-history", "forbearance", "zombie-bank"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The savings and loan crisis was the slow-motion detonation of a single mismatch: thrifts funded 30-year fixed-rate mortgages with deposits that could leave overnight, so when Paul Volcker's Federal Reserve drove short rates toward 20%, the spread inverted, the industry's net worth vanished, and a thousand-plus institutions died over the next decade.
>
> - A thrift earning 6% on its locked-in mortgage book but paying 12% to keep deposits loses money on every dollar it funds — its net interest margin (the spread between what it earns and what it pays) goes *negative*. That is the duration trap, the same one that killed Silicon Valley Bank in 2023.
> - Instead of closing the insolvent thrifts, regulators raised deposit insurance to \$100,000, loosened the rules, and let "zombie" institutions keep operating — so thrifts with nothing left to lose gambled on land deals and junk bonds, because heads they won and tails the insurance fund paid.
> - The cleanup body, the Resolution Trust Corporation, seized failed thrifts, paid off insured savers, and sold the wreckage for years.
> - **The number to remember: about 1,043 thrifts failed between 1986 and 1995, at a direct taxpayer cost of roughly \$124 billion (around \$160 billion all-in).**

In 1966, owning a savings and loan was about the safest, most boring business in America. The joke was the "3-6-3 rule": pay 3% on deposits, lend at 6% on home mortgages, and be on the golf course by 3 o'clock. The savings and loan — a *thrift*, an institution whose whole purpose was to take small savers' deposits and turn them into long-term home loans — was a community fixture, a sleepy utility, the place in *It's a Wonderful Life* where George Bailey explains that your money isn't in a vault, it's in Joe's house and the Kennedy house down the street.

Twenty years later that sleepy utility was a smoking crater. By the mid-1980s the entire thrift industry had a *negative* net worth on a market-value basis — it owed more than it owned. Over the following decade more than a thousand of these institutions failed. The cleanup cost American taxpayers around \$124 billion in direct money and consumed a federal cleanup agency built from scratch to dispose of the debris. It produced criminal convictions, a United States senator scandal (the "Keating Five"), and a phrase — *gambling for resurrection* — that economists still use to describe what a bankrupt institution does when you let it keep playing.

The astonishing thing is that almost the entire disaster traces back to one technical decision that sounds harmless: a thrift earns a fixed rate for thirty years and pays a floating rate that can change tomorrow. That is the whole story in one sentence. Everything else — the deregulation, the fraud, the bailout — is what happens when you try to escape that trap instead of closing the institutions it has already destroyed. This post is about how that one mismatch works, why it is lethal, why the cure made it worse, and why a smart reader watching a 2023 bank collapse on their phone was watching the exact same movie with better production values.

![The S&L duration trap from the borrow-short lend-long model to insolvency](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-1.png)

The diagram above is the mental model for the whole post. Read it left to right: a thrift's business model is to fund long fixed-rate mortgages with short-term deposits; the assets are locked in at a low yield for decades; a rate shock drives the cost of deposits above that yield; the spread inverts; the thrift bleeds on every dollar it funds; and its equity is wiped out. Hold that picture. Everything below is an expansion of those six boxes.

## Foundations: thrifts, the borrow-short-lend-long mismatch, and the words you need

Before we can tell the story we have to build the vocabulary from scratch. A curious reader with no banking background can follow every step if we define each term the first time it appears. Skim this section if you already know it; do not skip it if you don't.

### What a thrift actually is

A **savings and loan association** — also called an **S&L** or a **thrift** — is a special kind of bank. Where a commercial bank does a bit of everything (business loans, payments, trading, cards), a thrift was, by law and by tradition, a specialist: it took **deposits** from ordinary households and used that money to make **residential mortgages**, the loans people use to buy houses. In the United States this was deliberate public policy. After the Great Depression, the government wanted to channel small savers' money into home ownership, so it created a protected niche — thrifts got tax advantages and a slightly higher legal cap on the interest they could pay savers, in exchange for sticking to home loans.

The everyday analogy is a neighborhood cooperative. Picture a corner shop that does one thing: it collects the spare cash everyone in the neighborhood drops into a jar, and it lends that cash to neighbors who want to build a house, charging a little interest. As long as not everyone wants their cash back on the same day, the shop can keep most of the money out on long house loans and still hand back the few jars people ask for. That is maturity transformation, and it is the heart of all banking.

### The one trade every bank makes: borrow short, lend long

Here is the spine of this whole series, and the single idea you must internalize:

> A bank — and a thrift is just a focused bank — is a **maturity-transformation machine**. It borrows **short** (deposits, which the saver can pull out on short notice) and lends **long** (mortgages, which won't be repaid for many years). It earns the **spread** between the long lending rate and the short funding rate, and it survives only as long as depositors trust it and its thin layer of equity can absorb losses faster than they arrive.

Let's define those terms precisely:

- **Maturity** is just *how long until the money has to be paid back*. A checking deposit has a maturity of essentially zero — you can demand it now. A 30-year mortgage has a maturity of thirty years.
- **Borrowing short** means your funding (the money you owe) can be withdrawn or re-priced quickly. Deposits are short-term funding even when the saver leaves them for years, because they *could* leave tomorrow and the rate you pay them resets to whatever the market demands.
- **Lending long** means your assets (the money owed to you) are locked in for a long time at a fixed rate. A 30-year fixed-rate mortgage pays the thrift the same dollar coupon every month for three decades, no matter what happens to interest rates.
- The **spread** (also the **net interest margin** or **NIM**) is the difference between the average rate the bank *earns* on its assets and the average rate it *pays* on its funding. If a thrift earns 8% on its mortgages and pays 5% for deposits, its spread is 3 percentage points. That spread, multiplied across a big balance sheet, is essentially the entire profit of a traditional bank.

The trade is genuinely useful: society wants someone to turn skittish short-term savings into patient long-term housing finance, and the thrift does exactly that. But the trade is also structurally fragile, for a reason that took the entire industry by surprise.

### Duration risk: the trap hiding inside the spread

The fragility has a name: **interest-rate risk**, and its sharpest form is **duration risk**.

**Duration** is a measure of how sensitive the value of a stream of fixed payments is to a change in interest rates. The intuition: if you are locked into receiving \$60 a year for thirty years, and brand-new loans suddenly pay \$120 a year, your old \$60 stream is now worth far less than it was — nobody will pay full price for your stale, low-yielding contract when they could buy a fresh high-yielding one. The longer the stream is locked in (the higher its duration), the more its market value falls when rates rise. A 30-year fixed mortgage has a very long duration. A demand deposit has a duration near zero.

That asymmetry is the whole problem. The thrift's **assets** have long duration (30-year mortgages) and its **liabilities** have short duration (deposits). This is the **duration gap** — a large mismatch between how long your assets are locked in and how long your funding is locked in. When rates rise:

1. The cost of your short-duration funding goes *up immediately* — you must pay savers the new, higher rate or they leave.
2. The income from your long-duration assets does *not* go up — you're stuck collecting the old 6% on mortgages you wrote years ago.

So your spread gets squeezed from both ends, and if rates rise enough, it goes negative: you pay more for money than you earn on it. Worse, the *market value* of those long mortgages collapses, so even if you wanted to sell them and start over, you'd realize a huge loss. The trap has two jaws — an income jaw (the squeezed spread) and a value jaw (the collapsed asset price) — and they close together.

### Why the gap is invisible until it isn't

There's a subtle reason the duration gap was so dangerous: for decades it didn't show up anywhere a manager would look. A thrift in the 1960s and 1970s kept its books at **historical cost** — it recorded a mortgage at the dollar amount it lent, not at what the loan would fetch in the market today. As long as the loan kept paying, it sat on the balance sheet at face value, year after year, looking perfectly healthy. The duration gap was a *latent* exposure: invisible on the books, fully present in reality, waiting for the one event — a rate shock — that would convert it from a footnote into a sinkhole. This is exactly the trap that snared Silicon Valley Bank, which held billions of bonds in a "hold-to-maturity" bucket that didn't have to be marked down on the headline balance sheet even as their true value cratered. Historical-cost accounting doesn't make the duration loss go away; it just hides it until a forced sale or a run drags it into the light.

We'll meet several more terms as we go — **zombie bank**, **gambling for resurrection**, **moral hazard**, **forbearance**, **deposit insurance** — but each is easier to grasp once you've seen the trap spring, so I'll define them in context. For now, hold three things: thrifts borrowed short and lent long; the gap between those maturities is duration risk; and a big enough rate rise turns that gap from a quiet exposure into a fatal one.

## The original business: why this looked safe for thirty years

To understand the catastrophe you first have to understand why nobody saw it coming, and that requires understanding the world thrifts were built for.

From the 1930s into the late 1970s, American interest rates were low and remarkably stable. The Federal Reserve held short rates in a narrow band, inflation was modest, and a regulation called **Regulation Q** capped the interest rate banks and thrifts were allowed to pay on deposits. Thrifts even got a small advantage — they could pay a quarter-point more than commercial banks, to help them attract the savings they'd channel into housing.

In that world the borrow-short-lend-long trade was almost riskless. Why? Because the *short* side of the trade — the deposit rate — was nailed down by law. A thrift could write a 30-year mortgage at 6%, fund it with deposits costing 3% (capped by Regulation Q), and pocket a stable 3% spread for decades. The duration gap existed, but it didn't bite, because the price of the short funding couldn't run away. Regulation Q effectively froze one jaw of the trap.

#### Worked example: the sleepy thrift's net interest margin in 1965

Let's make the 3-6-3 world concrete with round numbers. Suppose a thrift has \$100 million in assets, all of it 30-year fixed-rate mortgages earning 6%. It funds those assets with \$92 million of deposits costing 3% and \$8 million of equity (the owners' own money — the 8% equity cushion is the realistic base case for a deposit-funded institution).

- Interest *earned* on assets: \$100M × 6% = \$6.0 million per year.
- Interest *paid* on deposits: \$92M × 3% = \$2.76 million per year.
- **Net interest income: \$6.0M − \$2.76M = \$3.24 million.**
- Net interest margin (net interest income ÷ assets): \$3.24M ÷ \$100M = **3.24%**.

That \$3.24 million covers salaries, the branch, a little loan loss, and leaves a tidy profit on \$8 million of equity. The one-sentence intuition: when the rate you pay is frozen by law below the rate you earn, the spread business prints money like a utility — and that is exactly why an entire industry walked, eyes closed, into a duration gap that would have terrified anyone who priced it.

There's a second reason the model looked safe: **deposit stickiness**. In a low-rate, capped world, savers had nowhere better to go. Your passbook account at the local thrift paid the legal maximum, and so did every other thrift and bank in town, so you left your money where it was. This is what bankers call a **sticky** deposit base — funding that stays put even though, in theory, it could leave overnight. Stickiness made the short-term funding *behave* like long-term funding, which made the duration gap look smaller than it really was. The mortgages were locked in for thirty years; the deposits, in practice, also seemed to stay for years. The gap appeared modest.

The lethal insight the thrifts missed is that **stickiness is conditional, not structural.** Deposits are sticky only as long as there's no better option and no reason to fear. The instant a money fund offers three times the rate, or a rumor of trouble spreads, sticky deposits become flighty deposits — and the true, terrifying duration gap reappears in full. The thrifts had built a permanent business on a temporary condition. When both the rate cap and the calm went away in the late 1970s, the funding that had felt like a 30-year anchor revealed itself as the overnight money it always legally was.

The thrift looked safe because the danger was disabled, not absent. Regulation Q was a clamp holding one jaw of the trap open. Take the clamp away, and the jaw snaps shut. That is precisely what happened next — but it wasn't a deregulator who removed the clamp first. It was inflation.

## Volcker's rate shock: the clamp comes off

In the 1970s American inflation broke loose. Oil shocks, loose monetary policy, and a wage-price spiral pushed consumer prices up 7%, then 11%, then 13% a year by 1979. Inflation is poison for any saver: if prices rise 13% and your thrift pays you 5% on your passbook account (the legal cap), you are losing 8% of your purchasing power every year by saving.

So savers started to leave. New money-market mutual funds — investment funds that bought short-term government and corporate IOUs and paid savers whatever those instruments yielded — could pay 10%, 12%, even 15%, because they weren't subject to Regulation Q. Money poured out of thrift passbook accounts and into money funds, a process the industry called **disintermediation** (savers cutting out the bank "intermediary" and buying market instruments directly). The thrifts' cheap, captive funding base was draining away.

Then came Paul Volcker. Appointed Fed chairman in 1979, Volcker decided the only way to break inflation was to crush it with brutally high interest rates. The Fed's policy rate, which had hovered around 5% for decades, was driven to extraordinary heights: the federal funds rate touched roughly 19–20% in 1981, the 3-month Treasury bill yielded around 16%, and the prime rate banks charged their best customers peaked at 21.5%. This was the deepest monetary shock in modern American history, and it worked — inflation collapsed over the following years. But it detonated the thrift industry on the way.

Here is why. To stop the bleed of deposits, Congress and regulators phased out Regulation Q — the deposit-rate cap — between 1980 and 1986. The clamp came off. Now thrifts had to pay market rates to keep savers, and market rates were 12%, 14%, 16%. But their *assets* were still the old 6% and 8% mortgages, written years ago, locked in for thirty years. Both jaws of the duration trap snapped shut at once.

![Deposit cost crossing above the locked mortgage yield in the early 1980s](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-2.png)

The chart above (a stylized version of the early-1980s squeeze) shows the two lines that matter. The blue line is the yield on the thrift's mortgage book — sticky and low, because it's mostly old loans that creep up only as new mortgages get added. The red line is what the thrift had to pay for deposits, which spiked with Volcker. Where the red line crosses above the blue line, the shaded zone opens: every day in that zone, the thrift earns less on its assets than it pays on its funding. That is a negative spread, and it is the financial equivalent of a slow-bleeding wound.

The cruelty of the chart is in the *slopes*. The red funding line is steep — it tracks the market almost immediately, because deposits reprice fast: a saver who can get 12% next door won't accept 6% from you for long. The blue asset line is nearly flat — it crawls upward only as old mortgages slowly pay off and get replaced by new, higher-rate ones, a process that takes years because the typical mortgage in the book has decades left to run. The faster your funding reprices relative to your assets, the deeper and longer the negative-spread zone. The metric that captures this is **deposit beta** — the fraction of a market-rate move that a bank has to pass through to its depositors. A deposit beta of 1.0 means every basis point of rate increase shows up in your funding cost immediately; a beta of 0 means your funding cost never moves. In the disintermediation panic of the early 1980s, thrift deposit betas effectively went to 1.0 — the cap was gone and savers demanded the full market rate — while the asset book's "beta" was close to zero for years. That mismatch in repricing speed *is* the income jaw of the duration trap, expressed as a number.

## The mismatch losses: when net interest margin goes negative

Let's now do the brutal arithmetic that the 3-6-3 thrift never planned for.

#### Worked example: the same thrift, now bleeding, in 1981

Take our \$100 million thrift from before — \$100M of 30-year mortgages earning 6%, funded by \$92M of deposits and \$8M of equity. The only thing that changes is the world: Regulation Q is gone and savers now demand 12% or they walk to a money fund.

- Interest *earned* on assets: \$100M × 6% = \$6.0 million per year. (Unchanged — the mortgages are locked at 6%.)
- Interest *paid* on deposits: \$92M × 12% = \$11.04 million per year. (The cost of funding has quadrupled.)
- **Net interest income: \$6.0M − \$11.04M = negative \$5.04 million.**
- Net interest margin: −\$5.04M ÷ \$100M = **negative 5.04%.**

Read that again. The thrift is now *losing* about \$5 million a year before it pays a single salary or absorbs a single loan default. Its margin isn't thin — it's negative. And remember the equity cushion is only \$8 million. At a \$5 million annual loss, the thrift's entire net worth is gone in under two years just from the carry, and that's before any other problem. The one-sentence intuition: a negative net interest margin means the bank loses money on the very act of existing — every dollar it funds is a dollar it pays to lose — and no amount of cost-cutting or good lending can fix a spread that is upside down.

This is the income jaw of the trap. Now the value jaw.

#### Worked example: the duration loss on the mortgage book

The thrift's mortgages are worth far less than face value once rates explode, because nobody will pay 100 cents for a 6% loan when fresh loans pay 14%. Let's estimate the hit with duration.

A 30-year fixed mortgage has a duration of roughly 8 years (the cash flows are spread out, and prepayments shorten it, so it's not the full 30). The rule of thumb: the percentage change in a bond's price is approximately *minus duration times the change in rate*.

- Duration of the mortgage book: about 8 years.
- Change in market rates: from 6% to 14% is +8 percentage points (+0.08).
- Approximate price change: −8 × 0.08 = **−0.64, or about a 64% loss in market value.**

So the \$100 million mortgage book, marked to what the market would actually pay, might be worth only about \$36 million. Even if the income jaw didn't exist, this alone obliterates the \$8 million of equity many times over. On a **market-value** basis the thrift is deeply insolvent: its assets are worth \$36M, it owes depositors \$92M, so its true net worth is around **negative \$56 million.** The one-sentence intuition: long-duration assets don't just earn too little when rates rise — they *are* worth dramatically less, so a thrift that looks fine on its old-cost books is a ghost the moment you mark its assets to the market.

By 1982, regulators and economists who did this arithmetic across the whole industry reached a stunning conclusion: the United States thrift industry, taken as a whole, had a *negative* market-value net worth — by some estimates the entire sector was underwater by \$100 billion or more. The industry was, collectively, already dead. It just hadn't been buried.

This is the moment the story should have ended cheaply. Had the government recognized the losses, closed the insolvent thrifts, and paid off the insured depositors in 1982, the bill would have been a fraction of what it became — most estimates put the early-1980s cost at well under \$50 billion. Instead, the policy response turned a manageable interest-rate disaster into a decade-long fraud-soaked catastrophe. To understand why, we need two more concepts: the **zombie bank** and **regulatory forbearance**.

## The zombie thrift and the temptation of forbearance

A **zombie bank** is an institution that is economically dead — its liabilities exceed the true value of its assets, so its real net worth is negative — but which is still allowed to operate, take deposits, and make loans. It walks among the living on borrowed time and other people's money. By the early 1980s the thrift industry was a colony of zombies.

Why weren't they closed? Two reasons, one practical and one self-serving.

The practical reason was money. Closing a thrift means the deposit insurer must pay off all the insured depositors out of its own fund. The thrift insurance fund — the **Federal Savings and Loan Insurance Corporation**, or **FSLIC** — was tiny relative to the hole in the industry. There was simply no way it could honor the claims of every depositor at every insolvent thrift in 1982. Recognizing all the losses at once would have bankrupted the insurer itself and forced an immediate, enormous appropriation from Congress, which nobody wanted to ask for.

The self-serving reason was politics and hope. Regulators and politicians told themselves a comforting story: the thrifts are only insolvent because rates are temporarily high; if we just keep them open, rates will come down, the spread will recover, and they'll earn their way back to solvency. Closing them now would "crystallize" losses that might evaporate on their own. This policy of looking the other way — relaxing accounting and capital rules so a dead institution can be declared alive on paper, in the hope it recovers — is called **regulatory forbearance**.

Forbearance is not a neutral wait-and-see. It is an active decision to let an insolvent institution keep gambling with insured money. And that is where it becomes lethal, because of one of the most important ideas in all of finance: **moral hazard**.

**Moral hazard** is the change in behavior that happens when someone is shielded from the consequences of their own risk-taking. The classic example: a driver with full insurance and no deductible drives less carefully than one who pays for every dent. When you don't bear your own downside, you take more risk. A zombie thrift is the purest case of moral hazard ever constructed, and we'll see exactly why in the next two sections.

![A sound thrift versus a zombie thrift gambling for resurrection](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-4.png)

The before-and-after figure makes the behavioral shift visible. On the left, the sound thrift has positive net worth, holds prime mortgages, and lends carefully — because the owner's own equity eats any loss. On the right, the zombie has negative net worth, holds land deals and junk bonds, and bets as big as the law allows — because there's nothing left of its own to lose, and the insurance fund (and ultimately the taxpayer) eats the downside. Same institution, opposite incentives. The difference is whether real equity is at stake.

## Deregulation: handing matches to an arsonist

Faced with a dead industry it couldn't afford to bury, Washington made a fateful bet in the early 1980s: instead of closing the zombies, *let them grow and gamble their way back to health.* The reasoning was that thrifts couldn't earn their way out of insolvency on thin home-mortgage spreads, so the answer was to give them new, more profitable (and far riskier) lines of business and the cheap funding to chase them. Two laws did the damage.

The **Depository Institutions Deregulation and Monetary Control Act of 1980** began phasing out Regulation Q (removing the deposit-rate clamp, as we saw) and — crucially — raised the **deposit insurance** limit from \$40,000 to **\$100,000** per account. Deposit insurance is the government's promise to make depositors whole if their bank fails; it exists so ordinary savers don't have to fear losing their money and don't start panicked runs. But raising the limit to \$100,000 had a second, darker effect: it made large deposits at *any* thrift completely safe regardless of how reckless that thrift was, which meant a zombie could attract unlimited funding by simply offering a high rate. Risk no longer scared money away. Insurance had severed the link between a thrift's recklessness and its ability to fund itself.

The **Garn–St. Germain Depository Institutions Act of 1982** went further. It let thrifts — institutions designed to make home loans — pour money into commercial real estate, land development, consumer loans, and corporate junk bonds. It let them put a huge share of assets into a single project or borrower. In some states, regulators went even further, allowing state-chartered thrifts to take *direct equity stakes* in real-estate developments — to become, in effect, property speculators with a federally insured checkbook. Capital requirements (the minimum equity cushion a thrift had to hold) were *lowered*, and creative accounting was blessed so that paper net worth could be conjured where real net worth had vanished.

Stack these up and you have built a doomsday machine:

1. **Unlimited cheap funding.** Thanks to \$100,000 insurance, a zombie could raise billions in **brokered deposits** — large, insured deposits gathered nationwide by brokers and steered to whoever paid the highest rate. Savers didn't care if the thrift was a death-trap; their money was insured.
2. **Broad gambling powers.** Garn–St. Germain let that money go into speculative land, commercial real estate, and junk bonds — far higher potential returns than home mortgages.
3. **Nothing left to lose.** The thrifts were already insolvent, so the owners had no real equity at stake.
4. **Heads I win, tails you pay.** If a wild bet paid off, the thrift kept the profit and might claw back to solvency. If it failed, the loss fell on the insurance fund and, behind it, the taxpayer.

This is the textbook setup for **gambling for resurrection** — the strategy a bankrupt institution rationally adopts when it's allowed to keep operating: take the biggest, riskiest bets available, because a long shot is the only thing that can save you, and you're not the one who pays if it misses.

![The moral-hazard loop of insured deposits forbearance and big bets](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-5.png)

The graph shows the loop in motion. Insured deposits mean savers don't run; forbearance means the regulator lets the zombie keep trading; together they hand the thrift cheap brokered funding to fund explosive growth. That money goes into big speculative bets. Then the asymmetry: heads, the bet pays and the thrift keeps the whole profit; tails, the bet fails and the insurance fund and taxpayer pay. When the downside isn't yours, you bet the maximum the rules allow.

## The gamble: how a bankrupt thrift bets the house

Let's make the gambling-for-resurrection logic concrete, because it's the hinge of the entire crisis and it is genuinely counterintuitive — it explains why deregulation made an interest-rate problem so much *worse* rather than better.

Imagine you run a thrift that is already insolvent. Its true net worth is, say, negative \$10 million. On a normal balance sheet you'd be closed tomorrow. But forbearance keeps you open, and Garn–St. Germain lets you do almost anything. What's your rational play?

You do *not* invest in safe, low-return assets. Safe assets earn maybe 8% — not nearly enough to dig out of a \$10 million hole, and if you just sit there, you eventually get caught and closed. Instead you take the biggest swing available, because only a big win can resurrect you, and a big loss costs you nothing you still own.

#### Worked example: the gamble-for-resurrection payoff to a zombie

Suppose the insolvent thrift uses its insured-deposit funding to make a \$100 million speculative loan to a commercial real-estate developer building a resort. The deal pays 18% if it works and defaults to a 50% loss if it busts. There's a 50/50 chance of each. Compare the thrift's perspective with a normal investor's.

A normal, solvent investor weighs the *expected* outcome:

- Win (50%): +18% × \$100M = +\$18M.
- Lose (50%): −50% × \$100M = −\$50M.
- Expected value: 0.5 × (+\$18M) + 0.5 × (−\$50M) = +\$9M − \$25M = **negative \$16 million.** A solvent investor with skin in the game would never touch this — it loses money on average.

But the zombie's calculus is different, because it doesn't bear the downside. Its equity is already gone; below zero, additional losses fall on the insurance fund, not on the thrift's owners. So from the *owners'* seat:

- Win (50%): the thrift pockets the +\$18M, which goes toward refilling the \$10M hole — they're alive and possibly solvent again. Payoff to owners: **+\$18M.**
- Lose (50%): the thrift fails, but it was failing anyway; the owners lose nothing more than the zero they already had. Payoff to owners: **\$0** (the −\$50M lands on the FSLIC).
- Expected value *to the owners*: 0.5 × (+\$18M) + 0.5 × (\$0) = **+\$9 million.**

The same bet that's worth *negative* \$16 million to society is worth *positive* \$9 million to the zombie's owners. The one-sentence intuition: deposit insurance plus forbearance flips the sign on risk — a bet that destroys value for the world creates value for the gambler, because the gambler keeps the wins and the public keeps the losses, so the rational move for a zombie is to bet as large and as wild as the law allows.

Multiply this logic across hundreds of insolvent thrifts, each raising brokered deposits to fund the riskiest projects it could find, and you get the second half of the crisis: a frenzied, debt-fueled construction and land boom across the Sun Belt — Texas, Arizona, California, Florida — funded by federally insured savings, much of it on projects that never had a prayer of paying off. When the regional real-estate boom turned to bust in the mid-1980s (oil prices crashed, wrecking Texas especially), the speculative loans went bad en masse, and the paper net worth that forbearance had conjured turned out to be exactly the fiction it always was.

Notice how the gamble *changed the kind of risk* the thrifts ran. The original problem was pure interest-rate risk — a duration mismatch. The cure layered a second, different risk on top: **credit risk**, the risk that a borrower simply doesn't pay back. A 30-year home mortgage to a creditworthy family rarely defaults; a speculative loan to a developer building a half-empty office park in a glutted market defaults constantly. So the deregulated thrifts traded a manageable rate problem for an unmanageable credit problem, and ran both at once. They didn't fix the duration trap — they couldn't, because the old mortgages were still on the books — they just stacked a credit bomb beside the rate bomb and lit both fuses. This is the deepest reason gambling for resurrection is so destructive: it doesn't remove the original risk, it *adds* a new one, financed by money that runs away the moment either one detonates.

## Fraud: where moral hazard meets crime

A system where you keep the winnings and the public keeps the losses doesn't just attract reckless gamblers. It attracts criminals, because it is the closest thing finance has ever offered to a legal money-printing machine — and where the rules were loose enough, plenty of operators stopped bothering with the "legal" part.

Some of the most expensive failures were outright fraud. Operators bought sleepy thrifts specifically to use the insured-deposit funding as a personal slush fund. The techniques became infamous:

- **Land flips.** Two insiders would sell the same piece of worthless land back and forth at ever-higher prices, each "sale" booked as a gain, until the thrift "lent" against the inflated value — pumping out cash that vanished into the insiders' pockets.
- **Nominee loans.** A thrift would lend to a straw borrower who secretly kicked the money back to the thrift's owners, disguising what was really self-dealing.
- **Dead cows and dead horses.** Loans were booked against collateral that didn't exist or was wildly overvalued.
- **ADC loans booked as profit.** Acquisition, development, and construction loans were structured so that the thrift booked large upfront fees as immediate income — manufacturing fictitious profits that justified bonuses and dividends even as the underlying projects sank.

The most notorious case was **Lincoln Savings and Loan**, run by Charles Keating. Lincoln took insured deposits and shoveled them into junk bonds, raw land, and speculative ventures while reporting fictitious profits. When regulators tried to rein it in, Keating enlisted five U.S. senators (the "Keating Five") to pressure them off. Lincoln's 1989 collapse cost the insurance fund over \$3 billion and wiped out thousands of elderly customers who'd been steered out of insured deposits and into Lincoln's own worthless bonds. Keating went to prison. He was not alone: the crisis ultimately produced over a thousand felony convictions of thrift insiders.

It's important to be precise here, because it's a common misconception that the S&L crisis was *mainly* a fraud story. It wasn't. The fraud was real, vivid, and it raised the bill, but most of the damage came from the legal-but-reckless gambling that forbearance and deregulation invited. Estimates attribute somewhere in the range of 10–25% of the total losses to outright criminal fraud; the larger share came from honest-but-doomed interest-rate losses and from speculative bets that were perfectly legal under Garn–St. Germain. Fraud was the lurid symptom. Moral hazard was the disease.

Why did fraud flourish so easily? Because every control that normally restrains a banker had been switched off at once. Normally, four things police a bank's risk-taking: its own shareholders (who lose if it gambles), its depositors (who pull money from a shaky bank), its regulators (who examine and close bad banks), and its accountants (who mark losses honestly). In the deregulated thrift world, all four were neutralized. Shareholders had nothing left to lose, so they cheered the gambling. Depositors were fully insured, so they happily funded the worst thrifts at the highest rates without a second thought. Regulators were understaffed, outgunned, and politically discouraged from closing thrifts — the examination force had actually been *cut* during the early-1980s growth surge, leaving a handful of examiners to watch an explosively growing industry. And the accounting had been loosened so that fictitious profits and conjured net worth passed muster. Remove every guardrail simultaneously and you don't merely permit fraud — you select for it, because the operators willing to push hardest into the gap are exactly the ones least troubled by the rules. The lesson is uncomfortable: fraud isn't only a moral failing of bad individuals; it's a predictable output of a system that has dismantled its own checks.

## The reckoning and the RTC cleanup

By the late 1980s the fiction was unsustainable. The FSLIC — the thrift insurance fund — was itself insolvent, with nowhere near enough money to close the hundreds of dead thrifts whose losses kept compounding. The longer regulators waited, the bigger the hole grew, because every quarter a zombie stayed open was another quarter of negative carry and fresh bad bets piling on top of the old ones. Forbearance hadn't saved money; it had run up the tab.

In 1989 Congress finally faced it with the **Financial Institutions Reform, Recovery, and Enforcement Act (FIRREA)**. FIRREA abolished the failed thrift regulator and the bankrupt FSLIC, moved thrift deposit insurance under the FDIC (the **Federal Deposit Insurance Corporation**, the body that insures commercial bank deposits), raised capital requirements, and — its central act — created the **Resolution Trust Corporation (RTC)** to seize the dead thrifts and dispose of the wreckage.

The RTC was a cleanup machine on an unprecedented scale. Its job was to take over failed thrifts, protect their insured depositors, and then sell off the rubble — the bad loans, the half-built resorts, the foreclosed land — for whatever the market would bear.

The genius of the RTC, in retrospect, was that it stopped trying to *avoid* losses and started trying to *minimize* them — a crucial difference. Forbearance had been built on the fantasy that if you held the assets long enough, they'd recover and the loss would never have to be booked. The RTC accepted the opposite premise: the loss is real, it already happened, and the only question is how much of the value can be salvaged by selling fast and selling smart. It pioneered techniques that are now standard in every bank-failure cleanup — bulk auctions of distressed property, equity partnerships where private investors took the upside in exchange for managing the workout, and the securitization of pools of bad commercial mortgages into bonds that could be sold to investors at a discount. None of this made the loss smaller in some accounting trick; it made the *recovery* larger by getting the dead assets into the hands of people who could actually do something with them. That is the difference between forbearance and resolution: one hides the loss and lets it grow, the other recognizes it and recovers what it can.

![How the RTC seized failed thrifts protected savers and sold the assets](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-8.png)

The pipeline above shows the RTC's four-step process. **Seize:** close the insolvent thrift and take control. **Protect savers:** pay off insured depositors up to the limit, so ordinary households lost nothing on their insured money. **Take the assets:** inherit the bad loans, the land, the foreclosed property onto the RTC's own books. **Manage and sell:** dispose of it all through auctions, bulk sales, and securitization — the RTC was a pioneer of packaging junk commercial mortgages into sellable securities. Then **tally** what it couldn't recover.

Between 1989 and 1995 the RTC resolved hundreds of thrifts holding hundreds of billions of dollars in assets. It is widely credited as one of the more effective government cleanups precisely because, unlike the years of forbearance that preceded it, it stopped pretending and started selling.

![The S&L crisis in two numbers about 1043 thrifts and the dollar cost](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-3.png)

The figure above shows the scorecard. Across 1986–1995, about **1,043 insured thrifts failed.** The direct cost to taxpayers came to roughly **\$124 billion**, with the all-in resolution cost (including the industry's own contributions) around **\$160 billion.** To grasp the scale, \$124 billion in early-1990s dollars is well over \$250 billion in today's money — at the time, the most expensive financial cleanup in American history, a record it held until 2008.

#### Worked example: the taxpayer bill, and what forbearance cost

Here's a calculation that captures the real lesson. Suppose the industry's true hole in 1982 — the market-value insolvency from the interest-rate losses alone — was on the order of \$25 billion. (Estimates vary; this is a defensible round figure for the early-1980s rate damage.) Had the government closed the insolvent thrifts then and paid off insured depositors, that's roughly what it would have cost.

- Cost of acting in 1982: about \$25 billion.
- Actual taxpayer cost by the time the RTC finished: about \$124 billion.
- **The price of waiting: roughly \$124B − \$25B = about \$99 billion of extra loss**, manufactured by the years of forbearance and the gambling it invited.

In other words, the policy of keeping zombies alive in the hope they'd recover didn't save the early loss — it *quadrupled* it. The one-sentence intuition: forbearance is almost never free; letting an insolvent bank keep operating doesn't postpone the loss, it compounds it, because a dead institution with insured funding and nothing to lose will gamble the bill ever higher. That single lesson — *recognize losses and close insolvent banks fast* — is the most expensive thing the S&L crisis taught, and it is the lesson regulators try to honor every time a bank fails today.

## Common misconceptions

**"The S&L crisis was mainly about fraud and crooks like Keating."** No. The fraud was real and produced over a thousand convictions, but most analysts attribute only about 10–25% of the losses to outright criminal fraud. The bulk came from the interest-rate mismatch (the duration trap) and from legal-but-reckless speculation that deregulation invited. The disease was moral hazard; fraud was a symptom. If you remember the crooks but forget the duration gap, you've learned the wrong lesson.

**"Deregulation caused the crisis."** Partly, but the order matters. The thrift industry was *already* insolvent from the Volcker rate shock *before* the worst deregulation arrived. Deregulation (Garn–St. Germain, the higher insurance limit, lower capital rules) didn't create the hole — it determined how the industry behaved while sitting in the hole, and it turned a roughly \$25 billion interest-rate problem into a roughly \$124 billion gambling-and-fraud catastrophe. The first cause was rates; the multiplier was deregulation plus forbearance.

**"Deposit insurance is what caused the moral hazard, so it's bad."** Deposit insurance is genuinely valuable — it stops bank runs by ordinary savers, and during the crisis it meant insured households lost essentially nothing. The problem wasn't insurance itself but insurance *without matching supervision*: raising the limit to \$100,000 while loosening capital rules and tolerating insolvency removed every market check on reckless thrifts at once. Insurance plus weak supervision is the toxic combination; insurance plus tough supervision (the post-FIRREA model) is the workable one.

**"The taxpayer lost \$124 billion that just vanished."** The \$124 billion was a real cost, but it's the *net* figure after the RTC recovered substantial value by selling the seized assets. The headline transaction sizes were much larger; the RTC clawed back a large fraction by working out and selling loans and property over years. The lesson is not that the money disappeared but that an enormous amount of real economic value — half-built malls, speculative land, bad loans — was destroyed by misallocated insured capital, and the taxpayer covered the gap the insurance fund couldn't.

**"This was a one-off; we fixed it and moved on."** The specific thrift structure was largely fixed, but the underlying trap — borrow short, lend long, get killed by a rate shock, then face the temptation to forbear — is permanent. It recurred at full systemic scale in 2008 and in miniature at Silicon Valley Bank in 2023. The names change; the duration gap does not.

## How it shows up in real banks

The S&L crisis is not a museum piece. The exact mechanisms — the duration trap, the temptation of forbearance, the moral hazard of a backstop — keep resurfacing. Here is the same disease in four other bodies.

### Silicon Valley Bank, 2023: the duration trap at digital speed

In March 2023, Silicon Valley Bank failed in about 36 hours. Strip away the modern details and it is the S&L crisis compressed into a weekend. SVB had taken in a flood of deposits from tech startups during the 2020–2021 boom and parked roughly \$91 billion of it in long-dated Treasuries and mortgage-backed securities — long-duration assets — funded by deposits that could leave instantly. When the Fed hiked rates aggressively through 2022, the market value of that bond book collapsed by more than \$15 billion, more than the bank's entire equity. SVB was, on a mark-to-market basis, insolvent — a zombie, exactly like a 1982 thrift. The difference was speed: instead of a decade-long grind, an online deposit run drained tens of billions in a day, because deposits now move at the speed of a mobile app and a group chat. Same trap, four decades and one technology cycle later.

![Same duration trap S and L crisis versus Silicon Valley Bank](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-7.png)

The matrix above lines them up. The mismatch is identical — deposits funding long fixed-rate assets. The rate shock is identical — a central bank hiking hard, inverting the spread and gutting the long assets' value. What differs is the speed (a decade of bleed versus a 36-hour digital run) and the asset type (30-year mortgages versus long Treasuries and MBS). The deep lesson is that interest-rate risk in the banking book is not an exotic edge case; it is the original, recurring way a maturity-transforming institution dies. For the full mechanics of measuring and managing it, see the duration-gap deep dive linked at the end.

### The 2008 mortgage meltdown: the same machine, securitized

The 2008 crisis rhymed with the S&L story in its incentives. Originate-to-distribute mortgage lenders, like the gambling thrifts, made loans they didn't have to live with the consequences of — they sold the risk onward, so the link between making a loan and bearing its loss was severed, exactly as deposit insurance had severed it for thrifts. Once again, an implicit backstop (the belief that house prices only rose, and that the government would step in) created moral hazard at scale, and once again the bill landed on the public. WaMu, the largest U.S. bank failure ever, was at its core a thrift that died of the 2008 version of the disease.

![FDIC insured bank failures per year from 2005 to 2025](/imgs/blogs/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures-6.png)

The chart above shows the modern record of bank failures, and it's worth holding next to the S&L numbers to feel the difference in shape. The 2008–2012 wave was sharp and concentrated — 157 banks failed in 2010 alone, the post-2008 peak — but it burned out in a few years. The S&L crisis, by contrast, was a slow grind: roughly a thousand thrifts over a *decade*, a steady drip of failures that never spiked as high in any single year but accumulated into the larger total. The shape tells you something about the cause. A fast spike (2008, 2023) is a panic — a run, a sudden mark-to-market shock. A long grind (the 1980s) is a structural insolvency that forbearance kept on life support for years. Both are duration-trap deaths; one happened all at once, the other in slow motion.

### Japan's lost decade: forbearance as national policy

After Japan's asset bubble burst around 1990, its banks were riddled with bad real-estate loans — they were zombies. Rather than force recognition and closure, Japanese authorities practiced forbearance on a national scale for years, letting zombie banks keep zombie borrowers alive. The result wasn't a quick, painful cleanup but a long, grinding stagnation — the "lost decade" (which stretched into two). Japan is the cautionary twin of the S&L story: the United States eventually bit the bullet with FIRREA and the RTC; Japan delayed, and paid in a different, slower currency. Both prove the same theorem: forbearance compounds the cost.

### Continental Illinois and the birth of "too big to fail"

In 1984, Continental Illinois — then the seventh-largest U.S. bank — suffered a wholesale-funding run after its bad energy loans came to light. Regulators, fearing a chain reaction, guaranteed *all* its depositors, even the uninsured ones above the limit. The phrase "too big to fail" entered the language. It is the macro version of the S&L moral hazard: once the market believes the government will backstop a large bank's creditors, those creditors stop policing the bank's risk, and the bank takes more of it. Every backstop buys stability today at the price of moral hazard tomorrow — the central, unavoidable trade-off the S&L crisis dramatized and that every regulator since has had to manage.

## The takeaway / How to use this

If you remember one thing from the savings and loan crisis, make it this: **a bank is a leveraged, confidence-funded maturity-transformation machine, and the maturity it transforms is exactly where it is most likely to die.** Borrowing short and lending long is the trade that makes banking useful and the trade that makes banking fragile — they are the same trade. The S&L crisis is what that fragility looks like when a generational rate shock hits an industry that had built its entire balance sheet on the assumption that short rates would stay put.

So here is how to actually *use* this when you read about banks:

**Always ask where the duration gap is.** When you look at any bank — a 1982 thrift, SVB in 2023, the next one — the first question is: how long are its assets locked in, and how fast can its funding reprice or run? A big gap is a loaded gun. It's harmless while rates are calm and lethal when they jump. You don't need to predict the rate shock; you only need to notice the gap and respect that it will eventually be tested.

**Distinguish insolvency from illiquidity, and watch how the backstop is used.** A thrift bleeding negative carry is *insolvent* — it has a real net-worth problem that time only worsens. The right response to insolvency is fast recognition and closure. The wrong response is forbearance, which feels merciful and is actually the single most expensive choice available, because it hands an insolvent institution a backstop and a license to gamble. When you see regulators relaxing rules to keep weak banks "alive," that is not stability — it is the cost being deferred and multiplied.

**Respect moral hazard as a force of nature.** Any time someone keeps the upside of a bet while the public keeps the downside, they will take more risk — not because they're evil, but because the incentives demand it. Deposit insurance, lender-of-last-resort support, and "too big to fail" guarantees are all genuinely valuable tools that all generate moral hazard, and the entire craft of bank regulation is the attempt to keep the stabilizing benefit while clawing back the risk-taking it invites. The S&L crisis is the case study of what happens when you take the backstop and drop the supervision.

The thrift that was the safest business in America in 1966 was a smoking crater by 1986 because the world stopped cooperating with its one fragile assumption. The lesson isn't that banking is doomed — it's that the spread business has a built-in trap, that the trap is the duration gap, and that the most expensive mistake a regulator can make is to keep a dead institution breathing in the hope the trap will spring itself open again. It won't. It never does.

## Further reading & cross-links

- [Interest-rate risk in the banking book (IRRBB) and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the full mechanics of the trap that destroyed the thrifts: repricing gap, duration of equity, and how a modern bank measures and hedges it.
- [Silicon Valley Bank 2023: the duration trap and the 36-hour digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) — the same mismatch four decades later, at the speed of a mobile app.
- [Deposit insurance, the lender of last resort, and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — why the backstop that stops runs also breeds the risk-taking the S&L crisis exposed.
- [Non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process) — what the RTC was really doing when it seized and sold the thrifts' bad assets: the workout, recovery, and write-off process at scale.

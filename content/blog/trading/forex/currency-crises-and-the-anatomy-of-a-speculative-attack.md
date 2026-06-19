---
title: "Currency Crises and the Anatomy of a Speculative Attack: How a Peg Breaks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a fixed exchange rate actually breaks — the misaligned peg, finite reserves, the asymmetric one-way bet, and the self-fulfilling run — told as the anatomy of a speculative attack, from 1992 to 1997 to 1998."
tags: ["forex", "currencies", "currency-crisis", "speculative-attack", "pegs", "reserves", "first-generation", "second-generation", "self-fulfilling", "erm", "asian-crisis"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A currency crisis is what happens when a fixed exchange rate that no longer reflects reality is attacked by traders who can lose only a little and win a lot. A misaligned peg, plus a finite pile of reserves to defend it, plus an asymmetric one-way bet, equals a speculative attack.
>
> - A peg is a *promise* — the central bank will trade its currency for dollars at a fixed price. That promise is only as good as the reserves behind it, and reserves are finite. When they run low, the promise becomes a bluff.
> - Shorting a peg is the best risk-reward trade in finance: if the peg holds you lose a small carry cost (a few percent a year); if it breaks you make 20%, 30%, 50% in days. Small capped loss, huge uncapped gain. That asymmetry is the engine of every attack.
> - **First-generation** crises are about bad fundamentals — the peg was always doomed and reserves simply run out. **Second-generation** crises are self-fulfilling — the peg *could* have survived, but because everyone expected it to break, attacking became rational and the expectation made itself true.
> - The one number to remember: in 1992 the Bank of England raised rates from 10% to 12% to 15% in a single day to defend the pound — and still lost. Reserves and rate hikes are the ammunition, and the ammunition runs out.

On the morning of 16 September 1992, the Bank of England did something it had not done in living memory. In the space of a few hours it raised its base interest rate twice — from 10% to 12%, then announced it would go to 15% — a brutal, panicked attempt to make holding the pound irresistible and shorting it ruinous. By the evening the rate hikes had been abandoned and the pound had been forced out of Europe's Exchange Rate Mechanism. The defence had failed completely. On the other side of that trade, a hedge fund run by George Soros had reportedly made over a billion dollars in a matter of days. The day is remembered, with very British understatement, as Black Wednesday.

What happened that day was not bad luck and it was not a single villain. It was the predictable end state of a particular structure: a fixed exchange rate that the market no longer believed in, a central bank with a finite war chest to defend it, and a wall of traders who understood that betting against the peg was a coin-flip where heads paid ten times what tails cost. That structure has a name — a **speculative attack** — and it has played out in almost the same shape over and over: the British pound in 1992, the Thai baht and Indonesian rupiah in 1997, the Russian ruble in 1998, the Swiss franc in 2015. Different countries, different decades, the same anatomy.

This post dissects that anatomy. We are going to take a currency crisis apart the way you would take apart a clock — laying every gear on the table so you can see exactly how the mechanism turns. We will keep the narrative; the formal mathematical model of the attack (the coordination game, the equilibrium algebra) lives in the game-theory track and we link out to it rather than re-deriving it here. What you will walk away with is the ability to look at any pegged currency and ask the three questions that decide its fate: *Is the peg misaligned? How much ammunition is behind it? And is the bet against it one-way?*

![The one-way bet against a peg as a before-and-after comparison](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-1.png)

The spine of this whole series is that you never own *a currency* in isolation — every position is a pair, a relative bet, and what moves it is the gap between two countries' interest rates plus the flow of money across borders. A speculative attack is that spine at its most violent. The "relative bet" is a trader selling the pegged currency to buy dollars. The "rate gap" is the interest the central bank must offer to keep money from leaving. And the "flow of money across borders" is the capital flight that drains the reserves. Hold those three ideas and the rest of this post is just mechanism.

## Foundations: The anatomy of a speculative attack

Let us build the thing from absolutely nothing. To understand a currency crisis you need four ideas, and once you have all four the crisis assembles itself almost automatically. The four are: what a peg actually *is*, why a peg becomes *misaligned*, why reserves are *finite*, and why the bet against a peg is *one-way*. We will take them one at a time, define every word, and use plain money examples before any number gets fancy.

### What a peg actually is — a promise, not a law

A fixed exchange rate, or **peg**, is a public promise by a country's central bank: *we will buy and sell our currency for a foreign anchor currency at a fixed price, no matter what.* Thailand in the 1990s promised to keep the baht at roughly 25 to the US dollar. Britain in the early 1990s promised, via the European Exchange Rate Mechanism, to keep the pound inside a narrow band against the German mark. Switzerland in 2011 promised the franc would never be allowed to strengthen past 1.20 per euro.

Here is the crucial part that beginners miss. A peg is not a physical law like gravity. It is not enforced by some external rule that the market *cannot* break. It is enforced entirely by the central bank standing in the market and trading. If too many people want to sell the currency and the price starts to fall below the promised level, the central bank itself must buy — it sells its hoard of dollars and buys up its own currency to prop up the price. The peg holds for exactly as long as the central bank keeps showing up to that trade, and not one second longer.

That is the whole vulnerability in one sentence: **a peg is a promise backed by a finite resource.** The resource is the central bank's stock of foreign currency, its *reserves*. As long as the reserves last, the promise is credible. When they run low, the promise is a bluff — and markets are very, very good at calling bluffs.

#### Worked example: what defending a peg actually costs

Suppose a central bank pegs its currency, call it the "ringgit", at exactly 4.00 ringgit per dollar. A wave of selling hits: exporters, importers hedging, and speculators all want to sell ringgit and get dollars, and they want to do it *at the peg* before it can fall. Say \$500 million of ringgit gets dumped on the market in a day. To hold the price at 4.00, the central bank must be the buyer of last resort — it must take the other side and buy that \$500 million worth of its own currency.

To buy ringgit it must *sell dollars*. So defending the peg for one day of \$500 million in selling costs the central bank \$500 million of its reserves, gone. If the bank started with \$30 billion of reserves, it can absorb that pace for sixty days. If the selling accelerates to \$2 billion a day — which it does in a real attack — the same \$30 billion lasts fifteen days. The intuition: every dollar of defence is a dollar of reserves spent, and the faster the attack, the faster the war chest empties.

### Why a peg becomes misaligned — the price stops telling the truth

A peg works beautifully when the fixed price is close to what the currency is *actually* worth. The trouble starts when the world moves and the peg does not. This is **misalignment**: the fixed price drifts away from the currency's fair value, and the gap keeps widening.

How does a peg become misaligned? Usually through inflation and trade. Suppose your country pegs to the dollar but runs much higher inflation than the United States — prices at home rise 8% a year while US prices rise 2%. After a few years your goods are far more expensive than American goods at the fixed exchange rate, so nobody wants to buy your exports and everybody wants to buy cheap imports. Money flows *out* to pay for those imports. The market's honest opinion is that your currency should be weaker — that the real exchange rate has gotten too strong — but the peg holds the nominal price fixed anyway. (This is the real-exchange-rate logic; the full treatment lives in [purchasing power parity and the real exchange rate](/blog/trading/forex/purchasing-power-parity-and-the-real-exchange-rate).)

A misaligned peg is a stretched rubber band. The longer it is held away from fair value, the more tension builds, and the more violent the eventual snap. The chart below shows the most famous snap of them all — the pound's exit from the ERM in 1992, where a currency held at an artificially strong level finally let go and fell more than 14% in three months.

![The pound breaking its ERM floor in autumn 1992](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-2.png)

The pound had been forced into the ERM at a rate that suited German reunification politics far more than it suited the British economy. Britain was in recession and needed *lower* interest rates; the peg to a strong mark demanded *higher* ones. The rubber band stretched for two years. When it snapped, it snapped hard.

#### Worked example: the misalignment gap in numbers

Take a currency pegged at 25 per dollar — the baht's old level. Suppose over five years the country runs 6% inflation versus 2% in the US, a 4% gap each year. Compounded over five years, domestic prices rise about 34% while US prices rise about 10%. Roughly speaking, the currency has become about \$22\%\$ "too expensive" in real terms at the fixed rate (`1.34 / 1.10 - 1 ≈ 0.218`).

A trader looks at that and thinks: the fair value of this currency is something like 30 per dollar, maybe worse, but the central bank is still selling dollars at 25. If the peg breaks and the currency falls to 30, that is a 20% move (`25/30 - 1 ≈ -0.167` in dollar terms, or a +20% rise in the price of a dollar). The misalignment *is* the size of the prize. The bigger the gap between the peg and fair value, the larger the payoff waiting for whoever bets on the snap.

### Why reserves are finite — the ammunition runs out

We have established that defending a peg means spending reserves. The third foundation is simply this: **reserves are finite, and everybody can see roughly how much is left.** A central bank publishes its foreign-exchange reserves, usually monthly. Markets watch that number like a fuel gauge. When it starts dropping fast, the attackers know the defence has a clock on it.

There is a standard yardstick for "how much is enough": months of import cover. If a country's reserves can pay for, say, three months of its imports, that is considered a bare-minimum buffer. Less than that and the country is one shock away from being unable to pay for the goods it needs. The chart below puts several emerging markets on that yardstick — and shows how thin some buffers are.

![Reserves measured in months of import cover across emerging markets](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-4.png)

Notice the logic the chart encodes. A country with fourteen months of cover, like China, has an enormous war chest — attacking its currency is like charging a fortress. A country sitting right at or below the three-month line has a thin defence, and attackers know that one determined wave of selling can crack it. Reserves are not just a number; they are the *credibility* of the promise. The smaller they are, the louder the market whispers that the bluff can be called.

And here is the cruel twist that makes reserves even more fragile than the headline number suggests: a central bank rarely spends its *last* dollar defending a peg. Long before reserves hit zero, the government needs them for other things — paying for imported food and fuel, servicing foreign debt, keeping the banking system solvent. So the *usable* reserves for FX defence are well below the total. The market knows this, which is why attacks often succeed with reserves still nominally positive. The fortress falls before the last brick is gone.

#### Worked example: how many days the war chest lasts

A country holds \$30 billion in reserves and pegs its currency. In calm times, daily two-way flow nets to roughly zero — the central bank barely touches its reserves. Then doubt sets in and the attack begins.

- **Day 1–10:** selling runs at \$500 million a day. Reserves fall \$5 billion, to \$25 billion. Manageable.
- **Day 11–20:** the published reserve drop spooks more sellers; the pace doubles to \$1 billion a day. Another \$10 billion gone — down to \$15 billion.
- **Day 21–25:** now everyone can see the fuel gauge, the attack goes vertical at \$3 billion a day. Five days erases \$15 billion. Reserves hit zero.

The peg that looked rock-solid with \$30 billion behind it is gone in about three and a half weeks. And in practice the central bank surrenders earlier — say at \$8 billion, the floor it needs for debt service — so the defence collapses around day 22. The lesson: **a reserve pile is a stock, and an attack is a flow, and a fast enough flow drains any stock.** This is exactly why the next idea matters so much — the asymmetry that makes the flow accelerate.

### Why the bet against a peg is one-way — the heart of the matter

Now we arrive at the engine of the whole thing, the single most important idea in currency crises: the bet against a peg is **asymmetric**. It is a one-way bet. Heads you win big; tails you lose a little.

Think through the position of a trader who *shorts* a pegged currency — that is, who borrows the currency and sells it for dollars, betting it will fall. Two things can happen:

1. **The peg holds.** The currency stays exactly where it was. The trader closes the position and has lost only the *carry cost* — the interest-rate difference they had to pay to borrow the currency and hold dollars, plus transaction costs. On a pegged currency that is typically a few percent a year, and the trader can scale that down by only holding the bet for the weeks the attack is live. Call it a 2–3% loss, capped and known in advance.
2. **The peg breaks.** The currency devalues — by 15%, 30%, 50%, sometimes more. The trader's short pays off in full. The gain is enormous and *uncapped*; the bigger the misalignment, the bigger the win.

That payoff is the figure that opened this post. A small, capped, known loss on one side; a large, uncapped, gain on the other. There is no symmetric downside, because a pegged currency *cannot rise* meaningfully — the whole point of the peg is to cap it. So the trader risks 3% to make 30%. A ten-to-one payoff on a bet where the central bank's own dwindling reserves are quietly tilting the odds in your favour.

This asymmetry is why attacks are not just possible but *attractive*, and why they snowball. The carry cost gates the early sellers, but as the central bank hikes rates to defend, something perverse happens — wait, the carry cost goes *up*, which should deter shorts. True. But the rate hike also screams that the defence is desperate, and a desperate defence is a signal that the prize is near. We will see this tension play out in a moment. For now, lock in the core: **the option-like payoff of shorting a peg — small capped loss, huge uncapped gain — is the gravitational force that pulls capital into the attack.** (Options traders will recognise this immediately as being *long a put on the currency for the price of the carry*; the vanilla-Greeks treatment lives in [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more).)

#### Worked example: the asymmetric payoff of a \$1B short

A macro fund puts on a \$1 billion short against a currency pegged at 4.00 per dollar, with \$30 billion of reserves behind it. The carry cost — the rate they pay to borrow the local currency minus what they earn on dollars — is 6% a year, but the fund expects the attack to resolve in roughly one month. So the *cost of being wrong* is about `6% × (1/12) × $1,000,000,000 ≈ $5,000,000`. Five million dollars at risk.

Now the upside. If the peg breaks and the currency devalues 30% — a routine crisis move — the \$1 billion short gains roughly `30% × $1,000,000,000 = $300,000,000`. Three hundred million dollars.

So the fund risks \$5 million to make \$300 million: a **60-to-1** payoff. Even if the fund thinks the peg has only a 1-in-20 chance of breaking this month, the math is a screaming "yes" — expected value is `(1/20 × $300M) − (19/20 × $5M) ≈ $15M − $4.75M = +$10.25M`. The asymmetry is so violent that you don't even need to be confident the peg will break to *want* the trade. That is what a central bank is up against: not a forecast, but a coin-flip the whole market wants to take.

### First generation versus second generation — two ways a peg dies

The last foundation is a distinction that organises everything. Economists sort currency crises into two families, and knowing which family you are looking at tells you whether the peg was *doomed* or merely *fragile*.

A **first-generation crisis** is about fundamentals. The country runs bad policy — big budget deficits financed by printing money, or persistent high inflation — and that policy is fundamentally incompatible with the peg. Reserves bleed out steadily because the currency is genuinely overvalued, and at some calculable point they hit the floor and the peg *must* break. The attack just brings forward the inevitable; speculators pick the moment, but the outcome was written into the fundamentals. The 1998 Russian crisis is the classic example — a government that could not fund itself and a peg that arithmetic doomed.

A **second-generation crisis** is different and more unsettling. Here the fundamentals are *ambiguous*. The peg is defensible — the central bank *could* hold it if it were willing to pay the price (sky-high interest rates, a deep recession). But that defence is costly, and whether the bank chooses to pay depends on how committed the market thinks it is. If traders believe the bank will defend, they don't attack, and the peg survives. If traders believe the bank will fold, they attack, the cost of defending rises, the bank folds, and the peg breaks. The crisis is **self-fulfilling**: the expectation creates the outcome. Britain in 1992 is the textbook second-generation case — the pound could arguably have been held, but the cost was a deeper recession the government would not accept, and once the market sensed that, the attack became unstoppable.

![A matrix comparing first-generation and second-generation currency crises](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-5.png)

The distinction matters enormously for how you read a crisis. In a first-generation case, defending is throwing good money after bad — the peg is dead, the only question is when. In a second-generation case, the *defence itself* is the variable; a credible enough central bank never gets attacked, while a wavering one invites the very attack it fears. The formal coordination model behind the second-generation case — where multiple equilibria exist and beliefs select between them — is exactly the structure of a bank run, and we treat it formally in [bank runs as coordination games](/blog/trading/game-theory/bank-runs-as-coordination-games-diamond-dybvig-and-svb) and [the central bank credibility game](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed). Here we keep the narrative; the algebra lives there.

## The mechanic of the attack — how the gears actually turn

We have the four foundations. Now let us watch them interlock into a running machine. A speculative attack is a chain of cause and effect that, once it gets going, has a brutal internal logic. The figure below is that chain.

![The mechanic of a speculative attack as a flow graph](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-3.png)

Trace it from the left. It begins with **doubt** — a published reserve number that fell more than expected, a worse-than-feared trade deficit, a political wobble, a neighbouring country that just devalued. The doubt is the spark. It tells traders the one-way bet might be live.

Doubt triggers **selling**. Traders sell the currency and demand dollars. Crucially, they demand those dollars *at the peg* — the whole point is to convert at the favourable fixed rate before it can move. Every seller who converts at the peg is, mechanically, draining the central bank's reserves, because the bank is the counterparty buying back its own currency. So selling becomes **reserves draining**, and the fuel gauge starts dropping where everyone can see it.

Now the central bank has two levers and both are bad. Lever one: keep selling reserves to hold the price — but that just empties the war chest faster. Lever two: **hike interest rates** to make holding the currency attractive and shorting it expensive. This is the classic defence. But raising rates is a sledgehammer that hits the whole economy: it chokes off lending, hurts borrowers, deepens any recession, and weakens the banks. So the rate-hike defence creates **economic pain**, and the more pain it inflicts, the less credible it becomes — because the market knows a government can only tolerate so much before it caves.

That is the fork at the right of the diagram. Either the defence works — rates high enough, reserves deep enough, resolve credible enough — and the peg **holds**, leaving the attackers with their small capped loss. Or the reserves run dry, or the political pain becomes unbearable, and the peg **breaks**, handing the attackers their enormous uncapped gain. The entire crisis is the resolution of that one fork.

### The rate-hike paradox — why the defence can backfire

The rate-hike lever deserves a closer look because it contains a genuine paradox. Raising interest rates is *supposed* to defend the currency — and in textbook theory it does, by raising the carry cost of shorting and by attracting yield-hungry capital. But in a live attack, a rate hike can *accelerate* the attack instead.

Here is why. A massive, panicky rate hike is a confession. It tells the market: *we are in trouble, the normal defence isn't working, we're reaching for the emergency lever.* And it inflicts real, visible economic damage — on mortgages, on businesses, on banks. Every observer now does a calculation: *how long can the government tolerate 15% rates in a recession before politics forces it to stop?* If the answer is "not very long", then the rate hike, far from deterring the attack, has just told everyone exactly how close the prize is. The defence has signalled its own fragility.

This is precisely the Black Wednesday story. The Bank of England's frantic intraday hikes — 10% to 12% to a promised 15% — were meant to crush the shorts. Instead they convinced the market that the defence was desperate and doomed, because everyone could see that 15% rates were politically impossible to sustain in a British recession. The hikes were a tell, not a deterrent. By evening the government folded.

#### Worked example: the rate-hike ladder that failed

Walk the famous ladder. On the morning of 16 September 1992 the UK base rate was 10%. As the pound hit its ERM floor and the selling intensified, the Bank raised it to 12% at 11:00. The selling did not stop. At 14:15 the Bank announced a further rise to 15%, effective the next day. The selling *still* did not stop.

Why didn't a 5-percentage-point hike — enormous by any standard — break the attack? Because of the very asymmetry we built. A short seller borrowing pounds now paid 15% a year instead of 10%, an extra 5%. But over the few days the attack would take to resolve, that extra cost was tiny: `5% × (3/365) ≈ 0.04%` for a three-day hold. Against an expected devaluation of 10–15%, an extra four *hundredths* of a percent in carry was a rounding error. The hike that looked devastating to the British economy was trivial to the attackers. By evening the UK left the ERM, the rate hikes were reversed, and the pound fell. The defence's biggest weapon barely scratched the trade it was aimed at.

### Two losing levers — why both defences feed the attack

Step back and notice the trap the central bank is in. It has exactly two tools to defend a peg under attack, and in a serious attack *both* of them quietly help the attacker.

The first lever is **direct intervention**: stand in the spot market and buy your own currency with reserves. This is the most natural defence and it works in the short run — every dollar you spend props up the price for that day. But it spends down the one resource the attackers are watching, and the spending is at least partly visible in the reserve report. So direct intervention buys price support today at the cost of revealing, tomorrow, exactly how much ammunition is left. The more aggressively you intervene, the faster the gauge falls and the louder the market hears that the defence has a clock on it.

The second lever is the **rate hike** we just dissected. It tightens the carry screw on shorts, but in a panic it broadcasts desperation and inflicts economic pain that caps how long the government can keep it up. So intervention spends your *reserves* and rate hikes spend your *political capital* — and both are finite, both are watched, and both signal weakness when used hard. A trader watching a central bank pull both levers at once is not seeing a fortress; they are seeing a defender burning through their last two resources in public. That is the deep reason a determined attack on a misaligned peg so often wins: the very act of defending hands the market more information that the defence is breakable.

There is a third "lever" that is really a confession of defeat — **capital controls**, where the government simply forbids people from selling the currency or moving money out. Malaysia reached for this in 1997. It can stop the bleeding, but at the cost of telling the world the peg could not survive on its merits, freezing foreign investment, and inviting a black-market exchange rate that prices in the devaluation anyway. Controls do not make the misalignment go away; they just hide it behind a wall. (This is the trilemma's iron logic — you cannot have a fixed rate, free capital flows, and an independent policy all at once, treated in [the impossible trinity](/blog/trading/forex/the-impossible-trinity-pick-two-of-three).)

#### Worked example: intervention versus the carry, side by side

Put numbers on the trap. A central bank with \$30 billion of reserves faces \$2 billion of daily selling. It can choose how to split its defence.

- **All intervention.** Spend \$2 billion a day buying the currency. The peg holds, but reserves fall to \$20 billion after five days, \$10 billion after ten — and the published drop draws in fresh sellers, so day eleven's selling is \$3 billion, not \$2 billion. The defence accelerates its own demise. Reserves are exhausted in roughly two weeks.
- **All rate hikes.** Raise rates by 8 percentage points to make the short painful. The extra carry over a two-week attack is `8% × (14/365) ≈ 0.31%`. Against a feared 25% devaluation, the attacker shrugs — 0.31% is a tax of barely one part in eighty on the prize. Meanwhile the 8% rate has cratered the domestic economy, and the market starts counting the days until the politics snap.

Neither lever alone works, and using both at once just empties both resources in parallel. The only thing that would have worked was *not being misaligned in the first place* — or having credibility so total that the attack never started. The defender's whole problem is that by the time they are pulling levers, it is already too late.

## Reserves draining — watching the fuel gauge

We touched on reserves as a foundation; now let us sit with the dynamic, because the *draining* of reserves is where an attack becomes visible and self-reinforcing. The reserve number is the one piece of the central bank's hand that the market can partly see, and the way it falls tells the attackers whether to press.

Consider the feedback. Reserves fall because of selling. The fall is published or estimated. The published fall *causes more selling*, because it tells every undecided trader that the war chest is emptying and the one-way bet is getting better odds. More selling drains reserves faster. Faster draining produces a worse next number. It is a doom loop with the reserve report as its heartbeat. (This is structurally the same wrecking-ball dynamic the dollar inflicts on emerging markets, treated in [the dollar as a wrecking ball](/blog/trading/forex/the-dollar-as-a-wrecking-ball-for-emerging-markets).)

This is why central banks fight so hard to *hide* the true state of their reserves during an attack. Thailand in 1997 is the infamous case: the Bank of Thailand had quietly committed most of its reserves through forward contracts that were not visible in the headline number. On paper it looked like it had over \$30 billion. In reality its *usable* reserves had been nearly exhausted defending the baht in the forward market. When the truth emerged, the defence collapsed almost instantly, because the market discovered the fuel gauge had been reading full while the tank was empty.

#### Worked example: the forward-book trick and its failure

Suppose a central bank has \$30 billion of headline reserves but wants to defend without the market seeing the drain. Instead of selling dollars in the spot market — which shows up immediately in the reserve report — it sells dollars *forward*: it agrees to deliver dollars in, say, three months at today's peg. The forward sale supports the currency now without touching the visible spot reserve number today.

So the gauge reads \$30 billion while the bank has actually pre-sold \$23 billion of dollars it must deliver in three months. Its *true* usable reserves are `$30B − $23B = $7B`. The moment the market learns this — through a leak, an IMF disclosure, or simply by inferring it from forward-market prices — the perceived war chest collapses from \$30 billion to \$7 billion overnight. The attack, which had been probing a fortress, suddenly sees a hollow shell, and the selling goes vertical. The trick buys time but converts an orderly drain into a cliff. This is roughly the mechanism by which the baht peg broke on 2 July 1997.

## The self-fulfilling run — when belief becomes the cause

The second-generation crisis deserves its own section because it is the most counterintuitive — and most important — idea in the whole subject. In a self-fulfilling run, *the peg breaks not because it had to, but because everyone expected it to.* The expectation is the cause.

The cleanest way to see it is to put yourself in one trader's seat. You hold the pegged currency. You are deciding whether to attack — to sell and join the run — or to sit tight. Your decision hinges entirely on what you think *everyone else* will do.

![The self-fulfilling run as a coordination graph](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-6.png)

Follow the two branches. If you expect everyone else to attack, then you expect reserves to drain and the peg to break — so the smart move is to attack *too*, and to do it early, before the devaluation. Your best reply to "everyone attacks" is "I attack." And when everyone reasons this way, everyone attacks, reserves drain, and the peg breaks. The fear *confirmed itself*.

But trace the other branch. If you expect everyone else to stay calm — to keep holding the currency — then you expect reserves to stay intact and the peg to survive. So your best move is to hold and collect your interest. Your best reply to "everyone stays calm" is "I stay calm." And when everyone reasons *that* way, the peg survives. The calm *also* confirmed itself.

Here is the dizzying part: *both* outcomes are stable. The same country, the same reserves, the same fundamentals can either keep its peg or lose it, and which one happens depends on nothing more than which expectation the market happens to settle on. This is what economists call **multiple equilibria**, and it is the defining feature of a second-generation crisis. There is no single "correct" answer the fundamentals dictate; there is a good outcome and a bad outcome, and beliefs choose between them. (The formal model — how a coordination game generates multiple equilibria and how a small "sunspot" can flip the market from one to the other — is exactly the [bank-run coordination game](/blog/trading/game-theory/bank-runs-as-coordination-games-diamond-dybvig-and-svb); we keep the story here and send the algebra there.)

This is why **credibility** is a central bank's most precious asset and why central bankers obsess over it. A perfectly credible central bank — one the market is *certain* will defend at any cost — never gets attacked, because no trader believes the others will attack, so the calm equilibrium locks in and the bank never has to fire a shot. A wavering central bank invites the attack it fears, because the moment the market doubts its resolve, the fear equilibrium becomes possible. The defence is won or lost in the market's *mind* before a single dollar of reserves is spent. "Don't fight the Fed" is the same coin's other face — a central bank with unquestioned credibility wins by not having to fight at all. (See [the central bank credibility game](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed).)

#### Worked example: the same peg, two destinies

Take a country with \$20 billion in reserves, a defensible but not bulletproof peg, and 1,000 traders each able to short \$50 million. Total potential attack: `1,000 × $50M = $50 billion` — more than the reserves. So *if* everyone attacks, the peg breaks for certain.

But no single trader's \$50 million can break a \$20 billion defence alone — it takes at least 400 of them coordinating (`400 × $50M = $20B`). Now each trader reasons: *if I think fewer than 400 others will attack, my short just loses the carry and the peg holds, so I stay out. If I think 400+ will attack, the peg breaks and I want to be in early.*

If the prevailing mood is calm, each trader expects fewer than 400 attackers, so each stays out — and the peg survives with reserves barely touched. If a shock — a bad headline, a neighbour's devaluation — makes each trader expect 400+ others to pile in, each attacks, more than 400 do, and the peg breaks. **Identical fundamentals, opposite outcomes, and the only thing that changed was the shared expectation.** That is a self-fulfilling currency crisis in one number: the difference between 399 and 400 expected attackers — which is exactly why the run figure above sits where it does.

## The attack timeline — the same movie, every time

Stand back from the gears and watch the whole film at normal speed. Currency crises rhyme so closely that you can write the screenplay in advance. It runs in roughly five acts.

**Act one — the calm before.** The peg has held for years. It feels permanent. Carry traders pile in to earn the high local interest rate, confident the currency won't move. Misalignment is quietly building under the surface — inflation, a widening trade deficit, an overvalued real exchange rate — but nobody is pricing it because the peg has "always" held. This complacency is itself part of the setup: it is what lets the imbalance grow large before anyone reacts.

**Act two — the spark.** Something cracks the complacency. A reserve number disappoints. A trade figure shocks. A neighbour devalues and suddenly your currency looks overvalued *relative to the neighbour's*. A government minister says something careless. The spark itself is often small; what matters is that it makes the one-way bet salient. Traders who had ignored the misalignment now see the prize.

**Act three — the attack.** Selling begins and feeds on itself. Reserves drain, the drain is published, the published drain triggers more selling. The central bank hikes rates and intervenes. For a while it may look like the defence is holding — a "successful" intervention, a bounce. But underneath, the reserves are bleeding and the credibility is fraying.

**Act four — the break.** The defence fails. Reserves hit the floor the government won't go below, or the political pain of high rates becomes intolerable, or the market simply calls the bluff and overwhelms the remaining reserves in a final vertical wave. The central bank capitulates: it floats the currency or formally devalues. The peg is gone, often in a single day.

**Act five — the overshoot and aftermath.** The currency does not fall to its fair value and stop. It *overshoots* — it falls far past fair value in the panic, because everyone who was short is now joined by everyone who must sell to cut losses, repay dollar debt, or flee. (This overshoot dynamic is the Dornbusch story, treated in [capital flows and the Dornbusch overshoot](/blog/trading/forex/capital-flows-and-the-dornbusch-overshoot).) Then comes the wreckage: inflation as imports get expensive, balance sheets blown up where firms borrowed in dollars, sometimes a banking crisis, sometimes an IMF program. Eventually the currency stabilises at a new, weaker level — and the cycle's seeds are sown again.

The reason the screenplay rhymes so reliably is that each act *causes* the next; the structure is not a coincidence but a chain. The calm of act one is what lets the misalignment grow large enough to be worth attacking. The size of that misalignment is what makes the spark of act two ignite rather than fizzle. The one-way bet is what turns the spark into the self-reinforcing attack of act three. The finite reserves and finite political tolerance are what guarantee the break of act four. And the forced, mechanical selling — dollar-debt repayment, stop-losses, capital flight — is what produces the overshoot of act five. Knowing the chain is causal, not random, is what lets a careful reader sometimes see act four coming while the market is still telling itself a story about act one. The people who made fortunes in these crises were rarely the cleverest forecasters; they were the ones who recognised the screenplay early and sized the asymmetric bet while everyone else was still insisting "the peg has always held."

#### Worked example: the five acts on a real clock — Thailand 1997

Put dates on the screenplay. Thailand pegged the baht near 25 per dollar (**act one**: years of calm, huge carry inflows, a property and credit boom built on the assumption the peg was forever). In late 1996 and early 1997 the export engine stalled and the current account deficit ballooned toward 8% of GDP — the misalignment (**act two**: the spark, as a 1996 export slump exposed an overvalued baht). Through the first half of 1997 the Bank of Thailand defended fiercely, burning through reserves and selling dollars forward to hide the drain (**act three**: the attack and the desperate, concealed defence). On 2 July 1997 it gave up and floated the baht (**act four**: the break). The baht did not settle at a modest discount — it collapsed from 25 to about 56 per dollar, a **54%** fall, far past any estimate of fair value, dragging the whole region down with it (**act five**: the overshoot and contagion). Five acts, eighteen months, the same movie. The next chart shows just how far the regional pegs fell once they snapped.

## Common misconceptions

A handful of intuitive-but-wrong beliefs cause people to misread currency crises. Let us correct them, each with a number.

**Misconception 1: "A peg is safe because the central bank controls the price."** The central bank controls the price only while its reserves last. Britain in 1992 had real reserves and the full machinery of a major central bank, and still lost in a single day. The Bank of Thailand looked like it had \$30+ billion and was effectively empty. Control of the price is rented from the reserve pile, and the rent comes due in an attack. The bank does not "set" the price so much as *subsidise* it until it can't.

**Misconception 2: "Raising interest rates always defends a currency."** In calm times, higher rates do attract capital and support the currency. In a live attack, a panicky hike can *signal weakness* and accelerate the run. The UK went from 10% to a promised 15% in one day in 1992 and still lost, because a `5% × (3/365) ≈ 0.04%` extra carry over a three-day hold was nothing against an expected 10–15% devaluation. The weapon was real but vastly outgunned.

**Misconception 3: "Only badly run countries get attacked."** This is the first-generation fallacy. Second-generation crises hit countries whose fundamentals were *defensible* — the peg could have survived, but the self-fulfilling run broke it anyway. The same \$20 billion of reserves either holds or breaks depending purely on whether the market expects 399 or 400 attackers. Good fundamentals reduce the odds; they do not make a country immune.

**Misconception 4: "Speculators cause currency crises."** Speculators are the *trigger*, not the cause. They pick the moment and they profit, but the misalignment was built by years of policy — inflation, deficits, an overvalued real rate. Soros did not make the pound overvalued; British and German policy did. He just noticed the rubber band was stretched and bet on the snap. Blaming the speculator is like blaming the person who points out the dam is about to fail.

**Misconception 5: "Once the peg breaks, the currency settles at fair value."** It overshoots. The baht's fair value in 1997 was perhaps 30–35 per dollar; it fell to 56, a 54% drop, before recovering somewhat. The break unleashes forced selling — dollar-debt repayment, loss-cutting, capital flight — that drives the currency far past fair value before any stabilising buyer steps in. The crisis low is almost never the eventual resting level.

**Misconception 6: "A central bank with huge reserves is safe."** Size helps, but it is the *usable* reserves and the *resolve* that matter, not the headline. A bank can have \$30 billion on paper and \$7 billion in reality if it has pre-committed the rest in the forward market, as Thailand did. And even genuine reserves are worthless if the market believes the government will fold before spending them — credibility, not the number, is the real defence. Switzerland in 2015 had *unlimited* ability to defend (it could print francs forever) and still broke the peg, because the willingness to bear the cost ran out. Reserves are necessary; they are never sufficient.

## How it shows up in real markets

The anatomy stays theoretical until you watch it run on real currencies. Three cases — one from each crisis family — show the gears turning in the wild.

### 1992: the ERM and the pound — a second-generation classic

Britain joined Europe's Exchange Rate Mechanism in 1990 at a rate (around 2.95 marks to the pound) that flattered German reunification politics more than British economic reality. By 1992 Britain was in recession and needed lower interest rates, but the peg to a strong mark — backed by a Bundesbank fighting its own reunification inflation with *high* rates — demanded the opposite. The rubber band stretched.

The crucial feature is that the pound's peg was *defensible* in a narrow technical sense — the UK could, in principle, have jacked rates sky-high and held it. But the *cost* was a deeper recession the government would not stomach, and once the market sensed that limit, the attack became a one-way bet on the government's pain threshold. Soros and others sold the pound massively. The Bank of England spent billions in reserves and ran the famous intraday rate-hike ladder — 10% to 12% to a promised 15% — and it all failed within hours, precisely because the hikes signalled desperation rather than resolve. Britain left the ERM on the evening of 16 September. This is the textbook **second-generation** crisis: ambiguous fundamentals, a self-fulfilling run, a defence lost in the market's mind before the reserves were even gone. (The full story, including Soros's positioning, is in [Soros and Black Wednesday](/blog/trading/forex/soros-and-black-wednesday-breaking-the-bank-of-england-1992).)

### 1997: the Asian crisis — misalignment, concealment, contagion

The Asian crisis began as a more fundamentals-driven event and then went self-fulfilling through contagion. Thailand, Indonesia, South Korea, Malaysia and others had pegged or heavily managed their currencies against the dollar, attracting a flood of carry-seeking foreign capital into property and credit booms. As the dollar strengthened in the mid-1990s and export growth stalled, the pegs became badly misaligned and current-account deficits ballooned.

Thailand cracked first. The Bank of Thailand's concealed forward-book defence — looking full while running on empty — meant the baht peg, when it broke on 2 July 1997, broke not as a gentle slide but as a cliff. And then came the truly dangerous part: **contagion.** Once the baht fell, every other regional peg suddenly looked overvalued *relative to a cheaper baht*, and the self-fulfilling logic jumped borders. Traders who would not have attacked the rupiah or won alone now expected everyone else to attack, so they all did. The chart below shows the result — five pegs that snapped within months of each other, with the Indonesian rupiah losing a staggering 83% of its value.

![The 1997 Asian pegs that snapped, ranked by their fall versus the dollar](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-7.png)

The Asian crisis is the case study in how a first-generation spark (real misalignment in Thailand) can ignite a second-generation wildfire (self-fulfilling contagion across the region). The full regional account is in [the 1997 Asian crisis](/blog/trading/forex/the-1997-asian-crisis-thb-idr-and-krw-in-freefall).

### 1998 Russia and the 2015 Swiss franc — two opposite ends (a teaser)

Two final cases bookend the spectrum and point at where this series goes next.

**1998 Russia** is the purest **first-generation** crisis. A government that could not fund its budget, financing itself with short-term debt at ruinous yields, with a peg that simple arithmetic doomed. When Russia defaulted and devalued in August 1998, the ruble went from about 6 to 21 per dollar. There was no ambiguity, no "defensible peg" — the fundamentals had written the ending in advance; the attack merely chose the date. The aftershock detonated the hedge fund Long-Term Capital Management and forced a Fed-orchestrated rescue, a reminder that a currency crisis in one country is a leverage crisis everywhere.

**2015 Switzerland** is the strange mirror image — a crisis of a peg breaking the "wrong" way. The Swiss National Bank had promised the franc would never strengthen past 1.20 per euro, and to hold that floor it had to *print francs and buy euros* without limit. The problem was not running out of reserves — a central bank can print its own currency forever — but the *cost*: its balance sheet ballooned to a size that threatened ruinous losses if the euro fell further. On 15 January 2015 the SNB simply abandoned the floor. The franc instantly surged, EUR/CHF collapsed from 1.20 toward parity in minutes, and traders short the franc were annihilated. The chart shows the gap.

![The SNB 2015 floor break, when a central bank abandoned its promise](/imgs/blogs/currency-crises-and-the-anatomy-of-a-speculative-attack-8.png)

The Swiss case flips the usual lesson on its head. Defending a peg against *strength* is not limited by reserves — you can print forever — but it is limited by the willingness to bear the balance-sheet cost. When that willingness runs out, the peg breaks just the same. A central bank can blink even when it has not run out of bullets. (The full account is in [the SNB 2015 peg break](/blog/trading/forex/the-snb-2015-peg-break-when-a-central-bank-blinks).)

## The takeaway: how to read a pegged currency

Put the clock back together and step back. What does the anatomy of a speculative attack actually tell you about how to read a currency?

First, **a peg is a promise priced in reserves, and you can read the price.** When you see a fixed or heavily managed exchange rate, the first thing to look at is not the rate but the reserves behind it — the level, the trend, and especially the *trend*. A reserve pile that is large and stable is a credible promise. A reserve pile that is falling, or that is large on paper but committed in the forward market, is a promise on borrowed time. The fuel gauge tells you more than the price.

Second, **misalignment is the size of the prize.** The further a pegged currency is held from its fair value — measured by inflation differentials, the real exchange rate, the current-account deficit — the larger the devaluation waiting on the other side of a break, and the more capital the one-way bet will attract. A peg that is roughly fairly valued can hold for decades; a peg held 20% away from fair value is a stretched rubber band, and the only question is the timing of the snap.

Third, **the one-way bet is the engine, and it never sleeps.** Because shorting a peg risks a small capped carry cost to make a large uncapped devaluation, the trade is permanently attractive whenever a break is even slightly possible. This is why central banks cannot win an attack with reserves and rate hikes once the market smells blood — the asymmetry means a 60-to-1 payoff overwhelms even a 5-percentage-point rate hike. The only durable defence is *credibility so complete that the bet never looks live* — the calm equilibrium that means the bank never has to fire a shot.

Fourth, **know which generation you are watching.** If the fundamentals are rotten — deficits, inflation, an arithmetic-doomed peg — you are in a first-generation crisis and the peg is simply waiting to die; defending it is throwing reserves into a hole. If the fundamentals are defensible but the defence is costly, you are in a second-generation crisis, and the outcome turns on belief — on whether the market settles into the fear equilibrium or the calm one. In the second case, the central bank's *resolve and credibility* are the whole ballgame, and a single shock can flip a stable peg into a broken one with nothing fundamental having changed.

And finally, tie it back to the spine of this entire series. You never own *a currency* in isolation. A speculative attack is the most extreme version of a relative bet: traders selling the pegged currency to buy the dollar, financed by the gap between two countries' interest rates, executed as a flood of money across the border that drains the reserves. The peg is just a promise to hold that relative price fixed — and a promise, in markets, is only as good as the reserves and the resolve behind it. Read the reserves, read the misalignment, read the credibility, and you can see the snap coming long before the front pages do.

## Further reading & cross-links

Within this series:

- [The Impossible Trinity: Pick Two of Three](/blog/trading/forex/the-impossible-trinity-pick-two-of-three) — the deeper reason pegs are so fragile: you cannot have a fixed exchange rate, free capital flows, and an independent monetary policy all at once.
- [Soros and Black Wednesday: Breaking the Bank of England, 1992](/blog/trading/forex/soros-and-black-wednesday-breaking-the-bank-of-england-1992) — the full 1992 ERM story and the trade that defined a second-generation crisis.
- [The SNB 2015 Peg Break: When a Central Bank Blinks](/blog/trading/forex/the-snb-2015-peg-break-when-a-central-bank-blinks) — the mirror-image crisis, where a peg broke against strength.
- [The 1997 Asian Crisis: THB, IDR and KRW in Freefall](/blog/trading/forex/the-1997-asian-crisis-thb-idr-and-krw-in-freefall) — misalignment, concealed defences, and self-fulfilling contagion across a region.
- [Capital Flows and the Dornbusch Overshoot](/blog/trading/forex/capital-flows-and-the-dornbusch-overshoot) — why a broken currency falls far past its fair value before settling.
- [The Dollar as a Wrecking Ball for Emerging Markets](/blog/trading/forex/the-dollar-as-a-wrecking-ball-for-emerging-markets) — how a strong dollar tightens the screws on pegged and managed EM currencies.

The formal models (kept out of this narrative on purpose):

- [Bank Runs as Coordination Games: Diamond-Dybvig and SVB](/blog/trading/game-theory/bank-runs-as-coordination-games-diamond-dybvig-and-svb) — the multiple-equilibria coordination model that underlies the self-fulfilling second-generation crisis.
- [The Central Bank Game: Credibility, Commitment, and Don't Fight the Fed](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed) — why credibility lets a central bank win without firing a shot, and why a wavering one invites the attack it fears.

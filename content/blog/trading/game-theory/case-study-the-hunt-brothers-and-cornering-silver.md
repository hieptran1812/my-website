---
title: "Case Study: The Hunt Brothers and Cornering Silver"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Two Texas oil heirs bought enough silver to corner the market and ran the price up eightfold, then learned the hardest lesson in game theory: the exchange that runs the game is a player too, and it can change the rules."
tags: ["game-theory", "corner", "silver", "hunt-brothers", "commodities", "futures", "leverage", "market-manipulation", "regulation", "case-study", "margin-call"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — In 1979 and early 1980, Nelson Bunker Hunt and William Herbert Hunt set out to *corner* the silver market: to own so much of the metal that everyone who had promised to deliver silver would have to buy it back from them at whatever price they named. The plan worked spectacularly, running silver from about \$6 an ounce to a record \$49.45, until the people who ran the marketplace did the one thing the Hunts never put in their model — they changed the rules.
>
> - A **corner** is a game where one player controls enough of the *deliverable supply* of an asset that the players who are short — who have sold contracts they must settle with the real thing — cannot find any to deliver, and so must buy from the cornerer at his price.
> - The corner is real power, but it ignores a hidden player: **the exchange and the regulator make the rules and can rewrite them mid-game.** COMEX's "Silver Rule 7" raised margins and restricted trading to *liquidation only* — you could sell, but you could not buy — which inverted the corner overnight.
> - The numbers: the Hunts and allies controlled roughly **100 million ounces** of physical silver plus another **~90 million ounces** through futures. Silver hit a record **\$49.45/oz spot** on Jan 18 1980, then crashed from **\$21.62 to \$10.80 in a single day** on "Silver Thursday," March 27 1980, leaving them facing a **~\$100 million margin call** they could not meet and a feared **~\$1.7 billion** loss.
> - The one rule to remember: **cornering is illegal market manipulation, and even when it is winning, the house can change the game.** Leverage that makes the gain enormous makes the unwind fatal. You are never the only strategic player at the table.

In the late 1970s, two brothers from Texas decided that paper money was doomed and silver was the only honest store of value left. They were not cranks shouting on a street corner. Nelson Bunker Hunt and William Herbert Hunt were sons of H.L. Hunt, one of the richest oil men in America, and they had the kind of money that lets a private belief become a market event. They started buying silver — not a little, not as a side bet, but in a way no private investors had ever attempted. They bought the physical metal by the ton and took it out of the market. They bought silver futures contracts and, instead of cashing them out, they demanded delivery of the actual silver. And as they bought, the price began to climb.

By January 1980, silver that had traded near \$6 an ounce a year earlier touched a record \$49.45. A pile of metal that historically sat in jewelry boxes and camera film was suddenly one of the most explosive trades on Earth, and the men who had bought most of the available supply were, on paper, billionaires several times over. Then, almost as fast as it rose, it collapsed. On a single Thursday in late March, the price was cut roughly in half. The brothers faced a margin call they could not pay, a consortium of the largest banks in America had to assemble a rescue loan to stop the failure from taking down brokers and banks with it, and within a few years both men were bankrupt and barred from trading commodities in the United States.

This was not a freak accident or a story about greed alone. It was a *game* with a precise structure — the oldest and most ruthless game in commodity markets, called a corner — and it ended the way corners almost always end, because the people who designed the squeeze forgot to count one of the players. The model below is the whole story compressed into two states: the corner holding, and then the house changing the rules. This piece is educational, not advice. The goal is to make the structure visible, so you understand both how a corner works and why it is illegal — and so you can recognize the moment the rules themselves become the weapon.

![Before and after diagram showing a corner controlling deliverable supply on the left and the exchange changing the rules on the right](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-1.png)

## Foundations: futures, delivery, deliverable supply, and what a corner really is

To understand why a man could become, in effect, the only seller of silver on Earth, we need five ideas built from zero. If you already trade commodities, skim this. If you do not, none of the rest of the story will make sense without it.

### A futures contract is a promise about the future, with a deadline

Most people buy things on the spot: you hand over cash, you get the thing, the deal is done. A *futures contract* is different. It is a standardized, legally binding promise to buy or sell a fixed amount of something — say, 5,000 troy ounces of silver, the standard COMEX contract size — at a price agreed today, but with the actual exchange of metal-for-money happening on a specific future date called the *delivery* or *expiry* date.

There are two sides to every contract. The **long** has promised to *buy* the silver at the agreed price; the long profits if the price rises. The **short** has promised to *sell* the silver at the agreed price; the short profits if the price falls. A *troy ounce* is the unit silver and gold are weighed in — slightly heavier than the ordinary ounce, about 31.1 grams. Every number in this story is quoted per troy ounce.

Here is the part most beginners miss, and it is the hinge of the whole corner. When the contract expires, it must be *settled*. There are two ways. Either the short delivers the actual physical silver to the long (this is called **physical settlement** or *taking delivery*), or — far more commonly — the two sides close out their positions for cash before expiry. A trader who is long can *roll* the contract: sell the expiring one and buy a later-dated one, never taking delivery, just keeping a bet on the price. The entire futures market is built on the quiet assumption that *almost nobody actually wants the metal.* They want the price exposure. The warehouse is a backstop, not a destination.

The Hunts broke that assumption on purpose. When their long contracts came due, they did not roll. They said: deliver the silver. Send us the metal.

#### Worked example: long, short, and the choice to take delivery

You and I trade one silver futures contract — 5,000 ounces — at \$10/oz. You are long (you promised to buy), I am short (I promised to sell). The total contract value is 5,000 × \$10 = \$50,000.

Suppose silver rises to \$12/oz at expiry. You are entitled to buy 5,000 ounces at the old \$10 price. Two ways to settle:

- **Cash close:** you sell your long position back into the market for \$12, I buy mine back. You gain (\$12 − \$10) × 5,000 = \$10,000; I lose the same \$10,000. No metal moves. This is what 98%-plus of contracts do.
- **Delivery:** you insist on the metal. I must now *find* 5,000 real ounces of deliverable silver and hand it over, receiving the agreed \$50,000. You then own physical silver worth 5,000 × \$12 = \$60,000 for which you paid \$50,000.

In the cash close, I only needed \$10,000 to settle my loss. In the delivery case, I need to source 5,000 physical ounces from *somewhere*. If silver is plentiful, fine — I buy it and deliver. But if someone has bought up most of the available metal, I am in trouble: I am legally obligated to deliver something I cannot find. The intuition: a futures market is a paper game right up until a big player decides to make it a physical one.

### Margin: the small deposit that controls a large position

One more piece of plumbing, because it is what makes a corner both possible and self-destructive. When you open a futures position you do not pay the full contract value. You post a fraction — the *initial margin*, typically around 5–10% of the notional — as a good-faith deposit with your broker, who in turn posts collateral with the exchange's clearinghouse. Every evening the exchange *marks the position to market*: it calculates your gain or loss for the day at the new settlement price and moves cash into or out of your account accordingly. If your account falls below a floor called the *maintenance margin*, you receive a *margin call* — a demand to wire in more cash by the next morning, or have your position forcibly liquidated.

Two consequences follow, and they bracket the whole story. On the way up, margin is a magnifier: a small deposit controls a large position, so a modest rise in the price produces an outsized gain on your equity, and the daily mark-to-market profits can even be withdrawn or used to finance buying *more* — a self-funding ascent. On the way down, margin is a trip-wire: because the daily loss is computed on the full notional, not on your thin deposit, a relatively small adverse move can blow through your margin and trigger a call you may not be able to meet. A trader who cannot meet a margin call is liquidated at the worst possible moment — into a falling market, alongside everyone else being liquidated. Hold that asymmetry in mind; it is the difference between the Hunts as paper billionaires in January and the Hunts facing a \$100 million call they could not pay in March.

#### Worked example: a single contract on margin, up and down

You buy one silver contract — 5,000 ounces — at \$20/oz. Notional = 5,000 × \$20 = \$100,000. At 10% initial margin you post \$10,000.

- **Silver rises \$2 to \$22.** Your gain is 5,000 × \$2 = \$10,000, credited to your account that evening. Your equity has gone from \$10,000 to \$20,000 — a **100% return** on a **10% price move.** That tenfold amplification is leverage working for you.
- **Silver instead falls \$2 to \$18.** Your loss is 5,000 × \$2 = \$10,000. Your \$10,000 of equity is gone; the position is at zero equity and you get a margin call to restore the maintenance level or be liquidated. A **10% price drop wiped out 100% of your cash.**
- **Now raise the margin requirement to 50% mid-trade** — exactly the kind of move COMEX made. To keep your one contract open you must now hold \$50,000 of equity against it, not \$10,000. If you cannot post the extra \$40,000, you are forced to sell — and so is everyone else caught by the same rule, all at once.

The same \$2 move is a fortune or a wipeout depending only on direction, and a rule change to the margin requirement can force you to sell even when the price has not moved at all. The intuition: margin turns the size of your *bet* loose from the size of your *bankroll*, and whoever controls the margin rule controls who is forced to sell and when.

### Deliverable supply is much smaller than the silver in the world

There is a lot of silver in the world — in jewelry, in flatware, in old coins, in factory inventories. But almost none of it is *deliverable* against a futures contract on short notice. To settle a COMEX contract, the silver has to be the right purity, in the right bar form, in an exchange-approved warehouse, available right now. The pool of metal that meets all those conditions — the **deliverable supply** — is a tiny fraction of all the silver that exists.

This distinction is the soft underbelly of every commodity market and the thing a cornerer attacks. You do not need to buy all the silver on Earth. You need to control enough of the *deliverable* supply that, on delivery day, the shorts cannot lay their hands on any. The grandmother's silver tea set is irrelevant; it is not in an approved warehouse and cannot be there by Thursday.

### A short sale has unbounded loss

A short in silver has done the mirror of buying. To be short, you sell silver (or a silver contract) you do not own, hoping to buy it back cheaper later. If the price falls, you win. But the asymmetry is brutal and it drives the panic in every corner. When you *buy* something at \$10, the worst case is it goes to zero and you lose \$10 — your loss is capped. When you go *short* at \$10, your gain is capped (the price can only fall to zero), but your loss is **unbounded**, because the price you must eventually pay to buy back has no ceiling. A short at \$10 can be forced to cover at \$49. That is not a typo; it is exactly the silver story.

### A corner is controlling the deliverable supply so the shorts are trapped

Now we can define the game precisely. A **corner** is a situation in which one player (or a coordinated group) buys up enough of the deliverable supply of an asset, while *also* holding long futures that demand delivery, that the players who are short cannot find any of the asset to deliver. The shorts are legally obligated to deliver metal that, in effect, only the cornerer has. So the shorts have exactly one way out: buy the metal — or buy back their contracts — *from the cornerer*, at whatever price he demands. The cornerer becomes a monopolist for a few desperate days, and the trapped shorts bid the price to absurd heights trying to escape.

A close cousin is the **short squeeze**, where a rising price forces shorts to cover and their buying pushes the price higher still, a self-reinforcing loop. A corner is the deliberate, engineered version of a squeeze: instead of waiting for the loop to start on its own, you *create* the scarcity by owning the supply. We dig into the general squeeze mechanics in [the short-squeeze game](/blog/trading/game-theory/the-short-squeeze-game-shorts-longs-brokers-and-gamma); here the point is that the Hunts did not get lucky with a squeeze, they manufactured a corner.

The economics of a corner are a fleeting, brutal **bilateral monopoly**. Ordinarily a market has many buyers and many sellers, and no single participant can dictate the price. In a corner, for a few days, the deliverable silver has effectively one seller — the cornerer — facing a crowd of buyers who are not buying because they want silver but because the law and their contracts *compel* them to. A monopolist facing buyers with inelastic, compelled demand can charge almost anything; that is why a corner price bears no relation to silver's industrial worth. The shorts' demand curve is nearly vertical: they will pay \$30, \$40, \$49, because the alternative — defaulting on a delivery obligation — is worse. The cornerer reads that vertical demand and raises his price to meet it. The whole grotesque spectacle of \$49 silver is just a monopolist pricing against captive demand, and the moment the captivity ends — when buying is no longer compelled or, as we will see, no longer *permitted* — the monopoly evaporates and the price falls back to what silver is actually worth.

One more vital word, because it shapes the ending: a corner is **market manipulation**, and it is illegal. Intentionally distorting the price of a commodity by controlling its supply is precisely the conduct that commodity law forbids. We will see the Commodity Futures Trading Commission, the CFTC, arrive at exactly that conclusion. So this is not a how-to. It is a study of why the game is powerful, why it is banned, and why even the player who builds it cannot win against the house.

## The setup: who the players were and how they accumulated

Every game needs a cast. The 1979–80 silver corner had four sets of players, and the fatal mistake — as we will see — was that the Hunts modeled three of them and forgot the fourth.

**The Hunts and their allies (the longs / cornerer).** Nelson Bunker Hunt and his brother William Herbert Hunt were the principals, with a third brother, Lamar, involved, plus a circle of wealthy allies including a group of Saudi investors through entities such as International Metals Investment Company. They believed, sincerely, that inflation would destroy the dollar and that silver was undervalued real money. Crucially, they were not only investors; by buying so much, they became the dominant force in the price. Through 1979 their accumulation accelerated. By late 1979, estimates put the Hunts and their allies in control of roughly **100 million troy ounces of physical silver** and another **~90 million ounces of long exposure through futures** — call it around 190 million ounces of silver controlled, an amount comparable to a large share of the entire annual world supply and a stunning fraction of what could actually be delivered.

**The shorts (commercial hedgers and speculators).** On the other side were the people short silver futures — silver miners and refiners hedging future production, industrial users, jewelers, and speculators betting the runaway price would reverse. Many of them were "commercials" who routinely sold futures as a normal part of business. As the price rose, their losses mounted and their need to either deliver metal or buy back contracts grew dangerous.

**The brokers and the clearinghouse.** Between the Hunts and the market sat their brokers — most notably Bache Halsey Stuart Shields — who financed the positions, collected margin, and stood between their client and the exchange. A broker who lets a client run an enormous leveraged position is itself exposed: if the client cannot pay, the broker eats the loss, and if the broker fails, the failure spreads. This is the channel through which a single trade became a systemic threat.

**The exchanges and the regulator — the house.** This is the player the Hunts under-weighted. The Commodity Exchange (COMEX) in New York and the Chicago Board of Trade (CBOT) were the venues where silver futures traded. They are not passive scoreboards. They write the rules — contract sizes, margin requirements, position limits, what trading is even allowed — and they can change those rules. Above them sits the CFTC, the federal regulator, with the power to investigate and punish manipulation. The exchanges also had their own conflict: many of their members — the brokers and floor traders — were on the short side and getting destroyed. **The house was not neutral; it was a player with its own losses to protect.**

It is worth being precise about why this fourth player is so easy to forget. In ordinary game theory we describe a game by its players, their strategies, and a fixed payoff table, then solve for equilibrium *within* that fixed structure. The Hunts, in effect, did exactly this — and did it well — for the three-player game of cornerer-versus-shorts-versus-brokers. Their analysis of *that* game was sound: trap the shorts, control the brokers' collateral, and the price has to rise. The error was a category error about the *boundary* of the game. The rules themselves are not nature; they are chosen by an agent, the exchange, who has preferences and the power to act on them. A player who can rewrite the payoff table mid-game is not playing the same game as everyone else — and treating that player's choices as fixed background is the single most expensive assumption in this entire story.

The model below puts the controlled silver next to the thin deliverable float the shorts had to settle against. This is the corner in one chart: the long side is a mountain, the deliverable side a sliver.

![Bar chart comparing about 100 million ounces of physical silver and about 90 million ounces of futures controlled by the Hunts against a thin deliverable float](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-3.png)

### Why "take delivery" was the weapon

Most large futures players are content with paper. The Hunts' innovation, if you can call manipulation an innovation, was to treat delivery as a tool. Every time they took delivery instead of rolling, they pulled real metal out of the deliverable pool and into private storage — some of it reportedly flown to Switzerland for safekeeping. Each delivery did two things at once: it made the Hunts longer in physical silver, and it shrank the supply available for the next set of shorts to deliver. The corner tightened with every expiry.

Think of the deliverable supply as the water in a bathtub and the shorts as people who have each promised to hand someone a full bucket on Thursday. The Hunts were quietly draining the tub into their own private tank. On Thursday, when everyone reached for water, the tub was nearly empty — and the only place to buy water was from the tank. The chart above is that tub-and-tank picture in numbers: roughly 190 million ounces controlled against a deliverable float that could not begin to cover the shorts' promises.

## The corner mechanics: monopoly power on delivery day

Let us make the squeeze mechanism explicit, because "they cornered the market" is a phrase people use without understanding the gears inside it.

On the day a futures contract expires, every open short must either deliver metal or have already bought back the contract. Normally this is routine: there is plenty of deliverable silver, the price of the contract converges to the spot price, and life goes on. In a corner, the routine breaks. The cornerer holds long contracts and *demands delivery*. The shorts go looking for metal to deliver and discover that the deliverable supply is locked up in the cornerer's vault. They have three options, all bad:

1. **Buy physical silver** to deliver — but the only meaningful holder is the cornerer, who will sell only at an extortionate price (if at all).
2. **Buy back their short futures** — but the buyers they need to trade with are the cornerer's longs, who will close only at a brutal price.
3. **Default** — fail to deliver, breach the contract, and face penalties and ruin.

Faced with options 1 and 2, the shorts bid against each other in a panic. Their forced buying drives the price up, which forces *more* shorts to cover, which drives the price up further. The price detaches from anything resembling silver's industrial or monetary value and instead reflects the desperation of trapped sellers. This is the squeeze loop, and a corner is the engineered version of it. We treat the self-watching, self-amplifying dynamic of prices like this in [reflexivity](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves); the corner is reflexivity weaponized.

#### Worked example: the squeeze arithmetic on a single contract

Suppose I am short 10 silver contracts — 50,000 ounces — that I sold at \$15/oz, expecting the bubble to pop. My promise: deliver 50,000 ounces, or buy back the contracts, on expiry.

The price runs to \$40/oz because the cornerer owns the deliverable metal. My choices:

- **Deliver:** I must source 50,000 ounces. The cornerer is the only seller, and he names \$40. Buying 50,000 oz at \$40 costs \$2,000,000. I receive only the \$15 contract price × 50,000 = \$750,000 on delivery. My loss: \$1,250,000.
- **Cover (buy back the contract):** I buy back at \$40 versus my \$15 sale. Loss = (\$40 − \$15) × 50,000 = \$1,250,000. Same number, by no coincidence — the two routes must cost the same or arbitrage would close the gap.

Either way I lose \$1.25 million on a position that, at the price I shorted, was \$750,000 of notional. My loss is bigger than the trade I put on, because a short's loss is unbounded. Now multiply this by every short in the market all reaching for the same nonexistent metal, and you have the engine that drove silver to \$49. The intuition: in a corner, the shorts are not trading against the market, they are trading against the one person who holds the only exit, and he sets the toll.

### Leverage: controlling far more than your cash

A corner needs scale, and scale in futures comes from **leverage** — controlling a large position with a small amount of cash. Futures are bought on *margin*: you post only a fraction of the contract's value, often around 5–10%, as a good-faith deposit, and the exchange marks your position to market every day. If it moves in your favor, your account grows; if it moves against you, you must wire in more cash — a **margin call** — to keep the position open.

Leverage is why the Hunts could control ~190 million ounces without ~190 million ounces' worth of cash. At, say, 10% margin, a \$1 of equity controls \$10 of silver. That is the up-escalator on the way up: a 10% rise in silver roughly doubles your equity. But the same leverage is a guillotine on the way down. Because your losses are measured against the *full notional*, not against your small margin slice, a fall of just a tenth of the price can wipe out 100% of a 10x-leveraged position. The chart below makes this concrete with an illustrative 100-million-ounce position carried at 10% margin: at the \$49 peak the equity is a few hundred million dollars; by the time silver falls to the low \$10s, the equity is not just gone, it is deeply negative, and the holder owes billions.

![Bar chart of mark-to-market equity on a leveraged silver long going from positive to deeply negative as the price falls from the peak to Silver Thursday](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-5.png)

#### Worked example: how a small price fall makes a giant margin call

Take an illustrative long of 100,000,000 ounces of silver carried via futures at 10% initial margin, marked at the \$49.45 peak.

- **Notional at the peak:** 100,000,000 × \$49.45 = \$4.945 billion.
- **Equity posted (10% margin):** about \$494.5 million. That is the cash actually at risk to *open* the position.
- **Now silver falls \$10/oz**, to \$39.45. The loss is the full notional move: 100,000,000 × \$10 = **\$1.0 billion**. Against \$494.5 million of equity, you are not down 20% — you are wiped out twice over and owe roughly \$500 million more than you put up.
- **A \$10 fall is only about a 20% move in the price**, yet it generates a billion-dollar loss and a colossal margin call.

This is the mathematical reason a leveraged corner is so fragile. The position that prints money as the price rises becomes an unpayable debt the instant the price reverses, and the reversal need not be large. The intuition: leverage does not change *whether* you can be wrong, it changes how fast being a little wrong becomes catastrophic.

## The spike: from \$6 to \$49 in a year

With the mechanics in hand, the price action becomes legible. Through 1979 the Hunts' buying, combined with genuine inflation fears — the late 1970s saw double-digit US inflation and a falling dollar — lit silver on fire. The number to anchor on: silver started 1979 at about **\$6.08 per troy ounce** and reached a record **\$49.45 spot on January 18, 1980**, with silver futures touching an intraday COMEX all-time high around **\$50.35**. That is roughly an **eightfold** rise in about a year.

The chart below traces it. Notice the *shape*: a steady climb through 1979 as the metal was drained from the market, an acceleration into late 1979 and early 1980 as the corner tightened and other speculators piled in, the violent peak in mid-January, and then the equally violent collapse. The peak is the moment the corner was, on paper, complete — and it is precisely the moment the house decided it had seen enough.

![Line chart of the silver price rising from about six dollars in early 1979 to a record about forty nine dollars in January 1980 and then crashing](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-2.png)

It is worth pausing on what \$49 silver *meant*. At that price the Hunts' roughly 190-million-ounce position was nominally worth on the order of \$9 billion. Refiners were melting down silverware and coins to capture the windfall; the US Mint and ordinary households sold heirlooms; photographers and electronics makers, who needed silver as an industrial input, were paying multiples of what their products could bear. The price had stopped describing silver's usefulness and started describing the desperation of the trapped shorts. A price that reflects coercion rather than value is the signature of a corner — and it is the signature regulators look for when they decide a market has been manipulated.

There is a second, subtler force at work in the climb, and it is the one that makes a corner *self-financing* for a while. As the price rose, the Hunts' existing positions generated enormous daily mark-to-market profits. Under the margin system, those paper gains could be pledged as collateral to borrow more or simply withdrawn and recycled into further buying. So the ascent funded itself: higher prices produced more borrowing power, which financed more buying and more demands for delivery, which drained the deliverable pool further and pushed prices higher still. This is the same reflexive loop that powers any bubble, but here it was deliberately engineered and physically enforced. The danger of a self-financing ascent is that it runs *in reverse* with equal violence: the moment prices fall, the collateral evaporates, the borrowing power shrinks, and the same leverage that fed the rise now forces the liquidation that feeds the fall. A corner built on rising-price collateral is a structure that can only stand while it is still rising — which is to say, it is a structure designed to collapse.

#### Worked example: the eightfold run in plain numbers

Start with \$6.08/oz at the beginning of 1979. End at \$49.45/oz on January 18, 1980.

- **Multiple:** \$49.45 ÷ \$6.08 ≈ **8.13×**.
- **Percentage gain:** (49.45 − 6.08) ÷ 6.08 ≈ **713%** in roughly twelve months.
- **What \$10,000 of unleveraged silver became:** \$10,000 × 8.13 ≈ **\$81,300**.
- **What \$10,000 of equity at 10x leverage became, on paper, near the peak:** the \$10,000 controlled \$100,000 of silver; an eightfold rise added \$700,000 of gains on the notional, so equity ballooned toward **~\$710,000** — about a 70x return on the cash, before the unwind.

That leverage-amplified, paper 70x is exactly why a corner is so seductive and so dangerous: the same factor that turned \$10,000 into \$710,000 on the way up turned it into a multi-hundred-thousand-dollar *debt* on the way down. The intuition: an eightfold move is enormous for the metal, but with leverage it is a life-changing fortune or a bankruptcy depending entirely on which side of the rule change you are caught on.

## The meta-move: the house changes the rules

Here is the heart of the game-theory lesson, and the part the Hunts got wrong. They had reasoned, correctly, about the shorts: trap them, and they must buy at your price. What they had not reasoned about was the *exchange* as a strategic player with its own payoffs and its own move — the power to change the rules of the game while the game is being played.

The exchanges were not bystanders. Many of their own member firms were short silver and facing ruin; a disorderly silver market threatened the brokers, the clearinghouse, and the integrity of the venue itself. So the people who ran the game did what a player with rule-making power can do: they rewrote the rules to neutralize the corner.

Through January 1980, **COMEX adopted what became known as "Silver Rule 7."** It sharply raised margin requirements, imposed strict position limits on how much silver any single account could hold, and effectively halted new speculative buying on margin. Then, on **January 21, 1980**, the exchange went further and imposed **"liquidation-only" trading**: you could *sell* existing positions, but you could *not* open new long positions. Read that again, because it is the kill shot. The corner depended on the cornerer being able to keep buying and demanding delivery, and on a market where buyers existed. Liquidation-only outlawed the buy side. New longs were forbidden by rule. The forced-buying loop that powered the squeeze was switched off at the source — and with only sellers permitted, the price had nowhere to go but down.

This is a *meta-game move*: a change not to a player's strategy *within* the rules, but to the rules themselves. In ordinary game theory we take the rules as fixed and solve for the best strategies. Real markets are not like that. The exchange and the regulator sit above the players and can alter the payoff structure mid-game. The Hunts solved the game brilliantly — and then the board was flipped. We model the rule-maker's own strategic behavior, the credibility of its threats, and "don't fight the house," in [the central-bank game](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed); the same lesson holds for any venue operator: never play a game against the entity that writes the rules and assume the rules are fixed.

The payoff matrix below makes the meta-move precise, with payoffs from the `nash_2x2` game model. The row player is the Hunts, choosing to PRESS the corner or RELEASE and take profits. The column player is the exchange, choosing to KEEP the rules or CHANGE them (Rule 7). The cell numbers are relative payoffs.

![Two by two payoff matrix of the Hunts pressing or releasing against the exchange keeping or changing the rules, with the rule change as a dominant strategy](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-4.png)

#### Worked example: solving the meta-game and finding the trap cell

Read the matrix as a game and look for each player's best response. Payoffs are written (Hunts, Exchange).

- **Top-left — Hunts PRESS, Exchange KEEPS:** (+10, −9). The squeeze works; the Hunts win big and the exchange's short members are slaughtered.
- **Top-right — Hunts PRESS, Exchange CHANGES:** (−8, +6). The corner is broken; the Hunts are caught long into a collapsing, liquidation-only market while the exchange protects its members.
- **Bottom-left — Hunts RELEASE, Exchange KEEPS:** (+3, +2). An orderly exit; the Hunts book a profit, the market calms.
- **Bottom-right — Hunts RELEASE, Exchange CHANGES:** (−2, +4). The Hunts sell into a falling but orderly market; the exchange has de-risked.

Now solve it. For the **exchange**, compare its payoffs column by column. If the Hunts press, KEEP gives −9 versus CHANGE +6 — change is better. If the Hunts release, KEEP gives +2 versus CHANGE +4 — change is *still* better. So **changing the rules is the exchange's dominant strategy**: it is the exchange's best move no matter what the Hunts do. Knowing that, the Hunts should anticipate CHANGE and pick their best response to it: PRESS gives −8, RELEASE gives −2, so they should RELEASE. The unique **Nash equilibrium is the bottom-right cell, (−2, +4): the Hunts release and the exchange changes the rules.**

But that is not where the Hunts ended up. They pressed on, landing in the **top-right trap cell, (−8, +6)** — the worst outcome for them — because they had not modeled the exchange as a player at all. They optimized against the shorts (the columns of a *different*, smaller game) and treated the rules as fixed scenery. The intuition: when you forget that a player exists, you cannot anticipate his dominant move, and you walk straight into the cell that player's move creates.

There is one more rule-maker in the frame. In January 1980, Federal Reserve Chairman Paul Volcker was waging war on inflation and pushed interest rates dramatically higher — the kind of move that made the borrowed money behind every leveraged silver position far more expensive to carry. Higher rates squeezed the cost of holding the corner at the exact moment the exchange was banning the buying that sustained it. Two different houses, the exchange and the central bank, were tightening the screws at once.

## The collapse: Silver Thursday

With buying outlawed and only liquidation permitted, the price did the only thing it could. From the January peak it slid, week after week, through February and into March. No new longs could enter to absorb the selling; every margin call forced more liquidation; each liquidation pushed the price lower and triggered the next margin call. The squeeze loop now ran in *reverse*, and the Hunts — the people who had built the loop — were standing directly in front of it.

The unwind detonated on **Thursday, March 27, 1980**, a day the markets remember as **"Silver Thursday."** The price fell roughly in half in a single session, from about **\$21.62 to \$10.80 per ounce**. For a 10x-leveraged book, a fifty-percent fall is not a setback; it is an extinction-level event many times over. The Hunts could not meet the margin calls. Reports put the immediate, unmet margin call from their broker Bache at around **\$100 million**, and the feared total exposure across the position at roughly **\$1.7 billion** — money the Hunts, asset-rich but cash-strapped, simply did not have on hand.

This was the moment the *systemic* danger became real. If the Hunts defaulted, Bache and other brokers would absorb losses they might not survive; the failure of a major broker could cascade into the banks that financed it and into the clearing system itself. To prevent a chain reaction, a consortium of major US banks assembled an emergency **line of credit of roughly \$1.1 billion** to the Hunts, secured against their other assets — including their stake in Placid Oil — so the positions could be wound down in an orderly way rather than blown up overnight. A private corner had become a public bailout: the financial system stepped in not to save the Hunts but to save *itself* from the Hunts.

The timeline below lays out the two-month death of the corner, from the January peak through the rule changes to the Thursday crash.

![Timeline of the silver corner from the January 1980 peak through Silver Rule 7 and liquidation only trading to the Silver Thursday crash](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-6.png)

#### Worked example: why a 50% fall is fatal at 10x leverage

Return to the illustrative 100,000,000-ounce long at 10% margin, but now mark it through Silver Thursday.

- **Equity at the \$49.45 peak (10% margin):** about \$494.5 million.
- **Price falls from \$49.45 to \$21.62** (the Mar 26 level): loss = 100,000,000 × (\$49.45 − \$21.62) = \$2.783 billion. Equity (~\$0.49B) is long gone; the position is roughly **\$2.29 billion underwater**.
- **Silver Thursday: \$21.62 → \$10.80** in one day: an additional loss of 100,000,000 × \$10.82 = **\$1.082 billion in a single session.**
- **Total drawdown from the peak to \$10.80:** 100,000,000 × (\$49.45 − \$10.80) = **\$3.865 billion** against ~\$0.49B of starting equity.

A position that needed only about a 5% adverse move to wipe out its margin took a 78% peak-to-Thursday fall. There was never any prospect of meeting the margin calls; the only question was who would absorb the loss, and the answer turned out to be the banking system. The intuition: leverage that high does not have a "bad day," it has a single fatal day, and the corner guaranteed there would be one as soon as the rules changed.

## The aftermath: bankruptcy and a manipulation finding

The corner did not just fail; it destroyed the men who built it and drew the verdict that cornering is illegal. The cost chart below collects the cited figures.

![Bar chart of the costs of the corner including the feared loss the bank rescue the Minpeco damages the margin call and the CFTC fine](/imgs/blogs/case-study-the-hunt-brothers-and-cornering-silver-7.png)

The fallout, in numbers:

- The brothers' personal fortunes cratered. By many accounts the family's net worth fell from around **\$5 billion in 1980 to under \$1 billion by 1988.**
- In **1988**, Nelson Bunker Hunt and William Herbert Hunt declared **personal bankruptcy.**
- Also in **1988**, a jury found the Hunts liable in a civil case for **conspiring to corner the silver market** and for damaging Minpeco, a Peruvian mineral-marketing company that had been short silver; the brothers were ordered to pay more than **\$130 million** in damages.
- In **December 1989**, the Hunts settled with the **CFTC**, each paying a **\$10 million fine** and accepting a **ban from trading in US commodity markets.**

The legal verdict is the part that matters for anyone trying to draw the wrong lesson. A corner is not a clever-but-legal trade that simply went bad on a technicality. It is **manipulation** — the deliberate distortion of a market's price by controlling supply — and it is precisely the conduct commodity law exists to punish. The exchanges' rule change broke the corner in the moment; the courts and the regulator confirmed afterward that the corner should never have been built. The Hunts ended barred from the very markets they had tried to own.

## Common misconceptions

**"The Hunts were just bullish investors who got unlucky."** No. Ordinary bullish investors buy and hold and accept the market price. The Hunts bought enough of the deliverable supply *and* demanded physical delivery on their futures specifically so that the shorts could not settle except by buying from them. That is the textbook structure of a corner, and a jury and the CFTC found it to be illegal manipulation. The "unlucky" framing misses that the conduct was unlawful by design.

**"The exchange changing the rules was cheating; the Hunts were robbed."** This is the most seductive misconception because there is a grain of fairness in it — the rules did change in the middle of the game, and the exchange's members benefited. But the deeper truth is that the rule-maker is *always* a player. An exchange exists to keep an orderly, non-manipulated market; halting a corner is within its mandate and was upheld. The lesson is not "the game was rigged," it is "you can never assume the rules are fixed when you are playing against the entity that writes them." A strategy that only works if the house never moves is not a strategy; it is a bet that the house is asleep.

**"They lost because silver's fundamentals were bad."** The fundamentals were almost beside the point at the peak. \$49 silver did not reflect industrial demand or monetary value; it reflected trapped shorts bidding for metal they could not find. The collapse did not come from a fundamentals reassessment; it came from a rule change that switched off the forced buying. Corners detach price from value on the way up *and* on the way down — the price was a coercion reading in both directions.

**"Taking physical delivery is illegal."** Taking delivery is perfectly normal and legal; commercial users do it constantly. What was illegal was the *combination* — accumulating control of the deliverable supply *in order to* force the shorts to pay a manipulated price. Intent and market power turn ordinary actions into manipulation. A jeweler taking delivery to make rings is fine; a billionaire taking delivery to empty the deliverable pool and trap the shorts is a corner.

**"Leverage is what killed them, so an all-cash corner would have worked."** Leverage made the death fast and the bailout necessary, but it was not the root cause. Even an all-cash holder of 190 million ounces faces the same fatal problem once buying is banned: you are holding an enormous position that, by rule, no new buyer may enter the market to take off your hands, into a price that can only fall. Less leverage would have meant a slower bleed and perhaps no bank rescue, but the corner still breaks the instant the house changes the rules. Leverage set the speed; the rule change set the direction.

**"If they had just sold at \$49, they would have walked away billionaires."** This is the most tempting hindsight, and it misunderstands the trap a cornerer builds for himself. You cannot quietly sell 190 million ounces at \$49. The \$49 price *existed only because* the Hunts were the dominant buyer demanding delivery; the instant they turned to sell at scale, the same monopoly logic runs in reverse — the buyers who were compelled to pay \$49 vanish, and the price collapses under the weight of the cornerer's own selling. A corner is a position you can build but cannot gracefully exit, because the exit destroys the very scarcity that created the gains. The paper \$9 billion was never liquid; it was a number on a screen that would have evaporated on the first serious sell order. This is precisely the exit problem of any crowded position, taken to its extreme.

## How it shows up in real markets

Corners are rare now precisely because the silver episode taught regulators and exchanges to watch for them and arm themselves with position limits and emergency powers. But the *structure* recurs, and recognizing it is the practical payoff of this study.

**The Volkswagen–Porsche squeeze, 2008.** The closest modern analog. Porsche quietly assembled control of about three-quarters of Volkswagen's ordinary shares using disclosure-free cash-settled options, while hedge funds piled into a crowded short. When Porsche revealed its stake, the buyable free float collapsed to a sliver, the trapped shorts bid VW from around €200 to over €1,000 intraday, and short sellers lost an estimated \$20–30 billion. It is the same corner game — control the deliverable float, trap the shorts — executed in equities through an information loophole rather than physical delivery. The full mechanics are in [the Volkswagen-Porsche squeeze](/blog/trading/game-theory/case-study-the-volkswagen-porsche-squeeze-of-2008). The difference from silver: VW's squeeze ended when Porsche voluntarily promised to free up float, not when an exchange banned buying — but both ended because a player above the shorts changed what was available.

**GameStop, 2021.** A crowd-sourced, distributed version: retail traders, coordinating openly, bought shares and call options in a heavily shorted stock, and the resulting squeeze and dealer hedging drove the price up many multiples. The "house move" arrived when brokers like Robinhood, citing clearinghouse collateral demands, restricted buying of the affected names to *closing only* — strikingly reminiscent of liquidation-only. Once buying was throttled, the squeeze deflated. Again: a player with control over market access changed the rules and the game turned. We cover the coordination side of that episode in [the GameStop case study](/blog/trading/game-theory/case-study-gamestop-2021-the-coordination-game-that-broke-wall-street).

**Why corners usually fail today.** Three defenses, all sharpened by the Hunt episode. First, **position limits** cap how much of a commodity any single account can hold, making it hard to amass enough of the deliverable supply legally. Second, **margin and circuit-breaker powers** let exchanges raise the cost of a runaway position or pause trading. Third, **emergency authority** lets exchanges and the CFTC impose liquidation-only or forced-settlement terms exactly as COMEX did. A would-be cornerer today is playing against a house that has already written the silver lesson into its rulebook.

**The general squeeze.** Even without a deliberate corner, the squeeze structure appears whenever a position is *crowded* and the exit is narrow — too many traders on the same side, not enough liquidity to let them all out. That is the everyday cousin of the corner, and we treat it directly in [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game). The corner is the engineered extreme; the crowded trade is the accidental version that the market produces on its own.

**Cartels and supply control.** A corner is a one-shot, hostile version of a broader game: controlling supply to control price. A cartel like OPEC plays a repeated, cooperative version — coordinating output to hold prices up — and faces its own instability, the constant temptation of each member to cheat. The supply-control logic and why it is fragile is in [cartels and the Cournot game](/blog/trading/game-theory/cartels-collusion-and-the-cournot-game-from-opec-to-algorithms). The thread connecting all of these: market power over supply is real power, but it is unstable, contested, and — in the corner's hostile form — illegal.

## The playbook: how to read a corner from the other side

You will almost certainly never build a corner, and you should not — it is illegal manipulation that the system is now built to crush. The practical value of this story is **defensive**: recognizing the game so you are not the trapped short, and understanding the rule-maker so you never bet on the house staying asleep.

**Who is on the other side.** In a corner, the player across from a trapped short is not "the market" — it is a single dominant holder who controls the only exit and sets the toll. In the everyday crowded short, the players on the other side are all the *other* shorts, each of whom will try to cover before you when the squeeze starts. Either way, your danger is the same: the asset you need to buy back is scarcer than you think, and everyone needs it at once.

**The game you are in.** Before you short anything, ask the corner questions. How much of the *buyable, deliverable* float is short? In silver, the shorts were promising metal that one player had locked away; in a stock, the analog is short interest as a fraction of free float. A short interest that exceeds the comfortably available float is a loaded trap, whoever set it. And ask: who controls the supply, and could a single player or a coordinated group be quietly accumulating it where you cannot see — through physical delivery, through options, through off-exchange holdings?

**Your edge.** Your edge as the smaller player is humility about the float and respect for the rule-maker. The trapped silver shorts and the trapped VW and GameStop shorts all made the same error: they sized a position as if the exit were as wide as the entrance. The defensive edge is to assume the exit is narrower than it looks and to never carry a short whose forced-cover cost you cannot survive. The unbounded-loss math of a short means a single corner or squeeze can cost more than your entire account.

**The invalidation.** The signal that you are on the wrong side of a corner is a price that has detached from value and a deliverable float that is shrinking. When the contract price runs far above any reasonable spot value and metal (or shares) cannot be located to deliver, the trade thesis is already dead; the only question is the size of the loss. Liquidation-only or buying restrictions are the final tell that the house has decided the game is over — and they always favor the side that can still sell.

**The sizing and exit.** Because a short's loss is unbounded and a corner makes the exit a single chokepoint, size any short so that even an eight-fold adverse move — yes, the silver number — does not threaten your solvency. That usually means a short is a small, defined-risk position (often expressed through options with capped loss, the subject of [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade)), never an unhedged, leveraged, open-ended bet. And the exit discipline is brutal but simple: cover early, before the squeeze, because the equilibrium of a forced-buying game is that everyone tries to exit at once and only the first few get out at a sane price.

**The meta-lesson.** Above all, internalize the one mistake that turned a winning corner into a bankruptcy: the rule-maker is a player. The Hunts solved the game against the shorts and treated the rules as fixed scenery. They were not scenery. Whenever your strategy depends on the exchange, the broker, the clearinghouse, or the central bank *not acting*, you have not modeled the whole game — you have modeled the part that is convenient and left the most powerful player off the board. The reliable money is made by reasoning one level deeper than the other side, and the deepest level is remembering who can change the rules.

## Further reading & cross-links

- [The short-squeeze game: shorts, longs, brokers, and gamma](/blog/trading/game-theory/the-short-squeeze-game-shorts-longs-brokers-and-gamma) — the general mechanics of a squeeze, of which a corner is the engineered extreme.
- [Case study: the Volkswagen-Porsche squeeze of 2008](/blog/trading/game-theory/case-study-the-volkswagen-porsche-squeeze-of-2008) — the same corner game executed in equities through a disclosure loophole.
- [Case study: GameStop 2021](/blog/trading/game-theory/case-study-gamestop-2021-the-coordination-game-that-broke-wall-street) — a distributed, crowd-sourced squeeze, and a broker "rule change" that ended it.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — the everyday cousin of the corner, where the exit is narrow because everyone is on one side.
- [Cartels, collusion, and the Cournot game: from OPEC to algorithms](/blog/trading/game-theory/cartels-collusion-and-the-cournot-game-from-opec-to-algorithms) — controlling supply to control price, in its repeated, cooperative form.
- [The greater fool and rational bubbles: the musical-chairs game](/blog/trading/game-theory/the-greater-fool-and-rational-bubbles-the-musical-chairs-game) — why a price can detach from value, and what happens when the music stops.
- [The central-bank game: credibility, commitment, and "don't fight the Fed"](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed) — the rule-maker as a strategic player you should never bet against.

---
title: "The SNB 2015 Peg Break: When a Central Bank Blinks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How the Swiss National Bank's 1.20 franc floor went from an iron promise to a 30% one-minute collapse — and why a peg the market doubts is a peg you will abandon."
tags: ["forex", "currencies", "swiss-franc", "central-banks", "currency-peg", "snb", "tail-risk", "leverage", "carry"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A currency floor is only as strong as the market's belief that the central bank will defend it forever; the moment that belief cracks, defending costs explode and the floor breaks in a single, gapping move.
>
> - From September 2011 to January 2015 the Swiss National Bank promised to cap the franc at 1.20 per euro, and defended it by printing francs to buy euros — ballooning its balance sheet toward 90% of GDP.
> - At 09:30 CET on 15 January 2015 the SNB abandoned the floor with no warning; EUR/CHF fell from 1.20 to about 0.85 in roughly three minutes — a near 30% move with almost no tradeable prices in between.
> - Leveraged long-franc-pair positions blew straight through their stop-losses into negative equity; retail brokers Alpari (UK) collapsed and FXCM needed a \$300 million rescue.
> - **The one number to remember: 30%.** A G10 exchange rate is supposed to move 0.5% on a busy day. The franc moved sixty busy days' worth in three minutes — because a defended floor that breaks doesn't drift, it gaps.

At 09:30 on the morning of Thursday, 15 January 2015, the Swiss National Bank published a short press release. It announced that it was "discontinuing the minimum exchange rate of CHF 1.20 per euro." Three sentences in, almost as an afterthought, it lowered its policy rate to minus 0.75%. The release was timed for a quiet mid-morning in Zurich, a moment chosen precisely because it was not a moment anyone was watching.

Within ninety seconds, the price of a euro in francs had fallen off a cliff. EUR/CHF had sat pinned just above 1.2000 for three and a half years — a number so stable that traders had stopped thinking about it, the way you stop noticing the floor you are standing on. Now it was 1.05. Then 1.00. Then, in a market with no buyers and no sellers willing to print a sane quote, it traded as low as roughly 0.85 before clawing back to settle near parity. A currency that the rich world treats as boringly stable had just appreciated by nearly a third in the time it takes to make a cup of coffee.

This post is the anatomy of that morning: why the SNB built the floor, why defending it quietly bankrupted the bank's own logic, why the abandonment had to be a surprise, what a 30% gap does to a leveraged book, and the one durable lesson the whole episode teaches about pegs, floors, and central-bank credibility. The spine of this whole series runs through it: an exchange rate is the relative price of two monies, set by the gap between two countries' interest rates and the flow of money across borders. The franc floor was an attempt to fix that relative price by force — and the story of 15 January 2015 is the story of what happens when force meets a flow it cannot absorb.

![Line chart of EUR/CHF collapsing from the 1.20 floor to about 0.85 on 15 January 2015](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-1.png)

## Foundations: The franc floor and why the SNB set it

Before we can understand the break, we need three pieces of plumbing: what a "safe-haven" currency is, what a one-sided floor actually is, and what it costs to hold one. None of these requires any finance background — just a willingness to follow the money.

### The franc as a safe haven

A **safe-haven currency** is one that investors run *toward* when the world looks dangerous. When a financial crisis hits, large pools of money — pension funds, banks, sovereign wealth funds, ordinary savers in shaky countries — want to park their wealth somewhere they trust will still be worth something next year. Historically a handful of currencies have played this role: the US dollar, the Japanese yen, and the Swiss franc.

Switzerland earns the role through boring, durable strengths. It runs a large and persistent **current-account surplus** — meaning the country as a whole sells more to the world than it buys, so foreigners are constantly net-acquiring francs to pay the Swiss. (In our data, Switzerland's current-account surplus runs around 6% of GDP, among the largest in the developed world.) It has low and stable inflation, deep political stability, strong rule of law, and a central bank with a reputation for sobriety. When money is frightened, it does not ask for a high return; it asks not to lose. The franc has spent a century answering "you won't."

Here is the key consequence, and it is the engine of the whole story. **Demand for a safe haven is strongest exactly when the rest of the world is in trouble.** The franc does not strengthen on good Swiss news; it strengthens on bad European news. And in 2010–2011, European news was very bad indeed.

One quick piece of vocabulary, because the rest of this post lives in it. **EUR/CHF** is a currency pair, quoted as the number of Swiss francs you get for one euro. The euro is the **base** currency (the thing being priced) and the franc is the **quote** currency (the thing it's priced in). So EUR/CHF = 1.20 means "one euro costs 1.20 francs." When EUR/CHF *falls*, the euro is getting cheaper in franc terms — equivalently, the franc is getting *stronger*. This is worth fixing firmly in mind, because the SNB's floor was a floor under EUR/CHF (it would not let the number go *below* 1.20), which is the same thing as a *ceiling* on franc strength. A falling EUR/CHF and a strengthening franc are the same event described from two sides; if you ever want the full grammar of reading a quote, see [base, quote, pips, and how to read an FX quote](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote). Throughout this post, "the floor breaking" and "the franc spiking" are the same sentence.

Why does the franc specifically — and not, say, the Swedish krona or the Australian dollar — get the safe-haven flows? Because a safe haven needs two things at once that are surprisingly hard to combine: it must be a currency people *want* to hold in a panic (deep, liquid, backed by a stable state), and it must be *available* in size when they reach for it. Switzerland's persistent surplus means the world is structurally short of francs relative to how many it wants in a crisis — the supply is tight. So when fear spikes and everyone reaches for francs at once, the price (in euros or dollars) jumps sharply, because there simply aren't enough francs to go around at the old level. The very property that makes the franc a haven — scarcity backed by stability — is what makes a flight into it so violent on the way up. The SNB spent 2010–2011 watching this scarcity premium tear its exporters apart.

### The euro-zone crisis and the franc's painful rise

In 2010 and 2011, the euro area was in an existential panic. Greece was sliding toward default. Markets were openly asking whether Italy and Spain could fund themselves, and whether the euro would survive at all. Money fled the periphery, and a great deal of it ran straight across the border into francs.

That capital flight pushed the franc up — and up, and up. EUR/CHF, which had spent the late 2000s in the 1.40s and 1.50s (meaning one euro bought about 1.50 francs), collapsed toward 1.10, and in August 2011 briefly touched almost 1.00, a hair from one-for-one parity. In plain terms, the euro had lost a third of its value against the franc in a matter of months, driven not by anything Switzerland did but by fear of the euro itself.

#### Worked example: what a soaring franc does to a Swiss exporter

Consider a Swiss watchmaker that sells a watch in the euro zone for €10,000. When EUR/CHF is 1.50, that sale brings home \$15,000-equivalent — 15,000 francs. The firm's costs (Swiss wages, Swiss rent) are in francs, say 12,000 francs, leaving a 3,000-franc profit.

Now the franc soars and EUR/CHF falls to 1.05. The same €10,000 watch now converts to just 10,500 francs. The firm's costs are still 12,000 francs. The sale that used to make 3,000 francs now *loses* 1,500 francs. The watch hasn't changed, the customer hasn't changed, the price in euros hasn't changed — but a currency move has flipped a profitable business into a loss-making one.

The intuition: for an export-driven economy, an over-strong currency is a slow-motion recession, because every foreign sale converts into fewer francs while costs stay fixed.

This was the SNB's nightmare. A franc rocketing toward parity threatened to gut Swiss exporters and tourism, and to drag the country into deflation (falling prices), because imported goods kept getting cheaper in franc terms. The SNB had already cut rates to near zero and intervened directly in the market, buying euros to slow the rise. It wasn't enough. So on 6 September 2011, it did something dramatic.

### The 1.20 floor: a one-sided promise

The SNB announced a **minimum exchange rate**: it would "no longer tolerate" a EUR/CHF rate below 1.20, and would defend that floor "with the utmost determination," buying "unlimited quantities" of foreign currency to do so.

It is worth being precise about what kind of peg this was, because the asymmetry is the whole point.

A classic **peg** is two-sided: the central bank promises to keep the exchange rate inside a band, defending both the ceiling (if the currency gets too weak) and the floor (if it gets too strong). A **floor**, as the SNB ran it, is one-sided. The SNB only promised to stop the franc from getting *too strong* (EUR/CHF too *low*). It said nothing about a weak franc; EUR/CHF was free to rise above 1.20 all it liked. The bank only stood in the market on one side.

This asymmetry has a beautiful and a terrible property, and they are the same property.

The beautiful part: defending a floor against a *strengthening* currency is something a central bank can, in principle, do forever. To stop the franc from getting too strong, the SNB sells francs and buys euros. And it can create francs out of nothing — it is the issuer of francs. Unlike defending a *weak* currency (which requires *spending* finite foreign reserves to buy your own currency, and you run out — that's the Bank of England's 1992 problem and the trap in [currency crises and the anatomy of a speculative attack](/blog/trading/forex/currency-crises-and-the-anatomy-of-a-speculative-attack)), defending a *strong* currency requires *printing* your own money, which is infinite.

The terrible part is the same sentence read backwards: defending the floor means printing unlimited francs and stuffing the proceeds into euro assets. The bank cannot run out of ammunition — but it accumulates an ever-larger pile of foreign currency it did not want, exposed to losses it cannot control.

There is a deeper structural reason the SNB had to pick *some* tool here, and it is worth naming because it explains why the choice was so constrained. Economists call it the **impossible trinity** (or trilemma): a country can have at most two of three things — a fixed exchange rate, free movement of capital across its borders, and an independent monetary policy. Switzerland was never going to give up free capital movement (it is a global financial center; capital controls are unthinkable). It had already pushed its own interest rates to the floor, so it had little independent monetary room left to lean on. That left the exchange rate as the lever to grab — but "grabbing the exchange rate" while keeping capital free and rates near zero is exactly what forces you to absorb unlimited flows through the printing press. The franc floor wasn't a free choice among many; it was what's left when a small, open, low-rate economy decides it can no longer tolerate its own currency. The macro-policy machinery behind that trade-off lives in [how monetary policy moves currencies through rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials).

![Pipeline diagram showing the SNB printing francs to buy euros and ballooning its balance sheet to defend the floor](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-2.png)

## Defending the floor: the mechanics of printing your way to a problem

Let us walk the mechanism step by step, because "the central bank defends a floor" is doing a lot of quiet work in that sentence, and the cost is hidden inside it.

Every time the franc tried to strengthen past 1.20 — every time a fresh wave of euro-zone fear pushed EUR/CHF down toward the floor — the SNB had to step in as the buyer of last resort for euros. The sequence is mechanical:

1. **Pressure arrives.** Investors sell euros and buy francs. EUR/CHF ticks down toward 1.2000.
2. **The SNB creates francs.** It credits new francs into the banking system — these show up as **sight deposits**, the franc balances commercial banks hold at the central bank. This is money creation, full stop.
3. **It spends those francs on euros.** It uses the newly created francs to buy the euros the market is desperate to sell, taking the other side of the trade at 1.20.
4. **The euros pile up as reserves.** The SNB now owns those euros, which it parks in euro-zone government bonds, equities, and other assets. Its **foreign-exchange reserves** balloon.
5. **The floor holds — at 1.20 exactly** — because the SNB is willing to absorb any quantity. But the bank's balance sheet has grown.

Repeat this thousands of times over three and a half years, and you get a central bank whose balance sheet has swollen to a size unlike almost any other in the developed world.

### The ballooning balance sheet

The cleanest single picture of the cost is the SNB's sight deposits — the franc liabilities it created to fund all that euro-buying. They tell the story of the floor's price.

![Bar chart of SNB sight deposits ballooning from 2011 to 2024, peaking while defending the floor](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-3.png)

In 2011, before the floor, sight deposits were around 100 billion francs. By the time the SNB introduced the floor and began defending it in earnest, they jumped to over 300 billion. By the end of 2014 — the last full year of the floor — the SNB's total balance sheet had grown to roughly 85% of Swiss GDP, an extraordinary figure. For comparison, the US Federal Reserve's balance sheet at its post-crisis peak was around 25% of US GDP. The SNB had built a foreign-currency reserve pile larger than three-quarters of its entire economy's annual output, almost all of it euros and dollars it had been forced to buy.

Now the second-order problem becomes visible. The SNB's *liabilities* (the francs it printed) are in francs. Its *assets* (the reserves it bought) are in euros and dollars. This is a giant **currency mismatch**, and it sits on the bank's own books.

#### Worked example: the SNB's paper loss on its own reserves

Suppose, to keep the arithmetic clean, the SNB has accumulated €400 billion of reserves, all bought at the floor rate of 1.20. On its books, those euros are worth 400 × 1.20 = 480 billion francs.

Now suppose the franc strengthens so that EUR/CHF falls to 1.00 — exactly the move the floor was holding back. Those same €400 billion are now worth only 400 × 1.00 = 400 billion francs. The reserves haven't changed; the bank still owns €400 billion. But measured in its home currency — the currency it reports in and the one its solvency is judged in — it has just lost 80 billion francs.

The intuition that makes the floor a trap: the SNB was buying more and more of the very asset (the euro) whose fall it was fighting, so the harder it defended the floor, the bigger the loss it was setting itself up to take the day the floor broke. Defending a floor is selling insurance against your own balance sheet.

This is the quiet bind. The SNB could print francs forever, so it could *never be forced* to abandon the floor for lack of ammunition. But every franc it printed bought more euros, and every euro it bought deepened the loss it would book if the franc ever rose. The constraint was not "can we?" but "how much loss are we willing to pre-commit to, and how much of our own economy's GDP do we want to hold as a leveraged bet on the euro?"

It is worth dwelling on why this is different from a normal central bank "money printing" worry. When most central banks expand their balance sheets — the Fed buying Treasuries, say — their assets and liabilities are in the *same* currency. The Fed's bonds are dollar bonds; its reserves are dollar reserves; a move in the dollar doesn't blow a hole in the middle of its balance sheet. The SNB's position was categorically riskier because of the mismatch: it had *printed* a domestic-currency liability to *buy* a foreign-currency asset, so it was running an enormous, leveraged, one-way FX position on its own account — long euros and dollars, short francs, in a size approaching its country's entire annual output. No hedge fund on earth would be allowed to run that book. The SNB ran it because it had no choice if it wanted the floor; the floor *was* that book. And the irony compounds: the position made money in franc terms whenever the floor was holding and the euro was stable, which made the policy look costless and even profitable in the good quarters — right up until the day the franc was allowed to rise, when the whole accumulated loss crystallized at once. A position that prints small steady gains and hides one enormous loss is the exact shape we will meet again when we look at who was on the *other* side of the floor.

The SNB also faced a subtler, reflexive problem. Its very willingness to buy unlimited euros at 1.20 was an open invitation. Any investor who wanted to dump euros — to de-risk away from the euro-zone crisis — could do so at a guaranteed price of 1.20, with the SNB as a captive buyer. So the floor didn't just absorb safe-haven inflows into the franc; it became a free exit door out of the euro for the whole world, which *increased* the flow the SNB had to absorb. The more credible and generous the floor, the more business it attracted, the bigger the balance sheet grew. A defence that gets more expensive the better it works is a defence with a built-in expiry date.

## The unholdable peg: why a floor the market doubts becomes a floor you abandon

A floor defended with infinite ammunition sounds permanent. It wasn't, and the reasons it wasn't are general — they apply to every peg, not just this one. The franc floor became unholdable for three reinforcing reasons, and the way they compound is the deepest idea in this post.

![Graph showing how persistent pressure, ECB QE, and a ballooning balance sheet make the floor unholdable](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-4.png)

**First, the political and financial limit on the balance sheet.** Buying unlimited euros means the SNB was, in effect, transferring Swiss national wealth into a giant directional bet on the euro and on euro-zone assets. A central bank's losses are ultimately the public's losses — in Switzerland's case, the SNB even distributes profits to the cantons (the regional governments), so the cantons had a direct stake. As the balance sheet pushed past 80% of GDP, the political question got louder: how big are we willing to let this get? There is no hard legal ceiling, but there is a real political one, and everyone could see it approaching.

**Second, and decisively, the European Central Bank was about to make the problem far worse.** By January 2015, it was widely expected — correctly — that the ECB would soon launch large-scale **quantitative easing**: creating new euros to buy euro-zone bonds. (Macro-trading owns the mechanics of QE; see [the central-bank toolkit of rates, QE, QT and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).) QE would flood the world with new euros, which would push the euro down against everything — including the franc. That meant the downward pressure on EUR/CHF was about to intensify enormously. The SNB looked at a future where it would have to print francs *even faster* to hold 1.20 against a tidal wave of new euros, and decided the wave was not worth fighting. The ECB announced its QE programme on 22 January — exactly one week after the SNB jumped.

**Third, the credibility doom loop.** This is the subtle one and the one that generalizes. A floor's defence is cheap precisely when the market *believes* it. If everyone is convinced the SNB will hold 1.20 forever, then nobody bothers to push hard against it, and the SNB barely has to print at all — the belief is self-fulfilling. But the instant the market starts to *doubt* the floor — once traders sense the balance sheet is getting politically uncomfortable and the ECB is about to make things worse — they pile in on the side of the break. Speculators short EUR/CHF *toward* the floor, betting the SNB will fold; that selling pressure forces the SNB to print even more to hold the line; the bigger balance sheet makes abandonment look more likely; which invites more speculative selling. Doubt is self-fulfilling in exactly the same way belief is. A floor the market trusts costs nothing to hold. A floor the market doubts costs more every day until it breaks.

This is the formal logic of a speculative attack — a coordination game where each speculator's willingness to bet against the peg depends on whether they think others will too — and game theory owns the model; see [the central-bank game of credibility, commitment and don't-fight-the-fed](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed). For our purposes the narrative is enough: once a defended price loses the market's belief, the defence gets more expensive precisely as it gets more pointless.

### The before-and-after of belief

The whole logic collapses into a single contrast: a credible floor versus a doubted one.

![Before-after diagram contrasting a credible floor that holds cheaply with a doubted floor that breaks](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-8.png)

When a floor is credible, the market believes it holds, so little defence is needed, so reserves stay contained, so the floor is self-sustaining — a virtuous circle. When a floor is doubted, the market bets it breaks, so endless printing is needed, so reserves balloon past tolerable limits, so the bank abandons it — a vicious circle. The terrifying thing is that the *same floor at the same level* can sit in either circle depending only on what the market believes. The SNB's floor lived in the virtuous circle for three years and slipped into the vicious one in its final months. Nothing about the number 1.20 changed. The belief did.

## The abandonment: why it had to be a surprise, and why "until the day before, forever"

Here is a fact that infuriated traders and politicians for years afterward, and that contains an important lesson about central banks. Just **three days before** the de-peg, on 12 January 2015, an SNB vice-chairman publicly called the minimum exchange rate "absolutely central" and reaffirmed the bank's commitment to defend it. The SNB had repeatedly described the floor as a "cornerstone" of its policy. Then, with no warning, it was gone.

Was this dishonesty? Not in the way it looks. It is a structural feature of how a credible exit from a peg *has* to work, and it follows directly from the doom loop above.

A central bank can never pre-announce that it is *thinking about* abandoning a floor. The moment it hints at doubt, the market front-runs the move: every speculator sells EUR/CHF immediately to get ahead of the break, the SNB has to print enormous sums to hold the line during the announcement window, and it accumulates even *more* loss-making reserves on the way out the door — buying euros at 1.20 that it knows are about to be worth far less. The only way to abandon a floor without first being run over by your own defence is to do it by surprise, having sworn loyalty to it right up until the morning you don't. "We will defend this forever — until the day we don't, and you will not get a warning" is not a contradiction a central bank can avoid; it is the only exit that doesn't bankrupt the bank on the way out.

So the public reassurance three days earlier was not (necessarily) a lie at the time it was said; it was the required camouflage for any exit. The lesson for anyone trading a pegged or floored currency is stark and permanent: **a central bank's promise to defend a level is informative about today and worthless about tomorrow.** The stronger the reassurance, the closer you may be to the break, because credible reassurance is exactly what a bank that's about to fold needs to issue. Don't fight the central bank — but don't trust its forever-promises either.

There is a second reason surprise is structurally required, beyond front-running, and it concerns the SNB's own losses on the way out. Recall the currency mismatch: the SNB held a vast pile of euros bought at 1.20. If it had pre-announced "we will exit the floor in two weeks," it would have spent those two weeks as a forced buyer of even more euros at 1.20 — euros it *knew* would be worth perhaps 1.00 the moment the floor lifted. Every euro bought during a pre-announced wind-down is a euro bought at a 20% known loss. The only way to stop pouring money into a loss you can already see is to stop *instantly*, without warning. Surprise isn't cruelty; it is loss control. The SNB chose to take its lump in one violent morning rather than bleed for two pre-announced weeks, and from the bank's narrow perspective that was the cheaper path.

This is also why you should be deeply skeptical of the genre of central-bank communication that says "we are *fully committed* to the current regime." That sentence carries almost no information about the future, because a bank that means it and a bank that's about to abandon the regime are *required* to say exactly the same thing. The committed bank says it because it's true; the exiting bank says it because the alternative — hinting at the exit — bankrupts its defence. You cannot distinguish the two from the words. You can only watch the *cost* of the defence: the balance sheet, the reserves, the external pressure. When the cost curve bends toward the unbearable, the forever-promise is at its loudest and its least trustworthy at the same time.

### The mechanics of the gap

When the floor went, EUR/CHF did not *fall* in the way prices usually fall. It **gapped** — it jumped from one price to a far lower one with no tradeable prices in between, because the one giant buyer of euros at 1.20 (the SNB) vanished in an instant and there was nobody behind it.

![Slope chart showing EUR/CHF jumping from the 1.20 floor to a 0.85 trough with no fills in between](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-5.png)

Think about who was standing at 1.20 the moment before the announcement. The SNB was the bid — it was buying every euro offered at 1.20, by its own promise. Behind it, real-money sellers and speculators were lined up to sell euros, leaning on the floor. There was almost no genuine private buyer of euros below 1.20, because for three and a half years there had been no need for one: the SNB was the floor.

When the SNB stepped away, all those euro sellers were suddenly looking for a buyer, and there was none until the price fell far enough to tempt fresh value-buyers and to trigger panicked short-covering. The first real prints came in around 1.05, then 1.00, then the market punched a hole down to roughly 0.85 — a 30% appreciation of the franc — before snapping back to settle near 0.98–1.00 over the next hours. Between 1.20 and the trough there were essentially *no fills*: orders that said "sell at 1.18" or "sell at 1.10" did not execute at those levels, because no buyer existed there. They executed wherever the next real buyer was — and that was a world away.

This is the property that turns a peg break from a loss into a catastrophe. In a normal market, a move passes *through* every price, so a stop-loss order set at a given level gets you out near that level. In a gapping market, a stop-loss is a request to be sold at the next available price, and the next available price can be 30% away. The protection you bought turns out not to exist at the moment you need it. We will see exactly what that does to a leveraged account next.

To feel the speed, it helps to lay out the morning minute by minute. At 09:30 CET the press release hit the wires. Within seconds, automated systems and dealers parsed the one sentence that mattered — the floor was gone — and pulled their bids. By roughly 09:31, EUR/CHF had already printed around 1.05; by 09:33 it had punched through parity and into the 0.85 air pocket. The whole vertical leg — 1.20 to the lows — took something on the order of *minutes*, not hours. Liquidity in EUR/CHF, normally one of the most liquid pairs on earth, simply evaporated: spreads that are usually a fraction of a pip blew out to hundreds of pips, and for a stretch there were effectively no two-way prices at all. Many trading platforms froze, rejected orders, or showed prices that no one could actually deal on. By late morning the market had stabilized somewhere around 0.98–1.02, and EUR/CHF closed the day near 0.985 — roughly an 18% one-day fall from the floor, having been down nearly 30% intraday.

The detail that matters for everyone who was positioned: the damage was done in the part of the move where there were *no prices*. A trader who could have gotten out at 1.18, or 1.10, or even 1.00 in an orderly market got out — if at all — somewhere in the 0.85–1.00 range, because those orderly prices never existed to trade on. The settle near 0.985 is almost a red herring; the accounts were destroyed in the gap, not at the close.

## Who blew up: from leveraged longs to broker insolvency

The franc shock is famous not just for the size of the move but for the bodies it left. To see why, we need to understand leverage and stop-losses in retail FX, and then trace the loss as it cascades upward.

### Leverage and the false comfort of a stop

Retail FX is traded with enormous **leverage** — borrowed money that magnifies both gains and losses. Before the shock, brokers commonly offered 50:1, 100:1, even higher. Leverage of 100:1 means that for every \$1 of your own money (your **margin**), you control \$100 of currency exposure. A 1% move in your favour doubles your money; a 1% move against you wipes it out.

The franc shock didn't move 1%. It moved 30%. Against a leveraged long-EUR/CHF position — and there were many, because being long EUR/CHF near the floor had looked like free money for years (you were buying just above a level the SNB swore to defend, collecting a small carry, with seemingly no downside) — a 30% adverse move was not a wipeout. It was a wipeout many times over.

![Tree diagram showing the 30% gap cascading from leveraged clients to brokers Alpari and FXCM](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-7.png)

#### Worked example: a leveraged long-EUR/CHF position destroyed

A retail trader believes the franc floor is rock-solid. He goes long EUR/CHF — buying euros, selling francs — at 1.2010, just above the floor, reasoning that the SNB will never let it fall below 1.20, so his downside is tiny and he collects a small interest carry. He puts up \$10,000 of margin and uses 50:1 leverage, controlling a \$500,000 position. He even sets a stop-loss at 1.1950, just below the floor, "to be safe" — capping his loss, he thinks, at well under his \$10,000.

The floor breaks. His stop at 1.1950 cannot fill — there are no buyers of euros there. The position is finally closed when the broker can find a price, around 1.00 (and some were filled far worse, near the 0.85 lows). The euro he bought at 1.2010 is now worth about 1.00 francs: a fall of roughly 0.20 francs, about 17% of the price.

On a \$500,000 position, a 17% adverse move is a loss of about \$85,000. His margin was \$10,000. He has not just lost his \$10,000 — he is *down \$75,000 below zero*. His account has **negative equity**: he owes the broker \$75,000 he never agreed to risk and likely cannot pay.

The intuition, and the lesson that priced into FX leverage limits worldwide afterward: a stop-loss caps your loss only if the market trades through your stop price. When a peg breaks and the price gaps, a stop is a polite request that the market ignores — and high leverage turns a gap into a debt you owe.

Multiply that one trader by thousands. That is the next link in the chain.

### From client to broker

When a client's account goes negative, the loss does not vanish — it lands on the **broker**. The broker had passed the client's trade through to the wider market (the bank or liquidity provider behind it), and *that* counterparty must be paid in full, at the real fill price, regardless of what the client can cover. If the client owes \$75,000 and can pay only the \$10,000 in the account, the broker eats the \$65,000 difference. Across a whole book of blown-up clients, the deficits added up to sums no retail broker held in capital.

#### Worked example: a broker's book of negative-equity clients

Suppose a mid-sized retail broker has 1,000 clients who were net long EUR/CHF into the shock, and after the gap their accounts are, on average, \$30,000 in negative equity each. That is 1,000 × \$30,000 = \$30 million of client debt the broker is contractually on the hook for to its own liquidity providers — but which it can only hope to collect from clients, many of whom simply walk away from a debt larger than their net worth.

If the broker's own regulatory capital was, say, \$20 million, then \$30 million of uncollectable client losses renders it insolvent on the spot: its liabilities to its liquidity providers exceed its capital plus whatever it can claw back from clients. The broker is bankrupt not because *it* took a bad position, but because it guaranteed its clients' trades to the market and the clients couldn't pay.

The intuition: a retail broker is a thin layer of capital standing between leveraged clients and the real market; a gap big enough to push clients past zero pushes the broker past zero too.

This is exactly what happened. **Alpari UK**, one of the larger retail FX brokers, announced insolvency within a day, stating that the franc move had left many clients with negative equity that exceeded the company's own resources. **FXCM**, a much larger US-listed broker, disclosed that its clients owed it roughly \$225 million in negative balances they couldn't pay — and that this left FXCM itself in breach of regulatory capital requirements. FXCM survived only by taking an emergency \$300 million rescue loan from the investment firm Leucadia, on punishing terms, and was effectively never the same company again. Other brokers and a handful of small funds around the world failed or took heavy losses. Even some banks reported nine-figure losses on the day.

The failure mode here is worth naming precisely, because it is a recurring feature of leveraged systems, not a quirk of retail FX. A broker that "passes through" client trades to the real market is implicitly *guaranteeing* its clients' performance to its liquidity providers. In normal markets that guarantee is nearly free, because clients who lose more than their margin are a rounding error — the market rarely gaps past everyone's stops at once. But the guarantee is a hidden short option: the broker collects steady fees in calm markets and is on the hook for an unbounded loss in a gap. The franc shock exercised that option for the whole industry on the same morning. Every broker discovered simultaneously that its risk model — "clients can't lose more than they deposit, because stops will fire" — was built on the assumption of a market that trades through prices, and that assumption had just failed for the entire book at once. The lesson generalizes far beyond FX: any business that profits in calm and bears a tail in the gap is short a hidden option, and the franc is the cleanest illustration finance has of that option being called.

There is also a grim asymmetry in who paid. Clients who were long EUR/CHF and went negative often simply *didn't* pay — you cannot squeeze a five-figure debt out of a retail customer who has walked away — so the realized loss landed on whichever institution sat above them in the chain and *couldn't* walk away: the broker, then the bank, then the prime broker. The deeper your pockets, the more of the gap you ended up wearing, regardless of whether you had any position at all. This is why the people most hurt by the franc were frequently not the speculators who took the bet, but the intermediaries who had merely guaranteed the speculators' access to the market.

The cascade — leveraged client → broker → the banks and prime brokers behind them — is the general shape of how a single violent price move propagates through a leveraged system. It is the same shape as a carry-trade unwind (see [carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller)), because being long EUR/CHF near the floor *was* a carry trade: collect a small, steady premium, and stand in front of a rare, enormous, one-directional risk.

### The CHF shock in context

How big was 30% in one minute, really? It helps to line the franc up against the other headline currency shocks of the modern era.

![Horizontal bar chart comparing the CHF de-peg against other modern FX shocks by magnitude](/imgs/blogs/the-snb-2015-peg-break-when-a-central-bank-blinks-6.png)

The pound's 8% slide on the 2022 mini-budget was a genuine crisis that nearly broke the UK pension system. The yen's roughly 20% collapse over 2022 reshaped global carry trades. The August-2024 yen-carry unwind, about 12%, briefly spiked the VIX above 65. Every one of those is a major event. And every one is dwarfed by the franc's ~30% — and crucially, those other moves played out over *days, weeks, or months*. The franc did most of its move in a single window of *minutes*. A move that big, that fast, in a major developed-market currency, is essentially without peer in the floating-rate era. It is the reference point traders mean when they say "a peg break."

## Common misconceptions

**"The SNB ran out of money to defend the floor."** No — this is the single most common error, and it gets the whole mechanism backwards. The SNB was defending against a *strengthening* franc, which it does by *printing* francs (which it can do infinitely) and buying euros. It could never run out of ammunition. It quit because the *cost* — a balance sheet near 90% of GDP and a fast-growing currency mismatch, about to get far worse under ECB QE — became politically and financially intolerable. Contrast this with the Bank of England in 1992, which was defending a *weak* pound and had to *spend* finite reserves; *that* central bank really could and did run out. See [Soros and Black Wednesday: breaking the Bank of England 1992](/blog/trading/forex/soros-and-black-wednesday-breaking-the-bank-of-england-1992). Defending a strong currency is unlimited but loss-accumulating; defending a weak one is finite and reserve-depleting. Different traps, opposite mechanics.

**"A stop-loss would have protected you."** A stop-loss caps your loss only if the market trades *through* your stop price. In a gapping market — no fills between 1.20 and ~0.85 — a stop at 1.1950 executed nowhere near 1.1950; it executed at the next real price, which could be 1.00 or worse. Tens of thousands of "protected" accounts were filled far below their stops and pushed into negative equity. A stop is a request, not a guarantee, and a peg break is exactly the moment the request gets ignored.

**"Being long EUR/CHF near the floor was low-risk."** It *looked* low-risk for three and a half years — that's precisely why so many were positioned that way, and why the blow-up was so widespread. It was actually a textbook negative-skew trade: a long stretch of small, steady gains (the carry) hiding a rare, enormous loss. The risk wasn't absent; it was *deferred and concentrated*. The quiet years were not the absence of risk — they were the accumulation of it.

**"The SNB lied; it had been planning this for weeks."** Possibly it had, and that is *necessary*, not scandalous. A credible exit from a floor cannot be pre-announced without triggering the very front-running stampede that bankrupts the defence on the way out. The reassurance three days before was the required camouflage. The lesson isn't "central banks lie" — it's "a central bank's forever-promise about a peg is, by structural necessity, worthless about tomorrow."

**"This was a freak one-off that can't recur."** The specific event was rare, but the *structure* is general and recurs constantly: any defended price (a peg, a floor, a ceiling, a managed band) can be held only as long as the market believes in it and the cost stays tolerable, and any such price can break in a gap when belief and cost cross. The Czech koruna floor (2013–2017) ended more gently only because the market never seriously doubted it and the koruna wasn't a global safe haven. The structure is always the same; only the violence of the exit varies.

## How it shows up in real markets

The franc shock is not just history; it permanently changed how the market prices and manages currency risk. Several concrete, durable effects:

**Leverage limits were slashed.** Regulators looked at the wall of retail negative-equity debt and acted. The US had already capped retail FX leverage at 50:1; afterward, regulators in Europe and elsewhere moved hard. The EU's ESMA later capped retail FX leverage at 30:1 on majors and lower on more volatile pairs, and many brokers introduced **negative-balance protection** — a promise that a retail client can lose at most their deposit, with the broker absorbing any gap beyond it. The franc shock is the event those rules were written against. Every time a retail trader today bumps into a 30:1 cap, that is the ghost of 15 January 2015.

**"Peg risk" became a permanent line item in FX options.** After the shock, the options market never again priced a defended currency as if the floor were free. The premium for out-of-the-money options that pay off if a peg breaks — the cost of insuring against a gap — carries a visible "this can break violently" charge for any managed currency. This is the **skew** of the vol surface doing its job: the market charges more for the tail it has actually seen. Options and vol own that machinery; see [trading skew, risk reversals, collars and the shape of fear](/blog/trading/options-volatility/trading-skew-risk-reversals-collars-and-the-shape-of-fear). The lesson the surface encodes is the lesson of this whole post: a quiet, pinned currency is not a safe currency; it is a coiled one.

**Every peg is now read as a clock, not a wall.** Traders watching the Hong Kong dollar band, the Danish krone's peg to the euro, the Gulf states' dollar pegs, or any managed-float regime now ask the franc questions: How big is the central bank's balance sheet or reserve pile getting? Is an external force (a big policy shift in the anchor currency) about to intensify the pressure? Is the market starting to doubt? A peg is not evaluated as "will it hold?" but as "what would make holding it cost more than breaking it, and how close are we to that?" The franc taught the market to price the *exit*, not the *level*.

Notice how the *direction* of the defence flips the whole risk profile, and the franc made traders fluent in that distinction. A bank defending a currency that is too *weak* — the Bank of England in 1992, an emerging market under capital flight — spends finite reserves to buy its own currency, and the warning sign is the reserve pile shrinking toward zero on a known clock. A bank defending a currency that is too *strong* — the SNB — prints infinite domestic currency and accumulates loss-making reserves, and the warning sign is the balance sheet *growing* without limit and the political tolerance for it shrinking. The Czech National Bank ran a strong-side floor on the koruna from 2013 to 2017 and exited it with barely a ripple, precisely because the koruna was not a global safe haven, the inflows were modest, and the market never doubted the exit would be orderly. Same structure, opposite outcome — which proves the structure isn't destiny. What determines whether a floor exits with a ripple or a 30% gap is the *size of the one-sided flow* it was holding back and *how crowded the trade against it* had become. The franc's flow was enormous and the trade against it was the most crowded in FX. That, not the level, is why it broke the way it did.

**The carry-trade lesson got a fresh, brutal data point.** Being long EUR/CHF at the floor was carry — getting paid a small premium to hold a position — with a catastrophic tail. The August-2024 yen-carry unwind rhymed with it: a beloved, crowded, "free money" trade that paid steadily and then detonated. The connective tissue is positioning: when everyone is on the same side of a trade because it has "always" worked, the unwind has no other side to sell into, so it gaps. See [carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) for the general anatomy.

**The 2015–2022 stress map starts here.** The franc de-peg opens a run of currency stress events — the franc in 2015, the yen's grind weaker through 2022, the gilt/pound crisis of September 2022, the EM stress cycle — that share a common thread: a relative price held in place by policy or by a one-sided flow, until the policy or the flow reverses. See [2015 to 2022: the franc, the yen and the EM stress cycle](/blog/trading/forex/2015-to-2022-the-franc-the-yen-and-the-em-stress-cycle) for how the franc fits the broader pattern.

**Risk managers now stress-test for "a peg that gaps."** Before 2015, a typical FX risk model assumed major-currency moves were roughly continuous — a 5-standard-deviation day was the bad case, and you could trade out of it. After the franc, serious desks added a discrete "gap scenario" to their stress tests: what does my book look like if a managed currency jumps 20–30% overnight with no fills, and my hedges and stops don't execute at their levels? That question reshapes how much leverage a desk will run against any currency that is even loosely managed — the Hong Kong dollar, the Gulf pegs, the Danish krone, the Chinese renminbi's daily fix. The number you can no longer assume away is the gap, and the SNB is the reason it sits in the model.

A useful way to carry all of this is to treat the *cost of the defence* as the real signal and the *level* as noise. The franc sat at 1.20 the whole time; that number told you nothing. What told you everything was the SNB's sight-deposit chart climbing toward 90% of GDP and the calendar ticking toward ECB QE. The level is what the central bank shows you; the cost is what it's trying to hide. Read the cost.

## The takeaway: a peg the market doubts is a peg you will abandon

Strip the franc shock down to its load-bearing idea and you get a sentence that applies to every fixed or managed exchange rate, everywhere, forever: **a peg, floor, or ceiling is a promise, and a promise is only worth what the market believes it's worth.**

While the market believes, the promise is nearly free to keep — belief is self-fulfilling, so almost no defence is needed. The moment the market doubts, the same promise becomes ruinously expensive — doubt is self-fulfilling too, so the defence escalates as it loses its point. The level never changes; only the belief does. And because a credible exit from a doubted peg cannot be pre-announced (announcing it triggers the stampede that bankrupts the defence), the break, when it comes, comes by surprise and comes in a gap — not a drift you can react to, but a hole in the price with no fills in it.

For how you actually read a currency, three durable rules fall out of this:

First, **treat a pinned currency as a coiled one.** Stability that comes from a central bank standing in the market is not the same as stability that comes from balanced flows. The first is a held breath; the second is calm. The franc near 1.20 was a held breath, and the longer it held, the more violent the exhale.

Second, **size for the gap, not the drift.** In any market where a defended price could break, your real risk is not the slow move you can stop out of — it is the gap you can't. That means: less leverage than the calm tape invites, position sizes you could survive a 20–30% overnight gap on, and a healthy distrust of stop-losses as protection against tail events. A stop protects you against a normal market; a peg break is the abnormal one. The arithmetic is unforgiving: at 50:1 leverage, a 2% adverse gap erases your margin, so a currency capable of a 30% gap is one where 50:1 is not "aggressive" — it is a guaranteed total loss the day the tail arrives. The right question is never "how much can I make if it holds?" but "can I survive the morning it doesn't?" If the answer is no, the position is too big regardless of how attractive the carry looks.

A concrete way to apply this: when you find yourself in a trade that has "always worked" and feels like free money — a pinned currency, a carry that pays every month, a level that "the central bank won't allow" — treat that *feeling* as the warning, not the reassurance. Free money that everyone can see is free money everyone is crowded into, and a crowded one-sided trade has no buyers on the other side when it unwinds. The franc at 1.20 felt safest in its final months, when it had held longest and the most people leaned on it. That is exactly the configuration that gaps the hardest. The comfort is the risk.

Third, and most general: **an exchange rate is the relative price of two monies, and no central bank can hold a relative price against the flow of money forever** — it can only hold it until the cost of holding exceeds the cost of letting go. The franc floor was a three-and-a-half-year attempt to override the gap between European fear and Swiss safety with the printing press, and it worked right up until the press itself became the problem. When a central bank blinks, it doesn't blink slowly. It blinks at 09:30 on a quiet Thursday, and a third of the value of a currency moves before you've finished reading the press release.

The franc didn't break because the SNB was weak. It broke because the SNB was honest with itself about a cost the market had stopped pricing — and the gap between those two is where every peg, eventually, comes apart.

## Further reading & cross-links

- [Currency crises and the anatomy of a speculative attack](/blog/trading/forex/currency-crises-and-the-anatomy-of-a-speculative-attack) — the general model of how a defended exchange rate gets attacked and breaks, of which the franc is one variant.
- [Soros and Black Wednesday: breaking the Bank of England 1992](/blog/trading/forex/soros-and-black-wednesday-breaking-the-bank-of-england-1992) — the mirror-image case: defending a *weak* currency with *finite* reserves, where the central bank really does run out.
- [How central banks intervene in the currency market](/blog/trading/forex/how-central-banks-intervene-in-the-currency-market) — the toolkit and mechanics of FX intervention, of which the franc floor was the most extreme example.
- [2015 to 2022: the franc, the yen and the EM stress cycle](/blog/trading/forex/2015-to-2022-the-franc-the-yen-and-the-em-stress-cycle) — how the franc de-peg fits into the broader modern run of currency stress events.
- [Carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) — why being long EUR/CHF at the floor was a negative-skew carry trade, and how those detonate.
- [The central-bank game: credibility, commitment and don't-fight-the-fed](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed) — the formal coordination-game logic behind why a doubted peg becomes self-defeatingly expensive to hold.

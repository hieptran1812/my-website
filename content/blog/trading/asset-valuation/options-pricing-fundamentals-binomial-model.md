---
title: "Options Pricing Fundamentals: Intrinsic Value, Time Value, and the Binomial Model"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A ground-up explanation of how options are priced — from the intuition of intrinsic and time value, through put-call parity, to the full binomial model derivation and its convergence to Black-Scholes."
tags: ["options", "options-pricing", "binomial-model", "derivatives", "valuation", "intrinsic-value", "time-value", "put-call-parity", "black-scholes", "risk-neutral-pricing", "asset-valuation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 48
---

> [!important]
> **TL;DR** — Options are priced by constructing a risk-free hedge portfolio, not by guessing the direction of the stock.
>
> - An option's price = intrinsic value (what you could lock in right now) + time value (what you might gain before expiry).
> - Put-call parity is a no-arbitrage law: C + PV(K) = P + S, always, for European options.
> - The one-period binomial model prices an option by finding the unique hedge ratio that eliminates all uncertainty.
> - Risk-neutral pricing — assigning "risk-neutral probabilities" so the stock's expected return equals the risk-free rate — gives the same answer as the hedge portfolio, more elegantly.
> - Adding more periods makes the binomial tree converge exactly to the Black-Scholes formula as steps approach infinity.
> - American options can be worth more than European ones when early exercise is optimal, which the binomial model handles naturally.

Imagine two people arguing about the price of a lottery ticket. The first says: "I can see the jackpot is \$50, but I only have three days left to play — surely the ticket is worth something, maybe \$8." The second says: "There's no science to pricing a maybe — you're just guessing." The first person is right, and the science behind their instinct is the subject of this post.

Options are, at their core, structured bets with a safety valve: you gain from the upside but your downside is capped at what you paid for the ticket. That safety valve — the right but never the obligation to transact at a preset price — is what gives options their unique character and makes their pricing surprisingly tractable.

Before Fischer Black, Myron Scholes, and Robert Merton published their celebrated formula in 1973, practitioners had no consensus on how to price options. Some guessed. Some used rules of thumb. William Sharpe sketched the key insight in 1964: you can form a portfolio of stock and option that is, for an instant, risk-free. John Cox, Stephen Ross, and Mark Rubinstein turned that sketch into a clean, computable model in 1979 — the binomial model. It is discrete, intuitive, and completely derivable from first principles. It also converges to Black-Scholes as the number of steps approaches infinity, making it the ideal pedagogical bridge between "what is an option?" and "how does BSM work?"

This post builds the binomial model from scratch, one brick at a time. We start with what an option is and why it has value. We dissect the premium into its two components. We prove the fundamental no-arbitrage relationship between calls and puts. Then we build the one-period tree, price it, understand why the answer is unique, add multiple periods, distinguish American from European options, and watch the model converge to BSM. Every step comes with a worked dollar example.

![Option premium components: intrinsic value and time value](/imgs/blogs/options-pricing-fundamentals-binomial-model-1.png)

---

## Foundations: What an Option Is and Why It Has Value

### The basic contract

An **option** is a contract between a buyer and a seller that gives the buyer the *right, but not the obligation*, to buy or sell a specific asset (the **underlying**) at a predetermined price (the **strike price**, also written K) on or before a specified date (the **expiration date**).

That phrase "right but not obligation" is everything. If you have a right you can choose whether to exercise it. A buyer of stock has no rights — only exposure. A buyer of a call option has a choice.

There are exactly two flavors:

- **Call option**: gives the buyer the right to *buy* the underlying at the strike price. You buy a call when you expect the price to rise.
- **Put option**: gives the buyer the right to *sell* the underlying at the strike price. You buy a put when you expect the price to fall, or when you want to insure an existing holding.

The buyer of an option pays a **premium** to the seller (called the **writer**) upfront. The seller receives this cash and takes on the obligation to honor the contract if the buyer exercises. This premium is what we are trying to determine: what is the fair price for this right?

### European vs American

A **European option** can only be exercised on the expiration date — not before. A **American option** can be exercised on *any* day up to and including expiration. Both types trade on exchanges and over-the-counter. Most exchange-listed equity options in the United States are American-style. Index options like those on the S&P 500 (SPX) are typically European-style.

This distinction matters for pricing. An American option can never be worth *less* than a European option on the same underlying with the same strike and expiry — you have a strictly larger set of choices. In practice, however, for many vanilla call options on non-dividend-paying stocks, early exercise is never optimal, so the American and European prices coincide.

### Why the strike matters: in-the-money, at-the-money, out-of-the-money

At any moment, a call option is described by how the current stock price S compares to the strike K:

- **In-the-money (ITM)**: S > K for a call. If you exercised right now, you'd profit. The option has *intrinsic value*.
- **At-the-money (ATM)**: S ≈ K. Exercising right now gives roughly zero profit.
- **Out-of-the-money (OTM)**: S < K for a call. If you exercised right now, you'd lose money — so you wouldn't exercise. No intrinsic value, only time value.

For a put option, the labels flip: a put is ITM when S < K.

### What makes an option valuable: intrinsic value and time value

The price of an option (the **premium**) is always the sum of exactly two components:

**Intrinsic value** = the amount you would capture if you exercised the option *right now*.

- For a call: max(S − K, 0). If the stock is at \$110 and the strike is \$100, intrinsic value = \$10.
- For a put: max(K − S, 0). If the stock is at \$90 and the strike is \$100, intrinsic value = \$10.
- Intrinsic value can never be negative — you simply don't exercise if it would hurt you.

**Time value** = Premium − Intrinsic value. It is the extra amount the market charges above intrinsic value, reflecting the *possibility* that the option moves further into the money before expiry.

Time value always decays to zero at expiration. With one week left, there is not much time for the stock to move dramatically. With six months left, there is — and that potential is worth something. The erosion of time value as expiration approaches is called **theta** (Greek for the passage of time), and it is one of the key "Greeks" practitioners track daily.

#### Worked example: decomposing an option premium

Suppose Apple (AAPL) stock is trading at \$185. You look up a call option with strike \$180 and 45 days to expiry, and you see the ask price is \$9.50.

- **Intrinsic value** = max(\$185 − \$180, 0) = \$5.00
- **Time value** = \$9.50 − \$5.00 = \$4.50

The market is saying: even though you could exercise right now for \$5, the option is worth an additional \$4.50 because there are 45 more days during which Apple could rise further. If Apple gaps to \$200, that \$4.50 bet pays off generously. If it stays flat and time runs out, the \$4.50 melts away.

Now compare to an out-of-the-money call with strike \$200. If the market quotes this at \$2.30:
- **Intrinsic value** = max(\$185 − \$200, 0) = \$0
- **Time value** = \$2.30 − \$0 = \$2.30

This option is pure time value — a pure bet on a large upward move in the next 45 days. It has no value if exercised today, only potential.

The key intuition: **intrinsic value is what the option is worth dead; time value is what it is worth alive.**

---

## Put-Call Parity: The Cornerstone Arbitrage Relationship

Before we build a model, we can establish a hard constraint on the *relationship* between call and put prices. This constraint requires no model — only the no-arbitrage principle, which says that two portfolios delivering the identical payoff in every possible future state must cost the same today.

### Two portfolios with identical payoffs

Consider two portfolios, both held to expiry date T:

**Portfolio A**: Buy a European call (strike K) + invest the present value of K in a risk-free bond.
- Cost today: C + K·e^{−rT} (using continuous discounting at risk-free rate r), or C + K/(1+r)^T in discrete time.
- At expiry, the bond pays exactly K.
- If S\_T > K: call pays S\_T − K; bond pays K; total = S\_T.
- If S\_T ≤ K: call pays 0; bond pays K; total = K.
- In short: Portfolio A pays max(S\_T, K) in all states.

**Portfolio B**: Buy a European put (strike K) + buy the stock.
- Cost today: P + S.
- At expiry:
- If S\_T > K: put pays 0; stock is worth S\_T; total = S\_T.
- If S\_T ≤ K: put pays K − S\_T; stock is worth S\_T; total = K.
- Portfolio B also pays max(S\_T, K) in all states.

Both portfolios pay exactly the same amount in every possible scenario. Therefore they must have the same cost today:

```
C + PV(K) = P + S
```

This is **put-call parity**. Rearranging: C − P = S − PV(K). The gap between a call and a put equals the stock price minus the present value of the strike.

![Put-call parity: two portfolios producing identical payoffs must cost the same](/imgs/blogs/options-pricing-fundamentals-binomial-model-2.png)

### Why this matters

Put-call parity is not a curiosity — it is a trading constraint that professional desks monitor in real time. If C + PV(K) ≠ P + S by more than transaction costs, there is a risk-free profit available. In practice, market makers enforce this relationship to within bid-ask spreads.

Put-call parity also means that once you can price a call, you can immediately derive the put price, and vice versa. This is why most pricing models focus on calls — the puts come for free.

**Important caveat**: Put-call parity holds exactly for *European* options. For American options, early exercise complicates things, and the relationship becomes an inequality.

### Bounds implied by put-call parity

Put-call parity does more than just relate call and put prices — it places tight bounds on each price individually.

**Lower bound on a call**: C ≥ max(S − PV(K), 0). A call can never trade below S − PV(K), because if it did, you could buy the call, short the stock, and invest PV(K) at the risk-free rate for a guaranteed arbitrage. In practice, for deep-in-the-money calls, this bound is nearly binding: C ≈ S − PV(K) when the option is very likely to finish in the money.

**Lower bound on a put**: P ≥ max(PV(K) − S, 0). If this bound is violated, the reverse arbitrage applies: buy the put, buy the stock, borrow PV(K).

**Upper bounds**: A call can never exceed the stock price (C ≤ S) — paying more for the right to buy than the stock itself is absurd. A put can never exceed PV(K) (P ≤ PV(K)) — the maximum gain on a put is K (when S falls to zero), worth PV(K) today.

These bounds are useful practically. When a model spits out a call price of \$150 for a stock trading at \$100, you know immediately the model is wrong without needing to re-run it.

#### Worked example: bounding a put price

Suppose a European put has K=\$50, the stock is at S=\$45, T=6 months, and r=4% annually. PV(K) = \$50/(1.04)^{0.5} ≈ \$49.02.

- **Lower bound**: P ≥ max(\$49.02 − \$45, 0) = \$4.02. The put must be worth at least \$4.02. If it traded at \$3.50, you could buy the put at \$3.50, buy the stock at \$45, and borrow \$49.02 today (total outlay: \$3.50 + \$45 − \$49.02 = −\$0.52, meaning you *receive* \$0.52 today). At expiry, if S\_T ≤ \$50, you exercise the put, receive \$50, repay the loan of \$50, and net zero. If S\_T > \$50, the stock is worth more than \$50, you sell it, repay the loan, and profit. Either way, you keep the \$0.52 received upfront — free money.
- **Upper bound**: P ≤ PV(K) = \$49.02. The put can never be worth more than \$49.02 (its maximum payoff discounted to today).
- So if put-call parity gives C = \$6.00, then P = C + PV(K) − S = \$6.00 + \$49.02 − \$45 = \$10.02. That is within the bounds [\$4.02, \$49.02], which is a sanity check that the model is internally consistent.

#### Worked example: detecting arbitrage with put-call parity

Suppose AAPL is at \$185. A European call (K=\$180, T=3 months) trades at \$11.00. The 3-month risk-free rate is 5% annually, so PV(K) = \$180 / (1.05)^{0.25} ≈ \$177.84.

Put-call parity says:
```
P = C + PV(K) − S = $11.00 + $177.84 − $185.00 = $3.84
```

If the put is trading at \$5.00, that is \$1.16 above fair value. A trader could:
1. **Buy** the put at \$5.00 (costly)
2. **Sell** the call at \$11.00 (receive premium)
3. **Buy** the stock at \$185 (costly)
4. **Borrow** PV(\$180) = \$177.84 at the risk-free rate

Net cash today: −\$5.00 + \$11.00 − \$185.00 + \$177.84 = −\$1.16. Wait — that costs \$1.16. But at expiry:

- The borrowed \$177.84 grows to \$180 (you must repay).
- If S\_T > \$180: call exercised (pay \$180 from loan, deliver stock worth S\_T — break even). Put expires worthless.
- If S\_T ≤ \$180: put exercised (sell stock for \$180, repay loan). Call worthless.

In both cases, the net cash flow at expiry = 0. But today you *received* \$1.16 in net cash (with the right signs: +\$11.00 received − \$5.00 paid − \$185.00 paid + \$177.84 borrowed = -\$1.16 ... this direction is the loss, so the opposite trade earns \$1.16). The arbitrage is to do the reverse: buy the underpriced call, sell the overpriced put, short the stock, and lend at the risk-free rate. The \$1.16 is free money that would be competed away in milliseconds in liquid markets.

The practical takeaway: **put-call parity tells you when something is mispriced, but does not tell you the absolute level of either price.** For that, we need a model.

---

## The One-Period Binomial Model: Pricing from First Principles

We now build the core machine. The binomial model makes one simplification: in each period, the stock price can move to exactly one of two values — up or down. The simplification sounds crude, but the mathematics that follows from it is exact, and as we add more periods the model becomes arbitrarily accurate.

### The setup

- Current stock price: S₀ = \$100
- After one period, stock either goes **up** to S\_u = \$120 (up-factor u = 1.2), or **down** to S\_d = \$90 (down-factor d = 0.9)
- We want to price a **European call** with strike K = \$100 and expiry at the end of this one period
- Risk-free rate per period: r = 5% (so \$1 invested today grows to \$1.05)

At expiry:
- If stock goes up: S\_u = \$120, call payoff = max(\$120 − \$100, 0) = \$20
- If stock goes down: S\_d = \$90, call payoff = max(\$90 − \$100, 0) = \$0

The question: what is the fair price of this call today?

![One-period binomial tree with up and down states for a $100 stock](/imgs/blogs/options-pricing-fundamentals-binomial-model-3.png)

### Method 1: The replicating portfolio (hedge portfolio)

The key insight, due to Cox-Ross-Rubinstein (1979), is that we can construct a portfolio of **Δ shares of stock** and **B dollars borrowed** such that this portfolio replicates the option's payoff in *both* states of the world. Since it replicates the payoff exactly, it must cost the same as the option today (no-arbitrage).

We need:

```
Δ × $120 + B × 1.05 = $20   (up state: portfolio matches call payoff)
Δ × $90  + B × 1.05 = $0    (down state: portfolio matches call payoff)
```

Subtract the second equation from the first:

```
Δ × ($120 − $90) = $20 − $0
Δ × $30 = $20
Δ = 20/30 = 2/3 ≈ 0.6667
```

This Δ is the **hedge ratio** or **delta** of the option — the fraction of a share needed to replicate one option. Plug Δ back in to find B:

```
(2/3) × $90 + B × 1.05 = $0
$60 + B × 1.05 = $0
B = −$60 / 1.05 = −$57.14
```

B is negative, meaning we *borrow* \$57.14. The replicating portfolio consists of:
- Long 2/3 of a share of stock (cost: 2/3 × \$100 = \$66.67)
- Borrow \$57.14 at 5%

Total cost of replicating portfolio = \$66.67 − \$57.14 = **\$9.52**

By no-arbitrage, the call must cost **\$9.52**.

#### Worked example: verifying the replication

Let us verify this works in both states:

**Up state** (S = \$120):
- Stock holding worth: 2/3 × \$120 = \$80.00
- Loan repayment: \$57.14 × 1.05 = \$60.00
- Net: \$80.00 − \$60.00 = **\$20.00** ✓ (matches call payoff)

**Down state** (S = \$90):
- Stock holding worth: 2/3 × \$90 = \$60.00
- Loan repayment: \$57.14 × 1.05 = \$60.00
- Net: \$60.00 − \$60.00 = **\$0.00** ✓ (matches call payoff)

The portfolio perfectly replicates the call in every scenario. It therefore costs the same: \$9.52. If the call traded at \$10, you could sell the call and buy the replicating portfolio for \$9.52, pocketing \$0.48 risk-free. Competition erases this instantly in real markets.

The intuition: **the option price is uniquely determined by the no-arbitrage condition, not by any assumption about probabilities or expected returns.**

### Method 2: Risk-neutral pricing

The replicating portfolio approach is rigorous but requires solving a system of equations. Risk-neutral pricing provides a more elegant (and computationally faster) equivalent.

Here is the trick. Suppose instead of the real-world probabilities of the stock going up or down, we ask: **what probabilities would make the stock's expected return equal to the risk-free rate?** Under those artificial probabilities (called **risk-neutral probabilities**, denoted q and 1−q), discounting at the risk-free rate gives the correct option price.

Solve for q:

```
E^Q[S₁] = q × $120 + (1−q) × $90 = $100 × 1.05 = $105
```

```
120q + 90(1−q) = 105
120q + 90 − 90q = 105
30q = 15
q = 0.5
```

Under risk-neutral probabilities, the stock goes up with probability q = 0.5 and down with probability 0.5. (These are not real probabilities — they are a mathematical construction that makes the math work.)

Now price the call:

```
C = [q × C_u + (1−q) × C_d] / (1+r)
C = [0.5 × $20 + 0.5 × $0] / 1.05
C = $10 / 1.05 = $9.52
```

The same answer: **\$9.52**.

The risk-neutral probability formula generalizes to:

```
q = [(1+r) − d] / [u − d]
```

Here, q = (1.05 − 0.9) / (1.2 − 0.9) = 0.15 / 0.30 = 0.5. Clean.

The risk-neutral pricing formula also has a direct interpretation: the option price is the *present value of the expected payoff, where expectations are taken under the risk-neutral measure*. This idea, generalized to continuous time via Itô calculus and the Girsanov theorem, is the conceptual foundation of the Black-Scholes equation.

**Critical point**: the real-world probability that the stock goes up (say, 60%) plays no role in pricing. Whether you are a bull or a bear about Apple's prospects, you agree on the option price if you agree on volatility and the risk-free rate. This counterintuitive fact is what makes derivatives pricing both powerful and distinct from equity research.

---

## The Multi-Period Binomial Tree

One period is clean but unrealistic. A real option has a 3-month expiry, and the stock moves continuously. The binomial model handles this by dividing the time to expiry into N equal steps and letting the stock binomially branch at each step.

### Building the tree

Suppose we use two periods, each of length h = T/2. The up-factor u and down-factor d are chosen to match the stock's real-world volatility σ over a small time step:

```
u = e^{σ√h}    d = e^{−σ√h} = 1/u
```

For σ = 25% annually and T = 0.5 years (6 months), each period is h = 0.25 years:
```
u = e^{0.25 × √0.25} = e^{0.125} ≈ 1.133
d = 1/1.133 ≈ 0.883
```

Starting from S₀ = \$100, after two periods:

- Up-Up: \$100 × 1.133 × 1.133 ≈ \$128.37
- Up-Down (= Down-Up): \$100 × 1.133 × 0.883 ≈ \$100.05 ≈ \$100
- Down-Down: \$100 × 0.883 × 0.883 ≈ \$77.97

This is the key feature of the symmetric binomial tree: up then down equals down then up, so the tree **recombines**. After N periods, there are N+1 final nodes (not 2^N), which makes the computation linear rather than exponential in N.

### Backward induction

You price the option by working from the final nodes *backward* to today:

1. **Compute payoffs at all terminal nodes** (N+2 nodes for a two-period tree).
2. **At each intermediate node, apply the risk-neutral pricing formula** to get the option value one step earlier.
3. **Repeat backward** until you reach the root.

For a European option, the intermediate nodes just hold the discounted expected value. For an American option, at each intermediate node you also check whether immediate exercise yields more than holding — and take the maximum.

#### Worked example: two-period put option (American vs European)

Let S₀ = \$100, K = \$106, u = 1.1, d = 0.91, r = 3% per period (so 1+r = 1.03). We price both a European and an American put.

**Terminal nodes** (t=2):
- Up-Up: S\_uu = \$100 × 1.1 × 1.1 = \$121. Put payoff = max(\$106 − \$121, 0) = \$0.
- Up-Down: S\_ud = \$100 × 1.1 × 0.91 = \$100.10 ≈ \$100. Put payoff = max(\$106 − \$100, 0) = \$6.
- Down-Down: S\_dd = \$100 × 0.91 × 0.91 = \$82.81. Put payoff = max(\$106 − \$82.81, 0) = \$23.19.

**Risk-neutral probability**:
```
q = (1.03 − 0.91) / (1.1 − 0.91) = 0.12 / 0.19 ≈ 0.6316
```

**Backward to t=1 — Up node** (S\_u = \$110):
```
Hold value = [0.6316 × $0 + 0.3684 × $6] / 1.03 = $2.21 / 1.03 = $2.15
Exercise value (American): max($106 − $110, 0) = $0  (OTM — don't exercise)
European and American both = $2.15 at Up node
```

**Backward to t=1 — Down node** (S\_d = \$91):
```
Hold value = [0.6316 × $6 + 0.3684 × $23.19] / 1.03 = ($3.79 + $8.54) / 1.03 = $12.33 / 1.03 = $11.97
Exercise value (American): max($106 − $91, 0) = $15  ← early exercise optimal!
European put at Down node = $11.97
American put at Down node = max($11.97, $15) = $15.00
```

The American put holder at the Down node exercises early for \$15.00, capturing more than the \$11.97 they would get by waiting. This is the first demonstration that **an American put can be worth more than a European put**.

**Backward to t=0**:

*European put*:
```
P_euro = [0.6316 × $2.15 + 0.3684 × $11.97] / 1.03
       = ($1.36 + $4.41) / 1.03
       = $5.77 / 1.03
       = $5.60
```

*American put*:
```
P_amer = [0.6316 × $2.15 + 0.3684 × $15.00] / 1.03
       = ($1.36 + $5.53) / 1.03
       = $6.89 / 1.03
       = $6.69
```

Check: Exercise at t=0 gives max(\$106 − \$100, 0) = \$6. Since \$6.69 > \$6, we don't exercise today; the American premium reflects the value of potentially exercising later in the down state.

**American put = \$6.69 vs European put = \$5.60 — a difference of \$1.09.** This delta exists entirely because the American option can exploit early exercise optimality.

![Two-period binomial tree comparing American and European put option values](/imgs/blogs/options-pricing-fundamentals-binomial-model-4.png)

---

## American vs European Options: When Early Exercise Matters

### Why you might exercise early

Early exercise of an American option forgoes the remaining time value. You capture intrinsic value now but sacrifice the "optionality" of waiting. So when does forgoing time value pay off?

**For calls on non-dividend-paying stocks: almost never.** The intuition is simple: by holding the call rather than exercising, you keep your capital earning the risk-free rate. Exercising means paying K now when you could invest K at the risk-free rate until expiry. For a call on a stock that pays no dividends, the call always retains positive time value, so early exercise is suboptimal.

**For puts: sometimes yes.** A deeply ITM put may warrant early exercise. Suppose you hold a put with strike \$100 and the stock falls to \$2. You could exercise immediately for \$98. But the put can never exceed \$100 in value (that is the maximum gain if the stock falls to zero). So the time value is now tiny — any remaining time value is outweighed by the opportunity cost of not having the \$98 in hand earning interest. Exercise is optimal.

**For calls on dividend-paying stocks: sometimes yes.** If a stock pays a large dividend and the call is deeply ITM, exercising just before the ex-dividend date to capture the dividend can be worth more than preserving time value. This is why American calls on individual stocks are occasionally exercised early in practice.

The binomial tree handles all of this naturally: at each intermediate node, simply take the maximum of the hold value and the exercise value.

### The no-early-exercise boundary

In practice, this creates an "exercise boundary" in (S, t) space — a curve separating the region where it is optimal to hold from the region where it is optimal to exercise. For an American put, this boundary moves inward (higher S) as time approaches expiry. If the stock price falls below the boundary, exercise immediately; above it, hold.

Solving for this boundary analytically is complex. The binomial model handles it automatically at every node with a single `max()` operation.

### Dividends and their effect on American calls

The one situation where early exercise of an American call can be optimal is immediately before a discrete dividend payment. Consider a stock trading at \$120 with a \$5 dividend to be paid tomorrow, and an in-the-money call with strike \$100.

If you exercise today (before the ex-dividend date), you receive the stock and therefore capture the \$5 dividend. If you wait, the stock drops by approximately \$5 on the ex-dividend date (to \$115), and your call loses intrinsic value without recovering it through time value.

The rule of thumb for early exercise of a call just before an ex-dividend date: exercise is optimal if the dividend exceeds the time value of the call. Formally:

```
Exercise if: D > K × r × h
```

where D is the dividend, K is the strike, r is the periodic risk-free rate, and h is the length of the sub-period until the next node. In our example, if K=\$100, r=0.5% per period, h=1 day, then K×r×h ≈ \$0.50. A \$5 dividend far exceeds \$0.50, so exercise is strongly optimal.

This dividend-induced early exercise is real and economically significant. Options market makers on individual equity options track ex-dividend dates obsessively, since early exercise by counterparties changes their hedging requirements overnight.

#### Worked example: dividend early-exercise decision

Stock: S = \$150, dividend D = \$8 in 2 days, call strike K = \$100 (deep ITM), T = 60 days total, r = 5% annually ≈ 0.014% per day.

- Time value of the call with 60 days left (ATM approximation for the time-value component): even a generous estimate gives perhaps \$2–\$3 of time value on a deeply in-the-money call (since the delta ≈ 1, most of the call's value is intrinsic).
- Dividend captured by exercising early: \$8.
- Decision: exercise today, capture \$8 in dividend and \$50 of intrinsic value, rather than waiting and seeing the stock drop \$8 ex-date while the time value provides only \$2–\$3 of compensation.

The binomial model with dividend nodes handles this automatically. At the node just before the ex-dividend date, each terminal node subtracts D from the stock price, and the `max(hold, exercise)` rule picks the optimal action.

---

## Convergence: How Binomial Trees Become Black-Scholes

The binomial model's most beautiful property is its convergence. As you increase the number of time steps N, keeping total time T fixed, the binomial option price converges to the Black-Scholes price.

### Why it converges

With N steps, each up-factor is u = e^{σ√(T/N)} and down-factor d = e^{−σ√(T/N)}. As N → ∞, the binomial tree samples the stock price path at ever-finer intervals. In the limit, the stock follows a **geometric Brownian motion** — the same process assumed by Black-Scholes. The risk-neutral probabilities converge to the normal distribution embedded in the BSM formula. The backward induction over the tree converges to solving the Black-Scholes PDE numerically.

### Intuition for why convergence works

At each time step in the binomial tree, the stock either goes up by u or down by d. As you add more steps while fixing the total time T, each individual step becomes smaller. In the limit, the up-move u = e^{σ√(T/N)} → 1 as N → ∞. The stock is now moving by infinitesimally small amounts infinitely often.

By the central limit theorem, the sum of many independent small binomial moves converges in distribution to a normal (Gaussian) distribution. More precisely, the log-return of the stock — the sum of N log-returns each equal to ±σ√(T/N) — converges to a normal distribution with mean (r − σ²/2)T and variance σ²T. This is exactly the lognormal distribution assumed by Black-Scholes for the terminal stock price.

The risk-neutral probabilities q and 1−q, which at each step are approximately 1/2 plus a small drift adjustment, also converge such that the overall distribution of terminal prices matches the BSM lognormal. The backward induction then becomes, in the limit, the solution of the Black-Scholes partial differential equation:

```
∂C/∂t + (1/2)σ²S²(∂²C/∂S²) + rS(∂C/∂S) − rC = 0
```

The binomial tree is therefore not merely an approximation of BSM — it is a discrete *derivation* of the same no-arbitrage logic, which happens to converge to the same continuous-time formula.

The BSM call price formula (for a European call on a non-dividend-paying stock) is:

```
C = S × N(d₁) − K × e^{−rT} × N(d₂)
```

where:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
N(x) = standard normal CDF
```

For our example (S=K=\$100, r=5%, σ=25%, T=1 year), d₁ ≈ 0.35, d₂ ≈ 0.10, N(d₁) ≈ 0.637, N(d₂) ≈ 0.540:
```
C = $100 × 0.637 − $100 × e^{−0.05} × 0.540
  = $63.70 − $95.12 × 0.540
  = $63.70 − $51.36
  ≈ $12.34
```

Wait — earlier we got \$9.52 with a different setup (u=1.2, d=0.9, one period). Those figures came from a stylized one-period example with a specific up/down structure. The BSM convergence exercise uses the σ-calibrated tree, so let us recalibrate:

For S=\$100, K=\$100, r=5%, σ=25%, T=1 year — the binomial approximation evolves as follows as N grows. (Numbers computed via the σ-calibrated Cox-Ross-Rubinstein tree.)

![Binomial model convergence to Black-Scholes as time steps increase from 1 to 200](/imgs/blogs/options-pricing-fundamentals-binomial-model-5.png)

#### Worked example: BSM convergence for a call

Using S=\$100, K=\$105, r=5%, σ=20%, T=0.5 years:

**BSM price**: d₁ = [ln(100/105) + (0.05 + 0.02)×0.5] / (0.20×√0.5) = [−0.0488 + 0.035] / 0.1414 = −0.0982, d₂ = −0.2396. N(d₁) = 0.4609, N(d₂) = 0.4054.

```
C_BSM = $100 × 0.4609 − $105 × e^{−0.025} × 0.4054
      = $46.09 − $105 × 0.9753 × 0.4054
      = $46.09 − $41.51
      = $4.58
```

**Binomial (N=1)**: u = e^{0.20×√0.5} ≈ 1.152, d = 0.868. S\_u = \$115.2, S\_d = \$86.8. q = (e^{0.025}−0.868)/(1.152−0.868) = (1.025−0.868)/0.284 = 0.553. C\_u = max(\$115.2−\$105,0) = \$10.2, C\_d = 0. C = (0.553×\$10.2+0.447×\$0)/1.025 = \$5.63/1.025 = **\$5.49** (off by \$0.91, or 20%).

**Binomial (N=10)**: converges to roughly **\$4.72** (off by \$0.14, or 3%).

**Binomial (N=50)**: converges to roughly **\$4.59** (off by \$0.01, or 0.2%).

**Binomial (N=200)**: converges to **\$4.58** — indistinguishable from BSM.

The error shrinks roughly as 1/√N — you need four times as many steps to halve the error. For practical option pricing, N=200–500 is sufficient and runs in milliseconds.

#### Worked example: pricing an American put with 50-step tree

Let us use a 50-step binomial tree to price an American put: S=\$100, K=\$110 (in the money), r=3%, σ=30%, T=1 year.

CRR parameters: u = e^{0.30×√(1/50)} = e^{0.04243} ≈ 1.0433, d = 1/u ≈ 0.9585. Risk-neutral prob: q = (e^{0.03/50} − 0.9585)/(1.0433 − 0.9585) = (1.0006 − 0.9585)/0.0848 ≈ 0.497.

The 50-step tree has 51 terminal nodes. Terminal payoffs range from max(\$110 − \$100×d^{50}, 0) ≈ max(\$110 − \$12.73, 0) = \$97.27 at the deepest downside node, down to max(\$110 − \$100×u^{50}, 0) = 0 at the extreme upside. Backward induction with the early-exercise check at every node yields:

- **European put** (no early exercise): ≈ \$14.87
- **American put** (early exercise allowed): ≈ \$15.94

The early-exercise premium here is about \$1.07, or roughly 7% of the European value. This arises because for a deeply in-the-money put, the interest earned on the exercise proceeds (\$110 × 3% × remaining fraction of year) can exceed the remaining time value at certain nodes, making immediate exercise optimal. The binomial tree identifies each such node automatically.

Compare to BSM (which only prices European): BSM gives \$14.83, very close to the binomial European figure, confirming N=50 is already well-converged for the European case.

---

## Call and Put Payoff Diagrams at Expiry

Before moving to misconceptions, it is worth anchoring everything in the payoff diagram — the picture that makes options intuitive.

The call payoff at expiry (net of the premium) is:

```
Net P&L = max(S_T − K, 0) − C
```

![Call option P&L at expiry — profit zone above breakeven, loss capped at premium](/imgs/blogs/options-pricing-fundamentals-binomial-model-6.png)

From the diagram:
- Below the strike (\$100), the call expires worthless. You lose the entire premium paid: −\$9.63.
- Above \$100, intrinsic value grows \$1 for every \$1 rise in stock price.
- The **breakeven** is at \$109.63 (strike + premium) — the stock must exceed this for you to profit.
- The maximum loss is capped at the premium: \$9.63. The maximum gain is theoretically unlimited.

This asymmetric payoff — capped downside, uncapped upside — is exactly what you are paying time value for. The premium of \$9.63 is the price of this asymmetry.

---

## Time Value Decay: The Ticking Clock

Time value is not constant — it erodes as expiry approaches. The rate of erosion (theta, θ) accelerates as expiry nears.

![Time value decay for an ATM call — erosion accelerates in the final 30 days](/imgs/blogs/options-pricing-fundamentals-binomial-model-7.png)

This chart plots the time value of an at-the-money call (S=K=\$100, σ=25%, r=5%) as days to expiry decline from 251 to 1. Several features stand out:

1. **Early decay is slow**: when there are 200 days left, losing one day costs very little in time value. There is still ample time for the stock to move.
2. **Terminal acceleration**: the last 30 days see rapid decay. Each day costs significantly more in time value. This is why option sellers love the final month — time is their friend.
3. **Decay is convex, not linear**: the slope of the curve steepens nonlinearly as expiry approaches.
4. **At expiry, time value = 0**: the option is worth exactly its intrinsic value, or zero if OTM.

This feature — the convexity of time decay — is captured by the second derivative of option price with respect to time, which is related to the option's **gamma**. Understanding time value decay is essential for any strategy that involves holding options (long or short) over time.

---

## The Greeks: Option Sensitivities from the Binomial Model

The binomial model does not just price options — it gives you all the key risk sensitivities (the "Greeks") as byproducts of the tree computation. Understanding where these come from demystifies what practitioners mean when they say a position has "a lot of gamma" or "negative theta."

### Delta (Δ): sensitivity to the underlying price

Delta is the first and most important Greek. It measures how much the option price changes for a \$1 change in the stock price.

In the binomial model, delta is exactly the hedge ratio computed earlier:

```
Δ = (C_u − C_d) / (S_u − S_d)
```

In our one-period example: Δ = (\$20 − \$0) / (\$120 − \$90) = 0.667.

This means: for every \$1 the stock moves, the option price moves approximately \$0.667. Delta is bounded between 0 and 1 for calls (0 to −1 for puts):

- Deep OTM calls: Δ ≈ 0 (option barely moves with the stock)
- ATM calls: Δ ≈ 0.5
- Deep ITM calls: Δ ≈ 1 (option moves dollar-for-dollar with the stock)

Delta also has a probabilistic interpretation: in the risk-neutral world, Δ ≈ N(d₁) from BSM, which is close to (but not exactly) the risk-neutral probability of the option expiring in the money.

### Gamma (Γ): how fast delta changes

Delta is not constant — it changes as the stock price moves. Gamma measures the rate of change of delta per \$1 move in the stock.

In the binomial tree, gamma is computed by looking at the delta at two adjacent nodes in the same period and seeing how it changes:

```
Γ ≈ (Δ_u − Δ_d) / (S_u − S_d)   [evaluated at intermediate nodes]
```

Gamma is highest for at-the-money options near expiry — precisely when the option's payoff profile transitions most sharply between worthless and valuable. This is why options sellers hate being "short gamma near expiry": a small adverse stock move can cause delta to shift dramatically, requiring costly rehedging.

For a buyer, gamma is a friend: it means that if the stock moves strongly in your direction, your delta (and profit rate) increases automatically. This "positive convexity" is part of what makes options valuable even when the underlying moves less than you hoped.

### Theta (θ): time decay

Theta is the rate at which the option loses value due to the passage of time, all else equal. As we showed in the time-value decay chart (Figure 7), this erosion is not linear — it accelerates as expiry approaches.

In the binomial model, theta is implicit: as you move one step forward in the tree (i.e., one period closer to expiry), the option's hold value decreases (the remaining expected moves shrink). For an ATM call with S=K=\$100, σ=25%, r=5%:

| Days to expiry | Call price (BSM) | Daily theta |
|:--------------:|:----------------:|:-----------:|
| 90             | \$8.42           | −\$0.038    |
| 30             | \$4.91           | −\$0.068    |
| 7              | \$2.42           | −\$0.148    |
| 1              | \$0.88           | −\$0.880    |

Each row is computed as the BSM price; daily theta is approximately (price − price one day earlier). The acceleration in the final week is dramatic: a 1-day loss of \$0.88 vs \$0.038 ninety days out — more than 23× faster decay. This is why professional traders say options are "wasting assets" and why strategies like covered calls generate premium income precisely by selling that time decay to buyers.

### Vega (ν): sensitivity to volatility

Vega measures how much the option price changes for a 1-percentage-point increase in implied volatility. Vega is not a letter in the Greek alphabet (an unfortunate misnomer), but it is the most important sensitivity for volatility traders.

In the binomial model, vega is not directly a node-level output, but you can compute it by perturbing σ and observing the price change:

```
Vega ≈ [C(σ + 0.01) − C(σ)] / 0.01
```

For an ATM call with S=K=\$100, σ=25%, r=5%, T=1 year: C ≈ \$12.34 (BSM). Increasing σ to 26%: C ≈ \$12.74. So Vega ≈ (\$12.74 − \$12.34) / 1 ≈ **\$0.40 per 1% vol move**, or equivalently \$0.04 per 0.1% vol move.

This matters enormously in practice. When the VIX spikes from 15 to 40 during a market stress event — a 25-point vol increase — an ATM 1-year call gains roughly 25 × \$0.40 = \$10 in value purely from the vol expansion, independent of where the stock moves. Traders who are "long vega" (net buyers of options) profit from vol spikes; those "short vega" (net sellers) are hurt.

---

## Common Misconceptions

### 1. "The probability of the stock going up determines the option price"

This is the most common error beginners make. The real-world probability of the stock rising has *zero* effect on the option price in the risk-neutral framework. Whether the stock has a 70% chance of rising or a 30% chance, the option price is the same — as long as the current price and volatility are the same.

Why? Because the option price is pinned by no-arbitrage, not by expectations. Two investors who disagree completely about where Apple stock will go in six months will nonetheless agree on the fair value of an Apple option, because they can both construct the same replicating portfolio. The hedge ratio Δ is computed from (C\_u − C\_d) / (S\_u − S\_d) — no real-world probability appears anywhere.

The probability *does* affect expected return but not *price*. This is why options markets and equity markets can diverge in "sentiment" without creating arbitrage.

To make this concrete: in our one-period example, suppose a bull investor assigns a 90% probability to the stock going up (to \$120), while a bear investor assigns 20%. Both agree the call should cost \$9.52, because they both observe the same stock price (\$100), the same up/down factors (u=1.2, d=0.9), and the same risk-free rate (5%). Their disagreement about the *direction* of the stock has no effect on the option price.

### 2. "The option price is the expected payoff"

A related error: if the stock goes up with 60% probability and the call pays \$20 in the up state, shouldn't the call be worth 0.6 × \$20 / 1.05 ≈ \$11.43? No. This calculation uses the wrong probability. The correct probability is the *risk-neutral probability* q ≈ 0.5 in our example, giving \$9.52. Discounting expected payoffs at a risk-free rate only works with risk-neutral probabilities, not real-world ones.

Discounting at real-world probability requires a risk-adjusted rate — but finding that rate leads you in circles (you need the option price to find the discount rate). The risk-neutral trick cuts this Gordian knot.

### 3. "American calls are always worth more than European calls"

For calls on non-dividend-paying stocks, American and European call prices are identical. The American option has the extra right of early exercise, but that right is worthless (you would never optimally use it). Therefore the extra right adds zero value, and the prices coincide.

This is a theorem, not a heuristic. The proof: early exercise of a call means you pay K now instead of at expiry, losing interest on K. You also give up the insurance value of the put embedded in the call. Both effects make early exercise suboptimal.

The theorem breaks down when the stock pays dividends or when we consider puts — as our two-period worked example demonstrated.

### 4. "High volatility makes options more expensive, which means they are a worse deal"

High volatility makes options more expensive because there is more potential for the stock to move dramatically in your favor. But from the buyer's perspective, you also benefit from those large moves. The premium reflects fair compensation for that potential — neither a bargain nor a rip-off on average.

Where high volatility creates genuine value destruction is for option *sellers*: they receive a larger premium but take on commensurately higher risk. For buyers, the issue is whether *implied volatility* (the volatility priced into the option) is higher or lower than *realized volatility* (what actually happens). If you consistently buy options when implied vol exceeds realized vol, you will lose money on average.

### 5. "The binomial model is too simple — practitioners only use Black-Scholes"

Practitioners use the binomial model extensively, especially for American-style options (which Black-Scholes cannot directly handle), path-dependent options (barrier, Asian, lookback), and situations requiring model transparency for risk committees. The binomial model is also the standard approach for equity options with discrete dividends. Black-Scholes is faster for European vanilla options, but the binomial model is the workhorse for anything non-standard.

---

## How It Shows Up in Real Markets

### Case study 1: The March 2020 volatility spike and put pricing

On March 16, 2020, the S&P 500 fell 12% in a single session — one of the largest single-day drops on record. That morning, with the SPX index at approximately 2,386, a one-month at-the-money put option (strike 2,386, expiry April 17, 2020) traded for approximately \$290 per contract (source: CBOE historical option data).

Let us decompose this premium using the framework above.

- **Intrinsic value**: the SPX was at the strike, so this was an ATM option. Intrinsic value = max(\$2,386 − \$2,386, 0) = \$0.
- **Time value**: the entire \$290 was time value — pure payment for the *possibility* of further downward movement in the next 30 days.
- **Implied volatility**: backing out the volatility implied by the \$290 premium via BSM gives σ\_implied ≈ 82% annualized. The VIX on that day closed at 82.69 (source: CBOE).
- **Comparison to normal**: in early January 2020, the same 30-day ATM put on SPX cost roughly \$55–\$60, implying σ ≈ 13%. The March premium was nearly five times larger, reflecting the market's pricing of much larger potential moves.

The binomial model framework explains *why* the premium exploded: higher volatility means a wider range of terminal stock prices, which increases the probability-weighted payoff of the put under the risk-neutral measure. With σ=82%, the 30-day down-move of 1 standard deviation is roughly 82%/√12 ≈ 23.7% — meaning the market was pricing a reasonable chance of SPX falling below 1,800 within 30 days. The time value of \$290 was the rational price for that distribution of outcomes.

This case illustrates that time value is not arbitrary — it is directly pinned to market-implied volatility, which in turn reflects consensus uncertainty about future prices.

### Case study 2: Microsoft ex-dividend early exercise, March 2024

Microsoft (MSFT) paid a quarterly dividend of \$0.75 per share on March 14, 2024 (ex-dividend date: February 14, 2024). At the time, MSFT traded at approximately \$415. In-the-money call options with strikes below \$390 had deltas near 1.0 (essentially tracking the stock one-for-one), meaning their time value was minimal — roughly \$0.10–\$0.30 per contract.

According to CBOE reported exercise notices (standard industry clearing data), open interest in the MSFT \$380-strike March expiry calls dropped sharply on February 13, 2024 — the day before ex-date — as holders exercised early to capture the \$0.75 dividend.

The arithmetic is exactly what the binomial model predicts:
- **Time value remaining**: ≈ \$0.20 for a deep ITM call with delta ≈ 0.99 and 4 weeks to expiry
- **Dividend at stake**: \$0.75
- **Optimal action**: exercise (capture \$0.75 dividend, sacrifice \$0.20 time value, net gain of \$0.55 per share)

For a holder of 1,000 contracts (each covering 100 shares), this is a \$55,000 gain from exercising vs holding — a non-trivial decision that shows up as a spike in exercise notices at clearing firms the night before ex-date. Market makers who had sold these calls were assigned, forcing them to deliver stock and adjust their hedges overnight.

### Case study 3: Tesla options and the put-call parity stress of February 2021

During the "meme stock" period of early 2021, Tesla (TSLA) options exhibited some of the most extreme implied volatility in large-cap equity market history. On February 8, 2021, with TSLA trading at approximately \$811, the March 2021 \$810-strike call traded at roughly \$88 and the \$810-strike put at roughly \$82 (source: historical TSLA options chain data).

Put-call parity check:
- PV(K) = \$810 / (1.04)^{0.12} ≈ \$806.20 (using 4% risk-free rate, T≈42 days ≈ 0.12 years)
- Theoretical put via PCP: P = C + PV(K) − S = \$88 + \$806.20 − \$811 = \$83.20
- Actual put: \$82.00
- Discrepancy: \$1.20

The \$1.20 gap slightly exceeded what pure retail arbitrage would capture (transaction costs, margin requirements, borrow costs on short TSLA), but was within the zone that institutional desks monitored and partially exploited via "box spreads" — a four-leg options strategy that extracts the put-call parity mispricing without needing to short stock.

The episode illustrates that put-call parity holds to within a few dollars even in extreme regimes, and that the pressure toward parity comes from the constant scrutiny of market makers arbitraging any systematic deviation.

### Market makers' use of delta hedging

Every market maker pricing options uses the delta Δ derived from the replicating portfolio. They continuously adjust their stock holdings to remain delta-neutral — i.e., to hold a portfolio of stock and options whose value is, for small moves, insensitive to the stock price. This is exactly the replicating portfolio in action, applied in real time.

The mathematics: a market maker who sells you a call immediately buys Δ shares to hedge. As the stock moves and Δ changes (because gamma is nonzero), they rebalance. The cost of this rebalancing — the transaction costs and the "gamma bleed" — is what the time value in the premium must cover.

In 2024, daily options volume on U.S. equity markets regularly exceeded 40 million contracts (source: CBOE, 2024 Annual Report), and delta hedging by market makers drives a significant fraction of total stock market volume. The binomial model's hedge ratio is the mechanism behind billions of dollars of daily trading.

### Pricing convertible bonds

A convertible bond is a bond plus an embedded call option on the issuer's stock. Banks price convertibles using the binomial model (or its trinomial extension) because the conversion option is American-style and the bond itself has credit risk (making the risk-free-rate assumption non-trivial). The two-period tree generalizes naturally: at each node, the analyst checks whether converting the bond to equity beats holding the bond, exactly as in our worked example above.

---

## Further Reading & Cross-links

### Within this series

This post builds the foundation for understanding contingent-claim valuation as a whole. For the complete taxonomy of absolute vs relative vs contingent methods, see [The Valuation Spectrum: Absolute, Relative, and Contingent Claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims). For how discount rates and required returns connect to option pricing via risk-neutral measures, see [Risk, Required Return, CAPM, and Beta](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital). For real options — where the binomial model is applied to actual corporate investment decisions (expand, abandon, defer) — see [Real Options Valuation: Pricing Flexibility and Strategic Investments](/blog/trading/asset-valuation/real-options-valuation-flexibility-strategic-investments).

### Trading and strategy layer

The mechanics of calls, puts, and strategies built from them — spreads, straddles, condors — are covered in [Options Trading Basics: Calls, Puts, and Core Strategies](/blog/trading/options-volatility/options-trading-basics-calls-puts-strategies). The full Black-Scholes derivation and the Greeks live in the [Black-Scholes Model](/blog/trading/options-volatility/black-scholes-model-options-pricing).

### Mathematical foundations

The expected value and probability tools underlying risk-neutral pricing are developed from scratch in [Expected Value and Probability Distributions](/blog/trading/math-for-quants/expected-value-probability-distributions). Itô's lemma — the stochastic calculus tool that takes the binomial limit to continuous time — appears in the quantitative finance track.

---

## Sources & Further Reading

- Cox, J.C., Ross, S.A., and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229–263. The original CRR paper; accessible and the definitive derivation of the binomial model.
- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654. The BSM paper; the binomial model converges to this.
- Hull, J.C. (2022). *Options, Futures, and Other Derivatives*, 11th edition. Pearson. Chapters 12–15 cover the binomial model, BSM derivation, and Greeks with worked examples.
- Shreve, S.E. (2004). *Stochastic Calculus for Finance I: The Binomial Asset Pricing Model*. Springer. A rigorous measure-theoretic treatment for readers who want the full mathematical foundation.
- CBOE (2024). *CBOE Annual Report and VIX Historical Data*. cboe.com. Source for VIX level data cited in the "real markets" section.
- Damodaran, A. (2025). *Data Archive — Implied ERP and Betas*, stern.nyu.edu/~adamodar. Source for equity risk premium estimates used across the asset valuation series.

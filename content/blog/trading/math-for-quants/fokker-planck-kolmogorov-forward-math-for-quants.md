---
title: "Fokker-Planck: evolving the whole distribution, not one path"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "An SDE tells you how one path moves. The Kolmogorov forward equation tells you how the density moves, and almost everything a risk manager wants is a property of the density. Stationary distributions, absorbing barriers, and why one PDE solve beats a million paths."
tags: ["fokker-planck", "kolmogorov-forward", "kolmogorov-backward", "stationary-distribution", "ornstein-uhlenbeck", "first-passage", "barrier-options", "local-volatility", "monte-carlo", "pde-pricing", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** A stochastic differential equation describes one path. The Kolmogorov forward equation, which physicists call Fokker-Planck, describes the density of where all the paths are. Almost every risk question is a question about the density, and reaching for Monte Carlo to answer it is usually the slow road.
>
> - The **backward** equation runs over the *starting* point and answers "what is the expected payoff if I start here?". The **forward** equation runs over the *ending* point and answers "where does the probability mass end up?". Confusing the two is the most common error on this topic.
> - The forward equation is a **conservation law**. Write the probability current $J = \mu p - \tfrac12 \partial_x(\sigma^2 p)$ and it reads $\partial_t p = -\partial_x J$: drift transports mass, diffusion spreads it, nothing is created or destroyed.
> - Set the time derivative to zero and the current to zero, and the **stationary density** falls out in one line. For a mean-reverting spread it is Gaussian with standard deviation $\sigma/\sqrt{2\theta}$, which at $\theta = 8$ per year and $\sigma = 40$ bps is exactly 10 bps.
> - **Boundary conditions carry the modelling.** On a \$5m one-touch at a \$120 barrier, using the terminal density instead of the absorbing-barrier solution prices it at \$835,688 rather than \$1,775,701, an error of \$940,012.
> - The number to remember: on that same trade a 100,000-path daily-step Monte Carlo carries a standard error of \$7,567 and a discretisation **bias of \$136,599**. The bias is 18 times the noise, and more paths do nothing about it.

## The question one path cannot answer

A risk manager asks three things before lunch. How often does this spread sit outside my stop? What is the chance this credit touches its default barrier before the bond matures? What does the book's P&L distribution look like at quarter end?

Every one of those is a question about a **density**: about where probability mass sits, not about where any particular path went. And yet the standard reflex is to answer them with paths. Write down the stochastic differential equation, simulate a hundred thousand of them, count.

That works, and for a complicated enough book it is the only thing that works. But notice what it costs. You started with an equation that describes the model exactly, discarded it, generated a hundred thousand noisy samples from it, and used the samples to rebuild, approximately and with error bars, an object the equation already contained.

The Kolmogorov forward equation is that object written down directly. It is a partial differential equation whose unknown is the probability density itself. Solve it once and you hold every probability, every moment and every quantile at the same time, with no sampling error at all.

![Two panels contrasting the SDE view, which follows one path at a time and needs one hundred thousand simulations per question, with the Fokker-Planck view, which moves the whole density and answers every question from a single PDE solve](/imgs/blogs/fokker-planck-kolmogorov-forward-math-for-quants-1.webp)

That figure is the argument of the whole article in one image. On the left, the SDE follows a single realisation, and any probability has to be assembled by simulating many paths and counting. On the right, the forward equation moves the density itself, and any probability is an integral of the answer. Same model, two questions.

## Foundations: what a density is, and what an SDE is not

A **probability density** $p(x)$ is a function whose area over an interval is the probability of landing in that interval. The chance of sitting between $a$ and $b$ is $\int_a^b p(x)\,dx$, and the total area is one because the quantity has to be somewhere.

A **transition density** adds time and a starting point. Write $p(y, t \mid x_0, 0)$ for the density of being at $y$ after time $t$, given that you started at $x_0$. At $t = 0$ all the mass sits on $x_0$: the density is a spike. As time passes the spike spreads out and drifts. The forward equation is the rule for how that spreading happens.

A **stochastic differential equation** describes the same model from the opposite end. In the general one-dimensional form,

$$dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t$$

the $dt$ term is the **drift**, the average direction of motion, and the $dW$ term is the **diffusion**, a random kick whose size is $\sigma$. If either of those is unfamiliar, the mechanics are built from zero in [stochastic differential equations: GBM, OU and CIR](/blog/trading/math-for-quants/sdes-gbm-ou-cir-math-for-quants), and the calculus that makes the $dW$ term behave is in [the Ito integral and Ito's lemma](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants).

The distinction that matters here is small and easy to slide past. The SDE is a statement about **one path**. It says how this particular realisation moves over the next instant, given where it is now. It says nothing directly about the crowd. To get from the path to the crowd you either simulate many paths, or you find the equation the crowd obeys. The second option exists, and it is the Kolmogorov forward equation.

## Two equations, two questions

Andrey Kolmogorov wrote down two equations in 1931, and they are not the same equation twice. They have different unknowns, run over different variables, and take their data at opposite ends of time.

The **backward equation** governs $u(x,t) = \mathbb{E}[\phi(X_T) \mid X_t = x]$, the expected value of some payoff $\phi$ at a fixed horizon $T$, viewed as a function of where you start:

$$\partial_t u + \mu(x,t)\,\partial_x u + \tfrac12 \sigma^2(x,t)\,\partial_x^2 u = 0, \qquad u(x,T) = \phi(x)$$

You know the answer at the end, $u(x,T) = \phi(x)$, and you integrate backwards to today. The payoff is fixed; the starting point is the variable. This is the equation behind option pricing, and the bridge from it to the expectation is the subject of [Feynman-Kac and the Black-Scholes PDE](/blog/trading/math-for-quants/feynman-kac-black-scholes-pde-math-for-quants).

The **forward equation**, or Fokker-Planck equation, governs the density $p(y,T)$ of where you end up, viewed as a function of the endpoint:

$$\partial_T\, p(y,T) = -\partial_y\big[\mu(y,T)\,p(y,T)\big] + \tfrac12\,\partial_y^2\big[\sigma^2(y,T)\,p(y,T)\big]$$

You know the answer at the start, a spike at $x_0$, and you integrate forwards. The starting point is fixed; the endpoint is the variable.

![Comparison table setting the Kolmogorov backward equation against the forward equation across five rows: the unknown, the variable each runs over, the terminal versus initial condition, where the coefficients sit relative to the derivatives, and when to reach for each](/imgs/blogs/fokker-planck-kolmogorov-forward-math-for-quants-2.webp)

The comparison in that figure is worth memorising row by row, because it is exactly what interviewers probe. Look at the last row of the equations above. In the backward equation the coefficients $\mu$ and $\sigma^2$ sit *outside* the derivatives, multiplying $\partial_x u$ and $\partial_x^2 u$. In the forward equation they sit *inside*, and you differentiate the products $\mu p$ and $\sigma^2 p$. That is not cosmetic. It is the signature of the two operators being **adjoints** of each other: if $L$ is the backward generator, the forward equation is $\partial_t p = L^{\ast} p$. Writing the forward equation with the coefficients pulled outside the derivatives is the single most common way to get this wrong, and the error is invisible the moment $\sigma$ stops being constant.

## Why the forward equation looks the way it does

You do not need the full derivation to read the equation, and the structure is more useful than the proof. Probability is a conserved quantity: it is not created or destroyed, it only flows. Any conserved quantity obeys a continuity equation of the form "the rate of change of the amount here equals the net flow in".

Define the **probability current**

$$J(x,t) = \mu(x,t)\,p(x,t) - \tfrac12\,\partial_x\big[\sigma^2(x,t)\,p(x,t)\big]$$

and the forward equation is exactly

$$\partial_t p = -\partial_x J.$$

![Diagram of a thin slice of the state space with probability current flowing in on the left and out on the right, the resulting conservation law, and the current split into a drift term that transports mass and a diffusion term that spreads it down the gradient](/imgs/blogs/fokker-planck-kolmogorov-forward-math-for-quants-3.webp)

That figure is the equation in words. Take a thin slice between $x$ and $x + dx$. The mass sitting in it is $p(x,t)\,dx$. Current $J(x)$ flows in from the left and current $J(x+dx)$ flows out to the right, so the mass in the slice changes at the rate of the net inflow, which is minus the gradient of $J$.

The current has two pieces and they do different jobs. The **drift term** $\mu p$ carries mass bodily: wherever there is mass and a drift, the mass moves in the direction of the drift, like a current in a river. The **diffusion term** $-\tfrac12\partial_x(\sigma^2 p)$ carries mass down its own gradient, from where the density is high towards where it is low. That is Fick's law, the same thing that makes ink spread in water.

Once the equation is written this way it becomes readable. Steepening the density anywhere creates a diffusive current that flattens it. Mean reversion creates a drift current pointing inward. And the two currents fight, which is what makes the next section work.

## The stationary distribution, in one line

A distribution is **stationary** when it has stopped changing: $\partial_t p = 0$. By the conservation law that means $\partial_x J = 0$, so the current is constant in space. On the whole real line the density has to vanish at infinity, which forces that constant to be zero. The two currents are in exact balance everywhere:

$$\mu(x)\,p(x) = \tfrac12\,\frac{d}{dx}\big[\sigma^2(x)\,p(x)\big].$$

For constant $\sigma$ this is a first-order equation you can solve by inspection. Divide through and you get $p'/p = 2\mu/\sigma^2$, so

$$p(x) \;\propto\; \exp\!\left(\frac{2}{\sigma^2}\int^x \mu(y)\,dy\right).$$

Now take the Ornstein-Uhlenbeck process, the standard model for a mean-reverting spread: $dX = \theta(m - X)\,dt + \sigma\,dW$, where $m$ is the level the spread is pulled towards and $\theta$ is how hard it is pulled. The integral is $\theta(mx - x^2/2)$, and completing the square gives

$$p(x) \;\propto\; \exp\!\left(-\frac{\theta}{\sigma^2}(x - m)^2\right),$$

which is a Gaussian centred at $m$ with variance $\sigma^2/(2\theta)$. The stationary standard deviation is $\sigma/\sqrt{2\theta}$. That is the whole calculation, and it took four lines.

![Chart of the stationary Ornstein-Uhlenbeck density on a basis-point axis, a bell curve centred at 50 basis points with a standard deviation of 10 basis points, marked with the mean, the 99th percentile at 73.3 basis points and the stop-out level at 80 basis points](/imgs/blogs/fokker-planck-kolmogorov-forward-math-for-quants-4.webp)

#### Worked example 1: sizing a \$50m spread book off the stationary density

A relative-value desk is long a spread that mean-reverts with $\theta = 8$ per year and a volatility of 40 bps per square-root year, around a long-run level of 50 bps. The position makes or loses \$40,000 per basis point, and the risk mandate stops the trade out at 80 bps. All the inputs here are assumed, and the arithmetic below is illustrative work on those assumptions, not a market observation.

1. **Stationary standard deviation.** $\sigma/\sqrt{2\theta} = 40/\sqrt{16} = 40/4 = 10$ bps exactly.
2. **Is the stationary answer even legitimate yet?** The half-life is $\ln 2 / \theta = 0.0866$ years, or 21.8 trading days. After three half-lives the memory of the starting point has decayed by a factor of exactly 0.125, and the spread's standard deviation has reached 0.992 of its stationary value. So from about 65.5 trading days after entry, the stationary density is the right description.
3. **The tail.** The stop sits at $(80-50)/10 = 3.00$ standard deviations. The Gaussian tail beyond three standard deviations is 0.135%, which is about one day in 741, or 0.34 days in a 252-day year.
4. **The sizing number.** The 99th percentile of the stationary density is $50 + 2.326 \times 10 = 73.3$ bps. That is an adverse move of 23.26 bps, which at \$40,000 per bp costs \$930,539. A full run to the 80 bps stop costs 30 bps, or \$1.20m.

The desk's mandate is a \$1m one-percent-day loss limit, and \$930,539 fits it with \$69,461 to spare. Double the position and the same 23.26 bp move costs \$1,861,078, which does not.

One honest caveat, and it is the reason the next section exists. The 0.135% above is the fraction of *time* the spread spends beyond 80 bps in the long run. It is not the probability of *touching* 80 bps over some horizon, which is larger, sometimes much larger. Occupancy and first passage are different questions, and the forward equation answers the second one only once you tell it what happens at the barrier.

## Boundary conditions are where the modelling lives

The forward equation on its own does not pin down a unique solution. You have to say what happens at the edges of the state space, and that choice is not a technicality. It is the model.

**Absorbing.** Mass that reaches the boundary leaves and never returns. The condition is $p(b,t) = 0$ at the barrier. Total mass inside then decays over time, and what it decays to is the survival probability. This is a knock-out barrier, a default threshold, a margin call that liquidates the position, a fund that hits its drawdown trigger and is shut.

**Reflecting.** Mass that reaches the boundary bounces back. The condition is $J(b,t) = 0$: no current crosses. Total mass is conserved. This is a hard position limit, a currency band a central bank defends, an inventory cap a market maker will not breach.

The same process with the same parameters gives completely different answers under the two, and choosing the wrong one is a modelling error that no amount of numerical care will rescue.

For constant coefficients and a flat barrier, the absorbing solution has a closed form through the **method of images**. Take the free density, subtract a mirror copy centred at the reflection of the starting point through the barrier, and the difference vanishes at the barrier by construction. The mass that remains is the survival probability, and one minus it is the first-passage probability. That construction gives the result below, where $\Phi$ is the standard normal distribution function:

$$P\!\left(\max_{t \le T}\,(\nu t + \sigma W_t) \ge b\right) = 1 - \Phi\!\left(\frac{b - \nu T}{\sigma\sqrt{T}}\right) + e^{2\nu b / \sigma^2}\,\Phi\!\left(\frac{-b - \nu T}{\sigma\sqrt{T}}\right)$$

#### Worked example 2: a \$5m one-touch struck at the \$120 barrier

A stock trades at \$100. A client wants a one-touch that pays \$5m if the stock trades at or above \$120 at any point in the next six months. Assume 30% volatility and zero rates and dividends, so that under the pricing measure the log price has drift $\nu = -\sigma^2/2$. These inputs are assumed for the illustration.

1. **Move to log space.** The barrier is $b = \ln(120/100) = 0.1823$, the drift is $\nu = -0.045$, and $\sigma\sqrt{T} = 0.30 \times \sqrt{0.5} = 0.2121$.
2. **First term.** $(b - \nu T)/(\sigma\sqrt{T}) = 0.9655$, and $1 - \Phi(0.9655) = 0.1671$.
3. **Image term.** Because $\nu = -\sigma^2/2$ exactly, the image factor $e^{2\nu b/\sigma^2}$ collapses to $e^{-b} = 100/120 = 5/6 = 0.8333$. The second argument is $(-b - \nu T)/(\sigma\sqrt{T}) = -0.7534$, and $\Phi(-0.7534) = 0.2256$, so the term is $0.8333 \times 0.2256 = 0.1880$.
4. **Touch probability.** ${0.1671 + 0.1880 = 0.3551}$, so 35.5%.
5. **The money.** The one-touch is worth $0.3551 \times \$5{,}000{,}000 = \$1{,}775{,}701$.

Now price it the lazy way, using only the terminal density and asking whether the stock *ends* above \$120. That is the first term on its own, 16.7%, worth \$835,688. The gap is \$940,012, and the correct answer is 2.12 times the lazy one.

That gap is the entire content of the absorbing boundary condition. The terminal density counts paths that finish above the barrier. The absorbing solution also counts every path that poked through \$120 in month two and came back. On a barrier product those paths are most of the value.

## The PDE answer against the Monte Carlo answer

The natural objection is that nobody needs a PDE for this, because Monte Carlo is easier to write. Here is what that trade actually costs on the same \$5m one-touch.

#### Worked example 3: what \$5m of Monte Carlo error looks like

The payoff is an indicator, so each simulated path returns 0 or 1 and the estimator is a sample proportion with $p = 0.3551$.

1. **Sampling noise.** $p(1-p) = 0.2290$. At 100,000 paths the standard error is $\sqrt{0.2290/100{,}000} = 0.001513$, which on \$5m of notional is \$7,567. A 95% interval is \$14,831 wide either side.
2. **What it takes to tighten it.** Standard error falls with the square root of the path count, so ten times tighter costs a hundred times the paths. Pinning the price to one basis point of notional, a \$500 standard error, needs 22,901,563 paths.
3. **The part that is not noise.** Simulate with daily steps and you only check the barrier once a day, so you miss every excursion that pokes above \$120 and comes back between two closes. The Broadie, Glasserman and Kou continuity correction says a discretely monitored barrier behaves like a continuous one shifted by $\exp(0.5826\,\sigma\sqrt{\Delta t})$. With 126 daily steps over six months, $\Delta t = 0.003968$ and the exponent is 0.01101, so the simulation is really pricing a barrier at \$121.33.
4. **Re-price at the effective barrier.** The same closed form at \$121.33 gives a touch probability of 32.8% rather than 35.5%. On \$5m that is a **bias of \$136,599**.

So the daily-step Monte Carlo converges, with beautiful tight error bars, to a number that is \$136,599 wrong. The bias is 18.1 times the standard error, which means the error bars are not merely useless here, they are actively misleading: they invite you to believe a number they have no opinion about.

More paths do not help, because the bias does not depend on the path count. Shrinking the time step does help, but only like $\sqrt{\Delta t}$: pushing the bias below the sampling noise takes roughly 41,064 steps instead of 126, which at 100,000 paths is 4.11 billion time steps. The PDE with an absorbing boundary gives the exact answer on a grid a laptop solves in under a second.

In sizing terms this is the difference between knowing your edge and guessing it. A desk that marks twenty such trades off the daily-step simulation is carrying \$2.73m of mismarked value, all of it in the same direction, and none of it visible in the reported confidence intervals.

## Where the forward equation earns its place

**Local volatility calibration.** The Dupire equation, the one that extracts a local volatility function from a surface of option prices, is the Kolmogorov forward equation in disguise. Options of every strike and maturity on the same underlying are prices of the *same* terminal density, so one forward solve in strike and maturity does what a separate backward solve per option would have to repeat thousands of times. The mechanics, the smoothing that a second derivative in strike demands, and the limits of local vol are covered in [the volatility surface](/blog/trading/quantitative-finance/volatility-surface).

**First passage and default.** Anything phrased as "does it touch this level before that date" is an absorbing-boundary problem: structural credit models, knock-outs, drawdown triggers, covenant breaches, stop-outs. The reflection principle handles the flat-barrier constant-coefficient case; everything else is a numerical forward solve.

**The whole distribution from one solve.** This is the quiet advantage. Monte Carlo gives you one number per question, each with its own error bar. The forward equation gives you $p$, and every probability, moment, quantile and expected shortfall is then an integral of the same object. Ask a new question and you integrate again, for free.

The limit is dimension. Grid-based PDE work is comfortable in one or two state variables, tolerable in three, and hopeless beyond that, which is where Monte Carlo becomes the only option. That crossover, not a general preference, is what should decide the method.

## Common misconceptions

**"Forward and backward are the same equation written twice."** They are not. The unknowns are different objects, they run over different variables, and their data sit at opposite ends of time. What is true is that the forward operator is the formal adjoint of the backward generator, which is a precise relationship and not an equality. The practical tell is where the coefficients sit: outside the derivatives in the backward equation, inside them in the forward equation. That distinction is invisible when $\mu$ and $\sigma$ are constants, which is exactly why people who learned on constant-coefficient examples get it wrong on the first state-dependent model they meet.

**"Fokker-Planck is a physics thing."** Dupire's local volatility formula is a forward-equation result. Breeden and Litzenberger reading the risk-neutral density off the curvature of call prices in strike is reading the solution of the forward equation. Structural default models are absorbing-boundary problems. Any mean-reversion trade that has ever been sized off a long-run standard deviation used $\sigma/\sqrt{2\theta}$, which is a stationary Fokker-Planck solution whether or not anyone said so.

**"Monte Carlo is always easier."** It is easier to *write*. In one or two dimensions it is slower, less accurate and, for barrier payoffs, systematically biased in a way its own error bars conceal. The honest rule is that Monte Carlo wins on dimension and on payoff complexity, and loses on everything else.

## In the interview room and on the desk

The question usually arrives as something plain. "How would you compute the probability that this spread breaches 80 basis points?" The weak answer starts simulating. The strong answer asks one clarifying question first, because "breaches" is ambiguous: do you mean the fraction of time it sits beyond 80, or the probability it touches 80 at some point before a date? Those are different calculations and different equations.

Then answer in order. Write the SDE. Say that the density obeys the Kolmogorov forward equation. For the occupancy question, set the time derivative to zero, set the probability current to zero, and integrate the drift over the diffusion to get the stationary density, which for an Ornstein-Uhlenbeck spread is Gaussian with standard deviation $\sigma/\sqrt{2\theta}$. Read the tail off that. For the touching question, put an absorbing boundary at the level, note that the method of images gives a closed form when the coefficients are constant, and quote the first-passage formula. Finish with the dimension caveat: in one state variable this is a PDE solve on a laptop, and Monte Carlo would be the slow way to a noisier answer.

Three things make a candidate look strong here. Knowing the stationary standard deviation is $\sigma/\sqrt{2\theta}$ without deriving it. Distinguishing occupancy from first passage without being prompted. And volunteering the discretisation bias in barrier simulation, because it shows you have actually been wrong about this once.

The trap is the forward and backward mix-up, and it is the single most common error on this topic. A candidate who says "I would solve the Fokker-Planck equation backwards from the payoff" has just merged the two equations, and an interviewer who is paying attention will follow up by making the volatility state-dependent and asking where the coefficients go. The other trap is writing the forward equation with $\mu$ and $\sigma^2$ outside the derivatives. Both are recoverable if you catch them yourself.

Two Sigma probes this hardest, as part of its general preference for candidates who reason about distributions rather than point estimates. Citadel's multi-strategy and rates seats care about the stationary and first-passage results because they are how spread trades get sized. Any exotics desk, at Citadel Securities, Jane Street or a bank, will expect the Dupire connection and the barrier discretisation bias as working knowledge rather than trivia.

## Sources and further reading

- Kolmogorov, A. N. (1931). "Über die analytischen Methoden in der Wahrscheinlichkeitsrechnung." *Mathematische Annalen* 104, 415-458. The paper that introduces both equations.
- Risken, H. (1989). *The Fokker-Planck Equation: Methods of Solution and Applications*, 2nd ed. Springer. The standard reference for stationary solutions, boundary conditions and the probability-current formulation.
- Shreve, S. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer. Chapters 6 and 8 for the Kolmogorov equations, Feynman-Kac and first-passage distributions in a finance setting.
- Dupire, B. (1994). "Pricing with a Smile." *Risk* 7(1), 18-20. Local volatility as a forward-equation calculation.
- Broadie, M., Glasserman, P. and Kou, S. (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance* 7(4), 325-349. The source of the 0.5826 barrier-shift constant used above.
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*, 4th ed. Springer. The clearest treatment of absorbing versus reflecting boundaries.

The dollar figures in the three worked examples are illustrative arithmetic on assumed inputs, not market observations. The only empirical constant borrowed from the literature is the 0.5826 continuity correction, cited above.

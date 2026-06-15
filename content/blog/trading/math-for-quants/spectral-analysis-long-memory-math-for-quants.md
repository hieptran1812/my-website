---
title: "Spectral analysis and long memory: finding cycles and persistence in markets"
date: "2026-06-15"
description: "How the Fourier transform turns a wiggly price chart into a map of its hidden cycles, how the Hurst exponent tells a trending market from a mean-reverting one, and how fractional differencing makes a series usable for machine learning without throwing its memory away -- built from zero with worked dollar examples."
tags: ["spectral-analysis", "fourier-transform", "power-spectral-density", "periodogram", "hurst-exponent", "long-memory", "fractional-differencing", "arfima", "time-series", "rescaled-range", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** -- a price chart is a sum of repeating cycles plus a tug of memory, and two tools let you see each part: the spectrum shows you the cycles, the Hurst exponent shows you the memory.
>
> - **The Fourier transform** rewrites any series as a stack of sine waves of different lengths. The *power spectrum* says how much of the wiggle comes from each cycle length -- a daily rhythm, a weekly rhythm, or pure noise.
> - **A flat spectrum means white noise** with no edge; a *peaked* spectrum means real structure you can sometimes trade. The hard part is telling a true peak from a fluke, because the raw estimate (the *periodogram*) is noisy and biased.
> - **The Hurst exponent $H$** measures memory on one dial: $H < 0.5$ is mean-reverting (fade the move), $H = 0.5$ is a random walk (no edge), $H > 0.5$ is trending (ride the move). It is the quickest regime check a quant has.
> - **Fractional differencing** lets you make a series statistically well-behaved *without* erasing its memory -- the trick behind López de Prado's machine-learning features, where ordinary differencing throws away exactly the signal you wanted.
> - **The one number to remember**: a series with $H = 0.65$ over a horizon where a momentum trade earns, say, \$3 per \$100 of turnover, is the same persistence that, at $H = 0.35$, would have paid a mean-reversion trade instead -- one exponent, opposite strategies.

## Why a price chart hides a song

Play a single piano chord and your ear instantly does something miraculous: it hears *three separate notes* inside one blended sound. The air carried a single squiggly pressure wave, yet your brain decomposed it into the individual pitches that made it. That decomposition -- taking one complicated wiggle and finding the pure tones hiding inside it -- is the single most useful idea in all of signal processing, and it has a precise mathematical name: the *Fourier transform*. The astonishing claim behind it is that **any** wiggle, no matter how messy, is secretly a sum of simple, regular waves of different speeds, and we can recover exactly how much of each one is in there.

Now look at a price chart instead of listening to a chord. A stock's intraday path looks like noise -- up a little, down a little, a lurch, a drift. But traders have suspected for a century that there are *rhythms* buried in that noise: a daily pulse where volume and volatility spike at the open and the close and sag at lunch; a weekly pulse where Mondays behave unlike Fridays; longer seasonal swings. If those rhythms are real, the same trick that pulls three notes out of a chord can pull the cycles out of a price series and tell us their length and their strength. That is *spectral analysis*, and the chart it produces -- the *spectrum* -- is to a time series what a prism's rainbow is to a beam of white light.

![A price series passes through the Fourier transform into a power spectrum, whose dominant cycle becomes a timing edge.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-1.png)

The diagram above is the mental model for the first half of this post. You feed in a price series -- a column of numbers, one per time step. The Fourier transform decomposes it into cycles and the power spectrum reports how much energy sits at each cycle length. A tall spike at "one day" means a strong daily rhythm; a flat line means no rhythm at all, just noise. If a peak is real and stable, the cycle it names can become a *timing edge* -- knowing when, within the day or the week, the move tends to happen.

The second half of the post is about a different question that the spectrum hints at but does not directly answer: not *which cycles* repeat, but *how much the past leans on the future*. A market with **memory** -- where today's direction makes tomorrow's direction more likely, for a long, long time -- behaves nothing like a coin-flipping random walk, and the difference is the whole ballgame for whether you should chase a trend or fade it. The tool that measures that memory on a single dial is the *Hurst exponent*, and the tool that lets you keep that memory while still doing honest statistics on the series is *fractional differencing*. By the end you will be able to compute a spectrum, read it without fooling yourself, estimate a Hurst exponent by hand, and choose a fractional-differencing parameter that keeps your signal alive.

We assume no finance and no math background beyond comfort with a square root. Every term is defined the first time it appears, every formula gets a plain-English picture before the symbols, and every idea is anchored in a worked example with real dollars. This is educational material about how these tools behave -- it is not investment advice, and every edge described here is fragile in ways the cautions section spells out.

## Foundations: cycles, power, and memory

Before any transforms, let us nail down the handful of ideas everything else is built from. A reader who knows them cold can skim; a beginner cannot proceed without them.

### What a "time series" and a "cycle" are

A *time series* is just a list of numbers in order, one per time step: a stock's closing price each day, a currency's exchange rate each minute, the temperature each hour. The order matters -- shuffle it and you have destroyed the thing we care about, which is how each value relates to the ones before and after it.

A *cycle* is a pattern that repeats. The cleanest possible cycle is a *sine wave*: a smooth up-and-down ripple that returns to where it started after a fixed interval. Three numbers describe any sine wave completely. Its *period* is how long one full up-and-down takes -- a daily cycle has a period of one day. Its *frequency* is just the period turned upside down -- one over the period -- and it counts how many full cycles fit into a unit of time. A daily cycle observed on hourly data has a period of 24 hours and therefore a frequency of $1/24$ cycles per hour. Its *amplitude* is how tall the ripple is -- how far it swings above and below its center. A big-amplitude daily cycle means a strong, obvious daily rhythm; a tiny-amplitude one means a faint one you would never spot by eye.

The single most important claim of this whole field, *Fourier's theorem*, is this: **every** time series, however jagged, can be written as a sum of sine waves of different frequencies, each with its own amplitude. Some series need only a few waves; some need thousands; but the recipe always exists and is unique. Spectral analysis is the act of finding that recipe -- of reading off how much of each frequency is present.

### What "power" and "the spectrum" mean

When we add up the contributions of all the cycles, we usually do not care about their raw amplitude so much as their *power*, which is just the amplitude squared. Squaring does two jobs. It makes everything positive (a cycle swinging "down" first contributes just as much rhythm as one swinging "up" first), and it matches the physical intuition that doubling a wave's height quadruples its energy -- the same reason a doubling of volatility quadruples variance.

The *power spectrum*, also called the *power spectral density* (PSD), is simply the chart of power against frequency: for each possible cycle length, how much of the series' total wiggle that cycle accounts for. A series that is all one strong daily rhythm has a spectrum that is a single tall spike at the daily frequency and nearly zero everywhere else. A series that is pure random noise has a spectrum that is *flat* -- every frequency contributes about equally, which is exactly why such noise is called *white*, by analogy with white light containing all colors equally.

![The same series shown two ways: a hard-to-read sequence of values over time and an obvious set of cycles in the frequency domain.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-2.png)

The figure above shows the two views side by side, and it is worth sitting with. On the left is the *time domain*: the value at 9:30, at noon, at 3:00 -- the raw series, where any repeating pattern is buried and hard to see by eye. On the right is the *frequency domain*: the same information re-sorted by cycle length, where a daily cycle stands tall, a half-day cycle is medium, and fast wiggles are tiny. No information is lost going from one to the other -- the Fourier transform is perfectly reversible -- but the pattern that was invisible on the left is obvious on the right. That re-sorting is the entire payoff of spectral analysis: it moves the structure to where your eye can find it.

### What "memory" means in a series

Memory is a separate idea from cycles, and confusing the two is a classic beginner mistake. *Memory* (also called *persistence* or *long-range dependence*) is about whether knowing the past helps you predict the future, and for how long. A *random walk* -- the path you trace by flipping a coin each step and moving up on heads, down on tails -- has *no* memory: the next step is independent of everything that came before, so the best forecast of tomorrow is always just today. A series with *positive* memory tends to keep going the way it has been going (an up move makes another up move more likely) -- that is *trending* or *persistent*. A series with *negative* memory tends to reverse (an up move makes a down move more likely) -- that is *mean-reverting* or *anti-persistent*.

The reason memory matters more than almost anything else to a trader is that it dictates which strategy can even work. In a trending series, *momentum* pays -- you buy what has been rising. In a mean-reverting series, momentum is suicide and *reversion* pays -- you sell what has risen and buy what has fallen. In a memoryless random walk, neither works and any apparent edge is luck. So the very first thing a quant wants to know about a new series is: does it have memory, and which kind? The number that answers that on a single dial is the *Hurst exponent*, and we build it up carefully later.

> A spectrum tells you *which rhythms* a series repeats. The Hurst exponent tells you *how stubbornly* it leans on its own past. They are different questions, and a series can have a strong cycle and weak memory, or strong memory and no cycle at all.

### The first worked example: reading a tiny spectrum by hand

Let us make "power per frequency" completely concrete on a series small enough to do by hand, before any heavy machinery.

#### Worked example: the power in a four-point series

Suppose a thinly traded micro-cap shows these four de-meaned hourly returns (de-meaned means we subtracted the average so it centers on zero), in cents per \$100 of position: $+2, -2, +2, -2$. Even by eye this is a perfect two-hour cycle: up, down, up, down, repeating every two steps.

With four data points, the Fourier transform checks for cycles at just three lengths: a flat component (the average, which we already removed, so it is zero), a slow cycle that fits once across the four points, and the fastest cycle that alternates every single step. Our data alternates every step, so *all* its power should land on that fastest cycle.

Running the numbers, the fastest-cycle coefficient picks up the full swing: it sums $(+2) - (-2) + (+2) - (-2) = 8$ in the alternating pattern, while the slow-cycle coefficient sums to zero because our data does not have a slow wiggle. Squaring and normalizing, essentially **100% of the series' power sits at the two-step cycle, and 0% at the slow cycle**. The spectrum is a single spike.

Now suppose your strategy is to buy this name whenever the last hour was down (expecting the alternation to continue) and sell whenever it was up. If each correct call captures the \$2 swing per \$100 and you are right as long as the cycle holds, the spectrum has just told you, quantitatively, that the alternation accounts for *all* the predictable wiggle -- so a two-hour reversion trade has the whole edge, and a slow trend trade has none.

The intuition: the spectrum is a budget of where a series spends its variance, and a single tall bar means a single cycle is worth chasing.

## 1. The frequency-domain view

Everything above was the picture; now we make it precise enough to compute. The good news is that you almost never compute a Fourier transform by hand -- one function call does it -- so the goal here is to understand what the function returns and how to read it, not to grind through the integral.

### From a series to its frequencies

Take a series of $N$ numbers, $x_0, x_1, \dots, x_{N-1}$. The *discrete Fourier transform* (DFT) turns it into $N$ new numbers, $X_0, X_1, \dots, X_{N-1}$, one for each frequency the data can resolve. The formula looks intimidating but says something simple:

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, k n / N}.$$

Here $k$ indexes the frequency (cycle number $k$ fits $k$ full cycles across your whole sample), $x_n$ is the $n$-th data point, and the $e^{-2\pi i k n/N}$ term is a compact way of writing "a cosine and a sine of frequency $k$" using the complex number $i$. The whole sum is a *correlation*: it slides a pure wave of frequency $k$ along your data and measures how strongly the data resonates with it. A big $|X_k|$ means your data contains a lot of that frequency; a small one means it contains little.

The frequency that index $k$ corresponds to is $f_k = k / (N \,\Delta t)$, where $\Delta t$ is the spacing between samples. So with hourly data ($\Delta t = 1$ hour) over $N = 168$ hours (one week), $k = 7$ corresponds to a frequency of $7/168 = 1/24$ cycles per hour -- the *daily* cycle, because seven daily cycles fit in a week. That arithmetic, mapping a coefficient index back to a real-world period, is the single most error-prone step in practice, and getting it wrong is how people "discover" cycles that are just counting mistakes.

In real code nobody uses the slow DFT sum directly; the *fast Fourier transform* (FFT) computes the same numbers in a flash, and it is one line:

```python
import numpy as np

x = returns - returns.mean()          # always de-mean first
X = np.fft.rfft(x)                     # complex Fourier coefficients
power = np.abs(X) ** 2                 # power at each frequency
freqs = np.fft.rfftfreq(len(x), d=1)  # frequency for each bin (d = sample spacing)
period = 1 / freqs[1:]                 # cycle length in samples (skip the zero-frequency bin)
```

The `power` array is your raw spectrum; the `period` array tells you what cycle length each entry means. The whole game from here is reading peaks in `power` and translating them through `period` into "a cycle of length X bars."

### Why the lowest and highest frequencies are special

Two frequencies bracket everything you can see, and both are set by your data, not by the market. The *lowest* frequency you can resolve corresponds to a single cycle spanning your *entire* sample -- you cannot detect a cycle longer than your data, because you have not watched it repeat even once. Watch a market for three months and you simply cannot confirm a one-year seasonal cycle; you have only a quarter of one swing. This is why claims about long cycles need long histories, and why a "40-day cycle" found in 50 days of data is almost certainly nonsense.

The *highest* frequency you can resolve is the *Nyquist frequency*: one cycle per two samples, the fastest alternation your sampling can capture. Anything wiggling faster than that gets *aliased* -- folded down and disguised as a slower cycle, exactly the way a fast-spinning wheel can look like it is turning slowly backward on film. If real five-minute structure exists but you only sampled hourly, it does not vanish; it sneaks into your spectrum wearing a false, slower period and can manufacture a peak that is not there. The fix is to sample fast enough, or to filter out the fast stuff before sampling -- never to trust a peak near the Nyquist edge without sanity-checking the sampling.

## 2. The periodogram and power spectral density

The raw `power = |X|^2` we just computed has a name: the *periodogram*. It is the most natural estimate of the power spectrum -- just the squared size of each Fourier coefficient -- and it is also, frustratingly, a deeply flawed one. Understanding *why* it is flawed is what separates a quant who finds real cycles from one who trades on noise.

### The periodogram, defined

The periodogram at frequency $f_k$ is

$$I(f_k) = \frac{1}{N}\, |X_k|^2 = \frac{1}{N}\left| \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N} \right|^2.$$

The $1/N$ just normalizes for sample size. $I(f_k)$ is your estimate of how much variance the data spends on the cycle of frequency $f_k$. Sum the periodogram across all frequencies and you recover the total variance of the series -- that is *Parseval's theorem*, and it is the precise sense in which the spectrum is a budget: it partitions the series' variance across cycle lengths, and the pieces add up to the whole.

The practical use is direct: a *tall, isolated* periodogram value at some frequency is a candidate cycle. Translate its frequency to a period, check that the period makes economic sense (a daily, weekly, or monthly rhythm has a plausible cause; a 37.4-bar rhythm probably does not), and you have a hypothesis to test.

#### Worked example: a dominant intraday cycle and a timing edge

You have a year of 30-minute returns for a liquid futures contract. The trading day is 6.5 hours, which at 30-minute bars is 13 bars per day. You de-mean the series, run the FFT, and find the periodogram has a clear, towering peak at a period of exactly **13 bars** -- a one-day cycle -- standing far above its neighbors, plus a smaller-but-real peak at 6.5 bars (a *half-day* cycle, the well-known morning-and-afternoon double pulse).

To turn this into a dollar edge, you look at *where in the cycle* the return concentrates. Averaging each of the 13 intraday slots across the year, you find the last bar of the day (3:30-4:00) averages **+\$4 per \$100** of position while the lunch bar (12:00-12:30) averages **-\$1 per \$100**. The 13-bar spectral peak was the statistical fingerprint of that closing drift; the slot-by-slot average is its dollar size.

A simple timing rule follows: hold the position into the close and stand aside at lunch. If the closing bar reliably pays +\$4 and trading it costs you, say, \$1 in fees and spread, the cycle is worth a net **+\$3 per \$100** per day it holds -- a modest but real intraday timing edge, *as long as the cycle is stable*, which the cautions section will immediately complicate.

The intuition: a periodogram peak is a "when," and a periodogram peak only becomes money when you also measure the "how much" at that point in the cycle.

### Why power spectral density, not just the periodogram

Practitioners often say "PSD" rather than "periodogram" because they mean a *smoothed*, better-behaved estimate of the same underlying spectrum. The periodogram is the rawest possible estimate; the PSD is what you get after you tame its wild variance with the techniques in the next section. They estimate the same true thing -- the power spectral density of the process -- but the periodogram is a noisy single sample of it, and the PSD is an averaged, trustworthy reading.

## 3. The periodogram, its bias, and spurious cycles

Here is the uncomfortable truth that the cheerful "find the peak" story hides: **the periodogram does not get more accurate as you collect more data.** Add a decade of history and each individual periodogram value is just as noisy as it was with a year. This is genuinely strange -- almost every other estimator in statistics tightens as the sample grows -- and it is the source of nearly every false cycle ever traded. Three problems compound it.

### Problem one: the periodogram is inconsistent

For most other quantities -- a mean, a variance -- more data shrinks the error toward zero. Not the periodogram. When you double your sample, you do not get more accurate estimates at the old frequencies; you get *estimates at twice as many frequencies*, each one still rattling around its true value with roughly 100% relative error. Statistically, each periodogram value behaves like the true spectrum times a random multiplier that averages to one but has a huge spread. So the raw periodogram of even an infinitely long pure-noise series is a jagged forest of spikes, and any one of those spikes, if you go hunting, looks like a "cycle."

The cure is *smoothing*. Because neighboring frequencies estimate nearly the same true power, you can average them: replace each periodogram value with the average of itself and its neighbors (this is *Welch's method* when done by averaging over overlapping data segments, or a *smoothed periodogram* when done by averaging neighboring bins). Averaging $m$ independent-ish bins cuts the noise by roughly $\sqrt{m}$, at the cost of blurring fine frequency detail. The art is choosing $m$ large enough to kill the noise but small enough to keep the peaks distinct -- the same bias-variance tradeoff that runs through all of statistics.

![A five-step spectral analysis workflow stacked from detrending the series through smoothing to sizing a bet on real cycles.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-4.png)

The stack above is the disciplined workflow that keeps you honest, and every step exists to defeat one of these three problems. First *detrend and demean* -- subtract the average and any obvious trend, because a trend masquerades as a giant low-frequency cycle and drowns everything. Second *compute the periodogram*. Third *smooth* it, to beat the inconsistency just described. Fourth *test the peaks against white noise*, so you only believe peaks that a pure-noise series would not have produced by chance. Only then, fifth, do you *size a bet* on the cycles that survive. Skipping straight from step two to step five is exactly how a backtest gets seduced by a cycle that was never there.

### Problem two: leakage and bias

The periodogram is also *biased*: power from a strong cycle does not stay put at its true frequency but *leaks* into neighboring frequencies, smearing the spectrum. The cause is that your data is a finite window -- you chopped an ongoing process off at both ends -- and that abrupt chopping is mathematically equivalent to multiplying the true infinite signal by a rectangular window, which itself injects a spray of spurious frequencies. A strong cycle whose period does not fit a whole number of times into your sample leaks the worst, building little side-lobes that can look like extra, smaller cycles.

The standard fix is a *taper* or *window function* -- multiply your data by a gentle bell shape (a Hann or Hamming window) that fades to zero at both ends before transforming, so there is no abrupt edge to leak. Tapering trades a little resolution for far less leakage, and it is close to mandatory whenever you care about *small* peaks sitting near *large* ones, because otherwise the big peak's leakage will bury the small one.

### Problem three: multiple testing and spurious cycles

The deepest trap is statistical, not numerical. A spectrum over $N$ data points offers roughly $N/2$ frequencies, and you are scanning all of them for the tallest. Even in *pure white noise* -- a series with no cycles whatsoever -- the tallest of several hundred random periodogram values will be substantially higher than the average just by chance. If you only test "is the biggest peak bigger than average," you will declare a cycle in literally random data most of the time. This is the *multiple-testing* problem wearing a spectral disguise, and it is the single most common way spectral analysis lies to traders.

The honest test is *Fisher's g-test* (or a simple simulation): compare the *height of your tallest peak relative to the total* against the distribution you would get from white noise of the same length. Only if your peak is taller than, say, 99% of the tallest peaks that random noise produces do you have evidence of a real cycle. Equivalently, simulate a thousand white-noise series of your length, record the biggest peak in each, and see whether your real peak beats nearly all of them. A peak that fails this test is a *spurious cycle*: a phantom your eye and your hope conjured out of noise.

#### Worked example: a spurious cycle that costs \$50,000

A trader runs a periodogram on 250 days of a stock's returns and finds a peak at a 17-day period -- a "tri-weekly cycle." The peak is the tallest in the spectrum and looks convincing. He builds a strategy that buys at the cycle's trough and sells at its crest, backtests it on the same 250 days, and sees a tidy **+\$50,000** profit on a \$1,000,000 book.

Then he runs the honest check: he simulates 1,000 white-noise series of 250 days each and records the tallest peak in every one. His 17-day peak turns out to be *taller than only 60%* of the random peaks -- nowhere near the 99% bar. In plain terms, six out of ten purely random series would have shown a peak at least this tall somewhere. The cycle is noise. He trades it live anyway, and over the next year it earns **-\$50,000** as the "cycle" dissolves -- the classic signature of an overfit pattern: it pays in-sample and bleeds out-of-sample.

The intuition: the tallest peak in a noisy spectrum is almost always real-looking and almost always fake, and the only defense is asking what pure noise of the same length would have produced.

## 4. Long memory and the Hurst exponent

We now switch from *which cycles* to *how much memory*, the second great theme. The two are linked -- long memory shows up in the spectrum as a peak at the very lowest frequencies, a spectrum that blows up as frequency goes to zero -- but memory has its own, more directly tradable measure: the *Hurst exponent*.

### The Nile, the desert, and the origin of $H$

The story is real and worth knowing because it builds the intuition perfectly. In the 1950s the hydrologist Harold Hurst was sizing a dam on the Nile. To size a reservoir you need to know the worst run of floods and droughts it must survive. Hurst measured the *range* of the river's accumulated flow -- how far above and below average the running total wandered -- over windows of different lengths. If the Nile's yearly flow were independent coin-flips, that range should grow with the square root of the window length, $\sqrt{n}$, the signature of a random walk. Instead Hurst found it grew faster, like $n^{0.7}$. Wet years clustered with wet years and dry with dry: the river had *memory*, far longer memory than chance allowed, which is why ancient Egypt suffered runs of famine no coin-flip model predicted.

That exponent, the power of $n$ at which the range grows, is the *Hurst exponent* $H$. It is defined by

$$E\!\left[\frac{R(n)}{S(n)}\right] \sim c \, n^{H},$$

where $R(n)$ is the *range* of the cumulative deviations over a window of length $n$, $S(n)$ is the *standard deviation* over that window (dividing by it makes the measure scale-free), $c$ is a constant, and $\sim$ means "grows proportionally to." The ratio $R/S$ is the *rescaled range*, which is why this is called *R/S analysis*. You estimate $H$ as the slope of $\log(R/S)$ against $\log(n)$ across many window sizes -- a single straight-line fit.

### Reading the dial

The Hurst exponent lives between 0 and 1, and the value 0.5 splits the world in two.

![A matrix mapping the Hurst exponent to behaviour, memory, and the matching strategy across three regimes.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-3.png)

The matrix above is the whole payoff of the Hurst exponent in one table. When $H < 0.5$ the series is *mean-reverting* and *anti-persistent*: an up move is more likely to be followed by a down move, the range grows *slower* than a random walk's, and the strategy that fits is to *fade* moves -- sell strength, buy weakness. When $H = 0.5$ the series is a pure *random walk* with *no memory*: the range grows exactly like $\sqrt{n}$, the past tells you nothing, and there is *no edge* of this kind to have. When $H > 0.5$ the series is *trending* and *persistent*: an up move makes another up move more likely, the range grows *faster* than $\sqrt{n}$, and the strategy that fits is to *ride momentum* -- buy strength, sell weakness. One number, three regimes, two opposite trades.

> A Hurst exponent is a momentum-versus-reversion verdict in a single decimal. Above one half, the market remembers and you ride it; below, the market forgets fast and over-corrects, and you fade it.

#### Worked example: classifying a price series by its Hurst exponent

You have 256 daily returns of a commodity and want to know whether to trade it with momentum or reversion. You run R/S analysis by hand-sized steps: split the series into windows of length $n = 8, 16, 32, 64, 128$, and for each window compute the rescaled range $R/S$, then average over all windows of that length.

Suppose you get these averages:

| Window $n$ | $R/S$ | $\log_2 n$ | $\log_2(R/S)$ |
|---|---|---|---|
| 8 | 2.6 | 3.0 | 1.38 |
| 16 | 3.9 | 4.0 | 1.96 |
| 32 | 5.7 | 5.0 | 2.51 |
| 64 | 8.6 | 6.0 | 3.10 |
| 128 | 12.8 | 7.0 | 3.68 |

The Hurst exponent is the slope of $\log(R/S)$ against $\log(n)$. From $n=8$ to $n=128$, $\log_2(R/S)$ rose from 1.38 to 3.68 -- a rise of 2.30 -- while $\log_2 n$ rose from 3.0 to 7.0 -- a rise of 4.0. The slope is $2.30 / 4.0 = \mathbf{0.58}$. So $H \approx 0.58$: meaningfully above 0.5, a *trending* series with positive memory.

The dollar implication: a momentum trade fits this name and a reversion trade does not. If the persistence is strong enough that buying after an up-day captures, say, **+\$2.50 per \$100** on average over the next few days net of costs, that edge exists *because* $H > 0.5$. Run the same calculation on a series that returned $H = 0.42$ and the verdict -- and the trade -- would flip to reversion. (Two honest cautions: $H$ estimates are noisy, especially below a few hundred points, and short-sample R/S is biased *upward* toward trending, so a single $0.58$ on 256 points is a hint, not a proof.)

The intuition: the Hurst exponent reads the growth rate of a series' wandering, and "wanders faster than a coin-flip" is the mathematical face of a tradable trend.

## 5. White noise versus AR spectra

Now we connect the spectrum to the kind of process generating the data, because the *shape* of a spectrum is a fingerprint that tells noise apart from structure. This is where spectral analysis pays off as a signal-versus-noise detector.

### The flat spectrum of white noise

*White noise* is a series of independent, identically distributed random draws -- the purest possible "no structure" series, with no memory and no cycles. Its spectrum is *flat*: every frequency carries the same expected power, because independence means there is no frequency the data prefers. (The raw periodogram of white noise is still jagged, because of the inconsistency problem from earlier; but its *true, smoothed* spectrum is a horizontal line.) A flat spectrum is the null hypothesis of spectral analysis -- the "nothing to see here" baseline that any claimed cycle must beat.

### The peaked spectrum of an AR process

An *autoregressive* (AR) process is one where each value depends on the previous ones plus a fresh shock. The simplest, AR(1), is $x_t = \phi \, x_{t-1} + \varepsilon_t$: today is $\phi$ times yesterday plus noise, where $\phi$ (between $-1$ and $1$) is the *persistence* parameter. When $\phi$ is positive and close to 1, the series is sticky and slow-moving -- it has long, lazy swings -- which means its power piles up at *low* frequencies (long cycles), and its spectrum *rises toward zero frequency*. When $\phi$ is negative, the series alternates fast and its power piles up at *high* frequencies. Either way, the spectrum is *peaked*, not flat, and the location of the peak tells you the character of the process. (If you want the full mechanics of AR and MA processes, see the [AR, MA, and ARIMA deep-dive](/blog/trading/math-for-quants/ar-ma-arima-math-for-quants).)

![A flat white-noise spectrum with no signal beside a peaked AR spectrum that bulges at low frequencies.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-5.png)

The contrast above is the diagnostic in one picture. On the left, white noise: low, mid, and high cycles all carry equal power, the spectrum is flat, and there is *no signal to trade* -- no frequency stands out, so no forecast beats "tomorrow equals the long-run average." On the right, an AR process with positive persistence: low-frequency cycles carry huge power, mid cycles less, high cycles almost none, the spectrum is *peaked*, and there is *structure to trade* -- the slow swings are partly predictable. Reading the shape of a spectrum is, at bottom, reading whether a forecast can beat a coin.

#### Worked example: telling an AR(1) signal from white noise

You have two candidate series, A and B, each 1,000 points, and a budget for exactly one strategy. You want the one with real structure. You compute and smooth each spectrum.

Series A's smoothed spectrum is essentially flat: the power at the lowest frequency (1.0 in arbitrary units) is about the same as the power at the highest (0.9). Series B's smoothed spectrum is steeply peaked: power of **6.0** at the lowest frequency falling to **0.4** at the highest -- a 15-to-1 ratio.

Series A is white noise. To confirm, you fit an AR(1) and get $\phi \approx 0.02$, statistically indistinguishable from zero -- no persistence, nothing to forecast. Series B fits $\phi \approx 0.45$: a meaningful, positive persistence whose spectral signature is exactly that low-frequency bulge. A one-step forecast of B using $\hat{x}_t = 0.45\, x_{t-1}$ explains a real slice of next-step variance; the same forecast on A explains essentially nothing.

The dollar implication: you allocate your capital to a strategy on B. If B's predictable component is worth, conservatively, **+\$1.50 per \$100** of turnover net of costs, while A's is worth \$0 (its flat spectrum guaranteed it), the spectrum alone steered \$1,000,000 of capital to the only book with an edge. The shape of the line was the entire decision.

The intuition: flat means fair coin and no forecast; peaked means a memory you can lean on, and the steeper the peak the stronger the lean.

## 6. Fractional differencing and ARFIMA

We arrive at the most modern and, for machine-learning quants, the most important idea in the post. It resolves a genuine dilemma: to do honest statistics or feed a model, a series usually must be made *stationary*; but the standard way to make it stationary *destroys the memory you wanted to trade*. Fractional differencing threads that needle.

### The stationarity dilemma

A series is *stationary* if its statistical character -- its mean, its variance, the way it correlates with its own past -- does not drift over time. Most statistical tools and most machine-learning models assume stationarity; feed them a wandering, trending price and they hallucinate relationships that fall apart. (For the full treatment of why, see the [stationarity and autocorrelation deep-dive](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants).)

The textbook fix is *differencing*: instead of modeling the price $P_t$, model the change $P_t - P_{t-1}$ -- the return. Returns are usually stationary even when prices wander. But here is the catch that López de Prado made famous: *full* differencing (subtracting the whole previous value, an integer order of $d = 1$) is a sledgehammer. It makes the series stationary, yes -- but a return series has *almost no memory*; the correlation between today's return and last week's return is essentially zero. You bought stationarity by throwing away exactly the long memory -- the slow, persistent structure -- that you wanted your model to learn from. You stationarized the signal to death.

![Raw integer differencing erases memory while fractional differencing of order 0.4 keeps both stationarity and memory.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-7.png)

The before-after above is the heart of the idea. On the left, integer differencing with $d = 1$: the series becomes stationary (good) but its correlation with its own past collapses near zero, its memory is erased, and it becomes a *weak predictor*. On the right, *fractional* differencing with $d = 0.4$: the series is *also* stationary, but it keeps a strong correlation with its past -- it preserves the memory -- and so it remains a *strong predictor*. The whole trick is that the amount of differencing need not be a whole number.

### Differencing by a fraction

How can you difference by 0.4 of a step? The answer is a beautiful piece of mathematics. The differencing operation can be written using the *backshift operator* $B$, where $B x_t = x_{t-1}$ (shift back one step). Full differencing is $(1 - B) x_t = x_t - x_{t-1}$. Differencing $d$ times is $(1-B)^d$. The leap is to allow $d$ to be *any* real number and expand $(1-B)^d$ as an infinite series using the binomial theorem:

$$(1-B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-B)^k = 1 - d\,B + \frac{d(d-1)}{2!}B^2 - \frac{d(d-1)(d-2)}{3!}B^3 + \cdots$$

For an integer $d = 1$ the series stops after two terms and gives back the ordinary first difference. For a *fractional* $d$, the series never stops -- the fractionally differenced value is a weighted sum of *all* past values, with weights that shrink slowly. Those slowly shrinking weights are precisely the long memory: each new value still remembers the distant past, just with gently fading importance, instead of remembering only yesterday (full differencing) or everything equally (no differencing). In practice you truncate the sum once the weights fall below a tiny threshold, so it is finite to compute.

The parameter $d$ is a dial between two extremes. At $d = 0$ you have the original (non-stationary, full-memory) series. At $d = 1$ you have the ordinary difference (stationary, no-memory). Somewhere in between sits the *smallest* $d$ that just barely achieves stationarity -- and because it is the smallest, it sacrifices the *least* memory. That sweet spot is what you hunt for.

### Finding the sweet-spot $d$

The recipe, due to López de Prado, is mechanical and worth memorizing. Sweep $d$ upward from 0 in small steps. At each $d$, fractionally difference the series and run a stationarity test -- the *augmented Dickey-Fuller* (ADF) test, which returns a statistic that must drop below a critical value for the series to count as stationary. Also track the *correlation* between the fractionally differenced series and the original, a proxy for how much memory survived. As $d$ rises, the ADF statistic falls (more stationary) and the correlation falls (less memory). You pick the *smallest* $d$ at which the ADF test passes -- the first moment you cross into stationarity -- because every step beyond that just throws away memory for no benefit.

#### Worked example: choosing $d$ to keep memory and pass the stationarity test

You have a price series that is badly non-stationary. The ADF test needs its statistic below the critical value of **-2.86** (the standard 5% threshold) to declare stationarity. You sweep $d$ and record both the ADF statistic and the correlation of the differenced series with the original price:

| $d$ | ADF statistic | Stationary? | Correlation with original |
|---|---|---|---|
| 0.0 | -0.4 | No | 1.00 |
| 0.2 | -1.6 | No | 0.97 |
| 0.4 | -3.1 | **Yes** | **0.86** |
| 0.6 | -4.8 | Yes | 0.55 |
| 0.8 | -6.5 | Yes | 0.21 |
| 1.0 | -8.9 | Yes | 0.04 |

Reading the table: $d = 0.2$ is not yet stationary (ADF $-1.6$ is above $-2.86$). At $d = 0.4$ the ADF statistic ($-3.1$) drops below the threshold for the first time -- *this is the sweet spot*. Crucially, at $d = 0.4$ the series still correlates **0.86** with the original price, meaning it retains the overwhelming bulk of its memory. Compare that to full differencing at $d = 1.0$, which also passes (ADF $-8.9$) but retains a pathetic **0.04** correlation -- memory gone.

The dollar implication: you feed the $d = 0.4$ series into your machine-learning model instead of plain returns. Suppose a model trained on plain returns ($d=1$, correlation 0.04) earns a Sharpe ratio of 0.3 and the same model trained on the $d=0.4$ features earns 0.8, because it can finally see the persistent structure. On a \$10,000,000 book at, say, 10% annual volatility, lifting Sharpe from 0.3 to 0.8 is the difference between roughly **+\$300,000** and **+\$800,000** of expected annual profit per unit of risk taken -- a half-million-dollar swing bought purely by choosing $d = 0.4$ instead of $d = 1$.

The intuition: integer differencing is the only kind most people know, and it is almost always too much -- the right amount of differencing is usually a fraction, chosen as the smallest dose that cures the non-stationarity.

### ARFIMA: the model that bakes memory in

When you take the standard ARIMA model -- AR for momentum, I for differencing, MA for shock-echoes -- and replace the integer differencing order $d$ with a *fractional* one, you get *ARFIMA* (the F is for *fractionally integrated*). ARFIMA can represent long memory directly: a single fractional $d$ between 0 and 0.5 produces a process whose autocorrelations decay *slowly*, like a power law, rather than the fast geometric decay of ordinary ARIMA. That slow decay is the mathematical definition of long memory, and it is exactly what the Hurst exponent measures -- in fact $d$ and $H$ are tied together by $H = d + 0.5$, so a fractional differencing order of $d = 0.15$ corresponds to a Hurst exponent of $H = 0.65$, the trending value from our TL;DR. The two halves of this post -- the Hurst exponent and fractional differencing -- are the same idea seen from two angles.

![A tree rooted at long-range dependence branching into how to measure it via the Hurst exponent and how to use it via fractional differencing and ARFIMA.](/imgs/blogs/spectral-analysis-long-memory-math-for-quants-6.png)

The tree above ties the long-memory family together. The root is *long-range dependence* -- the single phenomenon of a series leaning on its distant past. It splits into two questions. *How to measure it* gives you the Hurst exponent $H$, estimated by the rescaled range $R/S$. *How to use it* gives you fractional differencing of order $d$, and the ARFIMA model that builds that fractional order into a full forecasting machine. Measure with $H$, exploit with $d$, and remember they are linked by $H = d + 0.5$.

## Common misconceptions

**"A peak in the periodogram is a cycle."** No -- a peak is a *hypothesis* about a cycle, and in noisy data the tallest peak is usually a fluke. The periodogram is inconsistent (it never tightens with more data) and you are scanning hundreds of frequencies at once, so even pure noise reliably produces a tall-looking peak. A peak is real only if it survives a white-noise test (Fisher's g-test or a simulation) *and* holds up out of sample. Treat every peak as guilty until proven innocent.

**"More data makes the spectrum more accurate."** For the raw periodogram, false. More data buys you *more frequencies*, each still estimated with huge relative error, not *better* estimates of the old ones. Accuracy comes only from *smoothing* -- averaging neighboring frequencies or data segments -- which trades frequency resolution for statistical stability. The size of your sample sets the *longest* cycle you can see and the *finest* frequency you can resolve, but never the per-frequency precision; only smoothing does that.

**"Long memory means there are predictable cycles."** Different things. A cycle is a *repeating* pattern at a fixed period; long memory is *persistence* with no fixed period -- a tendency to keep drifting that shows up as a spectrum blowing up at zero frequency, not as a discrete peak. A trending market ($H > 0.5$) has long memory but may have no clean cycle at all. Spectral peaks and the Hurst exponent answer related but distinct questions, and conflating them leads to trading a "cycle" that is really just a slow drift.

**"Hurst above 0.5 guarantees a profitable momentum trade."** No. The Hurst exponent describes the *statistical character* of past data; it says nothing about whether the persistence is large enough to overcome transaction costs, whether it will persist into the future, or whether it is already arbitraged away. Worse, R/S estimates of $H$ are biased and noisy in short samples -- short series tend to read *above* 0.5 even when generated by a true random walk. A high $H$ is a reason to *investigate* momentum, never a license to trade it.

**"Fractional differencing is an exotic technique I will never need."** It is one of the most practical tools in modern quant machine learning. Any time you must feed a price-like series into a model that assumes stationarity, plain differencing into returns is throwing away signal. Fractional differencing is the standard fix for keeping memory while satisfying the stationarity requirement, and it is a few lines of code. If you build features from financial time series, you need it.

**"The spectrum and the autocorrelation are different tools telling you different things."** They are mathematically *the same information* in two costumes. The *Wiener-Khinchin theorem* says the power spectrum is the Fourier transform of the autocorrelation function -- a peaked spectrum and a slowly decaying autocorrelation are two views of the identical underlying dependence. If you have mastered the autocorrelation function, you already half-understand the spectrum; they are not competing tools but the same tool in the time domain versus the frequency domain.

## How it shows up in real markets

### 1. The intraday volatility smile of the trading day

The single most robust spectral feature in equity and futures markets is the *daily seasonality of volatility and volume*: both are high at the open, sag at lunch, and rise again into the close, forming a U-shape every single day. Run a periodogram on intraday volume and the daily-frequency peak towers over everything, and the half-day peak (the morning-and-afternoon double pulse) is usually clearly present too. This is one of the few spectral "cycles" that is genuinely real and stable, because it has a structural cause -- overnight news arriving at the open, end-of-day rebalancing and position-squaring at the close. Execution desks lean on it constantly: a *VWAP* (volume-weighted average price) algorithm front-loads and back-loads its trading to match the U-shaped volume curve precisely because the daily spectral cycle is dependable. Here the spectrum is not a fragile alpha; it is infrastructure.

### 2. The Joseph effect in commodities and the Nile

Hurst's original finding -- that the Nile's flow had long memory, $H \approx 0.7$ -- was named by Benoit Mandelbrot the *Joseph effect*, after the biblical seven fat years followed by seven lean years: persistence so strong that good runs and bad runs cluster far beyond chance. The same long memory appears in many commodity prices, where supply (planting decisions, mine capacity, drilling) adjusts slowly and demand shifts in waves, so prices trend persistently rather than oscillating randomly. A measured $H$ meaningfully above 0.5 on a commodity is the statistical echo of those slow real-world adjustment cycles, and it is part of why trend-following *managed-futures* funds -- which buy what is rising and sell what is falling across dozens of commodity contracts -- have a coherent reason to exist: they are harvesting the Joseph effect.

### 3. Mandelbrot, fat tails, and the failure of $H = 0.5$

Modern quantitative finance was born partly from Mandelbrot's insistence, starting in the 1960s, that markets are *not* the tidy random walks ($H = 0.5$, Gaussian, no memory) that the textbook models assumed. He documented both *long memory* (the Joseph effect, $H \neq 0.5$) and *fat tails* (the *Noah effect*, wild jumps far bigger than a normal distribution allows). The 1987 crash, when the S&P 500 fell **22.6% in a single day** -- an event a Gaussian random walk says should never happen in the lifetime of the universe -- was the violent proof. Spectral analysis and Hurst measurement are the descendants of that critique: they are the tools you reach for precisely when you suspect a series is *not* a memoryless random walk, which, Mandelbrot argued and 1987 confirmed, is most of the time.

### 4. Fractional differencing in machine-learning alpha

The practical revival of fractional differencing came from Marcos López de Prado's 2018 work on machine learning for asset management. The problem he named is concrete: quant ML pipelines need stationary features, so practitioners reflexively converted prices to returns -- and then watched their models underperform because returns are nearly memoryless. His fix, sweeping $d$ to find the minimum order that passes the ADF test (typically a fraction like 0.3 to 0.5, not 1.0), let models see the persistent structure in prices while still satisfying stationarity. Desks that adopted it reported features that retained correlations above 0.9 with the original price while passing every stationarity check -- the difference, in the worked example above, between a Sharpe of 0.3 and 0.8. It is now standard practice in serious quant-ML feature engineering, and a good example of how a piece of 1980s time-series theory became a 2020s production tool. It pairs naturally with the broader discipline of [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research), where feature stationarity quietly decides whether a model generalizes.

### 5. The endless graveyard of spurious market cycles

For over a century, traders have "discovered" cycles in markets -- the 4-year presidential cycle, the 54-year Kondratiev wave, lunar cycles, sunspot cycles, the "decennial pattern." Almost all of them are spurious in exactly the technical sense from earlier: a periodogram peak that fails a white-noise test, found by scanning many possible periods over a too-short history and remembering only the hits. The deep reason they persist is psychological -- humans are pattern-detection machines and a cycle story is satisfying -- compounded by the statistical trap that *any* finite noisy series will display *some* tall peak. The discipline of the white-noise null test and out-of-sample confirmation is the only known antidote, and it is why a careful quant trusts the boring, structurally-caused daily cycle while dismissing the exciting, story-driven multi-year one.

### 6. Pairs spreads and the search for low Hurst

The mirror image of trend-following is *statistical arbitrage*, which hunts for series that *mean-revert* -- exactly the $H < 0.5$ regime. The classic construction is a *pairs spread*: the difference between two related stocks (or a stock and a basket) that historically move together, so the spread should oscillate around zero. Quants screen thousands of candidate spreads and keep the ones with the lowest Hurst exponents, because a low $H$ is the statistical promise that the spread snaps back -- the precondition for a profitable fade-the-deviation trade. When such a spread's Hurst exponent drifts up toward 0.5 over time, it is a warning that the relationship is breaking and the mean-reversion edge is dying, a signal as actionable as the original low $H$ was. The whole pairs-trading playbook is, at root, an industrial-scale hunt for anti-persistence.

## When this matters to you

If you only ever trade by eyeballing charts, spectral analysis and the Hurst exponent will change how you *see* a series. The next time someone shows you a chart and claims a "cycle," you will know the right questions: How long is the history -- can it even contain a full cycle? Was the peak tested against white noise? Does it hold out of sample? And the next time you wonder whether a market is one to chase or to fade, you will reach for the Hurst exponent as a first, cheap diagnostic rather than guessing -- while remembering that it is noisy, biased in short samples, and a reason to investigate rather than a verdict.

If you build models on financial data, the practical takeaway is sharper: stop reflexively converting prices to returns. Fractional differencing is the standard, few-lines-of-code way to make a series stationary while keeping the memory your model needs, and skipping it may be quietly capping your performance. Sweep $d$, find the smallest one that passes the ADF test, and feed *that* to your model.

A closing dose of honesty, because this field seduces. The spectrum is a budget of variance, not a crystal ball; the Hurst exponent is a description of the past, not a promise about the future; and every "cycle" and every "memory" edge described here can evaporate the moment enough capital chases it, or can have been a statistical phantom all along. The tools in this post are at their best as *detectors and filters* -- ways to tell structure from noise and to avoid trading the noise -- and at their most dangerous as *generators* of confident-looking patterns. Use them to be skeptical first and opportunistic second. None of this is investment advice; it is a description of how the mathematics behaves.

For where to go next: the [stationarity and autocorrelation deep-dive](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants) builds the time-domain foundations the spectrum mirrors, and the [AR, MA, and ARIMA deep-dive](/blog/trading/math-for-quants/ar-ma-arima-math-for-quants) gives you the models whose spectra you have now learned to read. For the practitioner's view of how cycle-hunting and memory-detection fit into honest research, see [market-data EDA and the biases that ambush quant research](/blog/trading/quantitative-finance/market-data-eda-biases-quant-research), and for turning any of this into a tradable signal, [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research). The original sources are worth reading directly: Hurst's 1951 dam paper for the origin of $H$, Mandelbrot's work on the Joseph and Noah effects for the critique of the random walk, and López de Prado's *Advances in Financial Machine Learning* for fractional differencing in practice.

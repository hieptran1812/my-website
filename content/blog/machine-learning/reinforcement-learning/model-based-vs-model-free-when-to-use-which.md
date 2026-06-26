---
title: "Model-Based vs Model-Free RL: A Practitioner's Decision Guide"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-first-principles guide to choosing between model-based and model-free reinforcement learning, with the sample-efficiency math, the compounding-error analysis behind MBPO, and a five-step decision framework you can apply to a real project."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "model-based-rl",
    "sample-efficiency",
    "machine-learning",
    "pytorch",
    "mbpo",
    "world-models",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/model-based-vs-model-free-when-to-use-which-1.png"
---

A robotics team I worked with had a beautiful PPO policy. In simulation it walked their quadruped across broken terrain like a mountain goat. Then they tried to train the same policy on the real robot, and the math stopped them cold. PPO needed roughly ten million environment steps to converge. On the physical robot, one step is a tenth of a second of motor commands plus the wear on the servos plus an engineer standing by to catch it when it falls. Ten million steps is about twelve days of continuous, supervised, hardware-destroying operation. Nobody was going to sign off on that.

The fix was not a better policy-gradient estimator. The fix was to stop throwing away information. Every time the robot took a step, it observed a transition: from this state, that action led to this next state and this reward. A model-free method uses that transition exactly once — to compute a gradient — and then discards it (or, with a replay buffer, replays it a finite number of times). A model-based method does something greedier: it *learns the dynamics* from those transitions, and then generates as many synthetic transitions as it wants by simulating its own learned model. Suddenly a few hundred thousand real steps can support millions of policy updates. The robot was walking in two days.

That is the entire trade-off in one story, and the figure below makes it concrete across eight named algorithms. Model-based reinforcement learning (MBRL) wins on **sample efficiency** because it reuses each real interaction many times through planning or imagination. Model-free reinforcement learning (MFRL) wins on **asymptotic performance and simplicity** because it never has to learn a model, so it never inherits a model's errors. By the end of this post you will be able to look at a new RL problem — a trading bot, a recommender, a chemistry optimizer, an Atari clone — and decide, with numbers behind the decision, which family to reach for and which specific algorithm inside that family.

![Matrix comparing eight reinforcement learning algorithms across sample efficiency, asymptotic performance, model-error risk, visual observation support, and compute cost.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-1.png)

This is the decision post in the series. It assumes you have met the basic objects before — a Markov Decision Process (MDP) with states, actions, a transition function, and a reward; a policy that maps states to actions; a value function that estimates expected future reward. If those are fuzzy, the unified map post (`reinforcement-learning-a-unified-map`) lays them out. Here we take them as given and ask the engineering question: given a budget of real interactions, a simulator (or not), and an observation space, what do you build?

I want to be honest about why this post exists. There are a hundred tutorials that explain how SAC works and a hundred that explain how Dreamer works, and almost none that tell you which one to type into your terminal on a Monday morning when you have a deadline, a noisy reward signal, and a manager asking why the agent has not converged yet. The literature optimizes for novelty; your job optimizes for shipping. The two are not the same activity, and the gap between them is exactly where most RL projects die. So this guide is unapologetically a *decision* document. The math is here to support decisions, not to be admired. Every equation earns its place by changing what you would type next.

## 1. The central trade-off, stated precisely

Reinforcement learning is, at its core, a loop: an agent observes a state, picks an action, the environment returns a reward and a next state, and the agent updates itself to collect more reward over time. Every algorithm in the field is a different answer to two questions — *what objective do I optimize* and *how do I estimate its gradient from samples*. The model-based versus model-free split is a third, orthogonal question: **do I learn an explicit model of the environment's dynamics, or do I optimize the policy directly from raw experience?**

Write the MDP as a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ where $P(s' \mid s, a)$ is the transition distribution, $r(s, a)$ the reward, and $\gamma \in [0, 1)$ the discount. A **model-free** method estimates a value function $Q(s, a)$ or a policy $\pi(a \mid s)$ directly from sampled transitions, never representing $P$ or $r$ explicitly. SAC, PPO, DQN, and TD3 all live here. A **model-based** method learns approximations $\hat{P}(s' \mid s, a)$ and $\hat{r}(s, a)$ from data, and then uses them — by planning, by generating synthetic rollouts, or by differentiating through them — to improve the policy. Dyna-Q, PETS, MBPO, Dreamer, and MuZero live here.

It helps to see exactly where the two families part ways in the loop. Both observe a transition $(s, a, r, s')$. The model-free agent passes that transition straight into a gradient: it computes a temporal-difference error $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ and nudges $Q$ to reduce it, or it accumulates a return and nudges the policy toward actions that earned more than expected. The transition's only job is to carry a gradient, and once the gradient is taken the transition's informational value is largely spent. The model-based agent does that *and also* fits $\hat{P}$ and $\hat{r}$ to the transition, building a little simulator that it can re-query forever. The same byte of real experience does double duty: it trains a value or policy, and it trains a model that will go on to manufacture thousands more synthetic bytes. That is the whole game.

The fundamental asymmetry: a model is a *reusable* artifact. Once you have $\hat{P}$, you can query it billions of times for the cost of a forward pass, which is orders of magnitude cheaper than a real environment step on a robot, in a wet lab, or against a live recommender. That reuse is the source of MBRL's sample efficiency. But a model is also a *lie* — it is a learned approximation, and the policy optimizer will gleefully exploit any region where the model is optimistic but the real world is not. That exploitation is the source of MBRL's instability. The entire design space of model-based RL is a set of tactics for getting the reuse while containing the lie.

There is a sharp way to see why the optimizer is dangerous. The policy improvement step does not seek out the *average* behavior of the model; it seeks out the *maximum* predicted return. Anywhere your model is accidentally too optimistic — a corner of state space where it predicts a reward cliff that does not exist, a transition it thinks leads somewhere lucrative — the argmax will find it and camp there. This is not a bug you can engineer away with more layers. It is structural: optimization is adversarial to model error by construction, because optimization's job is to find extremes and model error is largest exactly at the extremes the policy has not yet explored. A model-based system that does not actively defend against this will produce a policy that is brilliant inside the model's hallucinations and incompetent in the real world. Every robust method in this post is, at heart, a different defense against the optimizer weaponizing model error.

Model-free RL makes the opposite bet. It refuses to learn a model, so it can never be fooled by a wrong one. The price is that it must visit the real environment enough times to estimate values directly, and value estimation from Monte Carlo samples is data-hungry. The variance of a Monte Carlo return estimate does not shrink fast — you need many independent samples of the same situation to pin down its value, and in a large state space you rarely revisit the exact same situation, so you lean on function approximation to generalize, which introduces its own bias. SAC on MuJoCo HalfCheetah needs on the order of a million environment steps to reach strong return. If those steps are free (a fast simulator), that is a non-issue and MFRL's simplicity and high asymptote make it the default. If those steps are expensive, a million of them may be impossible, and MBRL's tenfold-plus sample efficiency becomes the only viable path.

A useful one-line summary to carry through the rest of the post: **MBRL trades model bias for sample efficiency; MFRL trades sample efficiency for the absence of model bias.** Everything that follows is a refinement of that sentence. And one cultural note before we get numerical: the field's prestige attaches to the asymptote — the leaderboard cares about final score, not how few samples it took. Production almost never cares about the asymptote and almost always cares about the sample count, because samples cost money and time and the asymptote is usually "good enough" long before it is "best." Keep that mismatch in mind every time you read a paper's headline result.

## 2. Sample efficiency, quantified

Let us put real numbers on "sample efficiency," because the word gets thrown around loosely. Sample efficiency is *return achieved per unit of real environment interaction*. The cleanest way to compare two algorithms is to fix a target return and ask how many real steps each needs to reach it, or to fix a step budget and compare the returns. Both framings appear constantly in the literature, and they answer slightly different questions: the first is "how long until I clear the bar I care about?" and the second is "given the budget I actually have, who is ahead?" For a practitioner the first framing is usually the right one, because you almost always have a target return in mind — a robot that walks without falling, a recommender that beats the incumbent — long before you care about who eventually reaches the global optimum.

The figure below shows where the samples come from in the two regimes, using HalfCheetah-style numbers that match the published MBPO and PETS curves. SAC, a strong model-free baseline, climbs steadily and reaches a return around 4500 at roughly one million real steps, topping out near 4800. MBPO reaches a comparable 4500 at about 200k real steps — but those 200k real steps are accompanied by roughly 900k *simulated* steps drawn from the learned model. PETS, a pure planning method, reaches around 3000 at only 100k real steps, then plateaus lower because its planning horizon is short and it has no learned policy to push the asymptote up.

![Before and after comparison showing model-free SAC using one million real steps versus model-based MBPO using two hundred thousand real steps plus nine hundred thousand simulated rollouts.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-2.png)

It is worth tabulating these benchmark numbers explicitly, because the relationships between them are the empirical heart of the whole decision. On MuJoCo HalfCheetah, treating "real environment steps to reach a fixed return" as the currency:

| Algorithm | Family | Steps to target | Target return | Observation | Notes |
| --- | --- | --- | --- | --- | --- |
| SAC | Off-policy MF | ~1,000,000 | ~4500 | proprioceptive state | strong asymptote ~4800 |
| PPO | On-policy MF | ~3,000,000+ | ~4500 | proprioceptive state | needs more steps, very stable |
| MBPO | Hybrid MB | ~200,000 | ~4500 | proprioceptive state | ~5x more efficient than SAC |
| PETS | Planning MB | ~100,000 | ~3000 | proprioceptive state | plateaus lower, no learned policy |
| Dreamer | Latent-model MB | ~500,000 | ~4000 | **pixels** | competitive from raw images |

Read this table carefully because the columns interact. PETS reaches its target at the fewest *real* steps of anyone — 100k — but its target is lower (3000, not 4500), which is the whole point: pure short-horizon planning is wildly sample-efficient but asymptotically capped. MBPO needs twice PETS's real steps but clears a much higher bar, because SAC is doing the policy optimization underneath and there is no planning-horizon ceiling. Dreamer's 500k figure looks worse than MBPO's 200k until you notice the observation column: Dreamer is doing it *from pixels*, learning the dynamics of a visual scene, which is a categorically harder modeling problem than HalfCheetah's clean proprioceptive state. Comparing Dreamer's pixel result to MBPO's state result is comparing two different sports; the fair comparison is Dreamer-from-pixels against SAC-from-pixels, and there Dreamer's sample-efficiency edge is large.

The crossover structure is the thing to internalize. Early in training — the first hundred thousand steps — model-based methods dominate decisively, often by a factor of five to ten in sample efficiency, because they squeeze every real transition for all it is worth. Late in training, model-free methods catch up and frequently pass, because they are optimizing the true objective with no model bias capping their asymptote. The crossover point — where the two curves meet — typically sits somewhere between several hundred thousand and a few million real steps on standard continuous-control benchmarks, depending on the environment's dynamics complexity.

Why does the crossover happen at all? Because the model-based agent's asymptote is limited by the *residual model error that never goes to zero*. Even a beautifully trained dynamics model has some irreducible bias — the environment has stochasticity the model smooths over, contact dynamics it approximates, sensor noise it averages out. The policy that is optimal under the slightly-wrong model is slightly-wrong for the real world, and that gap is a ceiling no amount of synthetic data can break through, because more synthetic data just means more confident pursuit of the model's particular flavor of wrong. The model-free agent has no such ceiling: given enough real samples it converges toward the true optimum, full stop. So the two curves cross because one of them has a glass ceiling and the other does not. The model-based agent sprints out of the gate and then hits the ceiling; the model-free agent jogs and then keeps climbing past where the model-based one is stuck.

So the regime question becomes: *where does your real-step budget fall relative to the crossover?* If your budget is far below the crossover (robotics, biology, a slow recommender), you live in the regime where MBRL's advantage is enormous and you should pay the complexity tax. If your budget is far above the crossover (a fast simulator where you can take a billion steps overnight), the MBRL advantage has long since evaporated and you should take the simpler, higher-asymptote model-free method. The dangerous middle — a budget right at the crossover — is where you should run both for a few hundred thousand steps and let the curves decide, because no general rule beats a direct measurement when you are near the tie.

#### Worked example: MBPO's 5x advantage in concrete numbers

Suppose you are training on a physical manipulation task where one real step costs 0.2 seconds of robot time, and you want to reach a return of 4500. With SAC you need about 1,000,000 real steps, which is $1{,}000{,}000 \times 0.2\,\text{s} = 200{,}000$ seconds $\approx 55.6$ hours of continuous robot operation. With MBPO you need about 200,000 real steps to hit the same return, which is $200{,}000 \times 0.2\,\text{s} = 40{,}000$ seconds $\approx 11.1$ hours. That is a 5x reduction in wall-clock robot time. The model rollouts and SAC updates add GPU compute, but GPU seconds cost pennies and do not wear out a \$30,000 robot. The economics are not close. This is exactly the calculation the quadruped team ran, and it is why they switched families.

Now push the example one step further, because the 5x figure understates the real saving. The 55.6 hours of SAC operation are not 55.6 continuous hours — they are 55.6 hours that require a human supervisor present to reset and rescue the robot, spread across multiple days because no one supervises a thrashing robot for a 56-hour shift. Call it two weeks of calendar time with one engineer's attention. The MBPO version's 11.1 robot-hours might fit in two supervised working days. The hidden multiplier is the human in the loop: every real step is not just 0.2 seconds of motor time, it is a fraction of an expensive engineer's attention and a fraction of the robot's finite mechanical lifespan. When you fold those in, the effective cost ratio between model-free and model-based on real hardware is frequently larger than the headline sample ratio, not smaller. The samples are not the only thing that is scarce.

The lesson is not that one family is better. It is that *sample efficiency only matters when samples are expensive*, and the expense is a property of your problem, not your algorithm. Get that ordering right and most of the decision makes itself. And "expensive" is multi-dimensional — it is wall-clock, money, hardware wear, human supervision, and risk, any one of which can be the binding constraint. A simulator step that costs a microsecond of GPU time but might, in the real deployment it stands in for, correspond to a clinical decision is not cheap in the sense that matters.

## 3. Why model error compounds — the MBPO analysis

The central danger of model-based RL is that errors in the learned model compound over a rollout. If you understand exactly how they compound, you understand why the whole modern model-based playbook revolves around *short* rollouts. This is the analysis from Janner et al.'s 2019 MBPO paper, and it is worth deriving rather than asserting.

Suppose your learned model $\hat{P}$ has a bounded one-step error: under some distance measure on distributions (total variation), the model's predicted next-state distribution differs from the true one by at most $\epsilon_m$ for states the policy actually visits. Now roll the model forward for $H$ steps to generate a synthetic trajectory. At step 1 the state distribution is off by about $\epsilon_m$. At step 2, you are now predicting from a state that was itself already wrong, so errors stack: the deviation is roughly $2\epsilon_m$. At step $k$ the compounding gives a deviation on the order of $k \cdot \epsilon_m$. The figure below shows this stack-up: one step is fine, five steps are getting risky, ten steps drift into a danger zone, and twenty steps produce trajectories the policy can exploit but the real world will never honor.

![Stacked layers showing model rollout error growing roughly linearly with horizon, from a small one-step error to an unusable twenty-step rollout.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-4.png)

Let me make the compounding intuition airtight, because "errors stack" is the kind of phrase that sounds obvious and hides the actual mechanism. The reason error grows with horizon is a feedback of two sources. First, there is the per-step modeling error: even from a *perfectly correct* input state, the model's predicted next state is off by $\epsilon_m$. Second, and more insidiously, after step one you are no longer feeding the model correct input states — you are feeding it the model's own previous (slightly wrong) outputs. So at step two the model makes its fresh $\epsilon_m$ of error *on top of* an input that was already $\epsilon_m$ wrong, and because dynamics are typically sensitive to their inputs, that input error propagates forward and gets added to. The total is the sum of all the fresh per-step errors plus the propagated accumulation of every earlier error. In the simplest linear approximation those add up to roughly $H \cdot \epsilon_m$, but in systems with any instability — and most interesting control problems have unstable modes — the propagation term can grow *faster* than linearly, which is why the practical safe horizon is even shorter than the linear bound suggests.

Formally, MBPO bounds the gap between the true expected return $\eta[\pi]$ and the return $\hat{\eta}[\pi]$ estimated from model rollouts. The bound has the shape

$$\eta[\pi] \geq \hat{\eta}[\pi] - C(\epsilon_m, \epsilon_\pi, H, \gamma),$$

where the correction term $C$ grows with the model error $\epsilon_m$, with the policy shift $\epsilon_\pi$ (how much the current policy differs from the data-collecting policy), and crucially *with the rollout horizon $H$*. The takeaway from the bound is sharp: **for a fixed model error, there is an optimal rollout length, and it is short.** Long rollouts let compounding error dominate; the synthetic return becomes a fantasy that no longer lower-bounds the real return, and optimizing against it produces a policy that is good in the dream and bad in reality.

There is a clean piece of structure hiding in that bound that I want to surface, because it is the single most actionable result in model-based RL. The correction term $C$ has two competing pieces as a function of horizon $H$. On one hand, a longer horizon gives you *more useful synthetic data* and lets credit propagate further, which helps the policy — call this the benefit, and it grows with $H$. On the other hand, a longer horizon compounds more model error, which hurts — call this the cost, and it grows faster than the benefit as $H$ increases because the error accumulation is super-linear. Whenever you have a benefit that grows and a cost that grows faster, there is an interior optimum. Differentiating the bound's horizon-dependence and setting it to zero gives an optimal horizon of the form

$$H^* \approx \sqrt{\frac{C_0}{\epsilon_m}},$$

where $C_0$ is a problem-and-policy-dependent constant. The qualitative content is the part to carry around: **the optimal rollout horizon shrinks as model error grows.** A better model earns longer rollouts; a worse model demands shorter ones. This is not a heuristic someone tuned — it falls directly out of the return bound. It is also why the rollout schedule in real MBPO implementations starts short (when the model is freshly initialized and bad) and lengthens over training (as the model's held-out error drops). You are tracking $H^*$ as $\epsilon_m$ falls.

This is the single most important idea in modern model-based RL and the reason "model-based" no longer means "long-horizon planning." MBPO's resolution is to roll out the model for only $H = 1$ to $5$ steps, *branched from real states* sampled out of the real replay buffer rather than from a long imagined trajectory. Each synthetic rollout starts from a state the real world actually produced, so it never strays more than a few steps from reality before being thrown away. You get the data-multiplication benefit without the fantasy. The branching trick deserves emphasis: a single long imagined trajectory of length 1000 accumulates error all the way to step 1000, but a thousand independent length-1 rollouts each branched from a real state accumulate only one step of error apiece, while giving you the same thousand synthetic transitions. Same data volume, dramatically lower error budget. That asymmetry — between one long rollout and many short branched ones — is the geometric heart of why MBPO works and why naive "imagine a long trajectory" model-based RL does not.

#### Worked example: error propagation in a 10-step rollout

Take a model with a modest one-step state-prediction error of $\epsilon_m = 0.02$ in normalized total-variation distance — a genuinely good model on a control task. Under the linear-compounding approximation, a rollout of length $H$ has accumulated error on the order of $H \cdot \epsilon_m$. At $H = 1$ the error is $0.02$, negligible. At $H = 5$ it is $0.10$, still tolerable. At $H = 10$ it is $0.20$, and at $H = 20$ it is $0.40$ — meaning the model's predicted state distribution overlaps the truth by barely more than half. A policy trained to maximize reward under a distribution that is 40% wrong will learn to seek out exactly the states where the model is most optimistic, which are precisely the states where it is most wrong. This is why every robust MBRL system you will ship uses short rollouts. If you remember one number from this post, make it $H \in [1, 5]$.

#### Worked example: choosing the optimal horizon from measured model error

Now use the $H^* \approx \sqrt{C_0 / \epsilon_m}$ relationship to pick a horizon, the way you actually would on a project. Suppose you have measured your model's normalized one-step error and found it sitting at $\epsilon_m = 0.05$ early in training, and you have empirically calibrated the policy-dependent constant to roughly $C_0 = 0.45$ for your task (you calibrate it by sweeping a couple of horizons and seeing where return-improvement-per-real-step peaks). Then $H^* \approx \sqrt{0.45 / 0.05} = \sqrt{9} = 3$ — you should roll out three steps. Train for a while; the model improves and you re-measure $\epsilon_m = 0.0125$. Now $H^* \approx \sqrt{0.45 / 0.0125} = \sqrt{36} = 6$ — the better model has earned a horizon of six. This is exactly the *grow the horizon as the model earns trust* schedule, derived from numbers instead of intuition. If you ever see your model's held-out error tick *up* (the policy wandered somewhere the model has not seen), the formula tells you to *shrink* the horizon back down immediately, before the optimizer notices the new fantasy. Watching $\epsilon_m$ and adjusting $H$ to track $H^*$ is, in practice, ninety percent of operating a model-based system safely.

## 4. MBPO in detail — the recipe that actually works

Model-Based Policy Optimization (Janner et al., 2019) is the algorithm I reach for first when I need model-based sample efficiency on a continuous-control problem, because it is the one that most cleanly resolves the compounding-error problem. Its structure is worth knowing cold because it is the template most modern hybrids follow.

The figure below shows the data flow. Real transitions from the environment go two places at once: into a real replay buffer, and into training a learned dynamics model. The dynamics model is then used to generate short rollouts ($H = 1$ to $5$ steps) branched from states sampled out of the real buffer, and those synthetic transitions go into a separate model replay buffer. Finally, an off-policy model-free learner — SAC, specifically — trains the policy on a mixture of both buffers. The model is a data amplifier; SAC is the actual policy optimizer.

![Graph of the MBPO data flow showing real environment transitions branching into a dynamics model and a replay buffer, with short model rollouts merging into the SAC policy update.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-3.png)

Three design choices make MBPO robust. First, the dynamics model is a *probabilistic ensemble* — typically five to seven neural networks, each outputting a Gaussian over the next state, trained on bootstrapped subsets. The ensemble captures both aleatoric uncertainty (each net predicts a variance) and epistemic uncertainty (the nets disagree where data is scarce). When you roll out, you sample which ensemble member to use at each step, which keeps the policy from over-trusting any single net's hallucination. Second, the rollouts are short and branched, per the compounding-error analysis above. Third, the policy learner is *off-policy* (SAC), so it can consume the model buffer's synthetic data without the on-policy correction headaches that PPO would impose.

It is worth lingering on why each of the three choices is load-bearing, because the temptation when implementing MBPO is to skip the one that is annoying and the annoying one is usually the ensemble. A single deterministic dynamics network is much simpler to train and looks fine on held-out one-step error. It fails catastrophically in rollout because it has no notion of its own uncertainty — when the policy steers into unexplored state space, a single net does not get *less confident*, it just confidently extrapolates, and confident extrapolation is exactly the optimistic fantasy the policy will exploit. The ensemble's value is not better mean prediction; it is that disagreement between members *grows* in regions with little data, and that growing disagreement is a usable signal. By sampling a different member at each rollout step, you are effectively averaging over the model's own uncertainty, which blunts any single member's hallucination. The aleatoric piece (each net predicts a variance, not just a mean) matters separately: stochastic environments have irreducible noise, and a model that predicts a point estimate will systematically underestimate the spread of outcomes, leading the policy to be overconfident about controllability. You want both kinds of uncertainty represented, and the probabilistic ensemble is the cheapest thing that captures both.

The second choice — short branched rollouts — we have already justified from the compounding-error bound, but there is an implementation subtlety. The rollouts branch from states sampled out of the *real* buffer, which means the distribution of rollout start-states matches the distribution of states the real policy actually visits. This is not incidental. If you branched from arbitrary states — say, uniformly sampled from the whole state space — you would be training SAC on synthetic data from regions the policy never enters, wasting capacity and possibly destabilizing the value function with irrelevant targets. By anchoring every rollout to a real visited state, you keep the synthetic data distribution aligned with the on-policy distribution that matters. The third choice, using an off-policy learner, is what makes the whole mixing legal: SAC's replay-based off-policy updates do not care whether a transition came from the real environment or the model, so you can pour both into the same batch. PPO, being on-policy, would need importance-weighting corrections that get unstable fast when 95% of your data is synthetic, which is why nobody builds MBPO on top of PPO.

Here is the core MBPO loop in PyTorch-flavored pseudocode that maps directly to a real implementation (the Facebook Research `mbrl-lib` library implements exactly this):

```python
import torch
import numpy as np

def mbpo_train(env, dynamics_ensemble, sac_agent,
               real_buffer, model_buffer,
               rollout_horizon=5, model_rollouts_per_step=400,
               n_sac_updates=20, total_steps=200_000):
    state, _ = env.reset(seed=0)
    for step in range(total_steps):
        # 1. Collect ONE real transition from the current policy
        action = sac_agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        real_buffer.add(state, action, reward, next_state, terminated)
        state = next_state if not (terminated or truncated) else env.reset(seed=step)[0]

        # 2. Periodically retrain the dynamics ensemble on real data
        if step % 250 == 0 and len(real_buffer) > 1000:
            dynamics_ensemble.train(real_buffer, n_epochs=10)

        # 3. Generate many SHORT rollouts branched from real states
        if step % 250 == 0 and len(real_buffer) > 1000:
            start_states = real_buffer.sample_states(model_rollouts_per_step)
            s = start_states
            for h in range(rollout_horizon):          # H = 1..5 only
                a = sac_agent.act(s)
                member = np.random.randint(dynamics_ensemble.size)
                s_next, r = dynamics_ensemble.predict(s, a, member)
                model_buffer.add(s, a, r, s_next, done=False)
                s = s_next

        # 4. Update SAC mostly on synthetic data, anchored by real data
        for _ in range(n_sac_updates):
            batch = mix_batches(real_buffer, model_buffer, real_ratio=0.05)
            sac_agent.update(batch)
    return sac_agent
```

Notice the `real_ratio=0.05`: roughly 95% of each SAC update batch is synthetic. That is the data multiplication in action — a few thousand real transitions support hundreds of thousands of model-generated ones. The result reported in the paper, which I have reproduced approximately on HalfCheetah-v4, is near-SAC asymptotic performance at roughly 5x better sample efficiency. The ratio of SAC updates to real steps is the other dial that makes this work: notice that for every single real step we run `n_sac_updates=20` gradient updates on the policy. In ordinary SAC you do one update per step or so. MBPO can afford twenty because nineteen of them are mostly fed by synthetic data — the model is, in effect, lending the policy optimizer a vastly enlarged dataset to chew on between real interactions. This high update-to-data ratio is precisely what converts a small real-step budget into a large number of effective policy improvements.

A diagnostic you should always run when the model buffer is doing the heavy lifting: periodically measure the model's one-step prediction error on held-out *real* transitions, not on its own rollouts. If that error creeps up, your model is overfitting or the policy has wandered into a state region the model never saw, and your synthetic data is turning into fiction.

```python
@torch.no_grad()
def model_one_step_error(dynamics_ensemble, holdout_real_batch):
    s, a, _, s_next_true, _ = holdout_real_batch
    s_next_pred, _ = dynamics_ensemble.predict_mean(s, a)
    # normalized RMSE per state dimension
    err = torch.sqrt(((s_next_pred - s_next_true) ** 2).mean(dim=0))
    norm = s_next_true.std(dim=0) + 1e-6
    return (err / norm).mean().item()   # watch this number; rising = trouble
```

If that number drifts above roughly 0.1 normalized RMSE, shorten your rollout horizon or retrain the model more often before trusting it further. In practice I plot this number on the same dashboard as the episodic return, side by side, because the two together diagnose almost every MBPO failure. If return stalls while model error stays low, the bottleneck is the policy optimizer (try more SAC updates per step or a longer horizon). If return collapses while model error spikes, the model has lost the plot (shorten the horizon, retrain more often, or check whether the policy has discovered an exploit region). If both look fine but the real-environment performance lags the in-model performance, you are looking at the model-free-asymptote ceiling and it may be time to anneal toward a higher real-data ratio. The dashboard turns "MBPO is not working" — which is unactionable — into a specific knob to turn.

There is one more practical gotcha that bites everyone the first time. The model predicts a *reward* as well as a next state, and on many tasks the reward function is actually known analytically (you wrote it). If you know the true reward function, **use it** — compute reward from the predicted next state with your real reward function rather than letting the model predict reward, because reward-prediction error compounds into the value function just like state error does, and there is no reason to incur it when you have the ground truth. Only learn the reward model when the reward genuinely depends on unobserved environment internals you cannot recompute. This single change often improves MBPO stability more than any hyperparameter sweep, and it is the kind of thing the papers gloss over because they are benchmarking the general algorithm, not shipping your specific task.

## 5. When model-based RL dominates

The decision tree below collapses to two questions, and this section and the next answer them. The first question: *is a real environment step expensive?* When the answer is yes, model-based RL's sample efficiency is not a nice-to-have, it is the difference between a feasible project and an impossible one.

![Decision tree routing problems to a reinforcement learning family based on whether a fast simulator exists and whether observations are high-dimensional pixels.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-6.png)

**Physical robotics.** Every real step costs wall-clock time, electricity, mechanical wear, and usually a human supervisor. A million steps is days of operation and real hardware risk. This is the canonical MBRL domain — PETS, MBPO, and Dreamer were all benchmarked partly because robotics could not afford model-free sample counts. If you are learning on real hardware and cannot fully sim-to-real, you want model-based. The concrete published result to anchor on is Nagabandi et al. (2018), who learned neural-network dynamics models for legged and manipulation robots and used MPC to control them, reaching competent locomotion in a few minutes to tens of minutes of real robot experience — a regime where model-free methods would still be in their warm-up phase. The qualitative experience of working on real hardware is that you count steps the way a startup counts runway: every one is precious, and a method that gets 5x more learning per step is not a luxury, it is the only thing that fits in the budget.

**Biology and chemistry.** Optimizing a fermentation process, a reaction yield, or a protein design loop where each "step" is a wet-lab experiment taking hours or days and costing real reagents. Here the real-step budget might be a few hundred *total*. Model-free RL is simply inapplicable; you need a model (often a Gaussian process or a learned surrogate) and you plan against it. This shades into Bayesian optimization, which is model-based RL's close cousin for one-step-decision problems. The mental model to carry: when your total interaction budget is measured in hundreds rather than millions, you are so far to the left of any crossover that the question "model-based or model-free?" answers itself — there is no model-free method that converges in three hundred samples, so you build a surrogate model, quantify its uncertainty, and let it tell you which one experiment to run next. The entire field of experimental design is, in this light, just model-based decision-making under a brutal sample budget.

**Medical and clinical settings.** Adaptive trial design or treatment-policy learning where each interaction is a patient outcome you cannot rush and cannot run indefinitely. Beyond sample cost, you have a hard safety constraint, which pushes you toward model-based methods specifically because a model lets you *plan pessimistically* and screen actions for safety before executing them. The sample budget and the safety constraint reinforce each other here: you have very few samples *and* you cannot afford to explore freely to gather more, so you need every drop of efficiency the model offers and you need the model's uncertainty estimates to keep the policy inside the region the data supports. This is the domain where the two reasons to go model-based — sample efficiency and safe pessimistic planning — are both binding at once, which is why nearly all serious clinical RL work is model-based and offline.

**Slow-feedback recommenders.** A recommender whose reward (retention, subscription, long-term engagement) only resolves over days or weeks gets very few true reward signals per unit time. A learned model of user response can turn a trickle of real feedback into a usable training signal, and lets you do offline planning before exposing real users to a new policy. The subtlety in recommender systems is that the "expense" of a step is not compute — serving a recommendation is cheap — it is the *latency and risk of the reward*. If retention is your reward and retention is measured monthly, you get twelve reward observations per user per year, which is a sample budget so thin that model-free learning on the true long-horizon reward is hopeless. A learned user-response model lets you simulate the long-horizon consequences of a policy from the abundant short-horizon click data, converting a thin slow signal into a thick fast one — provided, and this is the eternal caveat, your user model is not so wrong that the optimizer learns to game it.

The common thread is a low *real-step budget* relative to the crossover point. In all of these, you are far to the left of the SAC-catches-up region, in the zone where MBRL's five-to-tenfold efficiency advantage is decisive and the asymptotic gap (model-based topping out slightly lower) is irrelevant because you will never have enough samples to reach the asymptote anyway. Notice the recurring shape of the argument: the asymptote is the model-free family's great strength, and in every one of these domains the asymptote *does not matter* because you cannot afford to reach it. A method's advantage is only real if you operate in the regime where it applies, and a low sample budget puts you squarely in the regime where the model-based advantage applies and the model-free advantage is moot.

## 6. When model-free RL dominates

Now the same decision tree's other branch. The first question again: is a real step cheap? When you have a fast, accurate simulator — a physics engine, a game engine, a trading backtester running on historical data — real steps are nearly free, and the entire premise of model-based RL evaporates.

**Fast simulators.** MuJoCo in simulation, Atari via the Arcade Learning Environment, Isaac Gym running thousands of parallel robots on a GPU. When you can take a billion environment steps overnight, sample efficiency stops mattering and the only thing you care about is the asymptote and the engineering simplicity. SAC, PPO, and DQN are battle-tested, have excellent open implementations in Stable-Baselines3, and reach higher final performance without the model-bias ceiling.

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Fast simulator: real steps are cheap, so just throw experience at SAC.
vec_env = make_vec_env("HalfCheetah-v4", n_envs=8)
model = SAC("MlpPolicy", vec_env, learning_rate=3e-4,
            buffer_size=1_000_000, batch_size=256, gamma=0.99,
            ent_coef="auto", verbose=1)
model.learn(total_timesteps=1_000_000)
mean_r, std_r = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"mean return {mean_r:.0f} +/- {std_r:.0f}")  # ~ 4800 +/- a few hundred
```

The engineering-simplicity point is underrated and worth dwelling on, because it is a real cost that does not show up in any benchmark plot. A model-free pipeline has one learned object — the policy (and its value function). A model-based pipeline has at least two — the dynamics model and the policy — and they interact in ways that create failure modes neither has alone. The model can be undertrained relative to the policy or overtrained relative to the data; the rollout horizon can be wrong for the current model error; the real-data ratio can starve or flood the policy optimizer; the ensemble can collapse to agreement and stop providing uncertainty signal. Each of these is a knob, each knob is a way to be subtly broken, and debugging a system with two coupled learned components is more than twice as hard as debugging one. When samples are free, you are paying this complexity tax for an efficiency you do not need. Take the simple thing. The cleanest engineering decision in RL is to use model-free whenever the sample budget lets you, precisely because it has fewer moving parts that can silently misbehave.

**Complex visual environments where the model is hard to learn.** Learning an accurate pixel-level dynamics model of a cluttered 3D scene is genuinely harder than learning a good policy for it. When the observation space is high-dimensional and visually rich, the model's errors are large and the compounding problem bites hard. Model-free methods sidestep this entirely — they never have to predict the next frame, only the value of acting. (The exception, covered below, is latent-space world models like Dreamer that learn dynamics in a compact latent rather than in pixel space.) The general principle: model-based RL is attractive exactly when the dynamics are *easier to model than the policy is to learn*, and that condition fails in rich visual environments unless you have the latent-space machinery to make the modeling tractable. A naive pixel-prediction model in a cluttered scene spends all its capacity on photorealism and none on the few decision-relevant variables, which is the worst possible allocation — we will return to this in section 8 as the deepest insight in the field.

**High-dimensional tasks where asymptotic performance is the goal.** If your objective is to set a benchmark record and you have the compute, model-free methods win the asymptote. OpenAI Five (Dota 2) and AlphaStar (StarCraft II) used massively parallel model-free training precisely because the simulator was fast and the goal was peak performance, not sample frugality. These projects burned through quantities of simulated experience — the equivalent of thousands of years of game time — that would be unthinkable on any real system, and that is the entire point: when the simulator is fast and the budget is effectively unlimited, you spend samples lavishly to buy asymptote, and the model-free family is the one with no ceiling to spend toward.

The comparison table below crystallizes the on-policy / off-policy / model-based distinctions that drive these choices:

| Family | Example | Sample eff. | Asymptote | Reuses data? | Best when |
| --- | --- | --- | --- | --- | --- |
| On-policy MF | PPO, A2C | Low | High | No (on-policy) | Fast sim, stable training wanted |
| Off-policy MF | SAC, TD3, DQN | Medium | High | Yes (replay) | Fast sim, want efficiency + asymptote |
| Dyna-style MB | Dyna-Q, MBPO | High | Near-MF | Yes (model + replay) | Expensive real steps, low-dim state |
| Planning MB | PETS, MPC | Very high | Medium | Yes (replan) | Very few steps, short horizon OK |
| Latent world model | Dreamer, PlaNet | High | High | Yes (imagination) | Expensive steps, visual observations |
| Value-equivalent MB | MuZero | High | Very high | Yes (search) | Discrete game tree, planning helps |

Read down the "reuses data?" column and you can see the entire sample-efficiency story compress into one idea: every family that reuses data well is sample-efficient, and on-policy PPO — which by construction must throw away data after one gradient step — sits at the bottom of the efficiency column. Off-policy replay (SAC, DQN) reuses each transition dozens of times and climbs the efficiency ladder; model-based methods reuse each transition *hundreds* of times by distilling it into a model and re-querying that model, and they top the ladder. Sample efficiency is, almost entirely, a question of how many times you extract value from each real interaction. That reframing — efficiency equals reuse — is the cleanest lens on the whole table.

## 7. The three flavors of model-based RL

"Model-based" is not one thing. There are three distinct families, and conflating them is the most common source of confusion. The figure below decomposes what a learned model even contains — a transition component, a reward component, and (for visual tasks) an observation component — all feeding policy improvement.

![Graph showing a learned world model factoring into transition, reward, and observation components that all feed policy improvement.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-7.png)

**(a) Explicit model plus planning.** You learn $\hat{P}$ and $\hat{r}$ in the original state space, then plan with them at decision time. Dyna-Q (Sutton, 1990) interleaves real experience with model-generated Q-updates. PETS (Chua et al., 2018) learns a probabilistic ensemble and plans with Model Predictive Control (MPC): at each real step, it samples thousands of candidate action sequences, rolls each through the model, and executes the first action of the best sequence — then replans next step. There is no learned policy at all; the "policy" is the planner. This is maximally sample-efficient and maximally interpretable, but planning is expensive at decision time and the short horizon caps the asymptote.

```python
import numpy as np

def pets_mpc_action(state, dynamics_ensemble, reward_fn,
                    horizon=15, n_candidates=1000, action_dim=6):
    # Random-shooting MPC: sample action sequences, score by model rollout.
    seqs = np.random.uniform(-1, 1, size=(n_candidates, horizon, action_dim))
    total_reward = np.zeros(n_candidates)
    s = np.tile(state, (n_candidates, 1))
    for t in range(horizon):
        a = seqs[:, t, :]
        member = np.random.randint(dynamics_ensemble.size)
        s, _ = dynamics_ensemble.predict(s, a, member)
        total_reward += reward_fn(s, a)
    best = np.argmax(total_reward)
    return seqs[best, 0, :]   # execute only the first action, then replan
```

The code above uses the simplest planner, random shooting, which just samples action sequences uniformly and keeps the best. In practice PETS uses the Cross-Entropy Method (CEM), which is smarter: it samples sequences, keeps the top-scoring fraction (the "elites"), refits a Gaussian to those elites, and resamples from the refined Gaussian, iterating a handful of times so the search concentrates around promising action sequences instead of blindly covering the whole space. The distinction matters for high-dimensional action spaces, where random shooting needs exponentially many samples to stumble onto a good sequence and CEM converges in a few iterations. Here is the CEM refinement sketched out:

```python
def cem_plan(state, dynamics_ensemble, reward_fn, horizon=15,
             action_dim=6, n_candidates=500, n_elites=50, n_iters=5):
    mu = np.zeros((horizon, action_dim))
    sigma = np.ones((horizon, action_dim))
    for _ in range(n_iters):
        seqs = mu + sigma * np.random.randn(n_candidates, horizon, action_dim)
        seqs = np.clip(seqs, -1, 1)
        scores = np.zeros(n_candidates)
        s = np.tile(state, (n_candidates, 1))
        for t in range(horizon):
            member = np.random.randint(dynamics_ensemble.size)
            s, _ = dynamics_ensemble.predict(s, seqs[:, t, :], member)
            scores += reward_fn(s, seqs[:, t, :])
        elites = seqs[np.argsort(scores)[-n_elites:]]   # keep the best
        mu, sigma = elites.mean(axis=0), elites.std(axis=0)  # refit, repeat
    return mu[0]   # execute first action of the refined mean sequence
```

The thing to internalize about the planning family is the cost structure: there is *no training-time policy optimization at all*, which is why it is so sample-efficient and so simple to reason about, but there is a *heavy decision-time cost* because you re-run this whole CEM search at every single step. On a robot taking ten decisions a second, you have a hundred milliseconds to run hundreds of model rollouts, which is feasible on a GPU but rules planning out for any high-frequency control loop. The trade is clear: PETS moves the computation from training time to inference time. If you can afford to think for a moment before each action, planning is wonderful. If you need a reflex, you need a learned policy, which points you at the next two families.

**(b) Latent-space models plus imagination.** Instead of modeling raw observations, you learn a compact latent state and model the dynamics *in the latent space*. PlaNet and Dreamer (Hafner et al., 2019, 2020, 2023) encode pixels into a latent, learn transition and reward models there, and then train an actor-critic purely on "imagined" latent rollouts — no decoding back to pixels needed for policy learning. This is what makes model-based RL viable on visual tasks: the model never has to predict pixels accurately, only the latent dynamics that matter for value. Dreamer-v3 reaches strong asymptotic performance across a huge range of domains, including collecting diamonds in Minecraft from pixels. The architectural key is the recurrent state-space model: Dreamer maintains a latent state that combines a deterministic recurrent component (for long-term memory) and a stochastic component (for capturing uncertainty in the dynamics), and it trains the encoder, the latent transition, the reward predictor, and an optional decoder jointly. Once the latent dynamics are learned, the actor and critic train entirely inside the latent imagination, which is fast because a latent rollout is just a sequence of cheap recurrent-cell forward passes with no pixel rendering — you can imagine thousands of trajectories per gradient step. The reason this beats pixel-space modeling is that the latent is *small* and *learned to be predictable*, so the compounding-error problem is far gentler than it would be predicting hundreds of thousands of raw pixels forward in time.

**(c) Model plus value or policy distillation.** Here the model exists only to generate data for a model-free learner; you never plan with it explicitly at decision time. MBPO (section 4) is the archetype: the model generates short rollouts, SAC learns the policy. STEVE (Buckman et al., 2018) goes further and uses the model to construct lower-variance value targets, weighting model-based and model-free estimates by their uncertainty. This family inherits the model-free asymptote (because a model-free learner is doing the policy optimization) while gaining model-based sample efficiency. The defining feature is that the model is a *means*, never an *end* — at deployment you ship the distilled model-free policy and the model is discarded entirely, which means deployment is exactly as cheap and reactive as plain SAC. You pay the model's cost only during training, and you reap a fast reflex policy at the end. For most production continuous-control problems where you need a real-time controller but cannot afford millions of real samples to train it, this distillation family is the sweet spot, and MBPO is its cleanest representative.

These three are not mutually exclusive — MuZero, which we treat as a case study, blends explicit planning (Monte Carlo Tree Search) with value-equivalent latent modeling. But knowing which flavor a method belongs to tells you immediately what its failure mode is: planning methods fail on long horizons, latent methods fail when the latent loses task-relevant information, and distillation methods fail when the model buffer poisons the model-free learner. Memorize the failure modes alongside the methods, because in production you will diagnose a system far more often by recognizing its characteristic failure than by recalling its architecture. A planning method that suddenly degrades is probably hitting its horizon limit; a latent method that plateaus has probably squeezed task-relevant signal out of its latent; a distillation method that diverges has a model buffer full of fiction. The flavor tells you where to look first.

## 8. A model does not need to be accurate — it needs to be useful

The deepest insight in modern model-based RL is that a *perfect* model is the wrong target. What you actually need is a model that is accurate *about the things that affect the policy's decisions*, and that is a far weaker requirement.

Consider a self-driving toy problem. A pixel-perfect generative model of the road — every reflection, every leaf, every cloud — is enormously hard to learn and almost entirely wasted, because the policy's decisions depend on a handful of factors: the lane geometry, the other cars' positions and velocities, the traffic light state. A model that predicts those well and gets the clouds wrong is perfectly useful. A model that predicts pixels beautifully but blurs the position of the car ahead is useless. Accuracy in the wrong subspace is worthless; accuracy in the decision-relevant subspace is everything.

This reframing dissolves a paradox that confuses many newcomers: why does pixel-space model-based RL struggle while latent-space model-based RL thrives, when the latent model is "less accurate" about the raw observation? The answer is that the latent model is *more accurate about the thing that matters*. A pixel-prediction model is graded on reconstruction loss, which is dominated by high-frequency texture and lighting — the clouds and leaves — so its gradients push it to spend capacity there, starving the decision-relevant variables. A latent model trained with a reward and value objective is graded, directly or indirectly, on whether its predictions support good decisions, so its capacity flows to the decision-relevant subspace by construction. "Less accurate on pixels, more accurate on what matters" is not a contradiction; it is the entire reason latent world models work. The lesson generalizes: a model's accuracy should be measured in the currency of the downstream decision, not in the currency of raw reconstruction, and an objective that confuses the two will allocate the model's limited capacity to the wrong places.

MuZero (Schrittwieser et al., 2020) takes this to its logical conclusion with **value equivalence**. MuZero learns a latent dynamics model, but it never trains the model to reconstruct the observation at all. Instead, the model is trained so that planning with it produces the *same value predictions and policy as the real environment would*. The latent state has no required meaning — it is not the board position, it is whatever internal representation lets the model predict reward, value, and policy correctly. MuZero matched AlphaZero's superhuman play in Go, chess, and shogi *and* set records on Atari, all without ever being told the rules of the game. The model is optimized for value prediction, not state reconstruction, and that is precisely why it works where pixel-reconstruction models would drown in irrelevant detail.

There is a spectrum here worth naming, because the right point on it depends on your task. At one end sits pure reconstruction (predict the next observation faithfully); at the other sits pure value equivalence (predict only reward and value, ignore everything else). Dreamer lives toward the reconstruction end — it does decode pixels, which gives it a rich, dense training signal at every step (every pixel is a supervision target), making the model easy to train, at the cost of spending capacity on irrelevant detail. MuZero lives at the pure value-equivalence end — no reconstruction at all, maximally focused on the decision, but trained from a sparser signal (only reward and value), which makes it harder to train and more reliant on lots of planning to generate that signal. TD-MPC2 (Hansen et al., 2024) deliberately sits in the middle, combining a value-equivalent latent objective with enough auxiliary structure to get a dense signal, and it has become a strong default for continuous control precisely because it captures both benefits. The practical reading: if your task gives a rich observation but a sparse reward, lean toward reconstruction for the training signal (Dreamer); if your task has a clear reward and the observation is full of irrelevant detail, lean toward value equivalence (MuZero); if you want one knob that works broadly, the hybrid middle (TD-MPC2) is the safe bet.

The practical lesson for your own systems: when you measure your model, measure it on the quantity that matters — *does planning or rolling out with this model produce good policy decisions?* — not on raw next-state reconstruction error. A model with high reconstruction error can still be excellent for control, and a model with low reconstruction error can still be useless if it is wrong about the few variables the policy depends on. The "good enough" threshold for a model is defined by the policy, not by a reconstruction loss.

#### Worked example: value-equivalence beats reconstruction

Imagine two models of the same task. Model A achieves 0.01 next-state reconstruction RMSE but systematically underestimates the velocity of approaching obstacles by 15%. Model B achieves a worse 0.08 reconstruction RMSE but nails obstacle velocity to within 2%. Train a planner against each. The policy from Model A will brake too late — it plans against obstacles that seem slower than they are — and will crash in the real environment despite the model's gorgeous reconstruction numbers. The policy from Model B brakes correctly. If you had selected models by reconstruction loss you would have shipped the worse one. This is the entire argument for value-equivalent training: optimize the model for the downstream decision, and let reconstruction fall where it may. The deeper point the example dramatizes is that reconstruction error *averages over the state dimensions*, so a model can drive its average error to near zero by nailing the many easy, high-variance, decision-irrelevant dimensions (texture, lighting, background) while quietly being wrong about the one low-variance, decision-critical dimension (obstacle velocity). The average lies. Always evaluate on the downstream policy, where the lie cannot hide.

## 9. Hybrid approaches and the tooling

In practice the line between the families is blurry, and the most effective systems are hybrids that take sample efficiency from the model side and asymptotic performance from the model-free side. The figure below shows the inner loop of the canonical hybrid, MBPO, as a pipeline: collect one real transition, add it to the real buffer, retrain the dynamics model, roll out short horizons from real states, add those to the model buffer, and run several SAC updates on the combined data.

![Pipeline of one MBPO update step from collecting a real transition through training the dynamics model to a SAC update on combined real and synthetic buffers.](/imgs/blogs/model-based-vs-model-free-when-to-use-which-8.png)

**MBPO** (covered above) is the workhorse hybrid: model for short rollouts, SAC for the policy. **STEVE** uses the model to build uncertainty-weighted value targets, interpolating between a pure model-free TD target and a model-based multi-step target based on which the ensemble trusts more — so when the model is unreliable it gracefully degrades to model-free. **MBMF** (model-based model-free) uses model-based MPC to initialize a policy, then fine-tunes it model-free, getting a fast start from the model and a high asymptote from the model-free phase. The common design pattern across all three is *graceful degradation toward model-free*: each is constructed so that when the model is bad, the system behaves like a model-free method rather than chasing the model's fantasies. STEVE downweights the model term when the ensemble disagrees; MBMF hands off to model-free fine-tuning once the model has given its fast start; MBPO shrinks its rollout horizon (and you can drop it to zero, recovering pure SAC) when model error climbs. The lesson for designing your own hybrid: build it so the failure mode of the model is "we fall back to a slower but correct method," never "we confidently optimize against a lie." A hybrid that cannot gracefully fall back is more dangerous than either pure family.

For practical implementation, the **`mbrl-lib`** library from Facebook Research (Pineda et al., 2021) is the reference codebase. It implements PETS and MBPO with the probabilistic-ensemble dynamics model, the short-rollout machinery, and the SAC integration, all configurable. A minimal MBPO configuration in its style:

```yaml
# mbrl-lib MBPO config (abridged)
algorithm:
  name: mbpo
  freq_train_model: 250        # retrain dynamics every 250 real steps
  num_sac_updates_per_step: 20
  sac_samples_action: true
  real_data_ratio: 0.05        # 5% real, 95% synthetic per SAC batch
dynamics_model:
  ensemble_size: 7             # probabilistic ensemble
  hid_size: 200
  num_layers: 4
overrides:
  rollout_schedule: [20, 150, 1, 5]  # grow horizon from 1 to 5 over training
  num_steps: 200000            # real environment steps
```

Note the `rollout_schedule`: MBPO often *grows* the rollout horizon over training, starting at $H=1$ when the model is poor and extending toward $H=5$ as the model improves. That schedule is a direct, practical application of the compounding-error analysis — you only lengthen rollouts as fast as the model earns your trust. If you build a model-based system from scratch, steal this idea before anything else. The four numbers in `rollout_schedule` read as `[start_epoch, end_epoch, start_horizon, end_horizon]`: linearly interpolate the horizon from 1 to 5 between training epochs 20 and 150. That is the $H^*$-tracking schedule from section 3 made operational — it is the library encoding the theory that a better model earns a longer horizon.

It is worth seeing how little code it takes to drive a full MBPO experiment once you lean on the library, because the perceived complexity of model-based RL is mostly the dynamics-model and rollout machinery that `mbrl-lib` already provides:

```python
import omegaconf
import mbrl.algorithms.mbpo as mbpo
import mbrl.util.env
import gymnasium as gym

# Load the abridged config above (extended with sac/ and dynamics_model/ blocks)
cfg = omegaconf.OmegaConf.load("mbpo_halfcheetah.yaml")
env = gym.make("HalfCheetah-v4")
term_fn = mbrl.util.env.EnvHandler.maybe_get_termination_fn("HalfCheetah-v4")

# mbrl-lib wires up the probabilistic ensemble, the model/real buffers,
# the short-rollout generator, and the SAC agent from the config.
mbpo.train(env, term_fn, reward_fn=None, cfg=cfg)
# logs episodic return AND held-out model error per epoch — watch both.
```

The termination function passed in is a small but important detail unique to model-based RL: when you roll out the model, you need to know when a synthetic trajectory should *end* (the cheetah flipped over, the pole fell), and the model does not predict the done flag reliably, so you supply the known termination logic analytically — the same spirit as supplying the known reward function from section 4. Anything about the task you know for certain, give to the system directly rather than asking it to learn; learning is for the dynamics you genuinely do not know.

A final piece of model-uncertainty tooling that pays for itself: before you ever trust the model for rollouts, sanity-check that the ensemble's *disagreement* actually tracks where data is scarce. A healthy ensemble disagrees a lot in unexplored regions and agrees in well-covered ones; a collapsed ensemble (all members trained to near-identical functions) disagrees nowhere and gives you no uncertainty signal, silently turning your "safe" model-based system into an overconfident one.

```python
@torch.no_grad()
def ensemble_disagreement(dynamics_ensemble, states, actions):
    # Each member predicts the next state; disagreement = epistemic uncertainty.
    preds = torch.stack([
        dynamics_ensemble.predict_mean(states, actions, m)[0]
        for m in range(dynamics_ensemble.size)
    ])                                          # (ensemble, batch, state_dim)
    return preds.std(dim=0).mean(dim=-1)        # per-state uncertainty scalar
# Sanity check: this should be LARGE for random/out-of-distribution states
# and SMALL for states drawn from the replay buffer. If it is flat, your
# ensemble has collapsed and provides no uncertainty signal.
```

## 10. The simulator shortcut

There is a tempting trap in this whole discussion, and it is worth naming explicitly because I have watched teams fall into it. The trap is reaching for model-based RL when you already have a *fast, accurate simulator*. If you have a physics engine or a game engine that runs faster than real time and is accurate enough to train against, then **you already have a model** — a hand-built, exact one — and learning a worse, approximate model on top of it is pure self-harm.

The right move when you have a fast simulator is massively parallel model-free RL. Spin up thousands of simulated environments, generate billions of cheap transitions, and let SAC or PPO chew through them. The model-based machinery — ensembles, short rollouts, uncertainty weighting — exists to manufacture cheap data when real data is scarce. When the simulator already manufactures cheap data perfectly, all that machinery is overhead that lowers your asymptote (because the learned model is worse than the simulator) while solving a problem you do not have.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

# Fast accurate simulator + massive parallelism = model-free wins outright.
def make_env(seed):
    def _f():
        env = gym.make("HalfCheetah-v4")
        env.reset(seed=seed)
        return env
    return _f

vec = SubprocVecEnv([make_env(i) for i in range(64)])  # 64 parallel sims
model = PPO("MlpPolicy", vec, n_steps=2048, batch_size=2048,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0, verbose=1)
model.learn(total_timesteps=10_000_000)  # cheap when sims are fast
```

There is a subtle and important wall-clock argument that often surprises people, and it is the real reason massively parallel model-free can beat model-based even on a *step-count* metric where model-based looks better. Model-based RL is sample-efficient per *real environment step*, but it is *serial* in a way that hurts wall-clock time: the loop is collect-one-step, retrain-model, generate-rollouts, update-policy, repeat, and the model training and rollout generation sit on the critical path between real steps. Massively parallel model-free, by contrast, collects 64 (or 4096, on a GPU sim like Isaac Gym) transitions simultaneously per wall-clock tick and runs the policy update on all of them at once. When you benchmark *total compute cost* and *wall-clock time to a target return* rather than real-step count, a fast simulator with thousands of parallel environments frequently reaches the target sooner than MBPO does, even though MBPO used fewer "steps," because MBPO's steps were serial and expensive in GPU-coordination overhead while the parallel model-free steps were free and concurrent. The metric you optimize determines the winner. If real steps are the scarce resource, count real steps and model-based wins. If wall-clock time and GPU-hours are the scarce resource (which they are, when the sim is fast), count those, and parallel model-free often wins. Never compare two methods on a metric that is not the one that actually constrains your project.

```python
# Pseudocode for the comparison you should actually run before committing:
def compare_total_cost(target_return):
    # Model-free with a fast sim: cheap parallel steps, serial-free.
    mf_walltime, mf_gpu_hours = train_parallel_ppo(n_envs=64,
                                                   until=target_return)
    # Model-based: fewer real steps, but serial model+rollout overhead.
    mb_walltime, mb_gpu_hours = train_mbpo(until=target_return)
    # Decide on the resource that is actually scarce for YOU, not on steps.
    print(f"MF: {mf_walltime:.1f}h wall, {mf_gpu_hours:.0f} GPU-h")
    print(f"MB: {mb_walltime:.1f}h wall, {mb_gpu_hours:.0f} GPU-h")
    # If the sim is fast, MF usually wins wall-clock despite "more steps".
```

The decision tree's left branch — "fast sim? steps cheap" routing to model-free — encodes exactly this. The one nuance: even with a fast simulator, if your task has a clean *game-tree* structure where planning genuinely helps (Go, chess, certain combinatorial problems), MuZero-style model-based planning can beat pure model-free because search at decision time finds better actions than a reactive policy. So the refined rule is: fast simulator and no game-tree structure points to model-free; fast simulator with strong planning structure points to MuZero-style search. The deeper principle behind that nuance is that planning buys you *lookahead at decision time*, and lookahead is worth a great deal precisely in domains where a small mistake is catastrophic and the consequences are far in the future — exactly the structure of board games and combinatorial optimization. In smooth continuous control, where a reactive policy can correct a small error on the next step, lookahead buys much less, which is why model-free reactive policies dominate MuJoCo while planning dominates Go.

## 11. Deployment: uncertainty and safe, offline RL

Sample efficiency is the headline reason to go model-based, but there is a second, quieter reason that matters enormously in production: **a model lets you reason about uncertainty and plan safely before you act.**

A model-free policy is a reflex — it maps states to actions and gives you no native way to ask "how confident am I that this action is safe?" A model-based agent with a probabilistic dynamics model can roll out a candidate action, see the *distribution* of outcomes the ensemble predicts, and refuse actions whose worst-case predicted outcome violates a safety constraint. This is the basis of safe model-based RL: plan against the *pessimistic* (lower-confidence-bound) outcome rather than the mean, so the agent only commits to actions that look good even under the model's uncertainty.

The same machinery powers offline RL, where you must learn entirely from a fixed dataset with no further interaction — common in healthcare, finance, and any setting where exploration is dangerous or forbidden. The danger in offline RL is distributional shift: the policy wants to take actions the dataset never demonstrates, and the value estimates for those actions are wild extrapolations. Model-based offline methods like MOPO and MOReL (both 2020) handle this by building a model, *penalizing reward in proportion to model uncertainty*, and planning against that pessimistic model. Where the model is confident (well-covered by data), the agent plans freely; where the model is uncertain (out of distribution), the uncertainty penalty steers the agent back to the safe, data-supported region.

```python
@torch.no_grad()
def pessimistic_reward(dynamics_ensemble, s, a, lam=1.0):
    # MOPO-style: subtract a penalty proportional to ensemble disagreement.
    preds = [dynamics_ensemble.predict_mean(s, a, m)[0]
             for m in range(dynamics_ensemble.size)]
    preds = torch.stack(preds)                 # (ensemble, batch, state_dim)
    disagreement = preds.std(dim=0).max(dim=-1).values  # epistemic signal
    base_reward = dynamics_ensemble.reward(s, a)
    return base_reward - lam * disagreement    # penalize the unknown
```

That `lam * disagreement` term is the whole trick: it converts the ensemble's "I don't know" into a reward penalty, so the optimizer learns to stay where the model is trustworthy. There is no clean model-free equivalent — you need the model to know what it does not know. For safety-critical deployment, this is often the decisive reason to choose model-based even when sample efficiency alone would not require it.

Worth naming explicitly, though, is when uncertainty estimates *fail*, because a practitioner who trusts them blindly will get burned. Ensemble disagreement is a good epistemic-uncertainty signal only when the ensemble members are genuinely diverse and the test states are not too far out of distribution. Two failure modes recur. First, *ensemble collapse*: if the members are trained too similarly — same initialization scheme, same data ordering, insufficient bootstrapping — they converge to nearly the same function and agree everywhere, including in regions where they are all confidently wrong together. Disagreement then reads near zero exactly where you most need a warning. Second, *far-out-of-distribution overconfidence*: neural networks do not reliably become uncertain arbitrarily far from their training data; they can extrapolate to a wrong-but-confident answer that all members happen to share, again collapsing the disagreement signal precisely where it should spike. The defenses are to maximize ensemble diversity (different initializations, different bootstrap subsets, sometimes different architectures), to validate that disagreement actually rises on held-out out-of-distribution probes before trusting it in deployment (the `ensemble_disagreement` sanity check from section 9), and to never treat the uncertainty penalty as a hard safety guarantee — it is a useful pressure toward the data-supported region, not a proof of safety. In genuinely safety-critical settings, layer a hard, hand-specified constraint check on top of the learned uncertainty penalty; do not let a learned signal be the only thing standing between your agent and a dangerous action.

## 12. The five-step decision framework

Here is the framework I actually use, distilled to five questions you answer in order. Stop at the first one that gives a decisive answer.

**Step 1 — How expensive is a real environment step?** If a step is cheap (fast simulator, historical-data backtest) and you can take millions to billions of them, go model-free and skip to Step 5 to pick the specific algorithm. If a step is expensive (real robot, wet lab, live users, clinical outcome), sample efficiency is your binding constraint — continue to Step 2. Remember that "expensive" is multi-dimensional: wall-clock, money, hardware wear, human supervision, and risk. If any one of those is binding, treat the step as expensive even if the others are cheap.

**Step 2 — Do you have a fast, accurate simulator?** If yes, you already have a model; do not learn a worse one. Use parallel model-free RL against the simulator. If no usable simulator exists and real steps are expensive, you are in genuine model-based territory — continue to Step 3. The special case of this branch is offline RL: if you have a *fixed dataset* and no interaction at all, you are forced into model-based offline methods (MOPO, MOReL) with uncertainty penalties, because there is no environment to query and the model is the only way to reason about actions the dataset does not demonstrate.

**Step 3 — Are observations high-dimensional pixels or low-dimensional state?** If pixels or rich visual input, a latent-space world model is your tool: **Dreamer** (or PlaNet for simpler cases). If low-dimensional proprioceptive state, an explicit-model method works well — continue to Step 4. The reason this question comes before the planning-vs-policy question is that observation type determines whether you can model dynamics in the raw space at all; pixels force you into a latent, and once you are committed to a latent world model, Dreamer's design handles the policy-learning question for you via imagined actor-critic rollouts.

**Step 4 — Do you need a learned policy or is planning at decision time acceptable?** If decision-time planning is affordable and the horizon is short, **PETS** with MPC (CEM planner) is simple and extremely sample-efficient. If you need a fast reactive policy at deployment or want a higher asymptote, **MBPO** (model for short rollouts, SAC for the policy) is the robust default. The deciding factor is your inference-time budget: PETS spends compute at every decision to plan, MBPO spends compute only during training and ships a cheap reflex. A high-frequency control loop cannot afford PETS; a system that can pause to think before each action can.

**Step 5 — Pick the model-free algorithm (if you landed here).** Continuous actions, want efficiency and high asymptote: **SAC**. Continuous actions, want stability and easy tuning: **PPO**. Discrete actions, image input: **DQN** and its modern variants (Rainbow). Continuous, want a deterministic-policy off-policy method: **TD3**. Game tree with planning structure even in a fast sim: **MuZero**.

Worked through some examples: A high-frequency trading agent trained on a fast historical backtester (cheap steps) lands at model-free SAC or PPO at Step 1. A surgical robot learning on real hardware with no good simulator and joint-angle state lands at MBPO at Step 4. An agent learning to play a new video game from pixels with a fast emulator lands at model-free DQN at Step 1 — unless the game has deep planning structure, in which case MuZero. A drug-dosing policy learned from a fixed clinical dataset with no interaction allowed lands in offline model-based territory (MOPO) — a branch of Step 2 where the "simulator" is a learned, uncertainty-penalized model. A warehouse robot arm doing pick-and-place on real hardware, with a slow but usable physics simulator available, lands at Step 2's "you have a simulator" branch — train parallel SAC in the sim and sim-to-real transfer, rather than learning a model on the real arm, because the existing simulator is already a better model than you would learn. Notice how the same domain (robotics) routes to completely different answers depending on whether a usable simulator exists; the family choice is a property of your *situation*, not your *domain*.

## Case studies

**Atari DQN (Mnih et al., 2015).** The landmark model-free result: a single DQN architecture reached human-level or above on 29 of 49 Atari games, learning from pixels and reward alone. It needed on the order of 200 million frames per game — utterly fine, because the Arcade Learning Environment is a fast simulator. This is the model-free-on-cheap-steps regime in its purest form; nobody would learn Atari model-based for the *efficiency*, because the samples are free. The interesting twist is that model-based methods like MuZero and Dreamer-v3 later *did* tackle Atari and beat DQN's sample efficiency dramatically (reaching strong play in a few million frames instead of 200 million) — not because Atari needs sample efficiency, but because Atari became the benchmark where model-based methods proved they could match model-free asymptote while crushing it on samples. The lesson: a benchmark's regime (cheap steps) tells you which family is *natural* for it, but researchers will still push the other family there to prove a point.

**MBPO on MuJoCo (Janner et al., 2019).** The reference model-based result for continuous control. MBPO matched SAC's asymptotic performance on HalfCheetah, Hopper, Walker2d, and Ant while using roughly 5x fewer real environment steps, by combining a probabilistic ensemble model with short branched rollouts and SAC. The headline demonstration that you can have model-based sample efficiency without sacrificing the model-free asymptote. The result that matters operationally is not just the 5x — it is that MBPO achieves it *robustly across four different environments with the same recipe*, which is what made it a default rather than a one-environment trick. A method that needs per-environment retuning is a research result; a method that works across a benchmark suite with one config is a tool you can reach for.

**Dreamer-v3 (Hafner et al., 2023).** A single latent-world-model agent with fixed hyperparameters reached strong performance across more than 150 tasks spanning continuous control, Atari, and — famously — collecting diamonds in Minecraft from pixels with no human data, all far more sample-efficiently than model-free baselines on the visual tasks. The case for latent-space world models on high-dimensional observations. The "fixed hyperparameters across 150+ tasks" claim is the genuinely remarkable part: it means the latent-world-model approach had matured from a finicky research artifact into something that generalizes without per-task babysitting, which is the threshold a method must cross before it belongs in a practitioner's default toolbox. Minecraft diamonds specifically is a deep-exploration, long-horizon, sparse-reward problem from raw pixels — about the hardest combination there is — and a world model solving it from scratch is strong evidence that learning the dynamics in a compact latent is the right move when observations are visual and the horizon is long.

**MuZero (Schrittwieser et al., 2020).** Matched AlphaZero's superhuman play in Go, chess, and shogi and set state-of-the-art on Atari, all without being given environment rules, by learning a value-equivalent latent model and planning with MCTS. The proof that a model optimized for value prediction rather than reconstruction can support superhuman planning. The under-appreciated detail is that MuZero learns the model *and* the rules of the game simultaneously from interaction — AlphaZero was handed a perfect simulator (the rules of chess), and MuZero matched it while having to *learn* the dynamics. That it lost nothing by learning the model rather than being given it is the strongest possible evidence for value equivalence: a model trained only to predict the decision-relevant quantities (reward, value, policy) was as good for planning as the exact rules, because the parts of the rules that did not affect the decision were never worth modeling in the first place.

**Neural dynamics for robots (Nagabandi et al., 2018).** Often forgotten next to the flashier results, this is the cleanest demonstration of model-based RL doing what it is for: learning competent locomotion and manipulation on real and simulated robots from a few minutes to tens of minutes of experience, using a learned neural dynamics model with MPC, then optionally distilling into a model-free policy for speed. It is the template for the robotics decision in section 5 — when real steps cost hardware time and you have no perfect simulator, a learned dynamics model plus planning gets you to competence in a sample budget model-free methods cannot touch, and you can hand off to a model-free policy afterward if you need a fast reflex at deployment.

## When to use this (and when not to)

Be decisive. **Use model-free RL** whenever real steps are cheap — a fast simulator, a backtester, a game emulator — because its simplicity and high asymptote win and you have no reason to pay the model-bias tax. SAC for continuous control where you want efficiency, PPO where you want stability, DQN for discrete pixel tasks.

**Use model-based RL** when real steps are expensive and you have no fast simulator: physical robots, wet-lab science, clinical or recommender settings with slow, costly feedback. Within model-based, use Dreamer for pixels, PETS for low-dimensional short-horizon problems where decision-time planning is affordable, and MBPO when you want a reactive policy with a high asymptote.

**Do not learn a model when you already have a simulator** — that is the most common and most wasteful mistake. **Do not use long model rollouts** — compounding error makes $H > 5$ a fantasy generator on most tasks. **Do not select a model by reconstruction loss** when what you care about is control; select it by downstream policy quality. **Do not trust ensemble disagreement blindly** as a safety signal without first verifying that disagreement actually rises on out-of-distribution probes, because a collapsed ensemble gives a confident green light exactly where it should warn you. And **do not reach for RL at all** if the problem is a one-step decision with a known structure — that is supervised learning or Bayesian optimization, and it will beat RL handily.

If there is a single meta-rule to leave with, it is this: *measure the resource that actually constrains your project, and choose the family that is efficient in that resource.* Steps, wall-clock, GPU-hours, money, hardware wear, human supervision, and risk are all different resources, and a method that is efficient in one can be wasteful in another. Most failed RL family choices I have seen came from optimizing a resource that was not the binding one — counting steps when wall-clock was scarce, or chasing sample efficiency when the simulator made samples free. Get the resource right and the framework in section 12 makes the rest of the decision almost mechanical.

## Key takeaways

- Sample efficiency only matters when samples are expensive; the expense is a property of your problem, not your algorithm — and "expensive" spans wall-clock, money, hardware wear, supervision, and risk. Answer "how costly is a real step?" first and most decisions follow.
- Model-based RL trades model bias for sample efficiency (often 5–10x on continuous control); model-free RL trades sample efficiency for the absence of model bias and a higher asymptote. The two curves cross because the model-based asymptote has a glass ceiling (residual model error) and the model-free one does not.
- Model error compounds roughly linearly with rollout horizon, so robust model-based RL uses short rollouts ($H = 1$ to $5$) branched from real states — this is MBPO's central idea, and the optimal horizon $H^* \approx \sqrt{C_0/\epsilon_m}$ grows as the model improves.
- A model does not need to be accurate, it needs to be useful: optimize and evaluate it on downstream policy quality (value equivalence), not on state reconstruction, because average reconstruction error hides being wrong about the few decision-critical dimensions.
- The three model-based flavors — explicit-model planning (PETS), latent imagination (Dreamer), and model-as-data-amplifier (MBPO) — have different failure modes; know which you are using and diagnose by the characteristic failure.
- If you have a fast, accurate simulator, you already have a model; use massively parallel model-free RL and do not learn a worse one — and benchmark on wall-clock and GPU-hours, not step count, because parallel model-free often wins the metric that actually constrains you.
- Models earn a second job in deployment: uncertainty-aware pessimistic planning enables safe and offline RL (MOPO, MOReL) in ways model-free policies cannot match — but verify the ensemble's disagreement signal actually rises out of distribution before trusting it.
- Grow the rollout horizon as the model earns trust; never trust the model further than its held-out one-step error justifies, and prefer to supply known reward and termination functions analytically rather than learning them.

## Further reading

- Janner, Fu, Zhang, Levine, "When to Trust Your Model: Model-Based Policy Optimization" (MBPO), 2019 — the compounding-error analysis and the short-rollout recipe.
- Chua, Calandra, McAllister, Levine, "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS), 2018.
- Hafner, Lillicrap, Norouzi, Ba, "Mastering Atari with Discrete World Models" (DreamerV2), 2020; and Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), 2023.
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero), 2020.
- Hansen, Su, Wang, "TD-MPC2: Scalable, Robust World Models for Continuous Control," 2024 — the value-equivalence-plus-auxiliary-structure middle ground.
- Nagabandi et al., "Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning," 2018 — the canonical learned-dynamics-on-real-robots result.
- Yu et al., "MOPO: Model-based Offline Policy Optimization," 2020; Kidambi et al., "MOReL: Model-Based Offline Reinforcement Learning," 2020.
- Buckman et al., "Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion" (STEVE), 2018 — uncertainty-weighted value targets.
- Pineda et al., "MBRL-Lib: A Modular Library for Model-Based Reinforcement Learning," 2021 — the reference implementation for PETS and MBPO.
- Sutton & Barto, "Reinforcement Learning: An Introduction," 2nd ed. — Dyna and the planning-learning unification (chapter 8).
- Within this series: the unified map (`reinforcement-learning-a-unified-map`), the SAC and PPO deep-dives for the model-free side, and the capstone (`the-reinforcement-learning-playbook`).

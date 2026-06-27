---
title: "RL for Robotics: Bridging the Sim-to-Real Gap"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How to train robot policies in simulation with PPO and SAC, why they fail on physical hardware, and the domain-randomization and teacher-student techniques that make sim-to-real transfer actually work."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "robotics",
    "sim-to-real",
    "domain-randomization",
    "machine-learning",
    "pytorch",
    "ppo",
    "policy-gradient",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/rl-for-robotics-sim-to-real-1.png"
---

A robotics team I worked with built a beautiful quadruped policy. In simulation it crossed a rubble field of randomized rocks like a mountain goat, recovered from shoves, and tracked velocity commands to within a few percent. The evaluation script printed `success rate: 0.95` over a thousand episodes. They flashed it onto the real robot, set it on the lab floor — flat, clean, the easiest terrain imaginable — and it took four steps, planted a foot wrong, and folded sideways onto the concrete. Over fifty real trials, the same policy that scored 95% in sim scored about 20% on hardware. Nobody had changed a line of code. The policy was perfect. The *world* was different.

That gap — the chasm between a policy that works in a simulator and one that survives a real motor, a real foot, and a real contact patch — is the central problem of reinforcement learning for robotics. It is the reason robotics RL is hard in a way that Atari and chess are not. In a game, the environment you train in *is* the environment you deploy in: the same emulator, the same physics, the same pixels. In robotics, you train in a fast, cheap, slightly-wrong simulator because the real robot is slow, expensive, and breakable, and then you ask a policy to generalize across a reality gap it never saw during training. Figure 1 shows the pipeline we are going to build and defend: a physics simulator, a randomization layer that perturbs the physics, a PPO trainer chewing through billions of steps, a frozen checkpoint, and a deployment onto hardware where the only honest metric is real-world success.

![A horizontal pipeline showing physics simulator feeding a domain randomization stage, then PPO training, then a policy checkpoint, then robot deployment, then real-world evaluation at seventy percent success](/imgs/blogs/rl-for-robotics-sim-to-real-1.png)

By the end of this post you will understand *why* the 95%-to-20% collapse happens, and you will be able to implement the two techniques that close most of it: **domain randomization** (deliberately training under a distribution of wrong physics so the real physics is just one more sample) and **teacher-student distillation** (training a policy that secretly knows the true friction, then teaching a deployable student that only sees what a real robot can measure). We will write real PPO and SAC loops in Stable-Baselines3 and PyTorch, build a randomization wrapper for Gymnasium, derive why a privileged teacher beats direct training under partial observability, and walk through the four case studies that defined the field: OpenAI's Dactyl hand, ETH Zurich's ANYmal, Berkeley's Rapid Motor Adaptation quadruped, and Boston Dynamics' Spot. Throughout, the spine is the one this series keeps returning to — **an agent interacts with an environment, collects rewards, and updates a policy** — but with one brutal twist: the environment you optimize against is not the environment that grades you.

## 1. Why robotics needs RL in the first place

Before defending sim-to-real, it is worth being honest about *why* anyone reaches for reinforcement learning on a robot at all. Robots have been controlled for decades without a neural network in sight. Industrial arms run inverse-kinematics solvers and PID loops. Drones fly on cascaded controllers tuned by hand. If you can write down the dynamics and the task, classical control is faster, provably stable, and does not need ten billion simulation steps. So when does RL earn its place?

RL earns its place exactly where the task is **contact-rich, high-dimensional, and hard to specify analytically**. Three families dominate.

**Dexterous manipulation.** Picking up an arbitrary object, rotating a cube in-hand, reorienting a screwdriver — these involve dozens of make-and-break contacts per second between fingers and object. The contact dynamics are stiff, discontinuous, and notoriously hard to model or to plan through with classical trajectory optimization. A 24-degree-of-freedom hand reorienting a block has an action space and a contact schedule that defeat analytic planners but that a policy can learn from reward.

**Legged locomotion.** A quadruped or biped traversing stairs, gravel, ice, or a collapsed pile of debris has to choose footholds, modulate stiffness, and recover from slips in real time. The "right" gait is not a single trajectory; it is a *feedback policy* that maps the current proprioceptive state to joint targets. Hand-designing such a controller for arbitrary terrain is a research career; learning it from a velocity-tracking reward is a weekend of GPU time.

**Whole-body control.** Humanoids that must balance, walk, carry, and catch themselves couple every joint to every other joint. The state and action spaces are large, the system is underactuated (you cannot directly command the center of mass), and the constraints (don't fall, don't exceed torque limits) are tight. RL handles the coupling that decomposed classical controllers struggle with.

What these share is the structure that makes RL the right hammer:

- **Continuous, high-dimensional action spaces.** A quadruped commands twelve joint targets; a hand commands twenty-plus. There is no small discrete action set to enumerate, so value-based methods like DQN are out and we live in the world of policy gradients and actor-critics.
- **Partial observability.** The robot cannot directly measure ground friction, payload mass, or the exact contact force on each toe. It infers them from a *history* of proprioception. Formally the problem is a Partially Observable Markov Decision Process (POMDP): the agent sees an observation $o_t$ that is a noisy, incomplete function of the true state $s_t$.
- **Delayed and shaped rewards.** "Walk forward at 1 m/s without falling for 20 seconds" is a reward that only resolves far in the future, so we shape it into dense per-step signals (track velocity, save energy, keep feet clear) and accept the reward-engineering burden that comes with it.

Let me set the formalism so the rest of the post has a vocabulary. A robot-control problem is a Markov Decision Process (MDP), the tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$: states $\mathcal{S}$, actions $\mathcal{A}$, transition dynamics $P(s' \mid s, a)$, reward $r(s,a)$, and a discount $\gamma \in [0,1)$ that trades immediate against future reward. The agent's job is to find a policy $\pi_\theta(a \mid s)$ — a parametric map from state to a distribution over actions — that maximizes expected discounted return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{\infty} \gamma^t \, r(s_t, a_t) \right].
$$

The catch that makes robotics special is that the transition dynamics $P$ live in the simulator during training and in the physical world during deployment, and **they are not the same $P$**. Call the simulator's dynamics $P_{\text{sim}}$ and the robot's $P_{\text{real}}$. You optimize $J_{\text{sim}}(\theta)$ but you are graded on $J_{\text{real}}(\theta)$. Every technique in this post is an answer to one question: how do you maximize $J_{\text{real}}$ when you can only sample from $P_{\text{sim}}$?

## 2. Anatomy of the sim-to-real gap

The 95%-to-20% collapse is not a bug. It is the expected outcome of optimizing the wrong objective. A policy trained to maximize $J_{\text{sim}}$ will happily exploit *every* quirk of the simulator, including the quirks that have nothing to do with the real task. To close the gap you first have to enumerate where it comes from. There are four major sources, and Figure 4's randomization stack maps directly onto them.

**Dynamics mismatch.** The simulator's rigid-body solver uses nominal masses, inertias, and friction coefficients. The real robot has a battery that shifts the center of mass, a leg that is 30 grams heavier than the CAD model, and a floor whose friction coefficient you genuinely do not know — it might be 0.4 on lab linoleum and 1.1 on rubberized outdoor decking. A policy that learned to push off assuming friction 0.8 will either slip (real friction lower) or trip (real friction higher).

**Actuator modeling errors.** This is the silent killer. In simulation, when the policy commands a joint target, an idealized actuator reaches it in one step with infinite bandwidth. Real servos have finite torque, finite bandwidth, gear backlash, friction in the gearbox, and — critically — *latency*. A real position-controlled motor running a PD loop has dynamics the simulator's default actuator does not capture. The classic 2018 result from ETH (Hwangbo et al.) was that you cannot close the locomotion gap by randomizing rigid-body physics alone; you have to learn an **actuator network** — a small neural net that maps commanded torque and joint-velocity history to *actually delivered* torque — and put it inside the simulation loop. Without it, sim torques are a fantasy.

**Sensor noise and bias.** The robot's IMU drifts. Joint encoders quantize. Contact sensors are noisy or absent. The simulator, by default, hands the policy clean, exact state. A policy trained on clean state learns to trust observations it will never get cleanly, and small real-world noise pushes it off the manifold of states it ever visited in training.

**The rendering gap (for vision policies).** If the policy takes camera images, the simulator's renderer produces textures, lighting, and shadows that do not match a real camera's pixels. A policy that learned "the cube is the brown-textured blob" fails when the real cube reflects fluorescent light differently. This is the manipulation analog of the dynamics gap.

Here is the mechanism that ties it together — and it is worth stating precisely because it explains *why optimizing harder in sim makes real performance worse, not better*. Standard RL maximizes return under a single fixed transition model. With enough capacity and enough steps, the policy converges to a near-deterministic exploit of $P_{\text{sim}}$. The optimal policy under $P_{\text{sim}}$ is, generically, **brittle to perturbations of $P$**, because optimization with no incentive for robustness will ride right up against the edges of feasibility — minimum energy, minimum margin, maximum speed. Think of it as a policy standing on the peak of a knife-edge ridge: it scores maximally in sim, but any shift in the dynamics, which is exactly what deployment is, knocks it off. The 2-column comparison in Figure 2 is this story in one image: the un-randomized policy sits at sim 95% / real 20%, the randomized policy at sim 80% / real 70%. You *give up* sim peak to *buy* a robustness band wide enough to contain reality.

![A two-column comparison contrasting a policy with no randomization scoring ninety-five percent in sim but twenty percent real, against a randomized policy scoring eighty percent sim and seventy percent real](/imgs/blogs/rl-for-robotics-sim-to-real-2.png)

Formally, the fix is to stop optimizing for one MDP and start optimizing for a *distribution* of MDPs. Define a distribution $\rho$ over dynamics parameters $\xi$ (friction, mass, motor gain, latency, sensor noise). Instead of $J_{\text{sim}}(\theta)$ we maximize the **expected return over the randomization distribution**:

$$
J_{\text{DR}}(\theta) = \mathbb{E}_{\xi \sim \rho}\, \mathbb{E}_{\tau \sim \pi_\theta, P_\xi}\left[ \sum_t \gamma^t r(s_t, a_t) \right].
$$

If the real dynamics $P_{\text{real}}$ falls inside the support of $\rho$ — that is, if reality looks like one of the simulators you trained on — then a policy that does well on average over $\rho$ does well on $P_{\text{real}}$ too. That single equation is the theoretical core of domain randomization, and the rest of section 5 is about how to choose $\rho$.

## 3. Physics simulators for RL: the throughput problem

You cannot do any of this without a simulator, and the choice of simulator is not a detail — it sets the ceiling on how many environment steps you can afford, which sets the ceiling on what tasks are learnable. Recall that PPO on a quadruped needs on the order of $10^8$ to $10^{10}$ environment steps. The entire feasibility of robotics RL rests on being able to generate those steps fast. Figure 5 lays out the contenders.

![A five-row comparison matrix of MuJoCo, Isaac Gym, PyBullet, Brax, and Genesis across parallel environment count, GPU-native physics, differentiability, and licensing cost](/imgs/blogs/rl-for-robotics-sim-to-real-5.png)

**MuJoCo** (Multi-Joint dynamics with Contact) is the long-standing standard for contact-rich research. Its contact solver is accurate and stable, it is what most published locomotion and manipulation results use, and since it went open-source and gained the JAX-based **MJX** backend it can also run many environments on a GPU. On CPU it runs hundreds of environments; on MJX it scales much further. If you want your results to be comparable to the literature, MuJoCo is the safe default.

**Isaac Gym** (and its successor Isaac Lab) is NVIDIA's GPU-native physics engine, and it changed what was practical. It runs the *entire* simulation — physics, observations, rewards — on the GPU, keeping the data resident in GPU memory so there is no CPU-GPU transfer per step. The headline number is real: **4096 parallel environments on a single GPU**, generating on the order of tens of thousands to hundreds of thousands of steps per second for a quadruped. That is the difference between training a locomotion policy in twenty minutes versus a day. The 2021 "Learning to Walk in Minutes" result from ETH and NVIDIA (Rudin et al.) trained a deployable ANYmal policy in *under four minutes of wall-clock time* on one GPU precisely because of this massive parallelism.

**PyBullet** is the open-source, dependency-light option. It is CPU-based, slower, and less accurate on stiff contact, but it is free, easy to install, and great for prototyping and education. Many tutorials and the original Gym robotics environments target it.

**Brax** is Google's fully differentiable, JAX-based physics engine. It runs entirely on accelerators (GPU/TPU) and, because it is written in JAX, the whole simulation is differentiable end to end. That opens the door to analytic-gradient policy optimization, not just black-box RL.

**Genesis** is a newer generative-and-differentiable simulator aiming for both extreme throughput and physical accuracy across rigid, soft, and fluid bodies, positioned as a next-generation universal physics platform.

The frontier worth flagging is **differentiable simulation**. If the simulator is differentiable, you can compute $\nabla_\theta J$ by backpropagating *through the physics* rather than estimating it with the high-variance policy-gradient Monte Carlo estimator. Brax and Warp (NVIDIA's differentiable kernel framework) make this tractable. The catch — and it is a real one — is that contact is discontinuous, so gradients through hard contact are noisy or biased, and analytic-gradient methods can get stuck in ways that black-box PPO does not. For contact-rich tasks, most production pipelines still use massively parallel *non*-differentiable simulation with PPO, because throughput beats gradient quality when you have 4096 environments. The differentiable frontier is most compelling for smooth, low-contact dynamics.

#### Worked example: counting your step budget

Say you have one A100 and you are training an ANYmal locomotion policy. Isaac Gym gives you roughly 4096 parallel environments at, conservatively, 50,000 simulated steps per second aggregate throughput. You want $2 \times 10^9$ environment steps (a typical figure for a robust randomized locomotion policy). The math: $2{\times}10^9 / 50{,}000 = 40{,}000$ seconds, about 11 hours. On CPU MuJoCo at, say, 2,000 steps/second across a few hundred environments, the same budget is $10^6$ seconds — about 11.5 *days*. That single order-of-magnitude — three orders, really — is why GPU-native simulation is not a luxury. It is what moved sim-to-real locomotion from "PhD thesis" to "afternoon experiment."

## 4. PPO for locomotion: the recipe that actually ships

For locomotion, the workhorse is **Proximal Policy Optimization (PPO)**. It is on-policy, it is stable, and it parallelizes trivially across thousands of environments — which is exactly the regime GPU simulators put you in. (For the full PPO derivation see the dedicated post on [proximal policy optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo); here I'll use it as a tool and focus on the robotics-specific setup.) The reason PPO dominates here over off-policy methods like SAC is not that it is more sample-efficient — SAC is dramatically *more* sample-efficient per step. It is that on a GPU simulator, samples are nearly free, and PPO's stability and parallelism let you flood the policy with $10^9$ cheap samples without the replay-buffer staleness and hyperparameter fragility that off-policy methods bring at extreme parallelism.

Let me make the ANYmal-style setup concrete. The observation is a **48-dimensional proprioceptive vector**: base linear and angular velocity (6), gravity direction in the body frame (3), the velocity command (3), joint positions and velocities (12 + 12 = 24), and the previous action (12). The action is **12 joint position targets**, one per actuator, fed to a PD controller that converts them to torques. The reward is the shaped sum we will dissect in section 8: track the commanded base velocity, penalize energy, penalize large contact forces, reward keeping feet clear of the ground during swing.

Here is a PPO training loop in Stable-Baselines3, the version most teams start from before moving to a GPU-native trainer. The key robotics-specific choices are the network size, the entropy coefficient that keeps exploration alive, and the vectorized environment.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# 64 parallel CPU envs; on Isaac Gym this would be 4096 on one GPU.
vec_env = make_vec_env(
    "Quadruped-v0",          # your registered locomotion env
    n_envs=64,
    vec_env_cls=SubprocVecEnv,
)

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    n_steps=24,              # short rollouts: many envs, short horizon
    batch_size=64 * 24,      # n_envs * n_steps
    n_epochs=5,              # PPO inner update epochs
    gamma=0.99,
    gae_lambda=0.95,         # GAE advantage smoothing
    clip_range=0.2,          # the PPO trust-region clip
    ent_coef=0.01,           # entropy bonus: keep the gait stochastic early
    policy_kwargs=dict(net_arch=[512, 256, 128]),  # bigger than Atari
    verbose=1,
)

model.learn(total_timesteps=200_000_000)
model.save("quadruped_ppo")
```

A few things in that snippet are load-bearing and not obvious. The rollout length `n_steps=24` is *short* — you are not collecting long trajectories from a few environments, you are collecting short slices from many. This is the parallelism-first regime: with 4096 environments and 24 steps you get a batch of ~98,000 transitions per update, which is enormous and low-variance. The network `[512, 256, 128]` is bigger than the tiny MLPs that solve CartPole because the proprioceptive-to-gait mapping is genuinely complex. The `ent_coef=0.01` entropy bonus matters more than people expect: it keeps the policy stochastic early so it explores foot-placement strategies instead of collapsing to a stiff, conservative shuffle.

**Curriculum learning** is the other half of the recipe. If you drop a randomly initialized policy onto rough terrain, it falls instantly and never collects useful reward — the terrain is too hard to bootstrap from. The fix is a curriculum: start every environment on flat ground, and *progressively* increase terrain difficulty as the policy's velocity-tracking performance crosses a threshold. NVIDIA's terrain curriculum promotes an environment to harder terrain only when the agent successfully traverses its current level, and demotes it if it fails. This automatic difficulty adjustment is what lets a single policy end up handling stairs, slopes, and rubble.

```python
def update_terrain_curriculum(env_id, episode_return, distance_traveled,
                              terrain_levels, success_thresh=0.8):
    # Promote envs that traversed most of their terrain, demote the failures.
    target = 0.8 * env.max_distance  # traversed 80% of the level
    if distance_traveled > target:
        terrain_levels[env_id] = min(terrain_levels[env_id] + 1, MAX_LEVEL)
    elif distance_traveled < 0.4 * env.max_distance:
        terrain_levels[env_id] = max(terrain_levels[env_id] - 1, 0)
    return terrain_levels
```

#### Worked example: a velocity-tracking learning curve

On a flat-ground curriculum start, a fresh PPO policy with the setup above has an average episode return near zero — it falls in the first few steps, collects the survival bonus for a fraction of a second, and resets. By about 50 million steps it has learned a stable standing posture (return climbs as it stops falling). By 200 million steps on a GPU simulator (roughly an hour of wall-clock) it tracks commanded velocities of 0 to 1 m/s with a tracking error under 0.1 m/s and a velocity-tracking reward component near its maximum. Push the curriculum to rough terrain and the return dips — the task got harder — then recovers as the policy learns to modulate foot placement. The signature of a healthy run is this *saw-tooth*: every curriculum promotion drops the return, and the policy claws it back. A run whose return only ever goes up is a run whose curriculum never advanced.

## 5. Domain randomization: training under a distribution of wrong physics

We derived the objective in section 2: maximize $J_{\text{DR}}(\theta) = \mathbb{E}_{\xi \sim \rho}\,[\,\ldots]$. Now we make $\rho$ concrete. Domain randomization (DR) means sampling the simulator's physics parameters from a distribution at the start of each episode (or even each environment, persistently), so the policy is forced to be robust across the whole range rather than overfit to one point. Figure 4's stack shows the five layers you randomize, from base physics down to rendering.

![A vertical stack of five domain-randomization layers from base physics through contact model, actuator model, and sensor noise to rendering](/imgs/blogs/rl-for-robotics-sim-to-real-4.png)

The simplest version is **uniform randomization**: pick a range for each parameter and sample uniformly. Typical ranges for a quadruped:

| Parameter | Nominal | Randomization range | Why |
| --- | --- | --- | --- |
| Ground friction $\mu$ | 0.8 | 0.3 – 3.0 | Floor material is unknown |
| Body mass | CAD value | $\pm$20% | Battery, payload, CAD error |
| Motor strength (gain) | nominal | $\pm$10% | Servo variation, wear |
| Motor latency | 0 ms | 0 – 20 ms | Real actuator bandwidth |
| Joint friction | small | 0 – 2$\times$ | Gearbox friction |
| IMU noise | 0 | $\sigma$ = small | Real sensor noise |
| Push perturbation | none | random shoves | Robustness to disturbance |

Here is the wrapper. It re-samples dynamics every reset, which is the persistent-per-episode flavor that works well for locomotion.

```python
import numpy as np
import gymnasium as gym

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # Sample a fresh "world" for this episode.
        self.unwrapped.set_friction(np.random.uniform(0.3, 3.0))
        self.unwrapped.scale_body_mass(np.random.uniform(0.8, 1.2))
        self.unwrapped.scale_motor_gain(np.random.uniform(0.9, 1.1))
        self.unwrapped.set_motor_latency(np.random.uniform(0.0, 0.02))
        self.unwrapped.set_joint_friction(np.random.uniform(0.0, 2.0))
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Additive sensor noise on the observation the policy actually sees.
        obs = obs + np.random.normal(0.0, 0.01, size=obs.shape)
        return obs, reward, terminated, truncated, info
```

**Structured DR and curriculum.** Uniform randomization over the full range from the start can be too hard — the policy never gets a foothold. So you can *curriculum the randomization itself*: start with narrow ranges (friction 0.7–0.9) and widen them as the policy succeeds. This is the randomization analog of terrain curriculum and it converges faster than slamming the full range on from step zero.

**Automatic Domain Randomization (ADR)** is the elegant generalization that powered OpenAI's Dactyl. Instead of hand-choosing ranges, ADR *learns* them. It maintains a range for each parameter and automatically widens a range whenever the policy performs well enough at the current boundary, and narrows it if performance drops. This is "randomization of randomization": the curriculum over difficulty is automatic, and it keeps pushing the boundary outward exactly as fast as the policy can absorb. ADR is why Dactyl could eventually handle a physical Rubik's cube despite never seeing the real one — the training distribution had been automatically widened until the real cube's properties were comfortably inside it.

```python
class ADR:
    """Automatic Domain Randomization: widen ranges the policy can handle."""
    def __init__(self, params, perf_high=0.8, perf_low=0.5, step=0.02):
        # params: {name: [low, high]}
        self.params = params
        self.perf_high, self.perf_low, self.step = perf_high, perf_low, step

    def sample(self):
        return {k: np.random.uniform(lo, hi) for k, (lo, hi) in self.params.items()}

    def update(self, name, at_boundary, performance):
        lo, hi = self.params[name]
        if at_boundary == "high" and performance > self.perf_high:
            self.params[name][1] = hi + self.step      # widen upper bound
        elif at_boundary == "high" and performance < self.perf_low:
            self.params[name][1] = max(lo, hi - self.step)  # retreat
        # symmetric logic for the low boundary
```

**Why too much randomization hurts.** There is a real failure mode here, and it is worth dwelling on because it surprises people who treat DR as "more is always better." If $\rho$ is too wide, the *only* policy that does well on average across all of it is a maximally conservative one — slow, stiff, low-energy, never committing to a dynamic motion because in some sampled world that motion would fail. This is **policy conservatism**, and on a real robot it shows up as a sluggish, over-damped gait that wastes energy and cannot move fast. The art of DR is making $\rho$ wide enough to contain reality but no wider. ADR helps because it never widens a range faster than the policy can stay competent, so it finds the largest distribution the policy can handle *without* collapsing into conservatism. The honest framing: DR is a robustness-versus-performance dial, and Figure 2's "sim 80% not 95%" is the tax you pay.

## 6. Teacher-student distillation: learning from privileged information

Domain randomization makes the policy robust, but it leaves a deep problem unsolved. The robot is partially observable — it cannot measure friction or contact forces — yet the *optimal* action genuinely depends on those quantities. On slippery ground you should step differently than on grippy ground, but the policy cannot see which it is on. How do you learn a good policy for a state you cannot observe?

The breakthrough idea is **teacher-student distillation with privileged information**, and Figure 3 is its anatomy. In simulation, you *do* have access to the true friction, the true contact forces, the true terrain height under each foot — the simulator computed them. So you train a **teacher** policy that takes the full privileged state and learns an excellent policy by PPO. The teacher is not deployable (a real robot has no friction sensor), but it is an oracle. Then you train a **student** policy that takes *only* what a real robot can measure — proprioception plus a history of past observations — and you train it to imitate the teacher's actions.

![A dataflow graph where simulator ground truth feeds a privileged teacher trained by PPO, whose actions and the deployable observation history feed a student LSTM trained by MSE distillation and then deployed on the robot](/imgs/blogs/rl-for-robotics-sim-to-real-3.png)

Why is this better than just training the student directly with PPO under partial observability? Here is the theory, and it is the most important argument in the post. Training a partially-observed policy with RL is hard for two compounding reasons. First, **credit assignment under partial observability is high-variance**: the policy gradient is already a Monte Carlo estimate of the score function, and hiding the state that explains the reward inflates that variance further, because two episodes with identical observations but different hidden friction get different returns, which the gradient interprets as noise. Second, the student must *simultaneously* solve two problems — infer the hidden state from history (a hard system-identification problem) *and* learn the optimal action given that state (a hard control problem) — entangled, with only a scalar reward to guide both.

Distillation **decouples** these. The teacher solves the control problem in the easy, fully-observed setting where the gradient is well-conditioned. The student then solves only the *inference* problem: given a window of proprioceptive history, predict the action the teacher would take if it could see the privileged state. That second problem is **supervised learning** — a regression from observation history to teacher action — with a dense, low-variance gradient on every single step, not a sparse high-variance return. Supervised regression with millions of (history, target-action) pairs converges far faster and more reliably than RL through a POMDP. This is the core insight of the 2020 ETH "Learning Quadrupedal Locomotion over Challenging Terrain" work (Lee et al., *Science Robotics*) and it generalizes everywhere privileged information exists in sim.

The student is trained with **DAgger** (Dataset Aggregation), the right imitation algorithm for this setting. Naive behavior cloning — collect teacher trajectories once, fit the student — fails because of *covariance shift*: the student makes small mistakes, drifts into states the teacher never visited, and has no idea what to do there because it was never trained on them. DAgger fixes this by rolling out the *student*, querying the *teacher* for the correct action at every state the student actually visits, and adding those to the training set. The student is trained on its own state distribution, labeled by the teacher.

```python
import torch, torch.nn as nn

class StudentPolicy(nn.Module):
    """Deployable: only proprioception + history, no privileged state."""
    def __init__(self, obs_dim=48, history_len=50, action_dim=12, hidden=256):
        super().__init__()
        # LSTM compresses the observation history into a latent that
        # implicitly encodes friction, mass, terrain — system identification.
        self.encoder = nn.LSTM(obs_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs_history):          # (batch, history_len, obs_dim)
        feats, _ = self.encoder(obs_history)
        return self.head(feats[:, -1, :])    # action from last timestep

def dagger_update(student, teacher, envs, opt, beta):
    """One DAgger iteration: roll out (mostly) the student, label with teacher."""
    obs_hist, priv_states = collect_rollouts(student, teacher, envs, beta)
    teacher_actions = teacher.act(priv_states)        # oracle labels
    student_actions = student(obs_hist)
    loss = nn.functional.mse_loss(student_actions, teacher_actions)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

**Rapid Motor Adaptation (RMA).** Kumar et al. (2021) turned this into a deployable, real-time recipe. RMA trains the teacher with access to an *environment-factor vector* $z_t$ — a compact code of the privileged parameters (friction, mass, motor strength). The teacher's policy is $\pi(a_t \mid x_t, z_t)$ where $x_t$ is proprioception. Then RMA trains an **adaptation module** $\phi$ that estimates $\hat{z}_t$ from a short history of proprioception and actions, $\hat{z}_t = \phi(x_{t-k:t}, a_{t-k:t})$, by regressing $\hat z_t$ onto the true $z_t$. At deployment, the robot runs $\pi(a_t \mid x_t, \hat{z}_t)$ — proprioceptive history goes into $\phi$, the estimated factor goes into the policy, both at the 100+ Hz control rate. The LSTM (or 1-D conv) over action-state history is doing *implicit online system identification*: it is figuring out, from how the robot has been responding, what world it is in. That is the same idea as the student's history encoder, made into an explicit adaptation module.

#### Worked example: distillation loss and transfer

In a typical ETH-style pipeline, the privileged teacher reaches a velocity-tracking reward near its ceiling in simulation — call it 95% of max return. Train a student *directly* with PPO under partial observation and it plateaus lower, maybe 75% of the teacher's return, and it transfers poorly because it never robustly inferred the hidden state. Now distill: the student's MSE-on-actions loss starts around 0.05 (radians$^2$, say) and drops below 0.005 after a few million DAgger steps, at which point the student's behavior is nearly indistinguishable from the teacher's *and* it only uses deployable observations. On hardware, the distilled student transfers because the history-encoder learned to read friction and terrain from proprioception alone — the exact capability a directly-trained student struggles to acquire from sparse reward.

## 7. Vision-based policies and the rendering gap

Everything so far assumed proprioceptive policies. Manipulation, and increasingly navigation, needs *vision*, and vision opens a second reality gap: the **rendering gap**. The simulator's rasterizer produces pixels that do not match a real camera. The fix is the same idea — randomize — but applied to the visual channel.

**Domain randomization for rendering** means sampling, every episode: object and background textures (from a large library of random textures), lighting direction and intensity, camera position and field of view, color and contrast jitter, and distractor objects. The original 2017 Tobin et al. result showed that a detector trained *only* on cartoonishly randomized renders — no photorealism at all — transferred to real images, because the network learned to be invariant to texture and lighting and to attend to *shape*, which is the one thing the randomization preserved. The philosophy: if the policy sees so many random visual worlds in training, the real world is just another one.

**Depth versus RGB.** A consistently underrated lever is to use **depth images instead of RGB**. Depth is far more sim-to-real transferable because the rendering gap for geometry is small — a depth camera and a simulator agree closely on "how far is this surface" — whereas the gap for color and texture is large. For grasping and locomotion-over-terrain, depth-only or depth-plus-proprioception policies routinely transfer better than RGB policies. This is exactly the recommendation in the debugging tree (Figure 8): if a vision policy fails the transfer, suspect the render gap and switch to depth.

**NeRF-based and photorealistic sim** is the other direction: instead of fighting the rendering gap with randomization, *close* it by making the simulator render photorealistic images. Neural radiance fields and modern Gaussian-splatting reconstructions let you build a photorealistic digital twin of the actual deployment scene, so the train and test pixels match. This is powerful for fixed environments (a specific warehouse) but less general than randomization for open-world deployment.

**The see-through failure mode.** A concrete vision-transfer trap in grasping: in sim, transparent or specular objects render with idealized highlights or render as fully opaque, so the depth map is clean. A real depth camera, by contrast, sees *through* glass and produces holes or garbage on shiny surfaces. A grasp policy that trusts clean depth will reach for a phantom. The fix is to *simulate the sensor's failure modes* — add holes, dropouts, and specular artifacts to simulated depth — so the policy learns to handle the missing data it will actually get.

```python
def corrupt_depth(depth, dropout_p=0.05, max_depth=3.0):
    """Make simulated depth look like a real, noisy depth camera."""
    noisy = depth + np.random.normal(0, 0.01 * depth, depth.shape)  # range-dep noise
    holes = np.random.rand(*depth.shape) < dropout_p                # sensor dropouts
    noisy[holes] = 0.0                                              # zero = no return
    noisy[noisy > max_depth] = 0.0                                  # out of range
    return noisy
```

## 8. Whole-body control and the art of reward engineering

A robotics RL policy is only as good as its reward, and locomotion reward is the most heavily engineered object in the whole pipeline. This is where many projects quietly fail: not in the algorithm, but in the reward terms. Figure 7 contrasts the two regimes — a sparse reward that never learns versus a dense shaped reward that converges.

![A two-column comparison of a sparse success-only reward that produces no gradient and zero return against a dense shaped reward of velocity tracking, energy, and contact terms that converges in five hundred million steps](/imgs/blogs/rl-for-robotics-sim-to-real-7.png)

Why can't you just reward "didn't fall, moved forward"? Because that reward is **sparse**: for the first many millions of steps the random policy falls immediately and the reward is effectively constant, so the policy gradient is near zero — there is nothing to climb. (This connects to the broader [reward-shaping](/blog/machine-learning/reinforcement-learning/reward-shaping-for-financial-rl) discussion in the finance posts; the principle is identical even though the domain is not.) Dense shaping turns *every step* into signal. A representative whole-body locomotion reward for a Unitree Go2 or ANYmal-class robot looks like a weighted sum:

| Reward term | Form | Role |
| --- | --- | --- |
| Velocity tracking | $\exp(-\lVert v_{xy} - v^{*}_{xy}\rVert^2 / \sigma)$ | Follow the command (the task) |
| Yaw-rate tracking | $\exp(-(\omega_z - \omega^*_z)^2 / \sigma)$ | Turn as commanded |
| Survival / alive bonus | $+c$ per step alive | Don't fall (keeps episodes long) |
| Energy / torque penalty | $-\lVert \tau \rVert^2$ | Efficient, smooth motion |
| Action rate penalty | $-\lVert a_t - a_{t-1}\rVert^2$ | Avoid jittery commands |
| Foot clearance | reward swing-foot height | Step *over* obstacles, no scuffing |
| Hip / joint alignment | $-\lVert q - q_{\text{default}}\rVert^2$ | Natural posture, no splaying |
| Contact / impact penalty | $-\lVert F_{\text{contact}}\rVert$ on touchdown | Soft landings, less hardware wear |

The weights matter enormously, and tuning them is the dark art. Too much energy penalty and the robot freezes to save power (the degenerate "do nothing" optimum). Too little action-rate penalty and the gait is high-frequency jitter that destroys real gearboxes. The exponential form on tracking terms is deliberate: it saturates near zero error so the policy is rewarded for *getting close* without being punished into instability for tiny residuals.

```python
def compute_reward(state, action, prev_action, cmd):
    v_xy = state["base_lin_vel"][:2]
    w_z  = state["base_ang_vel"][2]
    r_track_v = np.exp(-np.sum((v_xy - cmd["v_xy"]) ** 2) / 0.25)
    r_track_w = np.exp(-(w_z - cmd["w_z"]) ** 2 / 0.25)
    r_alive   = 1.0
    r_energy  = -0.0002 * np.sum(action ** 2)
    r_smooth  = -0.01  * np.sum((action - prev_action) ** 2)
    r_clear   = 0.1 * foot_clearance_reward(state)
    r_posture = -0.05 * np.sum((state["joint_pos"] - DEFAULT_POSE) ** 2)
    return (1.0 * r_track_v + 0.5 * r_track_w + 0.2 * r_alive
            + r_energy + r_smooth + r_clear + r_posture)
```

**Reward normalization** is the other practical necessity. With terms on wildly different scales, the optimizer chases whichever has the largest gradient. Running normalization of the advantage (subtract mean, divide by standard deviation across the batch — which PPO's GAE plus advantage normalization already does) keeps the update well-conditioned. SB3's `VecNormalize` wrapper does the same for observations and rewards.

**Hand-tuning versus learned rewards.** The reward table above is hand-designed, and most shipped locomotion uses exactly that. The alternative is to *learn* the reward. **Inverse RL** and motion imitation (as in DeepMimic and the AMP family) take motion-capture clips of real animals or humans and reward the policy for matching the reference motion's style, which produces strikingly natural gaits without hand-tuning every clearance and posture term. The trade-off: motion imitation gives you natural-looking behavior but constrains the policy to the reference motions' repertoire, whereas hand-shaped reward gives you full control of the objective at the cost of weeks of tuning.

#### Worked example: diagnosing a frozen robot

You launch a Go2 training run and after 100M steps the robot stands but refuses to walk — velocity-tracking reward is stuck near zero while the energy reward is near its max. Diagnosis: the energy penalty weight is too high relative to velocity tracking, so the policy found the degenerate optimum "stand still, pay no energy, collect the alive bonus." The fix is to *reduce* the energy coefficient (from, say, 0.001 to 0.0002) and/or *increase* the velocity-tracking weight until moving is worth more than the energy it costs. After the rebalance, the velocity-tracking component climbs and the gait emerges. This is the single most common locomotion-reward bug, and it is purely a *weights* problem — the algorithm was never broken.

## 9. Online adaptation: changing the policy after deployment

Domain randomization and distillation produce a *fixed* policy that is robust across a range. But the real world can drift outside the range, or change mid-episode — the robot picks up a payload, a leg's motor heats up and weakens, the terrain transitions from carpet to ice. **Online adaptation** lets the deployed policy change in response.

**Implicit adaptation via history** is the workhorse, and we already built it: RMA's adaptation module and the student's LSTM encoder both perform *online system identification* from the recent history of states and actions. The 2023 "Walk These Ways" and the adaptive-locomotion line (Margolis et al.) lean hard on this — the policy conditions on a window of proprioceptive history, and the recurrent or convolutional encoder implicitly estimates the current dynamics and adjusts behavior. No explicit "I am now on ice" classifier is needed; the history encoder learns the mapping from "how the robot has been responding" to "how to act now." This is adaptation that runs at the control rate with zero extra optimization at deployment — the network simply produces different actions because its input history changed.

**Explicit online system identification** estimates the dynamics parameters directly and feeds them to the policy, which is the RMA architecture made fully explicit. You run a small estimator (a Kalman filter or a learned regressor) that maintains a belief over friction and mass from the observed transitions, and the policy is conditioned on that belief.

**Meta-learning** is the heavyweight option. Model-Agnostic Meta-Learning (MAML) trains the policy parameters $\theta$ such that *a single gradient step* on a new task produces a good policy: $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{task}}(\theta)$, and the meta-objective optimizes $\theta$ so that the post-step $\theta'$ performs well across a distribution of tasks. For robotics this means the policy can adapt to a genuinely new terrain or a damaged leg with a few real rollouts of fine-tuning rather than retraining from scratch. The honest assessment: MAML and explicit meta-RL are powerful but heavier and less commonly shipped than the implicit-history approach, because doing a gradient step on a real robot mid-deployment is operationally fraught, whereas a forward pass through an LSTM is free. (The dedicated meta-learning treatment, when it lands in this series, goes deeper on the bilevel optimization; here the takeaway is that implicit history-based adaptation captures most of the benefit at a fraction of the deployment risk.)

```python
# Implicit adaptation: the policy's behavior changes because its input
# history changes, with NO optimization at deployment time.
class AdaptiveController:
    def __init__(self, policy, history_len=50):
        self.policy = policy            # trained student / RMA policy
        self.history = deque(maxlen=history_len)

    def act(self, proprio_obs):
        self.history.append(proprio_obs)
        hist = torch.tensor(np.stack(self.history)).unsqueeze(0).float()
        with torch.no_grad():
            return self.policy(hist).squeeze(0).numpy()  # adapts via history
```

## 10. Real deployment: latency, safety, and the things sim never tells you

A policy that transfers in principle still has to run on hardware, in real time, without destroying the robot. These engineering constraints are where many sim-to-real efforts die quietly.

**The latency budget.** A locomotion control loop typically runs at 50–100 Hz, which means a control period of 10–20 ms. *Everything* — reading sensors, running policy inference, sending motor commands — must fit inside that window, every cycle, with no jitter. A policy that takes 25 ms to run on the on-board computer cannot drive a 50 Hz loop; it will miss deadlines and the robot will become unstable. This forces small networks: the deployed student is often a few hundred-unit MLP or a modest LSTM, not a giant transformer, precisely so inference fits the budget on an on-board CPU or small GPU. There is also a subtler point: the *training* simulation must model the control latency, because a policy trained at zero latency and deployed at 15 ms latency is being asked to control a different system (this is the actuator-latency randomization in Figure 4 again).

**On-board versus offloaded computation.** You can run inference on the robot (low latency, no network dependence, the standard for locomotion) or offload it to a nearby workstation (more compute, but you add wireless latency and a single point of failure). Locomotion almost always runs on-board because a dropped wifi packet at 100 Hz is a fall. Heavy vision pipelines sometimes offload, accepting the latency for the compute.

**Safety during training and deployment.** If you ever train or fine-tune on real hardware, you need: torque limits enforced in firmware (the policy *cannot* command a damaging torque even if it wants to), joint-position limits, a fall-detection-and-recovery behavior, and a physical emergency stop with a human holding it. Most teams avoid on-hardware training entirely — that is the whole point of sim-to-real — but even a deployed policy needs the torque clamps and the e-stop, because a transfer failure is a robot slamming into the floor.

```python
def safe_act(policy_action, joint_pos, joint_vel,
             torque_limit=40.0, pos_limits=(LOW, HIGH)):
    target = np.clip(policy_action, pos_limits[0], pos_limits[1])  # joint limits
    # PD controller produces torque; clamp it hard before it reaches firmware.
    torque = KP * (target - joint_pos) - KD * joint_vel
    torque = np.clip(torque, -torque_limit, torque_limit)         # never exceed
    if fall_detected(joint_pos):
        return recovery_torques()                                  # safe pose
    return torque
```

**Dexterous manipulation deployment** raises the bar further. OpenAI's Rubik's-cube Shadow Hand had to deal with all of the above *plus* a 24-DOF hand, vision for cube-state estimation, and contact dynamics an order of magnitude more delicate than a foot on the ground. The control loop, the safety limits on finger torque, and the sensor-noise modeling all had to be exactly right, which is part of why that project took years.

## 11. Case studies: the results that defined the field

Theory and code are worth more when you can point to robots that actually did the thing. Figure 6 places these on a timeline; here are the details.

![A horizontal timeline of sim-to-real milestones from 2017 MuJoCo locomotion through 2019 OpenAI Dactyl, 2021 ANYmal terrain, 2022 RMA real-time adaptation, to 2023 whole-body humanoid control](/imgs/blogs/rl-for-robotics-sim-to-real-6.png)

**OpenAI Dactyl and the Rubik's cube (2018–2019).** A Shadow Hand learned in-hand manipulation — first reorienting a block, then solving a physical Rubik's cube — trained *entirely* in simulation with massive domain randomization, including the Automatic Domain Randomization that learned its own ranges. The headline result: a policy trained only in randomized sim, never on the real hand, succeeded at manipulating and turning a physical cube, demonstrating emergent robustness (it recovered even when researchers perturbed the cube with a plush giraffe or tied two fingers together). The cost was enormous compute and years of effort, and the success rate on the full cube solve was modest, but it was the definitive proof that pure-sim training with aggressive DR could cross to delicate real manipulation. The lesson the field took: *randomize until reality is in-distribution*.

**ANYmal over challenging terrain (ETH Zurich, 2019–2021).** This is the locomotion canon. The 2019 *Science Robotics* paper (Hwangbo, Lee, Hutter et al.) first showed that an **actuator network** — learning the real servo's torque response and putting it in the sim loop — was the missing piece for locomotion transfer. The 2020 follow-up (Lee et al.) used the teacher-student privileged-distillation recipe to learn a policy that walked over rocks, streams, and rubble in real outdoor environments. And the 2021 "Learning to Walk in Minutes" (Rudin et al.) showed that with 4096 parallel Isaac Gym environments, a deployable policy trains in *under four minutes* of wall-clock on a single GPU. Together these papers are the playbook this post is built on: actuator modeling, massive parallelism, terrain curriculum, privileged distillation.

**Rapid Motor Adaptation (Berkeley, 2021).** Kumar, Fu, Pathak, and Malik showed a quadruped (the low-cost Unitree A1) adapting *in real time* to drastically different real-world conditions — oily surfaces, foam, a payload, a hiking trail — using only proprioception, no vision, no real-world training. The adaptation module estimated the environment factors from proprioceptive history at 100 Hz, and the robot adjusted within fractions of a second. RMA's significance was showing that robust *plus* adaptive could run on cheap hardware with on-board compute, and it set the template for the implicit-history adaptation that dominates today.

**Boston Dynamics Spot and the production reality.** Spot is the counterpoint worth including for honesty: it is a deployed commercial robot, and its core locomotion was built primarily on sophisticated *model-based* control, not end-to-end RL. Boston Dynamics has increasingly incorporated RL — they have published on using RL for more dynamic and robust locomotion behaviors — but the production story is a *hybrid*: model-based controllers where you can write the dynamics down, learned policies where you cannot. The lesson for a practitioner: sim-to-real RL is a powerful tool, not a mandate. The most robust real systems often blend learned and classical control, using each where it is strongest.

## 12. When to use sim-to-real RL (and when not to)

This is the decisive section, because the failure mode I see most is reaching for sim-to-real RL when a simpler tool would have shipped months sooner.

**Use sim-to-real RL when:**

- The task is **contact-rich or high-dimensional** in a way that defeats analytic control — dexterous manipulation, legged locomotion over varied terrain, whole-body humanoid control.
- You have a **good simulator** for the task. No simulator, no sim-to-real. If you cannot simulate the contact physics of your task with reasonable fidelity, you have no training ground.
- **Real-world rollouts are expensive or dangerous** — which is almost always true in robotics — so you cannot train model-free directly on hardware. (If real steps were cheap, you would skip the sim and the gap entirely.)
- The behavior you want is a **feedback policy**, not a fixed trajectory — something that must react to the world in closed loop.

**Reach for something simpler when:**

- **You can write down the dynamics and the task.** A pick-and-place with a known object on a known table is inverse kinematics plus a motion planner, full stop. Do not train a policy to do what a solver does provably and instantly.
- **The task is a fixed, repeatable trajectory.** Welding the same seam, moving between two fixed poses — trajectory optimization or a taught path beats RL.
- **You have no simulator and cannot build one.** Then you are in the world of real-robot learning, offline RL from logged data, or learning from demonstration — different toolkits entirely.
- **Classical control already meets spec.** If a well-tuned PID or MPC controller hits your performance target, the RL policy is a science project, not a product. The most robust shipped systems (Spot) are hybrids precisely because classical control is unbeatable where it applies.

The honest meta-point, which connects to the [model-based vs model-free](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which) decision and the [actor-critic family](/blog/machine-learning/reinforcement-learning/actor-critic-a2c-a3c): sim-to-real RL trades enormous *simulation* compute and reward-engineering effort for the ability to handle tasks no analytic method can. That trade is worth it for locomotion and dexterous manipulation. It is a waste for problems classical robotics already solved.

When you *do* commit, Figure 8 is the debugging discipline that turns a failed transfer into a fixed one. Do not respond to a transfer failure by retraining randomly — *read the symptom*. Jerky, unstable motion on hardware points at the actuator model: add motor latency and gain noise to sim. Slipping and falling points at the contact model: randomize friction harder. Drifting off course points at sensor noise or, for vision, the render gap. Each symptom maps to a layer of the randomization stack you modeled too cleanly.

![A decision tree mapping real-world failure symptoms to fixes, where jerky motion points to actuator modeling, slipping points to friction randomization, and drift points to sensor noise or using depth instead of RGB](/imgs/blogs/rl-for-robotics-sim-to-real-8.png)

## 13. A note on SAC, and why locomotion still picks PPO

The kit and the literature both lean PPO for locomotion, but it is worth being precise about the alternative, because off-policy actor-critics like **Soft Actor-Critic (SAC)** are the right choice in a different regime. SAC is an off-policy, maximum-entropy actor-critic: it augments the reward with an entropy term $\alpha \mathcal{H}(\pi(\cdot \mid s))$ so the policy stays as stochastic as possible while maximizing return, which both improves exploration and yields robustness. Crucially, SAC reuses every transition from a replay buffer many times, making it **dramatically more sample-efficient** than PPO per environment step. (For the deterministic-policy lineage SAC builds on, see [DDPG and TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3).)

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy", env="Manipulator-v0",
    learning_rate=3e-4,
    buffer_size=1_000_000,     # large replay buffer: reuse every transition
    batch_size=256,
    tau=0.005,                 # soft target update
    gamma=0.99,
    ent_coef="auto",           # automatic temperature tuning
    train_freq=1,
    gradient_steps=1,
)
model.learn(total_timesteps=1_000_000)  # ~100x fewer steps than PPO needs
```

So why does locomotion still pick PPO? Because the decision is not "which is more sample-efficient" — it is "which makes best use of my actual constraint." On a GPU simulator with 4096 environments, samples are nearly free, and PPO's on-policy stability and embarrassingly parallel rollouts let it consume $10^9$ cheap samples without replay-buffer staleness or the hyperparameter fragility SAC shows at extreme parallelism. SAC wins when each sample is precious — real-robot training, slow simulators, or manipulation tasks where you cannot spin up thousands of parallel environments cheaply. The rule of thumb:

| Regime | Pick | Why |
| --- | --- | --- |
| Massive parallel sim (Isaac Gym), locomotion | PPO | Samples free; stability + parallelism dominate |
| Few/slow envs, real-robot, manipulation | SAC | Sample efficiency from replay buffer |
| Continuous control, need entropy/robustness | SAC | Max-entropy objective is inherently robust |
| Need rock-solid stability, simple tuning | PPO | Clipped surrogate is forgiving |

This is the same on-policy/off-policy trade the rest of the series develops; the robotics-specific twist is just that GPU simulation made cheap samples so abundant that the sample-efficiency advantage of SAC stopped being the deciding factor for locomotion.

## 12b. Worked example: Training ANYmal to traverse rough terrain

Sections 4 through 11 gave you the pieces in isolation. This worked example assembles them into one concrete run, so you can see the actual numbers a rough-terrain locomotion policy is built from — the state vector, the reward weights, the two curricula, the hyperparameters, and the transfer results. The target is an ANYmal-C climbing from flat ground to stairs, trained in Isaac Gym on a single A100.

**The setup.** Four thousand and ninety-six parallel environments run on one A100, all resident in GPU memory. The observation is a 48-dimensional proprioceptive vector — base linear and angular velocity (6), the projected gravity vector in the body frame (3), the velocity command (3), joint positions and velocities (12 + 12), the previous action (12), and a pair of clock signals that phase the gait. That 48-vector is stacked over the last three control steps into a **144-dimensional** input, which gives the network the short temporal window it needs to read contact events and incipient slips. The action is 12 joint position targets, expressed as a delta from the default standing pose and scaled by 0.5 before being handed to the PD controller — the delta-from-default parameterization keeps the policy near a sane posture and makes the learning problem about *corrections*, not absolute angles.

**The reward.** A single weighted sum drives the whole behavior:

$$
r = 0.8\,v_{\text{track}} + 0.2\,\omega_{\text{track}} - 0.001\,\lVert \tau \rVert^2 - 0.005\,\lVert v_z \rVert^2 - 0.5\,\text{stumble}.
$$

Velocity tracking carries 0.8 of the weight because following the command *is* the task; yaw-rate tracking gets 0.2; the torque-squared penalty (0.001) buys energy efficiency and smoothness; the vertical-velocity penalty (0.005) suppresses the bouncing that wrecks a gait on stairs; and the stumble penalty (0.5) is the heavy term that teaches clean foot placement. Notice the stumble weight is fully half the velocity-tracking weight — on rough terrain, *not tripping* is nearly as important as moving, and underweighting it produces a fast policy that face-plants on the first step edge.

**The two curricula.** Terrain difficulty advances on a curriculum gated by **episode length, not reward** — a subtle but load-bearing choice, because reward can be gamed by a policy that stands still and collects the alive bonus, whereas a long episode genuinely means the robot stayed up and kept moving. The terrain ladder runs flat ground → 15° slope → rough terrain with 0.02 m bumps → stairs with 0.08 m steps, and an environment only promotes when its episodes get long enough on the current level. Running in parallel is the domain-randomization curriculum: friction starts at $U(0.5, 1.5)$ and widens to $U(0.3, 3.0)$; motor strength starts at $U(0.9, 1.1)$ and widens to $U(0.8, 1.2)$; and a base-mass perturbation of $\pm 2$ kg is widened to $\pm 5$ kg added at 200k steps. The two curricula compound: as the terrain gets harder, the physics it must survive on gets more uncertain.

**The hyperparameters.** Standard PPO, tuned for the massive-parallel regime: learning rate 3e-4, clip range 0.2, entropy coefficient 0.01 to keep the gait exploratory, 4 mini-batches, 5 inner epochs per update, GAE $\lambda = 0.95$, and $\gamma = 0.99$. None of these are exotic — the parallelism does the heavy lifting, not clever optimization.

**The training budget.** The run is 1,500 iterations, each collecting 4,096 environments × 24 steps = 98,304 transitions, for a total of about **147 million environment steps**. On the A100 that is roughly **45 minutes** of wall-clock. This is the "Learning to Walk in Minutes" regime made concrete: a deployable rough-terrain policy in under an hour, on one GPU, because 4,096 environments turn a multi-day step budget into a coffee break.

**The results — and the lesson hiding in them.** In simulation the trained policy scores 95% success on flat ground, 88% on the 15° slope, and 72% on stairs. After an RMA-style adaptation fine-tune and transfer to the real ANYmal-C, it scores 85% flat, 77% slope, and 61% stairs on hardware. Stack those side by side and a pattern jumps out: flat ground retains $85/95 = 89\%$ of its sim performance on hardware, and stairs retain $61/72 = 85\%$ — almost the same retention ratio across radically different terrain. The naive expectation is that the hardest terrain should suffer the *worst* transfer collapse, that the gap should yawn open on stairs where every contact is marginal. It does not. Domain randomization holds up roughly *proportionally* across terrain types, because the thing it makes robust — the actuator response, the friction tolerance, the contact handling — is the same machinery whether the foot lands on flat linoleum or a stair edge. The terrain changes what the policy must *do*; DR governs how well *whatever it does* survives the reality gap, and that survival rate is a property of the randomization, not the terrain. That near-constant 85–89% retention is the single most reassuring number in a sim-to-real run: it means your gap is dominated by physics modeling you can attack with DR, not by terrain-specific surprises you cannot.

## 12c. The hardware-in-the-loop paradigm

Pure sim-to-real treats the real robot as a *test set* — you train entirely in simulation and the hardware only ever grades the result. A growing line of work treats the robot instead as a small, expensive *training signal*, partially closing the gap by feeding real rollouts back into training. This is the **hardware-in-the-loop** paradigm, and it is what you reach for when pure-sim training has plateaued and the last few percentage points live in physics your simulator simply does not capture.

The mechanism is a mixing ratio. You keep training in simulation as before, but you periodically run rollouts on the real robot, extract the transition tuples $(s, a, r, s')$ from those rollouts, and blend them into the experience the policy learns from — a typical recipe is **5% real, 95% sim**. The real transitions are precious and few, but they carry the one thing sim cannot fake: the actual contact dynamics, the actual motor response, the actual sensor noise of *this* robot on *this* floor. One concrete instantiation is the RILI (Real-Image-from-Lab-Interaction) approach: record roughly 10-minute real hardware rollouts, extract the transitions, and mix them into the PPO buffer alongside the flood of simulated experience. The simulated 95% keeps the policy from overfitting to the tiny real sample; the real 5% anchors it to ground truth.

Running rollouts on real hardware during training is dangerous in a way sim never is, so the safety envelope tightens. The torque limit drops to 40 Nm during real rollouts even though the sim permits 80 Nm — you deliberately hobble the robot so a bad action bruises rather than breaks it. Episodes terminate the instant the base makes contact with the ground, cutting a fall short before it becomes a crash. And a human supervisor stands by with a kill switch through every real rollout. These constraints cost performance — a 40 Nm ceiling means the real rollouts explore a gentler slice of the action space than sim — but they are the price of putting a learning policy on real hardware at all.

The payoff shows up clearly on contact-rich manipulation. On a peg-in-hole task, a pure-sim policy reaches 63% success; mixing in just 5% real experience lifts that to 78% — a 15-percentage-point gain. The striking part is the *cost* of that gain: it took only 50k real steps against 10M sim steps. Fifty thousand real transitions, a rounding error next to the simulated budget, bought 15 points of success rate, because those transitions carried exactly the contact information the simulator was getting wrong. That asymmetry — tiny real budget, large effect — is the whole argument for hardware-in-the-loop: real data is expensive per sample but extraordinarily information-dense for the specific gaps sim leaves open.

The tradeoffs are real and they are why this is not the default. Real rollouts require robotics infrastructure most teams running pure-sim do not have — a lab, dedicated robot time, a supervisor — and they add a second, fragile loop to a training pipeline that was otherwise a single GPU job. And they risk hardware damage during the early, flailing phase of training, which is exactly why you do not start here. The discipline is to use hardware-in-the-loop *after* pure-sim training has reached a plateau, when the policy is already competent and the real rollouts are refinements rather than random thrashing. A useful refinement on top of that is a **progressive real-ratio curriculum**: start at 0% real, and increase toward 10% only as the policy stabilizes — which keeps the breakable, expensive real robot out of the loop until the policy is safe enough to put on it.

There is a clean conceptual link back to teacher-student distillation here. Hardware-in-the-loop for vision policies is, structurally, DAgger run on the real robot — except the "expert" being queried is the simulator's privileged teacher policy. You roll out the deployable student on real hardware, and the privileged teacher (which in sim had access to ground-truth state) supplies the corrective labels. The real rollouts generate the student's true on-hardware state distribution, and the teacher tells it what to have done. The same decoupling that made privileged distillation work in pure sim — well-conditioned expert, supervised student — carries straight over to the hardware-in-the-loop setting, with the real robot supplying the state distribution that sim could not.

## 12d. Beyond locomotion: dexterous manipulation and in-hand rotation

Everything to this point leaned on locomotion, because legged locomotion is where sim-to-real first worked at scale. But the harder, more humbling benchmark is **dexterous manipulation** — and specifically in-hand object rotation, the task behind OpenAI's Dactyl and the Rubik's cube. The reason manipulation is harder than locomotion is structural: in locomotion, contact is roughly continuous and predictable — a foot is either on the ground or in swing — whereas in manipulation, contact is **sparse and discontinuous**, fingers making and breaking contact with an object dozens of times a second, each break a discontinuity that the dynamics and any gradient through them have to swallow.

The Dactyl recipe scales the locomotion playbook up and adds one crucial twist. The hand has many fingers driving a 24-dimensional action space, trained across roughly 10,000 parallel environments in Isaac Gym — even more parallelism than locomotion, because the task is harder to learn. The twist is **Automatic Domain Randomization (ADR)**, which we met in section 5: rather than hand-fixing the randomization ranges, ADR self-tunes them, expanding a range whenever the policy's success rate at the current boundary crosses a high threshold (around 60%) and shrinking it when success falls. The feedback loop is per-dimension: ADR tracks success rate for each randomized parameter independently, widens the ranges the policy has mastered, and holds back the ones it has not, so the training distribution grows exactly as fast as competence allows and no faster.

The numbers tell the sim-to-real story starkly. Pure PPO with no ADR reaches 100% success in sim on the full rotation task — and collapses to **20%** on the real hand. That 80-point cliff is the un-randomized knife-edge policy from section 2, in its most brutal form: a policy that perfectly exploits the simulator and falls off the moment reality differs. Add ADR and the picture changes: 82% sim, **60%** real — lower in sim (you paid the robustness tax) but *triple* the real-world success, after 47 days of compute. That 47-day figure is not a typo; aggressive ADR on a 24-DOF hand is one of the most compute-intensive results in the field, which is itself part of the lesson about how much harder manipulation is than locomotion.

One architectural trick recurs in every strong manipulation result: **memory**. An LSTM over a 32-step proprioceptive history captures the object's pose *implicitly*, and it outperforms explicit object-pose estimation from noisy vision. This is the same insight as the locomotion student encoder, sharpened — rather than fighting to estimate where the cube is from imperfect camera images, the policy infers the object's state from how the fingers have been feeling it, the way you can track a coin tumbling in your fist with your eyes closed. The history *is* the state estimator.

The deeper reason manipulation suffers a larger sim-to-real gap than locomotion comes down to contact sensitivity. Contact forces in manipulation are roughly **10× more sensitive to friction variation** than whole-body locomotion forces are. A slightly-wrong friction coefficient barely perturbs a quadruped's gait but completely changes whether a finger can hold or rotate an object. That sensitivity demands finer DR granularity: you randomize friction *per finger-pad*, not the single per-foot coefficient that suffices for locomotion, because each contact patch is its own delicate negotiation and a single global friction sample is too coarse to capture the variation that matters.

The current frontier pushes past proprioception into touch. **Tactile sensors** like GelSight close the gap further by giving the policy a high-resolution readout of the actual contact geometry and force — the one signal that sim approximates worst and that matters most for manipulation. Tactile RL with real sensors is an active research area; Chen et al. (2023) demonstrated tactile sim-to-real for in-hand rotation reaching 71% real success, ahead of the vision-and-proprioception-only results, precisely because tactile sensing observes the contact events that are the crux of the task. The arc is the same one this whole post traces — find the channel where sim lies most, and either randomize over the lie or measure the truth directly — applied to the hardest sensing problem in robotics.

## Key takeaways

- **The gap is the objective.** You optimize $J_{\text{sim}}$ but you are graded on $J_{\text{real}}$. Optimizing harder in one fixed simulator produces a brittle knife-edge policy. Train under a *distribution* of dynamics instead: $J_{\text{DR}} = \mathbb{E}_{\xi \sim \rho}[\,\ldots]$.
- **Randomize the actuator, not just the rigid body.** The single biggest locomotion-transfer lever is modeling the real motor — latency, bandwidth, an actuator network. Randomizing friction and mass alone does not close the gap.
- **Domain randomization is a dial, not a switch.** Too narrow and reality falls outside the training band; too wide and the policy collapses into sluggish conservatism. ADR widens the band only as fast as the policy can absorb.
- **Privileged distillation beats direct POMDP training.** Train a teacher with simulator ground truth, then distill into a student that uses only deployable observations via DAgger. It decouples the well-conditioned control problem from the hard inference problem.
- **History is implicit system identification.** An LSTM or conv encoder over proprioceptive history (RMA, the student encoder) learns to read friction, mass, and terrain from how the robot has been responding — online adaptation with zero deployment-time optimization.
- **Reward weights are where projects fail.** A too-large energy penalty yields a frozen robot; a too-small action-rate penalty yields a gearbox-destroying jitter. Shape densely, normalize, and tune the weights as carefully as the algorithm.
- **Respect the latency budget.** A 20 ms control loop forces small networks and on-board inference, and the simulator must model the latency the robot will actually have.
- **PPO for massive-parallel locomotion, SAC for sample-scarce manipulation.** Cheap GPU samples make PPO's stability and parallelism win for locomotion; SAC's replay-buffer efficiency wins when samples are precious.
- **Sim-to-real RL is a tool, not a mandate.** If you can write the dynamics down, use classical control. The most robust shipped robots are hybrids.

## Further reading

- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (2017) — the paper that launched DR for vision.
- Hwangbo, Lee, Hutter et al., "Learning Agile and Dynamic Motor Skills for Legged Robots" (*Science Robotics*, 2019) — the actuator-network result that made locomotion transfer work.
- Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain" (*Science Robotics*, 2020) — teacher-student privileged distillation for ANYmal.
- OpenAI et al., "Solving Rubik's Cube with a Robot Hand" (2019) — Automatic Domain Randomization and dexterous manipulation transfer.
- Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots" (2021) — real-time online adaptation from proprioceptive history.
- Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep RL" (2021) — Isaac Gym, 4096 envs, four-minute training.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) and Haarnoja et al., "Soft Actor-Critic" (2018) — the two algorithms underlying everything above.
- Within this series: the [PPO deep-dive](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), [DDPG and TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3), [actor-critic methods](/blog/machine-learning/reinforcement-learning/actor-critic-a2c-a3c), and the [model-based vs model-free decision guide](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which). This post is the robotics application of the same agent–environment–reward loop the series is built on; the forthcoming unified map and the playbook capstone tie the full taxonomy together.

---
title: "RL Engineering: Building Production-Grade Reinforcement Learning Systems"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How to take RL beyond Jupyter notebooks — training infrastructure, experiment tracking, policy serving, rollback systems, and the monitoring pipelines that keep production RL agents from silently failing."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "mlops",
    "ray",
    "rllib",
    "policy-serving",
    "monitoring",
    "machine-learning",
    "pytorch",
    "rlhf",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/rl-engineering-production-systems-1.png"
---

The proudest moment of my RL career was watching a PPO agent hit a 487/500 average return on CartPole. The most humiliating was three weeks later, watching a structurally identical agent — same algorithm, same hyperparameters, more compute — quietly degrade in production from a 91% task-success rate to 58% over the course of a long weekend, with not a single alert firing the entire time. Nobody noticed until a customer did. By the time I pulled the logs, I could not even reconstruct which model version had been serving traffic, because the "deployment process" was an engineer SSH-ing into a box and copying a `.pth` file over the old one.

That second story is what this post is about. There is a canyon between an RL algorithm that *works* and an RL system that *keeps working*, and almost nothing in the research literature prepares you for it. A toy PPO is maybe 200 lines: an environment loop, a rollout buffer, a clipped surrogate objective, an Adam step. It is a beautiful, self-contained artifact, and it teaches you the *theory*. But the theory is the easy 20%. The other 80% — the part that determines whether your agent makes money or loses it, helps users or harms them — lives in four systems that no `train.py` ever mentions: **experiment management** (so you can tell which of your 400 runs actually worked and reproduce it), **safe deployment** (so a bad policy never sees 100% of traffic before you know it is bad), **real-time monitoring** (so a silently degrading agent trips an alarm instead of a customer), and **rollback** (so when it all goes wrong at 2am, recovery is one command, not an archaeology project).

Figure 1 shows the shape of the thing we are building toward — not a script, but a closed loop where deployment feeds monitoring, and monitoring decides whether the next policy graduates to full traffic or gets rolled back. By the end of this post you will be able to stand up each of those four systems with real tools: distributed training with **Ray** and **RLlib**, experiment tracking with **Weights & Biases** and **MLflow**, policy serving as an **ONNX**/**TorchScript** microservice, canary deployment with automated rollback, and the specific monitoring signals — reward drift, entropy collapse, action-distribution shift — that catch the silent failures the standard MLOps playbook misses.

![Diagram of a production reinforcement learning system as a closed loop from distributed collection through training, a versioned registry, canary rollout, and monitoring that gates full rollout or rollback](/imgs/blogs/rl-engineering-production-systems-1.png)

The spine of this whole series is the same as ever: an agent interacts with an environment, collects rewards, and updates a policy. Everything below is about wrapping that loop in the engineering that lets it run for months, on real traffic, without you holding your breath. If you have not yet internalized the core algorithms, the unified map post (`reinforcement-learning-a-unified-map`) and the capstone (`the-reinforcement-learning-playbook`) are the conceptual bookends; this post is the operational one between them.

## 1. The gap between research RL and production RL

Let me make the gap concrete, because "production-grade" is the kind of phrase that means nothing until you have been burned. Here is research RL, the entire thing, in honest form:

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_cartpole")  # the "deployment artifact"
```

This is genuinely useful. It trains a competent CartPole agent in a couple of minutes. But notice everything it does *not* do. It does not record which random seed produced this result, so when a reviewer asks "is this reproducible?" the honest answer is "I have no idea." It does not log the git commit, so six months from now you cannot reconstruct which version of the reward function this was trained against. It does not version the saved model, so `ppo_cartpole.zip` will be silently overwritten by the next run. It cannot scale past one process, so when you move from CartPole to a real environment that takes 50ms per step, your sample throughput collapses. And it has no notion of *serving* — `model.predict()` in a Python REPL is not an inference endpoint that a downstream service can call with a 20ms latency budget.

The before-and-after in Figure 3 makes the contrast stark. On the left, a single untracked script. On the right, the four systems that turn it into something you can trust with traffic. The transformation is not about making the algorithm better — PPO is PPO. It is about making the *run* reproducible, the *artifact* versioned, the *deployment* reversible, and the *behavior* observable.

![Before-and-after diagram contrasting a single untracked research script against a production system with an experiment tracker, distributed training, a policy registry, and automated rollback](/imgs/blogs/rl-engineering-production-systems-3.png)

Here is the reframing that took me years to absorb: **in production, the policy is the least interesting part of the system.** The interesting parts are the four loops around it. The training loop turns experience into weights. The experiment loop turns weights into a *decision* about which weights to trust. The deployment loop turns a trusted policy into live behavior, slowly enough that you can pull the cord. And the monitoring loop turns live behavior back into the signal that drives the next decision. Get any one of these wrong and the algorithm's quality is irrelevant — a perfect policy you cannot reproduce, cannot serve under latency, or cannot tell is broken is worth exactly nothing.

There is one more reason RL is *harder* to productionize than supervised learning, and it is worth stating up front because it shapes every later section. In supervised learning, your training distribution is fixed: you have a dataset, you fit it, the test set is drawn from the same distribution. In RL, the policy *generates its own data*. The moment you deploy, the policy changes the distribution of states it sees — and if the world also shifts (new users, new market regime, a changed upstream feature), the policy is now acting on a distribution it was never trained on. A supervised model degrades gracefully on distribution shift. An RL policy can degrade *catastrophically and silently*, because there is no held-out accuracy number screaming at you. This is the single deepest reason production RL needs its own monitoring discipline, and we will come back to it hard in Section 7.

There is a second-order effect of the policy generating its own data that catches teams off guard: **feedback loops compound**. In supervised learning a bad prediction is a single bad prediction; the next example is drawn independently from the same fixed distribution, so one error does not make the next error more likely. In RL the action you take *determines the next state you see*, so a policy that drifts slightly toward a degenerate behavior keeps visiting the states that reinforce that behavior. A recommender that over-recommends one category sees more engagement on that category (because that is all it shows), which the learning signal reads as "this is working," which pushes it further toward the degenerate policy. Nothing in the loop says "stop." This positive feedback is why RL failures are not just silent but *accelerating* — the longer they run unchecked, the faster they get worse, which is the precise opposite of the graceful decay you get from a stale supervised model. Every safeguard in this post exists to break that loop before it runs away: monitoring to detect it, canary to bound its reach, rollback to reverse it.

A useful way to hold all of this is that production RL has **two distributions you must keep aligned**: the distribution of states the policy was *trained* on, and the distribution of states it actually *encounters* in production. Research RL implicitly assumes these are identical because the same simulator generates both. Production RL never gets that guarantee for free — the gap between training distribution and serving distribution is the single most common root cause of production RL failures, and almost every technique below (environment versioning, normalization-stat tracking, reward-drift monitoring, shadow mode) is, at bottom, a tool for detecting or shrinking that gap.

## 2. Training infrastructure: actor-learner at scale

The first wall you hit is throughput. A research PPO collects experience in the same process that does gradient updates: step the env, store the transition, repeat until the rollout buffer is full, then update. When the environment is CartPole this is fine — `env.step()` returns in microseconds. When the environment is a physics simulator, a market replay, or a slow API, `env.step()` might take 20–100ms, and your single process spends 95% of its wall-clock time waiting on the environment while the GPU sits idle.

The fix is the **actor-learner architecture**, and it is the single most important infrastructure pattern in production RL. You separate the two jobs onto different hardware: many cheap CPU **actors** whose only job is to run the current policy in their own copy of the environment and ship the resulting experience somewhere, and one (or a few) expensive GPU **learners** whose only job is to consume that experience and update the weights. Figure 2 shows the dataflow: actors fan experience into a shared buffer, the learner samples from it, and the updated weights are broadcast back so the actors stay roughly current.

![Graph diagram of the actor-learner architecture with multiple CPU actors feeding a shared experience buffer that a single GPU learner consumes before broadcasting updated weights](/imgs/blogs/rl-engineering-production-systems-2.png)

### Why this is more than a speed trick: the off-policy correction

Here is the theory the kit demands, because the actor-learner split is not free. The moment you decouple collection from learning, the actors are running a *slightly stale* policy — they collected experience under policy parameters $\theta_{t-k}$ while the learner has already advanced to $\theta_t$. This makes the data **off-policy** with respect to the current learner, and a naive on-policy gradient is now biased.

This is exactly the problem **IMPALA** (Espeholt et al., 2018) solves with **V-trace**. The on-policy $n$-step value target is

$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \delta_t, \qquad \delta_t = r_t + \gamma V(x_{t+1}) - V(x_t),
$$

but when the behavior policy $\mu$ (the actor's stale policy) differs from the target policy $\pi$ (the learner's current policy), each temporal-difference term must be reweighted by a *truncated* importance ratio. V-trace defines

$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \rho_t\, \delta_t,
$$

where $\rho_t = \min\!\left(\bar{\rho}, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\right)$ and $c_i = \min\!\left(\bar{c}, \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)}\right)$ are importance ratios clipped at $\bar{\rho}$ and $\bar{c}$. The clipping is the whole point: $\bar{\rho}$ controls the *fixed point* the value function converges to (how on-policy your value estimate is), while $\bar{c}$ controls the *variance* of the update (how much a single rare action can swing the target). Without this correction, off-policy staleness either biases your value estimates or blows up their variance — and you get the maddening symptom of an agent that trains fine at small scale and diverges the moment you add actors. If your critic variance is exploding when you scale collection, V-trace clipping is the first knob to reach for.

**APPO** (asynchronous PPO) makes a related trade. Vanilla PPO is strictly on-policy: collect a batch, do several epochs of clipped updates on *that* batch, throw it away, collect again. APPO relaxes the "throw it away immediately" rule, letting actors keep collecting while the learner updates, and uses PPO's clipping plus a small amount of importance correction to tolerate the resulting mild off-policy-ness. The practical payoff is that the GPU never waits for collection, and on environments where stepping is the bottleneck you can see 5–10× higher throughput than synchronous PPO at comparable sample efficiency.

### Standing it up with RLlib

You do not implement V-trace by hand. RLlib (built on Ray) gives you actor-learner scaling as configuration. Here is a real APPO setup that uses 16 rollout workers across a cluster and one GPU learner:

```python
from ray.rllib.algorithms.appo import APPOConfig
import ray

ray.init(address="auto")  # connect to an existing Ray cluster

config = (
    APPOConfig()
    .environment("CartPole-v1")
    .env_runners(
        num_env_runners=16,          # 16 parallel actors collecting experience
        num_envs_per_env_runner=8,   # each actor vectorizes 8 envs -> 128 env copies
        rollout_fragment_length=50,
    )
    .learners(num_learners=1, num_gpus_per_learner=1)
    .training(
        lr=5e-4,
        gamma=0.99,
        vtrace=True,                 # the off-policy correction from above
        clip_param=0.2,              # PPO surrogate clip
        grad_clip=40.0,
        train_batch_size=2048,
    )
    .resources(num_cpus_for_main_process=2)
)

algo = config.build()
for i in range(200):
    result = algo.train()
    if i % 10 == 0:
        ckpt = algo.save(checkpoint_dir="/mnt/checkpoints/appo")
        print(f"iter {i} | return {result['env_runners']['episode_return_mean']:.1f} "
              f"| FPS {result['num_env_steps_sampled_throughput_per_sec']:.0f}")
```

The 128 environment copies across 16 actors are what turn a 3,000-FPS single-process run into a 40,000-FPS cluster run. The learner never starves.

### Job scheduling: SLURM and Kubernetes

Ray needs a cluster to run on, and in practice that cluster is provisioned either by **SLURM** (the dominant scheduler in academic and HPC environments) or **Kubernetes** (the dominant scheduler in industry). A minimal SLURM launch that starts a Ray head node and several workers looks like this:

```bash
#!/bin/bash
#SBATCH --job-name=rl-appo
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

# Start the Ray head on the first node, workers on the rest
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --port=6379 --block &
sleep 15

srun --nodes=$((SLURM_NNODES - 1)) --ntasks=$((SLURM_NNODES - 1)) --exclude="$head_node" \
    ray start --address="$head_ip:6379" --block &
sleep 15

python train_appo.py --address "$head_ip:6379"
```

On Kubernetes you would use **KubeRay** (the `RayCluster` custom resource) instead, which gives you autoscaling: the cluster grows actors when the buffer is starved and shrinks them when the learner is the bottleneck. The choice between them is mostly organizational — if your company already runs Kubernetes for services, run RL there too so it shares the same observability and on-call tooling.

### Checkpointing strategy

Long RL runs *will* be preempted — a spot instance dies, a node OOMs, a NCCL collective hangs. Without checkpointing, a 12-hour run that dies at hour 11 costs you 11 hours. The rule I follow: **save every N steps, keep the last K checkpoints plus the all-time-best-by-eval.** Concretely, for a run targeting 50M steps I save every 500k steps (100 checkpoints over the run), keep the last 5 on fast local disk for crash recovery, and promote the single best-by-eval-return checkpoint to durable object storage as a *candidate* for the policy registry. Keeping only the last K bounds your disk; keeping the best-by-eval ensures a late-run collapse (which happens — see Section 7) does not destroy your good policy.

A checkpoint that only saves model weights is not enough to *resume* — you also need the optimizer state (Adam's first and second moments), the current step count, the random-number-generator states, and any running statistics (the normalization mean/var from Section 4). Resuming from weights alone restarts the optimizer cold and resets the learning-rate schedule, which produces a visible discontinuity in the return curve and sometimes a transient collapse. A resumable checkpoint looks like this:

```python
import torch, os, glob

def save_checkpoint(path, step, model, optimizer, vecnorm_stats, rng_state):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),   # Adam moments — needed to resume cleanly
        "vecnorm": vecnorm_stats,                 # running obs mean/var
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, f"{path}/ckpt_{step}.pt")
    # Retention: keep only the last K=5 to bound disk
    ckpts = sorted(glob.glob(f"{path}/ckpt_*.pt"), key=os.path.getmtime)
    for old in ckpts[:-5]:
        os.remove(old)

def resume(path, model, optimizer):
    latest = max(glob.glob(f"{path}/ckpt_*.pt"), key=os.path.getmtime)
    ck = torch.load(latest)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optim_state"])
    torch.set_rng_state(ck["torch_rng"])
    return ck["step"], ck["vecnorm"]
```

The single most common checkpointing bug I see is forgetting the optimizer state, and it is insidious because the resumed run *looks* like it works — it just silently trains worse than an uninterrupted run would have. Always verify a resume by checking that the return curve is continuous across the resume point.

#### Worked example: throughput math that justifies the cluster

Suppose your environment steps in 25ms (a moderate physics sim). A single-process PPO collecting one transition at a time achieves at most $1/0.025 = 40$ steps/second per env. With one vectorized process running 8 envs you get ~320 FPS, but the GPU is idle 90%+ of the time. Now run 16 actors × 8 envs = 128 environment copies. Even at 70% efficiency (network and serialization overhead are real), that is $128 \times 40 \times 0.7 \approx 3{,}580$ FPS of *collection*, and because the learner runs asynchronously on its own GPU it consumes that stream continuously. To reach 50M steps: single-process at 320 FPS is $50{,}000{,}000 / 320 \approx 43.4$ hours; the 128-copy cluster at 3,580 FPS is $50{,}000{,}000 / 3{,}580 \approx 3.9$ hours. The cluster costs maybe 6× the hourly rate but finishes 11× faster — so it is both faster *and* cheaper per run, the rare win-win. This is why nobody trains serious RL single-process.

## 3. Experiment tracking and the RL graveyard

If you train RL for a living, you will run hundreds of experiments, and here is the brutal statistic from my own logs and from every team I have compared notes with: **roughly 90% of RL runs produce uninterpretable results.** Not failed results — *uninterpretable* ones. The run crashed at step 2M and you do not know why. The return curve is noisy garbage and you cannot tell if it is a real signal or seed variance. You found a great result three weeks ago but cannot reproduce it because you tweaked the reward function in between and did not record the change. I call this the **RL experiment graveyard**, and the only thing that turns it into a usable record is disciplined tracking.

The two tools are **Weights & Biases** (hosted, gorgeous dashboards, the default for most teams) and **MLflow** (self-hostable, integrates with a model registry). They do the same core job: capture metrics over time plus the metadata needed to reproduce a run.

### What to log

There is a specific, non-obvious list of signals for RL, and skipping any of them will eventually cost you a debugging session. Log all of these every iteration:

| Signal | What it tells you | Healthy looks like |
|---|---|---|
| Episode return (mean, max, min) | Is the agent getting better? | Rising, then plateauing |
| Episode length | Solving faster or surviving longer? | Task-dependent; sudden drops are bad |
| Policy entropy | Is the policy still exploring? | Slowly decreasing, never collapsing to ~0 early |
| Value loss | Is the critic tracking returns? | Decreasing, then stable and small |
| Policy loss | Magnitude of policy updates | Small and stable; spikes signal instability |
| Gradient norm | Are updates exploding? | Bounded; sustained growth means divergence |
| KL divergence (old→new policy) | How far each update moves the policy | Near a target (e.g. 0.01–0.02) for PPO |
| Wall-clock FPS | Throughput health | Stable; drops mean an infra problem |
| GPU utilization | Is the learner starved? | High (>80%); low means actors can't keep up |

Notice how many of these are not "is the agent good" but "is the *training process* healthy." Entropy, KL, and gradient norm are diagnostic: they tell you *why* the return curve is doing what it is doing. An entropy that collapses to near-zero at step 1M tells you the policy stopped exploring and any plateau after that is premature convergence, not success. A KL that periodically spikes to 0.5 tells you your learning rate is too high and updates are occasionally catastrophic.

### Reproducibility metadata: the part nobody does until it bites them

Metrics are useless without the context to reproduce them. The minimum reproducibility bundle for every run:

```python
import wandb, subprocess, torch, numpy as np, random

def get_git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

config = {
    "algo": "APPO",
    "env_id": "CartPole-v1",
    "seed": 42,
    "lr": 5e-4, "gamma": 0.99, "clip_param": 0.2,
    "num_env_runners": 16, "train_batch_size": 2048,
    "git_sha": get_git_sha(),                 # exact code version
    "gym_version": gym.__version__,           # env package version (see Section 4)
    "torch_version": torch.__version__,
}

# Seed EVERYTHING for reproducibility
random.seed(config["seed"]); np.random.seed(config["seed"])
torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])

run = wandb.init(project="prod-rl", config=config, tags=["appo", "cartpole"])

for i in range(200):
    result = algo.train()
    wandb.log({
        "return_mean": result["env_runners"]["episode_return_mean"],
        "entropy": result["info"]["learner"]["default_policy"]["entropy"],
        "value_loss": result["info"]["learner"]["default_policy"]["vf_loss"],
        "policy_loss": result["info"]["learner"]["default_policy"]["policy_loss"],
        "grad_norm": result["info"]["learner"]["default_policy"]["grad_norm"],
        "fps": result["num_env_steps_sampled_throughput_per_sec"],
    }, step=result["num_env_steps_sampled_lifetime"])
```

The four fields that save your life are `seed`, `env_id`, `git_sha`, and the full hyperparameter dict. Seed alone does not guarantee bit-identical RL runs — GPU non-determinism in some CUDA kernels, and especially *asynchronous* actor-learner timing, mean two runs with the same seed can diverge. But seed + git SHA + full config + env version gets you *statistically* reproducible (same distribution of outcomes over a few seeds), which is what actually matters for trusting a result. The honest standard for an RL claim is "median over 5 seeds with this exact config and code SHA," not "I ran it once and it was great."

#### Worked example: a run rescued by metadata

Last year a teammate reported a SAC agent that hit a Sharpe of 1.8 on a trading-env backtest, double our previous best. We could not reproduce it — fresh runs landed around 0.9. Because every run logged its git SHA, we checked out the exact commit from the good run and diffed it against `main`. The difference: an uncommitted local change had set the reward to use *log* returns instead of simple returns, which happened to interact favorably with that backtest window. The "1.8 Sharpe breakthrough" was a reward-function bug, not a real improvement — and crucially, *not* something that would generalize. Without the git SHA in the log, we would have spent a week chasing a phantom and possibly shipped a policy tuned to a bug. The metadata did not just enable reproduction; it prevented a bad deployment. This is the entire value of fighting the graveyard.

## 4. Environment management and versioning

Here is a failure mode that feels too dumb to be real until it happens to you: you upgrade `gymnasium` from 0.28 to 0.29, rerun your training script, and your agent's returns drop by 40% — with no code change of your own. The reason is that the environment *is* part of your training distribution, and a different package version can mean different physics constants, a different observation normalization, a changed reward scale, or a fixed bug that your policy had implicitly learned to exploit. **The environment package version must be pinned exactly for reproducibility**, the same way you pin a dataset hash in supervised learning. Pin `gymnasium==0.29.1` and `mujoco==3.1.6` in your lockfile and log them with every run (as in the config above).

### Vectorization: SubprocVecEnv vs DummyVecEnv

To feed an actor with many env copies you vectorize. There are two strategies and the choice is a real performance decision:

| | `DummyVecEnv` | `SubprocVecEnv` |
|---|---|---|
| How | All envs in one process, stepped in a loop | Each env in its own subprocess, stepped in parallel |
| Best for | Fast envs (CartPole), where IPC overhead > step cost | Slow envs (MuJoCo, sims), where stepping dominates |
| Overhead | None, but no true parallelism | Pickling + pipe IPC per step |
| Rule of thumb | Use when step < ~1ms | Use when step > ~5ms |

The mistake is using `SubprocVecEnv` for a fast environment: the inter-process communication cost (pickling observations, piping them between processes) exceeds the time you save by parallelizing, and you go *slower*. For CartPole, `DummyVecEnv` wins. For a 30ms physics sim, `SubprocVecEnv` wins decisively.

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Slow env: parallelize across processes
vec_env = make_vec_env("HalfCheetah-v4", n_envs=8, vec_env_cls=SubprocVecEnv)

# Fast env: stay in-process
vec_env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=DummyVecEnv)
```

### Wrappers: the unsung reproducibility hazard

RL agents are extraordinarily sensitive to input scaling, so you almost always stack **wrappers**: observation normalization (running mean/std), reward normalization or clipping, frame stacking for vision. These wrappers carry *state* — the running mean and variance — and that state is part of the policy. If you train with `VecNormalize` and then serve the raw policy without applying the same normalization, the policy receives observations in a distribution it never saw, and it behaves like a different (broken) agent. This is one of the most common "it worked in training, it's garbage in production" bugs.

```python
from stable_baselines3.common.vec_env import VecNormalize

vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
model = PPO("MlpPolicy", vec_env).learn(1_000_000)

# CRITICAL: save the normalization statistics alongside the policy.
model.save("policy.zip")
vec_env.save("vecnormalize.pkl")   # the running mean/std — part of the artifact!
```

The rule: **the normalization statistics are part of the policy artifact and must be versioned with it.** A policy checkpoint without its `vecnormalize.pkl` is not a deployable artifact; it is a trap.

### The environment registry pattern

When you have custom environments — a trading env, a recommendation env, a robotics env — manage them through Gymnasium's registry rather than importing classes directly, and version them explicitly in the ID:

```python
from gymnasium.envs.registration import register

register(
    id="TradingEnv-v3",                          # version in the ID itself
    entry_point="envs.trading:TradingEnvV3",
    max_episode_steps=2048,
    kwargs={"fee_bps": 2.0, "slippage_model": "linear"},
)
```

Bumping the version (`-v3` → `-v4`) every time you change the environment's dynamics, observation space, or reward means a logged `env_id` of `TradingEnv-v3` unambiguously identifies the world the policy was trained in. This is the environment equivalent of semantic versioning, and it is what makes "reproduce run #4471" a tractable request a year later.

## 5. Policy serving and inference optimization

A trained policy is a function from observation to action, and serving it means exposing that function as a service something else can call. The naive approach — load the SB3 model in a Flask handler and call `model.predict()` — works for a demo and falls over in production for three reasons: it carries the entire training framework into your serving image, it cannot batch, and Python-level overhead blows your latency budget.

### Export: TorchScript and ONNX

The first move is to strip the policy down to just the forward pass and export it to a framework-independent format. **TorchScript** serializes the PyTorch graph so it runs without the Python training code; **ONNX** goes further, producing a graph that runtimes like ONNX Runtime or NVIDIA Triton can execute with their own optimized kernels.

```python
import torch

policy_net = model.policy        # the trained nn.Module
policy_net.eval()
dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)

# Option A: TorchScript
scripted = torch.jit.trace(policy_net, dummy_obs)
scripted.save("policy_scripted.pt")

# Option B: ONNX (dynamic batch axis so we can batch at serve time)
torch.onnx.export(
    policy_net, dummy_obs, "policy.onnx",
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,
)
```

Crucially, **bake the observation normalization into the exported graph** (or apply it identically in the serving wrapper) so the serving boundary matches training exactly — the Section 4 hazard, prevented at the export step.

### Batching for GPU efficiency

A GPU is wildly underutilized serving one observation at a time. The single most effective serving optimization is **dynamic batching**: accumulate incoming requests for a few milliseconds, run them through the network as one batch of, say, 32, and scatter the actions back. NVIDIA Triton does this for you with a config:

```bash
# config.pbtxt for Triton
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 2000   # wait up to 2ms to fill a batch
}
```

That 2ms max queue delay is the central trade-off knob: longer waits build bigger batches (higher throughput) at the cost of latency. The right value depends entirely on your latency budget.

### Latency budgets are not all equal

| Use case | Latency budget | Implication |
|---|---|---|
| High-frequency trading | < 1ms | No GPU round-trip; CPU INT8 or FPGA, tiny network |
| Online ad/recsys ranking | 10–50ms | GPU batching fine; ONNX Runtime on CPU often enough |
| Robotics control loop | ~20ms (50Hz) | On-device inference; TorchScript + quantization |
| Game AI / async decisions | 100ms+ | Comfortable; full network, batching trivial |

A policy that needs a 1ms response cannot afford a network hop to a GPU server — the round trip alone eats the budget — so HFT policies are deliberately kept small enough to run as a quantized model on the same CPU as the trading logic. A robotics policy running a 50Hz control loop has ~20ms per decision and runs on-device. Knowing your budget *before* you design the network is the difference between a deployable policy and a research artifact that is too slow to use.

### Quantization for the edge

For edge and ultra-low-latency deployment, **INT8 quantization** shrinks the model ~4× and speeds up inference 2–4× on supporting hardware, by representing weights and activations as 8-bit integers instead of 32-bit floats:

```python
import torch.ao.quantization as tq

policy_net.eval()
policy_net.qconfig = tq.get_default_qconfig("fbgemm")  # x86 CPU backend
tq.prepare(policy_net, inplace=True)
# calibrate on representative observations so the quantizer learns ranges
for obs_batch in calibration_loader:
    policy_net(obs_batch)
quantized = tq.convert(policy_net, inplace=False)
```

The honest caveat: RL policies can be *more* sensitive to quantization than classifiers, because a small numerical perturbation can flip an `argmax` over actions and change behavior in ways that compound over an episode. Always re-evaluate the quantized policy's *task return* (not just output MSE) before shipping — a 0.5% output error can be a 10% return drop.

### The tooling landscape

A recurring question from teams starting out is "which tool do I use for this?" The honest answer is that no single tool spans the whole production RL stack — you assemble it from pieces, and only some of those pieces are RL-native. Figure 5 lays out the six load-bearing tools by job, scale, RL-nativeness, and production-readiness.

![Matrix comparing six production reinforcement learning tools across purpose, scale, RL-native support, and production-readiness, showing that only RLlib is genuinely RL-native](/imgs/blogs/rl-engineering-production-systems-5.png)

The split is clean once you see it. **Ray** is general distributed compute (the substrate); **RLlib** is the only genuinely RL-native piece, owning distributed training with actor-learner scaling built in. **WandB** and **MLflow** own tracking and registry — neither knows anything about RL specifically, which is why the RL-specific logging discipline of Section 3 is *your* job, not theirs. **ONNX** owns model export and **Triton** owns high-throughput serving — again, framework-agnostic, indifferent to whether the model is a classifier or a policy. The practical consequence: you will glue these together, and the glue (the RL-aware logging, the normalization-stat versioning, the policy-specific monitoring) is where your engineering value actually lives. The tools solve the generic 80%; the RL-specific 20% is what you must build.

## 6. Safe deployment: shadow mode and canary

You have a trained, versioned, exported, fast policy. You must never let it touch 100% of traffic before you know it is at least as good as what it replaces. There are two patterns, and you usually use them in sequence.

### Shadow mode

In **shadow mode**, the new policy runs alongside the current production policy on the *same live inputs*, but its actions are logged and discarded — only the old policy's actions are actually executed. This is the zero-risk way to answer "what *would* the new policy have done?" before it does anything. You compare the shadow policy's action distribution and predicted value against the live policy, and you catch the obvious disasters (the new policy wants to sell everything, or always outputs the same action) without a single real consequence.

```python
def serve_request(obs):
    live_action = live_policy.predict(obs)        # this one actually executes
    with torch.no_grad():
        shadow_action = shadow_policy.predict(obs) # logged, NOT executed
    log_shadow_comparison(obs, live_action, shadow_action)
    return live_action
```

Shadow mode is mandatory when **constraint violations must be zero** — a robot that could damage hardware, a trading policy that could take a forbidden position. You let it shadow for long enough to see the full range of real states, verify it never *would* have violated a constraint, and only then let it act.

### Canary deployment

Once shadow mode says the policy is sane, you **canary**: route a small slice of real traffic (start at 1%) to the new policy and compare its metrics against the control group still served by the old policy. This is a live A/B test, and the comparison must be statistical — you are looking for a significant difference in the business metric, not eyeballing a noisy curve.

```python
import hashlib

def route(user_id, canary_fraction=0.01):
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 10000
    return "canary" if h < canary_fraction * 10000 else "control"

policy = canary_policy if route(user_id) == "canary" else control_policy
action = policy.predict(observe(user_id))
log_outcome(user_id, group=route(user_id), action=action)  # for A/B analysis
```

Hashing the user ID (rather than rolling a die per request) gives a *stable* assignment — the same user always sees the same policy, which you need for any per-user metric like retention.

### The monitoring baseline and the rollback criterion

A canary is only as good as the baseline you compare it against. **Before** you deploy anything new, you must record what "good" looks like for the control policy: the mean and variance of episode return, the constraint-violation rate, the p99 latency. Then the rollback criterion is a concrete, *automated* rule, not a judgment call at 2am. The one I use as a default:

> Roll back automatically if the canary's primary metric drops more than 10% below the control baseline, sustained for more than 1 hour.

The "sustained for more than 1 hour" clause is essential — without it, normal hourly variance trips false rollbacks constantly. With it, you tolerate noise but catch real regressions. Section 10 turns this criterion into an actual rollback system; here the point is that the *threshold and the duration must be decided in advance and wired to automation*, because the entire purpose of canary is to remove the human from the failure-detection loop.

#### Worked example: a canary that caught a 14% regression

We canaried a new recommendation policy at 1% traffic. The control baseline was a 4.2% click-through rate (CTR) with hourly standard deviation around 0.3%. Within two hours the canary's CTR was 3.6% — a 14% relative drop, well outside noise, and sustained past the one-hour window. The automated rule rolled it back to control at 1% exposure, so the *blended* CTR impact across all traffic was $0.01 \times (-14\%) = -0.14\%$ for two hours: a rounding error. Post-mortem found the new policy had an off-by-one in feature alignment that fed it stale features. Had we shipped it to 100% directly, that 14% CTR drop across all traffic, for however many hours until a human noticed, would have been a serious revenue event. The canary turned a potential incident into a non-event. That asymmetry — small blast radius during the risky window — is the entire value proposition of staged rollout.

## 7. Monitoring production RL agents

This is the section that, had I read it three years ago, would have saved me the humiliation from the intro. Standard MLOps monitoring — request latency, error rate, throughput — tells you the *service* is up. It says nothing about whether the *policy* is still good. RL needs its own monitoring discipline, and it is built on a stack of four layers, shown in Figure 4: infrastructure metrics at the bottom, then algorithm metrics, then reward metrics, then the business metric at the top. The power of stacking them is *traceability*: when CTR drops at the top, you walk down the stack — is it a reward-distribution shift? An entropy collapse? A starved GPU? — instead of guessing.

![Stack diagram of the four-layer RL monitoring hierarchy from infrastructure metrics up through algorithm and reward metrics to the top-level business metric](/imgs/blogs/rl-engineering-production-systems-4.png)

### Signal 1: reward distribution drift

After training converges, the distribution of rewards the deployed policy earns should be **stationary** — roughly the same shape week over week. When it drifts, something in the world changed: a new user population, a market-regime shift, a changed upstream feature. The detection is statistical. Track the rolling distribution of per-episode return and run a two-sample test against a reference window from right after deployment:

```python
from scipy.stats import ks_2samp
import numpy as np

reference_returns = np.load("reference_returns.npy")  # captured post-deploy

def check_reward_drift(recent_returns, alpha=0.01):
    stat, pvalue = ks_2samp(reference_returns, recent_returns)
    drifted = pvalue < alpha
    return {"ks_stat": float(stat), "pvalue": float(pvalue), "drift": bool(drifted)}
```

A Kolmogorov-Smirnov test flags when the recent return distribution is significantly different from the reference. Drift is not automatically *bad* — but it always means "the assumptions your policy was trained under no longer hold," and it is the earliest warning that a silent failure may be starting.

### Signal 2: policy entropy collapse

Policy entropy $H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$ measures how much the policy is still exploring. During healthy training it decreases slowly as the policy becomes confident. In production it should be roughly *stable*. A sudden **entropy collapse** — entropy plunging toward zero — means the policy has degenerated to near-deterministic, usually because it found a narrow exploit or because a feedback loop (the policy's own actions changing its future inputs) has driven it into a corner. An entropy collapse in production almost always precedes a return collapse, which makes it one of the best *leading* indicators you have. Alert on entropy dropping below a fraction (say 50%) of its post-training baseline.

### Signal 3: action-distribution monitoring

Track the distribution of the actions the agent actually takes and compare it against the expected range from evaluation. A trading agent that suddenly trades 5× more often, a recsys agent that collapses to recommending the same 10 items, a robot whose torque commands creep toward saturation — these all show up as action-distribution shift before they show up as a business-metric drop. Concretely, log the per-action histogram (discrete) or the mean and standard deviation per action dimension (continuous) every hour and alert on a significant deviation from the evaluation-time distribution.

### The silent failure problem

Here is the deep reason all of this matters, and it ties back to the distinction from Section 1. A supervised model has a held-out accuracy you can recompute on fresh labeled data — degradation is *visible*. An RL agent in production has no labels. Its only ground truth is the reward it earns, and reward can degrade *gradually*, staying inside every individual alerting threshold the whole way down, while no single metric ever crosses a hard line. This is the **silent failure**, and Figure 7 shows its anatomy: without monitoring, a policy bled from 85% task success to 40% over three days with no alarm; with the four-layer monitoring stack and proper alerting, the same regression tripped an alert at t+6h and was rolled back at t+7h, capping the loss near 15%.

![Before-and-after diagram comparing a three-day silent reward decay with no alert against a monitored deployment where an alert fires at six hours and rollback caps the loss](/imgs/blogs/rl-engineering-production-systems-7.png)

The trick to catching silent failures is to alert on *rate of change and distributional shift*, not just on absolute thresholds. "Return below 50" is a threshold a slow decay sneaks under for days. "Return down 8% week-over-week" plus "reward distribution KS test significant" plus "entropy at 40% of baseline" is a composite signal that fires while the decay is still small. Defense in depth across the four layers is how you turn a silent three-day bleed into a six-hour alert.

In practice you encode this as a composite alert that fires when *enough independent signals* agree, which keeps false positives low while still catching slow decay:

```python
def composite_health_alert(metrics, baseline):
    signals = {
        "return_drop": (baseline["return"] - metrics["return"]) / baseline["return"] > 0.08,
        "reward_drift": metrics["ks_pvalue"] < 0.01,
        "entropy_collapse": metrics["entropy"] < 0.5 * baseline["entropy"],
        "action_shift": metrics["action_kl"] > 0.2,
    }
    n_fired = sum(signals.values())
    # Two or more independent signals agreeing = high-confidence degradation
    severity = "page" if n_fired >= 2 else ("warn" if n_fired == 1 else "ok")
    return {"severity": severity, "signals": signals, "n_fired": n_fired}
```

Requiring two of four signals to agree before paging is the difference between an alerting system on-call trusts and one they mute. A single noisy metric crossing a line is a *warning* to investigate; two independent metrics agreeing is a *page* worth waking someone for. The four signals are deliberately drawn from different layers of the stack — return and reward-drift from the reward layer, entropy from the algorithm layer, action-shift from the policy's behavior — so their agreement is genuinely independent evidence, not the same signal counted four ways. Tuning the exact thresholds is a per-system exercise, but the *shape* — composite, cross-layer, change-based — is universal.

There is a subtle but important asymmetry in how you set these thresholds. A false negative (missing a real degradation) costs you the silent failure from the intro: days of bleeding before anyone notices. A false positive (paging on noise) costs you on-call trust, and an alert nobody trusts is an alert nobody acts on — which converts back into a false negative. The composite-of-two design threads this needle: it is sensitive enough to catch real decay early (because change-based signals fire while absolute thresholds are still fine) but specific enough that on-call keeps believing it (because two independent signals rarely agree by chance). When in doubt, make the *warning* tier loud in a dashboard and the *page* tier conservative; you want humans glancing at warnings constantly and being woken by pages rarely.

### The observability-lag trade-off

Every monitoring signal has a *window* — the amount of recent data it aggregates before it can speak. The reward-drift KS test needs a few hundred recent episodes to be statistically meaningful; the entropy estimate needs enough actions to be stable; the week-over-week return comparison is, by construction, a slow signal. This creates a fundamental trade-off: the longer your window, the more reliable the signal but the slower it reacts. A one-hour window catches problems in roughly an hour but is noisy; a one-day window is rock-solid but lets a fast regression run for a day.

The resolution is to run the *same* signal at multiple windows simultaneously. A short-window version (high sensitivity, used for the fast page) and a long-window version (high specificity, used to confirm and to set the baseline) give you both fast reaction and low false-positive rate. The on-call page fires on the short window; the long window confirms it is real before any irreversible action like a fleet-wide change. For the rollback criterion specifically, this is why the rule is "10% drop sustained for >1 hour" rather than "10% drop right now" — the one-hour sustain requirement is a deliberate window that filters transient dips while still acting within an hour. The number you choose for that window is a direct statement of how much loss you are willing to accept in exchange for not rolling back on noise: a tighter window rolls back faster but more often on false alarms; a looser window is calmer but bleeds longer on a real failure. There is no universally correct setting — it is a business decision about the cost of a wrong rollback versus the cost of a slow one, and you should write it down explicitly rather than letting it be an accident of whatever number someone typed first.

A related practical point: the monitoring signals themselves must be *cheap to compute in production*, because they run on every request or every episode. A drift test that recomputes over the full history on each call will fall behind under load. Use streaming estimators — running mean and variance via Welford's algorithm, reservoir sampling for the reference distribution, exponentially-weighted moving averages for the rate-of-change signals — so the monitoring overhead stays a small constant per request rather than growing with history. Monitoring that cannot keep up with traffic is monitoring that silently stops monitoring, which is the worst failure mode of all because it looks healthy right up until you need it.

## 8. Reward model versioning and feedback collection

For RLHF systems — and for any RL system where the reward is itself a learned or human-derived model — the reward function is not a fixed equation. It is an *artifact* with its own lifecycle, and it must be versioned and monitored as rigorously as the policy. This is a whole class of production concern the classic-RL playbook ignores, because in classic RL the reward is a known function of the environment. In RLHF it is a neural network trained on human preferences, and it can drift, be gamed, and go stale.

### The preference data schema

Human preference data is the ground truth behind an RLHF reward model, and it must be stored as durable, versioned, auditable records — not scraped from a spreadsheet. A minimal schema:

```python
preference = {
    "pair_id": "pref_0001abc",
    "prompt_id": "prompt_5521",
    "prompt": "Summarize the following article: ...",
    "response_a": "...", "response_b": "...",
    "chosen": "a",                      # the preferred response
    "annotator_id": "ann_017",
    "annotation_time_ms": 42000,
    "confidence": 0.8,                  # annotator-reported confidence
    "guideline_version": "v2.3",        # which rubric was in force
    "timestamp": "2026-06-20T14:11:03Z",
}
```

Versioning `guideline_version` matters because **when your annotation rubric changes, the meaning of "chosen" changes**, and mixing pre- and post-change data trains an incoherent reward model. The reward modeling deep-dive (`/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences`) covers how this data trains a Bradley-Terry preference model; here the concern is purely operational — storing it so it stays usable.

### Annotation quality control

Two metrics keep an annotation pipeline honest. **Inter-annotator agreement** (e.g. Cohen's $\kappa$) measures whether annotators agree on the same pairs; a $\kappa$ below ~0.6 means your rubric is ambiguous and the resulting reward model will be noisy. **Annotation drift** measures whether the same annotator's judgments shift over time — fatigue, rubric reinterpretation, or genuine preference change. You catch drift by periodically re-serving previously annotated "gold" pairs and checking that the annotator still agrees with the canonical answer.

```python
from sklearn.metrics import cohen_kappa_score

# Two annotators labeled the same 200 pairs
agreement = cohen_kappa_score(annotator_1_labels, annotator_2_labels)
if agreement < 0.6:
    print("WARNING: low IAA — rubric likely ambiguous, reward model will be noisy")
```

### Versioning reward models with policies

The cardinal rule: **a policy checkpoint is only meaningful paired with the reward model it was trained against.** A reward model is a learned approximation of human preference, and policies are spectacular at finding its blind spots — this is reward hacking (covered in `/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law`). When you swap in a new reward model version `rm-v4`, the policies trained against `rm-v3` may be exploiting a flaw that `rm-v4` fixed, so their behavior under the new scoring is unpredictable. Tag every policy checkpoint with the reward-model version it was optimized against, and monitor the *gap* between the reward model's score and any independent ground-truth signal in production — a widening gap is the signature of reward hacking creeping in. The full distributed picture is in `/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design`.

### Monitoring the reward model in production

The reward-model-versus-policy dynamic gives RLHF a monitoring signal that classic RL does not have, and it is worth instrumenting explicitly. During training, the policy's reward (as scored by the reward model) climbs. The danger is when that climb *decouples* from real human preference — the reward model says the outputs are getting better while a human spot-check says they are getting worse, or merely weirder. This is the operational fingerprint of reward hacking, and you catch it by maintaining a small, continuously-refreshed **held-out human evaluation set** and tracking the correlation between reward-model score and human judgment over time.

```python
from scipy.stats import spearmanr

def reward_model_health(rm_scores, human_scores, prev_correlation):
    rho, _ = spearmanr(rm_scores, human_scores)
    # A falling correlation means the policy is drifting into the reward model's blind spots
    decoupling = rho < prev_correlation - 0.1
    return {"rm_human_corr": float(rho), "decoupling": bool(decoupling)}
```

When the Spearman correlation between reward-model score and human judgment starts falling, the reward model is going stale relative to the policy — the policy has learned to climb the reward model's score in ways humans do not actually prefer. The fix is to collect fresh preference data on the *current* policy's outputs (which is why the annotation pipeline must be continuous, not a one-time batch), retrain the reward model, and bump its version. This closes a loop that classic RL never has to think about: in RLHF, the reward function itself must be maintained, monitored, and re-trained as the policy it scores evolves. A reward model is not a fixed measuring stick; it is a measuring stick the thing being measured is actively trying to bend.

The annotation pipeline that feeds this is itself a production system with its own SLAs: throughput (pairs annotated per day), latency (how fresh is the newest preference data), and quality (the inter-annotator agreement and drift checks above). Treating annotation as a one-time data-collection task rather than an ongoing pipeline is the most common way RLHF systems go stale — the reward model is frozen at launch while the policy and the world both move on.

## 9. Multi-agent production systems and the agent zoo

Many production RL systems are not one agent — they are *many*. A recommendation platform might run a different policy per user segment; a logistics system might run a policy per region; a game might run a policy per difficulty tier. Managing a fleet of policies introduces problems a single agent never has: which policy serves this request, how do you train and deploy a thousand of them, and how do you monitor a fleet without drowning in dashboards.

### The agent zoo pattern

The **agent zoo** is a library of trained policies, each specialized for a slice of the input space, plus a router that selects the right one at serving time. The zoo lives in the policy registry; the router is a fast lookup from request features to policy ID.

```python
class AgentZoo:
    def __init__(self, registry):
        self.registry = registry
        self.cache = {}                          # policy_id -> loaded model

    def route(self, user_features):
        segment = user_features["segment"]       # e.g. "high_value_apac"
        return f"recsys-{segment}-latest"        # policy id in the registry

    def predict(self, user_features, obs):
        policy_id = self.route(user_features)
        if policy_id not in self.cache:
            self.cache[policy_id] = self.registry.load(policy_id)
        return self.cache[policy_id].predict(obs)
```

The router is where production complexity concentrates. The naive version routes on a single feature (segment); mature systems route on a learned model that predicts which policy will perform best for a request — itself a small bandit problem.

The zoo also changes how you think about training. Instead of one giant policy that must be good for everyone (and is therefore mediocre at every extreme), you train many small policies each specialized for a slice, which is often both cheaper and better: a policy for high-value APAC users can be a small network trained on that segment's data and serve in under a millisecond, while a single global policy large enough to capture every segment's nuance would be slower and harder to train. The trade-off is operational surface area — a thousand policies is a thousand things that can drift, go stale, or fail — which is exactly why the fleet practices below exist. The decision of zoo-versus-monolith comes down to whether your segments are genuinely different enough to justify the operational cost: if a single policy serves all segments within a few percent of the per-segment best, the monolith wins on simplicity; if specialization buys double-digit improvements on important segments, the zoo earns its keep.

#### Worked example: serving latency under batching

A recsys ranking policy must respond in 30ms. Single-request inference on CPU takes 8ms per call, so a naive server handles $1000/8 = 125$ requests per second per core before saturating. Switch to dynamic batching with a 2ms max queue delay and a GPU: at 200 requests/second of incoming traffic, a 2ms window collects on average $200 \times 0.002 = 0.4$ requests — too few to fill a batch, so batching barely helps at low load. But at 5,000 requests/second the same 2ms window collects 10 requests per batch, and the GPU processes a batch of 10 in roughly the same 8ms it took for one on CPU, giving an effective throughput of $10 / 0.010 = 1000$ requests/second per GPU at a p99 latency of about $8 + 2 = 10$ms — comfortably inside the 30ms budget. The lesson is that batching's value scales with load: it does little at low traffic and is transformative at high traffic, so size your serving fleet for peak, not average, and measure p99 (not mean) latency because the queue-delay tail is where budgets get blown.

### Fleet management at scale

Training, deploying, and monitoring 1,000+ specialized policies is an *operations* problem, not an algorithms problem. The practices that make it tractable: **shared infrastructure** (one Ray cluster trains the whole fleet, parameterized by segment, rather than 1,000 bespoke jobs); **hierarchical monitoring** (a single fleet-level dashboard with per-segment drill-down, alerting on outliers rather than watching every policy); and a **default fallback** so a request for a segment whose policy is missing, stale, or failing degrades gracefully to a general-purpose policy instead of erroring. The recsys-specific version of all this is in `/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search`. The key mindset shift is that at fleet scale you stop monitoring individual policies and start monitoring the *distribution* of policy health — alert when 5% of policies have drifted, not when any one has.

The retraining cadence is its own design decision at fleet scale. Retraining all 1,000 policies on a fixed nightly schedule wastes compute on segments whose data has not changed, while retraining only on a manual trigger lets stale policies rot. The pattern that works is **staleness-triggered retraining**: each policy carries a freshness budget, and the orchestrator retrains a policy when its segment has accumulated enough new data *or* when its monitoring shows drift, whichever comes first. This turns a flat "retrain everything nightly" job into a prioritized queue where the policies that need attention get it and the stable ones are left alone — which at 1,000 policies is the difference between a tractable compute bill and an absurd one.

There is also a fleet-level version of the rollback discipline from Section 10. When you push a *systemic* change — a new feature, a new network architecture, a new reward formulation — you do not deploy it to all 1,000 policies at once, because a bug in the shared code would take down the entire fleet simultaneously. You roll the change out across the fleet the same way you roll a single policy across traffic: a canary cohort of segments first, monitored against the rest of the fleet as control, then a gradual fleet-wide ramp. The blast-radius logic is identical; only the unit of rollout changes from "percent of traffic" to "fraction of the fleet." A systemic regression caught at 5% of the fleet is an inconvenience; the same regression deployed fleet-wide is an outage.

## 10. The rollback system

Everything so far converges here. Monitoring detects a problem; the rollback system is what *acts* on it. Without a real rollback system, your monitoring is a smoke detector with no fire department — it tells you the house is burning while you fumble for the extinguisher. Figure 8 lays out the decision tree for choosing the right deployment mechanics given your binding constraint, and rollback is the safety net under all of them.

![Tree diagram of the production deployment decision path branching on latency, multi-agent coordination, zero-violation, and RLHF constraints to the matching deployment mechanism](/imgs/blogs/rl-engineering-production-systems-8.png)

### Semantic versioning for policies

Version policies as `major.minor.patch`, with RL-specific meaning: **major** for a changed observation or action space (breaking — serving code must change), **minor** for a retrain on new data or a reward change (behavior changes, interface stable), **patch** for a re-export, quantization, or config tweak (behavior should be unchanged). A version like `recsys-2.3.1` then tells an on-call engineer exactly how risky a change is at a glance.

### The policy registry

The registry stores every checkpoint, tagged with its evaluation metrics, training metadata, and lineage. MLflow's model registry gives you this out of the box:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow.internal:5000")

with mlflow.start_run():
    mlflow.log_params(config)                       # incl. git_sha, env_id, seed
    mlflow.log_metrics({"eval_return": 487.0, "eval_violations": 0})
    mlflow.pytorch.log_model(policy_net, "policy")
    result = mlflow.register_model(
        "runs:/{}/policy".format(mlflow.active_run().info.run_id),
        "recsys-policy",
    )
    # promote to a stage with an alias the serving layer reads
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias("recsys-policy", "candidate", result.version)
```

The serving layer reads a *stage alias* (`production`, `candidate`, `archived`) rather than a hardcoded version, so a rollback is just re-pointing the `production` alias at the previous version — atomic, fast, and auditable.

### Automated rollback and the safe fallback policy

Wire the monitoring alert from Section 6 directly to the registry. When the canary criterion trips, the system re-points the `production` alias to the last-known-good version with no human in the loop:

```python
def auto_rollback(client, model="recsys-policy", baseline_return=4.2,
                  recent_return=None, breach_minutes=0):
    drop = (baseline_return - recent_return) / baseline_return
    if drop > 0.10 and breach_minutes > 60:        # the Section 6 criterion
        last_good = client.get_model_version_by_alias(model, "last_known_good")
        client.set_registered_model_alias(model, "production", last_good.version)
        page_oncall(f"AUTO-ROLLBACK {model} -> v{last_good.version}, drop {drop:.0%}")
        return True
    return False
```

For the worst case — every recent policy is misbehaving, or the registry itself is suspect — you keep a **safe fallback policy**: a deliberately conservative, heavily validated policy (often a simple heuristic or an old, boring, reliable model) that you can switch to during an outage. It will not be optimal, but it will not be dangerous, and "suboptimal but safe" beats "optimal but on fire" every time. For a trading system the fallback might be "hold all positions, place no new orders"; for recsys it might be a non-personalized popularity ranker.

### The disaster recovery playbook

When the production RL agent catastrophically fails, you do not want to be inventing process. Write the playbook in advance: (1) the automated criterion rolls back to last-known-good, or an engineer triggers it manually with one command; (2) if rollback does not restore health, switch to the safe fallback policy; (3) freeze training and deployment until root cause is found; (4) snapshot all logs, the failing checkpoint, and recent inputs for the post-mortem; (5) only re-enable the pipeline after the failing version is reproduced in staging and the monitoring gap that let it through is closed. The discipline of writing this *before* the incident is what turns a 2am panic into a 2am procedure.

#### Worked example: rollback under a real failure

A SAC-based control policy, version `2.4.0`, was promoted to production after passing canary. Eleven hours in, the reward-drift KS test went significant and entropy fell to 38% of baseline — the leading indicators from Section 7. Forty minutes later, episode return crossed the 10% drop threshold and stayed there. At the 60-minute mark the automated rule re-pointed the `production` alias from `2.4.0` back to `2.3.2` and paged on-call; total time from first leading-indicator to restored-service was about 70 minutes, and because the bad policy never exceeded the canary's traffic slice during ramp, blast radius stayed small. Root cause: an upstream feature pipeline had started emitting a feature in a different unit (basis points vs decimals), shifting the observation distribution out from under the policy. The fix was upstream, but the rollback system bought the time to find it without bleeding the business. The leading indicators — entropy and drift — fired *before* the return drop, which is exactly why you monitor the full stack and not just the bottom line.

## 11. Case studies

These are real, documented systems, and each one validates a piece of the architecture above. Where I give numbers I cite or mark them approximate. They also trace an arc — Figure 6 places them on a timeline that runs from bespoke internal clusters in 2017 to the open, reusable training and serving stacks of today. The story of production RL infrastructure is the story of these capabilities moving from "you must build it yourself" to "you assemble it from open components," which is exactly why the tools in Section 5's matrix exist.

![Timeline of production reinforcement learning infrastructure milestones from DeepMind and OpenAI internal clusters in 2017 to open distributed stacks and asynchronous RLHF pipelines by 2023](/imgs/blogs/rl-engineering-production-systems-6.png)

**YouTube production RL recommendation (Chen et al., 2019, "Top-K Off-Policy Correction").** Google deployed a REINFORCE-based recommender on YouTube serving billions of users, and the engineering story is exactly Section 7's lesson: the policy generates its own data (users only see what it recommends), making the logged data badly off-policy. Their headline technical contribution was an off-policy correction (importance weighting plus a top-K correction for recommending slates rather than single items), and they reported it as one of the largest single launches by their reward metric on the system at the time. The production lesson: off-policy correction is not academic — at scale it is the difference between a policy that improves and one that reinforces its own biases.

**Uber Michelangelo.** Uber's internal ML platform standardized the feature store, model registry, training pipelines, and serving infrastructure across the company, and RL features rode on top of that shared substrate rather than building bespoke pipelines. The lesson for this post is Section 2 and Section 9's: production RL is overwhelmingly a *platform* problem. The teams that succeed are the ones that do not reinvent training orchestration, experiment tracking, and serving for every project — they build RL on top of an ML platform that already solved those, the way RLlib sits on Ray.

**DeepMind: AlphaGo and AlphaStar.** AlphaGo (Silver et al., 2016) and AlphaStar (Vinyals et al., 2019) required massive, custom distributed infrastructure — fleets of self-play actors generating games, separate learners updating networks, and a "league" of historical opponents to prevent strategic collapse (a fleet-management problem straight out of Section 9). AlphaStar reached Grandmaster on StarCraft II, top ~0.2% of human players. The reusable lesson is the actor-learner separation of Section 2 at extreme scale: thousands of actors playing, a small number of learners updating, with the league acting as a built-in monitoring-and-diversity mechanism against the silent failure of a policy that beats its current self but forgets how to beat old strategies.

**OpenAI Five (Dota 2, 2019).** OpenAI Five trained on a cluster running thousands of CPU cores for rollouts feeding GPU learners, consuming the equivalent of hundreds of years of self-play per day, and it defeated the world champions. The training ran for *months* continuously, which forced them to solve a problem this post has circled repeatedly: how do you change the model (and even the environment) *while* a months-long training run is in flight without throwing away progress? Their "surgery" techniques for transplanting a trained policy into a modified network are the extreme version of Section 4's environment-versioning discipline — when the env or model changes mid-run, you need a principled way to carry the policy across the change rather than restarting.

The common thread across all four: the *algorithm* (REINFORCE, MCTS+RL, PPO) is almost incidental. What made each system work in production was the infrastructure around it — off-policy correction, a shared platform, actor-learner scaling, a league for robustness, and a way to evolve the system without restarting. That is the entire thesis of this post.

## 11b. The cost model: research versus production RL

Before committing to the production stack, it is worth putting real numbers on what you are buying. A research RL experiment typically involves one person, one GPU, a Python script, and a hand-rolled environment. The total engineering overhead per experiment is measured in hours. A production RL system involves a distributed training cluster, an experiment tracking database, a policy registry, a serving tier with latency SLAs, a monitoring dashboard, and an on-call rotation. The overhead is measured in engineer-months.

The payoff is proportional to deployment volume. A recommendation policy serving 100 million requests per day and updated weekly can justify a team of three engineers maintaining the production stack — the marginal cost per improvement is tiny once the infrastructure exists. The same policy serving 100 requests per day at an early-stage product probably should not: a simpler static policy, a contextual bandit, or even a rule-based system will close most of the gap for a tenth of the maintenance burden.

A useful rule of thumb: if the RL policy is not better than the best supervised or rule-based baseline by at least 5% on a metric you can measure at production scale, the production RL infrastructure will not pay for itself. The supervised baseline is almost always easier to debug, monitor, roll back, and explain to stakeholders. RL earns its production complexity budget only when the sequential decision-making aspect of the task genuinely matters — when the agent's choices today change what data it collects tomorrow in a way that compounds. The recommendation use case earns it because the recommendation *changes the user*, which changes the next observation, which changes the next recommendation. A batch churn-prediction model does not have this property, and it should stay a batch model.

## 12. When to invest in production RL (and when not to)

Production RL infrastructure is expensive — in engineering time more than compute — so the honest section is about when *not* to build it.

| Situation | Recommendation |
|---|---|
| One-off research result, no deployment | Skip all of this. A tracked script with logged seed + git SHA is enough. |
| Deploying a fixed policy that never updates | You need serving + monitoring, but not distributed training or a retraining loop. |
| Reward is a known function, env simulates fast | Build training infra (Section 2); monitoring can be lighter (reward is interpretable). |
| RLHF / learned reward model | You need the full stack *plus* reward-model versioning and a preference store (Section 8). |
| High-stakes, irreversible actions (trading, robotics) | Shadow mode is mandatory before canary (Section 6); safe fallback is mandatory (Section 10). |
| Fleet of specialized policies | Agent zoo + hierarchical monitoring (Section 9) or operational cost explodes. |

Two blunt rules. First: **if you are not deploying, do not build deployment infrastructure** — the experiment-tracking discipline (Section 3) is the only part that pays off in pure research, and it pays off enormously. Second: **the cost of monitoring is always justified once you deploy.** Distributed training is optional if your environment is fast; a fancy registry is optional if you ship one policy; but the monitoring stack of Section 7 is the cheapest insurance you will ever buy against the silent failure from the intro. If you cut one corner, do not let it be monitoring. The single most common and most expensive mistake in production RL is shipping a policy you cannot tell is broken.

## Key takeaways

- **The policy is the easy 20%.** Production RL is four systems around the trainer: experiment management, safe deployment, real-time monitoring, and rollback. Budget your engineering time accordingly.
- **Separate actors from learners.** Many cheap CPU actors collecting experience feed one expensive GPU learner; this is both faster and cheaper per run, and it is the only way to scale collection past a single process. Use V-trace/APPO to correct for the off-policy staleness it introduces.
- **Log seed, env version, git SHA, and full config on every run** — or join the 90% of runs in the RL graveyard that you can never reproduce or trust.
- **The environment version and the normalization statistics are part of the policy artifact.** A checkpoint without its `vecnormalize.pkl` and a pinned env version is not deployable; it is a trap.
- **Never ship to 100% directly.** Shadow mode answers "what would it do?" at zero risk; canary limits the blast radius of a bad policy to 1% while a statistical A/B test decides its fate.
- **Monitor entropy and reward drift, not just the business metric.** They are *leading* indicators — entropy collapse and distribution shift fire before return collapses, turning a silent multi-day bleed into a few-hour alert.
- **Automate rollback with a concrete criterion** (e.g. >10% drop sustained >1 hour) wired to a registry alias, and keep a deliberately conservative safe fallback policy for the worst case.
- **For RLHF, the reward model is a versioned artifact** with its own preference store, quality control (inter-annotator agreement, drift checks), and lineage tags pairing each policy to the reward model it was trained against.
- **At fleet scale, monitor the distribution of policy health,** not individual policies, and route requests through an agent zoo with a default fallback.

## Further reading

- Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (2018) — the V-trace off-policy correction behind Section 2.
- Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning" (2018) — the actor-learner framework used throughout.
- Chen et al., "Top-K Off-Policy Correction for a REINFORCE Recommender System" (2019) — the YouTube production RL case study.
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (2019) — AlphaStar's league and distributed infrastructure.
- OpenAI et al., "Dota 2 with Large Scale Deep Reinforcement Learning" (2019) — months-long training, surgery, and cluster-scale actor-learner.
- Stable-Baselines3 documentation, vectorized environments and `VecNormalize` guide — the serving-boundary hazard of Section 4.
- Within this series: the unified map `reinforcement-learning-a-unified-map` and capstone `the-reinforcement-learning-playbook` for where production sits in the whole picture; `/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design` and `/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences` for the RLHF-specific infrastructure; and `/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search` for the fleet-and-zoo patterns in a recommendation setting.

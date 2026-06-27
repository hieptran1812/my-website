---
title: "Distributed RLHF System Design: Scaling Alignment to Billions of Parameters"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A systems-level tour of how RLHF actually runs at scale — the four-model memory problem, rollout-vs-trainer architectures, sync vs async, and the OpenRLHF, veRL and TRL designs that make 70B alignment runs fit on real clusters."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "llm-alignment",
    "distributed-training",
    "machine-learning",
    "pytorch",
    "ppo",
    "systems",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/distributed-rlhf-system-design-1.png"
---

The first time I tried to run RLHF on a 7B model, I did the obvious thing: load the policy, load a reward model, load a reference copy of the policy, add a value head, and call `trainer.train()`. The process died before the first gradient step with a CUDA out-of-memory error on an 80GB A100. I had assumed RLHF was "supervised fine-tuning plus a reward signal." It is not. It is a distributed systems problem wearing a machine-learning costume, and the moment you understand *why* it refuses to fit on one GPU, the entire architecture of OpenRLHF, veRL, and DeepSpeed-Chat snaps into focus.

The core of the trouble is that a single RLHF step is not one forward-backward pass over a batch. It is a small pipeline: the policy *generates* a response token by token (autoregressive, sequential, slow), a frozen *reference* policy scores how far that response drifts from the base model, a *reward* model assigns a scalar quality score, a *value* model estimates a baseline for variance reduction, and only then does Proximal Policy Optimization (PPO) compute a gradient and update the policy. Four models in memory at once. One of them runs generation, which behaves nothing like a training forward pass. The whole thing has to be choreographed across GPUs, and the naive layout wastes most of your hardware.

This post is the systems view. We will derive *why* the four-model memory wall is unavoidable, build the rollout-worker-versus-trainer split that every serious framework uses, work through synchronous and asynchronous designs and their staleness trade-offs, and then read the actual architectures of OpenRLHF, veRL, and TRL against that frame. By the end you will be able to estimate the memory and throughput of an RLHF run on the back of an envelope, choose the right framework for your scale, and recognize the bottleneck when your `train/step` time balloons. Figure 1 is the map: the four models and how a single PPO step threads through them.

![A dataflow graph showing the policy actor generating a rollout, which feeds the reference policy for KL, the reward model for scoring, and the value model for a baseline, all merging into the PPO trainer that produces an updated policy.](/imgs/blogs/distributed-rlhf-system-design-1.png)

If you want the algorithmic background on PPO itself — the clipped surrogate objective, generalized advantage estimation, the KL penalty — this post assumes it and builds the *systems* layer on top. We will recap just enough to make the memory and communication arguments precise. Throughout, tie everything back to the spine of this whole series: an agent interacting with an environment, collecting rewards, and updating a policy. In RLHF the "environment" is a prompt distribution plus a reward model, the "agent" is the language model, and the "reward" is a learned proxy for human preference. The systems challenge is that this agent has billions of parameters and the environment lives inside other billion-parameter models.

## 1. Why RLHF at scale is a different beast from SFT at scale

Supervised fine-tuning (SFT) at scale is a solved problem in the sense that the recipe is mature: shard the model with Fully Sharded Data Parallel (FSDP) or ZeRO, stream a dataset, do forward-backward, all-reduce gradients, step. The compute is dense and predictable. Every GPU does the same kind of work at the same time. You can saturate the hardware because there is exactly one model and one type of operation.

RLHF breaks every one of those comfortable assumptions. Let me list the structural differences, because each one forces a design decision later.

**You need multiple models resident simultaneously.** PPO-based RLHF requires, at minimum, the policy (the model you are training), a frozen reference policy (to compute a KL-divergence penalty that keeps the policy from drifting into reward-hacking gibberish), a reward model (to score completions), and a value model (a critic that estimates expected return for advantage computation). That is four distinct sets of weights. In SFT you have one. Direct Preference Optimization (DPO) collapses this to two models and removes the reward and value models entirely, which is exactly why DPO is so much cheaper — but DPO is not online RL, and many alignment recipes still want the on-policy signal that PPO or GRPO provide.

**Each step requires generation, which is sequential and expensive.** In SFT, the "forward pass" processes all tokens of a training example in parallel because the targets are known (teacher forcing). In RLHF, the policy must *generate* its own responses autoregressively: produce token 1, append it, produce token 2, append it, and so on for hundreds of tokens. Generation is memory-bandwidth bound, not compute bound, and it cannot be parallelized across the sequence dimension the way a training forward pass can. A single rollout of 512 generated tokens can take longer than the entire gradient update that consumes it. This asymmetry is the single most important fact in RLHF systems design.

**The reward model inference is a bottleneck and a synchronization point.** Every generated response must be scored before you can compute advantages. If the reward model lives on a different set of GPUs, that is a network round trip per batch. If it shares GPUs with the policy, it competes for memory. Either way it sits on the critical path.

**The data is on-policy and ephemeral.** In SFT you can shuffle a fixed dataset and reuse it across epochs. In on-policy RLHF the training data is generated *by the current policy*, so it goes stale the moment you update the weights. You generate, you consume, you discard, you regenerate. This is why a replay buffer in RLHF is fundamentally different from a DQN replay buffer — freshness matters far more than capacity.

Put these together and the conclusion is forced: the distributed architecture for RLHF cannot look like the one for SFT. You cannot just wrap the policy in FSDP and call it a day, because that leaves the generation phase starved (FSDP gathers and re-shards weights on every layer, which is murder for autoregressive decoding) and the other three models unaccounted for. You need a system that treats generation and training as distinct workloads with distinct hardware layouts. Figure 2 contrasts the doomed single-node layout with the distributed one we will build.

![A before-and-after comparison: on the left, all four models crammed onto one 80GB GPU hitting out-of-memory at 7B with serial execution; on the right, separate vLLM rollout workers and an FSDP trainer that overlap generation and training and scale to 70B.](/imgs/blogs/distributed-rlhf-system-design-2.png)

### The throughput accounting that should scare you

Here is a rough decomposition of where wall-clock time goes in a naive synchronous PPO step on a 7B policy generating 512-token responses for a batch of 1024 prompts, all on one node of 8 A100s with the policy under FSDP:

| Stage | Naive single-node share | Why |
| --- | --- | --- |
| Generation (rollout) | 55–70% | autoregressive, memory-bandwidth bound, FSDP re-gathers weights every decode step |
| Reward + reference + value forward | 10–15% | three extra forward passes over the full batch |
| Advantage / GAE computation | <2% | cheap elementwise math on the host or one GPU |
| PPO backward + optimizer step | 20–30% | the only part that resembles SFT |

The headline is that generation dominates, and under FSDP it is *pathologically* slow because FSDP is designed for training, not decoding. The first thing every production RLHF framework does is stop generating under FSDP and hand generation to an inference engine (vLLM, SGLang, or TensorRT-LLM) that uses tensor parallelism and paged attention. That one change can cut rollout time by 4–10×. We will quantify it in a worked example below.

## 2. The four-model memory wall, derived

Let me make the memory argument precise, because it is the constraint that dictates everything. Take a model with $P$ parameters. In BF16, each parameter is 2 bytes, so the raw weights are $2P$ bytes. For a 7B model that is 14 GB just to hold the weights once.

Now count what RLHF needs in memory:

- **Policy weights**: $2P$ bytes (BF16).
- **Policy gradients**: $2P$ bytes.
- **Adam optimizer states**: the master FP32 copy of the weights ($4P$), plus first moment ($4P$) and second moment ($4P$) in FP32 — that is $12P$ bytes if you keep FP32 optimizer state, which is standard for stability.
- **Reference policy**: $2P$ bytes, frozen, no gradients or optimizer.
- **Reward model**: $2P$ bytes (often the same size as the policy, sometimes smaller).
- **Value model**: $2P$ bytes plus its own gradients ($2P$) and optimizer states ($12P$) if it is a separately trained critic of comparable size.

The training-state cost of the policy alone — weights, gradients, Adam — is $2P + 2P + 12P = 16P$ bytes. The classic rule of thumb is "16 bytes per parameter for a model trained with Adam in mixed precision." For a 7B policy that is $16 \times 7\text{e}9 = 112$ GB. That already does not fit on an 80GB GPU, before we add the other three models. This is the wall.

#### Worked example: memory for a 7B RLHF run

Let me add it all up for a 7B-everything setup (policy, reference, reward, and value models all 7B), training the policy and value with Adam, references and reward frozen for inference.

- Policy training state: $16 \times 7\text{e}9 = 112$ GB
- Value training state: $16 \times 7\text{e}9 = 112$ GB
- Reference (inference only): $2 \times 7\text{e}9 = 14$ GB
- Reward (inference only): $2 \times 7\text{e}9 = 14$ GB
- Subtotal of model state: $112 + 112 + 14 + 14 = 252$ GB

And we have not counted activations or the KV cache for generation, which for a batch of 1024 prompts at 512 tokens with a 7B model can easily add tens of gigabytes. So a 7B RLHF run wants roughly **260–300 GB of GPU memory** as a floor. On 80GB A100s that is a minimum of four GPUs just to *fit*, and you want more to run efficiently. Compare this to SFT of the same 7B model, which fits comfortably on two or three GPUs. RLHF is roughly a 2–3× memory multiplier over SFT at the same model size, driven almost entirely by the extra models.

There are three escape routes from the wall, and real systems use combinations of all three:

**1. Model sharding (ZeRO-3 / FSDP).** Split the policy and value weights, gradients, and optimizer states across $N$ GPUs so each holds $1/N$ of the state. ZeRO stage 3 shards all three; FSDP is PyTorch-native equivalent. This is how you make the 112 GB of policy training state fit — across 4 GPUs it is 28 GB each. The cost is communication: every forward and backward all-gathers the shards layer by layer.

**2. LoRA / parameter-efficient fine-tuning.** Instead of training all $P$ parameters, freeze the base and train only low-rank adapters with $r \ll P$ parameters. Now the gradient and optimizer states are tiny — you carry $2P$ bytes of frozen base weights plus a few hundred MB of adapter state. A LoRA RLHF run can do 7B on a single 80GB GPU because the expensive $16P$ training-state term applies only to the adapters. The reference policy can even *be* the base weights (LoRA disabled), eliminating a whole model copy. This is the single biggest lever for fitting RLHF on modest hardware.

**3. Separate nodes / disaggregation per model.** Put the reward model on its own GPUs and serve it as an inference endpoint; put the policy and value on the training GPUs. This trades memory pressure for network traffic and is the path to genuine 70B-scale RLHF.

#### Worked example: LoRA collapses the wall

Run the same 7B RLHF but with rank-16 LoRA on the policy and value, and reuse the frozen base as the reference.

- Frozen base weights (shared by policy-LoRA-off as reference): $14$ GB
- Policy LoRA adapters + grads + Adam: rank-16 adapters on a 7B model are on the order of $20$–$40$M parameters, so $16 \times 30\text{e}6 \approx 0.5$ GB
- Value head + small LoRA: another $\approx 0.5$ GB
- Reward model (inference): $14$ GB
- Subtotal: roughly $14 + 0.5 + 0.5 + 14 = 29$ GB of model state

Now a 7B RLHF run fits on a *single* 80GB GPU with room for the KV cache. The headline number went from 252 GB to about 29 GB. This is why almost every hobbyist and many production RLHF runs use LoRA: it is the difference between needing a four-GPU node and needing one card. The trade-off is expressivity — full fine-tuning can move the model further, and some alignment objectives need that range — but for most preference-tuning, LoRA RLHF is the pragmatic default.

## 3. Rollout workers versus the trainer

The defining architectural move in scalable RLHF is to physically separate the *rollout workers* (which run the policy in generation mode to produce experience) from the *trainer* (which consumes experience and updates weights). Figure 3 shows the resulting layered stack, from orchestration down to the gradient update.

![A layered stack diagram: orchestration with Ray and NCCL on top, then vLLM rollout workers, then the reward inference server, then the experience buffer, then the PPO trainer on FSDP or ZeRO-3 at the bottom.](/imgs/blogs/distributed-rlhf-system-design-3.png)

Why separate them? Because generation and training want opposite things from the hardware:

- **Generation** wants the weights laid out for fast autoregressive decode: tensor parallelism (so each token's matmuls split across GPUs with low latency), paged KV cache, continuous batching. Inference engines like vLLM are built for exactly this and hit far higher token throughput than a training framework ever will.
- **Training** wants the weights laid out for fast gradient computation: FSDP or ZeRO-3 sharding, gradient checkpointing, fused optimizers. This layout is terrible for decode (it re-gathers shards every step) but excellent for backward passes.

If you force one layout to do both jobs, you cripple one of them. So the rollout workers hold a *copy* of the policy in inference layout, generate responses, and ship those responses (and their log-probabilities) to the trainer. The trainer holds the policy in training layout, computes the PPO loss, takes a gradient step, and then *synchronizes the updated weights back* to the rollout workers so the next batch of rollouts is on-policy.

That weight synchronization is the new piece of plumbing RLHF adds. After every (or every few) gradient steps, you must push the updated policy parameters from the trainer's sharded layout into the rollout workers' inference layout. For a 7B model that is 14 GB of BF16 weights to move; for 70B it is 140 GB. Done naively over the network this is brutal. Done well — using NCCL collectives over NVLink/InfiniBand and sometimes a custom weight-update path that streams directly into vLLM's parameter buffers — it is a few hundred milliseconds. OpenRLHF and veRL both spend significant engineering on making this fast, because it is on the critical path of every iteration.

```python
# Sketch of the rollout-worker / trainer split (conceptual, framework-agnostic).
# Rollout workers and trainer are separate processes / Ray actors.

class RolloutWorker:
    def __init__(self, model_path, tp_size):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, tensor_parallel_size=tp_size)
        self.sampling = SamplingParams(temperature=1.0, max_tokens=512,
                                       logprobs=1)  # need logprobs for PPO

    def generate(self, prompts):
        outs = self.llm.generate(prompts, self.sampling)
        return [(o.prompt, o.outputs[0].text,
                 o.outputs[0].cumulative_logprob) for o in outs]

    def update_weights(self, named_tensors):
        # Stream new policy weights into the running vLLM engine in-place.
        self.llm.llm_engine.model_executor.load_weights(named_tensors)


class Trainer:
    def __init__(self, policy, value, optimizer):
        self.policy = policy          # FSDP-wrapped, training layout
        self.value = value
        self.opt = optimizer

    def ppo_step(self, experience_batch):
        loss = compute_ppo_loss(self.policy, self.value, experience_batch)
        loss.backward()
        self.opt.step(); self.opt.zero_grad()
        return {name: p.detach() for name, p in self.policy.named_parameters()}
```

The `update_weights` call is where the magic and the pain live. The trainer's parameters are sharded across its GPUs; the rollout worker's vLLM engine expects full tensors in its own tensor-parallel layout. Bridging those two layouts efficiently — ideally over NVLink with no host round trip — is the difference between a 200 ms sync and a 20 second one.

### The throughput–staleness trade-off

Once rollout and training are separate processes, a question appears that did not exist in SFT: *do they run at the same time?* If the trainer waits for rollouts to finish, then rollouts wait for the trainer to update weights, then rollouts run again — that is **synchronous** RLHF, and the GPUs are half-idle by construction (the trainer GPUs idle during generation; the rollout GPUs idle during the update). If instead the rollout workers run *continuously*, generating experience with whatever weights they currently have while the trainer chews through earlier experience — that is **asynchronous** RLHF, and the GPUs stay busy, but the experience the trainer consumes was generated by a slightly *older* policy. That mismatch is **staleness**, and it is the central tension of the next two sections.

## 4. Synchronous versus asynchronous RLHF

Let me make the two regimes concrete, because the choice between them is the highest-leverage decision in an RLHF system.

**Synchronous RLHF** runs the pipeline in lockstep, exactly as Figure 4 lays out: sample prompts, generate rollouts with the current policy, score them with the reward model, compute advantages with generalized advantage estimation (GAE), run several PPO epochs over that batch, then broadcast the updated weights to the rollout workers and start over. Every piece of training data was generated by the immediately preceding policy. There is *zero* staleness — the algorithm matches the textbook PPO derivation exactly. The price is utilization: while the policy generates, the trainer's backward-pass kernels are idle; while the trainer updates, the rollout engine is idle. You are paying for GPUs that spend half their time waiting.

![A pipeline diagram of one synchronous RLHF step: sample prompts, rollout where the policy generates, score with the reward model, compute advantages with GAE, run the PPO update for four epochs, then sync workers by broadcasting weights.](/imgs/blogs/distributed-rlhf-system-design-4.png)

**Asynchronous RLHF** decouples the stages. Rollout workers generate experience continuously and push it into a shared buffer; the trainer pulls batches from the buffer whenever it has capacity, updates the policy, and pushes new weights back to the workers on its own cadence. Now nothing waits: rollout GPUs are always generating, trainer GPUs are always training. The catch is that a batch the trainer pulls might have been generated one, two, or more updates ago — its log-probabilities and advantages were computed under an older policy $\pi_{\theta_{old}}$ that no longer matches the current $\pi_\theta$.

Why does staleness matter, theoretically? Recall the PPO objective. The clipped surrogate is

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\right)\right]$$

where the probability ratio is

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}.$$

The whole point of PPO is that it is *almost* on-policy: it allows a few gradient steps away from the data-collecting policy $\pi_{\theta_{old}}$, and the clip term $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ keeps the update trustworthy only inside a small trust region around $\pi_{\theta_{old}}$. The importance ratio $r_t$ corrects for the mismatch between the policy that generated the action and the policy being optimized. When data is fresh, $r_t \approx 1$ and the correction is tiny and well-behaved. When data is stale — generated several updates ago — $\pi_\theta$ has drifted far from the $\pi_{\theta_{old}}$ that produced the data, $r_t$ can be far from 1, the importance weights have high variance, and many samples get clipped (their gradient zeroed). The estimator becomes biased and noisy. Staleness silently degrades the gradient.

So the trade-off is exact and unavoidable:

| Property | Synchronous | Asynchronous |
| --- | --- | --- |
| Staleness | zero | grows with buffer lag |
| GPU utilization | ~50% (stages idle each other) | ~90%+ (stages overlap) |
| Gradient quality | matches textbook PPO | biased/noisy if lag is large |
| Wall-clock to convergence | slower per-step efficient | faster if lag is bounded |
| Implementation complexity | low | high (buffer, versioning, sync) |
| Reproducibility | high | lower (nondeterministic ordering) |

**Why most large systems use a hybrid.** The pragmatic answer is *bounded asynchrony*: let rollout workers run ahead of the trainer, but only by a small, fixed amount — typically one update, sometimes two. This is often called "one-step off-policy" or "off-by-one async." The trainer is always consuming data generated by the policy from exactly one update ago, which keeps $r_t$ close to 1 (staleness is bounded and small, so the importance weights stay well-conditioned) while still overlapping generation and training so the GPUs stay busy. Empirically, off-by-one async recovers most of the utilization win with a barely measurable hit to gradient quality. Going to off-by-many is where convergence starts to suffer. Figure 7 will show this bounded-async architecture in detail.

#### Worked example: how staleness inflates the clip rate

Make the staleness cost concrete. Suppose each PPO update moves the policy such that, on average, $\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{old}}(a_t \mid s_t) \approx 0.05$ per token for the tokens the update favors. With off-by-one staleness, the data the trainer consumes is one update old, so the ratio is $r_t = e^{0.05} \approx 1.05$ — comfortably inside the clip band $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$ for the standard $\epsilon = 0.2$. Almost nothing clips; the gradient is clean.

Now suppose the rollout workers run four updates ahead before their data is consumed (off-by-four). The drift compounds to roughly $4 \times 0.05 = 0.20$, so $r_t \approx e^{0.20} \approx 1.22$ — *outside* the clip band. The clip term zeros the gradient for that whole population of favored tokens, and worse, the variance of the importance weights $r_t$ across the batch grows roughly exponentially in the drift, so a few tokens with large positive drift dominate the estimator. In practice you would observe `train/clipfrac` (the fraction of tokens clipped) jump from a healthy ~5–15% at off-by-one to 40%+ at off-by-four, and the effective learning slows because so much of the gradient is being discarded. This is the mechanism, not a hand-wave: bounded async keeps the drift small enough that $r_t$ stays inside the trust region, which is exactly why "off-by-one" is the sweet spot rather than "as-async-as-possible."

#### Why the KL-to-reference term is the safety rail

There is a second, subtler reason large-scale RLHF systems behave the way they do, and it is worth deriving because it explains a memory cost (the reference model) that otherwise looks gratuitous. The reward signal PPO optimizes is not the raw reward-model score $r_\phi(x, y)$; it is the score *minus a KL penalty* against the frozen reference policy:

$$R(x, y) = r_\phi(x, y) - \beta \, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{ref}(\cdot \mid x)\big).$$

Why subtract that KL term at all? Because the reward model $r_\phi$ is a *learned, imperfect* proxy for human preference, trained on a finite dataset of comparisons. It is accurate near the distribution of responses it was trained on and unreliable far from it. If you let the policy maximize $r_\phi$ without constraint, it will discover *adversarial* responses — strings that score high under $r_\phi$ but that no human would prefer, the textbook reward-hacking failure where a model learns to emit a particular flattering phrase or a wall of hedging because the reward model overweights it. The KL term penalizes the policy for moving away from the reference (the SFT model), which keeps generations in the distribution where $r_\phi$ is trustworthy. Formally, the objective is a KL-regularized reward maximization whose optimum is the reference policy reweighted by the exponentiated reward, $\pi^*(y \mid x) \propto \pi_{ref}(y \mid x)\, e^{r_\phi(x,y)/\beta}$ — a tilt of the reference, not a wholesale replacement of it. The coefficient $\beta$ tunes how far the tilt is allowed to go: too small and you reward-hack, too large and the policy barely moves. This is precisely why the reference model must be resident in memory at every step (it supplies $\pi_{ref}$ for the KL term) and why `train/kl` is the single most important health signal — an unbounded KL means the safety rail has failed and the run is drifting into the reward model's blind spots. The four-model memory wall, in other words, is not an accident of implementation; the reference model is load-bearing for correctness.

```python
# Bounded asynchronous loop with a max staleness of `max_lag` updates.
# The buffer tags each experience with the policy version that produced it.

POLICY_VERSION = 0
buffer = ExperienceBuffer(max_size=8192)

def rollout_loop(worker):
    while True:
        prompts = sample_prompts(batch=256)
        exp = worker.generate(prompts)          # uses worker's current weights
        buffer.put(exp, version=worker.version)  # tag with producing version

def train_loop(trainer, workers, max_lag=1):
    global POLICY_VERSION
    while True:
        # Only pull experience no older than max_lag updates.
        batch = buffer.sample(batch=1024,
                              min_version=POLICY_VERSION - max_lag)
        new_weights = trainer.ppo_step(batch)
        POLICY_VERSION += 1
        for w in workers:                        # broadcast on trainer cadence
            w.update_weights(new_weights)
            w.version = POLICY_VERSION
```

The `min_version` filter is the safety valve. If rollout workers race too far ahead, their stale experience is simply not sampled — it ages out of the eligible window. This bounds $r_t$ regardless of how fast the workers run, which is what makes async safe in practice.

## 5. The OpenRLHF architecture

OpenRLHF was, for many teams, the first framework that made 70B-scale RLHF tractable on commodity clusters, and its design is a clean expression of everything above. It is built on **Ray** for orchestration, uses **vLLM** for rollout generation, and **DeepSpeed ZeRO-3** for training. The defining choice is a **multi-controller**, actor-based layout: each model role is a separate set of Ray actors.

The component breakdown:

- **vLLM rollout actors** — hold the policy in inference layout, run continuous-batched generation. These are the throughput engine.
- **Actor (policy) training actors** — hold the policy under ZeRO-3 for the PPO gradient step.
- **Critic (value) training actors** — the value model, also under ZeRO-3, trained jointly.
- **Reward model actors** — inference-only, score completions.
- **Reference policy actors** — inference-only, frozen, supply the log-probs for the KL term.
- **Experience buffer** — collects `(prompt, response, reward, ref_logprob, value, advantage)` tuples between rollout and training.

Ray's role is to place these actors on GPUs, route data between them, and handle the weight-sync collectives. Because each role is a separate actor group, you can give each one its own parallelism and its own GPU count. The reward model might be small and need one GPU; the policy might be 70B and need an 8-GPU tensor-parallel group for vLLM plus a separate ZeRO-3 group for training. OpenRLHF lets you co-locate or disaggregate these flexibly. A common pattern is **hybrid co-location**: the policy's training actors and its vLLM rollout actors share the same physical GPUs but time-multiplex them — training offloads to CPU while vLLM generates, then vLLM offloads while training runs. This is OpenRLHF's answer to the "don't pay for idle GPUs" problem within the multi-controller model.

```bash
# Launch a 70B OpenRLHF PPO run across a Ray cluster.
# Roles are placed on disjoint or shared GPU groups via Ray placement.
ray start --head --num-gpus 8

python -m openrlhf.cli.train_ppo_ray \
  --pretrain meta-llama/Llama-3-70B \
  --reward_pretrain my-org/llama3-70b-reward \
  --actor_num_nodes 4 --actor_num_gpus_per_node 8 \
  --critic_num_nodes 2 --critic_num_gpus_per_node 8 \
  --vllm_num_engines 8 --vllm_tensor_parallel_size 4 \
  --colocate_actor_ref \
  --zero_stage 3 --bf16 \
  --micro_train_batch_size 4 --train_batch_size 1024 \
  --rollout_batch_size 1024 --generate_max_len 1024 \
  --init_kl_coef 0.02 --gamma 1.0 --lambd 0.95 \
  --async_train  # enable bounded-async rollout/train overlap
```

A few flags encode the lessons from earlier sections. `--colocate_actor_ref` reuses the policy GPUs for the reference forward pass (the reference is the same architecture, so co-locating saves a whole GPU group). `--vllm_tensor_parallel_size 4` puts generation under tensor parallelism, not ZeRO. `--zero_stage 3` shards the training state to beat the memory wall. `--async_train` flips on the bounded-async overlap.

#### Worked example: throughput, rollout versus train

Consider an OpenRLHF run on a 7B policy, batch 1024 prompts, 512 generated tokens, on one node of 8 A100-80GB. Suppose we measure per-stage wall-clock:

- Rollout (vLLM, TP=2 across 4 engines): 18 s for the batch.
- Reward + reference + value forward: 4 s.
- GAE / advantage: 0.3 s.
- PPO update (4 epochs, ZeRO-3): 9 s.

Synchronous step time: $18 + 4 + 0.3 + 9 \approx 31.3$ s, and during the 9 s of PPO the vLLM engines are idle, during the 18 s of rollout the ZeRO trainer is idle. Effective GPU utilization is roughly $(\text{useful work}) / (\text{total} \times \text{GPUs})$ — call it ~50%.

Now enable bounded-async with off-by-one. The trainer's 9 s PPO update overlaps with the next batch's rollout. The step is now gated by the *longer* of the two overlapping stages, the 18 s rollout, plus the unhidden reward/GAE on the critical path. Step time drops toward ~22 s, and utilization climbs past 80%. That is a roughly 30% throughput gain from a single config flag, paid for with off-by-one staleness — which, as argued in section 4, barely touches gradient quality. This is exactly the kind of measurement you should take before and after enabling async, using the per-stage timers we will cover in section 11. (These numbers are illustrative of the *shape* of the result; absolute times depend heavily on sequence length, hardware, and TP degree.)

OpenRLHF's reported numbers in its papers and repo claim meaningful speedups over DeepSpeed-Chat at 7B and the ability to run 70B PPO where naive setups cannot fit at all — driven by the vLLM rollout engine and ZeRO-3 sharding working together. The multi-controller design's cost is orchestration complexity: you are managing several actor groups and their placement, and debugging a Ray cluster is harder than debugging a single Python process.

## 6. The veRL architecture and the HybridEngine

veRL (Volcano Engine Reinforcement Learning, originally "HybridFlow") attacks the same problem from a different angle, and the contrast is instructive. Where OpenRLHF is multi-controller (many actors, orchestrated by Ray), veRL uses a **single-controller** programming model for the algorithm's control flow combined with **multi-controller** execution for the heavy compute. You write the PPO loop as if it were a single Python program — `rollout`, `compute_reward`, `compute_advantage`, `update_policy` as sequential function calls — and veRL's runtime maps each call onto the right distributed backend. This makes the algorithm readable and easy to modify (you can change the advantage computation without touching distributed plumbing) while still running each stage at scale.

The signature feature is the **HybridEngine**: instead of keeping rollout and training on separate GPUs, the HybridEngine *shares the same GPUs* between generation and training and reshards the model between the two layouts on the fly. During the rollout phase it arranges the policy weights in vLLM's tensor-parallel inference layout; when it is time to train, it reshards the *same physical weights* into FSDP/Megatron training layout — without a network round trip to a separate trainer, because the weights never left these GPUs. This directly attacks the weight-synchronization cost that OpenRLHF's disaggregated design pays: there is no "push weights from trainer to rollout worker" step, because they are the same GPUs holding the same tensors in two layouts.

The trade-off is the reverse of OpenRLHF's. HybridEngine maximizes hardware efficiency (every GPU does both jobs, no idle disaggregated rollout fleet, no big weight transfer) at the cost of *time-multiplexing*: the GPUs cannot generate and train simultaneously, because they are busy reshardng between the two modes. So the classic HybridEngine is essentially a very efficient *synchronous* design — it eliminates the disaggregation overhead rather than the synchronization barrier. Newer veRL versions add async and disaggregated options too, so the line has blurred, but the HybridEngine is the idea veRL is known for.

```python
# veRL's single-controller PPO loop reads like plain sequential Python.
# The runtime dispatches each call to the right distributed backend.
from verl.trainer.ppo import RayPPOTrainer
from verl import DataProto

trainer = RayPPOTrainer(config)   # config declares TP/PP/FSDP for each role
trainer.init_workers()

for step in range(num_steps):
    batch: DataProto = trainer.get_prompt_batch()

    # Rollout: HybridEngine puts policy in vLLM inference layout.
    gen = trainer.actor_rollout_wg.generate_sequences(batch)

    # Score with reward model and reference (separate worker groups).
    rewards = trainer.rm_wg.compute_rm_score(gen)
    ref_logp = trainer.ref_wg.compute_ref_log_prob(gen)

    # Advantages on the controller (cheap, single process).
    gen = compute_gae_advantage(gen, rewards, ref_logp,
                                gamma=1.0, lam=0.95)

    # Update: HybridEngine reshards SAME weights into FSDP training layout.
    trainer.actor_rollout_wg.update_actor(gen)
    trainer.critic_wg.update_critic(gen)
```

Notice there is no explicit weight-sync call between rollout and update. `actor_rollout_wg` is one worker group that does *both* generation and training, reshardng internally. That is the HybridEngine in one line of API.

When does veRL win over OpenRLHF? When weight-sync cost dominates — very large models where moving 140 GB of weights between a disaggregated trainer and rollout fleet every step is painful, and you would rather time-multiplex the same GPUs. veRL's published HybridFlow results report strong throughput at scale precisely because of this resharding trick. When does OpenRLHF win? When you want true rollout/train overlap (full async) and are willing to pay the weight-sync cost to keep both fleets busy simultaneously, or when you want the simpler design of independent actor groups you can scale independently.

| Dimension | OpenRLHF | veRL (HybridEngine) |
| --- | --- | --- |
| Control model | multi-controller (Ray actors) | single-controller logic, multi-controller execution |
| Rollout/train placement | disaggregated (separate GPUs) | co-located, reshards same GPUs |
| Weight sync | explicit push trainer→rollout | none (weights never move) |
| Concurrency | true async overlap possible | classic HybridEngine is synchronous |
| Best when | want full async, independent scaling | weight-sync cost dominates, want max efficiency |
| Mental model | several services to wire | one sequential PPO loop |

## 7. TRL: when simple is enough

Not every RLHF run needs Ray, vLLM resharding, and ZeRO-3. **TRL** (Transformer Reinforcement Learning) from Hugging Face is the framework you reach for when you want PPO, DPO, or GRPO on a single node or with plain data parallelism, integrated tightly with the `transformers` ecosystem. TRL's `PPOTrainer` keeps the four models in one process, uses `accelerate` for multi-GPU data parallelism (and can use DeepSpeed as the backend), and generates with the model's native `.generate()` rather than a separate vLLM fleet (though recent TRL versions can offload generation to vLLM too).

TRL is the right tool when: your model is small enough to fit the four-model state on the GPUs you have (often with LoRA), you are prototyping or doing research where the simplicity of one process matters more than peak throughput, or you are using DPO, where there is no reward or value model and no generation-during-training at all, so the heavy machinery is unnecessary. DPO trains directly on a preference dataset with a closed-form loss — it is essentially supervised learning with a clever loss — so TRL's `DPOTrainer` scales like SFT and is wonderfully simple.

```python
# TRL PPO on a single node with LoRA — the four models, minimal plumbing.
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from transformers import AutoTokenizer
from peft import LoraConfig

config = PPOConfig(
    model_name="meta-llama/Llama-3-8B",
    learning_rate=1.41e-5,
    batch_size=256, mini_batch_size=8,
    init_kl_coef=0.02, target_kl=6.0,   # adaptive KL controller
    cliprange=0.2, cliprange_value=0.2, gamma=1.0, lam=0.95,
)
tok = AutoTokenizer.from_pretrained(config.model_name)

# Policy carries a value head; LoRA keeps training-state tiny.
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, peft_config=lora, load_in_4bit=True)

# With LoRA, the reference is the base with adapters disabled — no extra copy.
ref = create_reference_model(policy)

trainer = PPOTrainer(config, policy, ref, tok)

for batch in dataloader:
    query_tensors = batch["input_ids"]
    response_tensors = trainer.generate(query_tensors, max_new_tokens=256)
    rewards = reward_model_score(batch["query"], response_tensors)  # list[Tensor]
    stats = trainer.step(query_tensors, response_tensors, rewards)
    trainer.log_stats(stats, batch, rewards)
```

This fits an 8B PPO run on a single 80GB GPU thanks to 4-bit base weights, LoRA adapters, and the reference being the adapter-disabled base. It will not match OpenRLHF's throughput at 70B — generation under `.generate()` is slower than vLLM, and one process cannot disaggregate roles — but for an 8B research run it is dramatically less to set up and debug. The decision tree in Figure 8 captures when to reach for each framework.

The rule I give people: **prototype in TRL, scale in OpenRLHF or veRL.** Get your reward model, your data pipeline, and your reward-shaping right on a small TRL run where you can iterate in minutes, then port the validated recipe to a distributed framework once you need 70B or production throughput. Porting is mostly config; the algorithm is the same PPO.

### Reading the framework landscape

Stepping back, the framework choice is a four-way trade among maximum model size, throughput, ease of use, and async support — and no single framework dominates on all four. Figure 5 lays out the landscape as a matrix so you can read the trade at a glance.

![A matrix comparing RLHF frameworks TRL, OpenRLHF, veRL, NeMo-Aligner, and XTuner across maximum model size, throughput, ease of use, and asynchronous support, showing that ease of use trades off against scale and throughput.](/imgs/blogs/distributed-rlhf-system-design-5.png)

The pattern in the matrix is the same trade we have been circling: the frameworks that scale highest and run fastest (OpenRLHF, veRL, NeMo-Aligner) are the ones that take on the most distributed machinery, and that machinery costs setup and debugging time. TRL sits at the easy-and-small corner; the production engines sit at the hard-and-large corner. NeMo-Aligner (NVIDIA) deserves a mention beyond the others because it leans on Megatron-LM's mature 3D-parallelism stack and has demonstrated RLHF on models in the hundreds of billions of parameters — it is the choice when you are already in the Megatron ecosystem and need the largest possible scale, at the cost of being harder to bend to a custom recipe. XTuner (from the InternLM team) targets the easy-to-use, moderate-scale niche with strong LoRA and QLoRA integration, closer in spirit to TRL but with better multi-GPU ergonomics.

The honest summary is that for the vast majority of teams the real choice is three-way: TRL for prototypes and DPO, OpenRLHF when you want true async overlap and independently-scalable roles, and veRL when weight-sync cost dominates and the HybridEngine's resharding wins. The others are specialists you reach for when their particular strength — Megatron scale for NeMo, LoRA ergonomics for XTuner — is exactly what you need.

## 8. 3D parallelism inside RLHF

To reach the largest models, RLHF inherits the same parallelism toolkit as large-scale pretraining — but applies it twice, because rollout and training have different optimal layouts. The three axes:

**Data parallelism (DP).** Replicate the model across groups of GPUs; each group processes a different shard of the batch; gradients are all-reduced. This is the easy axis and the first one you scale. In RLHF, DP applies to the trainer (each replica does PPO on a different slice of the experience batch) and to rollout (each rollout worker generates for a different slice of prompts).

**Tensor parallelism (TP).** Split individual layers across GPUs — the attention and MLP matmuls are partitioned so each GPU holds a slice of every weight matrix, and partial results are all-reduced within the layer. TP is essential for *generation* because it keeps per-token latency low (each decode step is split across GPUs). vLLM uses TP for the rollout fleet. TP is also used in training for models too big to fit even one layer's activations on a GPU.

**Pipeline parallelism (PP).** Split the *layers* across GPUs — GPU 0 holds layers 0–9, GPU 1 holds layers 10–19, and microbatches flow through the stages like an assembly line. PP is great for training very deep models (it bounds the activation memory per GPU) but is awkward for autoregressive generation because the pipeline bubble is expensive when you decode one token at a time. So in RLHF you often see **PP+TP+DP for the trainer and TP+DP for the rollout engine** — pipeline parallelism on the training side, tensor parallelism on the generation side.

Combining them, a 70B RLHF run might use, for the trainer: TP=8 within a node (NVLink-fast), PP=2 across two nodes, and DP across the remaining replicas, with ZeRO-1 sharding optimizer states on top. For rollout: TP=4 per vLLM engine, DP across multiple engines. The total parallelism degree is the product $\text{DP} \times \text{TP} \times \text{PP}$, and choosing the split is a memory-versus-communication optimization: more TP means more frequent all-reduces (keep TP within a node where bandwidth is high), more PP means pipeline bubbles, more DP means bigger all-reduce of gradients.

| Parallelism | Splits | Comm pattern | RLHF role |
| --- | --- | --- | --- |
| Data (DP) | the batch | all-reduce gradients | scale batch on trainer + rollout |
| Tensor (TP) | each layer's matmuls | all-reduce within layer | generation latency; big-layer training |
| Pipeline (PP) | layers into stages | point-to-point activations | very deep model training |
| ZeRO / FSDP | weights, grads, optimizer | all-gather / reduce-scatter | beat the memory wall |

The key RLHF-specific insight is that you tune these *separately* for rollout and training, because their bottlenecks differ. The rollout engine's enemy is per-token latency (favor TP within a node, avoid PP). The trainer's enemy is memory and gradient-comm volume (favor ZeRO-3 plus modest TP). A common mistake is to reuse the training parallelism config for generation and then wonder why rollout is slow — that is the FSDP-during-decode pathology again, just wearing a 3D-parallelism hat.

## 9. Experience replay buffer design

The buffer between rollout and trainer is small but consequential. Each entry is a tuple of everything PPO needs to recompute its loss without re-running the models: `(prompt, response, reward, old_log_probs, ref_log_probs, values, advantages, returns)`. Let me account for what must be stored and why.

- **prompt, response** — token ids; needed to recompute the new policy's log-probs during the PPO epochs.
- **old_log_probs** — the log-probability of each response token under the policy *that generated it*. This is the denominator $\pi_{\theta_{old}}$ of the PPO ratio $r_t$. You must store it at generation time, because after the first gradient step the policy has changed and you can no longer recover it.
- **ref_log_probs** — log-probs under the frozen reference policy, for the KL penalty. Computed once at collection time.
- **values** — the critic's value estimate per token, the baseline for advantage.
- **advantages, returns** — computed via GAE from rewards and values, then frozen for the PPO epochs.

The crucial design point is **freshness over capacity**. A DQN buffer is large (millions of transitions) and you sample uniformly from a long history because off-policy value learning tolerates old data. An RLHF/PPO buffer is the opposite: it holds *one* (or, in bounded-async, a couple of) batches of on-policy experience, and you *discard* it after consuming it, because PPO's trust-region guarantee only holds near $\pi_{\theta_{old}}$. Storing a long history would let stale data poison the importance ratios — exactly the staleness problem of section 4. So the buffer is sized to a single iteration's batch (plus a small async lag window), not to a giant replay memory.

```python
import torch
from dataclasses import dataclass

@dataclass
class Experience:
    prompt_ids: torch.Tensor      # [seq]
    response_ids: torch.Tensor    # [resp]
    old_logprobs: torch.Tensor    # [resp]  log pi_old per token
    ref_logprobs: torch.Tensor    # [resp]  log pi_ref per token
    values: torch.Tensor          # [resp]  critic baseline
    reward: float                 # scalar from reward model
    version: int                  # policy version that produced it

class ExperienceBuffer:
    def __init__(self, max_size):
        self.data, self.max_size = [], max_size

    def put(self, exp: Experience):
        self.data.append(exp)
        if len(self.data) > self.max_size:        # FIFO: oldest ages out
            self.data.pop(0)

    def sample(self, batch, min_version):
        # Freshness gate: never train on experience older than the window.
        fresh = [e for e in self.data if e.version >= min_version]
        idx = torch.randperm(len(fresh))[:batch]
        return [fresh[i] for i in idx]
```

Mini-batch sampling within a PPO iteration is its own small choice. After collecting a batch of, say, 1024 trajectories and computing advantages, PPO runs several epochs (typically 1–4) over that batch, shuffling into mini-batches of 8–64 each epoch. More epochs squeeze more learning out of each expensive rollout but push the policy further from $\pi_{\theta_{old}}$, so the clip term fires more often — another instance of the same trust-region tension. A practical default is 1–2 epochs for RLHF (lower than the 3–10 common in game RL), because the rollouts are so expensive that you want to move on and regenerate fresh data rather than over-optimize on stale-ish data.

One more knob: **whitening advantages.** Before the PPO epochs, normalize the batch's advantages to zero mean and unit variance. This is a cheap, almost-free trick that dramatically stabilizes the gradient scale across batches with different reward magnitudes — and at scale, where a single bad batch can spike the gradient norm and destabilize a 70B run, it is close to mandatory.

## 10. Communication bottlenecks at scale

Once you are on many nodes, the network becomes a first-class design constraint. Let me enumerate the traffic in an RLHF iteration and where it hurts.

**Weight synchronization (trainer → rollout).** The biggest single transfer in a disaggregated design. After a gradient step, the updated policy weights — $2P$ bytes in BF16 — must reach every rollout worker. For 70B that is 140 GB per sync. Over a 200 Gb/s InfiniBand link that is naively $140 \text{ GB} \times 8 / 200 \text{ Gb/s} \approx 5.6$ s if done point-to-point to each worker serially — unacceptable on the critical path. The fixes: (a) use NCCL broadcast/all-gather collectives so the transfer is tree-structured, not serial; (b) keep rollout and trainer on the same nodes connected by NVLink (~600 GB/s) so the transfer is intra-node; (c) sync less often (every few steps) accepting a touch more staleness; (d) overlap the sync with the next rollout (async). veRL sidesteps this entirely with the HybridEngine, as we saw — there is no transfer because the weights never move.

**Reward and reference log-probs (rollout → trainer).** Each response's reward is a scalar (trivial), but the reference log-probs are one float per response token — for a batch of 1024 × 512 tokens that is ~2M floats, a few MB. Small. The cost is not the bytes but the *forward pass*: the reference and reward models must each run a forward over the full batch, which is compute, and if they live on remote GPUs, the responses must travel to them and the scores back. Co-locating the reference with the policy (same architecture) avoids one network hop, which is why `--colocate_actor_ref` exists.

**Gradient all-reduce (within the trainer's DP group).** Standard data-parallel traffic: $2P$ bytes reduced across DP replicas per step. This is the same cost as SFT and is well-optimized by NCCL ring/tree all-reduce. Keep the DP group's interconnect fast.

**Experience movement (rollout → buffer → trainer).** The prompts, responses, and stored log-probs flow from rollout workers into the buffer and out to the trainer. For 1024 × (256 prompt + 512 response) token-ids plus per-token log-probs and values, this is tens of MB per batch — modest, but it must be pipelined so it does not stall the trainer waiting for the next batch.

| Transfer | Volume (70B, batch 1024) | Frequency | Mitigation |
| --- | --- | --- | --- |
| Weight sync trainer→rollout | ~140 GB | every step (or every k) | NCCL broadcast, NVLink co-loc, async, HybridEngine |
| Gradient all-reduce | ~140 GB | every step | NCCL ring/tree, fast DP interconnect |
| Reference/reward forward | compute + few MB | every step | co-locate ref with policy |
| Experience tuples | tens of MB | every step | pipeline with rollout |

The headline: **weight synchronization and gradient all-reduce are the two heavy hitters, both ~$2P$ bytes, and both want a fast interconnect.** This is why RLHF clusters care about NVLink and InfiniBand bandwidth as much as raw FLOPs, and why the framework's choice about whether weights move at all (disaggregated push versus HybridEngine resharding) is the single biggest determinant of network pressure. Figure 7 shows the async architecture where weight broadcast and rollout overlap to hide this cost.

![A dataflow graph of the asynchronous rollout architecture: three vLLM rollout workers fan out into a shared experience buffer with bounded staleness, the PPO trainer samples a batch from the buffer, and updated weights are broadcast back out to all three workers.](/imgs/blogs/distributed-rlhf-system-design-7.png)

## 11. Profiling and monitoring at scale

You cannot fix what you do not measure, and RLHF has more places to hide a bottleneck than almost any other ML workload. The instrumentation I insist on for any serious run:

**Per-stage wall-clock timers.** Break every iteration into `time/rollout`, `time/reward`, `time/reference`, `time/gae`, `time/train`, and `time/weight_sync`, and log them every step. The first thing you learn is the *shape* — almost always rollout dominates, and the question becomes whether async is hiding it. A sudden jump in `time/weight_sync` means your NCCL path regressed (fell back to host round trips, say). A jump in `time/train` with constant batch means a memory issue forced more gradient-accumulation microbatches.

**GPU utilization and the rollout/train idle pattern.** Watch `nvidia-smi dmon` or DCGM across the rollout GPUs and the trainer GPUs separately. In synchronous mode you will see them anti-phase — rollout GPUs at 95% while trainer GPUs sit at 5%, then flip. That visible anti-phase is your utilization tax, and the size of the gap tells you how much async can buy you.

**Model FLOPs Utilization (MFU).** MFU is the fraction of the hardware's peak FLOPs your run actually uses for useful model math:

$$\text{MFU} = \frac{\text{model FLOPs per step}}{\text{step time} \times \text{peak FLOPs} \times \text{num GPUs}}.$$

For dense SFT, well-tuned runs hit 40–55% MFU. RLHF is structurally lower — often 20–35% — because generation is memory-bandwidth bound (it cannot use the tensor cores fully) and because of the idle anti-phase. A *very* low MFU (single digits) usually means generation is running under the wrong layout (FSDP-during-decode) or the GPUs are idling on synchronization. MFU is the one number that tells you, honestly, how much of your expensive cluster is doing useful work.

**KL divergence and reward, the algorithmic health signals.** Beyond systems metrics, log `train/kl` (policy-to-reference KL) and `train/reward` every step. A KL that climbs unbounded means the policy is drifting and likely reward-hacking — the systems are fine but the run is failing. A reward that climbs while KL stays bounded is a healthy run. These are not systems metrics, but at scale a failing run wastes thousands of dollars of compute per hour, so catching divergence early *is* a systems concern.

```python
# Per-step instrumentation you should always have.
import time, torch

def timed_ppo_step(trainer, rollout, reward_fn, ref, batch):
    t = {}
    s = time.perf_counter()
    gen = rollout.generate(batch);            t["rollout"] = time.perf_counter() - s
    s = time.perf_counter()
    rew = reward_fn(gen);                      t["reward"]  = time.perf_counter() - s
    s = time.perf_counter()
    refl = ref.log_probs(gen);                 t["ref"]     = time.perf_counter() - s
    s = time.perf_counter()
    adv = compute_gae(gen, rew, refl);         t["gae"]     = time.perf_counter() - s
    s = time.perf_counter()
    stats = trainer.ppo_step(adv);             t["train"]   = time.perf_counter() - s

    # MFU: model_flops is 6 * params * tokens for the training fwd+bwd.
    tokens = batch.num_tokens
    model_flops = 6 * trainer.num_params * tokens
    peak = 312e12 * trainer.num_gpus          # A100 BF16 peak ~312 TFLOP/s
    mfu = model_flops / (sum(t.values()) * peak)
    return {**{f"time/{k}": v for k, v in t.items()},
            "perf/mfu": mfu, "train/kl": stats["kl"],
            "train/reward": stats["reward_mean"]}
```

#### Common bottlenecks and their fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `time/rollout` is 70%+ and trainer GPUs idle | synchronous, generation dominates | enable bounded-async; verify vLLM (not FSDP) does generation |
| `time/weight_sync` spikes to seconds | NCCL fell back to host transfer | pin collectives to NVLink/IB; check process placement |
| MFU in single digits | wrong generation layout / OOM thrash | move generation to vLLM TP; fix memory so no offload thrash |
| `train/kl` climbs unbounded | KL coef too low, reward hacking | raise `init_kl_coef`; use adaptive KL controller |
| Trainer waits on buffer | experience pipeline stalls | more rollout workers; prefetch next batch |
| OOM at start | four-model wall | ZeRO-3 + LoRA; co-locate reference; offload reward model |

The discipline is the same as any distributed system: measure the stages, find the dominant one, attack it, re-measure. In RLHF the dominant stage is almost always generation, which is why the entire field converged on "hand generation to a real inference engine and overlap it with training." Everything else is second-order.

## 12. Putting it together: a full async OpenRLHF-style loop

Let me assemble the pieces into one coherent training loop that reflects the real architecture — bounded-async, vLLM rollout, disaggregated reward and reference, ZeRO-3 trainer, instrumented.

```python
import ray, torch, time

@ray.remote(num_gpus=4)
class VLLMRolloutActor:
    def __init__(self, model_path):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.sp = SamplingParams(temperature=1.0, top_p=1.0,
                                 max_tokens=512, logprobs=1)
        self.version = 0
    def generate(self, prompts):
        outs = self.llm.generate(prompts, self.sp)
        return [pack_experience(o, self.version) for o in outs]
    def sync_weights(self, ipc_handles, version):   # zero-copy via NCCL/IPC
        self.llm.llm_engine.model_executor.update_weights(ipc_handles)
        self.version = version

@ray.remote(num_gpus=1)
class RewardActor:
    def __init__(self, rm_path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.m = AutoModelForSequenceClassification.from_pretrained(
            rm_path, torch_dtype=torch.bfloat16).cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(rm_path)
    @torch.no_grad()
    def score(self, texts):
        x = self.tok(texts, return_tensors="pt", padding=True,
                     truncation=True).to("cuda")
        return self.m(**x).logits.squeeze(-1).cpu().tolist()

def run(num_steps=2000, max_lag=1):
    ray.init()
    rollouts = [VLLMRolloutActor.remote(POLICY_PATH) for _ in range(4)]
    reward   = RewardActor.remote(REWARD_PATH)
    trainer  = build_zero3_trainer(POLICY_PATH, VALUE_PATH)  # FSDP/ZeRO-3
    ref      = build_reference(POLICY_PATH)                  # colocated, frozen
    buffer   = ExperienceBuffer(max_size=4096)
    version  = 0

    for step in range(num_steps):
        t0 = time.perf_counter()
        # Fan out generation across rollout workers (async, non-blocking).
        prompts = sample_prompts(batch=1024)
        futures = [r.generate.remote(chunk) for r, chunk
                   in zip(rollouts, split(prompts, 4))]
        for exp_list in ray.get(futures):
            for e in exp_list:
                buffer.put(e)

        batch = buffer.sample(1024, min_version=version - max_lag)
        texts = [decode(e) for e in batch]
        rewards = ray.get(reward.score.remote(texts))
        batch = attach_rewards_and_ref(batch, rewards, ref)
        batch = compute_gae(batch, gamma=1.0, lam=0.95, whiten=True)

        stats = trainer.ppo_step(batch, epochs=1, clip=0.2, kl_coef=0.02)
        version += 1

        # Broadcast new weights to all rollout workers (NCCL collective).
        handles = trainer.export_weight_handles()
        ray.get([r.sync_weights.remote(handles, version) for r in rollouts])

        log({"step": step, "time/iter": time.perf_counter() - t0,
             "train/kl": stats["kl"], "train/reward": stats["reward_mean"],
             "perf/mfu": stats["mfu"]})
```

Every architectural idea from this post is visible here: rollout workers as separate Ray actors running vLLM with tensor parallelism; the reward model disaggregated onto its own GPU; the reference co-located and frozen; the experience buffer with a freshness gate (`min_version=version - max_lag`); GAE with advantage whitening; the ZeRO-3 trainer doing the actual PPO update; and the weight broadcast back to rollout workers via NCCL handles rather than a slow host transfer. The `max_lag=1` makes it bounded-async. This is, in skeleton, what OpenRLHF runs.

## 13. GRPO and the reasoning-model shift

The systems story above was written for PPO, but the most consequential change in RLHF systems since 2024 came from a different algorithm: **Group Relative Policy Optimization (GRPO)**, popularized by reasoning-model training. GRPO matters here not because the math is exotic but because it *changes the systems bottleneck*, and any current treatment of distributed RLHF that ignores it is out of date.

The key move in GRPO is to **drop the value model entirely.** Instead of training a separate critic to estimate a baseline for advantage computation, GRPO samples a *group* of $G$ responses for each prompt (say $G = 8$ or $16$), scores all of them with the reward model, and uses the group's mean reward as the baseline. The advantage of response $i$ in a group is simply its reward minus the group mean, normalized by the group standard deviation:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}.$$

The systems consequence is immediate and large. Recall the four-model memory wall: the value model carried a full $16P$ bytes of training state, the same as the policy. GRPO deletes that model. The memory accounting drops from four models to three (policy, reference, reward), and the policy is the only one with the expensive $16P$ training state. For a 7B run, removing the value model's 112 GB of training state is the difference between needing four GPUs and needing two or three. This is a substantial simplification of exactly the constraint we spent section 2 deriving.

But GRPO gives with one hand and takes with the other. Because the baseline now comes from a *group* of sampled responses rather than a learned critic, you must generate $G$ times more rollouts per prompt. Generation was already the dominant cost (sections 1 and 11), and GRPO multiplies it. So GRPO trades the value model's memory and training cost for a $G$-fold increase in generation cost — which is precisely why the frameworks that win at GRPO are the ones with the fastest rollout engines and the best rollout/train overlap. veRL's strength at reasoning-model RL comes directly from this: when generation is $8\times$ more dominant, the HybridEngine's elimination of weight-transfer overhead and its efficient resharding pay off even more, and any framework that can run many parallel vLLM rollout workers (OpenRLHF) shines.

```python
# GRPO advantage: no value model, baseline is the group mean.
import torch

def grpo_advantages(rewards, group_size):
    # rewards: [num_prompts * group_size]
    r = rewards.view(-1, group_size)               # [num_prompts, G]
    baseline = r.mean(dim=1, keepdim=True)         # group mean per prompt
    adv = (r - baseline) / (r.std(dim=1, keepdim=True) + 1e-8)
    return adv.view(-1)                            # back to flat per-response

# Rollout must now produce G responses per prompt — generation cost x G.
prompts = sample_prompts(batch=256)
expanded = [p for p in prompts for _ in range(8)]  # G = 8
responses = rollout.generate(expanded)             # 256 * 8 = 2048 generations
rewards = reward_model.score(responses)
advantages = grpo_advantages(torch.tensor(rewards), group_size=8)
```

#### Worked example: GRPO memory versus generation trade

Take the 7B run from section 2 and switch PPO to GRPO. The value model's $112$ GB of training state vanishes, dropping the model-state floor from ~252 GB to ~140 GB (policy training state 112 GB + reference 14 GB + reward 14 GB). That alone can take a four-GPU requirement down to two. But generation now produces $G = 8$ responses per prompt. If a PPO rollout of 1024 prompts took 18 s on the vLLM fleet, a GRPO rollout of the same 1024 prompts at $G = 8$ generates 8192 responses and takes on the order of $8\times$ longer — roughly 140 s — unless you scale out the rollout fleet correspondingly. So the practical GRPO recipe is: spend the GPUs you *saved* by deleting the value model on *more rollout workers*, because generation is now even more of the budget. The net effect for reasoning-model RL, where long chains of thought make each generation expensive, is that the system becomes almost entirely a generation-throughput machine with a comparatively small training tail — the inverse of the SFT mental model, and the strongest possible argument for the rollout/trainer separation this whole post is built on.

This shift is also why the line between RLHF frameworks and inference-serving frameworks is blurring: when 80–90% of an RL run is generation, the quality of your inference engine (continuous batching, paged attention, speculative decoding, prefix caching for shared prompts) determines your training throughput. The best reasoning-model RL setups treat the rollout fleet as a first-class inference-serving problem, applying every serving optimization to it, because a 2× faster rollout engine is a nearly 2× faster training run.

## Case studies

The history of distributed RLHF tooling, sketched in Figure 6, reads as a steady march from research scripts toward throughput-optimized engines — each milestone a better answer to "how do we keep generation fast and the rest of the GPUs busy."

![A timeline of distributed reinforcement learning for large language models, running from InstructGPT in 2022 through the TRL library, DeepSpeed-Chat in 2023, OpenRLHF, veRL with its HybridEngine in 2024, and OpenRLHF v2 with async rollout.](/imgs/blogs/distributed-rlhf-system-design-6.png)

**InstructGPT (Ouyang et al., 2022).** The paper that put RLHF on the map aligned GPT-3 (175B) with PPO against a learned reward model, and a 1.3B InstructGPT model was preferred by human labelers over the 175B base GPT-3 on their prompt distribution. The systems lesson buried in the paper is that they were already wrestling the four-model problem at 175B scale, with the reward model and reference policy as separate components — the architecture we have been deriving was, in essence, born here, even if the public tooling to reproduce it cheaply came later.

**DeepSpeed-Chat (Microsoft, 2023).** The first widely-used open framework to make end-to-end RLHF (SFT → reward modeling → PPO) accessible, using ZeRO to shard the training state and a "Hybrid Engine" of its own to switch the policy between training and generation modes. It demonstrated RLHF on models up to tens of billions of parameters on commodity clusters and reported order-of-magnitude speedups over naive HuggingFace+DeepSpeed baselines, largely by not generating under the training layout — the same core insight veRL later refined.

**OpenRLHF (2023–2024).** Built the Ray + vLLM + ZeRO-3 architecture we read in section 5, and reported the ability to run 70B PPO and meaningful throughput gains over DeepSpeed-Chat at 7B, with vLLM-accelerated generation as the headline win. Its async-rollout addition (the "v2" lineage) brought bounded asynchrony to the open-source world.

**veRL / HybridFlow (ByteDance, 2024).** The HybridEngine resharding design (section 6) and the single-controller programming model. Its published results report strong throughput at scale by eliminating the weight-transfer cost between rollout and training, and it has become a common choice for very large RL runs, including reasoning-model RL (GRPO and variants) where rollout cost is even more dominant because chains of thought are long.

A throughline across all four: the bottleneck was always generation, and each generation of tooling found a better way to keep generation fast and the rest of the GPUs busy. The algorithm (PPO) barely changed; the systems engineering changed everything about what scale was reachable.

## When to use distributed RLHF (and when not to)

Be honest about whether you need any of this machinery. The decision tree in Figure 8 is the short version; here is the reasoning.

![A decision tree for choosing an RLHF framework: from the root, branch on whether the model exceeds 13B leading to OpenRLHF for async or veRL for a hybrid engine, or on whether it is a single-node run leading to TRL for a prototype or OpenRLHF and veRL for production scale.](/imgs/blogs/distributed-rlhf-system-design-8.png)

**Reach for distributed RLHF (OpenRLHF / veRL) when:** your policy is larger than ~13B and will not fit the four-model state on a single node even with LoRA; you need full fine-tuning (not LoRA) at scale; you are running long-rollout RL (reasoning models, GRPO) where generation cost is extreme and you must overlap it with training; or you are in production and throughput-per-dollar matters more than setup simplicity.

**Use TRL (single node, possibly LoRA) when:** your model fits the four-model state on the GPUs you have (8B with LoRA on one 80GB card is comfortable); you are prototyping and want to iterate on reward shaping and data in minutes; or you are doing DPO, where there is no generation-during-training and no reward/value model, so the heavy frameworks are pure overhead.

**Consider skipping online RL entirely when:** DPO or another offline preference method gets you the alignment you need. DPO removes the reward model, the value model, and the generation loop — it is supervised learning on preference pairs with a clever loss, scales like SFT, and is dramatically cheaper and more stable. Many teams found that for straightforward preference tuning, DPO matches PPO-RLHF at a fraction of the systems cost. Online PPO/GRPO earns its complexity when you need genuine on-policy exploration — the model trying new behaviors and getting scored on them — which offline methods cannot provide. If your reward signal is a fixed preference dataset, prefer DPO; if your reward signal is an interactive judge the model must learn to satisfy through exploration, you need online RL and therefore this whole architecture.

**A blunt cost note.** A 70B online RLHF run can burn a serious amount of GPU-hours — the kind of spend where a 30% throughput improvement from enabling async is worth many thousands of dollars over a campaign. That economic reality is why the systems engineering in this post is not academic: at scale, the architecture choices are budget choices.

## Key takeaways

- **RLHF is a four-model problem, and that is the whole story.** Policy, reference, reward, and value live in memory together; the $16P$-byte training-state rule applied to the policy and value alone blows past a single 80GB GPU at 7B. ZeRO-3, LoRA, and per-model disaggregation are the three escapes.
- **Generation is the bottleneck, always.** Autoregressive decode is sequential and memory-bandwidth bound; it dominates step time. The universal fix is to hand generation to a real inference engine (vLLM) under tensor parallelism, never to generate under the training (FSDP) layout.
- **Separate rollout workers from the trainer** so each gets its optimal weight layout; the new plumbing this adds is fast weight synchronization from trainer to rollout, which wants NVLink/NCCL, not host transfers.
- **Synchronous RLHF has zero staleness but ~50% utilization; async overlaps everything but adds staleness.** Bounded async (off-by-one) recovers most of the utilization with negligible gradient-quality loss, because the PPO ratio $r_t$ stays near 1 when lag is small.
- **OpenRLHF disaggregates (multi-controller, weights move); veRL co-locates (HybridEngine, weights reshard in place).** The choice turns on whether weight-sync cost or rollout/train overlap dominates your run.
- **Tune parallelism separately for rollout and training:** TP for low-latency generation, ZeRO-3 (+modest TP/PP) for memory-bound training. Reusing the training config for generation is the classic mistake.
- **The buffer values freshness over capacity.** Unlike a DQN replay buffer, it holds one (or a few) on-policy batches and discards them; a freshness gate bounds staleness.
- **Measure per-stage time, GPU anti-phase, and MFU.** RLHF MFU of 20–35% is normal; single digits means generation is in the wrong layout or the GPUs are idling on sync.
- **Prototype in TRL, scale in OpenRLHF/veRL, and consider DPO before any of it** — if your reward is a fixed preference dataset, offline DPO may give you the alignment without the four-model machinery at all.

## Further reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — the canonical RLHF recipe and the origin of the four-model PPO pipeline at scale.
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences," 2019 — the earlier work that established the reward-model + KL-to-reference structure this whole post relies on.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — the clipped surrogate objective and trust-region intuition behind the staleness analysis.
- Rafailov et al., "Direct Preference Optimization," 2023 — the offline alternative that removes the reward and value models; read it to know when you can skip distributed RLHF entirely.
- Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework," 2024 — the Ray + vLLM + ZeRO-3 architecture in detail.
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (veRL), 2024 — the single-controller model and HybridEngine resharding.
- Yao et al., "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales," 2023 — ZeRO-based RLHF and the original hybrid generation/training engine.
- Within this series: the unified map `reinforcement-learning-a-unified-map` for where RLHF sits among RL methods, the capstone `the-reinforcement-learning-playbook` for the decision framework, and `/blog/machine-learning/training-techniques/` for the reward-modeling and DPO companions that feed this pipeline.

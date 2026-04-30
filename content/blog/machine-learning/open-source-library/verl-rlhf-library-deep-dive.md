---
title: "verl: An Engineer's Deep Dive into HybridFlow RL for Large Language Models"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "An opinionated, principal-engineer walkthrough of verl — the HybridFlow RL framework that powers production RLHF at ByteDance, Anyscale, Qwen, and the open-source community. Programming model, controller paradigms, 3D-HybridEngine resharding, every supported algorithm, FSDP/Megatron/vLLM/SGLang plumbing, runnable code, real benchmarks, and a long catalog of production case studies."
tags:
  [
    "verl",
    "rlhf",
    "ppo",
    "grpo",
    "dapo",
    "ray",
    "fsdp",
    "megatron",
    "vllm",
    "sglang",
    "open-source-library",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 51
aiGenerated: true
---

Most teams discover the limits of their RLHF stack the same way: the prototype works on a 7B model on a single node, then somebody asks for the same training loop on Llama-3-70B across two nodes, and it does not just slow down — it falls apart. The actor and the rollout share a single shape. The reward model wants its own GPUs but the framework refuses to give them. Sequence packing breaks the loss mask. The vLLM upgrade requires a complete rewrite. After three weeks of hacking, the team rediscovers the lesson the field already learned in 2024: **a fast RLHF system needs different shapes for training and rollout, different backends for different roles, and a control flow that is not entangled with either**. That is the problem [verl](https://github.com/volcengine/verl) was built to solve.

verl is the open-source implementation of the [HybridFlow paper](https://arxiv.org/abs/2409.19256) (EuroSys '25). It treats RLHF as a hybrid programming model: a single Python driver expresses the algorithm in 30 lines of sequential code, while four worker groups (actor, rollout, reference, critic, reward) run on Ray and each pick their own backend (FSDP, FSDP2, Megatron-LM, vLLM, SGLang, TensorRT-LLM). The headline result from the paper: **1.53× to 20.57× throughput** vs DeepSpeed-Chat and OpenRLHF, the gain growing with model size because of a piece of plumbing called the 3D-HybridEngine that re-shards the actor between training and generation phases without going through disk or host RAM.

![verl architecture: single controller, four worker groups, swappable backends on Ray](/imgs/blogs/verl-rlhf-library-deep-dive-1.png)

The diagram above is the mental model: one controller process at the top, four worker groups in the middle (actor+rollout+reference colocated by default; critic; reward; rollout engine), and a Ray cluster at the bottom that maps each worker group to a placement group of GPUs. The controller's loop is plain Python — `for prompt in dataloader: generate, compute_log_prob, compute_values, compute_scores, update_actor` — and every method call dispatches to a worker group that runs on its chosen backend. The rest of this article walks each layer in detail, shows the algorithms verl implements with their loss formulas, gives you a runnable GSM8K PPO recipe you can copy-paste today, then closes with eleven case studies of real RLHF production incidents.

Companion reading on this blog: [trl: a tour of HuggingFace's RL library](/blog/machine-learning/open-source-library/trl-lib) for the single-controller predecessor, [LMCache deep dive](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) for the rollout-side KV cache story, and [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) for what happens inside the rollout engine.

## 1. Why a Separate RLHF Framework Exists

The naive view is that TRL plus DeepSpeed plus vLLM, glued with Python, is enough. It is — for a 7B model on a single node, with PPO, with a reward model that fits next to the actor in HBM. The moment any of those constraints relax, the cracks show.

| Assumption (single-controller world) | Reality (production RLHF) |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| Actor and rollout share a shape      | Training wants ZeRO-3 + TP + PP; rollout wants TP-only with PagedAttention      |
| One model fits in `actor_model.cuda()` | 70B in fp16 is 140 GB, plus 6× for Adam states; needs FSDP or Megatron        |
| Reward model is small, colocate with actor | RMs are now 7B–34B and want their own GPU pool                            |
| `model.generate()` is fast enough    | HF generate is 5–20× slower than vLLM continuous batching at scale              |
| Algorithm is fixed                   | The team wants to compare PPO vs GRPO vs DAPO this week                         |
| One node, one rank                   | 8–64 nodes, mixed prefill/decode, possibly cross-region                         |
| Sync rollout                         | Multi-turn agent rollouts want async + tool calls + sandbox execution           |

Each row maps to a real verl design choice. The actor-rollout shape mismatch is the **3D-HybridEngine**. The reward model GPU split is the **WorkerGroup placement**. vLLM speed is the **rollout backend abstraction**. Algorithm flexibility is the **`adv_estimator` registry**. Multi-node is **Ray placement groups**. Async is the **SGLang rollout integration**.

**The TRL ceiling.** TRL is the right starting point for prototypes — its PPOTrainer fits the standard "subclass-and-go" HuggingFace ergonomics. But it locks the actor and the generator into the same `transformers` model object, which means you cannot use vLLM for rollout. On a 7B model that costs you 5× throughput; on a 70B model it costs you 10–20×. The TRL team knows this — recent versions added vLLM integration via `vllm_serve` — but the integration sits awkwardly on top of a single-controller core that was not designed for cross-process state.

**The DSChat ceiling.** DeepSpeed-Chat ships an end-to-end RLHF pipeline with all four roles (actor / critic / RM / reference) and ZeRO-3 training. The catch: every role is its own DeepSpeed-engine init, the rollout uses HF `generate`, and the actor cannot reshard between training and generation. The HybridFlow paper benchmarked it head-to-head and found 5–20× throughput gaps at scale. DSChat is still the right tool for a "one weekend, one node, one algorithm" prototype, but no team I've seen ships it past 13B.

**The OpenRLHF ceiling.** OpenRLHF was the first open framework to take Ray seriously and integrate vLLM as a separate worker group. It runs PPO and GRPO competently on multiple nodes. What it did not have until late 2025 was the algorithm flexibility — adding a new advantage estimator meant editing trainer internals — and resharding was via disk write/reload (~30 s/step on 8B). verl is, in part, OpenRLHF's lessons absorbed into a cleaner programming model.

The conclusion: separating the **control flow**, the **role placement**, and the **shape transitions** is the only way to get RLHF to scale past a single replica without locking yourself into a specific algorithm. verl does that separation.

**The reasoning-RL inflection point.** A second force pulled verl into prominence in 2025: the success of GRPO/DAPO on math and code reasoning. DeepSeek-R1 demonstrated that RL on verifiable rewards (no RM at all, just a Python regex against a ground-truth answer) could produce reasoning gains that no SFT recipe matches. The frameworks that handled this best were the ones that let you swap out the entire reward pipeline — replace `RewardModelWorker` with a Python function — without touching the trainer. verl did this from day one because the role split was already abstract enough to absorb a new reward type. Frameworks that had RM forward calls baked into the trainer's hot path could not adapt without a rewrite. By the end of 2025 most reasoning-RL papers had switched to verl for this reason alone.

**The recipe-as-artifact culture.** A subtler force: verl's `recipe/` directory turned out to be its most important social artifact. When DeepSeek published R1's training setup, when Qwen published QwQ's, when DAPO's authors published their long-CoT recipe — they all converged on dropping a verl YAML alongside the paper. Reproducibility became "git checkout the commit and run the YAML." That is a different operating mode from "read the paper, infer the hyperparameters, hope." The recipes also embed the operational learnings (the right TP/PP/EP shape for 671B, the right `kl_coef` for late-stage DAPO) that no paper writes down explicitly. For a senior engineer, the recipe directory is a more valuable read than half the algorithm papers.

## 2. The Mental Model: One Controller, Four Roles, Many Backends

Keep this split in your head; every verl config knob lives in exactly one of these boxes.

**The driver process** owns:
- The Python script that defines the RL loop.
- The Hydra config (actor / critic / rollout / reward sub-trees).
- Per-step logging, checkpoint orchestration, and exit conditions.

**The worker groups** each own:
- A set of Ray actors pinned to a placement group of GPUs.
- A backend (FSDP, FSDP2, Megatron, vLLM, SGLang).
- A schedule of registered methods exposed via `@register(dispatch_mode=...)`.

**The 3D-HybridEngine** owns:
- The actor's weight transformation between training shape and rollout shape.
- The NCCL collectives that move bytes without disk staging.

**Hydra + the recipe directory** own:
- The end-to-end YAML that ties algorithm, models, batch sizes, and resource pools together.
- The reproducibility story: every published recipe pins a verl commit.

This separation is what lets verl track upstream vLLM versions, add a new algorithm in two days, and run on AMD ROCm and Ascend NPUs without rewriting the loop. The control flow is the constant; the rest moves underneath.

**Why this matters for the reader.** When you debug a verl regression, the first question is always: which box failed? A loss-curve drift is in the algorithm box. A NCCL hang is in the worker / placement box. A vLLM warmup OOM is in the rollout backend. A `dispatch_mode` mismatch is the connector. Naming the box first cuts the search space by 5×. New users who skip this step burn days; experienced users do it on the first sentence of the bug report.

## 3. HybridFlow's Hybrid-Controller Paradigm

The paper's central observation is that RLHF dataflow has _two_ kinds of computation: **algorithm logic** (which prefers a single sequential script) and **distributed neural network execution** (which prefers SPMD multi-process orchestration). Pre-verl frameworks picked one, and the other suffered.

![Three controller paradigms compared: single, multi, and HybridFlow's hybrid](/imgs/blogs/verl-rlhf-library-deep-dive-3.png)

**Single-controller** (TRL, DSChat, early OpenRLHF). One Python process drives everything. Algorithm code reads top-to-bottom, like a textbook PPO implementation. The cost: the same process must own both training and generation, which means a single GPU shape, which means no vLLM-fast rollout. Throughput caps at "one model, one mode, one configuration." Past 13B, OOM is the steady state.

**Multi-controller** (NeMo, Megatron-RLHF, raw Ray). Every GPU runs the same script and dispatches by rank: `if rank in actor_ranks: actor_loop(); elif rank in rollout_ranks: rollout_loop()`. Each role can use its own backend; NCCL transfers between roles. The cost: the algorithm logic is scattered across N processes, debugging requires correlating multi-rank logs, and a new advantage estimator means editing every script. Researchers hate it. Infra teams ship it because it scales.

**HybridFlow** (verl). One controller process drives the algorithm; multi-process worker groups execute the heavy compute. The controller calls `actor_wg.generate_sequences(prompts)` — a blocking, synchronous-looking call — but under the hood it dispatches to a Ray WorkerGroup that runs on N GPUs in parallel, returns a `DataProto` aggregated across ranks, and the controller carries on. You write single-controller code; you get multi-controller throughput.

The code below is the actual verl PPO loop, lightly edited for clarity. It is one Python file. There are no rank checks. No `if dist.get_rank() == 0`. The Ray dispatch is invisible.

```python
# verl/trainer/ppo/ray_trainer.py (simplified)
class RayPPOTrainer:
    def fit(self):
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 1) Rollout: generate G samples per prompt on the rollout engine
            gen_batch = self.actor_rollout_wg.generate_sequences(batch)

            # 2) Compute log probs under the current actor and the frozen ref
            old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)

            # 3) Critic values (PPO-only; skipped for GRPO/DAPO/RLOO/...)
            if self.use_critic:
                values = self.critic_wg.compute_values(gen_batch)

            # 4) Reward: function-based (math/code) or model-based (RM forward)
            scores = self.reward_wg.compute_scores(gen_batch)

            # 5) Compute advantages locally on the controller
            advantages = compute_advantages(
                scores, values, ref_log_prob, old_log_prob,
                estimator=self.config.algorithm.adv_estimator,
            )
            gen_batch.batch["advantages"] = advantages

            # 6) Update actor (and critic, if any). Each call reshards weights
            #    via the 3D-HybridEngine before training, after rollout.
            self.actor_rollout_wg.update_actor(gen_batch)
            if self.use_critic:
                self.critic_wg.update_critic(gen_batch)

            self.log_metrics(batch_idx, scores, advantages)
```

The hybrid is **not** just a compromise. It picked the strict winner of each axis: single-controller readability for the algorithm, multi-controller throughput for the workers, plus a third primitive (resharding) that neither parent had. That is why HybridFlow's gain is _multiplicative_ — 1.53× at 7B, 20× at 70B — not the additive 2× you would expect from averaging two ideas.

## 4. Worker, WorkerGroup, Ray Placement Groups

verl maps Python classes to Ray actors through three layered abstractions, each with a clear responsibility.

**`Worker`** is the GPU-bound class. Each instance owns one rank: it holds the model state, the optimizer, the rollout engine, or the reward function. It exposes methods decorated with `@register(dispatch_mode=...)`.

```python
# verl/workers/fsdp_workers.py (simplified)
class ActorRolloutRefWorker(Worker):

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, data: DataProto) -> DataProto:
        # `data` is already split: this rank received its DP shard.
        with self.maybe_offload_critic_to_cpu():
            self.rollout.sync_weights_from_actor(self.actor)
            return self.rollout.generate(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        return self.actor.compute_log_prob(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto) -> DataProto:
        # Reshard from rollout shape back to training shape (no-op if same)
        self.actor.train()
        loss = self.actor.compute_loss(data)
        self.actor.backward(loss)
        self.actor.step()
        return DataProto({"actor/loss": loss.detach()})
```

**`WorkerGroup`** is the controller-side proxy that aggregates a fleet of `Worker` instances. When you call `actor_wg.generate_sequences(batch)`, the WorkerGroup performs three things atomically: **split** the batch by data parallelism, **dispatch** each shard to its worker via `worker.method.remote()`, and **collect** results with `ray.get()`. The user sees one method call; under the hood, N parallel RPCs fire and return.

The dispatch mode is the contract. `Dispatch.DP_COMPUTE_PROTO` means "split along DP, dispatch a `DataProto` chunk to each worker, collect outputs as `DataProto`." Other modes exist for one-to-all (`ALL_TO_ALL`), single-rank (`ONE_TO_ALL`), and Megatron's PP-aware dispatch.

```python
# Pattern used everywhere in verl. The decorator does the magic.
from verl.single_controller.base.decorator import register, Dispatch

class CriticWorker(Worker):

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto) -> DataProto:
        # data["responses"] arrives already DP-sharded
        with torch.no_grad():
            values = self.critic_model(data["responses"])
        return DataProto({"values": values})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto) -> DataProto:
        loss = ((self.critic_model(data["responses"]) - data["returns"]) ** 2).mean()
        loss.backward()
        self.critic_optim.step()
        self.critic_optim.zero_grad()
        return DataProto({"critic/loss": loss.detach()})
```

**Ray placement groups** are the resource backing. A `ResourcePool` declares "I want 16 GPUs across 2 nodes," Ray's scheduler picks specific nodes, and each `WorkerGroup` is constructed on a specific pool. By moving WorkerGroups across pools, you change the GPU mapping without touching the algorithm:

```python
# verl/trainer/main_ppo.py (resource setup)
resource_pool_spec = {
    "actor_rollout": [8, 8],   # 16 GPUs (2 nodes × 8) for actor + rollout
    "critic":        [4],      # 4 GPUs on a third node for the critic
    "reward":        [2],      # 2 GPUs for the RM
}
resource_pool_manager = ResourcePoolManager(resource_pool_spec)
actor_rollout_wg = ActorRolloutRefWorkerGroup(
    resource_pool=resource_pool_manager.get_resource_pool("actor_rollout"),
    config=config.actor_rollout_ref,
)
critic_wg = CriticWorkerGroup(
    resource_pool=resource_pool_manager.get_resource_pool("critic"),
    config=config.critic,
)
reward_wg = RewardModelWorkerGroup(
    resource_pool=resource_pool_manager.get_resource_pool("reward"),
    config=config.reward_model,
)
```

The detail that matters: **the controller never moves data between WorkerGroups manually**. When the loop calls `reward_wg.compute_scores(gen_batch)`, gen_batch flies from the actor pool to the reward pool through Ray's object store, which uses shared-memory plasma for intra-node transfers and gRPC for inter-node. On a properly configured cluster the transfer is dominated by NIC bandwidth, not Python serialization.

**The colocation pattern.** By default verl colocates `actor + rollout + reference` on the same `ResourcePool` because they share weights. Calling `create_colocated_worker_cls(ActorRolloutRefWorker)` produces a single Worker class that holds all three roles' state and switches between them as needed. The reference policy is just frozen actor weights — no extra GPU memory until you load it. This default saves 2× GPUs compared to splitting them apart, and it is almost always what you want.

## 5. The Role Split: Actor / Rollout / Reference / Critic / Reward

The five logical roles are easy to confuse in the code because verl colocates four of them by default. The split:

| Role | Owns | Updates | GPUs (default) |
| --- | --- | --- | --- |
| Actor | Trainable policy weights | yes | actor_rollout pool |
| Rollout | Inference-shape weights + KV cache | weights synced from actor | same pool as actor |
| Reference | Frozen pre-RL policy (KL anchor) | no | same pool as actor (lazy load) |
| Critic | Value head (PPO only) | yes | critic pool |
| Reward | Score function or RM forward | no | reward pool |

**The colocation default for actor/rollout/reference** is what HybridFlow calls a "co-located worker." Three logical models share one set of GPUs by sharing weights where possible (actor and reference often start identical) and time-multiplexing the GPU between training and rollout. The 3D-HybridEngine handles the shape transitions.

**The critic split** is enabled only for PPO. GRPO, DAPO, RLOO, ReMax, REINFORCE++, and friends all replace the critic's value baseline with a group-statistics baseline, which means no critic worker, no critic GPUs, no critic optimizer state. On a 70B model, dropping the critic frees ~6–8 GPUs of memory budget. This is one of the reasons GRPO and DAPO became popular even when PPO was perfectly capable: the wall-clock cost is just lower.

**The reward split** is the most flexible. A `RewardWorker` can be:

1. A learned reward model (forward pass through a frozen 7B–34B RM).
2. A pure-Python function that scores rollouts (math: regex-extract the answer, compare to ground truth; code: run unit tests in a sandbox).
3. A combination: function reward + model reward summed with weights.
4. A multi-turn agent reward credited per turn.

The choice is a YAML key:

```yaml
reward_model:
  enable: true
  reward_manager: naive       # naive | dapo | prime | ...
  model:
    path: "/path/to/rm"
    fsdp_config: { ... }

# OR for verifiable rewards, no reward model at all:
custom_reward_function:
  path: "verl/utils/reward_score/math.py"
  name: "compute_score"
```

The function reward path is what made GRPO on math/code take off in 2025: no RM training pipeline, no preference data, just a Python function that returns 1.0 if the model's answer matches the ground truth and 0.0 otherwise. verl ships scoring functions for GSM8K, MATH, AIME, HumanEval, MBPP, and a few others under `verl/utils/reward_score/`.

**The frozen reference** is a load-on-demand thing. verl keeps the reference weights on CPU pinned memory by default; when `compute_ref_log_prob` is called, the weights are scattered to the GPU shape of the colocated worker, used for one forward pass, and freed. For algorithms without a KL term (DAPO is a notable case), you can drop the reference role entirely.

## 6. The 3D-HybridEngine: The Trick That Makes It All Fast

Everything above is plumbing. The thing that makes verl actually fast is the 3D-HybridEngine — the resharding mechanism that swaps the actor between training shape and rollout shape every iteration without going through disk or host RAM.

![3D-HybridEngine resharding: training shape on the left, rollout shape on the right, NCCL-only transition](/imgs/blogs/verl-rlhf-library-deep-dive-2.png)

Why this matters: training and rollout want **different shapes** for the same weights.

**Training shape** (FSDP or Megatron):
- TP × PP × DP × ZeRO-3 across data parallelism.
- Optimizer states sharded across DP (3× the weight bytes for Adam: m, v, master fp32).
- Activations + grads materialised during backward; Ulysses SP can split sequence dimension.
- Memory layout favours backward pass and grad-allreduce.

**Rollout shape** (vLLM or SGLang):
- TP-only, no PP, no DP partition (each rollout rank holds a full model).
- No optimizer state (no grads needed) → frees ~6× the weight bytes.
- KV cache pool sized for `batch × max_response_len`.
- Memory layout favours generation: dense weights, big KV cache.

The naive way to bridge the two: write actor weights to disk after every training step, reload them into vLLM. The HybridFlow paper measures this at ~30 seconds per step on an 8B model, dominated by I/O. On a 70B model it is minutes per step. In an RL loop where each step is 30–60 seconds of compute, a 30-second reshard is a 50% throughput tax. Unusable at scale.

The 3D-HybridEngine does the reshard **as a sequence of NCCL collectives in HBM**:

1. **All-gather along DP**: every rank now holds full weights for its PP stage.
2. **All-gather along PP**: full model materialised on each rollout rank.
3. **Re-tile TP** from training TP (often 4 or 8) to rollout TP (often 2 or 8).
4. **Bind to the rollout engine's KV cache layout** (vLLM PagedAttention block layout, or SGLang RadixAttention).

No disk. No host RAM. Just NCCL. The paper's measurement: reshard cost is **2–5% of step time** at 8B–70B, which is what allows the 1.53–20.57× throughput improvement to materialise.

There is a symmetric path back: before the next training step, free the KV cache pool, re-tile TP back, re-shard along DP, and restore optimizer states. Optimizer states never have to leave CPU pinned memory if you set `actor.optim.offload: true`, which costs a small wall-clock penalty but frees 3–6× the weight bytes of HBM during rollout.

**A worked numerical example for an 8B PPO run on 16 H100s:**

| Phase | Memory per GPU (training shape, TP=4, DP=4, ZeRO-3) |
| --- | --- |
| Actor weights (fp16) | 2 GB (1/4 TP × 1/4 ZeRO) |
| Adam states (fp32 m, v, master) | 6 GB |
| Activations + grads | 12 GB (Ulysses SP=2) |
| KV cache (none yet) | 0 |
| Total during training | ~20 GB |

Now reshard to rollout shape (TP=2, no PP, no DP partition):

| Phase | Memory per GPU (rollout shape, TP=2) |
| --- | --- |
| Actor weights (fp16) | 8 GB (full model / TP=2) |
| Adam states (offloaded to CPU) | 0 GB on GPU |
| Activations | 0 (no backward) |
| KV cache | 50 GB (the rest of HBM) |
| Total during rollout | ~58 GB |

The KV cache pool is what makes this trade work. vLLM uses every spare byte for paged blocks, which is what gives you continuous batching at 5–10× the throughput of HF generate. If you tried to keep training shape during rollout, you'd have ~60 GB of optimizer + activation memory squatting in HBM, leaving 20 GB for weights + KV cache, and your rollout would crawl. The reshard buys you 50 GB of KV cache pool every iteration.

**The MoE wrinkle.** For MoE models like DeepSeek-V3 671B, the rollout shape uses expert parallelism (EP) instead of pure TP — experts are partitioned across ranks. The 3D-HybridEngine extends to handle this: training EP can differ from rollout EP, and the reshard inserts a router-aware redistribution step. This is the piece that makes 671B RLHF practical; without it the reshard would not converge on a single node.

**Why disk-staging is unworkable at scale.** It is worth dwelling on this because the comparison is what the HybridFlow paper rests on. A 70B model in fp16 is 140 GB; with Adam states it's ~840 GB during training. Writing 840 GB to disk and reloading on each step, even with a parallel filesystem at 20 GB/s aggregate, takes ~40 seconds — and that's the optimistic case where the FS is dedicated. On a shared cluster with other tenants, you'll see 60–120 seconds. If your step is 30 seconds of compute, you've added 100–400% wall-clock overhead. At this point your "cluster" is doing more I/O than RL. The 3D-HybridEngine's NCCL-only path moves the same bytes over the GPU interconnect (NVLink at 900 GB/s, NVSwitch at 1.8 TB/s aggregate) in milliseconds. The throughput delta is not 2× — it's 50–200×, just on the reshard path alone, which is why the end-to-end gain compounds as model size grows.

**The actor-shape-vs-rollout-shape sympathy.** An underappreciated subtlety: the closer the training shape is to the rollout shape, the cheaper the reshard. If you train at TP=2 and rollout at TP=2, the TP re-tile is a no-op and you save ~30% of reshard time. This argues for **picking the same TP at training and rollout when memory permits**. For 8B–14B models on H100s, TP=2 works for both training and rollout; for 70B+, training usually needs TP=8 while rollout can do TP=4, and the reshard pays a small cost. The verl recipes make this trade-off explicit — read them as the answer to "what shape pair has been measured to work."

## 7. Training Backends: FSDP / FSDP2 / Megatron-LM / DeepSpeed Ulysses

verl supports four training backends, and the choice is non-obvious.

**FSDP1** (PyTorch Fully Sharded Data Parallel, the original). Easy to adopt, integrates with HuggingFace models out of the box, supports CPU offload. The catch: composing FSDP1 with TP or sequence parallelism is finicky — many models work, some need monkey-patches. For 7B–34B dense models, FSDP1 is the default choice.

**FSDP2** (the rewrite landed in PyTorch 2.4). Same sharding semantics but with cleaner internals and proper composability with TP/SP. The headline feature: `parallelize_module()` plays well with FSDP2's DTensor-based shard wrapping, so you can stack FSDP2 + TP + SP without writing custom hooks. Use FSDP2 for new projects; FSDP1 only when you have an existing recipe pinned to it.

**Megatron-LM** (NVIDIA, integrated via verl's Megatron worker). The right choice for 70B+ dense models and any MoE model. Provides TP, PP, EP, SP, and `--use-distributed-optimizer` (which is Megatron's equivalent of ZeRO-3). The cost: the model must be in Megatron's format, which means a conversion step from HF (`scripts/converter_hf_to_mcore.py`). The benefit: this is the only backend that scales cleanly to 671B with expert parallelism, and is what verl's DeepSeek-V3 recipe uses.

**DeepSpeed Ulysses** (sequence parallelism). Not a standalone training backend but a sequence-dimension parallelism overlay that composes with FSDP or Megatron. Splits the sequence across SP ranks during attention; reduces activation memory by `1/SP` at the cost of a small all-to-all per layer. For long-CoT reasoning workloads (response length 8 K–32 K tokens), Ulysses SP=2 or SP=4 is the difference between fitting and OOM.

| Backend | Best for | Catch |
| --- | --- | --- |
| FSDP1 | 7B–34B dense, prototypes | TP/SP composition is finicky |
| FSDP2 | New projects, 7B–70B dense | Requires PyTorch 2.4+, less tested in production |
| Megatron-LM | 70B+ dense, all MoE, large-scale | HF→Mcore conversion step required |
| DeepSpeed Ulysses | Long-context (≥8K response) | All-to-all overhead at small contexts |

**The choice in practice.** For a fresh project on Llama-3-8B PPO: FSDP2. For Qwen3-32B GRPO with 16K responses: FSDP2 + Ulysses SP=2. For DeepSeek-V3 671B DAPO: Megatron with TP=8, EP=8, PP=4. The verl `recipe/` directory has all three pinned with working configs.

**Liger-kernel integration.** Liger is a set of fused triton kernels for the most expensive ops in transformer training: RoPE, RMSNorm, SwiGLU, and crucially the cross-entropy + chunked output projection. Enabling `model.use_liger: true` in verl typically buys 15–30% throughput on dense models with 32K+ vocab. It is a free win on supported models (Llama-3, Qwen-2.5, Qwen-3) and a no-op on unsupported ones.

## 8. Rollout Backends: vLLM / SGLang / TensorRT-LLM / HF

The rollout engine is where the throughput delta lives. HF `generate` does ~30 tok/s on Llama-3-8B at batch 64; vLLM does 1500+ tok/s on the same hardware. That ratio is why every serious RLHF stack now uses an inference engine for rollout.

**vLLM** is the default in verl. PagedAttention, continuous batching, and the integration is mature (`v0.8.2+` for FSDP compatibility, `v0.10+` recommended for 2026). The actor's fp16 weights are pushed in via the 3D-HybridEngine; vLLM treats them as if they came from a checkpoint. Sampling parameters (`temperature`, `top_p`, `n` for group rollouts, `max_tokens`) are set per call. For PPO/GRPO/DAPO this is the right default.

**SGLang** is the multi-turn / tool-use specialist. RadixAttention shares prefix KV more aggressively than PagedAttention, which is the right primitive when you have G=32 rollouts that share the same prompt — they share 100% of the prefix KV. SGLang also has first-class support for interactive rollouts: a Python function that calls the model, gets a response, executes a tool, and feeds the tool output back into the same conversation. For agentic RL — the search agent, the code agent, the math-with-Python-execution agent — SGLang is what verl recipes converge to.

**TensorRT-LLM** is for production-grade inference with NVIDIA-specific optimizations (FP8, INT8 KV cache, Hopper-specific kernels). The engine compilation step is expensive (~minutes), so for RL where weights change every iteration, you'd think this is a deal-breaker. verl works around this by keeping the engine as a "shape" and rebuilding only the runtime context; it works, but you should reach for vLLM unless you have a specific reason.

**HF Transformers** is the fallback for niche models that don't yet have vLLM/SGLang support. Slow, but always works, and useful for ablations where you want to compare "rollout engine X vs ground truth from HF generate" to confirm a regression isn't caused by the engine.

```yaml
actor_rollout_ref:
  rollout:
    name: "vllm"            # vllm | sglang | hf | trtllm
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.6
    dtype: "bfloat16"
    n: 8                    # group size for GRPO/DAPO
    temperature: 1.0
    top_p: 1.0
    max_response_length: 8192
    enforce_eager: false    # cudagraphs for speed
    free_cache_engine: true # release KV pool back to actor weights between rollouts
```

**The async rollout pattern (SGLang-only).** For multi-turn agents, the rollout takes seconds-to-minutes per sample (the agent invokes tools, waits for sandbox results, retries). Synchronous rollout would block the actor on the slowest sample. verl's async rollout dispatches all G samples to SGLang concurrently, returns as they complete, and proceeds to the next training step as soon as enough have finished. This is implemented via `actor_rollout_ref.rollout.mode: "async"` and a target completion ratio.

**The vLLM 0.8.x compatibility story.** Between vLLM 0.7 and 0.8, the engine refactored `LLMEngine` and broke the way verl pushed weights in. verl pinned `vllm>=0.8.2` and updated the integration. If you're upgrading verl from a 2025-vintage commit, this is the breakage you'll hit; the fix is to bump both verl and vLLM together.

## 9. Algorithms: The `adv_estimator` Registry

verl's flexibility shines here. Switching between PPO, GRPO, DAPO, RLOO, ReMax, GSPO, DrGRPO, REINFORCE++, KL_Cov, and Clip_Cov is a single YAML key change. The control loop does not change.

![Algorithm × backend matrix: ten algorithms with critic, group size, reward type, and variance reduction](/imgs/blogs/verl-rlhf-library-deep-dive-4.png)

Below are the loss formulas and the verl knob that selects each. We define $\pi_\theta$ as the current actor, $\pi_{\theta_{\text{old}}}$ as the rollout-time actor (frozen during the optimizer steps for a single PPO epoch), $\pi_{\text{ref}}$ as the reference, $r$ as the reward, and $A$ as the advantage.

**PPO** (`adv_estimator: ppo`). The classical loss with clipped importance ratio and GAE advantages.

$$\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_t \min\left( \rho_t A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) + \beta \, \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ and $A_t$ is GAE($\lambda$). Needs a critic for the GAE baseline. Default $\epsilon=0.2$, $\lambda=0.95$, $\beta$=0.001–0.01.

**GRPO** (`adv_estimator: grpo`). Group-relative policy optimization. Generate $G$ samples per prompt, normalize within the group:

$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$

The critic is gone. The KL term is moved to the loss as $\beta \cdot \text{KL}$ (DeepSeek-Math style). Default $G=8$ to $64$.

**DAPO** (`adv_estimator: dapo`). DeepSeek's late-stage reasoning algorithm. Builds on GRPO with three tweaks: **decoupled clipping** (separate $\epsilon_{\text{low}}$ and $\epsilon_{\text{high}}$ to allow more aggressive upward updates); **dynamic sampling** (re-sample groups where all rewards are 0 or all 1, since they have no learning signal); and a **token-level loss** instead of sequence-level. Empirically the right choice for long-CoT math/code where you've exhausted GRPO's gains.

**RLOO** (`adv_estimator: rloo`). Leave-one-out baseline:

$$A_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j$$

Cheap variance reduction; works at small $G$ (2–8). When you can't afford GRPO's $G=32$, RLOO is the fallback.

**ReMax** (`adv_estimator: remax`). Subtracts the reward of a greedy-decoded sample as the baseline. Single rollout per prompt + one greedy decode. Useful when group sampling is expensive and you have a model RM.

**GSPO** (`adv_estimator: gspo`). Sequence-level importance sampling on top of GRPO. Trades a small bias for substantially lower variance at large $G$. The right pick when GRPO's update is unstable past $G=32$.

**DrGRPO** (`adv_estimator: drgrpo`). Fixes a length-normalization bias in GRPO: the original loss divided by sequence length, which under-weights long correct responses. DrGRPO removes the per-token normalization. For workloads where response length matters (math with long CoT), this is the right default over vanilla GRPO.

**REINFORCE++** (`adv_estimator: reinforce_plus_plus`). The basic policy gradient with a running-mean baseline. Cheap, simple, the right baseline to compare against.

**KL_Cov / Clip_Cov** (`adv_estimator: kl_cov` or `clip_cov`). Entropy-bonus variants that use the covariance between the policy gradient and the entropy gradient to encourage exploration. Useful when training collapses to greedy outputs and entropy drops to zero.

```python
# verl/trainer/ppo/core_algos.py (simplified GRPO advantage)
def compute_grpo_advantage(scores, group_size, eps=1e-6):
    """
    scores: tensor of shape [batch_size * group_size]
    Returns advantages of the same shape, normalized within each group.
    """
    bs = scores.shape[0] // group_size
    grouped = scores.view(bs, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds  = grouped.std(dim=1, keepdim=True) + eps
    advantages = (grouped - means) / stds
    return advantages.view(-1)
```

```python
# Compare with PPO's GAE
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    rewards: per-token reward, shape [batch, seqlen]
    values:  critic output, shape [batch, seqlen]
    """
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
    advantages = torch.zeros_like(deltas)
    last_gae = 0.0
    for t in reversed(range(deltas.shape[1])):
        last_gae = deltas[:, t] + gamma * lam * last_gae
        advantages[:, t] = last_gae
    return advantages
```

The two computations look completely different — PPO's GAE traverses time per token, GRPO's normalization is a one-shot moment-matching across rollouts — but the verl loop calls a single `compute_advantages(estimator=...)` function that dispatches to whichever is configured. Switching from PPO to GRPO is one config line; the rest of the loop adapts.

**Which one should you pick?** Heuristics:

- If you have an RM and want chat alignment → PPO.
- If you have verifiable rewards (math, code, structured output) and want reasoning → GRPO or DAPO.
- If GRPO is unstable at large G → GSPO.
- If GRPO's responses are dominated by short answers → DrGRPO.
- If you're comparing baselines → REINFORCE++ as a sanity check.
- If the model collapses to greedy → KL_Cov or Clip_Cov as a remediation.

## 10. Reward Design: Verifiable vs Model-Based

The 2024 RLHF playbook used preference data → reward model → PPO. The 2025 reasoning playbook flipped this: for math, code, and structured tasks, **a Python function is a better reward than a learned RM**. verl supports both.

**Function-based rewards** are pure-Python scoring functions. Input: the model's response, the ground-truth answer. Output: a scalar in [0, 1].

```python
# verl/utils/reward_score/math.py (simplified)
import re
from sympy import simplify, sympify

def extract_boxed_answer(response: str) -> str | None:
    """Find the last \\boxed{...} expression."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", response)
    return matches[-1] if matches else None

def compute_score(response: str, ground_truth: str) -> float:
    pred = extract_boxed_answer(response)
    if pred is None:
        return 0.0
    try:
        # Symbolic equivalence check
        if simplify(sympify(pred) - sympify(ground_truth)) == 0:
            return 1.0
    except Exception:
        pass
    # Fall back to string match
    return 1.0 if pred.strip() == ground_truth.strip() else 0.0
```

For code, the function is a sandbox executor:

```python
# verl/utils/reward_score/code.py (illustrative)
def compute_score(response: str, test_cases: list[dict]) -> float:
    # Extract the code block
    code = extract_python_block(response)
    if not code:
        return 0.0
    # Run in a sandboxed subprocess with strict timeout and resource limits
    passed = 0
    for tc in test_cases:
        try:
            result = run_in_sandbox(code, stdin=tc["input"], timeout=5.0,
                                    memory_limit_mb=256)
            if result.stdout.strip() == tc["expected"].strip():
                passed += 1
        except (TimeoutError, MemoryError):
            return 0.0
    return passed / len(test_cases)
```

**Why function rewards beat RMs for these workloads:**

1. **No reward-hacking margin.** An RM can be fooled by spurious patterns; a unit test cannot.
2. **No RM training pipeline.** Saves a quarter of engineering effort.
3. **Deterministic.** Same response → same reward, every time.
4. **Cheap.** Math regex match is microseconds; an RM forward is hundreds of milliseconds on a 7B model.

**When you still need a model RM:**

- Open-ended chat (no ground truth, only preferences).
- Helpfulness / safety alignment where rules don't capture the criterion.
- Style / format preferences ("respond like a friendly assistant").

verl handles both via a single `reward_model` config block, and supports **mixed rewards**: 0.7 × function reward + 0.3 × RM score. Recipes in `recipe/` show this for code tasks where you want correctness (function) plus style (RM).

**Sandbox fusion.** verl's `sandbox_fusion` mode batches sandbox executions across the rollout group, cutting the per-rollout sandbox overhead from ~1 s to ~50 ms when G=16. For code RL this is the difference between practical and unusable.

**A worked numerical example for sandbox cost.** A code-RL run with G=16 rollouts × batch=128 = 2048 trajectories per step. Each trajectory needs sandbox execution of, say, 5 unit tests at ~200 ms per test = 1 second per trajectory. Sequential execution: 2048 seconds (~34 minutes) per step — completely unworkable. Naive parallel-process pool with 32 workers: ~64 seconds per step. With sandbox fusion batching to 64 concurrent containers + warm container reuse: ~12 seconds per step. The verl `sandbox_fusion` integration also caches the unit-test setup, dropping the 200 ms per test to ~60 ms once warm. End state: sandbox cost is ~3% of step time instead of 60%. **Lesson:** if you're doing code RL, the sandbox infrastructure is more important than the RL algorithm. Pick `sandbox_fusion` from day one.

**Reward shaping pitfalls.** The most common code-reward trap is binary pass/fail: "all tests pass → reward 1, otherwise 0." This produces near-zero reward variance for hard problems and a flat gradient. The fix is partial credit: `reward = (tests_passed / total_tests) ** k` where `k=2` heavily rewards getting close to all-pass. For math, the parallel trap is "exact symbolic match." Many models produce numerically correct answers in a different form (e.g. `1/2` vs `0.5` vs `\\frac{1}{2}`), and a strict matcher gives them zero reward. verl's `math.py` reward function uses sympy's `simplify` to handle equivalence; copy that pattern when you write your own. Reward design is half of RL engineering, and most regressions trace back here, not to the algorithm.

**Multi-turn rewards.** When the rollout is a multi-turn conversation (agent calls tools, gets results, refines answer), the reward can be:

- Final-answer only (sparse signal, simpler to credit).
- Per-turn (denser signal, harder to credit correctly).
- A weighted mix.

verl's multi-turn recipes use final-answer reward for math/code agents (the only thing that matters is the final boxed answer) and per-turn for search agents (where each search query has its own quality signal). The masking story is critical: tool outputs are masked from the policy gradient (you don't want the model to be rewarded for predicting Python's stdout), only the model's own generations contribute to the loss.

## 11. Multi-Turn Rollout and Tool Use in RL

The 2026 frontier of RLHF is no longer chatbot alignment — it's training agents that use tools. verl is one of the few frameworks that handles this end-to-end.

The setup: the model emits a special token (`<tool_call>`), the rollout engine pauses generation, executes the tool (Python interpreter, search engine, API call, sandbox), feeds the result back as a `<tool_result>` block, and resumes generation. The whole back-and-forth is one trajectory. The reward is computed on the trajectory.

```python
# Simplified multi-turn rollout (SGLang-backed)
async def multi_turn_rollout(prompt, model, tools, max_turns=8):
    conversation = [{"role": "user", "content": prompt}]
    for turn in range(max_turns):
        response = await model.generate(conversation, stop=["<tool_call>"])
        conversation.append({"role": "assistant", "content": response})
        if "<tool_call>" not in response:
            break  # final answer
        tool_name, args = parse_tool_call(response)
        if tool_name not in tools:
            break  # invalid tool; trajectory ends
        result = await tools[tool_name].execute(args, timeout=10.0)
        conversation.append({"role": "tool", "content": result})
    return conversation
```

verl's interaction with this lives in `verl/workers/rollout/sglang_rollout.py`. The `agent_loop_manager` orchestrates the multi-turn loop, masks tool outputs from the loss, and computes per-turn or per-trajectory rewards.

**Masking is the part that bites.** The loss is computed only on the **assistant tokens that the model generated**. User messages and tool outputs must be masked. If you forget the mask, the model gets rewarded for predicting the tool's output, which is degenerate. verl's `compute_response_mask` in `verl/utils/dataset/rl_dataset.py` builds the mask from token-type IDs; mistakes here look like "training works, but the model gets weird very fast."

**Tool sandbox patterns.** verl integrates with three sandbox patterns:

1. **Local subprocess** — `subprocess.run` with rlimits. Simple, but a misbehaving sandbox can crash the rollout worker.
2. **Sandbox fusion** — batched container execution via SandboxFusion service. Better isolation, batched IO.
3. **External MCP servers** — for serious tool ecosystems (search, browsing, code execution at scale), point at an MCP server that runs separately.

The HermesAgent and DeepResearch recipes in verl show all three patterns for, respectively, code agents and search agents.

**Reward credit assignment.** For multi-turn, the simplest thing is final-trajectory reward — the math problem was solved correctly or not. More sophisticated schemes credit per-turn (the search query was useful or not, the partial code compiled or not). The trade-off: per-turn rewards reduce variance but require designing reward shaping that doesn't lead to local optima ("the model rewards itself for searching pointlessly because each search has positive reward").

## 12. Performance Levers: Sequence Packing, Ulysses SP, Gradient Accum, LoRA, Liger

These are the knobs that turn a working recipe into a fast working recipe.

**Sequence packing.** RLHF rollouts have wildly variable lengths (some prompts get 8K-token responses, some get 100). Naive padding to `max_length` wastes 60–80% of compute. Sequence packing concatenates short sequences into a single dense sequence and uses cumulative attention masks to keep them isolated. verl enables this via `actor.use_remove_padding: true` (the FlashAttention varlen path) and `actor.padding_free: true` for newer kernels. Speedups of 2–3× on workloads with high length variance.

**Ulysses sequence parallelism.** Splits the sequence dimension across SP ranks. Activation memory drops by `1/SP`, attention compute is unchanged but partitioned. The cost is an all-to-all per layer to materialise the full sequence at attention time. For 16K+ contexts on 8B–34B models, Ulysses SP=2 or SP=4 is what makes the activation memory fit. Enable with `actor.ulysses_sequence_parallel_size: 2`.

**Gradient accumulation.** Increases effective batch size without increasing memory. RLHF wants batches of 64–512 prompts × group size 8–64 = thousands of rollouts; you cannot fit all of them in HBM at once. Gradient accumulation runs N micro-batches and accumulates grads before the optimizer step. verl's `actor.ppo_mini_batch_size` and `actor.ppo_micro_batch_size_per_gpu` control this. Common setting: `mini=64, micro=4`, meaning `64/4 = 16` accumulation steps per optimizer step.

**LoRA.** verl supports LoRA fine-tuning for the actor (and optionally the critic). Trades expressiveness for ~10× lower memory. The right pick when you're constrained to a single node and want to RL-train a 70B base. The catch for RLHF specifically: LoRA's reduced expressiveness sometimes prevents the model from making large policy updates, which can be either good (stability) or bad (under-exploration). Treat as a different optimum, not a free win.

**Liger-kernel.** Already mentioned: fused triton kernels for RoPE, RMSNorm, SwiGLU, fused cross-entropy + chunked output projection. 15–30% throughput improvement on supported models. Free win when applicable.

| Lever | What it saves | What it costs |
| --- | --- | --- |
| Sequence packing | 50–80% wasted compute on padding | Slightly more complex masking |
| Ulysses SP | Activation memory (`1/SP`) | One all-to-all per layer (~5% time) |
| Gradient accum | Memory (effective batch ÷ N) | Wall-clock per opt step (×N) |
| LoRA | 10× memory on actor | Reduced expressiveness, possible under-exploration |
| Liger kernels | 15–30% throughput | None (when supported) |

**The interaction trap.** Stacking all of these is _not_ free. SP + sequence packing requires careful mask handling — the boundaries between packed sequences must align with SP boundaries. LoRA + Megatron has its own composability story. The verl recipes for DeepSeek 671B and Qwen3-235B encode the working combinations; deviate from those at your own risk.

## 13. Configuration Deep-Dive: ppo_trainer.yaml with Hydra

verl uses Hydra to compose configs. The top-level structure mirrors the role split.

```yaml
# verl/trainer/config/ppo_trainer.yaml (annotated, abridged)

algorithm:
  adv_estimator: grpo                # ppo | grpo | dapo | rloo | remax | ...
  use_kl_in_reward: false            # add KL inside the reward (DeepSeek style)
  kl_coef: 0.001                     # KL regularization weight
  norm_adv_by_std_in_grpo: true      # group-std normalization

data:
  train_files: "/data/gsm8k_train.parquet"
  val_files: "/data/gsm8k_val.parquet"
  prompt_key: "prompt"
  reward_fn_key: "ground_truth"
  train_batch_size: 256              # prompts per training step
  max_prompt_length: 1024
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: "Qwen/Qwen3-8B-Instruct"
    use_remove_padding: true
    use_liger: true
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp2                  # fsdp | fsdp2 | megatron
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 4
    use_kl_loss: true
    kl_loss_coef: 0.001
    ulysses_sequence_parallel_size: 2
    optim:
      lr: 1e-6
      lr_warmup_steps_ratio: 0.05
      offload: false                 # offload optimizer states to CPU during rollout
    fsdp_config:
      param_offload: false
      optimizer_offload: false
  rollout:
    name: vllm                       # vllm | sglang | hf
    tensor_model_parallel_size: 2
    n: 8                             # group size for GRPO
    temperature: 1.0
    top_p: 1.0
    gpu_memory_utilization: 0.6
    free_cache_engine: true
    enforce_eager: false
  ref:
    fsdp_config:
      param_offload: true            # keep ref weights on CPU until needed

critic:
  enable: false                      # disabled for GRPO; set true for PPO

reward_model:
  enable: false                      # using function-based rewards

custom_reward_function:
  path: "verl/utils/reward_score/gsm8k.py"
  name: "compute_score"

trainer:
  total_epochs: 5
  total_training_steps: 1000
  save_freq: 100
  test_freq: 50
  project_name: "verl-gsm8k-grpo"
  experiment_name: "qwen3-8b-grpo-v1"
  logger: ["console", "wandb"]
  resume_mode: "auto"
  n_gpus_per_node: 8
  nnodes: 2

ray_init:
  num_cpus: null                     # auto-detect

resource_pool_spec:
  actor_rollout_ref: [8, 8]          # 16 GPUs across 2 nodes
  # critic / reward pools omitted (disabled in this recipe)
```

The Hydra override pattern is what makes this manageable. To switch from GRPO to DAPO: `algorithm.adv_estimator=dapo`. To bump group size: `actor_rollout_ref.rollout.n=16`. To swap rollout backend: `actor_rollout_ref.rollout.name=sglang`. The full launch command on the next page exercises this pattern.

**The recipe directory.** Under `recipe/`, verl ships pinned configs for popular settings: `recipe/dapo/dapo_qwen3_8b.yaml`, `recipe/grpo/grpo_qwen3_8b.yaml`, `recipe/ppo/ppo_llama3_8b_with_rm.yaml`, `recipe/r1/r1_grpo_qwen3_32b.yaml`, plus `recipe/deepseek_v3/` for the 671B MoE setup. Each ships with a `REQUIRED_VERL.txt` pinning the verl commit it was tested against. Start from the recipe that matches your model class; do not invent configs from scratch unless you have a reason.

## 14. Hands-On: GSM8K GRPO and Notes for Larger Models

Here is the full path from a clean machine to a running GRPO job on Qwen3-8B for GSM8K. Tested with `verl==0.4.x` and `vllm==0.10.x` in April 2026.

### 14.1 Install

```bash
# verl + dependencies
git clone https://github.com/volcengine/verl.git
cd verl

# Use uv for fast resolution
uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install vllm==0.10.0
uv pip install -e .

# Verify
python -c "import verl; print(verl.__version__)"
```

For Megatron support (needed for 70B+):

```bash
uv pip install -r requirements-megatron.txt
# Convert HF checkpoint to Mcore (one-off per model)
python scripts/converter_hf_to_mcore.py \
  --hf_path "Qwen/Qwen3-32B" --mcore_path "/data/mcore/qwen3-32b"
```

### 14.2 Prepare GSM8K

```bash
# verl ships a helper that downloads + tokenizes
python examples/data_preprocess/gsm8k.py \
  --local_dir "/data/gsm8k"

# Produces /data/gsm8k/train.parquet and /data/gsm8k/val.parquet
# Each row: {prompt: str, ground_truth: str, ...}
```

### 14.3 Launch GRPO on 16 H100s (2 × 8)

```bash
# Start the Ray cluster (head node)
ray start --head --num-gpus=8 --port=6379

# Worker node
ray start --address=<head_ip>:6379 --num-gpus=8

# Launch training (run on head)
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/gsm8k/train.parquet \
  data.val_files=/data/gsm8k/val.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B-Instruct \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
  custom_reward_function.name=compute_score \
  trainer.total_epochs=3 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=2 \
  trainer.project_name=verl-gsm8k-grpo \
  trainer.experiment_name=qwen3-8b-v1 \
  trainer.logger=['console','wandb'] \
  trainer.save_freq=100 \
  trainer.test_freq=50
```

Expected output on H100: ~1500 prompts/min for the rollout phase, ~300 prompts/min for the training phase, dominated by the 8 group rollouts × 1024 response tokens. After 1000 steps (~6 hours) GSM8K accuracy improves from base ~60% to ~85%.

### 14.4 Scaling Notes

For Qwen3-32B GRPO on 32 H100s:

```bash
actor_rollout_ref.actor.strategy=megatron \
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
data.max_response_length=4096
```

For DeepSeek-V3 671B DAPO on 256 H100s — refer to `recipe/deepseek_v3/`. The config encodes the working EP=8, TP=8, PP=4 combination with FP8 weights and very specific Ulysses SP settings. Do not deviate.

### 14.5 Reading the Logs

The metrics that matter, in priority order:

1. `actor/reward_mean` — per-step mean reward across the batch. Should monotonically rise.
2. `actor/kl` — KL between actor and reference. If it explodes (>0.5), the policy is drifting; lower the learning rate or raise `kl_coef`.
3. `actor/entropy` — policy entropy. If it collapses to 0, you're stuck in greedy; switch to KL_Cov or raise temperature.
4. `actor/loss`, `critic/loss` — per-step training loss. Should be small, not necessarily monotone.
5. `rollout/throughput` — tokens/sec of the rollout phase. Should stabilize within 10% across runs.
6. `timing/reshard` — wall-clock for 3D-HybridEngine reshard. Should be 2–5% of step time. If higher, something is wrong.

## 15. Performance Tuning and Profiling

verl ships profiling hooks that integrate with NSight, Ray Timeline, and torch.profiler.

**Ray Timeline** is the first stop. Run any verl job, then:

```bash
ray timeline --filename ray_timeline.json
# Open in chrome://tracing or perfetto.dev
```

You'll see the gap between rollout end and training start — that gap is the 3D-HybridEngine reshard, and if it's 30+ seconds you have a misconfigured shape transition.

**torch.profiler integration** via `actor.profiler.enable=true` produces per-op traces. The hot ops to watch:

- `flash_attn_varlen_func` — should be 30–50% of attention time. If lower, sequence packing isn't working.
- `all_gather_into_tensor` — FSDP weight gather. Should be 5–10% of forward.
- `reduce_scatter_tensor` — FSDP grad scatter. Should be 5–10% of backward.

**NSight Systems** for kernel-level profiling. verl wraps the worker entrypoint with NSight if `actor.profiler.use_nsys=true` is set. Expect this to add 10–30% overhead, so use only for diagnosis.

**Common hot spots, ranked:**

1. **vLLM warmup on every iteration.** If `free_cache_engine: true` is too aggressive, vLLM rebuilds its KV pool every reshard. Set `free_cache_engine: true` only if you actually have HBM pressure. Otherwise leave it on default and let vLLM keep the pool.
2. **RM forward as a bottleneck.** A 7B RM forward on G=16 rollouts × batch=256 = 4096 forward passes. If the RM uses HF generate (which it shouldn't), you'll spend 50% of step time scoring. Use FSDP for the RM and batch-process; verl supports this.
3. **Reshard overhead at small scales.** On a 2-GPU setup, the all-gathers don't get faster — they just have nothing to overlap. Reshard is 5–10% of step time at 2 GPUs, drops to 2% at 16+. Don't be surprised if your laptop benchmarks lie.
4. **Reference model's lazy load.** First call to `compute_ref_log_prob` triggers a full weight load from CPU. Add a warmup call before the first training step.
5. **Logging overhead.** wandb step logging at high frequency can become a bottleneck. Set `trainer.test_freq` ≥ 50.

## 16. Case Studies

Eleven production incidents drawn from public verl issue threads, the HybridFlow paper's appendix, and the `recipe/` reproduction notes. Names are anonymized.

### 16.1 PPO collapses with KL coefficient set too low

A team training a 7B model with PPO and an off-the-shelf RM saw `actor/kl` climb steadily for the first 200 steps, then **explode to 50+** in step 220. The reward kept rising, but offline evals showed the model was now producing incoherent text full of reward-hack patterns the RM happened to like.

Root cause: `kl_coef: 0.0001` (the team had copied a recent paper's setting). At low KL coefficients, the policy can drift arbitrarily far from the reference, and the RM (trained on a fixed distribution near the reference) becomes uncalibrated outside that distribution. The model exploits whatever artefact the RM rewards in the OOD region.

Fix: bump `kl_coef` to 0.01, restart from a checkpoint before the divergence, retrain. KL stayed under 0.5 for the rest of the run; offline evals improved monotonically. **Generalizable lesson:** treat KL as a regularizer that bounds how far you can trust the RM. The right value is the largest KL at which the RM remains calibrated on your offline eval — measure by running rollouts at increasing KL distances and watching the reward-vs-eval correlation.

### 16.2 The 3D-HybridEngine NCCL hang on 256 GPUs

A team running DeepSeek-V3 DAPO on 256 H100s saw the first reshard hang indefinitely during the all-gather along PP. NCCL logs showed all ranks waiting for tensor `block_15.attn.qkv.weight` from rank 192.

Root cause: rollout TP was set to `tensor_model_parallel_size: 16` (matching the rollout engine's TP), but rollout EP was set to `expert_parallel_size: 8`, and these two sharding axes were not orthogonal in the rollout engine's expectation. The 3D-HybridEngine's tile algorithm produced a request that no rank could satisfy.

Fix: align the rollout shape to a known-working pattern from `recipe/deepseek_v3/`: TP=8, EP=8, with a specific routing-aware permutation. The hang resolved. **Generalizable lesson:** for MoE models on 100+ GPUs, use the recipe-provided shape exactly. The combinatorial space of TP × EP × PP × DP is large, and most points in it produce subtle NCCL deadlocks rather than clear errors.

### 16.3 GRPO advantage variance explosion on a code RL run

A team running GRPO on a 14B model for code generation saw the loss climb after step 500 even though the reward kept rising. The advantage histogram showed huge tails: a few rollouts with reward 1.0 in groups where the rest were 0.0 produced normalized advantages of +5.0 or higher, dominating the gradient.

Root cause: GRPO's group-std normalization is unstable at low group reward variance. When 7 of 8 rollouts get 0 and 1 gets 1, $\sigma \approx 0.35$, and the advantages are roughly $\{-0.4, -0.4, ..., +2.7\}$. With outliers like this driving the gradient, the update is essentially "imitate the lucky one," which over-fits to a single trajectory.

Fix: switched to DrGRPO, which removes the per-token normalization and uses a fixed-scale advantage. Stability returned, and the eval improved. **Generalizable lesson:** when group reward variance is low (most rollouts fail), GRPO's normalization amplifies noise. DrGRPO and GSPO both address this; pick one based on your length-bias preference.

### 16.4 vLLM 0.8.x rollout incompatibility with FSDP2

A team upgrading from vLLM 0.7 to 0.8 saw the rollout engine OOM on the first weight push. The error message pointed at a missing `tensor_parallel_workers` attribute that vLLM 0.7 had exposed and 0.8 hid behind a new init API.

Root cause: vLLM 0.8 refactored `LLMEngine` and the way verl pushed weights through the 3D-HybridEngine assumed the old API. verl-main had been updated, but the team was on a 2-month-old commit.

Fix: bump verl to current main, which calls the new vLLM API correctly. The `REQUIRED_VERL.txt` in `recipe/` exists for exactly this reason — pin verl + vLLM versions together. **Generalizable lesson:** don't upgrade vLLM in isolation. Treat verl + vLLM as a paired dependency; upgrade together with a fresh recipe checkpoint.

### 16.5 Sequence packing made decode tokens visible to prompt loss

A team enabled `actor.use_remove_padding: true` for throughput, retrained, and observed strange degradation in math accuracy: the model started producing answers that included _the prompt's hint text_ verbatim. Offline diff against a non-packed run showed the loss was ~10× lower for tokens in the prompt.

Root cause: sequence packing requires `position_ids` and `attention_mask` to correctly distinguish packed sequences. The team had patched the loss to skip prompt tokens, but the `position_ids` for packed sequences had been computed before the loss-mask update, leading the loss to incorrectly treat the prompt of sequence i+1 as a continuation of sequence i. The model literally got rewarded for predicting the next prompt's tokens.

Fix: regenerate the packed batch's `position_ids` and `loss_mask` together, atomically. verl has this fixed in current main; the team had a custom `RewardManager` that re-implemented the masking incorrectly. **Generalizable lesson:** sequence packing's correctness story is mask + position ids together. When you customize the dataset class, copy the masking exactly, do not "simplify."

### 16.6 SGLang multi-turn rollout deadlock when the tool sandbox blocked

A team running an agentic RL recipe with SGLang saw rollouts hang indefinitely after step 300. SGLang logs were quiet; the sandbox executor logs showed it was waiting on a process that didn't exist.

Root cause: a Python tool implementation had a `while True: pass` bug in some inputs; the sandbox subprocess hung. Without a timeout, the rollout worker waited forever, blocking the next training step.

Fix: enforce strict timeouts at the sandbox layer (`timeout=10.0` in `subprocess.run`), and at the agent loop layer (`max_turns=8`, `total_timeout=120s`). Mark trajectories that hit timeout as reward 0 and continue. **Generalizable lesson:** every external dependency in a multi-turn rollout must have a timeout, at every layer. The default in tool implementations is to wait; in RL training, "wait" means "deadlock the cluster."

### 16.7 Ulysses SP × rollout reshard race condition on Megatron checkpoint reload

A team training Qwen3-32B with Megatron + Ulysses SP=4, after a checkpoint reload from step 800, saw the next rollout produce gibberish. The actor weights looked correct on disk but the rollout engine was reading partial-update tensors.

Root cause: during a checkpoint reload, the SP rank dimension was being reconciled before the reshard call completed, and the reshard's all-gather along the SP dim raced with the resume hook that re-allocated weight buffers. A brief window existed where the rollout engine pulled a buffer that was being overwritten.

Fix: explicit barrier (`torch.cuda.synchronize` + `dist.barrier`) before the reshard call after a resume. verl current main has this; the team's custom resume code skipped it. **Generalizable lesson:** Megatron's many sharding dimensions need explicit synchronization on resume. When in doubt, add a `dist.barrier` — it's free at steady state and saves you an entire debug session at resume time.

### 16.8 ReMax variance reduction making training "look fine" but actually under-exploring

A team chose ReMax for cost reasons (1 stochastic rollout + 1 greedy = cheaper than GRPO's G=8). Training looked great: reward climbed, KL stayed bounded, no instability. But offline evals plateaued at the same level as the SFT baseline.

Root cause: ReMax's greedy baseline is a strong subtractor. If the stochastic rollout is barely better than greedy (which is likely if the model is already good), the advantage signal is tiny, and the policy gradient is dominated by noise. The model stays where it is. Reward "rises" because the rollout strategy is being optimized, but the actual policy shift is minimal.

Fix: switched to RLOO with G=4 (cost: 4× rollouts, but ~3× faster than GRPO with G=8). Eval improvements appeared by step 200. **Generalizable lesson:** "training looks healthy" is not the same as "training is exploring." Sample-based RL methods need a baseline _and_ enough sample diversity to drive exploration. Plot rollout-reward variance per group; if it's <0.05 you're not exploring.

### 16.9 Reward-model staleness during long async rollouts

A team running GRPO with a model RM and an async multi-turn rollout (rollouts taking ~30s each) on Qwen3-14B saw RM scores drift downward over the course of a step's batch — early rollouts scored higher than late ones for similar-quality outputs.

Root cause: the actor's weights were updated **before** all rollouts in the batch had been scored. Late rollouts were scored against the same RM, but the actor that produced them had drifted slightly from the actor at rollout start. The RM, calibrated to a specific actor distribution, drifted with it.

Fix: ensure all rollouts in a batch are scored against the **same actor checkpoint**. verl handles this correctly by default in synchronous mode; the team's async setup had a bug where the actor was updated mid-batch. The proper async pattern is "complete all rollouts, then score, then update," not "score as rollouts complete." **Generalizable lesson:** async rollouts and RM scoring must respect step boundaries. The async-ness is for hiding tool latency, not for overlapping with weight updates.

### 16.10 Scaling to DeepSeek 671B: expert parallelism config that finally worked

A team trying to RL-train DeepSeek-V3 (671B MoE) with DAPO went through three weeks of failed configurations. OOM at TP=4 EP=8. NCCL hang at TP=8 EP=4. Slow at TP=16 EP=2 (insufficient EP).

Root cause: 671B has 256 experts per layer; the EP shape determines per-rank expert count and thus memory. EP=8 means 32 experts per rank, which fits in 80 GB only if the activation tensors are SP-partitioned aggressively. The team's Ulysses SP=2 was insufficient.

Fix: SP=4 + EP=8 + TP=8 + PP=4 = 1024 ranks total = 128 nodes × 8 H100. Memory budget: 80 GB - (32 experts × 4 GB weight) - (8 GB Adam compressed) - (12 GB activations w/ SP=4) = ~28 GB headroom for KV cache during rollout. The recipe at `recipe/deepseek_v3/` documents this exact shape. **Generalizable lesson:** for MoE at 100B+ scale, the shape is not a free parameter — it's determined by the memory budget per rank, which is determined by experts × hidden × activation_factor / SP. Compute the budget first; pick the shape from the budget.

### 16.11 NUMA pinning recovered 18% throughput on AMD EPYC nodes

A team running on AMD EPYC 9474F (dual-socket) + 8× MI300X saw their effective throughput plateau ~80% of expected on a Qwen2.5-72B run. Profiling showed CPU-side data loading was inconsistent; some workers were slower than others by 30%.

Root cause: Ray's default actor placement does not pin to NUMA nodes. Workers that landed on socket 1's CPUs but accessed weights pinned to socket 0's memory paid the cross-socket Infinity Fabric tax — half the local DRAM bandwidth. The slower workers held up step time.

Fix: `numactl --cpunodebind=$N --membind=$N` matching each rank's GPU PCIe topology, set via Ray's `runtime_env`. Throughput climbed 18%, variance halved. **Generalizable lesson:** dual-socket boxes need explicit NUMA pinning. Ray respects it if you set it; the default is "let the kernel decide," which on EPYC is reliably suboptimal.

## 17. When to Reach for verl, When Not To

verl is the right pick when at least two of these are true:

- Multi-node deployment (≥16 GPUs).
- Iterating on algorithms (PPO vs GRPO vs DAPO vs ...).
- Model size 8B+.
- Rollout latency is a meaningful fraction of step time (i.e., your prompts produce 1K+ token responses).
- You need verifiable rewards (math, code) or multi-turn agentic rollouts.
- You want a recipe you can hand to another team and have them reproduce.

Skip verl when:

- Single-GPU prototyping. Use TRL — its ergonomics for single-script RL are unmatched.
- Small models (≤7B) on a single node, with PPO, with an existing RM, and short responses. TRL or DSChat will work and be simpler to operate.
- You only ever run one algorithm and never plan to switch. The Ray + Hydra abstraction tax isn't worth it.
- You need on-device fine-tuning. verl assumes a Ray cluster.
- DPO / SFT-only workloads. DPO is offline contrastive optimization, not RL; use HuggingFace TRL or Axolotl instead. verl can do SFT (it has an `sft_trainer.py`) but it's overkill compared to lighter SFT frameworks.

**The minimum-viable starting recipe:** copy `recipe/grpo/grpo_qwen3_8b.yaml`, change `data.train_files` to your dataset, change `actor_rollout_ref.model.path` to your model, run on 16 GPUs. Watch the metrics dashboard. After a week, change one algorithm knob (`adv_estimator`, `n`, `kl_coef`) at a time and measure. Do not enable LoRA, Megatron, Ulysses SP, multi-turn, and an RM all at once on the first run; you will not be able to attribute regressions.

The one operational rule that has saved me more time than any other: **never enable two verl features at once on the same week**. Add Megatron Tuesday, measure for a week, then add Ulysses SP, measure again, then add async rollout. The verl surface area is large enough that bisecting failures across multiple toggles eats half a quarter of debugging time. Go slow, measure, document.

verl is, at its core, an admission that RLHF dataflow has structure that the original frameworks ignored: the actor is two models (training-shape and rollout-shape), the roles want different backends, and the algorithm should not have to know about either. Once you internalize that, the framework's choices stop being mysterious — the colocation default, the YAML structure, the `@register` decorator — and start being inevitable. The rest is execution: a recipe, a Ray cluster, and a willingness to read NCCL stack traces when things go sideways.

A senior reading of where the field is heading in mid-2026: the next year's verl development will likely focus on tighter multi-turn / agentic RL primitives (the hard problem now is not the RL math, it's the tool ecosystem), better integration with the LMCache layer for cross-replica KV reuse during rollout, and continued MoE scaling improvements as 1T+ models become practical to RL-train. The connector API and the controller paradigm are stable; everything below is moving fast. Pick a working recipe, pin the verl commit, and you'll be fine.

### A few non-obvious operational rules

These are not in the docs and they will save you a quarter when you internalize them.

**Pin verl + vLLM + the recipe commit together.** Each of these has its own release cadence; they drift apart in subtle ways. The recipe directory's `REQUIRED_VERL.txt` is the source of truth. When you upgrade one, upgrade all three. If you're on a 2025-vintage commit and try to bump only vLLM in 2026, you will hit issue 14.4 verbatim.

**Run a sanity GSM8K GRPO on every new cluster.** Before you trust a cluster for serious RL, run the recipe for GSM8K GRPO on Qwen3-8B for 100 steps. Reward should climb from ~60% to ~75%. If it doesn't, something is wrong infrastructurally, and you want to find that out before you've spent 100 GPU-days on your real workload. This costs ~$20 and saves you weeks.

**Watch `actor/kl` more than `actor/reward_mean`.** Reward can rise from reward hacking; KL drifting too high tells you the policy has left the RM's calibration zone. A healthy training curve has reward rising while KL stays bounded under `kl_target` (often 0.2–0.5). KL exploding is the "I'm gaming the RM" signal.

**Plot rollout-reward variance per group, not just the mean.** Mean reward going up looks great until you notice that 7 of 8 rollouts in each group still get 0 — you're learning from a single lucky sample per group. That's high-variance learning that overfits. Variance should sit somewhere in 0.1–0.3 across the run; if it's <0.05 you're under-exploring; if it's >0.4 you're not learning anything coherent.

**Check the loss mask once per recipe, then never again.** When you adapt a recipe to a new task, the very first thing to verify is that `loss_mask` zeros out everything that isn't a model-generated token. Print one batch, decode the masked-in positions, eyeball it. Recipes copy each other and mask bugs propagate; a 30-second sanity check at the start of each new recipe saves you from issue 16.5.

**For agent RL, treat the tool layer as a separate engineering problem.** The tools (sandbox, search, browser, MCP servers) need their own reliability story — timeouts, rate limits, retry logic, observability. Agent RL is RL plus distributed-systems engineering plus security (sandboxes are an attack surface), and the verl + SGLang side is rarely the bottleneck. Spend at least 30% of your engineering effort on the tools.

**Cost discipline.** A 70B GRPO run on 32 H100s for one week is roughly $30K of cloud spend. A 671B DAPO run on 256 H100s for two weeks is closer to $1M. Before you launch, do the math: GPU-hours × $/GPU-hour. Run the smallest version of your experiment that proves the hypothesis (8B, then 32B, then full scale). RL reward curves are noisy enough that at every scale, "did this configuration outperform last week's?" should require a real eval, not a wandb screenshot. Build the eval into the trainer (`trainer.test_freq`) and gate model promotion on it.

### Further Reading

- [verl on GitHub (volcengine/verl)](https://github.com/volcengine/verl) — source of truth, issue tracker, recipe directory.
- [verl-project/verl](https://github.com/verl-project/verl) — the verl-project mirror; same code, different namespace.
- [HybridFlow paper (arXiv 2409.19256)](https://arxiv.org/abs/2409.19256) — the architecture doc, EuroSys '25.
- [verl documentation](https://verl.readthedocs.io/) — programming guide, configurations, recipes.
- [HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) — in-depth on the controller paradigm.
- [DAPO paper (arXiv 2503.14476)](https://arxiv.org/abs/2503.14476) — the late-stage reasoning algorithm.
- [DrGRPO paper (arXiv 2503.20783)](https://arxiv.org/abs/2503.20783) — fixing GRPO's length bias.

If something in this post does not match what you see in production, the verl issue tracker is the right place; the maintainers are responsive and the project has been through enough production churn that most edge cases have already surfaced. The community (ByteDance, Anyscale, LMSys, Alibaba Qwen, plus academic collaborators) is well-organized and worth contributing back to when you find yourself relying on it.

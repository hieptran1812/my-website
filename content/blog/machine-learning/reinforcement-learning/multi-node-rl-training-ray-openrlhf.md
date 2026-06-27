---
title: "Multi-Node RLHF with Ray and OpenRLHF: Scaling to 70B Parameters"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A practitioner's guide to running RLHF across many GPUs and nodes — the Ray actor model, OpenRLHF's four-model layout, veRL's HybridEngine, SLURM and NCCL setup, fault tolerance, profiling, and the dollar cost of a 70B run."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "ray",
    "openrlhf",
    "llm-alignment",
    "distributed-training",
    "machine-learning",
    "pytorch",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 68
image: "/imgs/blogs/multi-node-rl-training-ray-openrlhf-1.png"
---

The first time I tried to run RLHF on a 70-billion-parameter model, the job died before the first gradient step. Not with a clever error — just a flat, anticlimactic `torch.cuda.OutOfMemoryError` on GPU 0, after forty minutes of loading shards. I had eight A100s, 80GB each, and I had naively assumed that if a 70B model fits for inference on a couple of cards, surely eight would be plenty for training. I had forgotten that RLHF is not one model. It is four. A policy you are training, a critic that estimates value, a reward model that scores outputs, and a frozen reference policy that keeps the trainee honest. Four 70B models, plus optimizer states, plus activations, plus a KV cache for generation. Eight GPUs was not a little short. It was off by a factor of eight.

That failure is the whole reason this post exists. Single-node RLHF works beautifully up to about 7B parameters, and then it falls off a cliff. The cliff is not gradual. The moment your four models no longer fit on one machine's GPUs, you are forced into a completely different architecture — one where the policy lives on one set of nodes, the reward model on another, the reference policy on a third, and a distributed scheduler stitches them together over the network. The interesting part is that this forced restructuring is not just a workaround for memory. Done right, it is *faster*, because generation, scoring, and reference-log-prob computation can all run in parallel on different hardware instead of fighting for the same GPUs in sequence.

This is the post I wish I'd had that day. By the end of it you will understand why RLHF at scale needs many nodes, how Ray's actor model gives you the orchestration primitives to coordinate them, how OpenRLHF lays out its four model groups and overlaps their work, how veRL's HybridEngine takes the opposite bet by time-sharing one GPU set, how NeMo-RLHF reaches into 175B territory with 3D parallelism, and — the part nobody writes down — how to actually launch this on a SLURM cluster with the right NCCL environment variables so InfiniBand doesn't silently fall back to TCP and halve your throughput. We will profile a real run, find the bottleneck, and put a dollar figure on a full 70B RLHF job.

![A dataflow graph showing a prompt batch fanning out to three parallel model groups before merging in the trainer](/imgs/blogs/multi-node-rl-training-ray-openrlhf-1.png)

Throughout, keep the series spine in mind: RL is an agent interacting with an environment, collecting rewards, and updating a policy. In RLHF the "environment" is a prompt distribution plus a learned reward model, the "agent" is a language model generating token sequences, and the "policy update" is PPO (or a variant). Everything in this post is plumbing around that loop — but the plumbing is what decides whether you train a 70B aligned model in three days or never finish at all. If you want the conceptual map of where RLHF sits among RL methods, the unified-map post is the place to start; here we are deep in the systems weeds. For the algorithmic core of PPO itself, this post assumes you already know the clipped surrogate objective and the role of the KL penalty, and focuses entirely on making it run across a cluster.

## 1. Why 70B forces you off a single node

Let us do the memory arithmetic honestly, because it is the arithmetic that dictates the architecture. Take a 70B parameter model. In bfloat16, the weights alone are $70 \times 10^9 \times 2 \text{ bytes} = 140\,\text{GB}$. That already does not fit on a single 80GB A100. Now train it: with the Adam optimizer you carry, per parameter, the weight, a gradient, a first moment, and a second moment. In mixed precision the common recipe keeps an fp32 master copy of weights plus fp32 momentum and variance, which works out to roughly 16 bytes per parameter for the optimizer-related state on top of the bf16 weights. For 70B that is on the order of $70 \times 10^9 \times 16 = 1{,}120\,\text{GB}$ — over a terabyte — just for the policy's training state. Add activations for the backward pass, which for long sequences and large batches can be tens to hundreds of gigabytes more.

ZeRO-3 (the third stage of the Zero Redundancy Optimizer, which shards parameters, gradients, and optimizer states across data-parallel ranks) is what makes this tractable. It divides that terabyte-plus of state evenly across all the GPUs in the data-parallel group. If you have 16 A100-80GB cards, each holds roughly $1{,}120 / 16 = 70\,\text{GB}$ of optimizer state plus its shard of weights and activations — tight but feasible. So the *policy alone*, at 70B, needs on the order of 16 GPUs across two nodes just to train. That is the first model.

Now remember RLHF needs four. The reward model is typically the same scale or smaller, but at 70B-class alignment it is often another 70B model. The reference policy is a frozen copy of the initial policy — another 140GB of weights (no optimizer state, since it is never updated, but it still occupies memory and compute). The critic, in classic PPO-RLHF, is yet another model of comparable size with its own optimizer state. Stack those up and you are easily past 64 GPUs — eight nodes of eight A100s — before you have run a single rollout.

#### Worked example: counting GPUs for a 70B RLHF run

Let me make this concrete. Suppose every model is 70B and we use 80GB A100s. The policy needs full training state, so apply the ZeRO-3 arithmetic: ~1,120GB of state, divided so each GPU holds under ~65GB to leave room for activations and the bf16 working copy, gives $1{,}120 / 65 \approx 18$, round up to 16 GPUs (two nodes) with activation checkpointing trimming the activation footprint. The critic is the same: another 16 GPUs. The reward model runs inference only, so it needs roughly the 140GB of weights sharded — $140 / 65 \approx 3$, call it 4 GPUs with headroom for batched scoring. The reference policy is also inference-only: another 4 GPUs. That totals $16 + 16 + 4 + 4 = 40$ GPUs at the floor. In practice you give the rollout actor extra GPUs because generation is the throughput bottleneck, so a realistic layout is 16 for the policy/actor, 16 for the critic, 8 for the reward model, 8 for the reference — **48 GPUs, six nodes**. If you skip the separate critic by using a critic-free method like GRPO, you drop to roughly 32 GPUs. The number is never small.

Once you cross a single node's GPU count, the network becomes a first-class concern. Inside one DGX-style node, GPUs talk over NVLink at hundreds of gigabytes per second. Between nodes, you are limited by the network fabric. A good cluster uses InfiniBand at 200–400 Gb/s (25–50 GB/s); a cheap one falls back to Ethernet at 10–25 Gb/s. The gap matters enormously for ZeRO-3, which all-gathers parameter shards on every forward and backward pass. If those all-gathers cross slow inter-node links, your GPUs sit idle waiting for weights. This is the single biggest reason multi-node RLHF feels slow when it is misconfigured: the compute is fine, the network is starving it.

It is worth being precise about *why* ZeRO-3 is so network-hungry, because it explains a lot of the layout decisions later in this post. In ZeRO-3 no single GPU holds the full layer weights; each rank holds a $1/N$ shard. So to run the forward pass of a layer, the rank must first all-gather the full parameter tensor from all the other ranks, use it for the matmul, then discard it. The backward pass does the same all-gather again, plus a reduce-scatter of gradients. For a 70B model that is hundreds of gigabytes of parameter traffic per step, repeated layer by layer. If all $N$ ranks are inside one node on NVLink, this is nearly free. If they span nodes on InfiniBand, it is significant but tolerable. If they accidentally span nodes on Ethernet, the GPUs spend more time waiting for weights to arrive than computing with them — and your expensive A100s run at maybe 20% utilization. The practical implication: keep each ZeRO-3 group as topologically tight as you can, ideally filling whole nodes before spilling to the next, and never let a data-parallel all-reduce cross a slow link if you can place it on a fast one.

There is a second, subtler reason 70B forces multi-node that has nothing to do with the policy's training state: the *rollout* itself wants GPUs. Generation with a 70B model in vLLM needs the full 140GB of weights resident plus a KV cache that grows with batch size and sequence length. A large rollout batch — and you want a large batch, because RLHF's sample efficiency improves with more diverse experience per step — can need tens of gigabytes of KV cache on top of the weights. So even the inference side of RLHF wants its own multi-GPU footprint at 70B, and it wants that footprint tuned for throughput (high `gpu_memory_utilization`, tensor parallelism sized to keep latency low) rather than for training. That divergence in what the rollout GPUs and the trainer GPUs each want is the deepest reason RLHF resists a one-size-fits-all uniform process group — and the reason a flexible scheduler like Ray earns its place.

The figure above shows the consequence of all this arithmetic on the architecture. Rather than cram everything onto one node and run it serially, you spread the four model groups across nodes and let them work in parallel. That is the entire game.

## 2. The Ray actor model, from first principles

To coordinate four heterogeneous model groups spread across dozens of GPUs, you need a distributed runtime. You could hand-roll it with `torch.distributed` process groups and a lot of `MPI`-style bookkeeping, and the early RLHF codebases did exactly that — and they were miserable to extend. Ray exists to make this tractable, and OpenRLHF, veRL, and several others build directly on it. So before we touch RLHF specifics, understand what Ray actually gives you.

Ray has three core primitives. **Tasks** are stateless functions you run remotely: you decorate a function with `@ray.remote`, call it with `.remote(args)`, and Ray schedules it on some worker in the cluster, returning a future (an `ObjectRef`) immediately. **Actors** are stateful: a class decorated with `@ray.remote` becomes a remote object that lives on one worker, holds state in memory, and exposes its methods as remote calls. This is the key primitive for RLHF, because a model is exactly a stateful object — you load 70B of weights once and then call `.generate()` or `.forward()` on them many times. **The object store** is a shared-memory layer: large objects (like a batch of generated sequences) are put into a distributed store and passed between tasks and actors by reference, so you do not re-serialize a gigabyte of tensors every time you hand them off.

Here is the minimal code illustration. An actor that holds state and answers remote calls:

```python
import ray

ray.init(address="auto")  # connect to an existing cluster

@ray.remote(num_gpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self, by):
        self.value += by
        return self.value

# Create the actor — it now lives on some GPU worker in the cluster.
counter = Counter.remote()

# Call its method remotely; returns a future immediately (non-blocking).
future = counter.increment.remote(5)

# Block and fetch the actual result from the object store.
print(ray.get(future))  # 5
```

Two things in that snippet do all the heavy lifting for RLHF. First, `num_gpus=1` tells Ray's scheduler this actor needs a GPU, and Ray will not place two GPU-claiming actors on the same physical GPU unless you ask it to — this is how you guarantee your reward model and your policy do not collide. Second, `.remote()` returns immediately. That asynchrony is the whole reason we can run generation, scoring, and reference computation concurrently: we fire off all three remote calls, get three futures back instantly, and only block when we actually need the results.

#### Worked example: why asynchrony buys you wall-clock time

Suppose in one PPO step the rollout (generation) takes 40 seconds, the reward scoring takes 12 seconds, and the reference log-prob pass takes 8 seconds, and all three operate on the same prompts. Run them serially on the same GPUs and you pay $40 + 12 + 8 = 60$ seconds per step. Run them as three concurrent Ray actor calls on separate GPUs and you pay $\max(40, 12, 8) = 40$ seconds — the wall-clock time of the *slowest* stage, not the sum. That is a 1.5× speedup before you have tuned anything, and it is the direct payoff of spending money on extra nodes. The catch, of course, is that the reward and reference stages need the generated text, so they cannot literally start until generation produces tokens — in the real overlapped loop you pipeline across micro-batches so that while the policy generates batch $k+1$, the reward model scores batch $k$. We will see exactly that pattern in the OpenRLHF loop.

Why is Ray a better fit for RLHF than a plain `torchrun` launch? Because RLHF is a *heterogeneous* workload. Pure pre-training is homogeneous: every GPU does the same forward-backward on the same model, and `torchrun` with a uniform process group is perfect. RLHF mixes fundamentally different jobs — autoregressive generation (which is latency-bound and loves vLLM's paged attention), reward inference (a single forward pass, throughput-bound), reference inference (another forward pass), and PPO training (forward-backward with optimizer state). These want different parallelism strategies, different batch sizes, and often different numbers of GPUs. Ray lets you express each as its own actor group with its own resource footprint, and schedule them onto the cluster independently. That heterogeneity is impossible to express cleanly in a single uniform process group.

The object store deserves more attention than it usually gets, because it is what keeps the orchestration from drowning in serialization cost. When the rollout actor finishes generating a batch of sequences, those sequences are large — thousands of token IDs across a batch of a thousand, plus attention masks. In a naive RPC system you would serialize that tensor, send it over the wire to the reward server, deserialize it, and repeat for the reference and critic. Ray instead `put`s the tensor into the distributed object store once, and every downstream actor that needs it receives an `ObjectRef` — a lightweight handle. On the same node, the consuming actor reads the data from shared memory with zero copy; across nodes, Ray transfers it once and caches it. For RLHF, where the same batch of sequences feeds three or four consumers, this single-write-many-read pattern is a meaningful saving, and it is invisible in your code: you just pass the future around.

A related Ray feature that RLHF leans on is the ability to express *dependencies* between remote calls without blocking. You can pass an `ObjectRef` directly as an argument to another `.remote()` call, and Ray will wait for the upstream result before scheduling the downstream task — building a dependency graph that Ray executes with maximum concurrency. So `reward.score.remote(actor.generate.remote(prompts))` is a valid expression that says "score the generated sequences," and Ray figures out that scoring cannot start until generation finishes, scheduling everything else that *can* run in the meantime. This declarative dataflow is what lets the training loop read like straight-line Python while executing as a concurrent graph across the cluster. It is also why a bug in your dependency structure shows up as a performance problem (no overlap) rather than a correctness problem — the results are right, they just arrive slower than they should.

## 2b. Ray cluster setup in practice

Before any of the actor primitives matter, you need a Ray cluster that actually exists across your nodes. This is the step people skip in tutorials and then struggle with in production, so let me walk through it from a cold start, because the failure modes here are quiet — a worker that never joins, a driver that connects to a half-formed cluster, a dashboard that shows fewer GPUs than you allocated.

A Ray cluster has exactly one **head node** and zero or more **worker nodes**. The head runs the global control store (the cluster's metadata brain), the scheduler, and the dashboard; the workers contribute compute and register themselves with the head. You start the head with `ray start --head`, which prints the address other nodes use to join:

```bash
# On the head node:
ray start --head --port=6379 --num-gpus=8 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265
# Ray prints something like:
#   Local node IP: 10.0.0.5
#   To add another node, run:
#     ray start --address='10.0.0.5:6379'
```

The `--port=6379` is the GCS (global control store) port that workers connect to. `--num-gpus=8` tells Ray how many GPUs this node contributes — Ray will not auto-detect more than you declare, and declaring fewer than physically present is a common way to accidentally under-provision. `--dashboard-host=0.0.0.0` makes the dashboard reachable from outside the node (bind to `127.0.0.1` and you can only see it via an SSH tunnel), and `--dashboard-port=8265` is the port you point a browser at.

On every worker node you run the join command the head printed:

```bash
# On each worker node:
ray start --address='10.0.0.5:6379' --num-gpus=8
```

Each worker contacts the head's GCS, registers its resources (8 GPUs here), and from that moment the head's scheduler can place actors on it. Within a few seconds `ray status` on the head reflects the new node. Run this on all five workers of a six-node job and the cluster reports 48 GPUs total.

**The Ray dashboard (port 8265)** is the single most useful diagnostic during bring-up and during a run. Point a browser at `http://<head-ip>:8265` and you see every node, every actor, per-actor GPU and memory usage, the object-store occupancy, and a live log stream per actor. For RLHF specifically, this is where you confirm that your four model groups landed on the GPUs you intended — that the trainer actually got 16 GPUs across two nodes, that the reward server is on its own 8, that no two GPU-claiming actors collided. When a job "hangs at startup," the dashboard usually shows the actual state: an actor stuck in `PENDING_CREATION` because the placement group could not be satisfied, which means you asked for more GPUs than the cluster has.

**Connecting the driver** is the final piece. Your RLHF program — the driver — runs on the head node (or anywhere that can reach the GCS) and connects with:

```python
import ray

ray.init(address="auto")   # discover the local cluster's head automatically
# or be explicit across a network:
# ray.init(address="10.0.0.5:6379")

print(ray.cluster_resources())   # {'GPU': 48.0, 'CPU': 576.0, ...}
print(ray.available_resources()) # what's free right now
```

`address="auto"` tells Ray to find the head that this node already belongs to (it reads a small file Ray wrote during `ray start`). The two resource calls are worth running first thing in your driver: `cluster_resources()` should report the full 48 GPUs, and if it reports fewer, a worker failed to join and you should fix that *before* launching a four-hour job that will silently under-provision.

**Placement groups for NVLink co-location vs spreading.** This is where the topology of your hardware meets the layout of your actors. Some actors must be on the *same* node to share NVLink — most importantly a tensor-parallel group, where each forward pass does an all-reduce across the group and you want that all-reduce on NVLink at hundreds of GB/s, not across the inter-node fabric. Other actors should be *spread* so their memory pressure does not collide. You express this with a placement group's bundles and strategy (covered in depth in the next section), but the principle to internalize now is: a tensor-parallel vLLM engine of 8 GPUs wants a single `PACK`ed bundle that lands all 8 on one node's NVLink island, while two independent inference servers want `SPREAD` so a memory spike in one never touches the other.

**The `@ray.remote(num_gpus=8)` decorator** is how an actor claims GPUs. When you decorate an actor class with `num_gpus=8`, Ray's scheduler treats 8 GPUs as a hard requirement: it will only place the actor where 8 GPUs are free, and it logically reserves them so nothing else lands on top. Inside the actor, Ray sets `CUDA_VISIBLE_DEVICES` to exactly those 8 GPUs, so your model code sees a clean 8-GPU world starting at device 0. This is the mechanism that makes "the reward server gets its own 8 GPUs" a one-line guarantee rather than a manual device-assignment chore:

```python
@ray.remote(num_gpus=8)
class TensorParallelEngine:
    def __init__(self, model_path):
        # Ray has already set CUDA_VISIBLE_DEVICES to this actor's 8 GPUs.
        import torch
        assert torch.cuda.device_count() == 8
        self.model = load_tp_model(model_path, tensor_parallel_size=8)
```

**InfiniBand inside Ray.** Ray itself does not move your training tensors over InfiniBand — NCCL does, inside the process groups your framework builds. But the environment Ray's worker processes inherit decides whether NCCL finds the fast fabric. The two variables that matter most are `NCCL_IB_DISABLE` (set to `0` to *use* InfiniBand; if it is `1`, NCCL ignores IB entirely and crawls over TCP) and `NCCL_IB_GID_INDEX` (which selects the RoCE/IB GID — on many cloud RoCE setups the default GID is wrong and collectives hang until you set this to the right index, often `3`). You set these in the environment of the `ray start` commands so every worker inherits them, or pass them through `runtime_env` when you create actors:

```python
ray.init(
    address="auto",
    runtime_env={"env_vars": {
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_DEBUG": "INFO",
    }},
)
```

The trap is that these must be set *before* the worker process that builds the NCCL communicator starts — setting them in your driver after `ray.init` does not retroactively fix workers that already launched. Putting them in the `ray start` environment (as the SLURM script in section 8 does) is the robust place.

**A full SLURM + Ray launch.** The section-8 script is the production version; here is the minimal shape so the structure is clear before the annotated version: allocate nodes, start the head on node 0, start workers on the rest pointing at the head, wait for all to register, then run the driver. The one subtlety beginners miss is that `ray start` (without `--block`) returns immediately, so you either pass `--block` and background it with `&`, or poll `ray status` until the expected node count appears before submitting the driver. Submitting the driver against a cluster that is still forming is the single most common cause of "it only used half my GPUs."

## 3. Ray for RLHF orchestration: actors and placement groups

Now we map the four RLHF models onto Ray actors. The pattern, which OpenRLHF formalizes, is to create one **actor group** per model role:

- A **rollout actor group** running the policy in generation mode (vLLM internally for speed).
- A **reward model server** doing inference-only scoring.
- A **reference policy server** computing baseline log-probs for the KL term.
- A **trainer actor group** holding the policy (and critic) in training mode with optimizer state.

Each group can itself span multiple GPUs and multiple nodes — the rollout actor might be eight GPUs of tensor-parallel vLLM, the trainer sixteen GPUs of ZeRO-3. Ray treats each group as a collection of actors and schedules them as a unit.

The crucial scheduling tool is the **placement group**. A placement group reserves a set of resource "bundles" (e.g. "one bundle of 8 GPUs") with a placement strategy that controls how those bundles map to physical nodes. The two strategies that matter for RLHF are `PACK` (put all bundles on as few nodes as possible — co-locate) and `STRICT_SPREAD` (force each bundle onto a different node). You use `PACK` when actors talk to each other constantly and want NVLink, and `SPREAD` when you want to isolate a model group on its own node so it does not contend for memory.

```python
import ray
from ray.util.placement_group import placement_group

ray.init(address="auto")

# Reserve four bundles, each 8 GPUs, packed onto contiguous nodes.
# One bundle per model role: trainer, critic, reward, reference.
pg = placement_group(
    bundles=[{"GPU": 8, "CPU": 32} for _ in range(4)],
    strategy="PACK",
)
ray.get(pg.ready())  # block until the scheduler secures the resources

@ray.remote(num_gpus=8)
class RewardModelServer:
    def __init__(self, model_path, bundle_index):
        from vllm import LLM  # illustrative; reward uses a scoring head
        self.model = load_reward_model(model_path, tensor_parallel_size=8)

    def score(self, sequences):
        # Returns a scalar reward per sequence; runs on this actor's 8 GPUs.
        return self.model.score_batch(sequences)

# Pin the reward server to bundle index 2 of the placement group.
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
reward = RewardModelServer.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=2,
    )
).remote(model_path="/models/reward-70b", bundle_index=2)
```

The `placement_group_bundle_index` is how you say "this actor goes into bundle 2," which under `PACK` keeps it adjacent to the others but on its own 8 GPUs. If instead you wanted the reward server on a physically separate node so its memory pressure never touches the trainer, you would build it with `strategy="STRICT_SPREAD"` for that bundle. Getting these strategies right is most of the art of laying out a multi-node RLHF job: co-locate things that communicate per-token, spread things that only exchange batches.

A subtle point that trips people up: the trainer and the rollout actor are *the same model* (the policy), but in different modes. You do not want two full copies of 70B weights with optimizer state. The standard pattern is that the trainer owns the authoritative weights with optimizer state, and after each PPO update it *broadcasts* the new weights to the rollout actor's vLLM engine, which holds an inference-only copy. That weight broadcast — a NCCL collective from trainer GPUs to rollout GPUs — is one of the per-step costs we will profile later. OpenRLHF supports both a co-located mode (trainer and rollout share GPUs, swapping the model in and out) and a separated mode (they live on different GPUs and synchronize over the network). The separated mode costs more GPUs but overlaps generation with training; the co-located mode saves GPUs but serializes the two phases. This is precisely the trade-off veRL pushes to its logical conclusion, as we will see.

The weight broadcast itself is worth dwelling on because it is a piece of plumbing that is easy to get wrong and expensive when you do. The trainer holds the policy sharded under ZeRO-3 across, say, sixteen GPUs; the vLLM rollout engine holds it sharded under tensor parallelism across, say, eight different GPUs. Those are two *different* shardings of the same logical 70B tensor. To synchronize them, the framework must gather the full updated weights on the trainer side and re-scatter them into the vLLM engine's layout — a NCCL broadcast (or a series of them, layer by layer) across the network between two GPU groups that are not in the same process group by default. OpenRLHF sets up a dedicated communication group spanning the trainer and the vLLM engines specifically for this transfer, initialized once at startup. If that group is misconfigured, you see it as either a hang at the first weight sync or a silent correctness bug where the rollout engine keeps generating with the initial weights and your reward never improves. When debugging "the policy isn't learning," confirming that the weight sync actually lands updated weights in the vLLM engine is one of the first checks, and it is why some teams log a hash of a few weight tensors on both sides each sync during bring-up.

## 4. OpenRLHF architecture deep dive

![A layered stack diagram showing the Ray cluster, placement groups, model actor groups, experience buffer, and PPO trainer](/imgs/blogs/multi-node-rl-training-ray-openrlhf-3.png)

OpenRLHF is the most approachable production-grade framework for multi-node RLHF, and its design is worth understanding class by class because it makes the abstract Ray pattern concrete. The stack above shows the layering: Ray orchestrates, placement groups pin GPUs, four actor groups hold the models, a shared experience buffer in the object store moves data between them, and the PPO trainer sits on top. Let us walk through each model role as OpenRLHF implements it.

**The Actor class (rollout).** This is the policy in generation mode. OpenRLHF's defining design choice — and the reason it is fast — is that the Actor runs **vLLM** for generation rather than naive Hugging Face `model.generate()`. vLLM's paged-attention KV cache and continuous batching mean the rollout phase, which is the single most expensive part of RLHF, runs at a fraction of the time it would otherwise take. Generation is autoregressive and memory-bandwidth bound; vLLM is purpose-built for exactly that. The Actor takes a batch of prompts, generates completions, and puts the resulting sequences into the object store.

**The Critic class (value).** In classic PPO the critic estimates the state-value $V(s)$ used to compute advantages. It is a model (often initialized from the reward model or the policy) with a scalar value head, and it runs a forward pass over the generated sequences to produce per-token value estimates. The critic *is* trained — it has optimizer state — so it lives in the trainer's resource footprint. Methods like GRPO eliminate the critic entirely by estimating the baseline from a group of samples per prompt, which is why GRPO RLHF needs fewer GPUs.

**The Reward class.** This runs the reward model — a model with a scalar reward head, trained beforehand on human preference data — in inference mode over completed sequences, producing a single scalar $r(x, y)$ per (prompt, response) pair. It is forward-only, no optimizer state, so it is cheap on memory relative to the trainer.

**The Reference class.** The reference policy is a frozen copy of the initial (post-SFT) model. Its job is to compute log-probabilities of the generated tokens under the *original* distribution. The PPO objective adds a KL penalty $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ that keeps the trained policy from drifting too far from this reference — which is what prevents reward hacking, where the policy discovers degenerate outputs that fool the reward model but read as gibberish. The reference is forward-only too.

**The Trainer class.** This owns the policy weights with full optimizer state (and the critic), runs the PPO clipped-surrogate update over the collected experience, and after each update broadcasts new weights to the Actor's vLLM engine. It is the most memory-hungry actor group.

How do they communicate? Through Ray remote calls and the object store. The trainer's training loop calls `actor.generate.remote(prompts)`, `reference.forward.remote(sequences)`, and `reward.score.remote(sequences)`, getting futures back. The large tensors (sequences, log-probs, rewards) live in the object store and are passed by reference, so handing a batch of generated sequences from the Actor to the Reward server does not re-serialize gigabytes — it passes a pointer. This is where Ray's design pays off: the orchestration code reads almost like single-machine Python, while internally the data is moving between nodes efficiently.

Here is a stripped-down sketch of how OpenRLHF wires the actor groups together, close to the real API shape:

```python
import ray
from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    RewardModelRayActor,
    ReferenceModelRayActor,
    create_vllm_engines,
)

ray.init(address="auto")

# vLLM engines for fast generation (the rollout/Actor inference path).
vllm_engines = create_vllm_engines(
    num_engines=4,
    tensor_parallel_size=2,     # each engine spans 2 GPUs
    pretrain="/models/policy-70b",
    seed=42,
    enable_prefix_caching=True,
)

# Trainer-side actor group (policy with optimizer state, ZeRO-3).
actor_group = ActorModelRayActor.options(num_gpus=16).remote(
    pretrain="/models/policy-70b",
    zero_stage=3,
)
critic_group = CriticModelRayActor.options(num_gpus=16).remote(
    pretrain="/models/reward-70b",  # critic often inits from reward model
    zero_stage=3,
)
# Inference-only servers.
reward_group = RewardModelRayActor.options(num_gpus=8).remote(
    pretrain="/models/reward-70b",
)
ref_group = ReferenceModelRayActor.options(num_gpus=8).remote(
    pretrain="/models/policy-70b",  # frozen SFT checkpoint
)
```

The numbers in that snippet are not arbitrary — they are the 48-GPU layout from the worked example in section 1, expressed as Ray actor placements. This is the moment the memory arithmetic becomes literal code.

## 5. The OpenRLHF training loop, step by step

![A pipeline diagram of one OpenRLHF training step from prompt sampling through parallel async stages to PPO epochs and weight sync](/imgs/blogs/multi-node-rl-training-ray-openrlhf-4.png)

Now the loop itself. One PPO step in OpenRLHF proceeds as the pipeline above shows: sample prompts, fire off the three inference stages asynchronously, wait for all of them, compute advantages, run several PPO epochs over the experience, then sync the updated weights back to the rollout engine. Let me write it out in the order the framework executes it, because the *ordering and the asynchrony* are the entire performance story.

First, sample a batch of prompts from the dataset. Then the rollout: the Actor generates completions for those prompts via vLLM. This is the slow step — for long generations on a 70B model it dominates wall-clock time. Crucially, while generation produces sequences, the reference and reward work for the *previous* batch can still be in flight, which is how OpenRLHF overlaps stages.

Once a batch of sequences exists, three things happen on three different GPU groups, all kicked off as async remote calls:

1. The **reference** computes log-probs of the generated tokens under the frozen policy (for the KL term).
2. The **reward** model scores each complete sequence (the terminal reward signal).
3. The **critic** computes per-token value estimates (the baseline for advantages).

The trainer fires all three with `.remote()`, gets futures, and then calls `ray.get()` to block until all complete. Here is the loop body, lightly simplified:

```python
import ray
import torch

def ppo_step(prompts, actor_group, critic_group, reward_group,
             ref_group, kl_coef=0.02, ppo_epochs=4):
    # 1. Rollout: policy generates completions (vLLM-backed).
    sequences_ref = actor_group.generate.remote(prompts)
    sequences = ray.get(sequences_ref)  # tokens + attention masks

    # 2. Fan out three inference stages asynchronously on separate GPUs.
    ref_logprobs_fut = ref_group.forward.remote(sequences)
    reward_fut       = reward_group.score.remote(sequences)
    values_fut       = critic_group.forward.remote(sequences)
    actor_logprobs_fut = actor_group.compute_logprobs.remote(sequences)

    # 3. Block once for all of them (they ran concurrently).
    ref_logprobs, rewards, values, actor_logprobs = ray.get(
        [ref_logprobs_fut, reward_fut, values_fut, actor_logprobs_fut]
    )

    # 4. KL-shaped reward, then GAE advantages.
    kl = actor_logprobs - ref_logprobs
    shaped_rewards = rewards - kl_coef * kl   # penalize drift from reference
    advantages, returns = compute_gae(shaped_rewards, values,
                                      gamma=1.0, lam=0.95)

    # 5. PPO epochs: several passes of the clipped surrogate over experience.
    for _ in range(ppo_epochs):
        actor_group.ppo_update.remote(sequences, actor_logprobs,
                                      advantages)
        critic_group.value_update.remote(sequences, returns)
    ray.get(actor_group.barrier.remote())

    # 6. Broadcast fresh policy weights to the vLLM rollout engine.
    ray.get(actor_group.broadcast_to_vllm.remote())
```

A few things deserve emphasis. The KL term in step 4 is the theoretical heart of why RLHF is stable: without it, PPO will happily push the policy toward whatever maximizes the reward model's score, and since the reward model is itself an imperfect learned function, the policy finds adversarial outputs that score high but are nonsense. The penalty $r' = r - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ pulls the policy back toward the reference distribution. From an information-theoretic view, KL divergence measures how many extra bits you would need to encode samples from $\pi_\theta$ using a code optimized for $\pi_{\text{ref}}$; bounding it bounds how far the policy can wander into low-probability, out-of-distribution territory where the reward model's estimates are unreliable. This is not a heuristic — it is the constraint that keeps you inside the region where your learned reward signal means anything.

The GAE computation in step 5 (Generalized Advantage Estimation) turns the per-token rewards and the critic's value estimates into advantage targets. Note `gamma=1.0` is common for RLHF because the "episode" is a single generation with the reward at the end; there is no long-horizon discounting to do. The advantage tells PPO how much better each token was than the critic expected, and PPO's clipped surrogate increases the probability of above-baseline tokens while clipping the update so a single step cannot move the policy too far.

Step 6, the weight broadcast, is the synchronization that ties the trainer back to the rollout engine. In the separated architecture the trainer and the vLLM engine are different GPUs, so this is a NCCL broadcast across the network. It must complete before the next rollout, because otherwise you would generate with stale weights — though some advanced setups deliberately allow slightly-stale generation to overlap more, accepting a small amount of off-policy-ness for throughput. That is an "asynchronous RLHF" optimization and it trades a bit of sample quality for speed.

#### Worked example: OpenRLHF throughput for a 70B run

Let me put numbers on a step. Suppose a batch of 1,024 prompts, each generating up to 512 new tokens, on a 70B policy with vLLM across 8 rollout GPUs. With vLLM's continuous batching, a reasonable generation throughput for a 70B model on A100s is in the low thousands of tokens per second per GPU at this batch size — call it an aggregate of roughly 6,000 tokens/sec across the 8-GPU rollout group. Generating $1{,}024 \times 512 \approx 524{,}000$ tokens then takes about $524{,}000 / 6{,}000 \approx 87$ seconds. The reward and reference forward passes over those sequences are single passes — far cheaper, perhaps 15 and 10 seconds respectively, and they overlap with each other so they cost ~15 seconds of the wall clock, hidden largely behind generation in the pipelined version. The four PPO epochs over the experience, with ZeRO-3 on 16 trainer GPUs, might run ~40 seconds. The weight broadcast adds ~5 seconds. So a step is roughly $87 + 40 + 5 \approx 132$ seconds with good overlap, versus the ~150+ seconds you would get running everything serially. Over a full run of, say, 1,000 PPO steps, that is about 37 hours — a day and a half — versus over two days serial. These numbers swing widely with sequence length and batch size, but the shape is right: **generation dominates**, which is exactly why OpenRLHF spends its cleverness on vLLM and on overlapping the rollout with everything else.

The key operational metric to watch is rollout-to-train ratio. If generation is 87 seconds and training is 40, you are spending two-thirds of every step generating. That tells you immediately where to invest: more rollout GPUs, faster generation (vLLM tuning, shorter max-tokens, prefix caching), or asynchronous generation that overlaps the next rollout with the current training. It is almost never worth optimizing the PPO update when generation is 2× its cost.

## 5b. OpenRLHF experience buffer deep dive

The loop above glossed over where the collected experience actually lives between the rollout and the PPO epochs. That storage is the **experience buffer**, and understanding it is the difference between treating OpenRLHF as a black box and being able to reason about sample staleness, batch diversity, and the memory the buffer consumes. It is also where Ray's object store stops being a black box and becomes a concrete data structure you tune.

Each entry in the buffer is one **experience tuple**: for a generated sequence, OpenRLHF stores the prompt, the response tokens, the reward the reward model assigned, the old log-probs the policy had when it generated the sequence (`log_prob_old`), the critic's value estimates, and the computed advantage (and return). Written as a record, an experience is roughly `(prompt, response, reward, log_prob_old, value, advantage)`. The reason all six travel together is that PPO's clipped surrogate needs them all at update time: the ratio $\pi_\theta(a|s) / \pi_{\text{old}}(a|s)$ requires `log_prob_old` captured *at generation time* (it is wrong to recompute it after the weights have moved), the advantage drives the gradient direction, and the value and return drive the critic's regression. Storing `log_prob_old` rather than recomputing it is not an optimization — it is a correctness requirement, because PPO is an off-policy correction *relative to the policy that generated the data*, and that policy's log-probs must be frozen at collection time.

**The buffer is a Ray object store structure.** Rather than serializing these tuples and shipping them between actors, OpenRLHF `put`s the experience tensors into Ray's distributed object store and passes `ObjectRef` handles. A batch of a thousand sequences with their log-probs, values, and advantages is large — tens to hundreds of megabytes of tensors — and the single-write-many-read pattern means the trainer's PPO epochs read it from shared memory with no re-serialization. On a single node the PPO update reads the buffer with zero copy; the buffer's residency in the object store is also why you watch object-store occupancy on the dashboard, since a buffer sized too large for the store's memory triggers spilling to disk and a sudden slowdown.

**The buffer-size tradeoff.** How many steps of experience you retain before discarding is a real knob with a real tension. A **large buffer** retains experience collected under several past versions of the policy, which gives the PPO update more batch diversity — a wider spread of prompts and behaviors per gradient step, which stabilizes training — but at the cost of *staleness*: some of that experience was generated by a policy several updates old, so its `log_prob_old` is increasingly distant from the current policy and the importance-sampling correction stretches further, which PPO's clipping was designed to tolerate only within a modest range. A **small buffer** keeps experience fresh — every tuple was generated by nearly the current policy, so the on-policy assumption holds tightly — but each gradient step sees a narrower slice of behavior, which can make the update noisier and the training less stable. The standard OpenRLHF posture is a buffer sized to roughly one rollout's worth of experience, consumed over a handful of PPO epochs and then discarded, which keeps staleness bounded to within those epochs while extracting several gradient steps from each expensive generation.

**Mini-batch sampling with replacement across PPO epochs.** Once a rollout's experience is in the buffer, OpenRLHF runs `ppo_epochs` passes over it (4 in our examples). Within each epoch the buffer is shuffled and chopped into mini-batches of `micro_train_batch_size`, and the policy and critic are updated on each mini-batch. The shuffle every epoch matters: it decorrelates the mini-batch composition between epochs so the optimizer does not see the same grouping of sequences repeatedly, which would bias the gradient. Conceptually each epoch draws mini-batches that together cover the buffer once (sampling without replacement *within* an epoch, reshuffled *between* epochs), so across four epochs each experience tuple contributes to four gradient steps under four different mini-batch contexts. This is how RLHF squeezes multiple updates out of one rollout — the rollout is the expensive part, so you reuse its experience several times before throwing it away and generating fresh.

**The shuffle+sample loop and `make_experience`.** OpenRLHF centralizes the construction of the buffer in a function conventionally called `make_experience` (in the experience-maker component). Its job is exactly the fan-out we saw in the loop, packaged: given a batch of prompts, it drives the Actor's vLLM generation, then dispatches the reference forward pass, the reward scoring, and the critic value pass, gathers the results, computes the KL-shaped reward and the GAE advantages, and assembles the per-sequence experience tuples. The output is a list of experiences that get pushed into the buffer. Here is the architecture in stripped-down form:

```python
def make_experience(prompts, actor, critic, reward, reference,
                    kl_coef=0.02):
    # 1. Rollout: the Actor generates responses (vLLM internally).
    sequences = ray.get(actor.generate.remote(prompts))

    # 2. Fan out the three inference passes concurrently on separate GPUs.
    logprob_old_fut = actor.compute_logprobs.remote(sequences)
    ref_logprob_fut = reference.forward.remote(sequences)
    reward_fut      = reward.score.remote(sequences)
    value_fut       = critic.forward.remote(sequences)
    log_prob_old, ref_logprob, reward_scores, values = ray.get(
        [logprob_old_fut, ref_logprob_fut, reward_fut, value_fut]
    )

    # 3. KL-shaped reward, then GAE advantages/returns.
    kl = log_prob_old - ref_logprob
    shaped = reward_scores - kl_coef * kl
    advantages, returns = compute_gae(shaped, values, gamma=1.0, lam=0.95)

    # 4. Assemble experience tuples and return them for the buffer.
    return [
        Experience(prompt=p, response=r, reward=rw,
                   log_prob_old=lp, value=v, advantage=a, ret=ret)
        for p, r, rw, lp, v, a, ret in zip(
            prompts, sequences, reward_scores, log_prob_old,
            values, advantages, returns)
    ]

def train_on_buffer(buffer, actor, critic, ppo_epochs=4,
                    micro_batch=4):
    for _ in range(ppo_epochs):
        buffer.shuffle()                       # reshuffle each epoch
        for mb in buffer.iter_minibatches(micro_batch):
            actor.ppo_update.remote(mb)        # clipped surrogate
            critic.value_update.remote(mb)     # value regression
```

The split between `make_experience` (collect once, expensive, generation-bound) and `train_on_buffer` (consume several times, shuffle each epoch) is the structural reason OpenRLHF gets good GPU economics: the costly generation happens once per rollout, and the cheap reuse happens four times. When you tune `ppo_epochs` upward you extract more gradient steps per rollout — cheaper per update — but you also let the policy drift further from the data that generated the buffer, increasing staleness within the buffer's own lifetime. Four is the usual sweet spot for exactly that reason.

## 6. veRL and the HybridEngine: the opposite bet

![A graph showing the veRL HybridEngine sharing one GPU set between a vLLM generation phase and an FSDP training phase before merging to a single model footprint](/imgs/blogs/multi-node-rl-training-ray-openrlhf-7.png)

OpenRLHF's separated architecture spends GPUs to overlap stages: the rollout engine and the trainer live on different hardware, so generation and training can proceed concurrently, at the cost of holding (at least) an inference copy of the policy in vLLM *and* a training copy in the trainer. veRL (Volcano Engine Reinforcement Learning, sometimes written verl) makes the opposite bet with its **HybridEngine**, and the contrast is instructive because it is the central design fork in this whole space.

The HybridEngine puts generation and training on the *same* GPU set and time-shares them. During the rollout phase, those GPUs run vLLM with a paged KV cache, configured for fast inference. When generation finishes, the engine *resharding* — it tears down the inference layout and reconfigures the same weights into a training layout (FSDP or Megatron with gradients and optimizer state), runs the PPO update, then switches back. The figure above shows this: one shared GPU set branches into a generation phase and a training phase, with a phase switch between them, merging into a single resident model footprint.

The payoff is memory. In the separated design you pay for an inference copy of the policy *plus* a training copy; in the HybridEngine there is one set of weights that morphs between modes. For a 70B policy that can be the difference between needing 32 GPUs and needing 24, because you are not duplicating the model. The cost is that generation and training can no longer overlap — they are sequential phases on the same hardware, plus you pay the resharding overhead each transition (moving weights between the vLLM-friendly and FSDP-friendly layouts, which involves all-gather and re-partition collectives).

veRL also differs in control philosophy. OpenRLHF is **multi-controller**: each actor group runs its own controller logic, and they coordinate as peers via Ray calls. veRL is **single-controller**: a central driver process orchestrates the whole dataflow, issuing commands to worker groups. The single-controller design makes the dataflow programmable and easy to reason about — you write the RL algorithm as a more-or-less linear Python program in one place, and veRL handles distributing each operation. This is genuinely nicer for *research*, where you want to modify the algorithm (try a new advantage estimator, a different reward shaping) without touching distributed plumbing. The multi-controller design is sometimes faster at scale because there is no central bottleneck, but the single-controller design is far easier to extend.

Which wins on throughput? It depends on your bottleneck. If generation dominates and you have GPUs to spare, OpenRLHF's overlap of rollout-with-training wins — the trainer GPUs are not idle during generation. If GPUs are scarce and you would otherwise leave the rollout GPUs idle during training, the HybridEngine's GPU reuse wins because you are never holding idle duplicate hardware. In published comparisons the two trade leads depending on model size, sequence length, and cluster topology; the honest answer is "benchmark both on your workload." As a rule of thumb I have found: GPU-rich teams optimizing for fastest iteration reach for OpenRLHF's separated mode; GPU-constrained teams or researchers iterating on the algorithm itself reach for veRL.

```python
# veRL-style single-controller config (illustrative, hydra-style).
# One driver program describes the whole RLHF dataflow; veRL distributes it.
config = {
    "actor_rollout_ref": {
        "model": {"path": "/models/policy-70b"},
        "rollout": {
            "name": "vllm",
            "tensor_model_parallel_size": 8,
            "gpu_memory_utilization": 0.6,   # leave room for the training reshard
        },
        "hybrid_engine": True,               # share GPUs between gen and train
        "actor": {
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_epochs": 4,
        },
    },
    "critic": {"strategy": "fsdp", "model": {"path": "/models/reward-70b"}},
    "reward_model": {"model": {"path": "/models/reward-70b"}},
    "algorithm": {"kl_ctrl": {"kl_coef": 0.02}, "adv_estimator": "gae"},
}
```

The `hybrid_engine: True` flag and `gpu_memory_utilization: 0.6` are the tell-tale signs of the design: you deliberately leave 40% of GPU memory free during generation so there is room to reshard into the training layout without an OOM. That headroom is the price of the HybridEngine, and it is usually cheaper than a duplicate model.

### The HybridEngine, implemented

It is worth getting concrete about what "switching modes" physically means, because the engineering inside that switch is the whole reason the HybridEngine saves memory. On the same GPU set, the engine maintains *one* logical copy of the model weights but presents two completely different memory regimes around them.

In **generation mode**, the GPUs run vLLM. The weights are laid out for tensor-parallel inference, and the dominant memory consumer is the **paged KV cache** that vLLM manages for continuous batching — there are *no* gradient buffers, no optimizer state, no saved activations, because nothing is being trained. The engine configures `gpu_memory_utilization` so vLLM claims memory for the KV cache up to the headroom limit and no further.

In **training mode**, the picture inverts. The KV cache is gone — generation is finished — and the freed memory is reclaimed for the things training needs: **FSDP gradient buffers** (the sharded gradients for the backward pass), the optimizer state, and the activations saved for backpropagation. The same weights that vLLM was reading for inference are now the parameters FSDP shards and updates.

The transition between the two is the **reshard**, and the API that triggers it looks like a pair of calls:

```python
# Conceptual HybridEngine API (illustrative).
engine.switch_to_generation()   # free FSDP/optim state, build vLLM KV cache
responses = engine.generate(prompts)        # rollout phase

engine.switch_to_training()     # free KV cache, build FSDP grad/optim buffers
engine.ppo_update(experience)               # training phase
```

`switch_to_training()` tears down vLLM's KV cache, re-partitions the weights from the inference (tensor-parallel) sharding into the training (FSDP) sharding via all-gather and re-scatter collectives, and allocates the gradient and optimizer buffers. `switch_to_generation()` does the reverse: it frees the gradient/optimizer memory and rebuilds the KV cache for the next rollout. For a **7B model the switch takes roughly 2 seconds** — fast enough that, amortized over a rollout-plus-update step measured in tens of seconds, it is a small overhead. The reshard cost grows with model size (more weight bytes to re-partition) and with how different the two shardings are, which is why HybridEngine's per-step overhead is more noticeable at 70B than at 7B even though the relative memory saving is larger.

The **memory saving** is the payoff. In OpenRLHF's separated design you hold an inference copy of the policy in the vLLM engine *and* a training copy in the trainer — two resident copies of the weights. For a 7B model in bf16 that second copy is about **14 GB of weights you do not have to duplicate** (7B × 2 bytes), and at 70B it is 140 GB across the rollout GPUs that you reclaim by not duplicating. The HybridEngine pays for that saving with the reshard overhead and the lost ability to overlap generation with training — it is fundamentally sequential, generate-then-train on the same hardware — which is the trade the next table quantifies.

### veRL vs OpenRLHF: throughput at 7B and 70B

The honest comparison is "benchmark both on your workload," but it helps to have the shape of the trade in front of you. The table below sketches the characteristic differences; treat the throughput figures as illustrative orders of magnitude that swing with sequence length, batch size, and interconnect, not as a leaderboard.

| Dimension | OpenRLHF (separated) | veRL (HybridEngine) |
| --- | --- | --- |
| GPU layout | Rollout and trainer on *different* GPUs | Rollout and trainer *share* the same GPUs |
| Resident weight copies | Two (inference + training) | One (morphs between modes) |
| GPUs for 7B run | ~16 (separated groups) | ~8 (shared, time-shared) |
| GPUs for 70B run | ~48 (separated groups) | ~24–32 (shared) |
| Gen/train overlap | Yes — rollout overlaps training | No — sequential phases |
| Per-step overhead | Weight broadcast to vLLM (~seconds) | Reshard switch (~2s at 7B, more at 70B) |
| 7B throughput (relative) | Higher if GPU-rich (overlap wins) | Competitive; far better GPUs-per-dollar |
| 70B throughput (relative) | Higher with spare GPUs to overlap | Higher when GPUs are the binding constraint |
| Control model | Multi-controller (peer actors) | Single-controller (central driver) |
| Best for | GPU-rich, fixed recipe, fastest iteration | GPU-constrained, algorithm research |

The pattern the table encodes: at **7B**, where everything fits comfortably, veRL's GPU reuse means you finish a comparable run on roughly half the GPUs, so even if OpenRLHF edges it on raw wall-clock when given more hardware, veRL usually wins on cost-per-completed-run. At **70B**, the question becomes whether you have GPUs to spare. If you do, OpenRLHF's overlap keeps the trainer busy during generation and wins on wall-clock; if GPUs are the binding constraint, veRL's single-footprint design lets you run at all on hardware where OpenRLHF's duplicate copies would not fit. The reshard overhead is the tax veRL pays for that flexibility, and at 70B it is large enough that a GPU-rich team optimizing purely for speed will often still prefer OpenRLHF — which is exactly the GPU-rich-vs-GPU-constrained fork the rest of this section describes.

## 7. NeMo-RLHF and 3D parallelism for 175B and beyond

When you push past 70B into 175B and larger, even the OpenRLHF and veRL approaches strain, and you reach for NVIDIA's NeMo-RLHF (more recently exposed through the NeMo-Aligner toolkit), built on NeMo and Megatron-LM. Its distinguishing capability is **3D parallelism**: combining data parallelism, tensor parallelism, and pipeline parallelism simultaneously.

Recall the three axes. **Data parallelism** replicates the model and splits the batch — each replica processes different examples and gradients are all-reduced. **Tensor parallelism** splits individual layers across GPUs — a single matrix multiply is partitioned so each GPU computes a slice, requiring all-reduce within each layer. **Pipeline parallelism** splits the model by layers across GPUs — GPU 0 holds layers 1–10, GPU 1 holds layers 11–20, and micro-batches flow through the pipeline. At 175B, you need all three: tensor parallelism within a node (it is communication-heavy, so it wants NVLink), pipeline parallelism across a few nodes (it only passes activations at layer boundaries, so it tolerates slower links), and data parallelism across the remaining nodes for throughput.

The product of the three degrees must equal your total GPU count. For a 175B model on 256 GPUs you might use tensor-parallel 8 (within a node), pipeline-parallel 8 (across eight nodes), and data-parallel 4 — because $8 \times 8 \times 4 = 256$. NeMo's Megatron-Core provides the RL primitives that make PPO work under this layout: it knows how to compute log-probs, run the forward pass for the reward and reference models, and do the PPO update while the model is sharded three ways. Doing this by hand is brutal; NeMo-RLHF's value is that the 3D-parallel bookkeeping is already correct.

The trade-off is rigidity and infrastructure assumptions. NeMo-RLHF expects a Megatron-format checkpoint and a high-end NVLink-rich cluster (DGX SuperPOD-class) where tensor parallelism inside a node is fast. On a commodity cloud cluster with weaker inter-node fabric, the 3D-parallel communication can dominate and you lose the advantage. NeMo-RLHF is the right tool when you are NVIDIA-adjacent, training genuinely massive models, and have the NVLink hardware to feed tensor parallelism. For most teams at 70B on cloud A100s/H100s, OpenRLHF or veRL is the pragmatic choice; NeMo-RLHF is for the 175B+ frontier.

![A matrix comparing TRL, OpenRLHF, veRL, NeMo-RLHF, and DeepSpeed-Chat across max scale, async generation, multi-node support, and best fit](/imgs/blogs/multi-node-rl-training-ray-openrlhf-5.png)

The matrix above lays out the landscape. The honest summary: TRL is for prototyping at 7B on a single node and is where most people *learn* RLHF, but it does not scale to multi-node 70B comfortably. DeepSpeed-Chat was the first open end-to-end RLHF pipeline and proved the concept, but it lacks vLLM-class generation and async support, so it has been largely superseded. OpenRLHF, veRL, and NeMo-RLHF are the three live options for serious scale, distinguished by the design choices we have walked through. The timeline of how we got here is worth a glance.

![A timeline of scalable RLHF from InstructGPT in 2022 through DeepSpeed-Chat, OpenRLHF, veRL, and vLLM-accelerated rollouts in 2024](/imgs/blogs/multi-node-rl-training-ray-openrlhf-6.png)

## 8. SLURM and NCCL: actually launching on a cluster

Everything above is architecture. This section is the part that makes the difference between a job that runs and a job that hangs at startup with no error — the cluster launch and the NCCL environment.

Most HPC and many cloud GPU clusters use **SLURM** as the scheduler. You submit a batch script with `sbatch`, request nodes and GPUs, and `srun` launches your program across the allocated nodes. The pattern for a Ray-based RLHF job is: start the Ray head on the first node, start Ray workers on the rest, then submit the OpenRLHF driver to the head. Here is a working skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=rlhf-70b
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --exclusive

# --- NCCL / network configuration (the part that actually matters) ---
export NCCL_IB_DISABLE=0                 # USE InfiniBand (0 = enabled)
export NCCL_SOCKET_IFNAME=ib0            # the InfiniBand interface name
export NCCL_IB_HCA=mlx5                  # the IB host channel adapters
export NCCL_DEBUG=INFO                   # log the transport NCCL actually picks
export NCCL_P2P_LEVEL=NVL                # prefer NVLink for intra-node P2P

# --- Find the head node and its address ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(echo "$nodes" | head -n1)
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379

# --- Start the Ray head on node 0 ---
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_ip" --port=$port \
    --num-gpus=8 --block &
sleep 20

# --- Start Ray workers on the remaining nodes ---
worker_nodes=$(echo "$nodes" | tail -n +2)
for node in $worker_nodes; do
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$head_ip:$port" --num-gpus=8 --block &
done
sleep 20

# --- Submit the OpenRLHF driver to the Ray cluster ---
srun --nodes=1 --ntasks=1 -w "$head_node" \
    python -m openrlhf.cli.train_ppo_ray \
        --pretrain /models/policy-70b \
        --reward_pretrain /models/reward-70b \
        --actor_num_gpus_per_node 8 --actor_num_nodes 2 \
        --critic_num_gpus_per_node 8 --critic_num_nodes 2 \
        --vllm_num_engines 4 --vllm_tensor_parallel_size 2 \
        --zero_stage 3 --bf16 \
        --micro_train_batch_size 4 --train_batch_size 256 \
        --prompt_max_len 1024 --generate_max_len 512 \
        --init_kl_coef 0.02 --save_path /ckpt/rlhf-70b
wait
```

The NCCL block at the top is the single most important and most under-documented part. Let me explain each variable, because getting one wrong silently halves your throughput or hangs the job:

- `NCCL_IB_DISABLE=0` enables InfiniBand. The trap: if your fabric is misconfigured, NCCL silently falls back to TCP over Ethernet and your inter-node all-gathers crawl. Setting this to `0` (use IB) and then watching `NCCL_DEBUG=INFO` output to confirm it actually chose IB is essential. Many "RLHF is mysteriously slow" reports are an unintended TCP fallback.
- `NCCL_SOCKET_IFNAME=ib0` tells NCCL which network interface to use for its bootstrap and, on some setups, data. If you leave it unset, NCCL may bind to the management Ethernet interface (`eth0`) instead of the fast fabric. On clusters where the IB interface is named differently (`ibp...`, `bond0`), set it accordingly — `ip addr` on a node tells you the name.
- `NCCL_IB_HCA=mlx5` selects the Mellanox host channel adapters when a node has several network cards, so NCCL uses the high-speed ones.
- `NCCL_DEBUG=INFO` makes NCCL print the transport it selected at startup. The first time you bring up a cluster, you *read this log* and confirm it says it is using IB, not "via SOCKET." This one habit saves days.
- `NCCL_P2P_LEVEL=NVL` tells NCCL to use NVLink for peer-to-peer transfers within a node, which is what you want for tensor-parallel groups co-located on a node.

The structure of the launch is also worth internalizing: Ray needs a head and workers, and SLURM gives you a set of nodes but does not start Ray for you. You pick node 0 as the head, start the head process, give the workers its address, and wait for them all to register before submitting the driver. The two `sleep` calls are crude but real — Ray's gossip-based cluster formation takes a few seconds, and submitting the driver before all workers have joined leads to under-provisioning. In production you would replace the sleeps with a poll on `ray status` until the expected node count appears.

## 9. Fault tolerance: surviving node failures

A 70B RLHF run takes a day or more across dozens of nodes, and at that scale hardware *will* fail — a GPU will fall off the bus, a node will reboot, the network will hiccup. If a single node failure kills the whole job and you lose 30 hours of progress, you have a serious operational problem. So fault tolerance is not optional at scale; it is the difference between finishing and never finishing.

The first line of defense is **checkpointing**, and the question is frequency. Checkpoint too rarely and a failure costs hours; checkpoint too often and you waste GPU time writing terabytes of optimizer state to disk. The sensible policy is to checkpoint based on expected failure rate: if your cluster's mean time between node failures is, say, 12 hours across 48 GPUs, checkpointing every hour means you lose at most an hour of work and spend a small percentage of your time on I/O. A 70B model's full training state (sharded optimizer included) is on the order of a terabyte, so write it to a fast parallel filesystem and write it asynchronously if your framework supports it, so training does not stall during the save.

Ray adds a second layer: **fault-tolerant actor restart**. Because each model group is a Ray actor, Ray can detect when an actor dies (its worker process crashes or its node goes down) and restart it. You configure this with `max_restarts`:

```python
@ray.remote(num_gpus=8, max_restarts=3, max_task_retries=2)
class RewardModelServer:
    def __init__(self, checkpoint_path):
        # On restart, Ray re-runs __init__, so reload from the last checkpoint.
        self.model = load_reward_model(checkpoint_path)
        self.step = load_step_marker(checkpoint_path)

    def score(self, sequences):
        return self.model.score_batch(sequences)
```

`max_restarts=3` tells Ray to restart this actor up to three times if it dies. The actor's `__init__` re-runs on restart, so the right pattern is to make `__init__` reload from the latest checkpoint — that way a restarted actor comes back at the last saved state rather than from scratch. `max_task_retries` handles transient failures of in-flight method calls.

The subtle issue is **experience buffer recovery**. If a node dies mid-step, the partially-collected rollout experience in the object store may be lost. OpenRLHF's pragmatic approach is to treat the PPO step as the atomic unit: if a step fails partway, discard the partial experience and re-run the step from the last consistent checkpoint of model weights. You do not try to recover half a rollout; you re-generate. This is acceptable because a single step is minutes, not hours, and the experience is regenerable from the prompts. The data you cannot afford to lose is the model weights and optimizer state, which is what you checkpoint.

There is also **elastic training** — the idea that the job can continue with fewer nodes if some fail, rather than waiting for replacements. Ray supports elastic actor groups in principle, but for RLHF where parallelism degrees are baked into the layout (your ZeRO-3 group expects a fixed number of ranks), true elasticity is harder than for stateless workloads. The realistic posture for most teams is: checkpoint frequently, configure actor restarts, request a couple of spare nodes if the scheduler allows, and accept that a catastrophic multi-node failure means resuming from the last checkpoint. That is good enough to finish a multi-day run reliably.

### RLHF-specific failure modes and how to diagnose them

It helps to know *what* fails in RLHF specifically, because the failure modes are not the same as in plain pre-training. The three that bite most often:

- **OOM from long responses.** RLHF generates variable-length responses, and the moment a batch happens to contain many sequences that run to `generate_max_len`, the activation and KV-cache footprint spikes above what shorter batches needed — and you OOM mid-rollout on a step that looks identical to the thousand steps before it. The fix is to budget memory for the *worst-case* batch (all sequences at max length), not the average, and to cap `generate_max_len` realistically. This is the most insidious failure because it is intermittent and batch-dependent.
- **NCCL timeout on slow nodes.** Collectives (the ZeRO-3 all-gathers, the weight broadcast) have a timeout, and if one straggler node — thermal-throttled, sharing a link with a noisy neighbor — falls behind, the whole collective times out and the job dies with a `NCCL timeout` rather than a clean error. Raising `NCCL_TIMEOUT` masks the symptom; the real fix is finding and draining the slow node (the profiling in section 10 is how you find it).
- **Reward model crash.** The reward server is a separate actor running its own inference, and if it crashes (a CUDA error, an OOM on an unusually long sequence), the whole step stalls waiting on a reward future that never resolves. This is precisely the case Ray actor restarts are for — but only if you have wired the reward actor to reload from checkpoint on restart.

**Checkpoint strategy.** A correct RLHF checkpoint is more than model weights. To resume a PPO run cleanly you need the **model weights**, the **optimizer state** (Adam's moments — resuming with fresh optimizer state restarts the learning-rate warmup and momentum, perturbing training), and the **step counter / data position** so you resume at the right place in the prompt stream. The experience buffer is *not* checkpointed — as established above, it is regenerable, so a resumed run simply re-generates the rollout for the step it died on rather than trying to restore half-collected experience. Save every N steps to a fast parallel filesystem, asynchronously if your framework supports it so training does not stall on the write.

**Diagnosing actor crashes** is where Ray's state API earns its keep. When an actor dies, the error that killed it is captured in Ray's state, and you query it rather than grepping through dozens of per-node logs:

```python
from ray.util.state import get_worker  # state API surface
import ray

# After a step fails, inspect which actor died and why.
# The Ray state API exposes worker/actor errors and statuses; in practice:
#   - `ray list actors` / the dashboard show DEAD actors and their exit detail
#   - the state API surfaces the captured traceback for the crashed worker
errors = ray.util.state.get_worker_errors()  # captured worker tracebacks
for e in errors:
    print(e.worker_id, e.error_type, e.error_message)
```

The point is operational: instead of SSHing to six nodes hunting for the one with a CUDA stack trace, you ask Ray which worker died and read the captured error directly — turning a multi-node debugging slog into a single query. The dashboard's per-actor view (section 2b) shows the same information visually, with the dead actor flagged and its last logs attached.

**Elastic restart from checkpoint.** Combining the pieces: configure each model-group actor with `max_restarts`, make every actor's `__init__` reload from the latest checkpoint, and Ray will restart a failed actor in place and bring it back at the last saved state. For the inference-only servers (reward, reference) this is nearly free — they reload weights and resume. For the trainer it means resuming weights *and* optimizer state. The job-level recovery is then: the failed step is discarded, the restarted actors reload from the last checkpoint, and the loop re-runs that step's rollout from the recovered weights.

**Checkpoint overhead and the cadence it implies.** A full 70B checkpoint — sharded weights plus sharded Adam optimizer state, on the order of a terabyte — takes about **5 minutes to write** even to a fast parallel filesystem (asynchronous saving hides most of that from the training loop, but the I/O is real). That cost dictates the cadence: checkpoint **every 100 steps**. At our ~132-second step, 100 steps is roughly 3.6 hours of work, so a failure costs at most ~3.6 hours of recompute, while the 5-minute write amortized over 100 steps is well under 3% I/O overhead. Checkpointing every 10 steps would cut worst-case loss but spend 25%+ of wall-clock writing; every 1,000 steps would risk losing a day and a half of work to a single node failure. Every 100 steps is the balance point where the I/O tax is small and the worst-case loss is a few hours, not a day.

## 10. Profiling a multi-node RLHF job

You cannot optimize what you do not measure, and the failure mode of a distributed RLHF job is rarely a crash — it is that it runs, slowly, and you do not know why. The discipline is to instrument per-stage timing on every step and watch a small dashboard of numbers. The four stage times that matter are rollout time, reward time, reference time, and PPO-update time per epoch, plus the weight-sync time. Log them every step:

```python
import time

def timed_ppo_step(prompts, groups):
    t = {}
    t0 = time.time()
    sequences = ray.get(groups["actor"].generate.remote(prompts))
    t["rollout"] = time.time() - t0

    t0 = time.time()
    futs = {
        "reward": groups["reward"].score.remote(sequences),
        "ref":    groups["ref"].forward.remote(sequences),
        "value":  groups["critic"].forward.remote(sequences),
    }
    results = {k: ray.get(f) for k, f in futs.items()}
    t["inference_overlap"] = time.time() - t0   # wall time of the parallel block

    t0 = time.time()
    ray.get(groups["actor"].ppo_update.remote(sequences, results))
    t["ppo"] = time.time() - t0

    t0 = time.time()
    ray.get(groups["actor"].broadcast_to_vllm.remote())
    t["weight_sync"] = time.time() - t0

    print({k: round(v, 1) for k, v in t.items()})
    return t
```

When you read those numbers, you are looking for the dominant stage and for *imbalance*. The dominant stage tells you where to invest. The imbalance tells you about your hardware: if the reference forward pass takes wildly longer than the reward pass despite similar model sizes, you probably have the reference on a slower or more-contended node. Tracking per-node GPU utilization (via `nvidia-smi` or DCGM exported to your metrics system) reveals the *bottleneck node* — the one node that everyone waits on. In a multi-node job, a single straggler node (perhaps with a thermal-throttled GPU or a noisy neighbor) drags the whole synchronized step down to its speed.

The second thing to watch is **network bandwidth utilization**. If your steps are slow and the GPUs show low utilization with high wait time, the network is likely the culprit — the ZeRO-3 all-gathers or the weight broadcast are starving the GPUs. Tools like `ib_write_bw` (an InfiniBand bandwidth benchmark) let you verify your fabric delivers its rated speed point-to-point, and NCCL's own logs (from `NCCL_DEBUG=INFO`) tell you the algorithmic bandwidth NCCL achieves on its collectives. If `ib_write_bw` shows 24 GB/s but your job behaves as if it has 2 GB/s, you have a configuration problem — almost always the silent TCP fallback from section 8.

The third metric is the **rollout-to-train ratio**, which I mentioned earlier. It is the single most actionable number. If rollout is 70% of step time, the lever is generation: add rollout GPUs, shorten `generate_max_len`, enable vLLM prefix caching, or move to asynchronous generation. If PPO is the larger fraction (unusual but possible with very short generations and large models), the lever is the trainer: more aggressive ZeRO offloading, larger micro-batches, or better tensor parallelism. The mistake I see most often is teams optimizing the PPO update — recompilation, fused kernels — while generation quietly eats two-thirds of every step. Profile first, optimize second.

## 11. The dollar cost of an RLHF run

Compute is the constraint that turns architecture choices into budget decisions, so let us put real money on it. Cloud A100-80GB instances run roughly \$3–4 per GPU-hour on-demand at major providers and substantially less under committed-use or reserved pricing; H100s run higher, perhaps \$4–8 per GPU-hour on-demand, but deliver enough more throughput (especially with FP8 and faster interconnect) that cost-per-token often *favors* them. Spot/preemptible instances can cut prices by 60–80% but can be reclaimed at any moment, which is why the fault tolerance of section 9 is what makes spot viable for RLHF at all.

#### Worked example: estimating the cost of a full 70B RLHF run

Take the 48-GPU layout and the ~37-hour run estimate from section 5. At \$3.50 per A100-80GB GPU-hour on-demand, that is $48 \times 37 \times 3.50 = \$6{,}216$ for one full RLHF run. Round up to about \$6,000–7,000 to account for startup, checkpointing overhead, and a failed-then-resumed attempt or two. Now compare scales. A 7B RLHF run fits on a single 8-GPU node and finishes a comparable number of steps in maybe 6 hours: $8 \times 6 \times 3.50 = \$168$. A 13B run on two nodes (16 GPUs) for ~10 hours: $16 \times 10 \times 3.50 = \$560$. So the cost ladder from 7B to 13B to 70B is roughly \$170 → \$560 → \$6,500 — the jump to 70B is more than 10× the 13B cost, driven by both more GPUs and longer wall-clock per step. This is why teams iterate algorithm and reward-model choices at 7B and only commit to a 70B RLHF run once the recipe is locked.

How does this compare to the SFT (supervised fine-tuning) that precedes it? SFT is dramatically cheaper per unit of progress because it has no rollout — no generation loop, no four models, just forward-backward on labeled data. A 70B SFT pass over a few billion tokens might cost a fraction of the RLHF run for a comparable wall-clock budget, and crucially its cost-per-token is far lower because every GPU is doing the same dense training work with no idle generation time. The rule of thumb: **RLHF's cost-per-token is several times SFT's**, because generation is expensive and the four-model orchestration leaves GPUs less than fully utilized. This is part of why methods that simplify the loop matter economically — DPO (Direct Preference Optimization) eliminates the reward model and the rollout entirely, training directly on preference pairs like a supervised objective, which is why DPO runs cost a fraction of PPO-RLHF runs. If your alignment quality holds with DPO, it is the cheaper path; PPO-RLHF earns its cost when online generation against a reward model genuinely outperforms offline preference learning.

Instance selection follows from the bottleneck. Because generation dominates and benefits from memory bandwidth and fast interconnect, H100s with NVLink and a good InfiniBand fabric often deliver better cost-per-completed-step than A100s despite the higher hourly rate — you finish in fewer hours. The spot-instance strategy is to run the inference-only servers (reward, reference) on spot, since they are stateless and trivially restartable, while keeping the trainer (with its expensive-to-rebuild optimizer state) on on-demand instances. That hybrid often cuts 30–40% off the bill while keeping the run robust.

The before/after below captures the whole architectural argument of this post in one frame: the single-node design is cheaper in GPUs but caps you at small models with idle hardware, while the multi-node design costs more GPUs but is the only thing that reaches 70B and keeps the hardware busy.

![A before and after comparison contrasting single-node serial RLHF with multi-node parallel Ray RLHF](/imgs/blogs/multi-node-rl-training-ray-openrlhf-2.png)

## 12. A complete multi-node setup, end to end

Let me assemble the pieces into one coherent picture you could actually run. The full stack is: a SLURM allocation of six nodes (48 A100-80GB GPUs), a Ray cluster started across them by the launch script from section 8, and OpenRLHF's `train_ppo_ray` driver configuring the actor groups. The model layout is 16 GPUs for the policy trainer (ZeRO-3 across two nodes), 16 for the critic, 4 vLLM engines of tensor-parallel-2 (8 GPUs) for rollout, 8 for the reward server, and the reference folded into spare capacity. Here is the OpenRLHF driver invocation with the full set of flags that matter, annotated:

```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain /models/policy-70b \
    --reward_pretrain /models/reward-70b \
    --save_path /ckpt/rlhf-70b \
    `# --- model group placement (maps to the 48-GPU layout) ---` \
    --actor_num_nodes 2 --actor_num_gpus_per_node 8 \
    --critic_num_nodes 2 --critic_num_gpus_per_node 8 \
    --reward_num_nodes 1 --reward_num_gpus_per_node 8 \
    --ref_num_nodes 1 --ref_num_gpus_per_node 8 \
    `# --- vLLM rollout engines (fast generation) ---` \
    --vllm_num_engines 4 --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.85 \
    `# --- memory and precision ---` \
    --zero_stage 3 --bf16 --gradient_checkpointing \
    --adam_offload \
    `# --- batch and sequence sizes ---` \
    --train_batch_size 256 --micro_train_batch_size 4 \
    --rollout_batch_size 1024 --micro_rollout_batch_size 16 \
    --prompt_max_len 1024 --generate_max_len 512 \
    `# --- PPO / RLHF hyperparameters ---` \
    --max_epochs 1 --num_episodes 1 \
    --ppo_epochs 4 --init_kl_coef 0.02 \
    --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
    `# --- checkpointing for fault tolerance ---` \
    --save_steps 50 --ckpt_path /ckpt/rlhf-70b/resume
```

A few flags deserve a word because they encode lessons from the earlier sections. `--adam_offload` pushes the Adam optimizer state to CPU memory, which for a 70B model is the difference between fitting and OOMing on 16 GPUs — it trades a little speed (CPU↔GPU transfers each step) for the memory headroom. `--gradient_checkpointing` recomputes activations in the backward pass instead of storing them, cutting activation memory at the cost of ~30% more compute; at 70B with long sequences you almost always need it. `--vllm_gpu_memory_utilization 0.85` lets vLLM claim most of the rollout GPUs' memory for its KV cache, which is what makes generation fast — but note it is *0.85*, not the *0.6* we saw in the veRL config, precisely because OpenRLHF's separated design does not need to leave room to reshard into training on the same GPUs. That single number difference is the architectural fork made concrete.

The two learning rates differ by more than an order of magnitude — `5e-7` for the actor versus `9e-6` for the critic — which is standard RLHF practice: the policy must move slowly to stay near the reference (small LR plus the KL penalty), while the critic can learn its value estimates faster since it is just regressing returns. `--save_steps 50` checkpoints every fifty PPO steps, the fault-tolerance policy from section 9 turned into a flag.

When this runs, you watch three things in order: first, the NCCL debug log to confirm InfiniBand is actually in use; second, the per-stage timing to confirm generation dominates as expected and no node is a straggler; third, the KL divergence and reward curves to confirm the policy is improving without reward-hacking (a runaway reward with collapsing KL means the policy is exploiting the reward model, and you should raise `init_kl_coef`). Those three checks, in that order, catch the overwhelming majority of multi-node RLHF problems.

#### Worked example: designing an OpenRLHF cluster for a 70B model

Let me walk the full design from a blank sheet, the way you would size a real run, so every number in the launch script above has a derivation behind it.

**(a) GPU-count floor from memory.** Start with the raw arithmetic. A 70B model in BF16 is $70 \times 10^9 \times 4\text{ bytes} = 280\text{ GB}$ when you count the per-parameter footprint that training-state actually carries (weights plus the working copies and gradient that accompany them in the mixed-precision recipe). Across the four model roles of RLHF, the aggregate state is on the order of $280\text{ GB} \times 4 / 80\text{ GB per A100} \approx 14$ A100s as an absolute floor. Fourteen is the *minimum*, and you never run at the minimum: ZeRO-3 (FSDP) shards most efficiently when the rank count is a clean power-of-two-friendly number that fills whole nodes, so you **round up to 16** for FSDP efficiency — two full 8-GPU nodes — for the trainer alone.

**(b) Node layout.** Translate that into physical nodes. The trainer (policy with optimizer state, the most memory-hungry role) gets **2 nodes × 8 A100s = 16 GPUs** under ZeRO-3. The rollout, reward, and reference roles — generation plus two inference-only passes — are co-located onto **1 node × 8 A100s**, with the vLLM rollout engine taking the lion's share of that node's GPUs (generation is the throughput bottleneck, so it gets the headroom) and the reward and reference passes sharing the remainder since they are forward-only and cheap. That is a compact **3-node, 24-GPU** design — tighter than the 48-GPU separated-critic layout from section 1, because here the critic shares the trainer footprint and the inference roles share a node. It is the cost-conscious shape of the same architecture: fewer GPUs, less overlap headroom.

**(c) Expected throughput.** Size the rollout. A 70B policy in vLLM generates at roughly **150 tokens/sec per GPU** at a healthy batch size, so across the **8 rollout GPUs** that is about $150 \times 8 = 1{,}200$ tokens/sec aggregate. With generation as the dominant cost and the inference passes overlapped behind it, the PPO update on the 16 trainer GPUs lands the step rate at roughly **20 PPO steps per minute** for the batch and sequence lengths in the launch script.

**(d) Time for 10k steps.** At 20 steps/min, $10{,}000 / 20 = 500$ minutes $\approx$ **8.3 hours** of pure step time. Call it ~8 hours for the run, plus checkpointing overhead and the occasional resumed attempt.

**(e) Cloud cost.** Put money on it. At **\$3 per GPU-hour** for A100-80GBs and the 24-GPU design running ~8 hours: $\$3 \times 24 \text{ GPUs} \times 8 \text{ hours} = \$576$. Under \$600 for a full 10k-step 70B RLHF run on the compact layout — which is why a team locks the recipe at 7B (a sub-\$200 iteration) and only spends this once the reward model and hyperparameters are settled. Scale the layout up to the 48-GPU overlapped design from section 11 and the wall-clock shrinks but the bill climbs toward the \$6,000–7,000 figure there; the 24-GPU shape traded some speed for a far smaller invoice, which is the right trade when GPUs are scarce.

## When to use this (and when not to)

Multi-node RLHF with Ray and OpenRLHF is the right tool when you are training a genuinely large policy (≥ 30B parameters) with online PPO against a learned reward model, and you have a multi-node GPU cluster with a decent interconnect. That is a specific situation, and it is worth being honest about when something simpler wins.

If your model is 7B or smaller, do not reach for multi-node anything. A single 8-GPU node with TRL or single-node OpenRLHF will train it, and you will iterate far faster without the orchestration overhead. The whole point of going multi-node is that the model *does not fit* otherwise; if it fits, the added complexity is pure cost.

If you can get the alignment quality you need from **DPO** rather than PPO, prefer it — strongly. DPO has no reward model, no reference rollout loop, no four-model orchestration; it trains on preference pairs with a supervised-style objective. It is dramatically cheaper and simpler, and for many alignment tasks it matches PPO-RLHF. Only reach for online PPO-RLHF when you have evidence that online generation against a reward model beats offline preference learning for your task — for example, when the reward signal is something DPO's pairwise formulation cannot capture, or when you want the policy to explore beyond the preference dataset.

Within multi-node PPO-RLHF, choose the framework by the decision tree below. If you are GPU-constrained or iterating heavily on the algorithm itself, veRL's single-controller HybridEngine is more pleasant and more memory-efficient. If you are GPU-rich and want the fastest possible iteration on a fixed recipe, OpenRLHF's separated design with overlapped rollout-and-train wins. If you are training 175B+ on an NVLink-rich cluster, NeMo-RLHF's 3D parallelism is the only thing that scales. And if you are still learning RLHF or prototyping at 7B, stay on TRL until the recipe is locked — then graduate to multi-node.

![A decision tree for choosing a multi-node RLHF framework based on model size, parallelism needs, and GPU sharing](/imgs/blogs/multi-node-rl-training-ray-openrlhf-8.png)

Do not go multi-node to feel sophisticated. Every node you add is more network to misconfigure, more places for a straggler to hide, and more dollars per hour. Add nodes only when the memory arithmetic of section 1 forces you to, and then lay them out so the network does as little work as possible.

## Case studies

**InstructGPT (Ouyang et al., 2022).** The work that put RLHF on the map trained GPT-3-scale policies (up to 175B) with PPO against a reward model learned from human comparisons. It was done inside OpenAI on proprietary infrastructure, and the paper is light on systems detail — but it established the four-model recipe (policy, reward model, reference for KL, and a value function) that every open framework since has reimplemented. The headline result was that a 1.3B InstructGPT model was preferred by human labelers over the 175B GPT-3, demonstrating that alignment via RLHF could beat raw scale on helpfulness. That result is what made everyone want to run RLHF, and the systems frameworks in this post are the open-source answer to "how."

**DeepSpeed-Chat (Microsoft, 2023).** The first widely-used open end-to-end RLHF pipeline, built on DeepSpeed's ZeRO. It proved you could run the full three-stage pipeline (SFT → reward modeling → PPO) at billions of parameters on commodity clusters, and it introduced the "Hybrid Engine" idea of switching the same GPUs between training and inference modes — a concept veRL later refined. Its limitation, in retrospect, was the absence of a vLLM-class generation engine, so the rollout phase was slower than it needed to be; this is exactly the gap OpenRLHF closed.

**OpenRLHF and the vLLM integration (2023–2024).** OpenRLHF's contribution was to build the RLHF loop natively on Ray with vLLM as the generation engine, which directly attacked the dominant cost (rollout) identified by profiling. Published results and the project's own benchmarks report substantial throughput improvements over DeepSpeed-Chat at 7B–70B scale, driven by the combination of fast generation and the overlapped, separated actor architecture. It became the de facto reference implementation for "RLHF that scales to 70B on a few nodes."

**veRL / HybridFlow (ByteDance, 2024).** veRL formalized the single-controller, programmable-dataflow approach and the HybridEngine that time-shares GPUs between generation and training. Its reported advantage is GPU efficiency at scale and ease of algorithmic experimentation — you can express a new RL algorithm as a short driver program. It has become popular for research RLHF and for reasoning-model training (GRPO-style runs), where iterating on the algorithm matters as much as raw throughput.

## Key takeaways

- **The 7B cliff is real.** Single-node RLHF works up to ~7B; past that the four models stop fitting and you are forced into a multi-node, multi-actor architecture. The forced restructuring is also an opportunity to parallelize.
- **RLHF is four models, not one.** Policy, critic, reward, reference. The policy needs full training state (terabyte-scale at 70B with optimizer state); the reward and reference are inference-only. Count GPUs by adding up these footprints — expect 32–48 GPUs at 70B.
- **Ray's actor model is the right design choice** because RLHF is heterogeneous: generation, scoring, reference, and training want different parallelism, batch sizes, and GPU counts. Express each as an actor group; use placement groups to co-locate what communicates and spread what contends.
- **Generation dominates wall-clock time**, which is why OpenRLHF spends its cleverness on vLLM and on overlapping rollout with training. Profile the rollout-to-train ratio before optimizing anything else.
- **The central design fork is separate-vs-shared GPUs.** OpenRLHF separates them to overlap stages (more GPUs, faster iteration); veRL's HybridEngine shares them to save memory (fewer GPUs, sequential phases). Benchmark both on your workload.
- **NCCL configuration silently makes or breaks throughput.** Confirm InfiniBand is actually used (`NCCL_DEBUG=INFO`), set `NCCL_SOCKET_IFNAME` to the fast interface, and never accept an unintended TCP fallback. This is the most common cause of "mysteriously slow" runs.
- **Checkpoint for failure, not for ceremony.** At dozens of nodes for many hours, failures are expected. Checkpoint model and optimizer state on a cadence matched to your failure rate, configure Ray actor restarts, and treat the PPO step as the atomic unit so lost rollouts are simply re-generated.
- **The KL penalty is the theoretical guardrail.** It bounds how far the policy drifts from the reference, keeping it in the region where the learned reward model is reliable. A runaway reward with collapsing KL means reward hacking — raise the KL coefficient.
- **A 70B RLHF run costs thousands of dollars** (≈ \$6,000–7,000 at on-demand A100 rates for our example layout), roughly 10× a 13B run and far more per token than SFT. Iterate the recipe at 7B; consider DPO before committing to PPO-RLHF.

## Further reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the canonical RLHF recipe and the four-model structure every framework reimplements.
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019) — the earlier work that introduced the reward-model-plus-KL-penalty formulation for language RLHF.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — the PPO clipped surrogate objective at the heart of the trainer.
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023) — the simpler, reward-model-free alternative to consider before committing to multi-node PPO-RLHF.
- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (veRL, 2024) — the single-controller design and HybridEngine described in section 6.
- The OpenRLHF project documentation and the Ray distributed-computing documentation — the practical references for the actor, placement-group, and `train_ppo_ray` APIs used throughout.
- Within this series, see the unified map of RL methods (`reinforcement-learning-a-unified-map`) for where RLHF sits among algorithms, and the capstone (`the-reinforcement-learning-playbook`) for the decision framework across all of RL. For the supervised and preference-learning context that precedes RLHF, see the training-techniques posts on fine-tuning and DPO.

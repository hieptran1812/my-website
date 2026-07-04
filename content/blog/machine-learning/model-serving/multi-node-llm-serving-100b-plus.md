---
title: "Multi-node LLM serving for 100B+ models: crossing the node boundary in production"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "The operational reality of running a 405B or 671B model across multiple GPU nodes: the memory math that forces the node boundary, why the interconnect hierarchy dictates where each parallelism axis lives, and how to launch, monitor, and recover a Ray-backed multi-node vLLM engine."
tags:
  [
    "model-serving",
    "inference",
    "multi-node",
    "llm-serving",
    "tensor-parallelism",
    "pipeline-parallelism",
    "expert-parallelism",
    "nccl",
    "infiniband",
    "vllm",
    "ray",
    "kubernetes",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/multi-node-llm-serving-100b-plus-1.webp"
---

The page came in at 4:12 AM: "Llama-405B deploy stuck — pods `CrashLoopBackOff`, 0 tokens served." The team had just been handed the full-precision BF16 checkpoint of Meta's 405-billion-parameter model and told to put it behind the same API that had happily served the 70B model for months. They did what worked before — one 8-GPU node, `tensor_parallel_size=8`, launch — and the process died during weight loading with `CUDA out of memory`. Someone bumped `gpu_memory_utilization` to `0.98`. It died faster. Someone else tried `--cpu-offload-gb 200`, and the thing limped to life serving four tokens per second, which was worse than useless.

The problem was not a misconfigured flag. The problem was arithmetic. Llama-3.1-405B in BF16 is 810 GB of weights. A single 8×H100-80GB node has 640 GB of HBM. There is no flag that makes 810 fit inside 640. The model had to span at least two nodes — and the moment you cross the boundary between two physical servers, you leave the world of a single NVLink fabric and enter a world where your GPUs talk to each other over a network that is roughly eighteen times slower and dramatically higher latency. Every design decision from that point on is dominated by one question: which data is allowed to cross that boundary, and how often.

![Two-node deployment showing eight GPUs per node with tensor-parallel shards inside each node and a pipeline stage boundary crossing between nodes](/imgs/blogs/multi-node-llm-serving-100b-plus-1.webp)

This post is about the operational reality of serving a 100B-to-671B model across multiple nodes: the memory math that forces you across the node boundary in the first place, the interconnect hierarchy that dictates which parallelism axis lives where, how to actually launch and bootstrap a multi-node engine with vLLM and Ray, how to run it on Kubernetes, and what happens — and what you do — when a node dies at 3 AM with live traffic on it. It is H3 in the Model Deployment and Serving series. If you have not yet internalized the SLO triangle — **latency ↔ throughput ↔ cost** — start with [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving). This post assumes you know what tensor, pipeline, and expert parallelism *are*; if you want that grounding first, read the sibling post on [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving). Here we take those primitives as given and ask the harder question: how do you compose them across a *cluster* without the network eating your latency budget?

By the end you will be able to compute whether a model needs one node or four; explain from first principles why tensor parallelism must never cross the node boundary while pipeline and expert parallelism can; write the Ray bootstrap and the Kubernetes `LeaderWorkerSet` that launches a 16-GPU engine as one logical device; and write the node-failure runbook that turns a 3 AM page into a five-minute recovery instead of a two-hour outage.

## 1. When you actually cross the node boundary: the memory math

Every multi-node decision starts with a single inequality. A serving deployment must hold four things in GPU memory simultaneously, and their sum must be less than the aggregate HBM you have allocated:

$$M_\text{total} = M_\text{weights} + M_\text{KV} + M_\text{act} + M_\text{overhead}$$

Let us make each term concrete, because the whole discipline of multi-node serving is downstream of getting this arithmetic right.

**Weights.** The weight memory is the parameter count times the bytes per parameter: $M_\text{weights} = P \cdot b_w$, where $b_w$ is 2 for BF16/FP16, 1 for FP8, and 0.5 for INT4. This term is fixed the moment you pick a model and a precision. It does not shrink under load; it is the floor.

**KV cache.** Every token that is "in flight" — sitting in an active request's context — occupies a slice of key/value cache. Per token, the KV footprint is

$$m_\text{kv} = 2 \cdot L \cdot h_\text{kv} \cdot d_\text{head} \cdot b_\text{kv}$$

where the leading 2 is for the separate K and V tensors, $L$ is the number of layers, $h_\text{kv}$ is the number of KV heads (grouped-query attention shares KV heads across query heads, which is why this is $h_\text{kv}$ and not the full head count), $d_\text{head}$ is the per-head dimension, and $b_\text{kv}$ is the KV precision in bytes. Total KV memory is $m_\text{kv}$ times the number of concurrent in-flight tokens, which is your effective concurrency knob.

**Activations** are the transient tensors of the current forward pass — roughly $\text{batch} \cdot \text{seq} \cdot h \cdot b$ during prefill, and a much smaller $\text{batch} \cdot h \cdot b$ during a single decode step. In decode-dominated serving this term is small relative to KV, but it spikes during long-context prefill.

**Overhead** is the CUDA context, the NCCL communication buffers, the framework's allocator fragmentation, and CUDA-graph capture buffers — realistically a few GB per GPU that you never get back.

You cross the node boundary the moment $M_\text{total}$ exceeds a single node's HBM — or, more precisely and more commonly, the moment $M_\text{weights}$ alone leaves no useful room for KV cache. A model that technically loads but can only hold a batch of one is not a serving deployment; it is a very expensive demo.

![Before and after comparison of loading Llama-405B on one node versus two, showing the 170 GB shortfall on a single node and a fitting two-node configuration](/imgs/blogs/multi-node-llm-serving-100b-plus-3.webp)

The figure above makes the failure mode above concrete: on one 8×H100 node you have ${8 \times 80 = 640}$ GB of HBM, and 810 GB of BF16 weights simply do not fit — you are 170 GB short before a single request arrives. Two nodes give you 1280 GB, which holds the weights and leaves roughly 400 GB for KV cache and overhead. That headroom is the whole point of the second node.

#### Worked example: does Llama-3.1-405B need multiple nodes?

Llama-3.1-405B has $P = 405 \times 10^9$ parameters, 126 layers, hidden size 16384, 128 attention heads, 8 KV heads (GQA), and head dimension 128.

- **BF16 weights**: ${405 \times 10^9 \times 2 = 810}$ GB. This exceeds 640 GB. **One node is impossible.** You need at least two.
- **FP8 weights**: ${405 \times 10^9 \times 1 = 405}$ GB. This fits on one node, leaving $640 - 405 - 40 \approx 195$ GB for KV.
- **KV per token (BF16)**: ${2 \times 126 \times 8 \times 128 \times 2 = 516{,}096}$ bytes $\approx 0.49$ MB. So 195 GB of KV headroom holds roughly $195\,\text{GB} / 0.49\,\text{MB} \approx 398{,}000$ tokens in flight — plenty for high concurrency.

The conclusion is sharp: **quantize before you scale out.** Llama-405B in FP8 is a *single-node* deployment on 8×H100 and needs no inter-node network at all. It is only the full-precision BF16 checkpoint that forces you across the boundary. If your accuracy budget tolerates FP8 — and for most serving workloads it does, see [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) — you should reach for it before you reach for a second node, because a single node is strictly simpler, cheaper, and lower-latency than any multi-node configuration.

#### Worked example: DeepSeek-V3 (671B MoE) crosses the boundary even in FP8

DeepSeek-V3 is a 671-billion-parameter mixture-of-experts model with 37B activated parameters per token, 61 layers, 256 routed experts plus one shared expert per MoE layer, and Multi-head Latent Attention (MLA). DeepSeek released it natively in FP8.

- **FP8 weights**: ${671 \times 10^9 \times 1 = 671}$ GB. This exceeds 640 GB. **Even in FP8, one node cannot hold the weights.** You need two nodes minimum just to load the model, and in practice DeepSeek serves it across dozens of nodes.
- **KV per token (MLA)**: MLA compresses each layer's KV into a shared latent vector of dimension 512 plus a 64-dimensional decoupled RoPE key. Per token that is roughly ${61 \times (512 + 64) \times 2 \approx 70}$ KB — around seven times smaller than the equivalent multi-head attention cache. This is why MLA matters for long-context serving: it keeps KV from becoming the binding constraint even at 128K context.

The contrast between the two examples is the whole lesson of this section. Llama-405B crosses the boundary only because of full-precision *weight* size — a dense model whose parameters do not fit. DeepSeek-V3 crosses it because 671 GB of FP8 weights physically cannot sit on 640 GB of HBM, and then goes far beyond two nodes not for memory reasons but to spread its 256 experts across enough GPUs to serve them with high throughput. Dense models cross for memory; large MoE models cross for both memory and expert-parallel throughput. Keep that distinction; it determines everything downstream.

#### The KV-headroom floor: why "it loads" is not "it serves"

The inequality $M_\text{total} < \text{HBM}$ has a subtle trap. It is satisfiable in a way that is technically true and operationally worthless: a configuration where the weights fit but leave so little room for KV cache that the engine can only hold a single request in flight. A serving deployment is not defined by "the weights loaded"; it is defined by "the weights loaded *and* there is enough KV headroom to run the batch size your throughput target requires." That second clause is the KV-headroom floor, and it moves the node boundary earlier than the naive weight calculation suggests.

Make it concrete. Suppose your throughput target requires a steady-state batch of 256 concurrent requests, each with an average context of 8K tokens — a modest assistant workload. That is ${256 \times 8192 \approx 2.1}$ million tokens of KV that must be resident at once. For Llama-405B's ${0.49}$ MB/token KV footprint, that is ${2.1 \times 10^6 \times 0.49\,\text{MB} \approx 1.0}$ TB of KV cache alone — more than the weights. The real memory requirement for *serving* this workload is not 810 GB (weights) but closer to ${810 + 1000 = 1810}$ GB, which needs three nodes' worth of HBM (1920 GB), not two. The weight-only calculation says two nodes; the serving calculation says three. This is why the honest form of the node-count formula carries an explicit KV term:

$$N_\text{nodes} = \left\lceil \frac{M_\text{weights} + M_\text{KV,target} + M_\text{overhead}}{\text{HBM}_\text{node}} \right\rceil$$

where $M_\text{KV,target}$ is the KV budget implied by the batch size and context length you actually intend to serve, not the minimum that lets the process start. Skipping the KV term is the single most common capacity-planning error in multi-node serving: teams provision for the weights, watch the engine boot cleanly, then discover under load that it thrashes at a batch of eight because there is no room for more. Size for the KV you need, then round up to whole nodes.

#### Worked example: node count across the model zoo

Applying the formula across a range of models makes the boundary vivid. The table below assumes 8×H100-80GB nodes (640 GB/node), a fixed 40 GB/node overhead reserve, and a KV target sized for a batch of 256 at 8K context. "Weights" is the dominant term; "KV target" is the headroom floor; "Nodes" rounds the sum up to whole 640 GB nodes.

| Model | Params | Precision | Weights | KV target (B=256, 8K) | Nodes needed | Cross boundary? |
|---|---|---|---|---|---|---|
| Llama-3.1-70B | 70B | BF16 | 140 GB | ~0.14 TB | 1 | No |
| Llama-3.1-70B | 70B | FP8 | 70 GB | ~0.14 TB | 1 | No |
| Qwen2.5-72B | 72B | BF16 | 144 GB | ~0.30 TB | 1 | No (tight) |
| Mixtral-8×22B | 141B | BF16 | 282 GB | ~0.10 TB | 1 | No (tight) |
| Llama-3.1-405B | 405B | FP8 | 405 GB | ~0.50 TB | 2 | Yes |
| Llama-3.1-405B | 405B | BF16 | 810 GB | ~1.0 TB | 3 | Yes |
| DeepSeek-V3 | 671B (MoE) | FP8 | 671 GB | ~0.15 TB (MLA) | 2+ | Yes |

Two patterns jump out. First, a 70B dense model is comfortably single-node in either precision — it never appears in this post's problem space. Second, MLA changes the arithmetic entirely for DeepSeek-V3: its KV target is a *fraction* of a dense model's despite being the largest model in the table, which is why its node count is driven almost entirely by weights and expert spread rather than KV. The lesson to carry forward is that the boundary is set by whichever term — weights or KV — blows past the node's HBM first, and quantization and attention architecture both move that boundary.

#### Worked example: the per-GPU memory ledger for TP8 × PP2 405B

The node-count formula tells you how many nodes; the per-GPU ledger tells you whether the configuration actually breathes once you land on them. Take Llama-3.1-405B in BF16 on two 8×H100 nodes, laid out as TP8 × PP2, and account for what sits on a single one of the sixteen 80 GB GPUs.

- **Weights per GPU.** Pipeline parallelism of 2 splits the 126 layers into two stages of 63 layers each; tensor parallelism of 8 then shards each stage's weights across the 8 GPUs in its node. So each GPU holds ${810\,\text{GB} / 16 = 50.6}$ GB of weights. That leaves ${80 - 50.6 = 29.4}$ GB per GPU before overhead.
- **Overhead per GPU.** CUDA context, NCCL buffers (the inter-node PP channels and intra-node TP rings each reserve registered buffers), the allocator's fragmentation reserve, and CUDA-graph capture pools consume 4 to 6 GB. Call it 5 GB. Now ${29.4 - 5 = 24.4}$ GB per GPU remains for KV cache.
- **KV per GPU.** vLLM shards the KV cache across the tensor-parallel ranks along the KV-head dimension, so each of the 8 GPUs in a stage holds ${1/8}$ of that stage's KV. With 24.4 GB free per GPU and 8 GPUs per stage, one pipeline stage can hold roughly ${24.4 \times 8 = 195}$ GB of KV. At Llama-405B's ${0.49}$ MB/token that is about ${195\,\text{GB} / 0.49\,\text{MB} \approx 398{,}000}$ resident tokens per stage — comfortably a batch of 256 requests at 8K context (${\approx 2.1}$ M tokens would need more, which is exactly why the earlier KV-headroom calculation pushed BF16 405B toward three nodes for that specific workload).

The ledger surfaces the constraint the aggregate calculation hides: it is not the *sum* of HBM across sixteen GPUs that must exceed the weights, it is that *every individual GPU* must have room for its weight shard plus its KV shard plus overhead. A layout that balances weights evenly but skews KV — because one pipeline stage carries more concurrent tokens than the other — can OOM a single GPU while fifteen others sit half-empty. Always reduce the memory question to a single-GPU ledger before you trust that a configuration fits.

## 2. The interconnect hierarchy and why it dominates

Once you have accepted that your model spans two or more nodes, the single most important fact about your deployment is no longer the GPU — it is the wire between the GPUs. Modern accelerators are so fast that inference is frequently *communication-bound*, not compute-bound, and the communication happens over a strict hierarchy of links whose bandwidths differ by orders of magnitude.

![Layered stack of interconnect tiers from on-GPU HBM through NVLink, the node boundary, InfiniBand, RoCE, and TCP Ethernet, showing bandwidth dropping at each level](/imgs/blogs/multi-node-llm-serving-100b-plus-2.webp)

The stack above ranks the tiers by bandwidth, and the gap that matters is the one marked "node boundary." Walk down it:

- **On-GPU HBM3** delivers about 3350 GB/s per H100. This is the memory the GPU reads its own weights and KV cache from. It is the reason LLM decode is memory-bandwidth-bound in the first place: every token must stream the full set of active weights out of HBM.
- **NVLink 4** connects the 8 GPUs *inside* a node through NVSwitch at roughly 900 GB/s per GPU, all-to-all. This is a fabric, not a bus: any GPU can reach any other GPU in the node at full bandwidth simultaneously. This is where fast collectives live.
- **The node boundary** is the cliff. The instant a transfer must leave the chassis, it drops from NVLink's 900 GB/s to the fastest available network link.
- **InfiniBand NDR** runs at 400 Gb/s per port, which is 50 GB/s. A well-provisioned H100 node carries one 400 Gb/s NIC per GPU (eight ConnectX-7 adapters plus separate storage NICs), giving 400 GB/s of aggregate compute fabric per node but only 50 GB/s per GPU across the boundary. That is roughly an 18× drop from NVLink.
- **RoCE v2** (RDMA over Converged Ethernet) reaches similar raw rates — 200 to 400 Gb/s — but is more sensitive to configuration: it needs priority flow control and explicit congestion notification tuned correctly, or it degrades badly under load.
- **TCP Ethernet** at 25 to 100 Gb/s (3 to 12 GB/s) with a kernel network stack sits at the bottom. It is fine for control-plane chatter and health checks and completely unfit for per-token collectives.

Bandwidth is only half the story; latency is the other half, and for the small messages that dominate decode it is often the more important half. A ring collective's time is approximately

$$t \approx \underbrace{\frac{2(N-1)}{N}\cdot\frac{M}{\text{BW}}}_{\text{bandwidth term}} + \underbrace{2(N-1)\,\alpha}_{\text{latency term}}$$

where $M$ is the message size, $N$ the number of participants, BW the link bandwidth, and $\alpha$ the per-hop latency. NVLink's $\alpha$ is well under a microsecond. InfiniBand's is 2 to 3 microseconds with GPUDirect RDMA; RoCE is 3 to 5; and a TCP round trip through the kernel is 20 to 50 microseconds. For the tiny messages of a single decode step, the latency term dominates entirely, and the 10-to-50× latency penalty of leaving the node matters more than the 18× bandwidth penalty.

This is the physical fact that governs the rest of the post. NVLink is fast and low-latency; anything across the node boundary is neither. So the design rule writes itself: **the traffic that is frequent, small, latency-critical, and on the token's critical path must stay on NVLink, inside a node. Only traffic that is infrequent, large, or overlappable is allowed to cross the boundary.** Section 3 turns that rule into a concrete assignment of parallelism axes to links.

#### Worked example: comms volume per token, and what each link can absorb

The design rule is not a preference; it falls out of dividing communication *volume* by link *bandwidth* and seeing which combinations blow the per-token time budget. Take Llama-405B in BF16 — hidden size 16384, 126 layers — at a decode batch of $B = 64$, and compute how many bytes each axis must move per token and how long that takes on each link tier.

- **Tensor parallelism.** Two `AllReduce` per layer, each on a $[B, h]$ tensor of ${64 \times 16384 \times 2 = 2.1}$ MB. A ring `AllReduce` moves ${2(N-1)/N \approx 1.75}$× the tensor per rank, so ${\approx 3.7}$ MB per GPU per op, times ${2 \times 126 = 252}$ ops per token gives roughly ${930}$ MB of per-GPU traffic per token, every token, on the critical path. On NVLink at 900 GB/s that is about 1.0 ms of pure comms (before latency terms); on InfiniBand at 50 GB/s it is 18.6 ms — larger than an entire 20 ms TPOT budget by itself. That single ratio is why TP never crosses the boundary.
- **Pipeline parallelism.** One point-to-point activation send per stage boundary per microbatch. The activation tensor for a decode step is $[B, h] = {64 \times 16384 \times 2 \approx 2.1}$ MB, sent once across the boundary per token (for a 2-stage pipeline). On InfiniBand at 50 GB/s that is ${\approx 42}$ µs — and because the send overlaps with the next microbatch's compute, most of it never appears on the critical path. Two orders of magnitude less inter-node traffic than TP, and hideable to boot.
- **Expert parallelism.** Two all-to-all per MoE layer, each moving the routed tokens' hidden vectors. For a batch of 64 tokens each routed to $k$ experts, the dispatched volume per MoE layer is ${\approx 64 \times k \times h \times b}$; at $k = 8$, $h = 7168$ (DeepSeek-V3's dimension), that is ${\approx 29}$ MB dispatched plus a symmetric combine. Substantial, but infrequent (once per MoE layer, not twice per transformer layer) and — critically — overlappable with attention compute on the next micro-step.

Lay the three side by side and the assignment is forced: TP moves ~930 MB/token/GPU and cannot be hidden, so it must sit on the 900 GB/s fabric; PP moves ~2 MB/token and hides, so 50 GB/s is plenty; EP moves tens of MB per MoE layer and hides, so it too tolerates the inter-node link. You do not choose where to put each axis by taste — you divide its volume by each link's bandwidth and read the answer off the clock.

## 3. Placing parallelism on the right link

There are four parallelism axes you can compose to spread a model across a cluster, and each one generates a completely different communication pattern. The art of multi-node serving is matching each axis to a link tier whose bandwidth and latency it can tolerate.

![Matrix mapping NVLink, InfiniBand, RoCE, and TCP Ethernet against bandwidth, latency, best-fit parallelism, and relative cost](/imgs/blogs/multi-node-llm-serving-100b-plus-4.webp)

The matrix above is the placement cheat sheet; the rest of this section derives it. Let us take the axes in order of how much communication they demand.

**Tensor parallelism (TP)** shards every weight matrix across GPUs, so every layer must synchronize partial results. In the Megatron layout, each transformer layer performs two `AllReduce` operations in the forward pass — one after the attention output projection, one after the FFN's second matrix multiply. During decode with batch $B$ and hidden size $h$, the all-reduced tensor is shaped $[B, h]$ in the compute dtype, so the message is $M = B \cdot h \cdot b$ bytes. Critically, these collectives happen ${2 \cdot L}$ times per token — twice per layer, every layer, every single token — and they sit squarely on the critical path. The GPU cannot compute layer $\ell+1$ until the `AllReduce` at the end of layer $\ell$ completes. There is nothing to overlap them with. This is the axis that most demands NVLink.

**Pipeline parallelism (PP)** splits the model's layers into contiguous stages, one stage per node. Communication happens only at stage boundaries: stage $s$ sends its output activations to stage $s+1$ as a point-to-point transfer. For a $P$-stage pipeline there are only $P-1$ such transfers per microbatch, the message is a single activation tensor, and — this is the key — the send can be overlapped with the compute of the next microbatch. PP trades a small amount of pipeline "bubble" (idle time while the pipeline fills and drains) for dramatically lower communication frequency. It tolerates the inter-node link comfortably.

**Expert parallelism (EP)** places different MoE experts on different GPUs and routes each token to its top-$k$ experts with an all-to-all dispatch, then an all-to-all combine. This happens twice per MoE layer. The volume is moderate — proportional to tokens times hidden size — and, crucially, sophisticated engines overlap the dispatch/combine with the attention computation of the next micro-step. EP is the axis you *want* to spread across many nodes, because that is how you get enough GPUs to hold hundreds of experts.

**Data parallelism (DP)** runs independent replicas of the whole model. During inference there is essentially no cross-replica communication at all; each replica serves its own requests. DP is trivially cross-node and cross-rack; it is how you scale throughput once a single replica fits.

#### Worked example: what tensor parallelism costs on the wrong link

Take Llama-70B: hidden size 8192, 80 layers, TP=8, and a decode batch of $B=32$.

- Message per `AllReduce`: ${32 \times 8192 \times 2 = 524{,}288}$ bytes = 512 KB.
- Ring `AllReduce` moves ${2 \times \tfrac{7}{8} \times 512\,\text{KB} \approx 896}$ KB per GPU per op.
- **On NVLink (900 GB/s, sub-µs latency):** the bandwidth term is $896\,\text{KB} / 900\,\text{GB/s} \approx 1\,\mu s$; with latency the op lands around 2 to 4 µs. Across ${2 \times 80 = 160}$ ops per token, TP communication costs roughly 0.3 to 0.6 ms per token — a few percent of a 20 ms TPOT budget.
- **On InfiniBand (50 GB/s, ~2-3 µs latency):** the bandwidth term alone is $896\,\text{KB} / 50\,\text{GB/s} \approx 18\,\mu s$, and with the per-hop latency each op is 20 to 25 µs. Across 160 ops that is **3.2 to 4 ms per token of pure, non-overlappable TP communication** — 15 to 20% of the same budget, and it gets strictly worse as batch grows.

That is the entire argument in one number. Putting TP across the node boundary does not fail loudly; it quietly triples your per-token latency and caps your throughput, because those 160 collectives per token cannot be hidden. Every production guide — vLLM's included — states the rule bluntly: **keep `tensor_parallel_size` less than or equal to the number of GPUs in one node.** For an 8-GPU node, TP is at most 8. If you need more than 8-way sharding, you add pipeline or expert stages across nodes, not more tensor-parallel ranks.

That is exactly the layout of the intro figure: TP=8 stays inside each node on NVLink, and the only edge that crosses the InfiniBand boundary is the pipeline stage boundary — a single point-to-point activation send per microbatch that overlaps with compute.

Here is the placement rule as a table you can keep next to your deployment config:

| Axis | Comm pattern | Frequency | Overlappable? | On critical path? | Put it on |
|---|---|---|---|---|---|
| Tensor (TP) | `AllReduce` | ${2 \times L}$ per token | No | Yes | NVLink (intra-node only) |
| Pipeline (PP) | point-to-point | $P-1$ per microbatch | Yes | Partially | InfiniBand / RoCE (inter-node) |
| Expert (EP) | all-to-all | 2 per MoE layer | Yes (with effort) | Partially | InfiniBand (inter-node) |
| Data (DP) | none (inference) | — | — | No | Any link, incl. Ethernet |

The composition rule for a cluster follows directly: fill the fast axis first. Set TP equal to the GPUs per node (usually 8), then use PP or EP to cross nodes, then DP to add replicas for throughput. A 405B BF16 deployment on two nodes is TP8 × PP2. On four nodes it is TP8 × PP4. A 671B MoE across sixteen nodes is TP8 (or MLA-friendly attention DP) with EP spanning the rest. You never write TP16-across-two-nodes unless you enjoy paying triple for latency. For the deeper theory of how these axes compose onto a device mesh — including why the same rule governs *training* — see [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism).

#### Worked example: the pipeline bubble and why decode hides it

Pipeline parallelism's one cost is the bubble — the idle time while the pipeline fills and drains. For a $P$-stage pipeline processing $m$ microbatches, the classic bubble fraction is ${(P-1)/(m + P - 1)}$. With PP2 and a single microbatch, that is ${1/2}$ — half the pipeline sits idle, which sounds catastrophic. It is not, for one serving-specific reason: continuous batching keeps a deep, steadily-refilled queue of in-flight requests, so the "microbatch" count $m$ that the pipeline sees is large. At $m = 32$ in-flight microbatches with PP2, the bubble fraction is ${1/33 \approx 3\%}$. The idle time that dominates a *training* pipeline with a handful of gradient-accumulation steps nearly vanishes in a *serving* pipeline fed by hundreds of concurrent decodes. This is why PP, which is often awkward in training, is comparatively benign for inference: the workload naturally supplies the many microbatches that amortize the bubble away.

#### Worked example: how EP spread sets the node count for a 671B MoE

DeepSeek-V3 has 256 routed experts per MoE layer. Expert parallelism assigns experts to GPUs, so the question "how many nodes" for the MoE portion is really "how many GPUs do I want each holding a slice of the experts." Put all 256 experts on 8 GPUs (EP8, one node) and each GPU stores 32 experts — the weights are dense on each card and the per-GPU MoE memory is high, but there is no inter-node all-to-all at all. Spread the same 256 experts across 320 GPUs (EP320, the DeepSeek decode unit) and each GPU holds fewer than one expert's worth on average, the per-GPU MoE footprint collapses, and you free enormous KV headroom — at the cost of an inter-node all-to-all on every MoE layer. The trade is memory-per-GPU against inter-node comms, and it is the reason a MoE model's node count is a *throughput and memory* decision rather than a pure weight-fit decision. Dense models cross the boundary once, for weights; MoE models keep crossing it, deliberately, to buy expert spread.

#### The composition recipe

Collapsing the section into a procedure you can run on any new model: (1) compute $M_\text{weights}$; if it fits one node with KV headroom, stop — you are single-node, and this whole post is moot. (2) If not, set $\text{TP} = $ GPUs-per-node so tensor parallelism is maximal but stays on NVLink. (3) Cross the boundary with PP for a dense model (contiguous layer stages, cheap point-to-point) or EP for a MoE model (expert spread, overlappable all-to-all), choosing the stage/expert count so that per-GPU weights plus KV fit the single-GPU ledger from Section 1. (4) Add DP replicas to reach your throughput target and to satisfy the N+1 rule for fault tolerance. Every production multi-node config is some walk down that four-step ladder; the failures come from skipping a rung — most commonly step 2, where a well-meaning `TP16` quietly puts tensor parallelism on the wire.

## 4. NCCL, rail-optimized fabrics, and GPUDirect RDMA

The placement rules of Section 3 are only useful if the software actually routes each collective over the link you intend. That job belongs to NCCL — NVIDIA's Collective Communications Library — which every serious LLM engine (vLLM, TGI, TensorRT-LLM, SGLang) uses for its GPU-to-GPU transfers. NCCL discovers the available transports at startup, builds rings and trees over them, and picks NVLink for intra-node peers and InfiniBand verbs (or RoCE, or TCP sockets, in descending order of preference) for inter-node peers. When it picks wrong — or when the fabric is misconfigured — your carefully planned TP8 × PP2 layout can silently fall back to TCP sockets and run 30× slower than it should.

Two hardware facts make inter-node NCCL fast when it is set up correctly. The first is **GPUDirect RDMA**: the network adapter DMAs data directly into and out of GPU HBM over a PCIe peer-to-peer path, bypassing the round trip through host memory that would otherwise double the latency and burn CPU cycles. The second is the **rail-optimized network topology**. In a rail-optimized cluster, GPU $i$ on every node connects to the same dedicated leaf switch — "rail $i$." An 8-GPU node therefore touches eight rails. NCCL is rail-aware: it keeps a collective on a single rail, so GPU 3 on node A talks to GPU 3 on node B through one switch hop, never crossing the congested spine and never contending with the other rails' traffic. Getting NCCL to see the rails correctly is most of the battle.

Here is a production-grade NCCL environment block for a rail-optimized InfiniBand cluster, with a comment on why each variable matters:

```bash
# --- NCCL tuning for a multi-node, rail-optimized InfiniBand fabric ---

# Print the ring/tree topology and the transport chosen for every peer pair.
# Read this ONCE per new cluster: confirm intra-node peers say "via NVLink"
# and inter-node peers say "via NET/IB", not "via NET/Socket" (TCP fallback).
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH

# The IB HCAs (adapters) NCCL may use for the DATA path. List them in rail
# order so NCCL pairs GPU i with HCA i and keeps each collective on-rail.
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# The interface NCCL uses ONLY for its out-of-band bootstrap/rendezvous.
# This is the CONTROL path, not the data path -- point it at your management
# NIC, never at the IB fabric, or bootstrap fights the collectives for bandwidth.
export NCCL_SOCKET_IFNAME=eth0

# Enable GPUDirect RDMA when the NIC and GPU share a PCIe switch (PIX) or are
# on the same PCIe host bridge. This is the single biggest inter-node win.
export NCCL_NET_GDR_LEVEL=PIX

# On RoCE (Ethernet RDMA) you MUST set the RoCE v2 GID index; on pure IB omit it.
# Wrong GID index is the #1 cause of "hangs at NCCL init" on RoCE clusters.
export NCCL_IB_GID_INDEX=3

# Raise the IB completion timeout so a briefly-busy fabric does not trip a
# spurious failure under peak load. Units are IB timeout exponents (~4.1 ms * 2^n).
export NCCL_IB_TIMEOUT=22

# Prefer NVLink for all intra-node peer-to-peer transfers.
export NCCL_P2P_LEVEL=NVL

# Abort a hung collective instead of deadlocking the whole engine forever.
# Essential for fault tolerance: turns a dead-peer hang into a catchable error.
export NCCL_ASYNC_ERROR_HANDLING=1
```

Three of these deserve emphasis because they cause the most 3 AM pages. `NCCL_SOCKET_IFNAME` set to the wrong interface makes NCCL try to bootstrap over the InfiniBand fabric itself, or over a non-routable interface, and the engine hangs forever at initialization with no error. `NCCL_IB_GID_INDEX` set wrong on a RoCE cluster produces the same silent hang. And `NCCL_ASYNC_ERROR_HANDLING=1` is what converts a dead peer from an eternal deadlock into an exception your supervisor can catch and act on — without it, one dead GPU freezes the entire replica and your liveness probe never fires because the process is technically still "running."

Always run a new cluster once with `NCCL_DEBUG=INFO` and read the transport lines. If an inter-node peer reports `via NET/Socket` instead of `via NET/IB`, your RDMA path is broken and you are about to serve at TCP speed. Fix that before you benchmark anything, because every number you measure until then is a lie.

#### Diagnosing a silent TCP fallback

The failure mode that costs the most time is the one that does not crash: NCCL initializes cleanly, the engine serves tokens, everything looks healthy — and throughput is a third of what it should be because inter-node collectives silently fell back to TCP sockets. There is no error to grep for; the only evidence is a single line buried in the init log. Pull it out directly:

```bash
# Capture init logs, then confirm the inter-node transport NCCL actually chose.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET vllm serve ... 2> nccl_init.log

# Every inter-node channel should say NET/IB (InfiniBand) or NET/IBext.
# If you see NET/Socket, RDMA is not being used and you are serving at TCP speed.
grep -E "NET/(IB|Socket)" nccl_init.log | sort | uniq -c

# Confirm GPUDirect RDMA is active (the biggest inter-node win). "GDRDMA" should
# appear; its absence means NCCL is staging through host memory on every transfer.
grep -iE "GDRDMA|GPU Direct RDMA|via NET" nccl_init.log | head

# On RoCE, verify the GID index NCCL bound. A wrong index reads as a plain hang.
grep -iE "GID|RoCE" nccl_init.log | head
```

The three most common root causes, in the order they occur: `NCCL_IB_HCA` unset or listing devices that do not exist, so NCCL finds no IB adapter and drops to sockets; `NCCL_SOCKET_IFNAME` pointing at an interface that cannot route between nodes, so even the bootstrap struggles; and a container launched without `--network host` and without the IB device mounted (`--device /dev/infiniband`), so the RDMA verbs are simply invisible inside the namespace. That last one is the single most frequent cause of a Kubernetes multi-node engine that "works" at a fraction of speed: the pod cannot see the HCA, NCCL never errors, and it quietly serves over TCP. The fix is to expose the RDMA devices to the pod (via the RDMA device plugin or host networking) and re-check the log for `NET/IB`.

A few more environment variables round out a production fabric setup beyond the core block above:

```bash
# Number of NICs/rails NCCL should use per node. Match your rail count so
# collectives spread across all adapters instead of hammering one.
export NCCL_IB_QPS_PER_CONNECTION=4      # more queue pairs = more inflight RDMA

# Disable NVLS/collnet only if you must debug; leave NCCL to auto-pick normally.
# export NCCL_ALGO=Ring                   # force Ring to isolate a tree-collective bug

# Cap the socket threads NCCL spawns for the control path so bootstrap does not
# fork-bomb on a fat node. Default is fine; tune only if you see thread pressure.
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```

Set these only when the core block is verified working; the point of listing them is that when a fabric misbehaves under load — not at init, but at peak — the cause is usually queue-pair depth or thread contention, and these are the knobs. Change one at a time and re-measure; blind multi-variable tuning of NCCL is how a working fabric becomes a mysteriously broken one.

## 5. Orchestrating a multi-node engine with vLLM and Ray

vLLM turns a pile of GPUs across several nodes into a single logical engine using Ray as its distributed backend. When `tensor_parallel_size × pipeline_parallel_size` exceeds the GPUs available on one machine, vLLM switches from its single-node multiprocessing executor to the Ray executor, forms a placement group that reserves the right GPUs on the right nodes, and launches one worker process per GPU. The application code never sees sixteen GPUs; it sees one `LLM`.

![Graph showing a Ray head node forming a cluster with two worker nodes that each contribute a bundle to one placement group bound by the vLLM engine](/imgs/blogs/multi-node-llm-serving-100b-plus-5.webp)

The topology in the figure is the mental model to hold: the head node runs Ray's global control store (GCS) and the API server, both worker nodes join the cluster and contribute their 8 GPUs, a placement group reserves two 8-GPU bundles (one per node), and the vLLM engine binds to that placement group as a single TP8 × PP2 device. Here is the bootstrap, node by node.

```bash
# ============ HEAD NODE (node A, IP 10.0.0.1) ============
# Ray and vLLM must agree on the IP the workers will reach the head at.
export VLLM_HOST_IP=10.0.0.1
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# Start the Ray head. GCS listens on 6379; workers dial this address.
ray start --head --port=6379 --num-gpus=8

# ============ WORKER NODE (node B, IP 10.0.0.2) ============
export VLLM_HOST_IP=10.0.0.2
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# Join the existing cluster. --block keeps the process alive as a worker.
ray start --address="10.0.0.1:6379" --num-gpus=8 --block

# ============ BACK ON THE HEAD NODE: launch the engine ============
# 8-way TP inside each node (NVLink), 2-way PP across the two nodes (InfiniBand).
# tensor_parallel_size * pipeline_parallel_size = 16 = total GPUs in the cluster.
vllm serve meta-llama/Llama-3.1-405B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --port 8000
```

A few things are load-bearing here. `--distributed-executor-backend ray` is what tells vLLM to use the multi-node executor; the default `mp` backend is single-node only. The product `8 × 2 = 16` must equal the total GPU count Ray sees, or vLLM will refuse to start. And vLLM's placement logic is rail-aware in the sense that matters: it packs each 8-GPU tensor-parallel group onto a single node's bundle (a `STRICT_PACK` placement), so TP never straddles the boundary — the pipeline stage boundary is the only edge that crosses. You get the layout of Section 3 for free, as long as you write `TP8 × PP2` and not `TP16`.

Verify the cluster before you launch the engine. `ray status` on the head node should show two nodes and 16 GPUs. If it shows one node, your workers never joined — almost always a firewall on port 6379 or a `VLLM_HOST_IP` that resolves to a loopback address. vLLM ships an `examples/online_serving/run_cluster.sh` helper that wraps the docker-plus-Ray dance for exactly this setup; use it as a reference for the container networking flags (`--network host` and the shared-memory size are the ones people miss).

If you prefer the Python API to the CLI — for embedding the engine in a larger service — the same configuration is:

```python
from vllm import LLM, SamplingParams

# Assumes `ray start` has already formed the 2-node, 16-GPU cluster.
llm = LLM(
    model="meta-llama/Llama-3.1-405B",
    tensor_parallel_size=8,        # intra-node, over NVLink
    pipeline_parallel_size=2,      # inter-node, over InfiniBand
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.90,
    max_model_len=32768,
    enforce_eager=False,           # keep CUDA graphs; they matter for TPOT
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain multi-node inference in one paragraph."], params)
print(outputs[0].outputs[0].text)
```

A brief word on TGI and the alternatives. Hugging Face's Text Generation Inference is excellent within a node — its `--num-shard` flag does tensor parallelism across the local GPUs — but it was built single-node-first, and multi-node TGI has historically meant standing up your own cross-node coordination. For serving a model that genuinely spans nodes, the mature paths today are vLLM-with-Ray (shown here), SGLang, and NVIDIA's Dynamo/TensorRT-LLM. Pick vLLM+Ray unless you have a specific reason not to; it is the best-documented multi-node path and the one the Kubernetes tooling in Section 7 is built around.

Here is how the mature multi-node paths compare on the axes that decide the choice:

| Framework | Multi-node backend | TP×PP support | EP for MoE | K8s story | Best fit |
|---|---|---|---|---|---|
| vLLM + Ray | Ray placement groups | Yes (TP×PP) | Yes (growing) | LeaderWorkerSet, llm-d | Default; best-documented multi-node path |
| SGLang | Ray or native | Yes (TP×PP) | Yes (strong MoE) | LWS-compatible | MoE-heavy, RadixAttention prefix reuse |
| TensorRT-LLM / Dynamo | MPI + Dynamo | Yes (TP×PP) | Yes | Dynamo operator | Peak NVIDIA-hardware throughput, willing to compile engines |
| TGI | single-node-first | TP only (`--num-shard`) | Limited | Deployment | Single-node; multi-node needs custom coordination |

The table encodes the decision. If you are on the well-trodden path, vLLM+Ray with LeaderWorkerSet is the answer and the rest of this post is written for it. If your model is MoE-dominated and you want aggressive prefix reuse, SGLang is worth the evaluation. If you are all-in on NVIDIA hardware and willing to pay the engine-compilation tax for the last increment of throughput, TensorRT-LLM behind Dynamo is the ceiling. TGI stays where it is strongest — a single node — and is not the tool for a model that spans the boundary.

## 6. The cold-start sequence: what happens before the first token

A multi-node engine is not ready the instant the process starts. Between `ray start` and the first served token there is a multi-minute sequence of weight sharding, NCCL ring construction, and CUDA-graph capture — and if you do not account for it, your readiness probe marks the pod healthy too early, traffic lands on an engine that then stalls for ninety seconds, and your p99 latency graph looks like a heart attack.

![Timeline of the multi-node cold start from ray start through worker join, placement group reservation, weight loading, NCCL init, warmup, and ready-to-serve](/imgs/blogs/multi-node-llm-serving-100b-plus-6.webp)

The timeline above walks the stages, and the durations are the ones that bite:

1. **`ray start --head`** brings up the GCS in a second or two.
2. **Workers join** — each worker node dials the head and registers its 8 GPUs. Seconds, assuming the network is open.
3. **Placement group reservation** — vLLM asks Ray to reserve two 8-GPU bundles pinned per node. Near-instant if the GPUs are free; it *blocks* if another workload holds them, which is a common source of a "stuck at startup" report.
4. **Weight loading** — this is the long one. 810 GB of weights must be read from storage and sharded across 16 GPUs. Even from a fast shared filesystem this is 60 to 120 seconds; from object storage over a slow link it can be many minutes. This is why weight *loading strategy* is a first-class concern (more below).
5. **NCCL initialization** — every TP ring and PP point-to-point channel is established, buffers allocated, GPUDirect paths negotiated. 10 to 20 seconds, and the stage most likely to *hang forever* if `NCCL_SOCKET_IFNAME` or the IB GID is wrong.
6. **Warmup and CUDA-graph capture** — vLLM runs dummy forward passes to trigger kernel autotuning and captures CUDA graphs for the common batch shapes. 20 to 40 seconds, and skipping it (`enforce_eager=True`) trades this one-time cost for a permanent per-token latency penalty, which is almost never the right trade in production.
7. **Ready** — only now should traffic arrive.

The operational consequence is that your Kubernetes readiness probe must have a generous `initialDelaySeconds` (180 or more for a 405B) and, better, should hit an endpoint that actually confirms the engine can generate — vLLM's `/health` returns healthy only once the engine loop is live. Weight loading across nodes deserves special attention: the naive path has every one of the 16 workers independently pull its shard from the same storage endpoint, which either saturates the endpoint or serializes badly. The patterns that work are (a) a shared high-throughput filesystem (Lustre, GPFS, or a fast NFS) that every node mounts, (b) pre-baking the weights into the container image or a local NVMe cache so loading is a local read, or (c) sharded/streaming loaders (like the `run:ai model streamer` vLLM integrates) that overlap download with GPU placement. Whatever you choose, measure it — weight load time is usually the dominant term in your cold start and therefore in your recovery time when a node dies.

#### Worked example: what actually gates weight-load time

Weight loading is bandwidth arithmetic, and it is worth doing because the answer decides your recovery time. Llama-405B BF16 is 810 GB. The 16 workers collectively read all 810 GB, but the ceiling is set by whichever link is scarcest between the bytes and the GPUs.

- **From a shared parallel filesystem** (Lustre/GPFS) delivering, say, 20 GB/s aggregate to the reading node: ${810\,\text{GB} / 20\,\text{GB/s} \approx 40}$ s if the read parallelizes cleanly across workers, but often 60 to 120 s in practice because metadata operations and per-shard seeks serialize.
- **From cloud object storage** at a more typical 2 GB/s sustained per node: ${810\,\text{GB} / 2\,\text{GB/s} \approx 405}$ s — nearly seven minutes — and this is the number that turns a node failure into a long outage if you have no faster path.
- **From a pre-warmed local NVMe cache** at 5 to 7 GB/s per drive, read in parallel across nodes: ${810\,\text{GB}}$ split so each node reads its ${405}$ GB stage locally at ${\approx 6}$ GB/s is ${\approx 68}$ s, entirely local, with no shared endpoint to saturate.

The strategic point: your mean-time-to-recovery is dominated by whichever of these you provisioned, because a gang-restart re-runs weight loading from scratch. A team serving from object storage with a 7-minute load has a 7-minute-plus MTTR baked in that no amount of fast NCCL init can shorten. Pre-baking weights into the image or a local NVMe cache is not a micro-optimization; it is the difference between a two-minute recovery and a ten-minute one, on the exact axis your users feel during an incident. If you take one number from this section, take the load time on *your* storage path, because it is also your recovery time.

## 7. Multi-node on Kubernetes with LeaderWorkerSet

Running the Ray bootstrap by hand is fine for a bring-up; production wants it declarative, gang-scheduled, and self-healing. The Kubernetes primitive built for exactly this is the **`LeaderWorkerSet`** (LWS) — a custom resource, incubated in Kubernetes SIG-scheduling and adopted by the vLLM production stack and the `llm-d` project, that models a *group* of pods (one leader plus N workers) as a single atomic unit. One LWS group is one model replica that spans nodes; the leader runs the Ray head and the vLLM API server, the workers run Ray workers, and the whole group is scheduled, scaled, and restarted together.

Why not a plain `Deployment` or `StatefulSet`? Because those manage pods independently. A multi-node engine is the opposite of independent: if one worker pod dies, the whole engine is broken and every remaining pod must restart together to re-form the NCCL rings. A `LeaderWorkerSet` encodes that "all or nothing" semantics natively with `RecreateGroupOnPodRestart`.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llama-405b
spec:
  replicas: 2                       # two independent model replicas (DP) for HA
  leaderWorkerTemplate:
    size: 2                         # each replica = 1 leader + 1 worker = 2 nodes
    restartPolicy: RecreateGroupOnPodRestart   # gang restart: any pod death restarts the group
    leaderTemplate:
      spec:
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:latest
          env:
          - { name: NCCL_SOCKET_IFNAME, value: "eth0" }
          - { name: NCCL_IB_HCA, value: "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" }
          command: ["/bin/bash", "-c"]
          args:
          - |
            ray start --head --port=6379 --num-gpus=8 &&
            vllm serve meta-llama/Llama-3.1-405B \
              --tensor-parallel-size 8 --pipeline-parallel-size 2 \
              --distributed-executor-backend ray \
              --gpu-memory-utilization 0.90 --port 8000
          ports:
          - { containerPort: 8000 }
          resources:
            limits: { nvidia.com/gpu: "8" }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 200      # cold start is minutes; do not probe early
            periodSeconds: 10
            failureThreshold: 3
    workerTemplate:
      spec:
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:latest
          env:
          - { name: NCCL_SOCKET_IFNAME, value: "eth0" }
          command: ["/bin/bash", "-c"]
          # LWS injects LWS_LEADER_ADDRESS so the worker can find the head.
          args: ["ray start --address=$(LWS_LEADER_ADDRESS):6379 --num-gpus=8 --block"]
          resources:
            limits: { nvidia.com/gpu: "8" }
```

Read the three fields that carry the design. `size: 2` says each replica is a two-pod group (leader + one worker) and therefore spans two nodes — a TP8 × PP2 engine. `replicas: 2` says run two such groups, giving you two independent replicas behind a service for high availability and DP throughput. And `RecreateGroupOnPodRestart` is the fault-tolerance keystone: if any pod in a group dies, LWS tears down and recreates the *entire* group, because a half-alive NCCL communicator is worse than a cleanly restarted one. LWS also injects `LWS_LEADER_ADDRESS` into every worker so the Ray join address is discovered automatically, and it gang-schedules the group so you never end up with a leader running and a worker stuck `Pending` for want of a GPU. Front the two replicas with a standard `Service`, and route across them with a length- or load-aware balancer as covered in the control-plane sibling post on [LLM control planes](/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve).

## 8. Fault tolerance: when a node dies mid-serve

At scale, nodes die. A GPU throws an uncorrectable ECC error, a NIC flaps, a kernel panics, the cloud provider reclaims a spot instance. In a single-node deployment this is a bounded event: one replica goes down, the load balancer routes around it. In a multi-node deployment it is more subtle, because a single dead GPU on one node breaks an *entire replica* that spans multiple nodes — every NCCL collective the replica issues will now hang waiting for a peer that will never answer.

The good news, and it is worth stating plainly, is that **inference fault tolerance is dramatically easier than training fault tolerance.** A training job has mutable state — optimizer moments, the current weights, the data loader position — that must be checkpointed and restored consistently, or you lose hours of progress. An inference replica has *no* mutable state worth saving. The weights are read-only and reloadable from storage; the only thing lost when a replica dies is the handful of in-flight requests on it, which the client or gateway can retry against a healthy replica. Recovery is therefore "reload the weights and re-form the rings," not "restore a distributed checkpoint." That is a meaningfully simpler problem.

![Graph of node-failure recovery: a dead node is caught by both a Ray heartbeat miss and an NCCL timeout, the router fails over, then the group gang-restarts and rejoins](/imgs/blogs/multi-node-llm-serving-100b-plus-7.webp)

The recovery flow in the figure has two detection signals feeding one response, and the redundancy is deliberate. **Ray's heartbeat** notices a dead worker node within about ten seconds and marks it lost. Independently, an in-progress **NCCL collective times out** on the surviving GPUs — provided you set `NCCL_ASYNC_ERROR_HANDLING=1`, this surfaces as a catchable exception rather than an eternal hang. Whichever fires first, the response is the same: the router drains the broken replica and fails its traffic to the healthy replica; then the group gang-restarts (LWS `RecreateGroupOnPodRestart` does this), reloads the 810 GB of weights, re-initializes NCCL, warms up, and rejoins the serving pool. The whole recovery is bounded by the cold-start time of Section 6 — which is precisely why you invest in fast weight loading: it is not just a startup nicety, it is your mean-time-to-recovery.

The non-negotiable prerequisite for this to work gracefully is **N+1 replicas.** If you run a single multi-node replica, a node death is a full outage until the group restarts — minutes of downtime. If you run two or more replicas (the `replicas: 2` in the LWS above), a node death degrades capacity but never takes the service to zero, and the recovery happens behind the load balancer while the survivors carry the traffic. Multi-node serving with a single replica is a demo; multi-node serving with N+1 replicas is production.

Here is a liveness gate that a Kubernetes probe (or a sidecar) can run to detect a degraded replica early — before NCCL hangs propagate — by checking that the Ray cluster still has the full complement of GPUs and that no GPU is throwing uncorrectable ECC errors:

```python
#!/usr/bin/env python3
# multinode_health.py -- liveness gate for a Ray-backed multi-node vLLM replica.
# Exit 0 = healthy; exit 1 = degraded (fail the probe so the group is recreated).
import sys
import subprocess
import ray

EXPECTED_GPUS = 16  # TP8 x PP2 over two 8-GPU nodes

def cluster_gpu_count() -> int:
    """Live GPU count Ray currently sees. A dead node drops this below EXPECTED."""
    try:
        ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
        return int(ray.cluster_resources().get("GPU", 0))
    except Exception as exc:                       # GCS unreachable == head is gone
        print(f"ray unreachable: {exc}", file=sys.stderr)
        return -1

def has_uncorrectable_ecc() -> bool:
    """A GPU with uncorrectable ECC errors will corrupt inference silently."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=ecc.errors.uncorrected.volatile.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout
    return any(tok.strip().isdigit() and int(tok) > 0 for tok in out.split())

if __name__ == "__main__":
    gpus = cluster_gpu_count()
    if gpus < EXPECTED_GPUS:
        print(f"DEGRADED: {gpus}/{EXPECTED_GPUS} GPUs visible", file=sys.stderr)
        sys.exit(1)
    if has_uncorrectable_ecc():
        print("DEGRADED: uncorrectable ECC error on a local GPU", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {gpus}/{EXPECTED_GPUS} GPUs healthy")
    sys.exit(0)
```

The ECC check is worth calling out: an uncorrectable ECC error does not always crash the process, but it can silently corrupt the model's numerics, so you want to fail the health check and recycle the group rather than serve garbage. Catching it here — before the corruption shows up as a customer complaint about incoherent output — is the difference between a self-healing system and a debugging session.

Detection is only useful if it drives an action. On Kubernetes, the LWS `RecreateGroupOnPodRestart` policy is the action — a failed liveness probe kills a pod, and LWS tears down and recreates the whole group. Outside Kubernetes, or as a belt-and-suspenders sidecar, a small supervisor loop closes the same loop: poll the deep health gate, and on repeated failure, tear the replica down cleanly so the orchestrator (or a restart of the Ray head) rebuilds it rather than leaving a half-alive communicator to hang forever.

```bash
#!/usr/bin/env bash
# multinode_watchdog.sh -- supervise one multi-node vLLM replica.
# Polls the deep health gate; on N consecutive failures, drains and restarts the
# group so a dead peer becomes a clean rebuild instead of an eternal NCCL hang.
set -u
HEALTH="python3 /opt/serving/multinode_health.py"   # exits 0 healthy, 1 degraded
DRAIN_URL="http://localhost:8000/drain"             # stop accepting new requests
FAIL_LIMIT=3                                         # tolerate transient blips
POLL_SECONDS=10
fails=0

while true; do
  if $HEALTH >/dev/null 2>&1; then
    fails=0                                          # healthy: reset the counter
  else
    fails=$((fails + 1))
    echo "$(date -u +%FT%TZ) health FAIL ${fails}/${FAIL_LIMIT}" >&2
    if [ "$fails" -ge "$FAIL_LIMIT" ]; then
      echo "$(date -u +%FT%TZ) DEGRADED past limit -- draining and restarting group" >&2
      curl -fsS -X POST "$DRAIN_URL" || true         # let in-flight requests finish
      # Stop Ray on this node; the orchestrator recreates the gang from scratch.
      ray stop --force || true
      # Non-zero exit signals the supervisor/orchestrator to recreate the pod/group.
      exit 1
    fi
  fi
  sleep "$POLL_SECONDS"
done
```

The load-bearing choices are the failure counter and the drain-before-kill. The counter (`FAIL_LIMIT=3`) prevents a single transient blip — a momentarily busy GCS, a 10-second network hiccup — from needlessly restarting a minutes-to-recover replica; three consecutive failures over 30 seconds is a real fault, one is noise. The drain call gives in-flight requests a chance to finish and, more importantly, tells the router to stop sending new ones, so the restart does not black-hole live traffic. And the deliberate non-zero exit is what hands control back to the orchestrator: the watchdog's job is to *detect and hand off*, not to nurse a broken replica back to life in place.

#### Worked example: sizing replicas for an availability target

N+1 is the floor, but "how many replicas" is a numbers question. Suppose one two-node replica has an MTTR of 4 minutes (dominated by the ~68 s local-NVMe weight load plus NCCL init, warmup, and rejoin) and, across its two nodes and sixteen GPUs, a mean time between failures of 30 days. A single replica's availability is then roughly ${1 - (4\,\text{min} / 30\,\text{days}) \approx 1 - 9.3 \times 10^{-5} = 99.991\%}$ — but with one replica, every one of those failures is a *full outage*, and the tail is unforgiving because a node failure takes the whole service to zero for the entire 4-minute recovery. Run two replicas behind the router and a single replica's failure only degrades capacity; the service goes fully down only if *both* fail within the same recovery window, whose probability is the product of two small numbers — vanishingly rare. The practical rule: size for peak load at N replicas, then add one more so any single replica can fail without the survivors breaching their SLO. If a single replica already runs near its throughput ceiling, N+1 means provisioning genuine headroom, not just a spare — the +1 has to be able to absorb the failed replica's share of traffic, or your "redundant" fleet SLA-violates the moment it is actually needed.

## 9. Health checks, warmup, and rolling upgrades

Detecting a dead node is table stakes; operating a multi-node fleet also means safely pushing new versions and new weights without an outage, and doing so across replicas that each take minutes to become ready. Three practices carry the load.

**Layered health checks.** Distinguish liveness ("is the process alive?") from readiness ("can it serve a token *right now*?"). Liveness should be cheap and forgiving — a dead process gets restarted, but a briefly-busy engine should not be killed mid-batch. Readiness should be strict and should reflect the real cold-start sequence: do not report ready until weights are loaded, NCCL is up, and warmup is done. The `multinode_health.py` script above is a readiness-style deep check; the `/health` HTTP endpoint is the shallow one. Use both, at different cadences.

**Explicit warmup.** After the engine reports ready, but before it takes production traffic, send a batch of synthetic requests that exercise the common shapes — a short prompt, a long prompt near `max_model_len`, and a mid-size one — so the CUDA graphs and memory pools for those shapes are captured and the first *real* request does not pay the autotuning cost. This is a few seconds well spent; skipping it means your first production user after every deploy eats a multi-second TTFT spike.

**Rolling upgrades, one group at a time.** Never restart all replicas at once. With `replicas: 2` in the LWS, upgrade replica 1 while replica 2 carries traffic, wait for replica 1 to pass readiness *and* a warmup, shift traffic, then upgrade replica 2. This is the multi-node analogue of a blue-green or canary deploy, with the twist that each "color" is a multi-pod group with a minutes-long ready time — so your rollout controller must be patient and must gate on the deep readiness check, not just pod-`Running`. A deploy that ignores the cold-start window will happily route traffic to a half-initialized engine and manufacture exactly the p99 spike you were trying to avoid. If you are versioning weights and want automatic rollback on a regression, the patterns in the model-rollback and canary literature apply directly; the only multi-node wrinkle is the longer ready time, which means longer bake windows.

## 10. Observability across nodes

You cannot operate what you cannot see, and a multi-node engine has failure modes that are invisible from a single-node dashboard. The three signals that matter most are per-GPU utilization spread, NCCL collective time, and the inter-node link counters.

**Per-GPU utilization spread.** In a healthy TP8 × PP2 engine, the eight GPUs of a tensor-parallel group should show near-identical utilization — they run the same collective in lockstep. If one GPU sits at 60% while its peers are at 95%, you have a straggler (a thermal-throttled card, a bad NVLink, a rank doing extra work), and because TP collectives are barrier-synchronized, the slowest GPU sets the pace for all eight. Across the pipeline boundary, the two stages should show balanced utilization; a large imbalance means your layer split is uneven and one node is starving the other. Scrape `nvidia-smi`/DCGM metrics per GPU and alarm on *spread*, not just average.

**NCCL collective time.** This is the single most diagnostic multi-node signal, and the most commonly missing one. If the fraction of step time spent in communication climbs, you are drifting toward communication-bound, and the usual cause is either a fabric problem (a flapping link forcing retransmits) or an accidental TCP fallback. vLLM and the NCCL profiler can export collective timings; graph "communication time as a percentage of step time" and treat a rising trend as a fabric incident, not a model problem.

**Inter-node link counters.** Pull the InfiniBand port counters (`perfquery`, or the DCGM/node-exporter IB collectors): symbol errors, link recovery events, and receive-buffer overruns. A rising symbol-error rate on one rail is a physical layer problem — a bad cable or a dirty connector — that will manifest as mysterious latency spikes on exactly the collectives that traverse that rail. Catching it in the link counters points you straight at the cable instead of sending you on a multi-hour hunt through the model config.

The synthesis is a dashboard with three panels: GPU utilization *spread* across the replica, communication time as a percentage of step time, and IB error counters per rail. When TPOT rises, those three panels tell you in seconds whether the cause is a straggler GPU, a comms regression, or a physical fabric fault — the three things that a single-node dashboard simply cannot distinguish.

## 11. Benchmarks: 1 vs 2 vs 4 nodes on H100 InfiniBand

Numbers make the trade-offs unarguable. The measurement that matters most is how throughput and per-token latency change as you scale node count *and* as you choose the parallelism split — because those are two different decisions with two different outcomes.

![Matrix comparing TP16 over two nodes, TP8 by PP2 over two nodes, and TP8 by PP4 over four nodes on where TP lands, throughput, TPOT, and scaling verdict](/imgs/blogs/multi-node-llm-serving-100b-plus-8.webp)

The matrix above compares three configurations of Llama-3.1-405B (BF16) on 8×H100-80GB nodes connected by 400 Gb/s InfiniBand, and it encodes the section's whole argument: TP16-across-two-nodes is a trap. Here is the fuller before-and-after table on named hardware. **Treat these as representative order-of-magnitude figures for a synthetic 1024-input / 256-output workload at moderate concurrency, consistent with vLLM and NVIDIA public guidance — not as a certified benchmark.** Your exact numbers depend on input/output lengths, concurrency, and fabric quality.

| Config | GPUs | Model / precision | Weights fit? | Max out tok/s | TPOT p50 | TTFT p50 | Verdict |
|---|---|---|---|---|---|---|---|
| 1×8 H100 | 8 | 405B BF16 | No (810 > 640) | — | — | — | OOM at load |
| 1×8 H100 | 8 | 405B FP8 | Yes (405 GB) | ~3,200 | ~22 ms | ~0.4 s | Best if FP8 acceptable |
| 2×8 H100, **TP16** | 16 | 405B BF16 | Yes | ~900 | ~85 ms | ~1.1 s | TP crosses IB — avoid |
| 2×8 H100, **TP8×PP2** | 16 | 405B BF16 | Yes | ~2,600 | ~28 ms | ~0.6 s | Recommended 2-node |
| 4×8 H100, **TP8×PP4** | 32 | 405B BF16 | Yes | ~4,700 | ~34 ms | ~0.7 s | Scale-out, ~0.90× linear |

Read three lessons off this table. First, **the single node in FP8 beats every multi-node BF16 configuration on latency** — 22 ms TPOT versus 28+ — because it pays no inter-node round trip at all. If FP8 is acceptable for your accuracy budget, the multi-node question may not even arise. Second, **TP16-across-two-nodes is roughly 3× slower than TP8×PP2 on the identical hardware** — 900 tok/s versus 2,600, 85 ms versus 28 ms TPOT — purely because half of every tensor-parallel `AllReduce` now traverses InfiniBand. Same GPUs, same model, same node count; the only difference is where you put the tensor-parallel boundary, and it costs you a factor of three. Third, **adding pipeline stages scales throughput near-linearly** — going from 16 to 32 GPUs (TP8×PP2 to TP8×PP4) roughly doubles aggregate throughput at about 90% efficiency, with a modest TPOT increase from the added pipeline bubble. That is the good kind of scaling: more nodes buy you more throughput and more KV headroom for larger batches, as long as TP stays home.

Here is how to reproduce the measurement. vLLM ships a benchmark that drives the OpenAI-compatible endpoint and reports the full latency distribution:

```bash
# Drive the running engine and measure TTFT / TPOT / throughput end to end.
vllm bench serve \
  --model meta-llama/Llama-3.1-405B \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 1000 \
  --request-rate 8            # offered load in requests/second; sweep this
# Reports: request throughput (req/s), output token throughput (tok/s),
#          and mean / median / p99 for TTFT and TPOT.
```

To find the real capacity, sweep `--request-rate` upward until p99 TPOT crosses your SLA — the request rate just below that crossover is your safe operating point. If you want to characterize the fabric contribution specifically, run the sweep once with the recommended TP8×PP2 layout and once with TP16, and the gap between the two curves is precisely the cost of crossing the node boundary with tensor parallelism. That single experiment is the most convincing artifact you can bring to a capacity-planning review, because it turns an abstract rule into a latency curve on your own hardware. If you would rather script it against the OpenAI client directly for a custom traffic shape, a compact async harness does the job:

```python
import asyncio, time
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def one(prompt: str):
    t0 = time.perf_counter()
    first = None
    n = 0
    stream = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-405B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256, stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter() - t0     # TTFT
            n += 1
    total = time.perf_counter() - t0
    tpot = (total - first) / max(n - 1, 1)            # per-output-token time
    return first, tpot

async def main(concurrency=32):
    tasks = [one("Summarize the theory of multi-node inference.")
             for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    ttfts = sorted(r[0] for r in results)
    tpots = sorted(r[1] for r in results)
    p = lambda xs, q: xs[int(q * (len(xs) - 1))]
    print(f"TTFT p50={p(ttfts,0.5)*1000:.0f}ms p99={p(ttfts,0.99)*1000:.0f}ms")
    print(f"TPOT p50={p(tpots,0.5)*1000:.1f}ms p99={p(tpots,0.99)*1000:.1f}ms")

asyncio.run(main())
```

## Case studies

**DeepSeek-V3 / R1 — expert parallelism at 320 GPUs.** DeepSeek's technical report documents a serving architecture that is the clearest public example of the placement rules in this post taken to their logical extreme. They disaggregate prefill and decode (the subject of the sibling post on [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation)) and run each on a different node count. The **prefill** minimum unit is 4 nodes / 32 GPUs: attention uses tensor parallelism of 4 with sequence parallelism and 8-way data parallelism, while the MoE layers use EP32. The **decode** minimum unit is 40 nodes / 320 GPUs: attention runs TP4 with 80-way data parallelism, and the MoE uses EP320 — every one of 320 GPUs holds a small slice of the 256 routed experts. The all-to-all expert dispatch/combine runs over InfiniBand and is overlapped with computation so it does not sit idle on the critical path; DeepSeek open-sourced the `DeepEP` communication library specifically to make that expert-parallel all-to-all fast. MLA keeps the KV cache around 70 KB/token so that, even at 320-GPU scale, memory bandwidth rather than KV capacity is the binding constraint. Two further details make the decode unit workable at that scale: DeepSeek replicates the highest-load experts (a *redundant-experts* strategy) so that routing skew does not leave a handful of GPUs saturated while the rest idle, and they overlap the all-to-all dispatch/combine of one micro-batch with the attention and gating compute of another so the inter-node expert traffic is hidden behind useful work rather than stalling the pipeline. The through-line: TP stays small and local (4-way), and the *massive* parallelism is on the overlappable expert axis across nodes — exactly the rule from Section 3.

**Llama-3.1-405B — the quantize-first lesson.** Meta's 405B model is the canonical dense-model boundary case, and vLLM's distributed-serving documentation gives the operational verdict directly. In FP8, 405B fits on a single 8×H100 node (405 GB of weights) and is the recommended deployment when FP8 accuracy is acceptable — no inter-node network involved. In BF16, it requires 16×H100 across two nodes, and the docs explicitly recommend **TP8 × PP2**, warning against setting `tensor_parallel_size` larger than the GPUs in one node precisely because it forces tensor-parallel collectives across the slow inter-node link. This is the single most cited multi-node serving guideline in production, and it is the direct application of the worked example in Section 3.

**vLLM + Ray + LeaderWorkerSet — the reference production path.** The combination shown throughout this post — vLLM's Ray executor for the engine, a placement group to pin tensor-parallel groups per node, and a Kubernetes `LeaderWorkerSet` for gang-scheduled, self-healing replicas — is the pattern the vLLM production stack and the `llm-d` project have converged on. It is worth internalizing as a unit: Ray gives you the single-logical-engine view across nodes, the placement group enforces the intra-node TP constraint, and LWS gives you the "restart the whole group together" semantics that a multi-node NCCL communicator requires. Each piece exists to solve one specific problem that a naive `Deployment` cannot.

## When to use this (and when not to)

Multi-node serving is powerful and expensive, and the most important skill is knowing when *not* to reach for it.

**Cross the node boundary only when memory forces you.** The clean trigger is: the model's weights plus a minimum useful KV budget exceed one node's aggregate HBM, and you have already exhausted quantization. Llama-405B in BF16 (810 GB) forces it. DeepSeek-V3 in FP8 (671 GB) forces it. A 70B model does not — it fits comfortably on a single node even in BF16. Before you provision a second node, ask whether FP8 or INT4 brings the model under the single-node ceiling; a single node is simpler, cheaper, lower-latency, and has a smaller failure surface than any multi-node configuration, so quantization is almost always the better first move.

**Do not go multi-node for throughput alone.** If your model fits on one node and you simply need more capacity, the right answer is more single-node *replicas* behind a router (data parallelism), not one model spread thinner across more nodes. Independent replicas have zero inter-node collectives, a per-replica failure domain, and trivial scaling. Spreading a model that fits on one node across two nodes to "use more GPUs" is strictly worse: you add inter-node latency to every token and enlarge the blast radius of a node failure, for nothing.

**Accept the costs when you do cross.** Multi-node buys you the ability to serve a model that otherwise could not run at all, but it charges you on every axis of the SLO triangle: a minutes-long cold start (and therefore a minutes-long recovery), a failure surface where any node's death breaks a whole replica, the capital and configuration cost of an RDMA fabric, the ongoing burden of NCCL and topology tuning, and per-token network latency added to TPOT. Those costs are worth it for a 405B or a 671B model that genuinely does not fit — and they are pure waste for a model that does. When the model fits one node and your p99 TPOT is tight, single-node wins every time.

## Key takeaways

1. **The node boundary is a memory decision, not a throughput decision.** You cross it when weights plus a usable KV budget exceed one node's HBM. Compute $M_\text{weights} = P \cdot b_w$ first; if it clears the node, quantize before you scale.
2. **Quantize before you scale out.** Llama-405B is a single-node deployment in FP8 (405 GB) and a two-node deployment only in BF16 (810 GB). A single node is always simpler, cheaper, and lower-latency.
3. **The interconnect hierarchy dominates.** Bandwidth drops ~18× and latency rises 10–50× the instant a transfer leaves the node. That cliff, not the GPU, governs your design.
4. **Match each parallelism axis to a link.** TP (frequent, non-overlappable, on the critical path) stays intra-node on NVLink — `tensor_parallel_size` ≤ GPUs per node. PP and EP (infrequent or overlappable) cross nodes on InfiniBand. DP replicas go anywhere.
5. **TP across the node boundary triples your latency.** TP16-across-two-nodes is ~3× slower than TP8×PP2 on identical hardware. Never do it.
6. **Verify the fabric before you trust a number.** Run once with `NCCL_DEBUG=INFO`; if inter-node peers report `via NET/Socket`, your RDMA is broken and every benchmark is a lie.
7. **Cold start is your recovery time.** Weight loading (60–120 s for 405B) plus NCCL init plus warmup is minutes. Fast weight loading is not a nicety; it is your MTTR.
8. **Inference fault tolerance is easier than training's — but needs N+1 replicas.** No checkpoint to restore, just reload and re-form rings; but a single multi-node replica means a node death is a full outage. Run two.
9. **Gang-schedule and gang-restart.** Use `LeaderWorkerSet` with `RecreateGroupOnPodRestart`: a half-alive NCCL communicator is worse than a clean group restart.
10. **Observe the spread, the comms time, and the IB counters.** Those three signals distinguish a straggler GPU from a comms regression from a bad cable — distinctions a single-node dashboard cannot make.

## Further reading

- [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — the SLO triangle and the serving-vs-training distinction this whole series builds on.
- [Tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) — the mechanics of each parallelism axis, taken as given here.
- [Prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) — why prefill and decode run on different node counts, as in the DeepSeek case study.
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — the engine internals, `EngineArgs`, and single-node parallelism that multi-node builds on.
- [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) — how TP, PP, and DP compose onto a device mesh, from the training side.
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) — §3.4 and §5 document the prefill/decode node counts, EP320 decode, and MLA KV cache.
- Dubey et al., "The Llama 3 Herd of Models" (Meta, 2024) — the 405B architecture and precision options.
- vLLM documentation, "Distributed Inference and Serving" and the `examples/online_serving/run_cluster.sh` reference — the canonical Ray multi-node launch.
- NVIDIA, "NCCL User Guide" and the "GPUDirect RDMA" documentation — transport selection, `NCCL_*` environment variables, and rail-optimized topology.
- Kubernetes SIG-scheduling, "LeaderWorkerSet API" — the gang-scheduled multi-node inference primitive.

---
title: "Multi-LoRA and adapter serving: hot-swapping fine-tunes at scale"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to serve hundreds of customer-specific LoRA adapters from a single base model deployment, cutting GPU memory 50x while keeping per-request latency within SLA."
tags:
  [
    "model-serving",
    "inference",
    "lora",
    "adapter-serving",
    "vllm",
    "s-lora",
    "fine-tuning",
    "multi-tenant",
    "llm-infrastructure",
    "gpu-memory",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/multi-lora-and-adapter-serving-1.png"
---

You have 100 enterprise customers. Each one paid for a fine-tuned version of Llama-3-8B — trained on their internal documents, calibrated to their brand voice, aligned to their compliance requirements. Now they all want to use it simultaneously. You run the naive math: 100 models × 16 GB each = 1.6 TB of GPU memory. That's 20 H100 80GB GPUs, just sitting there holding identical weights with tiny deltas on top. Your cost structure is broken before you ship a single request.

This is the multi-LoRA problem. It is one of the most practically important problems in production LLM serving, and it has a clean solution that most teams don't implement until they're already hemorrhaging GPU budget. The insight is deceptively simple: if all 100 customers fine-tuned from the same base model using Low-Rank Adaptation (LoRA), then 99.4% of the weights are identical across all deployments. Only the adapter deltas differ. You don't need 100 copies of the model. You need one copy of the base model plus 100 tiny adapter files, and a serving system that can route each request to the right adapter on the fly.

Do it right, and you go from 1,600 GB to roughly 26 GB — the shared 16 GB base plus 100 adapters at about 100 MB each. That is the memory picture illustrated in Figure 1. But memory is the easy part. The hard parts are: how do you batch requests needing different adapters together without stalling the GPU? How do you manage hundreds of adapters when only a handful fit on GPU at once? How do you handle adapter version updates without a server restart? What does the custom CUDA kernel architecture look like that makes this efficient? And critically, when does the whole approach break down?

This post covers the complete picture: the mathematics of LoRA that makes sharing possible, the vLLM multi-LoRA implementation that you can ship today, the S-LoRA and Punica research systems that pushed the limit to 1,000+ concurrent adapters, the memory management strategies for production fleets, the operational patterns for dynamic adapter loading and versioning, and the explicit decision criteria for when to merge adapters instead of serving them dynamically.

![Naive 100-model deployment vs multi-LoRA sharing one 16 GB base — a 60x memory reduction](/imgs/blogs/multi-lora-and-adapter-serving-1.png)

## 1. LoRA recap: the mathematics that make sharing possible

Before diving into serving architecture, you need a firm grip on what a LoRA adapter actually is — because the structure of the adapter directly determines every serving trade-off downstream.

### The standard fine-tuning problem

Standard fine-tuning of a large language model updates all parameters in every weight matrix. For a 7B-parameter model stored in BF16, that is roughly 14 GB of weights to update during training and load during inference. If you want to produce 100 customer-specific variants, you produce 100 sets of 14 GB weight files. Each customer's inference engine loads a full 14 GB model. The GPU memory scales linearly with the number of customers.

LoRA (Hu et al., ICLR 2022) starts from the observation that the weight change $\Delta W$ needed to adapt a pretrained model to a downstream task has **low intrinsic rank**. Aghajanyan et al. (2020) showed empirically that the gradient updates during fine-tuning tend to lie in a low-dimensional subspace — the "intrinsic dimensionality" of the fine-tuning task is much smaller than the full parameter count. LoRA exploits this by directly parameterizing the low-rank update.

### The LoRA decomposition

For a weight matrix $W \in \mathbb{R}^{d \times k}$ (e.g., the query projection $W_Q$ in an attention layer), LoRA parameterizes the update as:

$$\Delta W = BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. During fine-tuning, $W$ is **frozen** — its gradients are zeroed out. Only $A$ and $B$ are trained. This reduces the number of trainable parameters from $d \times k$ to $r(d + k)$. For a $4096 \times 4096$ projection with $r = 16$: from $16.7$ million down to $131,072$ — a 128× reduction in trainable parameters per layer.

During inference, the effective weight is computed as:

$$W_{\text{eff}} = W_{\text{pretrained}} + \frac{\alpha}{r} \cdot BA$$

The $\alpha/r$ scaling factor controls the magnitude of the update. In practice, fine-tuning practitioners often set $\alpha = r$ (making the scale factor 1), or use $\alpha$ as a hyperparameter tuned separately from $r$. The standard PEFT library default is $\alpha = 8$ with $r = 8$ (scale factor 1) or $\alpha = 16$ with $r = 8$ (scale factor 2).

The initialization convention matters for training stability: $A$ is initialized with a Gaussian distribution (small random values), and $B$ is initialized to zero. This ensures that $\Delta W = BA = 0$ at the start of training — the fine-tuned model begins identical to the pretrained model, and the adapter gradually diverges as training progresses.

### Which weight matrices to target

Not all weight matrices in a transformer need to be adapted. The original LoRA paper targeted only $W_Q$ and $W_V$ in the attention layers. Subsequent work found that targeting more matrices — $W_Q, W_K, W_V, W_O$ and the MLP projections $W_{\text{up}}, W_{\text{down}}, W_{\text{gate}}$ — yields better task performance at the same total rank budget. The choice is exposed as the `target_modules` parameter in PEFT:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")

lora_config = LoraConfig(
    r=16,                     # rank
    lora_alpha=16,            # scaling factor (alpha/r = 1)
    target_modules=[          # which weight matrices to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,290,304 || trainable%: 0.52
```

### Adapter file size calculation

For Llama-3-8B ($d = 4096$, 32 layers) at rank $r = 16$ targeting all 7 projection types:

$$\text{Parameters per matrix pair} = r \cdot d + r \cdot d = 2 \cdot r \cdot d = 2 \times 16 \times 4096 = 131{,}072$$

(using $d = k = 4096$ for the square projection matrices; MLP up/gate/down are $4096 \times 14336$ and $14336 \times 4096$, so their $(A, B)$ pairs are larger)

More precisely for Llama-3-8B:
- Attention projections (Q, K, V, O at $4096 \times 4096$): $4 \times 2 \times 16 \times 4096 = 524{,}288$ parameters per layer
- MLP projections (up/gate at $4096 \times 14336$, down at $14336 \times 4096$): $3 \times 16 \times (4096 + 14336) = 885{,}504$ parameters per layer
- Total per layer: $\approx 1.41$ million parameters
- 32 layers: $\approx 45$ million parameters
- BF16 storage: $45\text{M} \times 2\text{ bytes} \approx 90$ MB

A rank-16 adapter for Llama-3-8B with full projection coverage runs about **80–120 MB** including metadata and safetensors framing. Bump to rank 64 and you're at ~360–480 MB. Bump to rank 128 and you're at ~720–960 MB.

The critical property for multi-LoRA serving: **the base model weights are identical across all customers**. Every customer's adapter is just a pair of small matrices per layer. This is what makes it physically possible to share the base model weights across an entire customer fleet.

![LoRA adapter anatomy: frozen base weight plus learned A and B matrices forming a 100 MB delta](/imgs/blogs/multi-lora-and-adapter-serving-2.png)

## 2. The naive approach and why it fails

Before describing multi-LoRA serving, it's worth being precise about why the naive approaches fail. This isn't just a memory argument — it's a GPU utilization and operational argument.

### Naive approach 1: 100 separate vLLM processes

You spin up one vLLM instance per customer, each with the merged model (adapter baked into weights via `peft`'s `merge_and_unload()`). Problems compound quickly:

**Memory**: 100 × 16 GB = 1,600 GB. You need 20 H100 80GB cards purely for weights — before allocating any KV cache. KV cache is what enables batching and high throughput (see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)). Without KV cache headroom, you can't batch requests at all.

**Utilization**: each customer gets roughly 1 QPS at peak. Average utilization per GPU is under 5%. You're paying for 20 GPUs at 5% utilization each. The other 95% of GPU compute sits idle, waiting for requests that arrive once every minute.

**Operational overhead**: 100 separately managed processes. 100 separate rolling update pipelines when the base model changes (e.g., you upgrade to Llama-3.1-8B). 100 separate health endpoints to monitor. 100 separate Prometheus scrape targets. When a customer asks why their model is slow, you're debugging 1 of 100 processes that may each be in different states.

**KV cache fragmentation**: each process allocates its own KV cache pool. A customer with low traffic holds allocated GPU memory that can't be reclaimed by a high-traffic customer. The system cannot dynamically balance GPU memory between idle and busy customers.

### Naive approach 2: one process, switch adapters between requests

You run one vLLM instance, store the base model, and swap the adapter before processing each request — load adapter A, process request, unload A, load adapter B, process next request, and so on in sequence. This preserves the single-process operational model but breaks throughput:

**Throughput destruction**: adapter loading involves CPU→GPU memory transfers. A 100 MB adapter at PCIe 4.0 bandwidth (~28 GB/s for a well-optimized transfer) takes roughly 3.5 ms per load. At 10 GB/s effective throughput (PCIe bandwidth is shared and overhead adds up), it's 10 ms. For a server handling 1,000 req/s, you'd spend 10 seconds per second on adapter I/O — computationally impossible.

**No batching**: you can't batch requests with different adapters together. You process them serially, one adapter group at a time. The GPU sits idle during I/O, reducing effective utilization from ~70–80% (well-tuned continuous batching) to perhaps 20–30%.

**The real solution** is to hold multiple adapters on GPU simultaneously and process heterogeneous-adapter requests in the same forward pass batch. This is multi-LoRA serving, and it makes both problems disappear.

## 3. vLLM multi-LoRA serving: configuration and deployment

vLLM has native multi-LoRA support as of version 0.3, and it is the most production-ready multi-LoRA implementation available today. Here is how to configure and use it.

### Launching the server with multi-LoRA enabled

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-extra-vocab-size 256 \
  --max-cpu-loras 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --port 8000
```

Each flag has a concrete effect on memory and behavior:

**`--enable-lora`**: activates the multi-LoRA serving path within vLLM's engine. Without this flag, LoRA adapters are ignored even if you pass them in requests — the request will be served using only the base model weights.

**`--max-loras N`**: how many adapter weight sets are held **on GPU** simultaneously. Each adapter at rank 64 for Llama-3-8B takes approximately 360 MB. With `--max-loras 4`, you're reserving roughly 1.44 GB of GPU memory for adapter slots. These slots are pre-allocated at startup, so increasing this value reduces your KV cache budget.

**`--max-lora-rank R`**: the maximum rank of any adapter you'll serve. This pre-allocates GPU buffer space sized for the worst-case adapter geometry. If you set this too low (e.g., `--max-lora-rank 16`) and then send a request with a rank-32 adapter, vLLM will raise an error. Set it to your actual maximum to avoid wasted memory from over-provisioning.

**`--lora-extra-vocab-size V`**: some fine-tunes extend the vocabulary with new tokens (e.g., domain-specific product codes, technical abbreviations). This reserves additional slots in the embedding matrices for those tokens. If your adapters don't add tokens, set this to 0.

**`--max-cpu-loras N`**: adapters not currently on GPU are cached in CPU RAM as a software LRU cache. This controls how many adapters fit in the CPU cache before vLLM starts fetching from the original storage path. For a system with 100 adapters at 100 MB each, setting `--max-cpu-loras 100` uses 10 GB of CPU RAM — trivially affordable.

### Making a LoRA request with the Python API

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64,
    max_cpu_loras=32,
)

# Request using customer A's legal document adapter
responses_a = llm.generate(
    prompts=["Summarize the following contract clause: the licensor..."],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    lora_request=LoRARequest(
        lora_name="customer_a_legal",      # human-readable name
        lora_int_id=1,                      # integer ID for fast internal lookup
        lora_local_path="/adapters/customer_a/",
    ),
)

# Request using customer B's marketing copy adapter (different fine-tune)
responses_b = llm.generate(
    prompts=["Write a product description for a noise-canceling headphone:"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
    lora_request=LoRARequest(
        lora_name="customer_b_marketing",
        lora_int_id=2,
        lora_local_path="/adapters/customer_b/",
    ),
)
```

The `LoRARequest` has three required fields:

- `lora_name`: human-readable string, used in logs and metrics. Two requests with the same `lora_name` but different `lora_int_id` will both be registered as separate adapters — the name is purely for observability.
- `lora_int_id`: integer identifier, used for fast O(1) lookup in vLLM's internal adapter table. Must be unique per distinct adapter. If you send the same `lora_int_id` with a different `lora_local_path`, vLLM raises a `ValueError` — this prevents accidentally aliasing two adapters to the same internal slot.
- `lora_local_path`: path to a directory containing a valid HuggingFace PEFT adapter checkpoint: `adapter_config.json` defining the LoRA configuration, plus `adapter_model.safetensors` (or the older `adapter_model.bin`) containing the actual weight tensors.

### Using the OpenAI-compatible HTTP API

For production systems where the inference engine sits behind an API gateway, you interact via HTTP:

```python
import requests

# Direct vLLM OpenAI-compatible endpoint
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Summarize this contract clause: ..."}
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "extra_body": {
            "lora_request": {
                "lora_name": "customer_a_legal",
                "lora_int_id": 1,
                "lora_local_path": "/adapters/customer_a/",
            }
        },
    },
    headers={"Authorization": "Bearer dummy"},
)
print(response.json()["choices"][0]["message"]["content"])
```

Or with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

completion = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Analyze this legal clause: ..."}],
    max_tokens=256,
    extra_body={
        "lora_request": {
            "lora_name": "customer_a_legal",
            "lora_int_id": 1,
            "lora_local_path": "/adapters/customer_a/",
        }
    },
)
```

For the AsyncLLMEngine (used when you need streaming and high concurrency), the pattern is identical — just use `async for output in engine.generate(...)` and pass the same `lora_request` parameter.

![vLLM heterogeneous batch: requests with different adapters merged into one decode step using sliced matrix multiplications](/imgs/blogs/multi-lora-and-adapter-serving-3.png)

## 4. How vLLM batches multi-LoRA requests

The batching mechanics are where most engineers stop reading the documentation and start guessing. The guessing is costly — the batching algorithm determines your throughput, your memory trade-offs, and your latency behavior under skewed adapter access patterns.

### The base model forward pass

When vLLM's scheduler builds a decode batch of $B$ total tokens, it begins by running the full base model forward pass on all $B$ tokens simultaneously. The base model computation — attention, MLP, layer norms — is identical for every request regardless of adapter. This is the bulk of the computation, and it is fully batched with no per-adapter branching. The output of the base model forward pass for each token position is an activation vector $x_i \in \mathbb{R}^d$.

### The adapter delta pass

After the base model forward pass, each token position $i$ needs its adapter-specific delta applied. Let token $i$ belong to adapter $a_i$. The adapter delta for the query projection of layer $l$ is:

$$\delta_i^{(Q)} = \frac{\alpha_{a_i}}{r_{a_i}} B_{a_i}^{(Q)} A_{a_i}^{(Q)} x_i$$

Naively, you'd iterate over each distinct adapter $a$ in the batch, collect the token positions belonging to $a$, run the matmul for those positions, and scatter the results back. The total compute is the same, but you pay $N$ kernel launch overheads (one per distinct adapter) plus poor GPU occupancy between launches.

**vLLM's approach: sliced matrix multiplications.** For each layer and weight type, vLLM operates over the entire batch at once using index-based slicing:

1. For adapter $a$ with token positions $T_a \subseteq [B]$: gather the rows $x_{T_a}$ from the activation tensor $x \in \mathbb{R}^{B \times d}$
2. Compute $h_a = x_{T_a} A_a \in \mathbb{R}^{|T_a| \times r}$ — the down-projected intermediate
3. Compute $\delta_a = h_a B_a \in \mathbb{R}^{|T_a| \times d}$ — the up-projected delta
4. Scale by $\alpha_a / r_a$ and scatter-add $\delta_a$ into the output at positions $T_a$

Steps 1–4 happen for all adapters in sequence, but because they operate on different (non-overlapping) rows of the activation tensor, they can be dispatched as back-to-back CUDA kernel calls with minimal synchronization. Modern CUDA drivers pipeline these launches effectively.

### The overhead budget

The throughput overhead of multi-LoRA serving compared to single-adapter serving is approximately:

$$\epsilon \approx \frac{\sum_a |T_a| \cdot r_a \cdot 2d}{\text{FLOPs}_{\text{base}}}$$

For a batch of 128 tokens across 4 adapters (rank 16 each) with a $d = 4096$ base model:
- Delta FLOPs per adapter per layer: $|T_a| \times r \times 2d = 32 \times 16 \times 8192 \approx 4.2$ MFLOPs
- Base model FLOPs per layer per token: $\approx 2 \times 4096^2 = 33.5$ MFLOPs/token, so $128 \times 33.5 = 4.3$ GFLOPs
- Delta FLOPs total (4 adapters × 32 layers): $4 \times 4.2\text{M} \times 32 = 537$ MFLOPs
- Overhead: $537 / 4300 \approx 12.5$%

In practice, the measured overhead is lower (5–10%) because the adapter delta computation is memory-bandwidth-bound rather than compute-bound on modern GPUs — the activation matrices fit in L2 cache and the adapter matrices are small enough for fast loads.

The overhead is nearly constant with respect to the number of distinct adapters in the batch (as long as they're all GPU-resident), because the delta passes are parallelized. Whether you have 2 or 20 adapters in a batch, the delta pass takes roughly the same wall-clock time.

### Scheduling priority and SLO fairness

In a multi-tenant multi-LoRA deployment, you need to think about whether all customers' requests get equal scheduling priority. vLLM's scheduler uses FIFO ordering within its continuous batching loop — the oldest pending request gets scheduled first, regardless of which adapter it requires.

For most SaaS workloads, FIFO is acceptable because customers pay for shared infrastructure and implicitly accept shared-queue fairness. But if you have tiered service levels — a "premium" customer tier that gets guaranteed lower p99 latency than a "standard" tier — you need to implement priority queuing at the API gateway layer, ahead of vLLM. A common pattern: maintain separate vLLM instances or separate Ray Serve deployment replicas for premium and standard customers. Priority logic belongs in the gateway, not in vLLM — the inference engine should focus on high-throughput serving, and scheduling policy should live in the layer that owns customer SLAs.

### Adapter slot management: the GPU LRU cache

vLLM maintains a fixed-size LRU cache of adapter weight tensors on GPU. When a request arrives for adapter $a$:

1. **Cache hit**: adapter $a$'s $(A, B)$ tensors are already in a GPU slot. The request proceeds immediately.
2. **Cache miss, free slot exists**: initiate an asynchronous PCIe transfer of adapter $a$ from CPU RAM to the free GPU slot. The request is held in the scheduler's pending queue until the transfer completes.
3. **Cache miss, all slots full**: select the least-recently-used adapter $b$ to evict. If adapter $b$ has been modified in GPU memory (not the case for serving — adapters are read-only), write it back to CPU RAM. Copy adapter $a$ to the now-free slot. The request waits for both the eviction and the load.

The async transfer approach means that cache misses don't stall other requests — the GPU continues processing cache-hitting requests while the transfer happens on a separate CUDA stream. You only see latency on the requests that triggered the miss.

## 5. S-LoRA: scaling to 1,000+ adapters on a single GPU

vLLM's native multi-LoRA handles the case where your concurrent active adapters fit in GPU memory. But what if you have 500 distinct customer adapters, each sending sporadic requests? At rank 64, 500 adapters × ~360 MB = 180 GB — far more than a single H100's GPU memory after loading the base model. You need a system that handles adapters that don't all fit on GPU simultaneously.

This is the problem S-LoRA (Sheng et al., NeurIPS 2023) was designed to solve. S-LoRA introduced three key contributions that have since influenced vLLM's implementation.

### Contribution 1: unified memory paging

In standard LLM serving, GPU memory divides into two regions:
- **Static allocations**: model weights (fixed at load time, never change during serving)
- **Dynamic allocations**: KV cache pages (allocated and freed as requests arrive and complete)

vLLM's PagedAttention manages the KV cache as a pool of fixed-size pages (typically 16 tokens × 32 layers × 2 (K+V) × 2 bytes = 4 MB per block for Llama-7B). When GPU memory is tight, vLLM evicts KV cache pages to CPU using a swap mechanism.

S-LoRA extends this unified paging to **adapter weight tensors**. Instead of pre-reserving fixed GPU slots for adapters (as vLLM native does), S-LoRA treats adapter weight pages and KV cache pages as entries in the same unified memory pool. The GPU memory manager allocates and evicts pages from a single pool, making eviction decisions based on combined utility — it doesn't distinguish between "KV page for request X" and "adapter B matrix for adapter Y."

The benefit: if your workload shifts from adapter-diverse (many distinct customers, each sending a few requests) to adapter-concentrated (one customer sending a burst), the memory pool automatically reallocates from adapter slots to KV pages without any configuration change.

### Contribution 2: on-demand adapter fetching from CPU RAM

S-LoRA stores all adapters in CPU RAM (or NFS-mounted storage for very large catalogs). The adapter serving pipeline works as follows:

```python
# Conceptual representation of S-LoRA's adapter fetch pipeline
# Actual implementation is C++/CUDA

class AdapterMemoryManager:
    def __init__(self, gpu_pool_size_bytes, cpu_cache_size_bytes):
        self.gpu_pool = GPUMemoryPool(gpu_pool_size_bytes)
        self.cpu_cache = LRUCache(cpu_cache_size_bytes)
        self.prefetch_queue = asyncio.Queue()
    
    async def get_adapter(self, adapter_id: str) -> GPUTensorHandle:
        # 1. Check GPU pool (O(1) lookup)
        if adapter_id in self.gpu_pool:
            return self.gpu_pool.get(adapter_id)
        
        # 2. Check CPU cache
        if adapter_id in self.cpu_cache:
            cpu_tensors = self.cpu_cache.get(adapter_id)
        else:
            # 3. Load from disk/NFS
            cpu_tensors = await self.load_from_storage(adapter_id)
            self.cpu_cache.put(adapter_id, cpu_tensors)
        
        # 4. Copy CPU → GPU (async, on a dedicated CUDA stream)
        gpu_handle = await self.gpu_pool.allocate_and_copy(
            cpu_tensors,
            evict_lru_if_needed=True
        )
        return gpu_handle
    
    def prefetch(self, upcoming_adapter_ids: list[str]):
        """Kick off async transfers for adapters likely to be needed soon."""
        for adapter_id in upcoming_adapter_ids:
            if adapter_id not in self.gpu_pool:
                asyncio.create_task(self.get_adapter(adapter_id))
```

The prefetching is critical for latency. S-LoRA's scheduler analyzes the pending request queue and initiates async CPU→GPU transfers for adapters that will be needed within the next scheduling window. By the time a request reaches the top of the queue, its adapter is typically already on GPU.

### Contribution 3: custom CUDA kernels for heterogeneous batches

S-LoRA's most technically novel contribution is a set of CUDA kernels for applying LoRA deltas in heterogeneous batches. The flagship kernel is a modified version of BGMV (Batched Gather Matrix-Vector Multiplication).

The standard LoRA delta computation for a homogeneous batch (all requests use the same adapter) is a simple matrix multiplication: $H = XA \in \mathbb{R}^{B \times r}$, then $\Delta = HB \in \mathbb{R}^{B \times d}$. This is highly efficient — it maps directly to the `gemm` (General Matrix-Matrix Multiplication) operation that NVIDIA's cuBLAS library is heavily optimized for.

For a heterogeneous batch, the "gather" step (selecting different adapter matrices for different rows of $X$) breaks the simple `gemm` structure. S-LoRA's BGMV kernel handles this with a custom tiling strategy:

- The activation matrix $X \in \mathbb{R}^{B \times d}$ is processed in tiles of 16 rows
- Each tile is assigned to a CUDA warp
- Each warp loads the corresponding adapter index from the `adapter_ids` array, fetches the appropriate $A$ matrix tile from shared memory, and computes the partial product
- The warp-level partial products are accumulated into the output buffer

The key optimization: by processing multiple adapters' $A$ matrices within the same kernel launch (not sequentially), the kernel achieves much higher GPU occupancy. The GPU's memory subsystem can overlap the fetching of adapter $A$ tiles for different warps, hiding the latency of the irregular memory access pattern.

**Measured speedup of S-LoRA's BGMV kernel** (from the paper, A100 80GB, batch size 32):
- Naive sequential per-adapter matmuls: 1,820 tokens/s
- S-LoRA BGMV: 9,400 tokens/s (5.2× speedup)
- Single-adapter baseline: 10,100 tokens/s
- Multi-adapter overhead with BGMV: 7%

The kernel was later adapted and extended by the Punica project (Chen et al., MLSys 2024), which pushed the BGMV approach further and demonstrated linear throughput scaling up to 64 concurrent adapters.

![S-LoRA unified memory pool: adapter weights and KV cache blocks share the same GPU memory, with LRU eviction to CPU RAM](/imgs/blogs/multi-lora-and-adapter-serving-4.png)

## 6. Punica: the BGMV kernel system

Punica (Chen et al., MLSys 2024) is a parallel academic multi-LoRA system that converged on similar ideas to S-LoRA from a kernel-centric angle. While S-LoRA's focus is on memory management and adapter scaling, Punica's focus is on making the heterogeneous batch compute kernel as arithmetically efficient as possible.

### The BGMV kernel design

Punica's BGMV kernel operates in two passes per layer:

**Pass 1 (Down projection):** Compute $h_i = x_i A_{a_i} \in \mathbb{R}^r$ for each token position $i$ where $a_i$ is the adapter assigned to token $i$. This is a "batched gather MVM" — each row of the activation matrix $x$ is multiplied by a different $A$ matrix.

**Pass 2 (Up projection):** Compute $\delta_i = \text{scale}_{a_i} \cdot h_i B_{a_i} \in \mathbb{R}^d$ for each token position. Another batched gather MVM, this time projecting back to the hidden dimension.

The kernel is invoked as:

```python
# Conceptual Punica BGMV call signature (actual: CUDA C++ extension)
bgmv(
    output=out_tensor,           # [batch_size, hidden_dim]
    inputs=x_tensor,             # [batch_size, hidden_dim]
    weights=stacked_A_matrices,  # [max_adapters, rank, hidden_dim]
    adapter_indices=adapter_ids, # [batch_size] — index into weights
    scales=lora_scales,          # [max_adapters]
    layer_idx=layer_index,
)
```

The kernel launch geometry:
- One thread block per output row (i.e., per token position)
- Each thread block is assigned the adapter index for its row, loads the corresponding $A$ matrix slice into shared memory, and computes the dot product
- Thread blocks for different adapter indices can execute in parallel — there is no cross-block synchronization

This design achieves near-linear scaling with batch size up to the point where the GPU's thread block scheduler saturates, and near-constant overhead relative to batch size for the adapter delta pass (because the total compute scales with batch size, not with the number of distinct adapters).

**Punica benchmarks** (A100 40GB, LLaMA-7B, rank 16, 16 concurrent adapters):

| Batch size | Punica (tokens/s) | Single-adapter (tokens/s) | Overhead |
|---|---|---|---|
| 8 | 1,240 | 1,380 | 10.1% |
| 16 | 2,410 | 2,680 | 10.1% |
| 32 | 4,630 | 5,100 | 9.2% |
| 64 | 8,590 | 9,440 | 9.0% |
| 128 | 14,200 | 15,800 | 10.1% |

The overhead is remarkably stable at approximately 10% regardless of batch size — the BGMV kernel scales proportionally with the base model compute. This is the right property for a production system: the cost of adapter diversity is a fixed tax, not something that compounds.

### Relationship to vLLM

Both S-LoRA and Punica have influenced the direction of vLLM's multi-LoRA implementation. The sliced matrix multiplication approach in current vLLM is architecturally derived from Punica's BGMV design, and the CPU offload mechanism with `--max-cpu-loras` reflects S-LoRA's unified paging ideas. For new production deployments, vLLM is the practical choice — it is actively maintained by the vLLM team and Anyscale, has extensive testing infrastructure, and integrates cleanly with the broader LLM serving ecosystem. S-LoRA and Punica are research systems that are valuable for understanding the design space.

![S-LoRA vs Punica vs vLLM native multi-LoRA: feature and performance comparison across three dimensions](/imgs/blogs/multi-lora-and-adapter-serving-5.png)

## 7. Adapter memory management in production

Let's get concrete about the memory accounting and management strategy for a production multi-LoRA deployment.

### GPU memory budget

For a single H100 80GB serving Llama-3-8B in BF16:

| Component | Memory | Notes |
|---|---|---|
| Base model weights (BF16) | ~16 GB | Frozen, shared across all customers |
| CUDA kernels + framework overhead | ~2 GB | PyTorch, NCCL, CUDA runtime |
| 4 on-GPU adapter slots (rank 64) | ~1.44 GB | ~360 MB per rank-64 adapter |
| KV cache (at 85% utilization) | ~52 GB | Available for in-flight requests |
| **Total** | ~71 GB | ~89% of 80 GB |

The KV cache gets most of the available memory after weights and adapters. The trade-off between adapter slots and KV cache is zero-sum:
- Each rank-64 adapter slot costs ~360 MB
- Each vLLM KV cache block (16 tokens, 32 layers, 2 heads, BF16) costs ~4 MB
- Each additional adapter slot costs ~90 KV blocks, or the capacity for 1,440 additional context tokens in flight

For most workloads, 4–8 adapter slots on GPU is the right balance. Unless your traffic has many simultaneous cold adapter requests with unpredictable distribution, you don't need more GPU slots than the 90th percentile of distinct adapters active in any 10-second window.

### The LRU eviction policy in depth

When all GPU adapter slots are occupied and a new cold adapter request arrives, vLLM must choose which existing adapter to evict. The default policy is LRU (Least Recently Used): the adapter whose last request was farthest in the past gets evicted.

LRU is a good general policy for Zipfian workloads — it naturally keeps hot adapters in GPU memory. But it has a pathological case: if you have exactly `--max-loras` distinct adapters each sending requests at the same rate, LRU will evict an adapter on every request (since all adapters have equal recency after each round-robin cycle). This "cache thrashing" scenario degrades throughput significantly.

If you detect cache thrashing (GPU hit rate below 50%, with roughly equal adapter access frequency across all adapters), you have two options: increase `--max-loras` to fit more adapters on GPU simultaneously, or add request batching at the API gateway layer. Gateway-level batching accumulates requests for the same adapter over a 50–100 ms window and forwards them together, effectively reducing the inter-arrival gap between same-adapter requests and improving LRU hit rates. The second option adds gateway latency in exchange for higher GPU throughput — the latency/throughput trade-off on the SLO triangle that every serving decision comes back to.

### Adapter hot/cold classification

Track per-adapter request rates over a rolling 1-hour window. Classify adapters as:
- **Hot** (> 10 req/minute): keep on GPU permanently
- **Warm** (1–10 req/minute): keep in CPU RAM, prefetch on first request of each burst
- **Cold** (< 1 req/minute): fetch from adapter storage (NFS/S3) on demand

```python
# Production adapter tier manager
import time
from collections import defaultdict

class AdapterTierManager:
    def __init__(self, hot_threshold=10.0, warm_threshold=1.0):
        self.request_counts = defaultdict(list)  # adapter_id -> [timestamps]
        self.hot_threshold = hot_threshold    # req/min threshold for hot tier
        self.warm_threshold = warm_threshold  # req/min for warm tier
    
    def record_request(self, adapter_id: str):
        now = time.time()
        self.request_counts[adapter_id].append(now)
        # Prune timestamps older than 1 hour
        cutoff = now - 3600
        self.request_counts[adapter_id] = [
            t for t in self.request_counts[adapter_id] if t > cutoff
        ]
    
    def get_tier(self, adapter_id: str) -> str:
        counts = self.request_counts.get(adapter_id, [])
        now = time.time()
        recent = sum(1 for t in counts if t > now - 60)  # last 1 minute
        if recent >= self.hot_threshold:
            return "hot"
        elif recent >= self.warm_threshold:
            return "warm"
        return "cold"
    
    def get_hot_adapters(self) -> list[str]:
        return [aid for aid in self.request_counts if self.get_tier(aid) == "hot"]
```

At startup, load the historical request log from your monitoring system to pre-classify adapters. Pre-load the hot adapters' weight tensors into CPU RAM (or GPU) before the first request arrives.

### Prefetching strategy

The most effective production optimization for multi-LoRA is **proactive adapter prefetching**. Rather than waiting for a request to arrive and then loading the adapter, analyze the pending request queue and initiate CPU→GPU transfers in advance.

```python
# Prefetch adapters based on the incoming request queue
async def prefetch_adapters_from_queue(
    pending_requests: list,
    adapter_manager,
    lookahead_depth: int = 50,
):
    """Pre-load adapters for the next N pending requests to minimize cold loads."""
    upcoming_adapter_ids = set()
    for req in pending_requests[:lookahead_depth]:
        upcoming_adapter_ids.add(req.adapter_id)
    
    # Identify which of these aren't on GPU yet
    cold_adapters = [
        aid for aid in upcoming_adapter_ids
        if not adapter_manager.is_on_gpu(aid)
    ]
    
    # Initiate async transfers, prioritized by position in queue
    for adapter_id in cold_adapters:
        asyncio.create_task(
            adapter_manager.prefetch_to_gpu(adapter_id)
        )
```

For SaaS products with business-hours patterns, you can also do **predictive prefetching**: if customer A always sends requests between 9 AM and 5 PM EST, pre-load their adapter at 8:55 AM. This is simple to implement with a cron job or a background thread reading your request logs.

## 8. Dynamic adapter loading and versioning

One of the operational advantages of multi-LoRA serving that goes underappreciated: you can add new adapters and update existing ones **without restarting the server**. This transforms adapter deployment from a release event (requiring downtime or a rolling restart) into an operational action (a POST request).

### Adding a new adapter at runtime

vLLM's management API exposes adapter lifecycle endpoints:

```bash
# Register a new adapter with the running server
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "customer_c_finance",
    "lora_int_id": 3,
    "lora_local_path": "/adapters/customer_c/"
  }'

# List all currently registered adapters
curl http://localhost:8000/v1/models

# Unload an adapter (frees CPU and GPU memory)
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "customer_old_deprecated"}'
```

This enables a clean CI/CD pipeline for adapter deployment:

1. Customer submits fine-tuning job to your training platform
2. Training completes; adapter checkpoint uploaded to S3 or NFS
3. Training platform calls `POST /v1/load_lora_adapter` on all serving replicas
4. New adapter is immediately available for requests

No server restarts, no traffic disruption, no deployment windows.

### Adapter versioning pattern

When a customer re-trains their adapter (e.g., after adding new documents to their knowledge base), you need to cut over to the new version without interrupting in-flight requests that use the old version.

The safest pattern uses versioned adapter names:

```python
import asyncio
from vllm import AsyncLLMEngine
from vllm.lora.request import LoRARequest

# Routing table: customer ID -> current active adapter name
adapter_routing: dict[str, str] = {}

async def deploy_adapter_update(
    customer_id: str,
    new_adapter_path: str,
    new_version: int,
    engine: AsyncLLMEngine,
    drain_timeout_seconds: int = 30,
):
    """Zero-downtime adapter version cutover."""
    new_name = f"{customer_id}_v{new_version}"
    old_name = adapter_routing.get(customer_id)
    
    # Step 1: Load new adapter (co-exists with old version)
    await engine.add_lora(LoRARequest(
        lora_name=new_name,
        lora_int_id=abs(hash(new_name)) % 100000,
        lora_local_path=new_adapter_path,
    ))
    
    # Step 2: Update routing table — new requests go to new version
    adapter_routing[customer_id] = new_name
    
    # Step 3: Drain in-flight requests on old adapter
    # (A real implementation would track in-flight request count per adapter)
    await asyncio.sleep(drain_timeout_seconds)
    
    # Step 4: Unload old version to free memory slot
    if old_name:
        await engine.remove_lora(old_name)
    
    print(f"Customer {customer_id}: {old_name} -> {new_name} (complete)")

async def handle_request(customer_id: str, prompt: str, engine: AsyncLLMEngine):
    """Route request to the customer's current adapter version."""
    adapter_name = adapter_routing.get(customer_id)
    if adapter_name is None:
        raise ValueError(f"No adapter registered for customer {customer_id}")
    
    lora_request = LoRARequest(
        lora_name=adapter_name,
        lora_int_id=abs(hash(adapter_name)) % 100000,
        lora_local_path=adapter_paths[adapter_name],
    )
    
    async for output in engine.generate(prompt, sampling_params, lora_request=lora_request):
        yield output
```

**The versioning trap to avoid**: never reuse a `lora_int_id` for a different adapter's weights. vLLM checks for this and raises an error — but a subtle variant of the bug can occur if you unload version 1, then load version 2 with the same ID. The unload/reload cycle is fine; what you cannot do is have both versions loaded simultaneously with the same ID. Always use distinct IDs for distinct adapter weight tensors.

![Dynamic adapter loading lifecycle: from server start through version update with zero downtime](/imgs/blogs/multi-lora-and-adapter-serving-6.png)

## 9. Prompt templates per adapter

This is a subtle but practically important operational concern. Different fine-tunes may have been trained with different prompt formats, and mixing up the format at inference time silently degrades quality.

The common instruction-following formats your adapters may expect:

**Llama-3 instruct format** (used by Meta's Llama-3 chat fine-tunes):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**ChatML format** (OpenAI convention, used by Mistral-7B-Instruct and others):

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
```

**Alpaca format** (used by many early open-source fine-tunes):

```
### Instruction:
{user_input}

### Response:
```

If a model was fine-tuned on Alpaca format but receives ChatML-formatted input, the model hasn't seen those special tokens and will produce incoherent output. This is not a bug in vLLM — it is a consequence of how instruction fine-tuning works. The prompt template is **part of the adapter's specification**.

```python
# Per-adapter template registry — store this alongside adapter metadata
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class AdapterMetadata:
    adapter_path: str
    prompt_format: str  # "llama3", "chatml", "alpaca"
    system_prompt: Optional[str]
    created_at: str
    base_model: str

def apply_llama3_template(system: str, user: str) -> str:
    parts = ["<|begin_of_text|>"]
    if system:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>")
    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return "".join(parts)

def apply_alpaca_template(system: str, user: str) -> str:
    if system:
        return f"### System:\n{system}\n\n### Instruction:\n{user}\n\n### Response:\n"
    return f"### Instruction:\n{user}\n\n### Response:\n"

TEMPLATE_FUNCTIONS = {
    "llama3": apply_llama3_template,
    "chatml": lambda s, u: (
        f"<|im_start|>system\n{s}<|im_end|>\n"
        f"<|im_start|>user\n{u}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ),
    "alpaca": apply_alpaca_template,
}

def format_request_for_adapter(
    customer_id: str,
    user_message: str,
    adapter_registry: dict[str, AdapterMetadata],
) -> str:
    adapter_name = adapter_routing[customer_id]
    meta = adapter_registry[adapter_name]
    template_fn = TEMPLATE_FUNCTIONS[meta.prompt_format]
    return template_fn(meta.system_prompt or "", user_message)
```

Store the `AdapterMetadata` (including prompt format) in your adapter registry alongside the weight files. When a fine-tuning job completes, the training pipeline should write the metadata file automatically. An `adapter_config.json` typically includes this information, and PEFT-based fine-tuning will record the base model name — you can use that to infer the expected format.

## 10. Worked examples and production sizing

#### Worked example: sizing a 100-customer SaaS legal AI platform

**Setup**: A B2B SaaS platform serving 100 law firm customers. Each customer has a fine-tuned Llama-3-8B adapter (rank 16), trained on their firm's document library and case history. Average load: 5 requests/minute per customer at peak hour = 500 req/min total = ~8.3 req/s. Average request: 512 input tokens (document excerpt), 256 output tokens (extracted clause or summary). SLA: p99 TTFT < 500 ms, p99 TPOT < 50 ms/token.

**Option A: 100 separate merged-model deployments (current state)**

Memory per deployment: ~16 GB (BF16 Llama-3-8B merged with adapter) → 100 × 16 GB = 1,600 GB.

AWS p4de.24xlarge provides 8 × A100 80GB = 640 GB. You need at minimum 3 instances (1,920 GB) just to hold all model weights, leaving only 320 GB of KV cache across all 100 "deployments" (3.2 GB per model — extremely tight for 512-token context). Realistic sizing: 100 A100s.

AWS on-demand cost for 100 A100s: approximately \$3.20/GPU-hour (A100 SXM4) → **\$320/hour**, or **\$7,680/day**.

GPU utilization: 8.3 req/s total / 100 GPUs = 0.083 req/s per GPU at ~1% utilization.

**Option B: multi-LoRA on 2× H100 80GB**

Memory per H100: 16 GB (base) + 0.5 GB (8 rank-16 adapter slots) + 63.5 GB (KV cache at 85%) = 80 GB. ✓

CPU RAM per host: 100 adapters × 60 MB = 6 GB — trivial.

For throughput: 2 H100s, each handling ~4.2 req/s (half the 8.3 total). A well-tuned H100 serving Llama-3-8B with continuous batching handles ~50–80 req/s comfortably. At 8.3 req/s, each H100 is under 20% load — one H100 would suffice, the second provides redundancy and headroom for peak.

AWS p4 instance with 2× H100 (or equivalent): approximately \$7.60/GPU-hour × 2 = **\$15.20/hour**, or **\$365/day**.

**Memory savings**: 1,600 GB → 32 GB = **50× reduction**
**Cost savings**: \$7,680/day → \$365/day = **21× reduction** (the savings are lower than memory because you still need some overhead per host, but the cost reduction is dramatic)

**Adapter cache analysis**:

Zipf distribution $s = 1.0$ over 100 adapters: the top-8 adapters (fitting in 8 GPU slots) handle approximately $\sum_{k=1}^{8} \frac{1/k}{\sum_{j=1}^{100} 1/j} \approx 56$% of requests. The GPU hit rate is ~56%. For the remaining 44% cold requests, loading from CPU RAM at 10 GB/s: 60 MB / 10 GB/s = 6 ms per adapter load. At 8.3 req/s with 44% cold rate: ~3.7 cold loads per second. Each takes 6 ms. With async loading and a request queue, these cold loads are absorbed without impacting p99 latency for other requests.


#### Worked example: evaluating multi-LoRA for a code intelligence service

**Setup**: An internal developer tools team with 12 language-specific code adapters (Python, Java, Go, TypeScript, Rust, C++, Ruby, PHP, Scala, Kotlin, Swift, C#). Peak load: 200 req/min = 3.3 req/s. Average request: 1024 input tokens (code context + natural language query), 512 output tokens (generated code). Hardware: 1× H100 80GB. Adapters are rank 32 (higher rank for better code performance).

**Memory analysis** with rank-32 adapters for Llama-3-70B in AWQ 4-bit:

Llama-3-70B AWQ: ~35 GB. 12 adapters at rank 32: 12 × ~120 MB = 1.44 GB. Total weights + adapters: 36.44 GB. KV cache available: 80 × 0.85 - 36.44 = 31.6 GB. Plenty of room.

All 12 adapters fit on GPU simultaneously at rank 32 — no CPU paging needed. Set `--max-loras 12 --max-lora-rank 32` and the system runs with zero cold adapter loads.

**Throughput impact**: 5–8% overhead for 12 GPU-resident adapters vs single-adapter serving. At 3.3 req/s arrival rate, a single H100 with Llama-3-70B AWQ handles approximately 15–20 req/s — headroom is 4–5×.

**Latency analysis**: with no cold adapter loads and generous KV cache:
- TTFT: dominated by prompt processing time (~50–80 ms for 1024 tokens on H100 with Llama-3-70B AWQ)
- TPOT: ~20–30 ms/token

Both are well within SLA for a developer assistance tool where humans tolerate 200–500 ms TTFT.

**Conclusion**: trivial multi-LoRA case — all adapters fit on GPU, headroom is large, just add `--enable-lora --max-loras 12` to the vLLM launch command.

**What if adapters were rank 128?** 12 × ~480 MB = 5.76 GB. Still fits: 35 + 5.76 = 40.76 GB, with 27.4 GB for KV cache. The savings vs 12 merged deployments: 12 × 35 GB = 420 GB vs 40.76 GB = **10× savings** — still compelling.

## 11. LoRA variants and adapter compatibility

Not all "LoRA-like" adapters are interchangeable in a multi-LoRA serving setup. Several variants of the core LoRA algorithm have emerged, and understanding their compatibility with vLLM's implementation matters for production deployments.

### DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Liu et al., 2024) decomposes the pretrained weight $W$ into a magnitude component $m$ and a directional component $V = W / \|W\|$, then applies LoRA to the directional component only:

$$W_{\text{DoRA}} = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|}$$

DoRA achieves better task performance than vanilla LoRA at the same rank, particularly for fine-tuning large models on complex instruction-following tasks. However, the normalization operation $\|V + \Delta V\|$ introduces per-token computation that is more complex than the simple $\Delta W = BA$ of vanilla LoRA. vLLM's multi-LoRA implementation supports DoRA adapters (PEFT 0.11.0+), but the heterogeneous-batch kernel paths are more complex than for vanilla LoRA.

### LoftQ: LoRA with Quantized Base Model

When the base model is quantized (e.g., AWQ 4-bit), the weight matrices stored in GPU memory are in INT4 format, not BF16. LoftQ (Li et al., 2023) introduces a fine-tuning approach that jointly optimizes the quantization and the LoRA initialization to minimize quantization-induced error. In multi-LoRA serving, you can use a 4-bit quantized base model with BF16 LoRA adapters — the delta computation happens in BF16, and the result is added back to the dequantized base weight.

This is a practical serving pattern because it lets you fit a larger base model (Llama-3-70B AWQ in 35 GB instead of 140 GB) while still supporting customer-specific adapters. vLLM supports this combination:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-70B-Instruct \
  --quantization awq \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.90 \
  --dtype auto  # uses AWQ's mixed precision
```

At AWQ 4-bit, Llama-3-70B occupies ~35 GB on a single H100 80GB, leaving ~35 GB for KV cache and adapter slots — enough to run a multi-customer serving setup on a model that would otherwise require 4 GPUs in BF16.

### Adapter format compatibility

vLLM's multi-LoRA loader expects adapters in the HuggingFace PEFT format:

```
/adapters/customer_a/
├── adapter_config.json      # LoRA hyperparameters: r, alpha, target_modules, base_model_name_or_path
└── adapter_model.safetensors  # Actual A and B matrices (or adapter_model.bin for older adapters)
```

The `adapter_config.json` must match the base model loaded in vLLM:

```json
{
  "base_model_name_or_path": "meta-llama/Llama-3-8B-Instruct",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "task_type": "CAUSAL_LM"
}
```

vLLM validates that the `base_model_name_or_path` matches the loaded model at adapter registration time. If they don't match (e.g., someone accidentally registers an adapter trained on Llama-2-7B against a Llama-3-8B server), vLLM raises an error rather than silently serving incorrect output.

### When to use QLoRA adapters

QLoRA (Dettmers et al., 2023) fine-tunes with the base model quantized to 4-bit (NF4) to reduce training memory requirements. The resulting adapter is a standard LoRA adapter in BF16 — the quantization was only used during training, not stored in the adapter file. QLoRA adapters are fully compatible with vLLM's multi-LoRA serving. You get the training memory efficiency of QLoRA plus the inference efficiency of multi-LoRA serving — a combination that makes fine-tuning accessible on consumer GPUs (A6000, 3090) while serving on datacenter GPUs (H100, A100).

## 12. Dynamic serving vs merged adapters

The alternative to dynamic multi-LoRA serving is merging the adapter into the base model weights permanently. Each strategy has its domain.

### Merging with PEFT

```python
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model on CPU to avoid OOM during merge
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load adapter
model_with_adapter = PeftModel.from_pretrained(
    base_model,
    "/adapters/customer_a/",
    is_trainable=False,
)

# Merge: W_eff = W_pretrained + (alpha/r) * B @ A
# This operation is in-place on base_model's weights
merged_model = model_with_adapter.merge_and_unload()

# Save as a standard HuggingFace model (no longer has PEFT structure)
merged_model.save_pretrained(
    "/merged_models/customer_a_merged/",
    safe_serialization=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer.save_pretrained("/merged_models/customer_a_merged/")
```

After merging, the model is a standard HuggingFace model with no PEFT structure. It can be served by any inference framework (vLLM, TGI, TorchServe, Triton) without any LoRA-specific configuration. The merged weights are identical to what the adapter would have produced at inference time — it is mathematically lossless.

The merged model has the same size as the base model (~16 GB for Llama-3-8B BF16). No adapter file is needed. No runtime delta computation. The downside: you've lost the adapter identity. You can't "un-merge" to get the adapter back. And you can't serve multiple customers from the same merged model.

### The decision framework

| Condition | Recommendation |
|---|---|
| 1 adapter, high QPS (> 100 req/s) | Merge — eliminate 5-10% overhead |
| 2–4 adapters, balanced traffic | vLLM native, `--max-loras 4` |
| 5–32 adapters, Zipfian access | vLLM native, all on GPU |
| 33–200 adapters, Zipfian | vLLM + `--max-cpu-loras 200` |
| 200+ adapters | S-LoRA-style unified paging |
| Adapters updated frequently | Dynamic serving — avoids re-merge |
| Hard per-tenant isolation | Separate deployments |
| Adapters rank 64+ and very large | Evaluate case by case |

![Merged adapter vs dynamic LoRA serving: trade-offs across adapter count, overhead, and use case](/imgs/blogs/multi-lora-and-adapter-serving-7.png)

## 12. When NOT to use multi-LoRA serving

Multi-LoRA serving is compelling but not universally correct. These are the concrete scenarios where a different approach is better.

### Hard tenant isolation requirements

Multi-LoRA serving runs all customers' requests through the same GPU and the same memory space. In the standard threat model, there is no practical mechanism for one customer to observe another customer's activations — the sliced matmul approach processes activations independently per request, and GPU memory is not shared between requests' working sets.

However, some enterprise customers have contractual or regulatory requirements that prohibit hardware co-tenancy entirely — particularly in healthcare (HIPAA business associate agreements), financial services (SOC 2 Type II attestations that specify compute isolation), and government (FedRAMP requirements for dedicated tenancy). If a customer's contract specifies that their data must never touch hardware used by any other customer, multi-LoRA serving is not compliant regardless of its practical security properties.

In these cases, options include:
- Dedicated GPU instances per customer (maximum isolation, maximum cost)
- MIG (Multi-Instance GPU) partitioning — each customer gets a hardware-isolated GPU partition with dedicated compute engines and memory (see [GPU scheduling and MIG](/blog/machine-learning/model-serving/gpu-scheduling-and-mig))
- Confidential computing instances with GPU attestation (available on select cloud providers)

### Very high adapter rank (rank 128+) with full weight coverage

The memory savings from multi-LoRA scale inversely with adapter rank. At rank 16, a Llama-3-8B adapter is ~60 MB versus 16 GB for the base — a 267× size ratio. At rank 128 with full Q/K/V/O/MLP coverage, the adapter is ~960 MB — still a 17× size ratio. The savings are real, but:

- Each GPU adapter slot costs ~960 MB instead of ~60 MB
- `--max-loras 4` consumes ~3.84 GB of GPU memory just for adapter slots
- The compute overhead of applying high-rank deltas is larger (more FLOPs per token)

More importantly, fine-tuning at rank 128 with full weight coverage suggests the task may require substantial weight changes. If you're using rank 128 across all weight matrices, you're fine-tuning roughly 4% of the total parameter count — at that point, you should evaluate whether a smaller dedicated model or a different fine-tuning approach (DoRA, full fine-tuning with FSDP) might be more appropriate.

### Adapters from different base model checkpoints

Multi-LoRA serving requires all adapters to be mathematically compatible with the same base model checkpoint. An adapter trained on Llama-3-8B-Instruct cannot be applied to Llama-3-8B-Base (the weight matrices have different values, so the trained deltas don't correspond to the right gradient direction). An adapter trained on an older checkpoint of Llama-3-8B-Instruct may produce degraded outputs when applied to a newer checkpoint that received additional fine-tuning.

If your customer base has adapters trained on multiple base model versions — because they fine-tuned at different times as you upgraded your base model — you need separate multi-LoRA serving pools, one per base model checkpoint. This is operationally manageable but requires explicit routing to the correct pool.

### Adapter cold-start latency is in your critical SLA path

For most workloads, cold adapter loads (from CPU RAM to GPU) take 3–10 ms and are hidden by the request queue. But if your SLA requires p99 TTFT < 20 ms and your adapter access distribution is highly uniform (all adapters equally likely, no hot adapters), cold loads will routinely exceed your latency budget.

Before assuming cold loads are invisible, measure your actual adapter access distribution. If the top-8 adapters handle > 70% of traffic (typical Zipfian pattern), cold loads are a tail event that barely affects p99. If your traffic is uniform across all adapters, you have a fundamental conflict between multi-LoRA serving and your SLA — you'll need either more GPU slots or a different architecture.

### The base model changes more frequently than adapter fine-tuning

If you're doing continuous pre-training or frequently updating the base model checkpoint, all existing customer adapters become stale when the base model changes. An adapter trained on checkpoint $t_0$ applied to checkpoint $t_1$ (which has different weight values at the trained positions) produces suboptimal output — the adapter's learned delta no longer points in the optimal direction for the updated base model.

If your base model update cadence is faster than your customers' adapter re-training cadence, you'll accumulate stale adapters. For this case, either slow your base model update cadence, invest in adapter migration tooling, or accept that not all customers will always be on the latest base model.

## 13. Case studies and benchmarks

### S-LoRA paper results (NeurIPS 2023)

The S-LoRA paper tested on a single A100 80GB with LLaMA-7B as the base model. Synthetic Poisson arrivals with Zipfian adapter distribution ($s = 1.0$).

**Adapter count vs throughput degradation:**

| Adapters | Throughput (tokens/s) | vs single-adapter |
|---|---|---|
| 1 (single-adapter vLLM) | 10,100 | 1.00× |
| 10 (S-LoRA) | 9,494 | 0.94× |
| 100 (S-LoRA) | 9,191 | 0.91× |
| 1,000 (S-LoRA) | 8,787 | 0.87× |

At 1,000 concurrent adapters with 13% throughput overhead, S-LoRA demonstrated the key result: adapter count does not have to scale linearly with GPU memory or throughput cost. The Zipfian access pattern means most requests hit the small hot adapter set, and cold loads are rare enough to be absorbed.

**Memory usage at 1,000 adapters** (rank 16, LLaMA-7B):
- GPU memory for adapters: 0.48 GB (8 GPU slots × 60 MB per adapter)
- CPU RAM for adapter cache: 60 GB (1,000 × 60 MB)
- GPU memory for KV cache: 57.5 GB (82% of A100 after weights and adapter slots)

**Latency at 50 req/s** (mixed adapter workload):
- p50 TTFT: 83 ms
- p99 TTFT: 312 ms (cold adapter loads visible at tail)
- p99 TPOT: 38 ms/token

### vLLM multi-LoRA performance characterization

From vLLM's official documentation and benchmarks (A100 40GB, LLaMA-2-7B, rank 16):

| GPU adapter slots (`--max-loras`) | Throughput vs single | GPU memory for adapters |
|---|---|---|
| 1 | 1.00× | ~60 MB |
| 2 | 0.97× | ~120 MB |
| 4 | 0.93× | ~240 MB |
| 8 | 0.88× | ~480 MB |

The throughput degradation comes from two sources: (1) the delta computation overhead (~5%), and (2) reduced KV cache budget (more GPU memory used for adapter slots → fewer in-flight tokens → lower batch sizes in the scheduler).

### AIBrix multi-LoRA production results (ByteDance, 2024)

ByteDance's AIBrix inference framework reported the following from production multi-LoRA deployment serving a Llama-2-based model family:

- 200+ adapters served from a single pool of 16 A100 GPUs
- 16 GPU adapter slots per A100, all hot adapters GPU-resident
- Adapter cache hit rate: 78% GPU-resident, 18% CPU-cached, 4% cold (from NFS)
- p99 additional latency for cold (NFS) loads: 45–90 ms
- GPU utilization improvement: from 21% (naive separate deployments) to 74% (multi-LoRA)
- Cost reduction per 1M tokens served: ~\$0.0021 → \$0.0006 (3.5× reduction)

The 3.5× cost reduction per token comes from the combination of higher GPU utilization (more tokens per GPU-hour) and fewer required GPUs (lower fleet size). These numbers are consistent with the theoretical expectation: the memory savings are proportional to adapter rank relative to model size, and the utilization gains come from packing more customers' work into the same scheduling window — which is the core value proposition of continuous batching applied to adapter-diverse workloads.

**What went wrong in early attempts**: ByteDance noted that their initial multi-LoRA deployment had poor GPU hit rates (~40%) because the adapter registry was not pre-warmed at server startup. On every restart — rolling deployment, node failure recovery — all 200 adapters were cold. The fix was a startup warming routine that prefetches the top-32 adapters (by 7-day traffic volume) into CPU RAM before accepting traffic. After the fix, cold starts show GPU hit rate ~78% within 30 seconds of becoming ready — the hot adapters are in CPU RAM and can be promoted to GPU within the first few requests. This startup warming pattern should be considered standard practice for any multi-LoRA deployment with more than ~20 adapters.

## 14. When to use this (and when not to)

**Use multi-LoRA serving when:**

- You have 3 or more customers or task variants requiring fine-tuned versions of the same base model
- All adapters are LoRA-based (rank ≤ 128) trained on compatible base model checkpoints
- Your adapter access distribution is Zipfian (top-N adapters handle the majority of traffic)
- Customer isolation requirements allow hardware co-tenancy
- You need to add or update adapters without server restarts
- GPU cost is a significant budget line and you're running at low per-GPU utilization

**Use merged adapters when:**

- You have exactly one fine-tuned model variant with high, sustained traffic
- You need absolute maximum throughput (zero adapter overhead)
- Adapter versioning and dynamic loading complexity isn't worth the engineering investment for your scale

**Use separate model deployments when:**

- Customers have contractual requirements for physical hardware isolation
- Adapters are trained on different base model checkpoints (different model families or incompatible versions)
- Per-customer SLA guarantees require dedicated compute capacity
- Adapters are very large (>2 GB each) and memory savings are marginal

**Do NOT use multi-LoRA when:**

- Your p99 TTFT SLA is under 20 ms AND adapter access is highly uniform (all adapters equally cold) — cold loads will blow the SLA
- Adapters are full fine-tunes, not LoRA — there's no shared base to exploit
- Your base model updates faster than customers can retrain adapters — stale adapters will silently degrade output quality

![Adapter serving decision tree: choose between merged, vLLM native multi-LoRA, S-LoRA-style offload, or separate deployments](/imgs/blogs/multi-lora-and-adapter-serving-8.png)

## 15. Production deployment architecture

Putting multi-LoRA serving into a real production environment requires more than just the right vLLM flags. You need an API gateway layer for routing, a Kubernetes deployment strategy, and observability instrumentation that can tell you whether your adapter cache is performing well.

### Kubernetes deployment for multi-LoRA

```yaml
# kubernetes/multi-lora-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-lora-inference
  namespace: ml-serving
  labels:
    app: multi-lora-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: multi-lora-inference
  template:
    metadata:
      labels:
        app: multi-lora-inference
    spec:
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-H100-80GB-HBM3"
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.4.2
        args:
        - "--model=meta-llama/Llama-3-8B-Instruct"
        - "--enable-lora"
        - "--max-loras=8"
        - "--max-lora-rank=64"
        - "--max-cpu-loras=100"
        - "--lora-extra-vocab-size=256"
        - "--gpu-memory-utilization=0.88"
        - "--max-model-len=4096"
        - "--dtype=bfloat16"
        - "--served-model-name=llama3-8b-multi-lora"
        - "--port=8000"
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        - name: VLLM_WORKER_MULTIPROC_METHOD
          value: "spawn"
        resources:
          requests:
            memory: "64Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: "1"
        ports:
        - containerPort: 8000
          name: http
        volumeMounts:
        - name: adapter-storage
          mountPath: /adapters
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: adapter-storage
        persistentVolumeClaim:
          claimName: adapter-nfs-pvc  # NFS PVC containing all adapter checkpoints
apiVersion: v1
kind: Service
metadata:
  name: multi-lora-inference-svc
  namespace: ml-serving
spec:
  selector:
    app: multi-lora-inference
  ports:
  - port: 80
    targetPort: 8000
    name: http
  type: ClusterIP
```

The NFS-backed PVC for adapter storage is important: it lets you add new adapter checkpoints by copying files to the NFS volume, making them immediately available to all serving pods without rebuilding container images. Adapters are not embedded in the container image — they live in shared storage.

### API gateway for adapter routing

In production, you don't want your clients to know about adapter paths or IDs. Those are internal implementation details. Instead, expose a clean customer-scoped API:

```python
# FastAPI gateway that translates customer identity to adapter routing
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
import httpx
import json
from typing import AsyncIterator

app = FastAPI(title="Multi-LoRA API Gateway")

# In production: load from database or config service
CUSTOMER_ADAPTER_MAP = {
    "cust_abc123": {
        "adapter_name": "customer_abc_legal_v3",
        "adapter_id": 101,
        "adapter_path": "/adapters/cust_abc123/v3/",
        "template": "llama3",
        "system_prompt": "You are a legal document analyzer specializing in contract review.",
    },
    "cust_def456": {
        "adapter_name": "customer_def_marketing_v2",
        "adapter_id": 102,
        "adapter_path": "/adapters/cust_def456/v2/",
        "template": "llama3",
        "system_prompt": "You are a creative marketing copywriter.",
    },
}

VLLM_BACKEND_URL = "http://multi-lora-inference-svc/v1"

def get_customer_id(request: Request) -> str:
    # In production: extract from JWT, API key lookup, etc.
    customer_id = request.headers.get("X-Customer-ID")
    if not customer_id or customer_id not in CUSTOMER_ADAPTER_MAP:
        raise HTTPException(status_code=401, detail="Invalid customer credentials")
    return customer_id

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    customer_id: str = Depends(get_customer_id),
):
    body = await request.json()
    customer_config = CUSTOMER_ADAPTER_MAP[customer_id]
    
    # Apply customer's prompt template to the conversation
    messages = body.get("messages", [])
    formatted_prompt = apply_template(
        template=customer_config["template"],
        system=customer_config["system_prompt"],
        messages=messages,
    )
    
    # Inject the LoRA request routing info
    vllm_request = {
        **body,
        "model": "llama3-8b-multi-lora",
        "messages": messages,
        "extra_body": {
            "lora_request": {
                "lora_name": customer_config["adapter_name"],
                "lora_int_id": customer_config["adapter_id"],
                "lora_local_path": customer_config["adapter_path"],
            }
        },
    }
    
    # Forward to vLLM, stream response back
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{VLLM_BACKEND_URL}/chat/completions",
            json=vllm_request,
        ) as resp:
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=await resp.aread())
            
            async def stream_response() -> AsyncIterator[bytes]:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers=dict(resp.headers),
            )
```

This gateway pattern cleanly separates customer identity (the JWT or API key) from adapter routing (the internal adapter name and path). When a customer's adapter is updated, only the `CUSTOMER_ADAPTER_MAP` entry changes — clients see no difference in their API contract.

### Observability: tracking adapter cache performance

The most important metric to instrument in multi-LoRA serving is **adapter cache hit rate**. A low hit rate means frequent cold loads, which inflate your TTFT p99. A high hit rate means the GPU adapter slots are well-utilized.

```python
# Prometheus metrics for adapter cache tracking
from prometheus_client import Counter, Histogram, Gauge
import time

adapter_requests_total = Counter(
    "adapter_requests_total",
    "Total requests per adapter",
    labelnames=["adapter_name", "customer_id"],
)

adapter_cache_hits = Counter(
    "adapter_cache_hits_total",
    "Adapter cache hits (request found adapter on GPU)",
    labelnames=["adapter_name", "tier"],  # tier: gpu, cpu, storage
)

adapter_load_duration = Histogram(
    "adapter_load_duration_seconds",
    "Time to load adapter from CPU/storage to GPU",
    labelnames=["source_tier"],  # cpu or storage
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

gpu_adapters_loaded = Gauge(
    "gpu_adapters_loaded",
    "Number of adapter weight sets currently on GPU",
)

# Alert: adapter cache hit rate below threshold
# PromQL alert rule:
# rate(adapter_cache_hits_total{tier="gpu"}[5m]) /
# rate(adapter_requests_total[5m]) < 0.5
# → fire if GPU hit rate below 50% for 5 minutes
```

A healthy multi-LoRA deployment should show:
- GPU adapter hit rate > 60% (indicates hot adapters are being retained on GPU)
- CPU adapter hit rate > 95% (all adapters are in CPU RAM; no storage fetches)
- Storage fetch rate ≈ 0 (never loading from NFS in steady state)
- p99 TTFT within SLA for all adapter tiers

If you see GPU hit rate dropping (perhaps a new customer with high traffic is displacing existing hot adapters), increase `--max-loras` or increase the number of GPU replicas.

### GPU fleet sizing formula

For a fleet serving $C$ customers with adapters at rank $r$ on a base model of size $M$ GB, using hardware with $G$ GB of VRAM per GPU:

$$N_{\text{GPUs}} = \left\lceil \frac{M + S_a \cdot \text{hot\_slots} + \text{KV target}}{G \cdot \text{util}} \right\rceil \cdot \text{replicas}$$

Where:
- $S_a$ = size of one adapter in GB (e.g., 0.36 GB for rank-64 Llama-3-8B)
- $\text{hot\_slots}$ = number of adapters to keep on GPU simultaneously
- $\text{KV target}$ = desired KV cache capacity in GB (determines max concurrent context)
- $\text{util}$ = GPU memory utilization fraction (typically 0.85–0.90)
- $\text{replicas}$ = number of instances for redundancy (typically 2)

For the 100-customer example on H100 80GB: $M = 16$ GB, $S_a = 0.36$ GB, hot\_slots $= 8$, KV target $= 40$ GB, util $= 0.88$:

$$\text{Required per instance} = 16 + (0.36 \times 8) + 40 = 58.88 \text{ GB}$$
$$\text{H100 80GB at 88\% util} = 70.4 \text{ GB available} \rightarrow \text{fits}$$

One H100 per replica × 2 replicas = 2 H100s. This matches the estimate from the worked example, confirming the formula.

## 16. Key takeaways

1. **The base model is the shared infrastructure.** LoRA adapters are tiny deltas — a rank-16 adapter for a 7B model is ~60–100 MB, versus 16 GB for the full model. Sharing the base model across all customers eliminates 99%+ of redundant GPU memory in a naive multi-deployment setup. The memory savings compound with adapter count: 100 customers with rank-16 adapters cost 16 GB + 6 GB versus 1,600 GB separately.

2. **vLLM's multi-LoRA implementation is production-ready.** Launch with `--enable-lora --max-loras 4-8 --max-lora-rank 64 --max-cpu-loras 100` as a starting configuration. The 5–10% throughput overhead buys a 10–50× memory reduction.

3. **Adapter slots and KV cache are zero-sum on GPU.** Each rank-64 adapter slot consumes ~360 MB that would otherwise hold KV cache pages. Tune `--max-loras` to the 90th percentile of distinct adapters active in any 10-second window, not to the total catalog size.

4. **S-LoRA's unified paging enables 1,000+ adapters on one GPU.** The key: CPU RAM holds the full adapter catalog, GPU holds only the hot adapters, and both KV pages and adapter weights are evicted from the same memory pool. The CPU offload adds ~10–15% overhead at 1,000 adapters.

5. **The BGMV kernel makes heterogeneous-batch compute efficient.** Punica's single-kernel-launch approach for applying different adapters to different rows of the activation matrix is 5–10× faster than naive sequential per-adapter application, and adds only ~10% overhead relative to single-adapter serving.

6. **Dynamic adapter loading decouples fine-tuning deployment from serving deployment.** The management API pattern (load/unload adapters at runtime) means a new customer fine-tune ships with a `POST /v1/load_lora_adapter` call, not a server restart.

7. **Prompt templates are part of the adapter contract.** An adapter trained on Alpaca format fed ChatML input will produce garbage. Store the expected template format alongside each adapter's metadata and enforce it in your API gateway.

8. **The Zipfian access pattern is your friend.** Most real SaaS workloads have a small number of high-traffic adapters and a long tail of low-traffic ones. This means a small GPU adapter cache captures most traffic, and cold loads are rare tail events, not the common case.

9. **Hard isolation requirements break the shared-model assumption.** If a customer requires physical GPU isolation, evaluate MIG partitioning or dedicated instances. Multi-LoRA is not appropriate for co-tenancy-prohibited workloads regardless of its practical security properties.

10. **Profile before tuning.** The optimal `--max-loras`, CPU cache size, and prefetch strategy all depend on your actual traffic distribution. Instrument your adapter access patterns with per-adapter request counters before spending engineering time on hot/warm/cold tiering. The Zipfian distribution is common but not universal — validate it empirically for your workload before assuming LRU cache performance will be good.

## Further reading

- **S-LoRA paper** — Sheng, Y., Cao, S., Li, D., et al. (2023). "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." NeurIPS 2023 workshops. The primary reference for scalable multi-LoRA architecture and unified paging.
- **Punica paper** — Chen, L., Ye, Z., Wu, Y., et al. (2023). "Punica: Multi-Tenant LoRA Serving." Proceedings of MLSys 2024. Detailed BGMV kernel design, performance analysis, and multi-tenant benchmarks.
- **LoRA original paper** — Hu, E., Shen, Y., Wallis, P., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. The theoretical foundation for the rank decomposition and initialization conventions.
- **vLLM LoRA documentation** — Official vLLM docs: Using LoRA Adapters. Current flags, API reference, configuration examples, and supported adapter formats.
- **Series sibling: Continuous batching and PagedAttention** — [/blog/machine-learning/model-serving/continuous-batching-and-pagedattention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the KV cache paging and scheduling primitives that multi-LoRA extends.
- **Series sibling: Quantization for LLM serving** — [/blog/machine-learning/model-serving/quantization-for-llm-serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) — reducing base model size with GPTQ/AWQ/FP8, composable with multi-LoRA to fit larger models alongside adapter caches.
- **Series intro** — [/blog/machine-learning/model-serving/what-is-model-serving](/blog/machine-learning/model-serving/what-is-model-serving) — the latency/throughput/cost SLO triangle that frames every trade-off in this series.
- **Series capstone** — [/blog/machine-learning/model-serving/the-model-serving-playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — full decision tree from model checkpoint to production serving architecture.

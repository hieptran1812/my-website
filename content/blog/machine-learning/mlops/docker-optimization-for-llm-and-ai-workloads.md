---
title: "Docker for LLM & AI Workloads: A Senior Engineer's Optimization Playbook"
date: "2026-04-28"
description: "A deep, practical guide to Docker for LLM training and inference — image layering, BuildKit, GPU runtime, NCCL, vLLM/Triton, multi-stage builds, and a catalog of real production case studies."
tags:
  [
    "Docker",
    "LLM",
    "MLOps",
    "GPU",
    "Inference",
    "vLLM",
    "Triton",
    "Kubernetes",
    "BuildKit",
    "NVIDIA",
  ]
category: "mlops"
author: "Hiep Tran"
featured: true
readTime: 40
---

# Docker for LLM & AI Workloads: A Senior Engineer's Optimization Playbook

Most "Docker for ML" tutorials stop at `FROM nvidia/cuda` and `pip install torch`. That works on your laptop and falls apart the moment you ship a 70B model to a multi-GPU node behind a load balancer. This article is the playbook I wish someone had handed me three years ago: what actually matters when containerizing LLM training and inference, why it matters, and a long catalog of failure modes you only learn by being paged at 2 AM.

![Docker LLM Optimization Architecture](/imgs/blogs/docker-llm-optimization-architecture.png)

The diagram above is the mental model: four layers (host, image, runtime, serving), each with its own optimization surface. Get any one wrong and the others cannot compensate. The rest of this article walks through each layer, then closes with twelve detailed case studies of real production incidents and the way a senior engineer should reason about them.

## 1. Why Docker for AI Is Different

Containers were designed for stateless 12-factor web apps: small images, no persistent state, ephemeral processes. LLM workloads violate every assumption.

| Assumption            | Web App   | LLM Workload                                                 |
| --------------------- | --------- | ------------------------------------------------------------ |
| Image size            | 50–500 MB | 5–30 GB (CUDA + PyTorch + flash-attn + Triton kernels)       |
| Process state         | Stateless | 70 GB of model weights pinned in VRAM                        |
| IPC                   | None      | NCCL all-reduce across 8 GPUs per node                       |
| Shared memory         | Unused    | Required for PyTorch DataLoader workers and TP workers       |
| Boot time             | <1s       | 30s to 5min (weight loading dominates)                       |
| Networking            | HTTP only | RDMA, GPUDirect, NVLink topology aware                       |
| Storage               | Logs only | Multi-TB checkpoints, dataset shards                         |
| Isolation expectation | Strong    | Often relaxed (`--ipc=host`, `--privileged`) for performance |

Treating an LLM container like a Flask app is the root cause of perhaps eighty percent of the production issues I have personally debugged. Everything below follows from this mismatch. When you find yourself reaching for an unusual flag, ask: _which of these table rows am I servicing?_ That question alone keeps the design honest.

## 2. The Host Layer: What Stays Outside the Container

A senior rule of thumb: **do not virtualize what cannot be virtualized cleanly.** The NVIDIA driver, kernel modules, and device files belong on the host. The container brings the userspace CUDA libraries.

```
Host (bare metal / VM)
├── nvidia-driver (e.g. 550.x)         ← kernel module, must match GPU silicon
├── nvidia-container-toolkit            ← injects /dev/nvidia*, libs, into containers
├── containerd / runc                   ← OCI runtime
└── cgroups v2 + systemd                ← required for proper GPU resource isolation
```

The CUDA _toolkit_ version inside the image must be ≤ the driver's max supported CUDA. The driver is forward-compatible; the toolkit is not backward-compatible. A `nvidia/cuda:12.4` image works on a driver supporting 12.4 _or newer_. Reverse breaks at runtime with a cryptic `CUDA driver version is insufficient`.

```bash
# Verify the host once
nvidia-smi                                   # driver + max CUDA
nvidia-container-cli info                    # what the toolkit will inject
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```

If `nvidia-smi` works on the host but fails inside the container, ninety-five percent of the time it is the toolkit not being registered with your runtime. On containerd:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

There is a second, less-discussed concern: **driver upgrades require coordination.** A driver bump requires a reboot or a `nvidia-smi --reset` of every GPU on the host, which means draining workloads first. In a fleet of 200 nodes, you do not want to do this manually. Bake it into your node maintenance loop with cordon → drain → upgrade → uncordon, and version-pin the driver via `apt-mark hold nvidia-driver-*` so a runaway `apt upgrade` cannot break a hundred training jobs at once.

A third concern: **CUDA forward compatibility packages**. NVIDIA ships a `cuda-compat` package that lets newer CUDA toolkits run on older drivers (within limits). This is occasionally useful when you are stuck on driver 525 but need CUDA 12.4 for a new wheel. Install `cuda-compat-12-4` on the host and set `LD_LIBRARY_PATH` inside the container. It is a band-aid; plan a real driver upgrade.

## 3. Image Engineering: Layer Order Is a Performance Decision

Docker layers are content-addressed and cached top-down. Every line in your Dockerfile that changes invalidates everything below it. For ML images this matters enormously because the layers below are the expensive ones: multi-GB downloads of `torch`, `flash-attn`, `xformers`, `vllm`.

**Order layers from coldest (rarely changes) to hottest (changes every commit).** That single discipline is responsible for turning fifteen-minute CI builds into ninety-second cache hits.

```dockerfile
# syntax=docker/dockerfile:1.7
# ---------- Stage 1: build ----------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# 1. System packages — pinned, rarely change
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git build-essential ninja-build \
        libaio-dev libnccl2 libnccl-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# 2. Python build deps (separate from runtime deps)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel setuptools

# 3. Heavy wheels first — change rarely
COPY requirements-heavy.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-heavy.txt
    # torch, flash-attn, vllm, transformers, xformers

# 4. Light deps — change more often
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

# Only userspace CUDA runtime — no nvcc, no headers (saves ~3 GB)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 libnccl2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages \
                    /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 5. App code last — changes every commit
WORKDIR /app
COPY ./src ./src

ENTRYPOINT ["python", "-m", "src.serve"]
```

### Why this is non-negotiable

The `runtime` versus `devel` choice alone is a three-to-five GB swing on your final image. `devel` ships `nvcc`, headers, static libs. You only need `devel` to _build_ C++/CUDA extensions like flash-attn. The final shipped image must use `runtime`. I have seen teams ship `devel` images for years simply because nobody knew the difference; cutting their image from 18 GB to 11 GB took an afternoon.

BuildKit cache mounts (`--mount=type=cache`) keep pip and apt caches across builds without bloating the layer. Re-installing `torch` becomes a two-second hash check instead of a four-GB download. The `sharing=locked` flag prevents two parallel CI jobs from corrupting the apt cache directory; on shared CI runners this matters.

Multi-stage builds let the final image inherit only the dist-packages directory, not the build toolchain. Typical savings: 8 GB to 4 GB. The trick people miss is that the runtime stage should be a _different base image_, not the same one. Otherwise you are paying for compilers you will never use.

`TORCH_CUDA_ARCH_LIST` controls which GPU architectures custom kernels (flash-attn, xformers, your own CUDA ops) are compiled for. Default is "all", which doubles compile time and image size. Set it to your actual fleet: `8.0` (A100), `8.6` (RTX 30/40), `9.0` (H100/H200), `10.0` (Blackwell). If your fleet is heterogeneous, list the architectures explicitly.

### Stop putting model weights in the image

I see this constantly. Someone does `COPY ./models /models` with a 14 GB Llama-3-8B safetensors checkpoint and wonders why CI pushes take twenty minutes and every node pulls 40 GB on first schedule. **Weights are data, not code.** Treat them that way.

| Strategy                                     | When to use                                    | Trade-off                                 |
| -------------------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| Mount as read-only volume                    | Single-node, weights on local NVMe             | Fast, but tied to one node                |
| Pull from S3/GCS at startup with `s5cmd`     | Cloud, weights in object storage               | Cold-start latency; need credentials      |
| HuggingFace cache volume (`HF_HOME=/models`) | Multi-tenant nodes sharing the same checkpoint | Coordination on cache eviction            |
| Init container in K8s                        | Need verification, decryption, warmup          | Two containers per pod                    |
| Bake into image                              | Tiny models (<1 GB), offline batch             | Image bloat                               |
| Sidecar model server                         | Fleet shares one weight host                   | Network hop; bandwidth becomes bottleneck |

For HF, set the cache once on the host and reuse:

```bash
docker run --rm --gpus all \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  vllm/vllm-openai:v0.7.0 \
  --model meta-llama/Llama-3.1-70B-Instruct
```

`HF_HUB_ENABLE_HF_TRANSFER=1` flips the downloader to a Rust implementation that saturates 10 Gbps NICs. The default Python downloader caps around 200 MB/s; a 140 GB checkpoint becomes a twelve-minute wait instead of four. On a fleet that auto-scales to ten replicas during a traffic spike, that is the difference between a graceful response and a queueing storm.

### A second-order optimization: model weight deduplication

If you serve multiple finetuned variants of the same base model (which is the common case), the base weights are identical across variants. Storing each variant as its own monolithic checkpoint multiplies storage by N. Two patterns help.

**LoRA-style deltas:** keep the base in the shared cache, ship adapters as small (50–200 MB) deltas applied at load time. vLLM, SGLang, and TGI all support this natively now. One container can serve dozens of finetunes from a single 140 GB base.

**Content-addressed safetensors:** chunk safetensors files at the tensor boundary and store them by SHA256. A new finetune that only modifies the LM head reuses ninety-nine percent of the base's blocks on disk. This is what HuggingFace Hub's xet backend does internally; you can replicate it with a custom loader. For a team running fifty finetunes, this turned a 7 TB cache into 200 GB.

## 4. Runtime: The Flags Nobody Mentions in Tutorials

The defaults of `docker run` are tuned for web apps and actively hostile to ML workloads. Here is the survival kit.

### `--shm-size`: the silent killer

Docker defaults `/dev/shm` to **64 MB**. PyTorch DataLoader workers communicate via shared memory; a single batch of image tensors blows past 64 MB instantly. Symptom: `RuntimeError: DataLoader worker (pid X) is killed by signal: Bus error` or, worse, silent hangs that look like a slow data pipeline.

```bash
docker run --shm-size=16g ...           # bare minimum for training
docker run --ipc=host ...               # alternative: full host IPC namespace
```

For multi-GPU NCCL, use `--ipc=host` _or_ enlarge `--shm-size` and set `NCCL_SHM_DISABLE=0`. Tensor-parallel inference in vLLM and SGLang relies on this for the inter-process worker handoff. A common mistake is to set `--shm-size=2g` thinking that is "plenty"; the workers happily fill 8 GB during a long-context request and OOM the shared memory segment.

### `--gpus`, MIG, and visibility

```bash
docker run --gpus all ...                                        # all GPUs
docker run --gpus '"device=0,3"' ...                             # specific GPUs
docker run --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 ...    # belt and suspenders
```

Use **MIG** (Multi-Instance GPU) on A100/H100 when you want to slice a single GPU into isolated partitions for multiple tenants. The container sees only its slice, with no noisy neighbor on memory bandwidth or SM scheduler.

```bash
sudo nvidia-smi mig -cgi 9,9,9 -C        # three 1g.10gb instances on H100
docker run --gpus '"device=MIG-<uuid>"' ...
```

Common MIG profiles on H100 (80 GB): `7g.80gb` (whole), `4g.40gb`, `3g.40gb`, `2g.20gb`, `1g.10gb`, `1g.20gb`. Pick based on your smallest deployable unit. If your worst-case model needs 18 GB of VRAM (e.g., a 7B model with 8K context), `2g.20gb` gives you three tenants per H100 with hardware-enforced isolation. The downside: MIG instances cannot do peer-to-peer NVLink, so anything tensor-parallel must live within one instance.

### Memory locking and pinned host memory

GPUDirect and pinned-memory transfers require unlimited locked memory. The default `memlock` is 64 KB; enough to make `cudaMallocHost` fail in obscure ways.

```bash
docker run \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=IPC_LOCK \
  ...
```

`IPC_LOCK` is required for RDMA verbs if you use InfiniBand or RoCE between nodes. Forgetting it gives you a confusing `ibv_reg_mr failed` deep inside NCCL initialization that no Stack Overflow answer will help with.

### NCCL across containers and nodes

Multi-node training breaks in three classic ways with containers.

The first: NCCL picks the wrong interface. It defaults to docker0 (bridge), which has terrible MTU and goes through iptables NAT. Force the right NIC:

```bash
-e NCCL_SOCKET_IFNAME=eth0       # or ib0 for InfiniBand
-e NCCL_IB_DISABLE=0             # enable IB if available
-e NCCL_IB_HCA=mlx5_0,mlx5_1     # which HCAs
-e NCCL_DEBUG=INFO               # log topology selection
```

The second: topology detection fails inside containers. Mount `/sys` and run with `--privileged` (training nodes only, never serving) or use `--cap-add SYS_ADMIN`. NCCL needs PCIe topology to pick rings versus trees. A misdetected topology can cut all-reduce throughput in half silently.

The third: MTU mismatch between bridge and host network. Use `--network=host` for tightly-coupled multi-node training. Yes, it sacrifices network namespace isolation; for a closed training cluster that is the right trade-off.

```bash
docker run --network=host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA=mlx5_0 \
  -e MASTER_ADDR=node-0 -e MASTER_PORT=29500 \
  -e WORLD_SIZE=16 -e RANK=$RANK \
  trainer:latest torchrun --nproc_per_node=8 train.py
```

A subtler variant is the **NCCL P2P over PCIe** problem. On certain server topologies (e.g., GPUs on different CPU sockets), NCCL falls back from NVLink to PCIe-host-PCIe routing, which is roughly four times slower. Enable `NCCL_P2P_LEVEL=NVL` to assert NVLink-only, and use `nvidia-smi topo -m` from inside the container to confirm the matrix shows `NV{n}` everywhere it should.

### Pull policy and image distribution

A 20 GB image pulled cold over a 1 Gbps link takes about three minutes. Across a 100-node cluster, that is five hours of bandwidth contention. Mitigations, in order of effort:

A registry mirror in-region (Harbor or AWS ECR pull-through cache) is usually the first move. Lazy loading with eStargz or Nydus pulls layer metadata first and fetches blocks on demand; first container starts in seconds. A pre-pulling DaemonSet in K8s ensures every node has the image before pods schedule. Image streaming services (Google GCFS, AWS SOCI) are the managed version of the same idea.

For a fixed inference fleet I prefer pre-pulling. For autoscaling clusters with bursty scale-out, lazy loading is worth the operational overhead. The crossover is roughly: if your image is over 10 GB and you scale faster than your registry can serve, lazy loading wins.

## 5. The Inference Stack: vLLM, SGLang, TensorRT-LLM, Triton

Picking the serving runtime drives ninety percent of your container concerns.

| Runtime                     | Container story                                                                                                         | Best for                                   |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **vLLM**                    | Official `vllm/vllm-openai` image. PagedAttention, continuous batching, OpenAI-compatible API.                          | Default choice for self-hosted LLM APIs.   |
| **SGLang**                  | Slightly leaner image, RadixAttention prefix caching, faster on long-prompt agentic workloads.                          | Agentic / structured-output workloads.     |
| **TensorRT-LLM**            | Compile-once, run-fast. Ships as part of NGC's `nvcr.io/nvidia/tritonserver`. Compilation is GPU-architecture specific. | Maximum throughput, fixed model + GPU SKU. |
| **Triton Inference Server** | Multi-backend (TRT-LLM, vLLM, ONNX, Python). Model repository on disk, dynamic batching across models.                  | Multi-model, multi-framework production.   |
| **TGI (HuggingFace)**       | OCI image, supports streaming, tool-use, JSON-grammar. Slightly behind vLLM on latency.                                 | Tight HF ecosystem integration.            |
| **Ollama**                  | Single-binary container, GGUF format, CPU/GPU hybrid.                                                                   | Local dev, edge inference, offline tools.  |

### A production vLLM container

```bash
docker run -d --name vllm-llama-70b \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 8000:8000 \
  -v /mnt/models:/models:ro \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_TRITON_FLASH_ATTN=1 \
  --health-cmd='curl -f http://localhost:8000/health || exit 1' \
  --health-interval=30s --health-start-period=600s \
  vllm/vllm-openai:v0.7.0 \
    --model /models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

`--health-start-period=600s` because loading 140 GB of fp16 weights takes minutes. A 30-second startup probe will kill the container in a restart loop. `--gpu-memory-utilization 0.92` leaves headroom for activations and CUDA graphs; 0.95 sometimes OOMs under burst load. `--ipc=host` is mandatory for tensor-parallel >1 (the workers are separate processes). Models mounted **read-only**; anything else means a buggy worker can corrupt weights for the other replicas sharing the volume.

### Triton model repository pattern

Triton's superpower is the model repo abstraction: your container is stateless, the disk holds versioned models.

```
/models
├── llama-70b-trtllm
│   ├── config.pbtxt
│   └── 1/
│       └── model.engine        # compiled TRT-LLM
├── embedding-bge
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── reranker-bge
    └── ...
```

```bash
docker run --gpus all --shm-size=8g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /mnt/models:/models:ro \
  nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 \
    tritonserver --model-repository=/models \
                 --model-control-mode=poll \
                 --repository-poll-secs=30
```

`--model-control-mode=poll` means: drop a new version directory on disk, Triton hot-loads it. No container restart, no traffic interruption. This is how you do A/B model rollouts properly.

## 6. Image Size Optimization Beyond the Obvious

After multi-stage and runtime base, the next wins:

### Strip unused CUDA libs

`libcublas`, `libcudnn`, `libnvjpeg`, `libcufft`, `libnvrtc` together are about 3 GB. Most inference workloads use cuBLAS plus cuDNN plus maybe NCCL. Strip the rest:

```dockerfile
RUN find /usr/local/cuda -name 'libnvjpeg*' -delete \
 && find /usr/local/cuda -name 'libcufft*'  -delete \
 && find /usr/local/cuda -name 'libcurand*' -delete
```

Verify the image still passes a smoke test before shipping; some unexpected paths exist (e.g., PyTorch's `torch.compile` indirectly pulls `libnvrtc`).

### Compress with zstd

BuildKit supports zstd compression for image layers (smaller and faster than gzip):

```bash
docker buildx build \
  --output type=image,compression=zstd,force-compression=true \
  --tag registry/llm-server:v42 \
  --push .
```

Typical 12 GB image becomes 7 GB compressed. Faster pulls everywhere.

### Distroless and Chainguard for inference-only

If you do not need a shell in production, `gcr.io/distroless/cc-debian12` or Chainguard's `chainguard/python` images cut another 200 MB and shrink CVE surface dramatically. For LLM serving this works only if your runtime is fully self-contained (compiled binary or single-file Python with all deps installed). vLLM is borderline; Triton is fine.

### Squash-then-split

A single fat image is hard to pull. A useful pattern: keep heavy "base" images (CUDA + torch + vllm) versioned slowly, and ship a thin "app" image that uses the base as `FROM`. Now ninety-nine percent of your CI rebuilds only ship a 200 MB application layer. This is what NVIDIA NGC, HuggingFace, and most ML platforms do internally.

## 7. Observability: What to Wire Before You Need It

By the time you need metrics in production, it is too late to add them. Bake these in from day one.

```yaml
# Minimum surface
- /metrics              Prometheus endpoint (vLLM, Triton both expose this)
- /health               liveness
- /ready                readiness — flips false during model load and warmup
- structured JSON logs  one event per request with model, tokens, latency, status
```

Inside the container, install `dcgm-exporter` as a sidecar for GPU telemetry: SM utilization, memory bandwidth, ECC errors, throttling reasons, NVLink traffic. Vanilla `nvidia-smi` polling will not catch a thermal throttle that drops your throughput by thirty percent; DCGM will.

```bash
docker run -d --gpus all --cap-add SYS_ADMIN \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.7-3.4.2-ubuntu22.04
```

Key SLI for LLM serving (set alerts on these, not on CPU%):

- `vllm:time_to_first_token_seconds` for TTFT p99
- `vllm:time_per_output_token_seconds` for ITL p99
- `vllm:gpu_cache_usage_perc` for KV cache pressure
- `dcgm_gpu_utilization` for SM busy percentage
- `dcgm_fb_used` for VRAM used
- `dcgm_clock_throttle_reasons` for thermal/power throttling

## 8. Security: The Boring Section That Saves Your Job

Security on AI containers is often skipped because "it is just internal." Then your inference endpoint gets exposed via a misconfigured ingress and someone runs arbitrary Python through your tool-calling API.

Run as non-root. Even with GPU access. `USER 1000` in the Dockerfile, then `--user 1000:1000` at runtime. Most CUDA libs do not need root. Mount the root filesystem read-only with explicit `tmpfs` for writes: `--read-only --tmpfs /tmp --tmpfs /var/run`. Drop all capabilities by default and add only what NCCL or RDMA need. Never use `--privileged` in serving containers. Period. Training is a different conversation.

Sign and scan images. `cosign sign` plus `trivy image` in CI. The Hugging Face supply chain has had real incidents (typosquatted packages, malicious safetensors with embedded code paths via `pickle`). Pin everything, including the base image digest. For models from untrusted sources, prefer `safetensors` over `pickle`-based formats. `pickle.load` on a malicious checkpoint is RCE.

`--security-opt no-new-privileges` prevents setuid escalation. Combine with seccomp profiles tuned for your runtime; the default Docker seccomp profile blocks a few syscalls that PyTorch occasionally uses (e.g., `clone3` on older glibc), so test before deploying.

## 9. Kubernetes-Specific Nuances

If you are deploying to K8s, a few things change.

Use the **NVIDIA device plugin** plus GPU Feature Discovery. Resource requests are `nvidia.com/gpu: 1` (or `nvidia.com/mig-1g.10gb: 1`). NVLink-connected GPUs matter for tensor parallelism; use `topology.kubernetes.io/zone` and node selectors that pin to NVLink islands. Init containers for weight pulls keep the main container image small and isolate network failures.

Pod spec must include the right shared-memory volume:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
securityContext:
  capabilities:
    add: ["IPC_LOCK"]
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 16Gi
volumeMounts:
  - mountPath: /dev/shm
    name: dshm
```

`terminationGracePeriodSeconds: 120` (or more) so in-flight requests finish during rolling updates. Default 30s drops live streams mid-token. Horizontal scale on KV cache pressure, not CPU. Use `vllm:gpu_cache_usage_perc` with KEDA or a custom HPA.

## 10. Case Studies: Real Production Incidents

These are taken from incidents I have personally debugged or watched debugged. If you recognize one, you are not alone. Each one ends with the _senior takeaway_: the lesson that generalizes beyond the specific bug.

### Case 1 — "CUDA OOM only on the second deploy"

**Symptom.** A vLLM serving container deployed cleanly on a fresh node. The pod was restarted (rolling update), and the new pod fails to start with `CUDA out of memory`. `nvidia-smi` on the host shows the GPU as "in use" with seven gigabytes of memory consumed but no process attached.

**Investigation.** The previous container was killed with `SIGKILL` because the K8s `terminationGracePeriodSeconds` was 30s and weight unload took longer. CUDA contexts on the GPU are owned by the process; when the kernel OOM-kills a process, the driver is supposed to clean up but in practice the context can leak, especially with multi-process tensor parallel where the worker processes were children of the killed parent.

**Fix.** Two layers: (1) immediate — `nvidia-smi --gpu-reset` (requires no other workload on the GPU), or reboot the node. (2) durable — add a SIGTERM handler that calls `engine.shutdown()` and `torch.cuda.empty_cache()` before exit, and bump `terminationGracePeriodSeconds` to 120. Also add a pre-start init container that runs `nvidia-smi --query-gpu=memory.used --format=csv` and fails the pod if VRAM is non-zero, so you fail fast instead of hitting OOM mid-load.

**Senior takeaway.** GPU state outlives the process. Treat it like a hardware resource that needs explicit deinit, not like RAM that the OS auto-reclaims.

### Case 2 — "Training hangs at step zero"

**Symptom.** A multi-node DDP training job starts, all eight ranks log `[INFO] Initializing process group`, then hang forever. No error, no traceback. CPU usage on all nodes is zero.

**Investigation.** Strace on the master process shows it stuck in a `connect()` on port 29500 to an IP that is not the other node. Container DNS resolved `MASTER_ADDR=node-1` to the docker0 bridge IP of node-1 (172.17.0.x) instead of its real interface. NCCL was waiting on a peer that would never arrive.

**Fix.** Use `--network=host` for training containers. If you must keep the network namespace isolated, set `MASTER_ADDR` to the explicit host interface IP, never a hostname that goes through container DNS.

**Senior takeaway.** Anywhere two containers on different hosts must talk, the default Docker bridge network is wrong. For tightly-coupled compute, host network. For loosely-coupled (web tier), proper service discovery (K8s headless service, Consul).

### Case 3 — "vLLM container CrashLoopBackOff every ten minutes"

**Symptom.** A vLLM pod in K8s logs healthy startup, serves traffic for about ten minutes, then gets killed and restarted. `kubectl describe` shows `Liveness probe failed: HTTP probe failed with statuscode: 500`.

**Investigation.** During a long-context request (32K tokens in, 8K out), the request held the GPU for 90 seconds. The liveness probe had `timeoutSeconds: 5` and `failureThreshold: 3`, so after fifteen seconds of unanswered probes the kubelet killed the pod. This in turn dropped 200 in-flight requests from other clients.

**Fix.** Liveness probe should test that the _process_ is alive, not that it can answer in five seconds. Move the latency check to the readiness probe (which removes the pod from the service but does not kill it), and use a cheap `/health/live` endpoint for liveness that just returns 200 without touching the GPU. Bump `timeoutSeconds: 30` and `failureThreshold: 5` for readiness.

**Senior takeaway.** Liveness probes kill containers; readiness probes only remove them from rotation. For LLM serving where occasional long requests are normal, never put a latency-sensitive check in liveness. The cost of being wrong is restarting a process that took ten minutes to load.

### Case 4 — "Inference latency p99 doubles after six hours"

**Symptom.** A Triton + TRT-LLM service runs cleanly for six hours, then latency p99 climbs from 80ms to 160ms over thirty minutes and stays there. Throughput drops in proportion. No code change, no traffic spike.

**Investigation.** DCGM showed `clock_throttle_reasons` flipping to `SW_THERMAL_SLOWDOWN`. The data center had a chiller maintenance window earlier that day and never returned to normal cooling. The GPU was thermally throttled from 1980 MHz to 1410 MHz, exactly matching the latency ratio.

**Fix.** Operations issue, not Docker issue. But the Docker side mattered: alerts on `dcgm_clock_throttle_reasons != 0` had not been wired up, so the team only noticed via a customer complaint. Wired the alert, raised an incident with facilities, latency returned within forty minutes of cooling restoration.

**Senior takeaway.** Throughput is a downstream signal; clock frequency, ECC errors, and throttle reasons are upstream. If you only alert on outputs (latency, error rate), you discover hardware problems by their effects, hours late. Wire DCGM the day you go to prod.

### Case 5 — "flash-attn works locally, ImportError in prod"

**Symptom.** Developer builds the image locally on an A100 box, everything works. Pushes to staging on H100 hardware, container fails on `from flash_attn import flash_attn_func` with `undefined symbol: ...cuda...sm_90`.

**Investigation.** Local build had `TORCH_CUDA_ARCH_LIST="8.0"` (just A100). The compiled wheel did not include sm_90 (H100) kernels. Production loaded the wheel, found no matching arch, and aborted.

**Fix.** Set `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` in the builder stage. Or use the prebuilt wheels from the flash-attn GitHub releases that target multiple arches. Verify with `python -c "import flash_attn; print(flash_attn.__version__)"` in CI on a target-arch GPU.

**Senior takeaway.** "Works on my machine" for CUDA ops means "works on my arch." Your CI must build _and run_ on every architecture you ship to, or your build matrix is wrong.

### Case 6 — "Image went from 12 GB to 28 GB overnight"

**Symptom.** The team's daily image build, which had been stable at 12 GB for months, jumped to 28 GB. CI started timing out on the push step. Nobody could pinpoint which commit caused it.

**Investigation.** A new contributor had added a Jupyter notebook with embedded plotly outputs to the repo root. `COPY . /app` slurped it in. They had also added `data/` with a 12 GB sample dataset for "easier local debugging."

**Fix.** A proper `.dockerignore`. The senior move is to make `.dockerignore` strict by default (deny-list) rather than permissive (allow-list).

```
# .dockerignore — start strict
**
!src/**
!pyproject.toml
!requirements*.txt
!entrypoint.sh
```

This pattern, "ignore everything, then allow-list," is the only way to prevent regressions long-term. Five different teams I worked with adopted it after their own version of this incident.

**Senior takeaway.** Build hygiene is a one-way ratchet. It is easy to bloat, hard to debloat, impossible to debloat without a deny-list strategy.

### Case 7 — "GPU works, but nccl-tests times out"

**Symptom.** Inside the container, `nvidia-smi` prints all eight GPUs. `python -c "import torch; print(torch.cuda.device_count())"` returns 8. Running `all_reduce_perf -b 8 -e 128M -f 2 -g 8` (the official NCCL benchmark) hangs for thirty seconds and times out with `unhandled cuda error`.

**Investigation.** The container was launched without `--ipc=host` and with `--shm-size=64m` (default). NCCL initialization tries to mmap a shared-memory region for the inter-rank handshake; with 64 MB, it succeeds for 2 ranks and silently fails for 8.

**Fix.** `--ipc=host --shm-size=16g`. As a sanity check, `df -h /dev/shm` inside the container should show >= 16 GB.

**Senior takeaway.** NCCL failure modes are some of the worst-debuggable in the stack because the error messages are written for the kernel team, not for users. Know the four flags (`--ipc=host`, `--shm-size`, `--ulimit memlock=-1`, `NCCL_DEBUG=INFO`) and set them by default.

### Case 8 — "Cold start latency in autoscaler is killing us"

**Symptom.** Black Friday traffic spike. The HPA scaled vLLM from 4 to 12 replicas. New pods took six minutes each to become ready. By the time they came up, the spike had passed and they immediately scaled down. P99 latency during the spike was 30 seconds because the four original pods were saturated.

**Investigation.** Six minutes broke down as: 90 seconds to pull a 14 GB image (cold node), 180 seconds to download Llama-70B from S3 (60 GB at ~330 MB/s), 90 seconds to load weights into VRAM and warm up CUDA graphs.

**Fix.** Three changes in priority order. (1) Pre-pull the image to all GPU nodes via a DaemonSet, so image pull becomes zero on warm starts. (2) Move weights to a `ReadOnlyMany` PV backed by NFS or FSx for Lustre, mounted on every node; weight load drops from 180 to 30 seconds. (3) Use `startupProbe` separately from `readinessProbe` so K8s does not panic during the legitimate startup window. Combined: 6 minutes to 90 seconds. (4) Long-term: keep a "warm pool" of pre-loaded replicas at 0% utilization, scaling up the pool an hour before known peaks.

**Senior takeaway.** Cold start is not one number; it is a sum of four. Optimize the largest term, not the most visible one. Measure _each phase separately_ in your dashboard so you know which one to attack.

### Case 9 — "The container can see the GPU, but `torch.cuda.is_available()` returns False"

**Symptom.** A new dev environment image. `nvidia-smi` works inside. `python -c "import torch; print(torch.cuda.is_available())"` prints `False` with a warning about UserWarning: CUDA initialization: ...

**Investigation.** The image had Python 3.12. PyTorch 2.1 wheels were compiled for CUDA 11.8 and Python 3.10/3.11. The container had CUDA 12.4 system libs. The PyTorch import succeeded (it falls back to CPU), but couldn't find the right CUDA runtime for its bundled kernels.

**Fix.** Match the PyTorch wheel to the container's CUDA. Easiest path: use the official `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime` base image and let them handle the version dance. Or pin: `pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124`.

**Senior takeaway.** CUDA versioning is a four-way handshake: GPU silicon, host driver, container CUDA, framework wheel. Document the chosen versions in a `VERSIONS.md` next to your Dockerfile, and treat any drift as a release-blocking change.

### Case 10 — "We are paying $40K/month for VRAM we don't use"

**Symptom.** A finance review flags that the inference fleet's H100s are at thirty percent average VRAM utilization. The team had provisioned for worst-case batch size during launch, then the actual traffic profile turned out to be smaller queries with shorter contexts.

**Investigation.** Each replica was reserving 75 GB of VRAM via `gpu_memory_utilization=0.95`, but the KV cache p99 was using only 22 GB of that. The over-provisioning was a constant cost, twenty-four hours a day.

**Fix.** Two levers. (1) Drop `gpu_memory_utilization` to 0.55, freeing the GPU for a second co-located replica via MIG. The H100 was now serving two pods, each with its own 1g.40gb slice, doubling effective replicas without adding hardware. (2) For variable-context workloads, switch to vLLM's `max_num_seqs` autoscaling and batch size = 256 (was 32), which improved throughput within the smaller VRAM budget. Net savings: forty-two percent of the monthly H100 bill.

**Senior takeaway.** Capacity planning by worst case is expensive. Capacity planning by actual percentile usage, with MIG slicing as the granular knob, is where the real money lives. Measure first, provision second.

### Case 11 — "The image is poisoned"

**Symptom.** A security audit caught that the team's nightly build, generated by an OSS pipeline, had a `cryptominer.so` bundled into `/usr/local/lib`. Investigation traced it back to a dependency on a specific Python package that had been hijacked on PyPI two days earlier.

**Investigation.** The team used a wide pin (`tokenizers>=0.13`) which let dependency resolution pick a malicious version. Their CI ran with no SBOM, no `trivy` scan, no `cosign` verification. The image got pushed to ECR, deployed to staging, and was caught only because a security tool elsewhere noticed an unusual outbound connection.

**Fix.** End-to-end supply chain hardening. (1) Generate SBOM at build time with `syft`. (2) Run `trivy image --exit-code 1 --severity HIGH,CRITICAL` in CI. (3) Sign with `cosign sign --key <kms-key>` and verify on `kubectl apply` via Sigstore policy controller. (4) Pin every Python dependency with `==`. (5) Use `pip-audit` or `safety` as a second scanner. (6) Build from a private registry mirror, not directly from PyPI, so a compromise gives time to react.

**Senior takeaway.** Supply chain attacks against AI workloads are real and increasing because the dependency graphs are deep and the maintainers under-resourced. Pin, scan, sign, mirror. Operations cost is real but the alternative is unreviewed code running with GPU access on production data.

### Case 12 — "Half the traffic is slow, half is fast, no pattern"

**Symptom.** A 16-replica vLLM service shows a bimodal latency distribution: half of requests at 80ms TTFT, half at 180ms. No correlation with prompt length or model.

**Investigation.** All 16 replicas were on a 2-node cluster, eight pods per node. Profiling showed that pods on certain GPUs ran at fifty percent NVLink bandwidth. `nvidia-smi topo -m` revealed the GPUs were split across two CPU sockets, and pods scheduled on the wrong NUMA node went through QPI for every PCIe access.

**Fix.** NUMA-aware pinning via `numactl --cpunodebind=$NODE --membind=$NODE` in the container entrypoint, with a downward-API env from K8s that exposes the chosen GPU's NUMA affinity. Plus a topology-aware scheduler (NVIDIA GPU Operator can do this with the right config). Latency converged to 80ms across all replicas.

**Senior takeaway.** Below the container abstraction, hardware topology still matters. PCIe, NUMA, NVLink, and CPU socket layout are invisible to Kubernetes by default. For high-throughput serving you must surface them. The good news: this is a one-time setup, not a per-deploy concern.

## 11. Applied Problems: Worked Examples

A reference of common shapes of "I need to ship this" problems and the container recipe that solves them.

### Problem A — Self-hosted ChatGPT-equivalent on a single 8×H100 node

You have a single DGX-H100 box and want to serve Llama-3.1-70B-Instruct to internal users with an OpenAI-compatible API.

Recipe: vLLM in tensor-parallel-size=4, run two replicas (each on 4 GPUs) behind a small nginx that load-balances. Mount weights from `/mnt/nvme/models` (read-only). Use `--ipc=host`, `--shm-size=32g`, `--ulimit memlock=-1`. KV cache fp8, prefix caching on. Prometheus + Grafana sidecar for `vllm:*` metrics, dcgm-exporter for hardware. Fronted by an authenticated proxy that does rate limiting per user.

Pitfall: do not run two TP=4 replicas on the same eight GPUs by GPU index alone. Use `--gpus '"device=0,1,2,3"'` for replica A and `--gpus '"device=4,5,6,7"'` for replica B. The default `--gpus all` will fight over GPUs and OOM at startup.

### Problem B — RAG service with embedding + reranker + LLM

You need an end-to-end RAG endpoint: BGE embeddings, BGE reranker, and a 7B LLM. Latency budget is 800ms p95.

Recipe: One Triton container hosting all three models (BGE embedding ONNX, BGE reranker ONNX, Llama-7B as a TRT-LLM engine) on a single L40S or H100. Triton's dynamic batching across models lets you serve embed and rerank requests with implicit batching while the LLM streams. Front with a thin Python orchestrator container (FastAPI) that calls Triton via gRPC for sub-millisecond overhead. Vector store (Qdrant or pgvector) in a separate container.

Pitfall: do not serve all three from three separate Python processes. The cross-process overhead and triple GPU context cost you 200ms+ of latency you cannot afford. Triton co-location pays for itself.

### Problem C — Fine-tuning loop on Spot/Preemptible GPUs

You are running supervised fine-tunes of 13B models, each taking eight hours on 4×A100. Spot instances are seventy percent cheaper but get reclaimed with two minutes notice.

Recipe: Container that mounts a checkpoint volume (S3-backed, e.g., via s3fs or rclone). Trainer (axolotl or torchtune) writes a checkpoint every 500 steps and on SIGTERM. Wrap the training command with a shell that traps SIGTERM, calls `kill -USR1` to the trainer, and waits for `checkpoint_saved.flag` before exiting. K8s pod with `terminationGracePeriodSeconds: 120`. On preemption, a controller pod re-schedules the job pointing at the latest checkpoint URI.

Pitfall: writing a 26 GB optimizer state checkpoint to S3 takes longer than two minutes. Use sharded checkpointing (`torch.distributed.checkpoint`), each rank writes its shard in parallel. Or keep optimizer state on local NVMe and only persist model weights on preemption (accepting that you may lose Adam momentum on resume; usually fine after a few steps of warmup).

### Problem D — Inference at the edge (no internet, limited VRAM)

A factory client wants on-prem inference of a 7B model on a single RTX 4090 (24 GB) with no outbound internet for compliance.

Recipe: Pre-build a self-contained image with the model baked in (this is one of the few cases where baking is correct). Use AWQ or GPTQ 4-bit quantization to fit a 7B comfortably in 8 GB, leaving 16 GB for KV cache. Ollama or llama.cpp server (CPU+GPU hybrid, GGUF) for simplicity. Image signed and verified at install time. Auto-updates blocked.

Pitfall: 4-bit quantization changes inference quality. Validate on the customer's eval set before signing off. Have a fp16 fallback image ready in case quality regression is unacceptable.

### Problem E — Multi-tenant inference platform

You run an internal "model platform" where different teams deploy their own finetuned models, each as a separate endpoint, sharing a fleet of H100s.

Recipe: One vLLM-with-LoRA container per base model (e.g., one for Llama-3.1-70B base, one for Mistral-7B base). Tenants supply LoRA adapters by uploading to an S3 bucket. The vLLM container hot-loads adapters on demand via its `/v1/load_lora_adapter` API. MIG to isolate teams that need hard guarantees, time-share for the rest. Per-tenant API gateway tracks usage and enforces rate limits.

Pitfall: noisy neighbors on a shared base model can starve others. Set vLLM's per-request priority via the LoRA name and use cooperative scheduling. Or hard-isolate via MIG for teams with SLA contracts.

### Problem F — Batch inference of millions of documents

A nightly batch job scores ten million documents through a 13B classifier. Total budget: four hours, eight A100s.

Recipe: This is _not_ a serving problem. Use Ray or Dask, not an HTTP API. Container with vLLM in `offline` mode (no server, direct LLM API). Each worker reads a shard from S3, runs `llm.generate()` with batch_size=256, writes results back. KV cache reuse via prefix caching since most docs share a system prompt. No Triton, no nginx, no autoscaler.

Pitfall: people often try to serve batch traffic through their online inference fleet "to reuse infra." This OOMs the online fleet because batch traffic doesn't respect rate limits, and is two to four times slower than offline mode because of HTTP/network overhead per request. Use the right tool.

### Problem G — Latency-critical real-time inference (streaming voice)

Voice-agent product requires TTFT < 250ms p99 on a 7B model with 2K-token system prompt.

Recipe: SGLang container with RadixAttention prefix caching (the system prompt is hit on every request). Pin the container to a single GPU (no TP overhead). FP8 KV cache. CUDA graphs enabled. Disable any feature that adds branching: speculative decoding off, no LoRA, fixed batch size. Run two replicas behind a sticky-session load balancer so the prefix cache stays warm per user.

Pitfall: cold cache on autoscaling kills the SLA. Keep `min_replicas` >= 2 even at low traffic, and pre-warm the prefix cache on startup with a synthetic request.

## 12. Best Practices: The Optimization Cookbook

A consolidated reference. Each item is one sentence of _what_, one of _why_, and one of _how_. Use it as a deploy-time checklist; if you cannot justify a deviation, follow the default.

### 12.1 Build-Time Best Practices

**Pin the base image by digest, not tag.** Tags like `nvidia/cuda:12.4-runtime` are mutable; the maintainers can rebuild them. A six-month-old reproducible build today won't be reproducible tomorrow. Use `FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:abc123...`. Update digests deliberately via Renovate or Dependabot, not implicitly.

**Use BuildKit, always.** Set `DOCKER_BUILDKIT=1` (or `# syntax=docker/dockerfile:1.7` on the first line). Cache mounts, secret mounts, and parallel stage builds only exist with BuildKit. The legacy builder is functionally a different product.

**Order RUN steps from coldest to hottest.** Each `RUN` invalidates the cache for everything below it. System packages first, heavy Python wheels next, light packages, app code last. A one-line app change should never re-download torch.

**Combine RUN steps for cleanup, not for "cleanliness."** `RUN apt-get install ... && rm -rf /var/lib/apt/lists/*` is correct because the cleanup must be in the same layer. But splitting unrelated commands across multiple RUNs is _better_ because it preserves cache granularity. Combine for correctness, not aesthetics.

**Always specify `--no-install-recommends` for apt.** Default `apt-get install` pulls "recommended" packages which on Ubuntu means dragging in 200 MB of stuff you don't need. One flag, fifteen percent image reduction.

**Build with `--platform=linux/amd64` explicitly on Apple Silicon dev machines.** Otherwise Docker silently builds an arm64 image that fails on x86_64 GPUs at deploy. Add `--platform` to your Makefile and CI; never rely on the host default.

**Use `.dockerignore` as a deny-list, not an allow-list.** Start with `**` then `!`-allow specific paths. This makes new files default-excluded, so a contributor adding `data/` cannot bloat your image without an explicit allow rule.

**Build once, promote across environments.** The image you test in staging must be the _exact same digest_ you ship to prod. Never rebuild on promote. Use registry tags (`:staging`, `:prod`) that point at the same SHA.

**Generate SBOM at build time.** `syft sbom oci-dir:./image -o spdx-json` produces a complete dependency manifest. Store alongside the image. When CVE-2025-XXXXX drops, you can answer "are we affected?" in seconds without re-scanning the fleet.

**Sign images with `cosign`.** A signed image plus an admission controller policy means an attacker cannot inject a tampered image into your cluster even with registry write access.

**Run `hadolint` in CI.** Catches Dockerfile anti-patterns automatically. Five-minute setup, prevents the next ten "why is the image 5 GB bigger than expected" tickets.

**Use multi-arch only when you actually deploy multi-arch.** Building `linux/amd64,linux/arm64` doubles build time. If your fleet is x86_64, do not pay this cost.

**Cache the wheel index, not just the wheels.** `pip install` resolves the dependency graph by hitting PyPI metadata. On flaky networks, that resolution fails before any download starts. Mirror via Devpi or a pull-through Artifactory; reduces "PyPI is down" outages on builds.

### 12.2 Runtime Best Practices

**Default flags for any GPU container:**

```bash
--gpus all \
--ipc=host \
--shm-size=16g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ulimit nofile=1048576:1048576
```

Memorize these six. They cover ninety percent of "weird hangs / OOMs / NCCL errors" in production.

**Always set `--restart=unless-stopped` (or `Always` in K8s) for serving, but never for training.** Training failures need human investigation; auto-restart hides bugs. Serving failures need fast recovery; auto-restart is a feature.

**Set `OOM_SCORE_ADJ` for serving containers.** `--oom-score-adj=-500` makes the kernel prefer killing other processes over your serving process when memory pressure hits. Pair with a memory limit so it cannot starve other tenants.

**Run as non-root unless GPU access requires it.** With nvidia-container-toolkit modern versions, non-root works. Add `USER 10001:10001` in the Dockerfile and `--user 10001:10001` at runtime. Not for paranoia; for limiting blast radius.

**Mount config as files, not env vars.** Env vars leak into `docker inspect`, `ps aux`, error reports, and crash dumps. For secrets and large configs, mount a tmpfs file: `--mount=type=bind,src=./config.yaml,dst=/app/config.yaml,readonly`.

**Pin container CPU affinity to the GPU's NUMA node.** `docker run --cpuset-cpus=0-31 --cpuset-mems=0` for a GPU on NUMA node 0. Cross-NUMA traffic costs ten to twenty percent of throughput on data-heavy pipelines.

**Always set explicit memory limits.** A container without `--memory` can hit the host OOM killer and bring down siblings. Set to ninety percent of physical RAM divided by expected co-tenants. The container will OOM itself instead of the kernel killing the wrong process.

**Use `tini` (or `--init`) as PID 1.** Without it, your Python process is PID 1, doesn't reap zombies, and ignores SIGTERM by default. `docker run --init ...` or build with `ENTRYPOINT ["tini", "--", "python", ...]`. One line; eliminates a class of "container never shuts down cleanly" bugs.

**Set health checks that don't load the GPU.** `/health/live` returns 200 from a 1-line handler that doesn't touch the model. `/health/ready` checks that the model is loaded. `/health/deep` (called manually, not by k8s) does a 1-token inference. Three endpoints; never one fat probe.

**Make logs structured JSON from day one.** `{"ts":"...","level":"info","model":"llama-70b","tokens_in":..., "tokens_out":...,"latency_ms":...}`. Trivial to grep and aggregate. Switching from text logs to JSON six months in is a bigger project than you think.

**Send container logs to stdout/stderr, never to files inside the container.** The container is ephemeral. Files in it disappear on restart. The Docker logging driver handles rotation, the file approach does not.

**Use `READ-ONLY` filesystems with explicit `tmpfs` mounts for writes.** `--read-only --tmpfs /tmp:size=512m --tmpfs /var/run:size=64m`. Defends against persistence by an attacker who gets RCE; the moment they try to drop a file, it fails.

### 12.3 Image Size Best Practices

**Use the right base for the right job:**

| Job                              | Base image                                    |
| -------------------------------- | --------------------------------------------- |
| Training (need nvcc)             | `nvidia/cuda:X.Y-devel-ubuntu22.04`           |
| Inference (compiled wheels only) | `nvidia/cuda:X.Y-runtime-ubuntu22.04`         |
| Inference, ultra-thin            | `nvcr.io/nvidia/distroless/cuda-runtime`      |
| Pure CPU preprocessing           | `python:3.11-slim`                            |
| Edge / embedded                  | `chainguard/python` or `gcr.io/distroless/cc` |

The wrong base costs three to five GB. The right base is one search-and-replace away.

**Strip CUDA libs you don't use.** A typical inference image only needs cuBLAS, cuDNN, NCCL, and cuFFT (occasionally). Delete `libnvjpeg`, `libcurand`, `libnpp*`, `libnvgraph*`. Document which ones you stripped so the next person doesn't re-add them.

**Use `pip install --no-deps` once you know the dep tree.** After the first `pip install -r requirements.txt`, pip-compile the resolved versions, then in the Dockerfile install with `--no-deps`. Faster, deterministic, no surprise transitive bumps.

**Compress with zstd at push time.** `docker buildx build --output type=registry,compression=zstd`. Half the bytes on the wire. Modern Docker daemons support it natively; older ones fall back to gzip.

**Prefer `wheel`-only installs.** `pip install --only-binary=:all: -r requirements.txt`. Forces pip to use prebuilt wheels and fail loudly if a source build is needed (which you don't want in production images anyway).

**Keep build artifacts out of the runtime image.** Multi-stage `COPY --from=builder` only what you need. A surprising amount of your image bloat is `.pyc` caches, `__pycache__` dirs, and pip's wheel cache. Set `PYTHONDONTWRITEBYTECODE=1` and `PIP_NO_CACHE_DIR=1`.

**Vendor model code, don't `pip install -e .` from a mounted git repo.** The latter requires git in the image, drags `.git/` history (often gigabytes), and breaks reproducibility. Either install a tagged release or vendor the source tree.

### 12.4 GPU & CUDA Best Practices

**Match the four-way version handshake explicitly.** Document GPU silicon → driver version → container CUDA → framework wheel in a `VERSIONS.md`. Treat any drift as a release-blocking change.

**Set `TORCH_CUDA_ARCH_LIST` to your actual fleet.** Not "all". For a fleet of A100 + H100, set `"8.0;9.0"`. Saves compile time and binary size; eliminates a class of "works on dev box, fails in prod" bugs.

**Pre-warm CUDA graphs in startup probes.** First inference after model load takes longer because cuDNN autotune runs. Make startup probe wait until at least one warmup inference has completed. Otherwise the first user request is a tail-latency outlier.

**Use FP8 KV cache where supported.** On H100/H200/B100, `--kv-cache-dtype fp8` cuts KV memory by half with negligible quality loss for most models. Free throughput.

**Enable prefix caching for repeated prompts.** vLLM `--enable-prefix-caching`, SGLang RadixAttention. For agentic / few-shot / system-prompt workloads, this is a two-to-five-times throughput win.

**Calibrate `gpu_memory_utilization` empirically.** Default 0.9 is conservative on stable workloads, aggressive on bursty ones. Run a load test, watch the OOM threshold, set to ninety percent of the maximum stable value.

**Disable ECC if you can tolerate it (training only).** `nvidia-smi -e 0` saves about six percent of VRAM and gives modest throughput gains. Never for training runs you can't restart, never for inference. Decide per-cluster, document the choice.

**Use NCCL persistent threads for high QPS workloads.** `NCCL_LAUNCH_MODE=GROUP`, `NCCL_NTHREADS=<num>`. Reduces per-call overhead for tensor-parallel inference where you do many small all-reduces.

**Always start NCCL with `NCCL_DEBUG=INFO` on first deploy of a new topology.** Read the topology dump. Confirm rings/trees match expectation. Then turn it off (it is verbose). The five minutes of setup beat five hours of "why is training half-speed" later.

**Run nccl-tests on every new cluster, before any real workload.** `all_reduce_perf -b 1M -e 8G -f 2 -g 8`. If you don't see at least seventy percent of theoretical NVLink bandwidth, your cluster is misconfigured. Find out before training, not after.

### 12.5 Networking Best Practices

**Use `--network=host` for tightly-coupled compute (training, multi-worker TP).** Worth the namespace loss for the bandwidth win. Document the security implications.

**Never use `--network=host` for serving.** It exposes every port the container opens to the host without firewall consideration. Use bridge with explicit `-p` mappings.

**Pin NCCL to the right interface.** `NCCL_SOCKET_IFNAME=eth0` (or `^docker,lo` to exclude). Default detection picks docker0 in containers and silently drops to terrible performance.

**Set jumbo frames on training networks.** MTU 9000 on InfiniBand or RoCE. Default 1500 means six-times the packet count for the same data, six-times the per-packet overhead. Coordinate with network ops.

**Use connection pooling for downstream calls.** Inference services that call vector stores, databases, or external APIs should pool. A new TCP connection per request adds 1–10ms of TTFT.

**Set explicit DNS in the container.** `--dns=10.0.0.2 --dns=10.0.0.3`. Container DNS resolution timeouts have caused more "intermittent slow startup" tickets than any other single cause.

### 12.6 Storage & Volumes

**Models on read-only volumes, period.** `-v /mnt/models:/models:ro`. Eliminates a class of accidental-corruption bugs.

**Logs, caches, and tmp on tmpfs or a dedicated volume.** Never on the container's writable layer. The writable layer is slow (overlayfs) and balloons if uncapped.

**Use NVMe for HF cache, NFS for cold storage.** First-pull download to NFS, then copy hot models to NVMe per-node. Three-tier: S3 (cold) → NFS (warm) → local NVMe (hot). Latency and bandwidth improve at each tier.

**Set explicit `sizeLimit` on `emptyDir: medium: Memory`.** Default is half the node RAM. A misbehaving DataLoader can fill 200 GB of tmpfs on a fat node and OOM the kubelet.

**Snapshot model weights with content-addressed storage.** Whether HF Hub xet, IPFS, or a custom system. The win is on diffs: a finetuned variant of a 70B model only stores changed tensors.

**Pre-fetch with a DaemonSet on K8s.** Run a "model warmer" DS that pulls hot models to local NVMe on every GPU node. New pods skip the download.

### 12.7 Observability Best Practices

**Three log levels, not five.** INFO for request lifecycle, WARN for recoverable issues, ERROR for failures. DEBUG and TRACE belong in dev, not prod (they fill disk and leak PII).

**Trace every request.** OpenTelemetry instrumentation. Span per phase (auth → tokenize → prefill → decode → detokenize → respond). Even at one percent sampling, you can answer "where did the latency go" in production.

**Export both vLLM/Triton metrics and DCGM metrics, correlated by timestamp.** Service-level SLI plus hardware-level cause analysis. Without DCGM you can see a regression but not its cause.

**Alert on rates, not absolute values.** "Error rate > 1% over 5min" beats "errors > 100". Latter is noisy on traffic spikes; former is invariant.

**Define SLOs explicitly: TTFT, ITL, throughput, availability.** "Fast and reliable" is not a target. "TTFT p99 < 500ms over 28 days at 99.5% availability" is. Document, alert, review.

**Synthetic monitoring from outside the cluster.** A canary client that sends one request per minute to your public endpoint and pages on failure. Catches issues that internal metrics miss (DNS, ingress, auth).

### 12.8 Security Best Practices

**Drop all capabilities, add back only what you need.** `--cap-drop=ALL --cap-add=IPC_LOCK` (for RDMA, only if needed). Defaults are too permissive.

**`--security-opt=no-new-privileges`.** Cheap, prevents setuid escalation, no downside.

**Apply seccomp profiles.** Docker default is reasonable; for hardened deployments, use a custom profile that blocks unused syscalls. Profile your app, generate the profile, apply.

**Scan every image before push.** `trivy image --exit-code 1 --severity HIGH,CRITICAL` in CI. Block builds with critical CVEs unless explicitly waived.

**Rotate registry credentials regularly.** Every ninety days minimum. Scoped to single repos, not org-wide. Audit access.

**Sign with `cosign`, verify with admission controller.** Sigstore policy controller in K8s. Unsigned images cannot run.

**Use safetensors, not pickle.** `pickle.load` on an untrusted weight file is RCE. Safetensors is structurally incapable of executing code.

**Keep secrets out of images entirely.** No `.env` in `COPY`. Use Vault, Sealed Secrets, or cloud secret manager mounted at runtime.

**Run security policies even in dev.** Bypassing security in dev guarantees those bypasses ship to prod by accident eventually. Same `--cap-drop`, same non-root, same read-only filesystem from day one.

**Isolate untrusted model serving.** A model from an external source runs in its own network namespace, on its own GPU (MIG-isolated), with no access to internal services. Treat new models like third-party code: untrusted until proven otherwise.

### 12.9 CI/CD & Deployment Best Practices

**Build images once per commit, tag with the git SHA.** `image:abc123` is immutable. `image:latest` is a lie. Promote SHAs across environments; never rebuild on promote.

**Test the image, not the source.** Your test suite runs against the built image (`docker run image:abc123 pytest`). This catches packaging bugs that source-only tests miss.

**Smoke test on real GPU hardware in CI.** A self-hosted runner with one GPU. Every PR runs `docker run --gpus all image python -c "import torch; assert torch.cuda.is_available()"`. Catches CUDA/driver mismatches before merge.

**Rolling updates with `maxSurge=1, maxUnavailable=0` for serving.** New pod comes up, old pod drains in-flight requests, then terminates. Zero downtime.

**Canary five percent of traffic to new versions for thirty minutes.** Watch p99 latency, error rate, and DCGM metrics. Auto-rollback on regression.

**Always have a `kubectl rollout undo` rehearsed.** Practice it in staging quarterly. The first time you do it under pressure should not be in prod.

**Image lifecycle policy in the registry.** Keep last 30 prod tags + tagged releases; garbage-collect everything else. Otherwise your registry storage cost grows unbounded.

**Test cold-start time as a first-class metric in CI.** A test that fails if startup exceeds N seconds. Cold start regresses subtly when someone adds a heavy import; catch it early.

**Document the runbook in the repo, not in Confluence.** `RUNBOOK.md` next to the Dockerfile. On-call engineer reads from where they are already looking.

### 12.10 Cost Optimization Best Practices

**Right-size with MIG instead of replicas-per-GPU when possible.** Hardware isolation, deterministic performance, simpler debugging.

**Use spot/preemptible instances for batch and training, never for serving.** Serving needs availability SLAs that spot cannot offer. Training can checkpoint and resume.

**Co-locate small models on the same GPU.** A 1B classifier and a 3B embedder can share a single L4. Use Triton's instance grouping or run two vLLM containers with `gpu_memory_utilization=0.4` each.

**Quantize aggressively for non-critical paths.** AWQ/GPTQ 4-bit for re-rankers and classifiers, fp8 for primary LLMs, fp16 only where quality demands. Each tier roughly halves cost.

**Autoscale on KV cache pressure, not request rate.** Request rate is a lagging indicator; KV cache pressure tells you the GPU is about to choke. Scale up before latency degrades.

**Schedule batch jobs to off-peak GPU hours.** If you have an inference fleet that idles at night, run nightly batch on the same GPUs. Resource sharing across workload types compounds.

**Track $/1M tokens as your unit cost.** Not "$/hour". Hour-based thinking hides the impact of throughput optimizations. A change that doubles throughput halves $/token.

**Review GPU utilization weekly.** Average SM utilization < 60% means you can downsize or co-locate. Average > 90% means you're at risk and should pre-emptively scale.

### 12.11 Disaster Recovery Best Practices

**Image registry geo-replication.** Single-region registry is a single point of failure. ECR replication or Harbor multi-replica across regions.

**Model weights in two regions.** S3 cross-region replication for weights. Restoring from cold storage takes hours; replication is automatic.

**Test restore-from-zero quarterly.** "How long does it take to bring up the inference service in a fresh region from scratch?" Time it. Reduce it. Document it.

**Capture the entire runtime environment in code.** Helm chart or Terraform module that recreates the deployment. No "click here in the AWS console" steps.

**Have a fallback model ready.** When your primary model has an incident, can you switch to a smaller / older / different vendor model in five minutes? Plan it in advance.

## 13. The Senior Mental Model

Internalize this and most decisions become obvious.

The container is the unit of _deployment_, not the unit of _state_. Weights, caches, logs all live outside. The image is reproducible; the running container is disposable. This single principle resolves about half of the case studies above.

Optimize for the slowest path you actually hit. If you redeploy every commit, optimize layer cache. If you autoscale every minute, optimize cold start. If you train for weeks, optimize NCCL, not image size.

Every flag you add (`--privileged`, `--network=host`, `--ipc=host`) trades isolation for performance. Know which trade you are making and why. The flags are not free; they expand the blast radius if anything inside the container is compromised.

Measure before you optimize. A 12 GB image is a problem if you pull it 1000 times per day; it is a non-issue if you pull it once per week. Pick battles. Senior engineers know the difference between a hot path worth optimizing and a cold path worth ignoring.

Reproducibility is downstream of pinning. Pin base image digest (`@sha256:...`), pin all `pip` versions, pin CUDA, pin driver. "It works on my machine" is a pinning failure, full stop.

## 14. Closing Thought

Docker for AI is not Docker plus a GPU flag. It is a different operational discipline that happens to share the same CLI. The tooling rewards engineers who treat the container as a contract — between the host's silicon, the image's libraries, the runtime's flags, and the model's expectations — and punishes everyone who treats it as a black box.

Build the contract deliberately. Test each layer in isolation. Wire DCGM and structured logs before you need them. And the next time someone asks "why does the model load fast on staging and slow on prod," you will already have the answer ready: it is the `HF_HUB_ENABLE_HF_TRANSFER` flag, the NVMe mount, or the registry pull path. Probably all three.

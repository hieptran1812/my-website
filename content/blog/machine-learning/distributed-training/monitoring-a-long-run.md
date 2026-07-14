---
title: "Monitoring a Long Run: The Dashboards That Catch a Failing Job Before You Waste a Week"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The four families of signal a multi-week training run emits, the metric-to-failure-signature lookup that tells you what a bad value means, the per-rank instrumentation that catches the straggler an average hides, and the handful of alerts worth a 3am page — with runnable loggers, a hardware poller, and a grad-norm guard."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "monitoring",
    "observability",
    "dcgm",
    "mfu",
    "pytorch",
    "gpu",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

The run had been going for nine days. Sixty-four H100s, a 7-billion-parameter model, a token budget that put the finish line another eleven days out. On day nine somebody finally opened the dashboard — really opened it, scrolled past the loss curve — and found that the aggregate throughput had been sitting about seven percent lower than launch for the past four days. Not a cliff, not an alert, just a gentle sag nobody had been watching. It turned out one node had a GPU that had been thermal-throttling since a rack-cooling hiccup on day five. Four days at seven percent slower is roughly seven GPU-days of an sixty-four-GPU cluster thrown away — a couple thousand dollars of rented H100 time, gone, because the one number that would have caught it was averaged across sixteen nodes and the sick one disappeared into the mean.

That is the cheap version of the story. The expensive version is the run that quietly starts diverging, or the fp16 job whose gradients climb for two hundred steps and then NaN, or the node that dies at 3am and takes the whole all-reduce down with it while everyone sleeps. A large training run is not a script you fire and forget. It is a **long-lived distributed system** — the longest-lived one most ML engineers ever operate — and like any production system it has failure modes that cost you days if you find them late and minutes if you find them early. The difference between those two outcomes is almost entirely **observability**: the metrics you emit, the dashboards you put them on, and the small set of alerts you trust enough to answer at 3am.

![a tree that splits a long training run into four signal families with one failing metric drawn under each family](/imgs/blogs/monitoring-a-long-run-1.webp)

This post is the operations manual for watching a multi-week job. The figure above is the whole shape of it: a long run emits four families of signal, and each family answers one question about whether the job is healthy — is it *learning* (training health), is it *fast* (throughput), is a GPU *sick* (hardware), is the pipeline *fed* (system). By the end you will know what to put on each of those four panels, what a bad value in each one actually means, why the average across ranks is the single most dangerous view on your dashboard, which six things are worth a page and which twenty are noise, and you will have runnable code for a metrics logger, a hardware poller, a straggler detector, and a grad-norm guard that alerts and checkpoints before the NaN. This is the operations chapter of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series; it sits on the fourth wall the whole series is built around — the run costs too much — because every day a broken job runs unwatched is a day of GPU-hours you paid for and threw away.

## A training run is a production system

Start with the framing, because it changes every decision that follows. When a web service goes down, nobody argues about whether it needs monitoring — of course it does; there are dashboards, there are on-call rotations, there are pages. But a training run *feels* different. It feels like a batch job: you submit it, it churns, it produces a model. The instinct is to check on it now and then and otherwise leave it alone.

That instinct is wrong at scale, and the reason is duration multiplied by cost. A multi-week run on a large cluster is burning money continuously — an eight-GPU H100 node is on the order of \$25 to \$100 per hour depending on where you rent it, and a sixty-four-GPU job is eight of those. Every hour the job runs in a broken state — throttled, diverging, starved for data, or dead-but-not-crashed — is an hour of that bill spent producing nothing, or producing a model you will have to throw away and rewind. The economics are exactly the economics of a production system: uptime and efficiency are money. So you monitor it like one.

The failure modes map cleanly onto the production-system playbook, too. A **loss spike** is an error rate climbing. A **throughput regression** is a latency regression. A **straggler** is one slow replica dragging a distributed request. A **dead node** is an instance that fell out of the pool. A **NaN** is a crash — except that in training it is often a *silent* crash, because the job keeps running, keeps burning GPU-hours, and keeps writing checkpoints full of `inf`. Every one of these has a signal that precedes or accompanies it, and the entire discipline of monitoring a long run is: emit that signal, watch it, and alert on the few that need a human.

There is one wrinkle that makes training monitoring harder than web monitoring, and it is worth stating up front because it recurs through this whole post: **the average lies.** A web service with a hundred replicas can often be summarized by aggregate p99 latency. A training job cannot be summarized by aggregate throughput, because the job runs at the speed of its *slowest* rank. Every step ends with a collective — an all-reduce, an all-gather — and a collective is a barrier: the fastest GPU waits for the slowest. So one sick rank does not show up as a small bump in the average; it silently sets the pace for all the others, and the average, which includes fifteen healthy ranks, looks almost fine. The whole back half of this post is about instrumenting *per rank* so that the outlier cannot hide.

## The four signal families

Everything worth putting on a dashboard falls into one of four families. The families are not arbitrary — they follow the physical structure of the job: the *math* (is the model learning?), the *speed* (how fast is the math running?), the *silicon* (are the chips healthy?), and the *plumbing* (is data reaching the chips and are checkpoints leaving them?). Instrument all four and there is essentially no failure mode of a training run that has no signal. Skip one and you have a blind spot exactly the size of that family.

Let me walk each one: what it contains, what a healthy value looks like, and what the bad value is trying to tell you.

### Family 1 — training health: is it learning?

This is the family everyone already watches, because it is the family that decides whether the run was worth doing. But most people watch only the loss, and the loss is the *least* early of the training-health signals. By the time the loss visibly misbehaves, the thing that caused it happened tens or hundreds of steps ago. The earlier signals are in the gradients.

The signals, roughly in order of how early they warn you:

- **Gradient norm** — the single most valuable early-warning signal in the whole run. The global grad norm is $\lVert g \rVert_2 = \sqrt{\sum_i \lVert g_i \rVert_2^2}$ over all parameter gradients, and you get it for free: `torch.nn.utils.clip_grad_norm_` *returns* it. A healthy run has a grad norm that is noisy but stationary — it wanders around some baseline. When it starts a sustained *climb*, the run is becoming unstable, and it will typically climb for tens to hundreds of steps before it finally produces a value large enough to overflow and NaN. That climb is your window. Alert on the climb and you can lower the learning rate or skip the bad batch *before* the loss ever moves.
- **Loss** — the headline, but a lagging one. Watch it for the slow story (is it descending on schedule?) and for the sharp story (a sudden spike means a bad batch or an instability that already fired). A loss that goes flat is its own failure: the model has stopped learning, usually because the learning rate is wrong or the data is broken.
- **Learning rate** — not a health signal so much as *context* for the others. A grad-norm spike right at the peak of the warmup is a different story than one deep in the decay. Always plot the LR schedule alongside the loss and grad norm so you can read them together.
- **Parameter norms and update ratios** — the ratio of update magnitude to parameter magnitude, per layer, tells you whether some layer is being pushed far harder than the rest. A layer whose param norm is exploding while the others are stable is a localized instability you would never see in the global loss.
- **Eval metrics** — periodic downstream evaluation (perplexity on a held-out set, a few task accuracies). Cheaper to run than people think and the only signal that catches "the loss is descending beautifully but the model is memorizing / the eval set leaked / the data is subtly wrong."

The question this family answers: **is it learning, is it diverging, is it about to NaN?** Grad norm answers the last two earliest. If you instrument nothing else in this family beyond loss, instrument grad norm.

### Family 2 — throughput: is it fast?

This family is the subject of its own chapter — [throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions) — so I will be brief and defer the deep treatment there. The signals:

- **Tokens per second** — the north-star progress metric, because the model learns from tokens, not steps. Crucially, watch tokens per second and *not* steps per second: if your batches are variable-length, a shard of long documents makes each step heavier while the tokens-per-second is flat, and steps-per-second will read that as a phantom regression.
- **MFU (Model FLOPs Utilization)** — tokens per second normalized by the hardware's theoretical ceiling, so it is comparable across model size and GPU type. For a dense transformer with $N$ parameters on $G$ GPUs each with peak $P_\text{peak}$, MFU is
  $$\text{MFU} = \frac{6 N \cdot (\text{tokens/s})}{G \cdot P_\text{peak}}$$
  using the standard `6N` FLOPs-per-token approximation (roughly `2N` forward, `4N` backward). MFU is the number to put on an incident, because "throughput dropped from 84,700 to 59,000 tokens/s" means nothing without the hardware, but "MFU dropped from 45% to 31%" is portable and sanity-checkable.
- **Step time** — wall-clock per optimizer step, and specifically **per-rank step time**, which is the straggler detector. The aggregate throughput hides a slow rank; the distribution of per-rank step times does not.

The question this family answers: **is it fast, and is it regressing?** A sustained MFU drop of more than about ten percent is a real regression worth investigating — usually a straggler, a thermal problem, a data-loader stall, or a comms fallback.

### Family 3 — hardware: is a GPU sick?

This is the family almost nobody instruments until the first time a bad GPU costs them a run, and then they instrument it forever. It comes not from your training loop but from the GPU itself, via NVIDIA's **DCGM** (Data Center GPU Manager) or, more crudely, `nvidia-smi`. The signals, per GPU:

- **SM utilization and memory utilization** — how busy the compute units and memory system are. Healthy training pins SM utilization high and steady. Utilization that *periodically drops to zero* is the classic loader-stall signature: the GPU finished its work and is sitting idle waiting for the next batch. Utilization that is chronically mediocre is a kernel-efficiency or comms problem.
- **Temperature and clock speed** — the throttle detector. GPUs reduce their clock frequency when they get too hot (thermal throttling) or when they hit a power limit. A GPU running at 88–90°C with its clock notched down from its boost frequency is doing *less compute per second* than its healthy neighbors, which makes it a straggler that only the hardware panel can see. The training-health and throughput families will never tell you *why* a rank is slow; the temperature and clock will.
- **Power draw** — a coarse proxy for how much work each GPU is doing; a GPU drawing far less power than its peers is either idle or throttled.
- **ECC errors** — memory error counts. HBM (the GPU's high-bandwidth memory) has error-correcting code that silently fixes single-bit flips (correctable errors) but *cannot* fix double-bit flips (uncorrectable errors). A GPU that starts logging correctable ECC errors is a GPU on its way to failing; an uncorrectable error will typically crash your process, corrupt a tensor, or — in the worst case — silently corrupt an all-reduce and inject a NaN. Rising ECC counts are the earliest warning that a specific GPU is dying.
- **XID errors and PCIe replay counts** — driver-level and link-level fault indicators. An XID error in the kernel log is the GPU telling you something went wrong at the hardware or driver level; certain XIDs (like a fallen-off-the-bus or an ECC page-retirement) mean that GPU is done for this run.

The question this family answers: **is a specific GPU sick, throttling, or dying?** Because this is the family that lets you name the *rank* and *node* to evict — connecting directly to [the straggler](/blog/machine-learning/distributed-training/the-straggler) and, when a bad GPU corrupts an all-reduce, to [a silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale).

### Family 4 — system: is it fed?

The last family is the plumbing that keeps the GPUs busy and the run recoverable. It comes partly from your training loop and partly from the orchestration layer:

- **Data-loader wait time** — the fraction of each step spent *waiting for data* rather than computing. This is the single most common cause of unexplained MFU below target: the GPUs are idle a slice of every step because the `DataLoader` cannot produce the next batch fast enough. Measure it directly (time the `next(iterator)` call), because it is invisible in every other family — a data-starved GPU looks like a slow-but-busy GPU from the outside. This connects to [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale).
- **Checkpoint success and duration** — did the last checkpoint actually write, to all shards, without error, and how long did it take? A checkpoint that silently fails is a landmine: you find out only when a node dies and you try to resume from a checkpoint that was never completely written. Emit a success signal on every save and alert if it is ever missing.
- **Node liveness and NCCL health** — is every rank still alive and participating in collectives? A hung all-reduce, a NCCL timeout, or a node that fell out of the job are the failures that turn a running job into a *stalled* job that keeps holding the allocation without making progress. A heartbeat per rank and a watchdog on collective completion catch these.
- **Restart and interruption count** — on a long run, hardware *will* fail; the question is whether your job survives it. Tracking how often the job has restarted (and why) is a health metric in its own right and ties to [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training).

The question this family answers: **is the pipeline fed and is the run recoverable?** These are the signals that separate a job that loses ten minutes to a dead node from one that loses a whole night.

The most under-measured signal in this family is data-loader wait, so it is worth showing exactly how to capture it, because it is invisible everywhere else. You measure it by timing the moment you *ask* for the next batch — if that call blocks, the GPUs were idle for the duration:

```python
import time, torch

def train_with_loader_timing(model, loader, monitor):
    data_iter = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)            # this is where the GPU starves
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)
        loader_wait = time.perf_counter() - t0  # seconds spent waiting for data

        t1 = time.perf_counter()
        loss = train_step(model, batch)         # forward + backward + optimizer
        compute_time = time.perf_counter() - t1

        wait_frac = loader_wait / (loader_wait + compute_time)
        # a healthy pipeline keeps this near zero; > 0.05 sustained = starving
        monitor.log({"loader_wait_frac": wait_frac})
        if wait_frac > 0.10:
            print(f"[LOADER] GPUs idle {wait_frac*100:.0f}% of the step — "
                  f"raise num_workers / prefetch_factor")
```

A healthy pipeline keeps `wait_frac` near zero: the `DataLoader`'s worker processes stay far enough ahead that the next batch is always ready before the previous step finishes. A `wait_frac` that climbs above five percent and stays there is the GPUs stalling on data — the exact failure that shows up as periodic zero-utilization on the hardware panel and as unexplained MFU below target on the throughput panel, but that *only* this measurement names as a loader problem rather than a slow-GPU problem.

## Metric to failure signature

The four families give you the panels. The skill is reading them: mapping a metric that is *moving* to what actually broke and the one thing to do about it. This is the lookup table I keep pinned next to every long run.

![a table matching each moving training metric to what a bad value means its likely cause and the one action to take](/imgs/blogs/monitoring-a-long-run-2.webp)

Written out, with the reasoning behind each row:

| Signal | Bad value means | Likely cause | First action |
|---|---|---|---|
| Loss spiking up | Instability already fired | Bad batch, LR too high, fp16 overflow | Lower LR, skip/inspect the batch, check the [NaN path](/blog/machine-learning/distributed-training/silent-nan-at-scale) |
| Grad norm climbing | Instability building, pre-NaN | LR too high, a bad data shard, fp16 range | **Alert before the NaN** — lower LR now |
| Loss flat, not descending | Not learning | LR too low or zero, frozen params, broken data | Check LR schedule, verify data, overfit one batch |
| MFU dropping > 10% | Real throughput regression | Straggler, thermal, loader stall, comms fallback | Bisect: hardware, then data, then code |
| Per-GPU temp up, clock down | Thermal throttle | Hot rack, fan fault, bad airflow | Cool or evict the node |
| One rank's step time diverges | A straggler | Sick GPU, bad NIC, throttling on that node | Find the rank, evict the node |
| GPU util periodically 0% | Data-loader stall | Too few workers, slow storage, cold cache | More `num_workers`, prefetch, warm the cache |
| ECC errors rising on a GPU | Failing hardware | HBM degrading | Drain that GPU, replace before it corrupts |
| Node heartbeat missing | Dead node / hung collective | Hardware failure, NCCL timeout | Restart from checkpoint, evict the node |
| Checkpoint success missing | Silent save failure | Storage full, permission, one shard failed | Fix storage, verify the checkpoint restores |

Two rows deserve emphasis because they are where the real time is saved. The **grad-norm-climbing** row is the highest-leverage alert in the whole table: it fires *before* the loss moves and *before* the NaN, which is the difference between lowering the learning rate in a calm moment and rewinding two hours of a dead run. And the **MFU-dropping** row is deceptively hard, because "MFU dropped" has at least four plausible causes across three different families — which is exactly why you need all four families instrumented, so you can look at the hardware panel and the loader-wait panel *at the same time* and see which one moved.

## The tooling: what emits the metrics and where they go

Knowing what to watch is half the job; the other half is a pipeline that actually collects it. There are three kinds of emitter and one store, and they compose into a small system.

![a dataflow where the training loop per-rank timing and a hardware poller merge into one metric store that feeds a dashboard and an alerter](/imgs/blogs/monitoring-a-long-run-3.webp)

The three emitters, as the figure shows, merge into one store, which then fans back out to a dashboard humans watch and an alerter that pages them:

1. **The training loop** logs the training-health and throughput signals — loss, grad norm, LR, tokens/s, MFU. On rank 0 for the scalars that are already all-reduced (loss), and **per rank** for anything that can differ across ranks (step time, peak memory). This is code you write; it is the first snippet below.
2. **A hardware poller** logs the DCGM signals — per-GPU util, memory, temperature, clocks, power, ECC. This runs as a sidecar or a background thread, independent of the training loop, because you want hardware metrics *even when the training loop is hung*. This is the second snippet.
3. **The orchestration layer** (SLURM, your launcher, `torchrun`) logs liveness, restarts, and node events.

All three write to a **metric store**, and here is the honest comparison of the three you will actually choose between:

| Store | Strength | Weakness | Reach for it when |
|---|---|---|---|
| Weights & Biases | Zero-setup, great UI, run comparison, built-in alerts | Hosted (or self-host is heavy), per-seat cost | Default for research runs; you want it working today |
| TensorBoard | Free, local, ships with PyTorch | No alerting, weak at multi-run/multi-rank, scales poorly | Small runs, single node, no infra budget |
| Prometheus + Grafana | Industry-standard alerting, scrapes DCGM natively, per-rank labels | You run the infra; not ML-run-aware out of the box | Production clusters; you already have the ops stack; you need real alert routing |

In practice the common setup is **both**: Weights & Biases (or TensorBoard) for the *training-health* and *throughput* signals that an ML engineer stares at, and **Prometheus + Grafana scraping DCGM** for the *hardware* signals and the *alerting*, because Grafana's alert routing (to PagerDuty, Slack, or a phone) is what actually pages a human at 3am. NVIDIA ships a `dcgm-exporter` that turns DCGM into Prometheus metrics with one container, which is why the hardware family tends to live in Grafana even when the loss lives in W&B.

### The training-loop metrics logger

Here is the logger I paste into every run. It captures the grad norm from `clip_grad_norm_`'s return value (free — you were clipping anyway), times the step, computes tokens/s and MFU, reads peak memory, and — critically — gathers per-rank step time so a straggler cannot hide in an average. It logs to Weights & Biases and to the console.

```python
import time, torch, torch.distributed as dist

H100_BF16_PEAK = 989e12  # bf16 dense FLOP/s per GPU (approx, vendor headline)

def transformer_flops_per_token(num_params: int) -> float:
    return 6 * num_params  # 2N fwd + 4N bwd, the standard approximation

class RunMonitor:
    def __init__(self, num_params, num_gpus, peak_flops=H100_BF16_PEAK,
                 tokens_per_step=None, use_wandb=True):
        self.num_params = num_params
        self.num_gpus = num_gpus
        self.peak = peak_flops
        self.tokens_per_step = tokens_per_step
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.use_wandb = use_wandb and self.rank == 0
        if self.use_wandb:
            import wandb; self.wandb = wandb
        self._t = None

    def step_start(self):
        torch.cuda.synchronize()          # do NOT time async launches
        self._t = time.perf_counter()

    def step_end(self, step, loss, grad_norm, lr, tokens_this_step=None):
        torch.cuda.synchronize()          # wait for the step to truly finish
        dt = time.perf_counter() - self._t
        tokens = tokens_this_step or self.tokens_per_step

        # per-rank step time -> all_gather so rank 0 sees the whole distribution
        local = torch.tensor([dt], device="cuda")
        if self.world > 1:
            gathered = [torch.zeros_like(local) for _ in range(self.world)]
            dist.all_gather(gathered, local)
            step_times = torch.cat(gathered).tolist()
        else:
            step_times = [dt]

        peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
        torch.cuda.reset_peak_memory_stats()

        if self.rank == 0:
            slowest, fastest = max(step_times), min(step_times)
            tok_s = tokens * self.world / slowest        # job runs at the slowest rank
            mfu = transformer_flops_per_token(self.num_params) * (tok_s) \
                  / (self.num_gpus * self.peak)
            straggle = slowest / fastest                 # 1.0 = perfectly balanced
            metrics = {
                "loss": float(loss), "grad_norm": float(grad_norm), "lr": lr,
                "tokens_per_sec": tok_s, "mfu": mfu,
                "step_time_slowest": slowest, "step_time_fastest": fastest,
                "straggle_ratio": straggle, "peak_mem_gb": peak_mem,
            }
            if self.use_wandb:
                self.wandb.log(metrics, step=step)
            if step % 20 == 0:
                print(f"step {step:>6} | loss {loss:6.3f} | gnorm {grad_norm:6.2f} "
                      f"| {tok_s/1e3:6.1f}k tok/s | MFU {mfu*100:4.1f}% "
                      f"| straggle {straggle:4.2f} | mem {peak_mem:4.1f}GB")
            return metrics
```

The two lines that people get wrong and that make every number a lie are the `torch.cuda.synchronize()` calls. CUDA kernels launch asynchronously; if you call `time.perf_counter()` without synchronizing, you are timing how long it took to *enqueue* the work, not to *do* it, and your step times will be nonsense. Synchronize before you start the timer and before you stop it. (Do this only in the monitored steps if the sync cost matters; in practice one sync per step is negligible next to the step itself.)

The other subtlety is `tok_s = tokens * self.world / slowest`. Throughput is computed from the **slowest** rank's step time, not the average, because that is the rank the whole job waits for at the barrier. The `straggle_ratio` — slowest over fastest — is the cheapest straggler alarm you can build: 1.0 is perfectly balanced, and anything sustained above about 1.2 means one rank is dragging the run.

### The hardware poller

The training loop cannot report a throttling GPU, because from inside the loop a throttled GPU just looks like a slow one. You need to ask the *hardware* directly. Here is a poller using `pynvml` (the Python binding for NVML, the same library `nvidia-smi` uses) that reads temperature, clocks, power, utilization, and ECC error counts per GPU and logs them. Run it as a background thread on every node.

```python
import time, threading
import pynvml

def poll_gpu_hardware(interval=15, logger=None):
    """Background hardware poller. One reading per GPU every `interval` seconds.
    Logs temp, clock, power, util, and ECC error deltas. Never touches CUDA
    contexts, so it keeps reporting even if the training loop is hung."""
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    last_ecc = [0] * n
    while True:
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu   # %
            # correctable + uncorrectable ECC (aggregate, all memory locations)
            try:
                ecc = pynvml.nvmlDeviceGetTotalEccErrors(
                    h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_AGGREGATE_ECC)
            except pynvml.NVMLError:
                ecc = 0
            ecc_delta, last_ecc[i] = ecc - last_ecc[i], ecc

            reading = {f"gpu{i}/temp_c": temp, f"gpu{i}/sm_clock_mhz": sm_clock,
                       f"gpu{i}/power_w": power, f"gpu{i}/util_pct": util,
                       f"gpu{i}/ecc_uncorrected_delta": ecc_delta}
            if logger:
                logger(reading)
            # local alarms the poller can raise without a metric store
            if temp >= 87:
                print(f"[HW] gpu{i} HOT {temp}C clock {sm_clock}MHz — throttle risk")
            if ecc_delta > 0:
                print(f"[HW] gpu{i} UNCORRECTED ECC +{ecc_delta} — failing memory")
        time.sleep(interval)

# launch once per node, e.g. from the local_rank==0 process:
threading.Thread(target=poll_gpu_hardware, kwargs={"interval": 15},
                 daemon=True).start()
```

If you would rather not run a Python poller, the same data is one shell command, which is exactly what a Prometheus `dcgm-exporter` scrapes internally:

```bash
# stream per-GPU temp, SM clock, power, util, ECC every 5s to the log
dcgmi dmon -e 150,100,155,203,319 -d 5000
# 150=GPU temp  100=SM clock  155=power  203=GPU util  319=uncorrected ECC

# or the nvidia-smi equivalent, one CSV line per GPU
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,\
utilization.gpu,ecc.errors.uncorrected.aggregate.total \
  --format=csv,noheader,nounits -l 5
```

The rule that makes the hardware poller earn its keep: **it must run independently of the training process.** If your only hardware visibility is inside the training loop and the loop hangs on a stuck collective, you lose exactly the metrics you need to diagnose *why* it hung. A sidecar container, a `dcgm-exporter`, or a `daemon=True` thread with its own NVML context all survive a hung loop. Your training-loop logger will go silent the moment the job stalls; the hardware poller keeps talking.

## The average lies: per-rank versus aggregate

This is the recurring theme of the whole series and the single most important idea in this post, so it gets its own section and its own figure. The claim: **rank-0 metrics and cluster averages will hide the exact failure you most need to catch.**

![a comparison of an averaged throughput panel that looks healthy against per-rank panels that reveal one node throttling](/imgs/blogs/monitoring-a-long-run-4.webp)

Consider the throttling-node story from the intro, drawn out in the figure. The left panel is what most dashboards show by default: one aggregate throughput line. It reads 82,000 tokens/s, MFU 44%, comfortably inside the healthy range. Nothing on that panel is alarming, which is precisely why the problem ran for four days. The right panel is the same moment instrumented per rank: sixteen step-time lines instead of one, and fifteen of them sit together while rank 11 rides consistently 1.4× higher. Cross-reference the hardware panel for that rank and there it is — 88°C, SM clock notched down to 1200 MHz from its boost. The throttle was always there; the *average* is what hid it.

The mathematics of why the average is so misleading is worth making explicit. Suppose sixteen ranks, fifteen doing a step in 100ms and one throttled rank doing it in 140ms. The job's step time is set by the barrier — the slowest rank — so it is 140ms, a 40% throughput loss. But the *average* step time is $(15 \times 100 + 140)/16 = 102.5$ms, a mere 2.5% above baseline. So a failure that is costing you 40% of your throughput shows up as a 2.5% wobble in the metric most dashboards plot. The average does not just soften the signal; it divides it by the world size. At sixty-four ranks a rank running 40% slow moves the average by well under one percent — completely inside the noise. **The straggler is invisible to the mean by construction.**

The fix is in the logger above: `all_gather` the per-rank step time and plot the whole distribution, or at minimum the max and the `straggle_ratio`. Three views make the outlier impossible to hide:

- **Per-rank step time**, all ranks on one chart (or a heatmap of rank × time). The straggler is the line that separates from the pack.
- **The straggle ratio** (slowest / fastest) as a single scalar you can alert on. Above ~1.2 sustained, something is wrong.
- **Per-GPU hardware** (temp, clock, ECC) laid out by rank, so the moment step-time diverges you can look at the *same rank's* silicon and see whether it is a throttle, a dying GPU, or a bad NIC.

Here is the straggler detector as a standalone check, the thing you run when the straggle ratio trips. It gathers every rank's recent step time and names the outliers by rank so you know exactly which node to drain — the same idea developed in depth in [the straggler](/blog/machine-learning/distributed-training/the-straggler).

```python
import torch, torch.distributed as dist

def detect_stragglers(recent_step_time: float, threshold: float = 1.20):
    """Every rank calls this with its own recent median step time.
    Rank 0 returns the list of (rank, ratio) whose step time exceeds
    `threshold` * the fastest rank. Empty list = balanced."""
    rank, world = dist.get_rank(), dist.get_world_size()
    local = torch.tensor([recent_step_time], device="cuda")
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    times = torch.cat(gathered)                 # [world]

    fastest = times.min().item()
    ratios = (times / fastest).tolist()
    stragglers = [(r, round(x, 3)) for r, x in enumerate(ratios) if x > threshold]

    if rank == 0 and stragglers:
        for r, ratio in stragglers:
            node = r // torch.cuda.device_count()  # which node this rank lives on
            print(f"[STRAGGLER] rank {r} (node {node}) at {ratio:.2f}x the fastest — "
                  f"check temp/clock/ECC on that GPU")
    return stragglers if rank == 0 else []
```

## Alerting: what wakes you at 3am and what shouldn't

Dashboards are what you look at when you are already worried. **Alerts** are what make you worried at the right time — including at 3am when you are asleep and the run has another twelve days to go. Getting alerting right is a discipline of *restraint*: the failure mode of alerting is not missing a page, it is sending so many pages that people mute the channel, and then the one page that mattered gets muted with the rest. The single rule that governs everything: **alert on things that need a human to act, and on nothing else.**

That rule sorts every candidate alert into three buckets:

- **Page (wake a human)** — the run is dead or about to die, and a human must intervene *now* or you lose hours. NaN/inf loss, a node down or a NCCL timeout, a grad-norm spike crossing the danger threshold, an uncorrected ECC error, a checkpoint that failed to write.
- **Ticket (tell a human, don't wake them)** — something is degraded but the run is still making progress. A sustained MFU drop above the threshold, a rising correctable-ECC count on a GPU, a straggle ratio that is high but not catastrophic. These get looked at in the morning.
- **Log only (record, never notify)** — everything else. A single slow step, a transient temperature blip, a one-off loader stall. Record it so you can investigate patterns later; never notify on it.

Here are the alerts that have earned a page, with the threshold and the reasoning:

| Alert | Threshold | Severity | Why this threshold |
|---|---|---|---|
| Loss is NaN or inf | Any occurrence | Page | The run is producing garbage; every further step wastes GPU-hours and poisons checkpoints |
| Grad norm spike | > 4× the trailing median, sustained 3+ steps | Page | Catch the instability *before* the NaN; the multiplier avoids paging on single-step noise |
| Node down / NCCL timeout | Heartbeat missing > 2 min, or collective timeout | Page | The job is stalled holding the whole allocation; every minute is the full cluster idle |
| Uncorrected ECC error | Any occurrence on any GPU | Page | The memory is failing; it may already have corrupted a tensor or an all-reduce |
| Checkpoint failed | Any save that did not confirm all shards | Page | Silent until you need it; a run with no valid checkpoint is a run you cannot recover |
| MFU regression | Drop > 15% from trailing baseline, sustained 10+ min | Ticket | Real but not fatal; investigate in the morning unless it is also a straggler |
| Straggle ratio high | Slowest/fastest > 1.25, sustained 10+ min | Ticket | One node is dragging the run; drain it at the next checkpoint, no need to wake anyone |
| Correctable ECC rising | > 100/hour on one GPU | Ticket | That GPU is degrading; schedule it out before it produces an uncorrected error |

The two thresholds worth dwelling on are the grad-norm spike and the loss NaN, because together they are the difference between a saved run and a lost one. You alert on the grad-norm spike as a **page** even though the run is still technically fine, precisely *because* it is the early warning — by the time the NaN fires, the window to lower the learning rate and continue has closed and you are into rewind-and-restart. The grad-norm alert is you catching the fire while it is still smoke.

Here is that alert and the NaN guard together — the guard checks for NaN/inf, and if it finds one, it *checkpoints the last known-good state and raises*, so you do not spend another second training on `inf`:

```python
import math, collections, torch, torch.distributed as dist

class InstabilityGuard:
    def __init__(self, grad_norm_window=50, spike_mult=4.0, alerter=None):
        self.history = collections.deque(maxlen=grad_norm_window)
        self.spike_mult = spike_mult
        self.alert = alerter or (lambda msg: print(f"[ALERT] {msg}"))
        self.consecutive_high = 0

    def check(self, step, loss, grad_norm, save_fn=None):
        loss_v = float(loss)
        # --- NaN / inf guard: a PAGE, and we stop before wasting more steps ---
        if math.isnan(loss_v) or math.isinf(loss_v):
            self.alert(f"NaN/inf loss at step {step} — halting")
            if save_fn and dist.get_rank() == 0:
                save_fn(tag=f"preNaN_step{step}")   # keep the corpse for a postmortem
            raise FloatingPointError(f"loss={loss_v} at step {step}")

        # --- grad-norm spike: page if we climb above N x the trailing median ---
        g = float(grad_norm)
        if len(self.history) >= 10:
            median = sorted(self.history)[len(self.history) // 2]
            if g > self.spike_mult * median:
                self.consecutive_high += 1
                if self.consecutive_high >= 3:       # 3 steps -> not just noise
                    self.alert(f"grad norm {g:.1f} > {self.spike_mult}x median "
                               f"{median:.1f} at step {step} — lower LR NOW")
            else:
                self.consecutive_high = 0
        self.history.append(g)
        return self.consecutive_high >= 3
```

And on the infrastructure side, the Prometheus alerting rules that watch the hardware family scraped from `dcgm-exporter` — the alerts that fire without any Python at all, which matters because they keep working when the training process is dead:

```yaml
groups:
  - name: training-hardware
    rules:
      - alert: GpuThermalThrottle
        expr: DCGM_FI_DEV_GPU_TEMP > 87
        for: 2m
        labels: { severity: ticket }
        annotations:
          summary: "GPU {{ $labels.gpu }} on {{ $labels.instance }} at {{ $value }}C"

      - alert: GpuUncorrectedEcc
        expr: increase(DCGM_FI_DEV_ECC_DBE_AGG_TOTAL[5m]) > 0
        for: 0m
        labels: { severity: page }
        annotations:
          summary: "Uncorrected ECC on GPU {{ $labels.gpu }} — failing memory"

      - alert: NodeHeartbeatLost
        expr: up{job="training"} == 0
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Rank on {{ $labels.instance }} stopped reporting — job may be hung"
```

Notice the design choices baked into those rules. The thermal alert is a `ticket` with a two-minute `for:` clause — a GPU that touches 88°C for one scrape is noise; one that holds it for two minutes is really throttling. The uncorrected-ECC alert is a `page` with `for: 0m` — zero tolerance, fire immediately, because a double-bit flip may have already corrupted a tensor. The design *is* the discipline: severity and duration encode "does a human need to act, and how fast."

## Worked example: a grad-norm climb, caught before the NaN

Let me put the grad-norm alert through a real scenario, because this is the single highest-value thing monitoring does for a long run.

![a timeline of grad norm climbing over several hundred steps until an alert fires and the learning rate is lowered before a NaN](/imgs/blogs/monitoring-a-long-run-5.webp)

The setup: a 7B model, fp16 mixed precision (which has a narrow dynamic range and is prone to this), 64 H100s, day six of a run. The trailing-median grad norm has been sitting around 0.4 for days.

- **Step 8000** — grad norm 0.4, business as usual. The timeline in the figure starts here.
- **Step 8300** — grad norm 2.1. That is above baseline but under the 4× spike multiplier (4 × 0.4 = 1.6... it has crossed, but not for three consecutive steps yet). The `consecutive_high` counter starts ticking. A single elevated reading is not an alert; the run *is* getting noisier.
- **Step 8500** — grad norm 8.7, and it has been above 1.6 for well more than three consecutive steps. The guard fires a **page**: "grad norm 8.7 > 4× median 0.4 — lower LR NOW." An engineer (or, better, an automated policy) sees it.
- **Step 8520** — the learning rate is cut by 3× and the last few batches are inspected; one shard had a run of pathological sequences. The run continues from where it was — no rewind, no restart.
- **Step 8800** — grad norm back to 0.5. The run is stable again. Total cost: a slightly lower LR for a few hundred steps and fifteen minutes of an engineer's attention.

Now the un-monitored counterfactual, the same instability with no grad-norm alert:

- The grad norm climbs exactly the same way, but nobody is watching it — the dashboard shows loss, and the loss is *still descending* at step 8500 because the spike has not propagated yet.
- **Step 8600** — grad norm finally exceeds the fp16 range, a gradient overflows to `inf`, the loss becomes `NaN`. Now it is visible, but now it is too late: the optimizer has already stepped on garbage.
- The job keeps running — training loops do not crash on NaN by default, they cheerfully keep multiplying `inf` — writing NaN checkpoints for however long until someone notices, potentially hours.
- Recovery is a rewind to the last good checkpoint (say two hours back), plus the time to notice, plus re-launching 64 GPUs. Call it **two-plus GPU-hours per GPU lost** — 128+ GPU-hours across the cluster, several hundred dollars, and a chunk of a night.

![a comparison of an unmonitored run that hits a NaN and loses hours against a monitored run that pages early and keeps going](/imgs/blogs/monitoring-a-long-run-6.webp)

The two runs differ by exactly one thing, as the figure lays out side by side: an alert on the *climb* rather than on the *NaN*. The grad norm gave hundreds of steps of warning; monitoring is what turned that warning into an action. This is why the grad-norm alert is a page even though the run is still nominally healthy when it fires — the whole value is in acting during the smoke, not the fire.

## Worked example: the MFU dashboard that found a throttling node

The second example is the intro story, now with the numbers and the resolution, because it is the canonical case for *per-rank* instrumentation.

The setup: 64 H100s, a 7B model, running at a healthy 45% MFU (≈84,700 tokens/s) since launch. On day five, aggregate throughput sagged to about 79,000 tokens/s — a 7% drop, MFU from 45% to about 42%. On the *aggregate* dashboard this was barely perceptible: a line that was flat before now had a gentle downward tilt, well inside the day-to-day noise band of a real run. No alert fired, because a 7% aggregate drop with a `for: 10m` sustained clause on a 15%-threshold ticket does not trip. It sat there for four days.

What the aggregate hid: one GPU on one node had been thermal-throttling since a cooling event. Its step time was about 1.4× the others — 140ms against 100ms. Recall the arithmetic from the per-rank section: one 1.4× rank out of 64 moves the *aggregate average step time* by well under one percent, but it sets the *barrier* step time for the whole job. So how did throughput drop 7% and not the naive ~40%? Because this cluster used gradient accumulation and the throttled GPU was slow but not catastrophically so, and the barrier cost was partially amortized across the accumulation micro-steps — the real-world number was a 7% job-level loss, not the textbook 40%, but still real, still four days long, still thousands of dollars.

The resolution, once someone looked at the right panel:

- **Open the per-rank step-time chart.** Fifteen nodes' ranks cluster together; rank 11 rides consistently higher. Sixty seconds to see it.
- **Cross-reference the hardware panel for rank 11's GPU.** Temperature 88°C, SM clock pinned at 1200 MHz instead of its ~1980 MHz boost. There is the throttle, named down to the exact GPU.
- **Drain and evict at the next checkpoint.** The elastic launcher ([fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training)) restarts the job on 60 healthy GPUs (or a hot spare swaps in), MFU returns to 45%, and a ticket goes to data-center ops to check the rack cooling.

Total avoidable loss: four days at 7% on a 64-GPU cluster ≈ 4 × 24 × 64 × 0.07 ≈ 430 GPU-hours, several hundred to a couple thousand dollars depending on your rate. Avoidable in sixty seconds *if* the per-rank panel and the straggle-ratio alert had existed. They did not, so the average hid it for four days. The fix was not a smarter model or a faster kernel — it was a `straggle_ratio > 1.25` ticket alert that would have fired on day five and a per-rank chart to confirm it. That is the entire return on instrumenting per rank.

## Case studies: what the big runs actually logged

The reliability numbers from published large-scale runs make the case better than any argument, because they show that at scale, hardware failure is not an edge case — it is the weather.

**Llama 3 (Meta, 2024).** The Llama 3 paper reports, for a 54-day snapshot of pre-training on a cluster of 16,384 H100 GPUs, **419 unexpected interruptions** (plus 47 planned), and attributes roughly **78% of the unexpected ones to hardware issues** — with GPUs and their HBM3 memory the single largest category. That is on the order of eight unexpected hardware interruptions *per day*. And yet they report maintaining **over 90% effective training time**, which is only possible with exactly the monitoring-and-recovery machinery this post describes: detect the failure fast, checkpoint often, and restart elastically without a human in the loop for the routine cases. The lesson is stark — at 16k GPUs the question is not *whether* hardware fails during your run but how many times per day, and your effective throughput is decided by how fast you detect and recover.

**OPT-175B (Meta, 2022).** The OPT logbook — released publicly, and required reading — documents a run on 992 A100 80GB GPUs that was interrupted constantly: dozens of loss spikes requiring manual intervention (lower the LR, roll back, skip data), and a steady drumbeat of hardware failures that cycled the team through a large pool of spare hosts over the run. The logbook reads like an on-call diary precisely because that is what operating a large run is. Many of the loss spikes were caught and mitigated by watching exactly the training-health signals in Family 1; the ones that were caught *late* cost rollbacks.

**BLOOM (BigScience, 2022).** The 176B BLOOM run on the Jean Zay supercomputer (384 A100 80GB GPUs for the main run, with spare nodes held in reserve) documented periodic GPU failures on the order of one to two per week and relied on frequent checkpointing plus held-back spare nodes to swap in failed hardware — the reliability pattern that the [fault-tolerance chapter](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) makes concrete.

**PaLM (Google, 2022).** On the throughput side, PaLM reported **46.2% MFU** training 540B parameters on 6,144 TPU v4 chips — a useful anchor for what a *well-tuned* very-large run achieves, and a reminder that "good MFU" at extreme scale is in the 40s of percent, not the 50s. If your 64-GPU run is at 45% MFU you are in healthy territory; the monitoring job is to keep it there and catch the day it slips.

The common thread: every one of these teams treated the run as a system to be operated, not a job to be submitted. They logged the four families, they watched them, and they built the recovery loop that turns "a GPU failed" into "the job restarted in ten minutes" instead of "we lost a night." Cite the numbers with the caveat that exact figures vary by report and I have rounded for clarity — but the order of magnitude (many hardware failures per day at 16k GPUs; MFU in the 40s at extreme scale) is solid.

## When to reach for this — and when it's overkill

Monitoring is a cost like any other: instrumentation code to write, dashboards to maintain, alert thresholds to tune, and a small runtime overhead. Here is the honest calculus of how much is worth it.

**Do the full stack — all four families, per-rank, Prometheus alerting, the works — when** the run is long (days to weeks), wide (tens to thousands of GPUs), or expensive (a real GPU bill). Above roughly a node-week of committed compute, the full instrumentation pays for itself the first time it catches a throttling node or a grad-norm climb, which on a run of that size is essentially every run. At the 16k-GPU scale of the case studies, monitoring is not optional; it is the only thing standing between you and losing the run.

**Scale it down when** the run is short (hours) or small (one node). For a single-node fine-tune that finishes overnight, the training-health family (loss, grad norm) plus a basic tokens/s and a NaN guard is enough — you do not need a Prometheus cluster to watch an eight-hour job. The per-rank machinery still matters the moment you are multi-node, but within one node the aggregate is much closer to the truth (though even within one node, a single throttling GPU still hides in the average — the straggle ratio is cheap enough to keep everywhere).

**Do not skip, ever, even on small runs:** the NaN guard and the grad-norm history. They are ten lines of code, they cost nothing, and the downside they protect against — training for hours on `inf` and writing poisoned checkpoints — is the same on one GPU as on a thousand.

**Do not over-alert.** The most common failure of a monitoring setup is not too few signals but too many pages. If your alert channel has fired more than a couple of times this week for things that turned out not to need action, your thresholds are wrong and you are training your on-call to ignore the channel — which is worse than no alerting, because now the real page gets muted too. When in doubt, make it a ticket, not a page. A signal you look at in the morning is almost always fine; reserve the 3am page for "the run is dead or dying and only a human can save it." This is the same discipline that governs any production on-call rotation, and it applies verbatim here.

The stress test for your monitoring setup is the same as for the run itself: *what happens at 64 GPUs when one node throttles — does the straggle ratio catch it before the aggregate hides it? What happens on PCIe instead of NVLink — does your MFU baseline account for the lower comms bandwidth so you do not chase a phantom regression? What happens when the batch is tiny and the data loader can't keep up — does the loader-wait panel show the GPUs starving? What happens when a node dies at 3am — does the heartbeat alert fire, and does the run restart itself?* If your dashboards and alerts answer all four, you are watching the run. If any of them would go unnoticed, that is your next panel to build.

## The monitoring checklist for a long run

Pulling it together into the thing to set up *before* you launch a multi-week job — instrument these five layers and you have covered essentially every failure mode a long run has.

![a five layer checklist stacking training health throughput hardware system and alert routing](/imgs/blogs/monitoring-a-long-run-7.webp)

The stack, top to bottom:

1. **Training health** — log loss, grad norm (from `clip_grad_norm_`'s return), LR, and periodic eval. Alert: NaN/inf (page), grad-norm spike > 4× median (page).
2. **Throughput** — log tokens/s and MFU (rank 0), and per-rank step time and straggle ratio. Alert: MFU drop > 15% (ticket), straggle ratio > 1.25 (ticket).
3. **Hardware** — run an independent DCGM/`pynvml` poller for per-GPU temp, clock, power, util, ECC. Alert: uncorrected ECC (page), thermal throttle > 87°C sustained (ticket), correctable ECC rising (ticket).
4. **System** — log data-loader wait, checkpoint success per save, and a per-rank heartbeat. Alert: node down / heartbeat lost (page), checkpoint failed (page).
5. **Alert routing** — wire pages to something that wakes a human (PagerDuty, phone), tickets to something that does not (Slack, email). Tune thresholds so the page channel fires only for run-is-dying events. Review the thresholds after the first week — the first run always teaches you which alerts were noise.

Set those five up before launch, not after the first failure, because the failure you did not instrument for is the one that costs you the week.

## Key takeaways

- **A long training run is a production system.** It burns money continuously for weeks, and its failure modes — spikes, regressions, stragglers, dead nodes, silent NaNs — cost you days if caught late and minutes if caught early. Monitor it like one.
- **Instrument four families:** training health (is it learning?), throughput (is it fast?), hardware (is a GPU sick?), and system (is it fed?). Skip a family and you have a blind spot exactly its size.
- **Grad norm is the highest-value signal you are probably not alerting on.** It climbs for tens to hundreds of steps before a NaN. Alert on the climb, not the NaN, and you lower the LR in a calm moment instead of rewinding hours.
- **The average lies.** A straggler that costs 40% of throughput moves the *mean* step time by a fraction of a percent, because the job runs at the barrier — the slowest rank — not the average. Instrument per rank or the outlier is invisible by construction.
- **Watch tokens/s, not steps/s**, and normalize to MFU so the number is comparable across hardware and portable onto an incident report.
- **Run the hardware poller independently of the training loop**, so it keeps reporting temp, clocks, and ECC even when the loop is hung — which is exactly when you need it.
- **Alert on things that need a human, and nothing else.** Sort every candidate into page (dying, act now), ticket (degraded, act in the morning), or log (record, never notify). Over-alerting mutes the one page that mattered.
- **The NaN guard and grad-norm history are ten lines and belong in every run**, one GPU or a thousand — the downside they prevent (hours of training on `inf`, poisoned checkpoints) is scale-independent.
- **At scale, hardware failure is the weather, not the exception.** The published large runs logged many hardware interruptions per day; their >90% effective training time came entirely from fast detection and automatic recovery.

## Further reading

- [Throughput Regressions: When Yesterday's Job Was Faster](/blog/machine-learning/distributed-training/throughput-regressions) — the deep treatment of MFU, tokens/s, and bisecting a slowdown; the sibling to this post's throughput family.
- [The Straggler: One Slow GPU Halving Your Throughput](/blog/machine-learning/distributed-training/the-straggler) — how to find and evict the sick rank the per-rank panel exposes.
- [Silent NaN at Scale](/blog/machine-learning/distributed-training/silent-nan-at-scale) — bisecting a NaN across ranks; where the grad-norm alert leads when the guard fires.
- [Profiling a Distributed Run](/blog/machine-learning/distributed-training/profiling-a-distributed-run) — when the dashboard says "slow" and you need the profiler to say *why*.
- [Fault Tolerance and Elastic Training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) — the recovery half of the story: turning "a node died" into "the job restarted."
- [Why Distributed Training](/blog/machine-learning/distributed-training/why-distributed-training) and [The Distributed Training Playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the series intro and the capstone checklist that ties monitoring to the four walls.
- The [Llama 3 paper](https://arxiv.org/abs/2407.21783) (interruption and reliability statistics), the [OPT-175B logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) (the on-call diary of a real run), and the [NVIDIA DCGM documentation](https://docs.nvidia.com/datacenter/dcgm/latest/) (the hardware-family metrics).

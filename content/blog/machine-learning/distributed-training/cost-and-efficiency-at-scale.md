---
title: "Cost and Efficiency at Scale: What a Training Run Actually Costs, and How to Halve It"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A training run's bill is GPU-hours times price, and GPU-hours are inversely proportional to MFU — so doubling your MFU halves your invoice. Derive the cost equation, price real runs, and work the levers that cut the bill in half."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "mfu",
    "gpu-hours",
    "spot-instances",
    "cost-optimization",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 31
---

Two teams train the same 7-billion-parameter model on the same 800 billion tokens, on the same H100 cluster, at the same cloud price. One team's finance partner signs an invoice for roughly 79,000 dollars. The other signs one for roughly 43,000 dollars. Neither team used a smaller model, fewer tokens, cheaper hardware, or a different cloud. The only thing that differed was a single number that never appears on the invoice: **Model FLOPs Utilization**, or MFU — the fraction of the GPU's theoretical peak that the run actually sustained. One team ran at 30 percent. The other ran at 55 percent. That gap, and nothing else, is 36,000 dollars.

This is the post where the whole series turns into money. Every prior post — overlapping communication with compute, sharding the optimizer, choosing the right parallelism, killing the straggler, keeping the loss stable — was, at bottom, a fight to raise that one utilization number. Here we make the fight quantitative. We will derive the cost equation from first principles, show that at fixed compute the dollar cost is **inversely proportional to MFU**, and then walk the concrete levers that move it: precision, overlap, parallelism choice, spot instances, right-sizing the cluster, and not wasting compute on runs that were doomed at step 500. By the end you will be able to price any run before you launch it, tell whether your cluster is the right size for your deadline, and know which optimization is worth your engineering time and which is a rounding error.

![a layered stack that decomposes a training bill into total compute divided by peak throughput and MFU, then multiplied by the GPU price, ending in dollars proportional to one over MFU](/imgs/blogs/cost-and-efficiency-at-scale-1.webp)

The figure above is the entire post in one column. Total compute at the top is fixed by your model and your token budget — you cannot cheat it. The peak FLOP/s of the GPU is fixed by the hardware you rented. The price is fixed by your contract. The one term in the middle that you control, the one that decides whether the same run costs 79k or 43k dollars, is MFU. Read the stack top to bottom and the thesis falls out: cost is proportional to `$1/\text{MFU}$`. Halve the waste and you halve the bill. This ties directly back to [why we do distributed training at all](/blog/machine-learning/distributed-training/why-distributed-training) — the fourth of the four walls was always *cost too high* — and it is the accounting layer under the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## The cost equation, derived

Let us build the bill from nothing. Three quantities, three multiplications, and the whole thing is exact enough to plan a budget around.

**Step one: total compute.** A forward pass through a dense transformer costs about two floating-point operations per parameter per token — one multiply and one add in each of the matrix multiplications that dominate the model. Call the parameter count `$N$` and the number of training tokens `$D$`. The forward pass is then about `$2ND$` FLOPs. The backward pass computes gradients with respect to both the activations and the weights, and each of those is roughly the same cost as the forward, so the backward is about `$4ND$`. Add them and the total training compute is

$$ C_\text{train} \approx 6 N D. $$

This `$6ND$` rule is one of the most useful approximations in all of ML systems, and it comes straight from the Kaplan scaling-laws work and the [Chinchilla compute-optimal analysis](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling). It also ignores anything that is not the model's own weight matmuls — the optimizer step, layer norms, the data loader — because those are cheap relative to the matrix multiplications.

The one correction worth knowing is attention. The score and value matmuls in each attention layer cost roughly `$12 \cdot L \cdot s \cdot h$` FLOPs per token per layer, where `$L$` is the layer count, `$s$` the sequence length, and `$h$` the hidden dimension — a term that grows with sequence length while the `$6ND$` weight term does not. As a fraction of the total it is on the order of `$s / (6h)$`. For a typical configuration with hidden dimension 4096 and sequence length 2048 that is about 8 percent; at a 32K context it climbs past a quarter of the compute; at 128K it dominates. The practical rule: use `$6ND$` for planning at ordinary context lengths and treat it as a 5-to-15-percent underestimate, but if you train long-context, add the attention term explicitly or your budget will be low — this is exactly why long-context training is disproportionately expensive, and why the [sequence and context parallelism](/blog/machine-learning/distributed-training/sequence-and-context-parallelism) work exists.

**Step two: wall-clock time.** Compute is a count of operations; time is that count divided by the rate at which you perform them. A cluster of `$G$` GPUs, each with a peak throughput of `$P_\text{peak}$` FLOP/s, sustaining a utilization of `$\text{MFU}$`, performs useful model FLOPs at a rate of `$G \cdot P_\text{peak} \cdot \text{MFU}$`. So the wall-clock time to finish the run is

$$ T_\text{wall} = \frac{C_\text{train}}{G \cdot P_\text{peak} \cdot \text{MFU}} = \frac{6 N D}{G \cdot P_\text{peak} \cdot \text{MFU}}. $$

MFU is doing all the work in that denominator. It is defined — the definition comes from the PaLM paper, which introduced the metric precisely so that runs could be compared honestly across hardware — as the achieved model FLOP/s divided by the hardware's theoretical peak FLOP/s. An MFU of 0.5 means half of every GPU's advertised throughput turned into forward-and-backward math; the other half evaporated into communication stalls, memory-bandwidth limits, kernel launch gaps, and pipeline bubbles. Everything else in this series exists to push that fraction up.

**Step three: the bill.** You rent `$G$` GPUs for `$T_\text{wall}$` hours at a price `$p$` per GPU-hour. The cost is the product:

$$ \text{Cost} = T_\text{wall} \cdot G \cdot p = \frac{6 N D}{G \cdot P_\text{peak} \cdot \text{MFU}} \cdot G \cdot p = \frac{6 N D \cdot p}{P_\text{peak} \cdot \text{MFU}}. $$

Stare at that last form, because it contains the two most important facts in this post.

**The number of GPUs cancels.** `$G$` appears in the denominator of the wall-clock and in the numerator of the rental, and it divides out. To first order — holding MFU fixed — a run costs the same number of GPU-hours whether you run it on 64 GPUs or 1024 GPUs. More GPUs finish the run proportionally faster, but you rent proportionally more of them, and the product, the money, is unchanged. GPU count buys you *time*, not *cost*. This is the single most counterintuitive thing about training budgets, and it is why the right question is never "how many GPUs can I get" but "what wall-clock deadline am I paying to hit." The only reason the cost is *not* actually independent of `$G$` is that MFU falls as you add GPUs — scaling efficiency below one — and we will spend a whole section on exactly that correction.

**Cost is inversely proportional to MFU.** The other term in the denominator is the one you own. Double MFU and the cost halves. There is no diminishing return hidden in the algebra: going from 25 to 50 percent MFU saves exactly as large a fraction of the bill as going from 40 to 80 percent would. Every point of MFU is worth the same slice of the invoice, which is why the systems work in this series pays off linearly in dollars and does not plateau.

### Measuring MFU honestly

You cannot manage what you will not measure, and MFU is easy to fool yourself about. The honest way to compute it is from observed throughput. If your run logs `$T$` tokens per second across the whole job, the achieved model FLOP/s is `$6 N T$` (per-token training FLOPs times tokens per second), and

$$ \text{MFU} = \frac{6 N T}{G \cdot P_\text{peak}}. $$

Two traps sink most MFU numbers. First, `$P_\text{peak}$` must be the *dense* number in the precision you are actually using. An H100 SXM does about 989 dense bf16 TFLOP/s; the 1979 TFLOP/s you see on the spec sheet is the 2:4-sparsity marketing figure and no dense transformer touches it. Quote MFU against 989 for bf16, or against the fp8 peak if you train in fp8, and be consistent. Second, measure throughput in *steady state*, not on the first few steps. Call `torch.cuda.synchronize()` before you read the clock, discard the first dozen warm-up steps while cuDNN autotunes and caches fill, and average over a window long enough that a slow data-loader batch or a checkpoint save does not dominate. Here is the measurement I actually drop into a training loop:

```python
import time
import torch

def measure_mfu(model_params, tokens_per_step, step_times_s, world_size,
                peak_flops_per_gpu=989e12):
    """Compute MFU from observed steady-state step times.

    model_params:   N, total parameter count (dense model)
    tokens_per_step: global tokens processed per optimizer step
    step_times_s:   list of per-step wall times, warm-up steps already dropped
    world_size:     G, number of GPUs
    peak_flops_per_gpu: dense peak in the precision you train in
    """
    steady = sorted(step_times_s)[len(step_times_s) // 10:]   # drop fastest decile (noise)
    mean_step = sum(steady) / len(steady)
    tokens_per_s = tokens_per_step / mean_step
    achieved_flops = 6 * model_params * tokens_per_s           # 6ND per second
    peak = world_size * peak_flops_per_gpu
    return achieved_flops / peak, tokens_per_s

# Example: 7B model, global batch 4M tokens/step, ~11.9s/step on 64 H100
mfu, tps = measure_mfu(7e9, 4_000_000, [11.9, 12.0, 11.8, 12.1], 64)
print(f"MFU = {mfu:.1%}, throughput = {tps:,.0f} tok/s")
# MFU = 55.9%, throughput = 336,700 tok/s
```

That number — recomputed every few hundred steps and logged next to the loss — is your north star. It is the same north star as in [throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions), viewed through the lens of money instead of tokens per second. When it drops, your bill is rising in real time.

## What a training run actually costs

Now put numbers in. The prices below are order-of-magnitude cloud figures for mid-2026 and you should treat them as approximate — they move monthly and vary three-fold between providers, commitment levels, and regions — but they are close enough to plan with.

| Instance | Rough on-demand | Reserved (1yr) | Spot / preemptible |
|---|---|---|---|
| A100 80GB SXM | ~\$2.5–4/GPU-hr | ~\$1.5–2.5/GPU-hr | ~\$0.8–1.5/GPU-hr |
| H100 SXM | ~\$3–5/GPU-hr | ~\$2–3/GPU-hr | ~\$1.5–2.5/GPU-hr |
| Owned / on-prem H100 | ~\$1–2/GPU-hr amortized | — | — |

I will use a round \$2.50 per H100-hour as the working on-demand price for the rest of this post, and a dense bf16 peak of 989 TFLOP/s. From the cost equation, the GPU-hours for a run are `$C_\text{train} / (P_\text{peak} \cdot \text{MFU} \cdot 3600)$`, and the dollar cost is that times the price. Here is the same calculation as a function you can paste and adapt:

```python
def run_cost(n_params, n_tokens, mfu, price_per_gpu_hour=2.50,
             peak_flops=989e12):
    """GPU-hours and dollar cost for a dense training run."""
    total_flops = 6 * n_params * n_tokens          # 6ND
    gpu_seconds = total_flops / (peak_flops * mfu) # note: independent of GPU count
    gpu_hours = gpu_seconds / 3600
    dollars = gpu_hours * price_per_gpu_hour
    return gpu_hours, dollars

for name, N, D in [("7B",  7e9,  2.0e12),
                   ("70B", 70e9, 2.0e12),
                   ("175B",175e9,3.0e11)]:
    for mfu in (0.30, 0.45, 0.55):
        h, d = run_cost(N, D, mfu)
        print(f"{name:>4}  MFU {mfu:.0%}  {h:>10,.0f} GPU-hr  ${d:>12,.0f}")
```

Running it produces the cost surface below — the thing every training budget is a point on.

![a matrix of training cost with model and token sizes down the rows and three MFU levels across the columns, each cell showing GPU-hours and dollars](/imgs/blogs/cost-and-efficiency-at-scale-2.webp)

| Model × tokens | Compute (FLOPs) | MFU 30% | MFU 45% | MFU 55% |
|---|---|---|---|---|
| 7B × 2.0T | 8.4e22 | 79k GPU-hr / \$197k | 52k GPU-hr / \$131k | 43k GPU-hr / \$107k |
| 70B × 2.0T | 8.4e23 | 786k GPU-hr / \$1.97M | 524k GPU-hr / \$1.31M | 429k GPU-hr / \$1.07M |
| 175B × 300B | 3.15e23 | 295k GPU-hr / \$737k | 197k GPU-hr / \$491k | 161k GPU-hr / \$402k |

Three things jump out. First, the numbers are large and they are real: a 70B model on a 2-trillion-token budget is a seven-figure run, and the gap between a sloppy 30 percent MFU and a respectable 55 percent is nearly a million dollars for that single row. Second, read any row left to right and the cost falls by the ratio of the MFU values — 55/30 is 1.83, and indeed \$1.97M / \$1.07M is 1.84. The inverse law is not a slogan; it is arithmetic you can check against the table. Third, the 175B row costs *less* than the 70B row despite being a bigger model, because it was trained on far fewer tokens (300B versus 2T). Cost tracks `$N \times D$`, not `$N$` alone — a smaller model trained long can easily out-cost a larger model trained short. Budget by compute, never by parameter count.

### The comparable unit: dollars per million tokens

Absolute run cost is the number finance signs, but it is a terrible unit for comparing efficiency across runs of different sizes, because a bigger or longer run costs more for reasons that have nothing to do with how well it was engineered. The comparable unit — the one that lets you benchmark this run against last quarter's, or your cluster against a competitor's published numbers — is **cost per million tokens processed**. Divide the bill by the token budget:

$$ \frac{\text{Cost}}{\text{million tokens}} = \frac{6 N \cdot p}{P_\text{peak} \cdot \text{MFU}} \times 10^6. $$

Notice what dropped out: the token count `$D$` cancels, leaving a per-million-token cost that depends only on model size, price, and MFU. This is the true efficiency yardstick of your training stack, and it is invariant to how long you train. Two runs of the same model at the same price have the same dollars-per-million-tokens if and only if they hit the same MFU — which makes it a direct, size-normalized report card on your systems work.

```python
def cost_per_million_tokens(n_params, mfu, price_per_gpu_hour=2.50,
                            peak_flops=989e12):
    """Size-normalized training efficiency: dollars per 1M tokens processed."""
    flops_per_token = 6 * n_params                 # 6N, per training token
    gpu_seconds_per_token = flops_per_token / (peak_flops * mfu)
    gpu_hours_per_token = gpu_seconds_per_token / 3600
    return gpu_hours_per_token * price_per_gpu_hour * 1e6

for N, name in [(7e9, "7B"), (70e9, "70B"), (175e9, "175B")]:
    lo = cost_per_million_tokens(N, 0.30)
    hi = cost_per_million_tokens(N, 0.55)
    print(f"{name:>4}:  ${lo:6.2f}/M tok @30% MFU   ->   ${hi:6.2f}/M tok @55% MFU")
# 7B:  $  0.25/M tok @30% MFU   ->   $  0.13/M tok @55% MFU
# 70B: $  2.46/M tok @30% MFU   ->   $  1.34/M tok @55% MFU
# 175B:$  6.14/M tok @30% MFU   ->   $  3.35/M tok @55% MFU
```

The per-million-token cost scales linearly with model size (it is proportional to `$N$`) and inversely with MFU, exactly as the equation says. It is worth computing once for your stack and pinning to the wall: it turns "did this run go well" into a single comparable number, and it is the natural bridge to inference economics, where cost-per-million-tokens is already the standard currency. When someone quotes you a training cost, ask for it per million tokens — that is the only form in which two different runs can be honestly compared.

## The levers that actually move the bill

There are exactly three roots to any cost saving, and every specific tactic hangs off one of them. You either raise MFU (same hardware, more useful work), rent cheaper hours (spot, reserved, owned), or buy fewer hours (right-size the cluster, don't waste compute). The tree below organizes them, and the ordering is deliberate: the branches near the top are the ones that pay the most for the least risk.

![a decision tree rooted at cutting dollars per run that branches into raising MFU, cheaper hours, and fewer hours, with concrete tactics as leaves under each branch](/imgs/blogs/cost-and-efficiency-at-scale-3.webp)

| Lever | Mechanism | Typical saving | Where it lives in this series |
|---|---|---|---|
| Overlap comms with compute | Hide all-reduce/all-gather behind the backward | +10–30 pts MFU | [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) |
| fp8 / bf16 where it validates | Higher peak, less comms volume | 1.3–1.8× throughput | [Mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) |
| Right parallelism, no over-provision | Least parallelism that fits; no needless collectives | +5–20 pts MFU | [Picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) |
| Spot / preemptible instances | Rent at ~1/3 on-demand price | up to 60–70% | [Fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) |
| Right-size the cluster | Stop before scaling efficiency craters | avoids paying 1.5× for time | this post, below |
| Kill bad runs early | No compute on doomed runs | eliminates 100% of wasted spend | [Throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions) |

The reason MFU sits at the top of the tree is that it is the only lever that costs you nothing but engineering time. Spot instances trade price for reclaim risk. A bigger cluster trades money for wall-clock. But raising MFU from 35 to 50 percent is pure profit: the same GPUs, the same run, the same result, 30 percent less money, with no new failure mode introduced. That is why the entire rest of this series — which is almost entirely about raising MFU — is, viewed from finance, the highest-return work an ML-systems engineer does.

### Lever one: raise MFU, and watch the bill fall

Because cost is `$\propto 1/\text{MFU}$`, this lever is worth pricing out concretely.

#### Worked example: the 79k versus 43k dollar run

You are pre-training a 7B model on 800 billion tokens on an H100 cluster at \$2.50 per GPU-hour. The compute is fixed: `$6 N D = 6 \times 7\text{e}9 \times 8\text{e}11 = 3.36\text{e}22$` FLOPs. Nothing you do changes that number; it is set by the model and the data.

Your first launch runs at **30 percent MFU**. Perhaps the gradient all-reduce is not overlapping with the backward, the data loader stalls the GPU between steps, and you left the model in fp32 attention. The GPU-hours are `$3.36\text{e}22 / (989\text{e}12 \times 0.30 \times 3600) = 31{,}500$` GPU-hours. At \$2.50 that is **\$78,600**. On 64 GPUs that run takes `$31{,}500 / 64 = 492$` hours — about 20.5 days.

Now you do the systems work. You enable bucketed gradient overlap so the all-reduce hides behind the backward pass. You switch to bf16 with fp8 matmuls in the linear layers where the loss curve confirms it is safe. You fix the loader to prefetch enough that the GPU never waits. MFU climbs to **55 percent**. The GPU-hours drop to `$3.36\text{e}22 / (989\text{e}12 \times 0.55 \times 3600) = 17{,}200$` GPU-hours — **\$42,900**, and the run now finishes in 11.2 days on the same 64 GPUs.

Same model, same data, same hardware, same price. The difference is **\$35,700 and nine days**, bought with a week of engineering. That is the return on MFU, and it is why "the run is training, ship it" is an expensive sentence.

![a before and after comparison showing a run at thirty percent MFU costing seventy-nine thousand dollars versus the same run at fifty-five percent MFU costing forty-three thousand dollars](/imgs/blogs/cost-and-efficiency-at-scale-4.webp)

The stress test on this lever is honesty about where it saturates. MFU has a practical ceiling — very few real dense runs sustain above 55 to 60 percent, and the best-published large-model runs land in the high 40s to low 50s. If you are already at 50 percent, the remaining headroom to a theoretical 65 is worth only about a 23 percent cost cut, and squeezing it may cost more engineering than it saves. The inverse law cuts both ways: the first 20 points of MFU are enormous, the last 10 are marginal. Know which side of the curve you are on before you spend another week on kernels.

## Spot and preemption: renting at a third of the price

The second lever is the price per hour, and the biggest single move there is spot (or preemptible) instances: the cloud's spare capacity, rented at roughly a third of on-demand, with one catch — the provider can reclaim your machines with a couple of minutes' notice. For a tightly-coupled synchronous training job this sounds fatal, because losing one node kills the whole collective. It is not fatal, if the run can checkpoint and resume. This is where the [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) machinery pays for itself in cash.

The economics are an expected-value calculation. Let the useful work be `$W$` GPU-hours (fixed by the run). Spot costs `$p_\text{spot} = p_\text{od}/3$` per hour. Each reclaim forces you to redo the work since your last checkpoint — on average half a checkpoint interval — plus the time to reload and re-establish the process group. If reclaims arrive at rate `$\lambda$` per wall-clock hour and each one wastes `$c$` hours of wall-clock, the run stretches and you pay for the wasted GPU-time too. The model:

```python
def spot_vs_ondemand(useful_gpu_hours, n_gpus,
                     price_od=2.50, spot_discount=1/3,
                     interrupt_per_hour=0.05,       # job-level reclaim probability/hr
                     ckpt_interval_h=1.0,           # checkpoint every hour
                     restart_h=1/6):                # ~10 min to reload + re-rendezvous
    price_spot = price_od * spot_discount
    useful_wall = useful_gpu_hours / n_gpus
    # each reclaim wastes: half a ckpt interval recomputed + restart idle
    waste_per_event_wall = ckpt_interval_h / 2 + restart_h
    # fixed point: total wall = useful + interruptions * waste
    #   interruptions = interrupt_per_hour * total_wall
    total_wall = useful_wall / (1 - interrupt_per_hour * waste_per_event_wall)
    interruptions = interrupt_per_hour * total_wall
    wasted_gpu_hours = interruptions * waste_per_event_wall * n_gpus
    billed_gpu_hours = useful_gpu_hours + wasted_gpu_hours

    cost_spot = billed_gpu_hours * price_spot
    cost_od = useful_gpu_hours * price_od
    return {
        "on_demand_$": round(cost_od),
        "spot_$": round(cost_spot),
        "spot_wall_days": round(total_wall / 24, 1),
        "od_wall_days": round(useful_wall / 24, 1),
        "wasted_pct": round(100 * wasted_gpu_hours / useful_gpu_hours, 1),
        "interruptions": round(interruptions),
    }

print(spot_vs_ondemand(17_200, 64))
# {'on_demand_$': 43000, 'spot_$': 14733, 'spot_wall_days': 11.6,
#  'od_wall_days': 11.2, 'wasted_pct': 3.4, 'interruptions': 14}
```

#### Worked example: spot on a checkpointed 7B run

Take the same 55 percent MFU run from before: 17,200 useful GPU-hours, 64 GPUs, on-demand cost \$42,900, wall-clock 11.2 days. Move it to spot at \$0.83 per GPU-hour, with a 5 percent per-hour job-level reclaim probability, a checkpoint every 60 minutes, and a 10-minute reload. Over the run you eat about 14 reclaims. Each wastes on average half an hour of recompute plus 10 minutes of restart, so the run stretches from 11.2 to 11.6 days and you burn about 3.4 percent extra GPU-hours redoing lost work. The billed compute rises to ~17,750 GPU-hours, but at one-third the price the bill is **\$14,700** — a **66 percent saving** versus on-demand, for a 3-percent-longer wall-clock.

![a left to right timeline of a spot training run that checkpoints hourly, absorbs a reclaim by reloading, and lands at fifteen thousand dollars versus forty-three thousand on-demand](/imgs/blogs/cost-and-efficiency-at-scale-5.webp)

When does spot stop winning? The break-even is clean. Spot beats on-demand whenever the billed-to-useful multiplier stays below three (since spot is a third of the price). That multiplier is `$1/(1 - \lambda c)$` in the model above, so spot loses only when reclaim rate times waste-per-event approaches two-thirds — an interruption every few hours combined with a checkpoint interval measured in many hours. In practice the price is almost never the thing that kills spot; two other things do.

First, **the run must be able to checkpoint and resume at all.** A synchronous job that has no distributed checkpoint, or whose checkpoint takes 20 minutes to write and reload on a model that gets reclaimed every 40 minutes, spends more time saving and restoring than training. Spot demands cheap, frequent, sharded checkpoints — the distributed-checkpointing discipline that async, sharded saves exist to provide. Second, **reclaim probability compounds across nodes.** A single instance at 5 percent per hour is fine; sixteen instances that each die at 5 percent per hour give a job-level reclaim probability near `$1 - 0.95^{16} \approx 56$` percent per hour if any death kills the job. Elastic training — where the job shrinks to the surviving nodes and re-expands rather than dying — is what keeps large spot runs viable, and without it, spot on a big tightly-coupled job can thrash so hard the wall-clock never converges. Spot is a lever for fault-tolerant, well-checkpointed, ideally elastic runs. For a run you cannot checkpoint, it is a trap.

## Right-sizing the cluster: buying time with money

Recall the cancellation: at fixed MFU, GPU-hours are independent of GPU count. If that held exactly, cluster sizing would not be a cost question at all — you would pick the size that hits your deadline and pay the same either way. It does not hold exactly, because **MFU falls as you add GPUs.** Communication grows, the batch gets split thinner per GPU, collectives take longer relative to compute, and scaling efficiency — the speedup you get divided by the number of GPUs — drops below one. Since GPU-hours are `$W_\text{ideal} / \eta$` where `$\eta$` is scaling efficiency, and `$\eta$` shrinks with `$G$`, the GPU-hours *rise* with cluster size even as the wall-clock falls.

That is the real trade, and it is a Pareto curve, not a free lunch. Here is the computation:

```python
def scaling_cost_curve(n_params, n_tokens, gpu_counts, mfu_at_count,
                       price=2.50, peak=989e12):
    """Wall-clock, GPU-hours, and cost as the cluster grows and MFU falls."""
    total_flops = 6 * n_params * n_tokens
    rows = []
    for g in gpu_counts:
        mfu = mfu_at_count[g]
        gpu_hours = total_flops / (peak * mfu * 3600)
        wall_days = gpu_hours / g / 24
        dollars = gpu_hours * price
        rows.append((g, mfu, wall_days, gpu_hours, dollars))
    return rows

# 7B x 800B run; measured MFU falls as we scale out
mfu_at = {64: 0.55, 256: 0.48, 1024: 0.36}
for g, mfu, wall, h, d in scaling_cost_curve(7e9, 8e11, [64, 256, 1024], mfu_at):
    print(f"{g:>4} GPU  MFU {mfu:.0%}  wall {wall:>4.1f} d  {h:>7,.0f} GPU-hr  ${d:>8,.0f}")
# 64 GPU  MFU 55%  wall 11.2 d   17,158 GPU-hr  $42,895
# 256 GPU  MFU 48%  wall  3.2 d   19,661 GPU-hr  $49,153
# 1024 GPU  MFU 36%  wall  1.1 d   26,214 GPU-hr  $65,536
```

![a three by three grid showing GPU count and MFU across the top, wall-clock in the middle row, and GPU-hours with dollars in the bottom row, as the cluster grows from sixty-four to one thousand GPUs](/imgs/blogs/cost-and-efficiency-at-scale-6.webp)

Read the grid across. Going from 64 to 1024 GPUs collapses the wall-clock from 11.2 days to 1.1 days — more than ten times faster — but the bill climbs from \$43k to \$66k, a 53 percent premium, because MFU eroded from 55 to 36 percent as communication came to dominate. You did not get 16× the speed for your 16× the GPUs; you got about 10×, and you paid a half-again markup on the money for the privilege. That markup is the price of time.

This reframes cluster sizing as a purchasing decision with a clear question: **how much is a day worth to you?** The figure below is the shape of that decision.

![a branching and merging diagram where adding GPUs forks into a faster wall-clock and a falling scaling efficiency that raises GPU-hours, the two branches merging at a chosen Pareto point set by deadline and budget](/imgs/blogs/cost-and-efficiency-at-scale-7.webp)

If a model has to ship for a launch in a week, paying the 53 percent premium to finish in one day instead of eleven is obviously worth it — the deadline value dwarfs \$23k. If it is a research run with no deadline, paying 53 percent extra to finish ten days sooner is usually a poor trade, and you should run on the smaller, more efficient cluster. The Pareto-optimal cluster size is wherever your marginal value of a day equals the marginal dollars that day costs. There is no single right answer, only the right answer *for your deadline*. And crucially, past the point where scaling efficiency has fallen off a cliff — where doubling GPUs barely moves the wall-clock because you are drowning in communication — you are paying strictly more money for strictly no time, and that is never right. That cliff is your hard ceiling on cluster size, and finding it is a matter of measuring MFU at two or three cluster sizes and extrapolating, exactly as [scaling a 7B model from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) walks through.

### The critical batch size: where more GPUs stop helping convergence

There is a second, subtler ceiling on useful cluster size, and it is not about systems efficiency at all — it is about optimization. Scaling out with data parallelism grows the global batch size: each added GPU processes more tokens per step. Up to a point, a larger batch means each optimizer step makes proportionally more progress, so you can take fewer steps and wall-clock falls. But there is a **critical batch size** beyond which doubling the batch no longer halves the number of steps to a given loss — the gradient is already estimated well enough that extra samples are largely redundant. Past that point, adding GPUs to grow the batch buys you almost no reduction in steps, so the run does not converge meaningfully faster, and you are simply paying for GPUs that add tokens the optimizer cannot use.

The practical consequence for cost is sharp: there is a data-parallel width past which more GPUs are pure waste for convergence, independent of any communication overhead. If your global batch is already at 4 million tokens and the critical batch size for your model and learning-rate schedule is around there, throwing another 512 GPUs at pure data parallelism will not finish the run sooner — it will just widen the batch into the regime of diminishing returns. When you hit that wall, the way to keep using more hardware is to switch axes: add pipeline or tensor parallelism to go faster without growing the batch, as the [parallelism-strategy decision framework](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) lays out. Spending GPUs past the critical batch size on data parallelism is one of the quietest ways to burn money in distributed training, because nothing crashes and the throughput graph even looks fine — you are just paying for tokens that do not help.

### Not wasting compute: the cheapest lever of all

The single largest cost saving is the compute you never spend. A run that diverges at step 40,000 and gets killed at step 60,000 wasted a third of its budget. A hyperparameter sweep that launched six configurations when two would have been decisive wasted two-thirds. A loss spike after a checkpoint resume that corrupted the run and forced a restart from an old checkpoint threw away everything in between. None of these show up as low MFU — the GPUs were busy the whole time — but every one is money for nothing.

The defenses are unglamorous and enormously valuable. Log grad-norm and loss at high frequency and alert on divergence so a doomed run dies at step 2,000 instead of 60,000. Run a short "canary" at small scale to validate the config, the data order, and the precision before committing the full cluster. Checkpoint often enough that a crash costs minutes, not days, so a preemption or a node failure never rewinds you far. Get resume-correctness right so a restart does not silently corrupt the run — the [loss-spike-after-resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) failure is a direct tax on the budget. A team that catches bad runs in the first hour and never loses a checkpoint routinely spends 20 to 40 percent less than an equally skilled team that lets doomed runs cook overnight, and the difference is entirely operational discipline, not any deeper systems insight.

## Case studies: what real runs actually cost

The numbers above are models; here are measured runs from public reports, which anchor them to reality. Treat each as approximate and read against its own precision and parallelism assumptions.

**Llama 2 70B — the honest MFU baseline.** Meta reported that pre-training Llama 2 70B on 2 trillion tokens took about 1,720,320 A100-80GB GPU-hours. Back out the MFU: the compute is `$6 \times 70\text{e}9 \times 2\text{e}12 = 8.4\text{e}23$` FLOPs, the GPU-seconds are `$1{,}720{,}320 \times 3600 = 6.19\text{e}9$`, so the achieved throughput was `$8.4\text{e}23 / 6.19\text{e}9 = 1.36\text{e}14$` FLOP/s per GPU, against an A100 peak of 312 TFLOP/s — an MFU of about **43.5 percent**. That is a strong, realistic number for a large dense model on A100, and it tells you the 40-to-45-percent band is where good real runs live, not the 55 percent I use as a stretch target. At a nominal \$2 per A100-hour that run's compute alone is on the order of \$3.4M.

**DeepSeek-V3 — the modern cost-transparency benchmark.** The DeepSeek-V3 technical report is unusually candid: pre-training the 671B-parameter (37B-activated) Mixture-of-Experts model on 14.8 trillion tokens took about 2.664 million H800 GPU-hours, and the full run including context extension and post-training came to about 2.788 million GPU-hours. At their assumed \$2 per H800-hour, that is roughly **\$5.576 million** — a headline that reset a lot of people's intuition about what a frontier-class model costs. Two caveats keep the comparison honest: it is an MoE, so only the ~37B activated parameters enter the `$6ND$` FLOP count, not the full 671B; and they trained in FP8, so the "MFU" you compute depends entirely on whether you measure against the bf16 or the fp8 peak. The dollar figure, though, is exactly the kind of number this post teaches you to produce and sanity-check.

**PaLM 540B — where the MFU metric came from.** The PaLM paper introduced Model FLOPs Utilization precisely to make cross-hardware comparisons honest, and reported achieving about **46.2 percent MFU** on 6,144 TPU v4 chips training a 540B dense model. That is among the highest MFU figures published for a model that large, and it stands as a practical ceiling: when a well-resourced team optimizing hard lands at 46 percent on a 540B model, a 50-plus percent target on your own large run is ambitious, not routine.

**GPT-3 175B — the order-of-magnitude anchor.** GPT-3 175B on ~300B tokens is about `$6 \times 175\text{e}9 \times 3\text{e}11 = 3.15\text{e}23$` FLOPs. Contemporary estimates put the training cost in the low millions of dollars on the V100-era hardware of its time; the exact figure was never published and the widely-repeated numbers are estimates, so I quote it only as "single-digit millions" and note the uncertainty. The FLOP count, however, is exact from the `$6ND$` rule and is the right way to compare its scale to a modern run: it is roughly a third of the compute of DeepSeek-V3's activated-parameter budget, which is a striking illustration of how far efficiency and token budgets have moved.

## When to reach for each lever, and when to stop

Not every lever is worth pulling, and the discipline of *not* optimizing is as valuable as the optimization itself. Here is the decisive version.

**Always measure MFU first.** Before any cost conversation, compute the real MFU of the run from throughput. If it is already in the high 40s to low 50s for a large dense model, the systems work is largely done and further kernel-tuning is a rounding error — move on to price and sizing. If it is in the 20s or low 30s, stop everything and fix it, because that is where the money is: you are quite possibly paying double. The inverse law makes this triage automatic.

**Reach for spot when the run is fault-tolerant and checkpointable; avoid it otherwise.** If you have distributed checkpointing, cheap frequent saves, and ideally elastic rendezvous, spot is a 50-to-70-percent price cut for a small wall-clock penalty and you should take it. If the run cannot resume cleanly, or checkpoints are slow and huge relative to the reclaim interval, or the job is so tightly coupled that any node death kills it and you have no elasticity, stay on on-demand or reserved capacity — the cheaper hourly rate is a false economy when you spend it thrashing on restarts.

**Right-size to your deadline, not to the biggest cluster you can get.** More GPUs buy time, not savings — and past the scaling-efficiency cliff, they buy neither. Pick the cluster size where the wall-clock meets your deadline at the smallest premium, and refuse to scale past the point where MFU has cratered. If you have no deadline, run small and efficient.

**Stop optimizing when the marginal saving is worth less than the engineering.** This is the budget principle from [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy), applied to cost: every optimization has a price in engineer-time and added complexity, and complexity is itself a source of expensive bugs. A week of work to lift MFU from 30 to 50 percent on a million-dollar run is obviously worth it. The same week to lift a well-tuned run from 52 to 55 percent, or to shave 5 percent off a \$40k run, usually is not. The right amount of cost optimization is exactly enough to get onto the good part of the `$1/\text{MFU}$` curve and no further — the goal is a run that trains efficiently and ships, not a monument to utilization.

**Don't forget the cheapest lever.** Before any of the above, make sure you are not wasting whole runs. Alerting on divergence, a small canary before the big launch, frequent correct checkpoints, and killing bad configurations early save more money, more reliably, than any kernel. Wasted compute has an MFU of zero and a cost of everything.

## Key takeaways

- **Cost = GPU-hours × price, and GPU-hours = `$6ND / (P_\text{peak} \cdot \text{MFU})$`.** Compute is fixed by your model and tokens; price is fixed by your contract; MFU is the term you own.
- **Cost is inversely proportional to MFU.** Doubling MFU halves the bill, with no diminishing return in the algebra — every point of MFU is worth the same slice of the invoice.
- **GPU count cancels out of the cost.** At fixed MFU, a run costs the same GPU-hours on 64 or 1024 GPUs; more GPUs buy wall-clock time, not money. The only reason cost rises with cluster size is that MFU falls.
- **Budget by compute (`$N \times D$`), not by parameter count.** A small model trained long can out-cost a large model trained short — the 175B-on-300B run is cheaper than the 70B-on-2T run.
- **Spot is a ~1/3-price lever for fault-tolerant, checkpointed, ideally elastic runs.** It almost always wins on price; it loses when you cannot resume cheaply or when reclaim risk compounds across nodes into thrashing.
- **Right-sizing is a Pareto trade of money for time.** Scale out to hit a deadline at the smallest premium; never scale past the scaling-efficiency cliff, where you pay more for no time.
- **Respect the critical batch size.** Past it, more data-parallel GPUs stop speeding convergence — switch axes instead of widening the batch into diminishing returns.
- **The cheapest compute is the compute you never waste.** Alert on divergence, canary before the big run, checkpoint often and correctly, and kill doomed runs in the first hour.
- **Measure MFU honestly, against the dense peak in your precision, in steady state** — then let the `$1/\text{MFU}$` law tell you exactly how many dollars each point is worth.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls, of which "cost too high" is the fourth, and the map of the whole series.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist this cost model plugs into.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) and [Mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) — the two largest MFU levers, priced here in dollars.
- [Picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) — the fit-first framework and the budget principle for when to stop.
- [Fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) — what makes spot instances viable, and how elasticity prevents reclaim thrash.
- [Throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions) — MFU as the north star and the confounds that quietly raise your bill.
- The PaLM paper (Chowdhery et al., 2022) — the origin of the MFU metric and the 46.2 percent large-model reference point.
- The Llama 2 (Touvron et al., 2023) and DeepSeek-V3 (2024) technical reports — the clearest published GPU-hour and dollar figures to calibrate your own estimates against, plus the [Chinchilla compute-optimal analysis](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) behind the token budgets.

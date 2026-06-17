---
title: "Edge MLOps: model registries, OTA updates, on-device A/B, and drift"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Operate a fleet of optimized models across millions of heterogeneous devices: version compressed artifacts, ship OTA updates safely with staged rollouts and rollback, run privacy-preserving on-device A/B, and detect silent drift without labels."
tags:
  [
    "edge-ai",
    "model-optimization",
    "mlops",
    "ota-updates",
    "drift-detection",
    "model-registry",
    "ab-testing",
    "monitoring",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/edge-mlops-1.png"
---

A speech-wakeword model shipped to four million phones. It was the team's proudest result: a 240 KB int8 network, p99 latency of 9 ms on a mid-tier NPU, a false-accept rate that beat the previous generation by a third in the lab. They bundled it into the app, shipped it, moved on to the next thing. Eleven weeks later a support ticket trickled in — "the assistant doesn't wake up anymore in my car." Then a few dozen more. Nobody had a dashboard that could see it, because the model ran on-device and the team had no way to look at the audio. When they finally instrumented the wake-confidence distribution and pulled aggregates back, the histogram had quietly collapsed: a new car-infotainment Bluetooth codec, rolled out by a phone-OS update, had shifted the microphone gain on a slice of devices. The model wasn't broken. The world had moved underneath it, and the model — frozen in the app binary — couldn't move with it. Six weeks of silent degradation on a couple hundred thousand devices, discovered by accident.

Shipping the model once is the easy part. The hard part — the part nobody warns you about when you spend three months getting a network down to 240 KB — is *operating* it: keeping a fleet of optimized artifacts healthy across millions of devices you cannot SSH into, updating them safely without bricking anyone, and knowing when they have silently rotted. That discipline is **edge MLOps**, and it is its own engineering domain, not a footnote to cloud MLOps. This post is the operating manual. By the end you will be able to design a model registry that makes a *compressed* artifact reproducible byte-for-byte, ship an over-the-air model update with a staged rollout and an automatic rollback, run an on-device A/B test that never exfiltrates a raw input, and build a label-free drift monitor that catches the wakeword failure above on day two instead of week eleven. The whole thing is a closed loop — telemetry feeds retraining, retraining feeds re-optimization on the accuracy-latency [Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier), and the winner rides an OTA back to the fleet (Figure 1).

![A six-stage timeline showing the edge MLOps feedback loop from device telemetry through drift detection, retraining, re-optimization, re-validation, and a staged over-the-air update](/imgs/blogs/edge-mlops-1.png)

This sits at the very top of the series' four-lever frame — [quantization, pruning, distillation, and efficient architecture](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) all produce the artifacts that edge MLOps has to version, ship, and watch. You can compress a model perfectly and still lose in production because you could not operate it. Let's build the operating layer.

## Why edge MLOps is a different animal

If you have run cloud MLOps you already have a mental picture: a model registry (MLflow, Vertex, SageMaker), a CI pipeline that retrains and re-validates, a serving deployment behind a load balancer, and a monitoring stack (Prometheus, Evidently, whatever) that watches predictions because you log every request and response. Rolling back a bad model is a `kubectl rollout undo`; it takes seconds, you own the runtime, and you can replay yesterday's exact traffic against a candidate. Three assumptions hold in the cloud that all break at the edge, and every break forces a different design (Figure 2).

![A before-after comparison contrasting cloud MLOps assumptions of full input visibility, instant redeploy rollback, and one uniform server image against edge MLOps with invisible inputs, OTA-gated rollback, and one artifact per hardware tier](/imgs/blogs/edge-mlops-2.png)

**Assumption 1: you can see the inputs.** In the cloud, every inference is a request you logged. You can compute input distributions, replay them, build a labeled set from production traffic. On-device, the input is a camera frame, an audio buffer, a keystroke stream — and the entire reason the model is on-device is often privacy: the input must *never* leave the phone. You are blind to the one thing cloud monitoring depends on. Edge monitoring has to run on proxies: the model's *outputs* and *confidences*, aggregated, plus pure systems telemetry (latency, crashes, battery).

**Assumption 2: rollback is instant and you own the runtime.** At the edge, "the runtime" is someone else's phone, on a flaky cellular connection, possibly in airplane mode, possibly an OS version you stopped testing two years ago. You cannot push a fix and have it live in seconds. An update is a download the device has to choose to fetch, verify, and apply — measured in hours-to-days across a fleet, not seconds. Rollback is *also* an OTA, which means a bad model can be live on millions of devices for the entire propagation window. That single fact is why staged rollouts and on-device guardrails are not optional polish; they are the only thing standing between a bad build and the whole fleet.

**Assumption 3: the hardware is uniform.** Cloud serving runs one container image on machines you chose. A consumer fleet spans a [bewildering range of silicon](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) — flagship NPUs, mid-range GPUs, ancient CPU-only devices — and the *same* logical model needs a *different optimized artifact* per tier: int8 for the NPU, fp16 for the GPU, an XNNPACK int8 build for the CPU. So the registry does not track one model; it tracks a fan-out of artifacts, each with its own measured Pareto point. We will come back to this heterogeneity problem in its own section because it shapes everything.

Put those three together and edge MLOps is not "cloud MLOps, but smaller." It is a discipline built around blindness, latency, and fan-out. The rest of this post walks the five pillars — registry, OTA, on-device A/B, monitoring/drift, and the feedback loop — with the science and the code for each.

## Pillar 1: a registry for compressed artifacts

A cloud model registry tracks a checkpoint and some metrics. That is not enough at the edge, and the gap is exactly the thing this whole series is about: **the artifact you ship is not the model you trained.** It is the model after quantization, pruning, graph fusion, and conversion to a target-specific binary — a `.tflite`, a `.gguf`, a TensorRT `.engine`, a Core ML `.mlpackage`. If your registry only stores the fp32 checkpoint and an accuracy number, you have lost the ability to answer the question that matters most when something breaks: *how exactly did this 240 KB binary on this phone come to exist, and can I rebuild it bit-for-bit?*

So the unit of the edge registry is not a model — it is a **(recipe, target, runtime, artifact, metrics, provenance)** tuple, pinned together and made immutable (Figure 3). Let me define each.

![A six-layer vertical stack showing the fields of a registry entry: identity, optimization recipe, target plus runtime, the artifact binary with its hash, the measured metrics, and the provenance of training commit and calibration set](/imgs/blogs/edge-mlops-3.png)

- **Identity**: a model name and a monotonic build number. `wakeword-v3 build 412`.
- **Recipe**: the *exact* optimization specification — quantization scheme (int8 per-channel? int4 group size 128?), pruning ratio and structure (30% structured channel pruning? 2:4 sparse?), the calibration set hash, the converter version and flags. This is what makes the artifact reproducible. Two artifacts with the same recipe and the same training checkpoint and the same converter version must produce the same bytes.
- **Target + runtime**: the hardware tier and the runtime version it was built and validated against. `Pixel 8 NPU, LiteRT 2.16`. An int8 artifact that was great on LiteRT 2.15 can regress on 2.16 if a kernel changed — the runtime version is part of the identity.
- **Artifact**: the actual binary, content-addressed by hash. `model.tflite, sha256 9f2a…`. The hash is the link between "what the registry says" and "what is actually on the device."
- **Metrics**: the measured Pareto point — accuracy on the held-out set, latency p50/p99 on the *named target*, model size in MB, peak memory. Not the training-time metrics; the on-target ones.
- **Provenance**: training commit, dataset version, calibration set hash, who promoted it and when. The audit trail.

### Why content-addressing is non-negotiable

The hash matters more than it looks. At the edge you ship a binary over a flaky channel to a device that then applies it. You need an end-to-end integrity guarantee: the bytes the device runs are *exactly* the bytes the registry validated. Content-addressing gives you that for free. The OTA payload carries the expected SHA-256; the device recomputes it after download and refuses to install on mismatch. The same hash that names the registry entry is the same hash the device verifies, so "the build I tested" and "the build that ran" are provably identical. Combine that with a *signature* over the hash (more on signing under OTA) and you have integrity plus authenticity.

Here is a registry entry as the JSON you would actually store. Note it captures the recipe in enough detail to rebuild, and the metrics are on-target, not training-time.

```json
{
  "model": "wakeword",
  "build": 412,
  "created_utc": "2026-06-10T14:22:03Z",
  "recipe": {
    "base_checkpoint": "wakeword-fp32@a91c4e2",
    "quantization": {
      "scheme": "int8",
      "granularity": "per_channel",
      "activations": "int8_dynamic",
      "calibration_set_sha256": "4d1e...88",
      "calibration_samples": 512
    },
    "pruning": { "type": "structured_channel", "sparsity": 0.30 },
    "converter": "litert==2.16.1",
    "converter_flags": ["--optimizations=DEFAULT", "--target=npu"]
  },
  "targets": [
    {
      "tier": "npu_flagship",
      "runtime": "LiteRT 2.16",
      "artifact": {
        "uri": "s3://models/wakeword/412/npu/model.tflite",
        "sha256": "9f2a1c...e7",
        "size_bytes": 245760
      },
      "metrics": {
        "accuracy_top1": 0.972,
        "far_per_hour": 0.21,
        "latency_p50_ms": 6.1,
        "latency_p99_ms": 9.0,
        "peak_mem_kb": 410
      }
    }
  ],
  "provenance": {
    "train_commit": "a91c4e2",
    "dataset_version": "ww-2026.05",
    "promoted_by": "ci-bot",
    "signature": "MEUCIQ...=="
  }
}
```

The recipe block is the part people skip and regret. When the wakeword model regresses, the first question is "did the recipe change?" — and if the calibration set hash drifted, or the converter bumped a minor version, you have your suspect in thirty seconds instead of a week of bisection. This connects directly to the [end-to-end deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle): the lifecycle is how you *make* the artifact; the registry is how you *remember exactly how you made it* so you can do it again. Tools like MLflow and TFX pioneered the registry-as-source-of-truth idea for cloud; the edge twist is that the "model" field expands into the recipe-plus-per-target fan-out above.

### Reproducibility is harder than it sounds

A cloud checkpoint is reproducible if you fix the seed and the data. An *optimized* artifact has more moving parts, and several of them are silently non-deterministic:

- **Calibration sampling.** Post-training int8 quantization picks activation ranges from a calibration set. If you sample those 512 examples randomly each run, you get different scale factors and a different binary. Pin the calibration set by hash and freeze the sample order.
- **Converter version.** TFLite/LiteRT, ONNX Runtime, TensorRT, and Core ML Tools all change op implementations and fusion patterns across versions. Pin the exact converter version in the recipe; a minor bump is a *new recipe*, not the same one.
- **Kernel autotuning.** TensorRT and TVM autotune kernels to the device, and the tuning is wall-clock-budgeted and can pick different kernels run to run. For reproducibility, cache and pin the tuning result (the TensorRT timing cache, the TVM tuning log) as part of the recipe.

If you pin all three, the artifact hash is stable and reproducibility becomes a property you can *check in CI*: rebuild from the recipe, diff the hash, fail the pipeline if it moved.

### Stage transitions and the promotion gate

A registry entry is not just stored; it moves through *stages*, and the transitions are where quality gets enforced. The minimal lifecycle is `candidate → validated → staged → production → deprecated`, and each transition has a gate:

- **candidate → validated.** The artifact built from the recipe; CI re-runs the on-target benchmark (accuracy on the held-out set, latency p50/p99 on the named target via an emulator or a device farm) and checks the metrics meet the bar. A candidate that does not measure on-target does not advance — training-time numbers are not enough.
- **validated → staged.** The Pareto check: is the candidate non-dominated against the current production build? At least as accurate at equal-or-lower latency, or faster at equal accuracy. A candidate that is strictly worse on both axes is rejected even if it "passed validation," because shipping a dominated artifact is pure downside.
- **staged → production.** The staged rollout itself, gated by the guardrail telemetry from Pillar 2. Production is not a flag flip; it is the *successful completion* of a 1% → 10% → 100% ramp with clean guardrails at every ring.
- **→ deprecated.** When a successor reaches 100%, the predecessor is deprecated but *kept* (you may need to roll back to it), and only garbage-collected after a safety window.

Encoding these transitions as code — a promotion gate that refuses to advance an artifact that fails its check — is what turns "we have a registry" into "the registry enforces quality." Here is the Pareto-and-reproducibility gate as the function CI runs:

```python
import hashlib, json

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def promote(candidate: dict, production: dict, rebuilt_artifact_path: str) -> str:
    # 1. Reproducibility: the rebuilt artifact must hash to the registered value.
    got = sha256_file(rebuilt_artifact_path)
    want = candidate["artifact"]["sha256"]
    if got != want:
        return f"REJECT: hash drift {got[:8]} != {want[:8]} (non-reproducible recipe)"

    # 2. Pareto: candidate must not be dominated by production on (acc, latency).
    c, p = candidate["metrics"], production["metrics"]
    worse_acc = c["accuracy_top1"] < p["accuracy_top1"]
    worse_lat = c["latency_p99_ms"] > p["latency_p99_ms"]
    if worse_acc and worse_lat:
        return "REJECT: dominated on both accuracy and latency"

    # 3. Guardrail floor: never ship a build that fails an absolute floor.
    if c["latency_p99_ms"] > candidate["targets_max_p99_ms"]:
        return "REJECT: p99 over hard ceiling"

    return "PROMOTE: to staged, begin 1% canary"
```

The reproducibility check (step 1) is the one teams skip and the one that saves a 2 a.m. incident: if the recipe no longer reproduces the registered hash, *something in your toolchain moved silently* — a converter auto-updated, a calibration set was resampled — and you want to know that in CI, not after you have shipped a different binary than the one you validated.

## Pillar 2: OTA model updates without an app release

Here is the single most important capability edge MLOps buys you: **the ability to update the model without shipping a new app.** Think of why this matters. An app release goes through a store review (days), then propagates to users at whatever pace they update (weeks; a meaningful fraction of users never update). If your model is welded into the app binary, your model's update cadence is your app's update cadence, and a critical model fix is hostage to App Store review and user laziness. Decoupling the model from the binary is the whole game.

### Asset bundling versus true OTA

There is a spectrum here, and the [mobile end-to-end deployment](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end) choice sits right at its center:

- **Bundled in the binary.** The `.tflite` is in the app's assets. Simplest, works offline on first launch, zero infra. But the model's lifecycle is chained to the app's. Fine for a small, static model that rarely changes.
- **On-demand asset / asset pack.** The model is a downloadable asset (Android's Play Asset Delivery, iOS's On-Demand Resources). The app ships without it and fetches it on first run. Decouples *size* from the binary, but updates still ride app releases unless you add your own delivery.
- **True OTA model delivery.** A first-class model-delivery service: the device asks "what build should I be running for my tier?", downloads the artifact, verifies, and hot-swaps it at runtime. This is what lets you ship build 413 to fix a drift problem this afternoon, gated to 1% of devices, with no store involvement at all.

True OTA is the target for any fleet you intend to operate. Below is the device-side config it consumes — a manifest the device fetches, telling it which build to run, where to get it, how to verify it, and the rollout gate.

```yaml
# /v1/fleet/wakeword/manifest  -> served per device, gated by rollout ring
model: wakeword
device_tier: npu_flagship
target_build: 412
rollout:
  ring: canary          # canary | early | broad | full
  percent: 1            # this build is exposed to 1% of the npu tier
  min_app_version: "8.4.0"
  min_runtime: "LiteRT 2.16"
artifact:
  uri: "https://cdn.example.com/models/wakeword/412/npu/model.tflite"
  sha256: "9f2a1c...e7"
  size_bytes: 245760
delivery:
  delta_from_build: 411     # device has 411; send only the diff
  delta_uri: "https://cdn.example.com/models/wakeword/411_to_412.bsdiff"
  delta_bytes: 38912
signature:
  alg: "ecdsa-p256-sha256"
  value: "MEUCIQ...=="      # signs the sha256 + manifest fields
guardrails:
  max_crash_rate: 0.002
  max_p99_ms: 14
  min_confidence_psi_ok: 0.25
```

Five things in that manifest do real work, and each maps to a hard constraint of the edge:

**Staged rollout (`rollout`).** The build is exposed to a *ring* of the fleet — 1% canary, then 10%, then broad, then 100%. The device hashes its stable ID into a bucket and only fetches the build if its bucket is inside the current percentage. This is exactly the Play Console / App Store Connect staged-rollout model, applied to the model artifact instead of the whole app. We will do the statistics of "how big a ring do I need" in the science section, because that is where most teams guess wrong.

**Delta updates (`delivery`).** Shipping a fresh 245 KB binary to 4 million devices is ~980 GB of egress *per rollout*. But build 412 differs from 411 by a fraction of its bytes — same architecture, slightly different weights. A binary diff (bsdiff, zstd `--patch-from`, or Courgette-style) ships only the changed bytes. The math is in the science section; the practical effect is an order-of-magnitude bandwidth cut, which on cellular is also a cost and a user-goodwill cut.

**Signing and integrity (`signature`).** The manifest and the artifact hash are signed with a private key the device verifies against a pinned public key. This is non-negotiable: a model is *executable behavior*: an attacker who can swap your model can change what the app does. Verify the signature, then verify the downloaded bytes hash to the signed value, then and only then install. Integrity (hash) plus authenticity (signature) plus a transport you trust (TLS).

**Heterogeneity (`device_tier`, `min_runtime`).** The manifest is served *per device*: an NPU-flagship phone gets the int8 NPU build, a CPU-only phone gets the XNNPACK build. The `min_runtime` and `min_app_version` gates prevent shipping an artifact a device's runtime cannot execute — the single most common cause of a "rollout looked fine in canary, exploded at 10%" incident.

**Rollback (`target_build` can go backwards).** Rollback is just publishing a manifest whose `target_build` points at the previous good build. The device sees a new target, fetches it (or, ideally, still has it cached locally for instant revert), verifies, swaps. Because rollback is an OTA, it inherits the same propagation latency as a forward update — which is the deep reason staged rollouts matter: you want to catch the problem at 1% so the rollback only has to reach 1%, not 100%.

### The "keep N builds" rule

A practical pattern that saves you during incidents: the device keeps the **current build plus the previous known-good build** on disk. Rollback then is not a download at all — it is a pointer swap to a binary already present, applied on next launch. You trade a little flash (one extra artifact, here ~240 KB, trivial; for a 4 GB LLM, a real budget decision) for near-instant local rollback. For a large [on-device LLM](/blog/machine-learning/edge-ai/making-on-device-llms-fast) you cannot keep two copies, so you accept a re-download on rollback and lean harder on catching problems in canary.

### The device-side update state machine

The manifest is the server's view; the device has its own job, and getting it wrong bricks the feature. The device-side update is a small state machine with one hard requirement: **the swap must be atomic and the model must always be runnable.** You never want a state where the old model is deleted, the new one is half-downloaded, and the app launches with no model at all. The states:

1. **idle** — running the current build, periodically polling the manifest endpoint (on a backoff, on a metered-connection-aware schedule).
2. **download** — a new `target_build` appeared and this device's bucket is in the ring. Fetch the delta (or full artifact) to a *staging* location, resumable across connection drops. Never touch the live model yet.
3. **verify** — recompute the SHA-256 of the reconstructed artifact and check it against the signed manifest value; verify the signature against the pinned public key. Any mismatch → discard and stay on the current build. This is the integrity gate, on-device.
4. **apply** — atomically swap the active-model pointer to the new artifact on next launch (or next idle moment). The previous build stays on disk as the rollback target.
5. **confirm** — run the new build for a probation window; if it crashes on launch or trips a *local* guardrail (e.g. it fails to load, or first-inference latency is absurd), auto-revert to the previous build without waiting for a server rollback. This local self-heal is what protects you during the propagation window when a server rollback has not reached the device yet.

That last point — **client-side self-healing** — is underrated. A device that can detect "this new model crashes on load" and revert *itself* turns a fleet-wide incident into a per-device non-event for the worst failure mode (an artifact that does not even run on some hardware). The server-side staged rollout catches *statistical* regressions; the client-side confirm step catches *catastrophic* per-device failures instantly. You want both.

```python
# Device-side update FSM (faithful pseudocode for the on-device updater)
def update_tick(state, manifest, local):
    if state == "idle":
        if manifest["target_build"] != local.active_build and in_ring(local.device_id, manifest):
            return "download"
        return "idle"
    if state == "download":
        ok = fetch_resumable(manifest["artifact"]["uri"],
                             delta=manifest["delivery"].get("delta_uri"),
                             base=local.active_artifact)
        return "verify" if ok else "idle"          # retry later on failure
    if state == "verify":
        bytes_ok = sha256(local.staged) == manifest["artifact"]["sha256"]
        sig_ok   = verify_sig(manifest, PINNED_PUBKEY)
        if bytes_ok and sig_ok:
            return "apply"
        local.discard_staged()                      # tamper or corruption
        return "idle"
    if state == "apply":
        local.keep_previous(local.active_build)     # retain rollback target
        local.activate(local.staged)                # atomic pointer swap
        return "confirm"
    if state == "confirm":
        if local.first_run_crashed() or local.first_inf_ms() > manifest["guardrails"]["max_p99_ms"] * 5:
            local.revert_to_previous()              # CLIENT-SIDE self-heal
            return "idle"
        local.commit()                              # probation passed
        return "idle"
```

The `max_p99_ms * 5` threshold on the confirm step is deliberately loose — the client-side check is a catastrophe detector ("is this 5× worse, i.e. obviously broken?"), not a fine-grained quality gate. Fine-grained regressions (the 40% p99 bump from the worked example) are too subtle for one device to judge from a handful of inferences; those are the server's job, decided across thousands of devices via the sample-size math. Splitting the responsibility this way — catastrophes local and instant, subtle regressions central and statistical — is the design that keeps a fleet both safe and well-measured.

## Pillar 3: on-device A/B and shadow testing

You have two candidate models — build 412 and a more aggressively pruned build 413 — and you want to know which is actually better *in the field*, not in your lab. In the cloud you would split traffic, log both arms' predictions and outcomes, and read the metrics. On-device you cannot log the inputs or, often, the ground-truth outcomes. So on-device A/B is built on two ideas: **assign by device, measure by aggregate.**

**Assignment.** Each device hashes its stable ID into an experiment bucket and runs exactly one arm. This is deterministic (a device stays in its arm across launches, so you are not flip-flopping a user's experience) and it is sticky to the device, not the request. The control arm runs the current production build; the treatment arm runs the candidate.

**Measurement.** Each arm reports *aggregates only* — never raw inputs. The confidence histogram, the output-class rate, latency p50/p99, crash rate, battery delta — bucketed and summed on-device, uploaded periodically. The server compares arm A's aggregate to arm B's. You learn "treatment has 8% lower p99 and an indistinguishable confidence histogram, at a 0.4-point-lower mean confidence" without a single raw audio frame leaving a phone.

### Shadow mode: the safest way to test a candidate

Before you even A/B, you can **shadow** a candidate: run *both* models on the same input on a small slice of devices, but only the production model's output is used — the candidate's output is computed, compared, and its aggregate logged, then thrown away. The user never sees the candidate's behavior, so there is zero risk, and you get a direct on-device comparison of the two models *on the same real inputs*. The cost is running two inferences (battery, latency), so you shadow on a tiny slice and for a bounded window.

```python
# On-device experiment + shadow logic (pseudocode-faithful, runs on the phone)
import hashlib

def bucket(device_id: str, salt: str, n_buckets: int = 1000) -> int:
    h = hashlib.sha256(f"{device_id}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) % n_buckets

class OnDeviceExperiment:
    def __init__(self, device_id, prod_model, candidate_model, cfg):
        self.device_id = device_id
        self.prod = prod_model
        self.cand = candidate_model
        self.cfg = cfg                       # from the OTA manifest
        self.arm = self._assign()
        self.agg = ConfidenceAggregator()    # privacy-safe histogram

    def _assign(self):
        b = bucket(self.device_id, self.cfg["experiment_id"])
        if b < self.cfg["shadow_pct"] * 10:        # e.g. 2% shadow
            return "shadow"
        if b < self.cfg["treatment_pct"] * 10:     # next 10% treatment
            return "treatment"
        return "control"

    def infer(self, x):
        if self.arm == "treatment":
            y = self.cand(x)
        elif self.arm == "shadow":
            y = self.prod(x)                       # user sees prod
            y_cand = self.cand(x)                  # computed, not shown
            self.agg.observe_shadow(conf(y), conf(y_cand))
            return y
        else:
            y = self.prod(x)
        self.agg.observe(conf(y), arm=self.arm)    # bucketed counts only
        return y

    def flush(self):
        # uploads aggregates ONLY: histogram bins, p50/p99, counts
        return self.agg.to_payload(arm=self.arm)
```

### Guardrail metrics gate everything

An A/B test optimizes a *success* metric (accuracy proxy, engagement) but it must not regress a *guardrail* metric (crash rate, p99 latency, battery). The discipline: pre-register one or two guardrails with hard thresholds, and if the treatment arm trips a guardrail you halt and roll back *regardless* of the success metric. A candidate that improves accuracy by a point but raises the crash rate from 0.1% to 0.5% is not a win; it is an incident waiting for scale. Guardrails are the same numbers the staged-rollout gate watches — A/B and staged rollout are two views of the same telemetry, which is why we unify them in the feedback-loop section.

### Reading an on-device A/B without fooling yourself

On-device A/B has two statistical traps that the cloud version mostly avoids, and both come straight from the privacy-and-aggregation constraints. The first is **peeking**. Telemetry trickles in over days as devices wake, connect, and flush their windows; the temptation is to watch the dashboard and call the experiment the moment treatment looks ahead. That inflates your false-positive rate badly — if you test at $\alpha = 0.05$ but peek ten times, your real false-positive rate is closer to 20–30%, because each peek is another chance for noise to cross the line. The fix is either to pre-commit to a fixed sample size (compute $n$ up front from the effect you care about, then look exactly once) or to use a **sequential test** that is valid under continuous monitoring. Always-valid sequential methods — mixture sequential probability ratio tests, or the more recent confidence-sequence approaches — let you peek as often as you like and stop the moment the confidence interval excludes zero, while controlling the error rate. For a fleet where data dribbles in asynchronously, a sequential test is not a luxury; it is the correct tool, because you will look at the dashboard whether you are supposed to or not.

The second trap is that **aggregation costs you statistical power**. In the cloud you have one observation per request; on-device you have one *bucketed, windowed* observation per device per day. Coarser data has higher variance per unit of information, so detecting a given effect needs more devices than the naive per-inference calculation suggests. If you add local differential privacy on top — each device perturbing its counts with noise of variance $\sigma_{\text{dp}}^2$ — the effective variance of the aggregate is the sum of the sampling variance and the privacy noise variance, and the detectable effect floor rises accordingly. Concretely, the minimum detectable effect scales like

$$
\text{MDE} \;\propto\; \sqrt{\frac{\sigma_{\text{sampling}}^2 + \sigma_{\text{dp}}^2}{n}},
$$

so doubling the privacy noise variance is roughly equivalent to halving your device count for detection purposes. This is the quantitative version of the trade-off you make every time you turn up the privacy dial: more protection, less sensitivity, bigger rings needed to see the same win. Pre-registering the experiment with this in mind — picking $n$ from the MDE you can tolerate at your chosen privacy level — is what separates an A/B program that ships real improvements from one that ships noise.

A practical decision rule that has served me well: run the candidate in **shadow** first (zero user risk) to confirm it does not produce wildly different outputs on real inputs; then a small **treatment** ring to measure the success metric with the guardrails armed; only then promote into the staged rollout. Shadow answers "is it broken?", A/B answers "is it better?", and staged rollout answers "is it safe at scale?" — three distinct questions, three distinct mechanisms, run in that order.

## Pillar 4: monitoring and drift without seeing the inputs

This is the pillar that the wakeword team in the intro was missing, and it is the hardest one because of Assumption 1: you cannot see the inputs. Cloud drift detection compares the *input* distribution today against training; you cannot, because the input never leaves the device. So edge drift detection runs on **proxies** — quantities you *can* aggregate without violating privacy (Figure 6).

![A five-row matrix mapping monitoring signals confidence PSI, output-rate, latency p99, crash and ANR, and battery delta to what each catches, its privacy cost, and whether it is label-free](/imgs/blogs/edge-mlops-6.png)

The proxies, and what each catches:

- **Confidence distribution (PSI / KL).** The model's own output confidences, bucketed into a histogram and aggregated. When the input distribution shifts, the confidence histogram shifts — usually toward lower confidence as inputs move off the training manifold. This is your primary drift signal and it is *label-free*. The math is below.
- **Output-class rate.** The fraction of inferences landing in each output class. If "wake" events triple or a classifier suddenly calls everything "class 3", the world or the model changed. Catches class-mix shift; pure counts, maximally private.
- **Latency p99.** A regression here usually means an *operational* fault, not a data one: an NPU op fell back to CPU after a runtime update, or thermal throttling kicked in. Catches the "looked fine in canary, exploded at 10%" failure that is op-support, not accuracy.
- **Crash / ANR rate.** Out-of-memory, a bad delegate, a malformed artifact. Catches deployment faults the moment they appear.
- **Battery / energy delta.** A model stuck in a runaway inference loop, or a fallback path that runs 8× more compute, shows up as a battery regression before users complain.

Crucially, **none of these requires labels.** You do not know if any individual prediction was correct — but you can detect that the *distribution* of the model's behavior has moved, which is the strongest signal you can get on a device where ground truth never arrives.

### Data drift versus concept drift — and why proxies catch both partially

Two distinct things can rot a model, and they are worth separating:

- **Data drift (covariate shift):** the input distribution $P(x)$ changes while the input-to-label mapping $P(y \mid x)$ stays the same. The car-codec microphone-gain shift is pure data drift — the audio statistics moved, but a wake-phrase is still a wake-phrase. Confidence and output-rate proxies catch data drift well, because off-manifold inputs lower confidence.
- **Concept drift:** the mapping $P(y \mid x)$ itself changes — the same input should now produce a different label. A fraud model facing a new fraud pattern, or a content classifier facing a new slang. This is harder: the inputs may look in-distribution (high confidence) while the model is confidently *wrong*. Proxies catch concept drift only partially; you often need a trickle of labels (from a federated or human-in-the-loop channel) to see it.

The honest framing: proxy monitoring is excellent at catching data drift and operational faults label-free, and it is a *partial* net for concept drift. For high-stakes concept drift you supplement with a small labeled stream. Do not oversell what confidence histograms can do.

### Making the monitor actually page correctly

A drift number is only useful if it pages at the right time and not at the wrong time. Three operational details decide whether your monitor is a help or a source of alert fatigue:

**Choosing the reference.** PSI compares *current* against a *reference* distribution. The wrong reference is a common own-goal. Use the confidence histogram captured at *validation time for the exact build in the field* — not the training distribution, not last week's field window. If you compare against a rolling recent window, slow drift becomes invisible because the reference drifts with it (the boiling-frog problem). A fixed validation reference per build is the honest baseline; it answers "has the model's behavior moved since we said it was healthy?"

**Windowing.** PSI on too few samples is pure noise — a 50-inference window will trip thresholds on sampling wobble alone. Aggregate to a window with enough mass (a few thousand inferences across the ring) before computing PSI, and require the alert to *persist* across two consecutive windows before paging. A single-window spike is often a transient (a software update rolling through a region); a two-window-sustained shift is real drift. This `persist=2` rule cuts false pages dramatically at the cost of one window of detection latency, which is a good trade when a page means waking someone.

**Per-tier and per-region baselines.** A confidence shift on the CPU tier may not appear on the NPU tier, and a shift in one region (the car-codec example was geographically concentrated) may be invisible in the fleet-wide average. Computing PSI per tier *and* per coarse region surfaces localized drift that a single global number washes out — the intro incident was a ~5% device slice, and a fleet-average PSI would have stayed under threshold while a per-tier-per-region PSI lit up. The cost is more dashboards; the benefit is catching the failures that hide in subpopulations, which are exactly the ones that produce mystifying support tickets.

```python
# Server-side drift alerting over windowed, per-tier confidence histograms
def drift_alert(reference_hist, windows, tier, psi_alert=0.25, persist=2):
    """windows: list of recent per-window histograms, newest last."""
    recent = windows[-persist:]
    scores = [psi(reference_hist, w) for w in recent]
    if len(scores) == persist and all(s >= psi_alert for s in scores):
        return {
            "level": "page",
            "tier": tier,
            "psi": round(scores[-1], 3),
            "action": "escalate_to_retrain",
            "note": f"PSI >= {psi_alert} for {persist} consecutive windows",
        }
    if scores and scores[-1] >= 0.10:
        return {"level": "watch", "tier": tier, "psi": round(scores[-1], 3)}
    return {"level": "ok", "tier": tier, "psi": round(scores[-1], 3) if scores else 0.0}
```

The `watch` level (PSI between 0.10 and 0.25) does not page — it shows up on a dashboard and in a weekly digest. The discipline of *three* levels (ok / watch / page) rather than a single binary threshold is what keeps the on-call rotation trusting the monitor: pages are rare and real, watches accumulate context, and the team is not numb to the alert by the time a real one fires.

## The science: deriving the drift detector

Let's make the drift detector rigorous, because "the confidence looks different" is not a threshold you can page on. The workhorse is the **Population Stability Index (PSI)**, which is a symmetrized, binned approximation of the KL divergence between the reference confidence distribution and the current one.

Bin the model's output confidence (or any scalar proxy) into $B$ buckets. Let $p_i$ be the fraction of reference-period outputs in bucket $i$ (the distribution when build 412 was validated and healthy), and $q_i$ the fraction in the current field window. PSI is

$$
\text{PSI} = \sum_{i=1}^{B} (q_i - p_i)\,\ln\!\frac{q_i}{p_i}.
$$

Where does this come from? Recall the KL divergence $D_{\mathrm{KL}}(q \,\|\, p) = \sum_i q_i \ln(q_i/p_i)$, which measures how surprised you are seeing $q$ when you expected $p$. KL is asymmetric. PSI symmetrizes it by adding the two directions:

$$
\text{PSI} = D_{\mathrm{KL}}(q \,\|\, p) + D_{\mathrm{KL}}(p \,\|\, q) = \sum_i q_i \ln\frac{q_i}{p_i} + \sum_i p_i \ln\frac{p_i}{q_i},
$$

and collecting terms over the shared bins gives the compact $\sum_i (q_i - p_i)\ln(q_i/p_i)$ form above. So PSI is the symmetric KL between two histograms — a single scalar that grows as the distributions diverge, in either direction.

The conventional thresholds, from credit-scoring practice where PSI originated, are:

- $\text{PSI} < 0.10$: no significant shift, healthy.
- $0.10 \le \text{PSI} < 0.25$: moderate shift, investigate.
- $\text{PSI} \ge 0.25$: major shift, alert and consider retraining.

A subtlety that bites in production: empty bins. If a current bin has $q_i = 0$, the $\ln(q_i/p_i)$ term blows up to $-\infty$; if a reference bin has $p_i = 0$, you divide by zero. The standard fix is **Laplace smoothing** — add a small $\epsilon$ (e.g. $10^{-4}$) to every bin before normalizing, which is equivalent to a weak Dirichlet prior and keeps the index finite and stable for the typical few-thousand-sample windows you get per fleet slice.

Here is the drift monitor in code — the function the server runs against the uploaded confidence histograms.

```python
import numpy as np

def psi(reference_hist, current_hist, eps=1e-4):
    """
    Population Stability Index between two confidence histograms.
    Inputs are RAW BIN COUNTS (privacy-safe aggregates from devices),
    not raw inputs. Returns a single scalar; >=0.25 => alert.
    """
    p = np.asarray(reference_hist, dtype=float) + eps   # Laplace smoothing
    q = np.asarray(current_hist,   dtype=float) + eps
    p /= p.sum()                                         # normalize to probs
    q /= q.sum()
    return float(np.sum((q - p) * np.log(q / p)))

# Reference: build 412's confidence histogram when it was validated healthy.
# 10 bins over confidence in [0, 1]; counts aggregated across the canary ring.
reference = [ 40,  60, 110, 180, 260, 420, 720, 1100, 1600, 2100]

# Current field window from the same ring, two weeks later.
current   = [320, 540, 880, 990, 760, 540, 410,  280,  180,  120]

print(round(psi(reference, current), 3))   # -> 0.84  (major shift, alert)

# A healthy window for comparison (small sampling wobble, no real shift):
healthy   = [ 38,  64, 105, 176, 271, 410, 700, 1120, 1590, 2150]
print(round(psi(reference, healthy), 3))   # -> 0.001 (no shift)
```

The PSI of $0.84$ in that example screams: the reference mass was concentrated in high-confidence bins (the model was sure of its answers), and the field window has slid hard into low-confidence bins — the exact signature of the off-manifold microphone-gain shift from the intro, visible *with no labels at all*.

#### Worked example: a drift alert with no labels

The wakeword model, build 411, has been live for two weeks on a canary ring of devices. The reference confidence histogram (captured at validation) put 62% of wake-decisions in the top two confidence bins. The field window now reports 34% there, with a swollen low-confidence tail — exactly the before-after in Figure 5.

![A before-after view of the confidence histogram: a healthy reference with mass in the high-confidence bins and PSI near zero versus a field-shifted distribution with a swollen low-confidence tail and PSI of 0.31 over the alert threshold](/imgs/blogs/edge-mlops-5.png)

Running `psi(reference, current)` over the ten-bin histograms returns $0.31$. That is above the $0.25$ alert threshold, so the monitor pages. Note what we did *not* need: no labels, no raw audio, no idea yet of the root cause. The PSI tells us the model's behavior distribution moved materially; the output-class rate (wake events dropped 19%) corroborates a real failure rather than a sampling artifact; the latency p99 is unchanged, which *rules out* an op-fallback and points at a data shift, not an operational one. Three privacy-safe proxies triangulate to "data drift on a device slice, build 411, escalate to retrain" — on day two of the canary, not week eleven of the full fleet. That single change in *when* you find out is the entire ROI of edge MLOps.

## The science: how many devices to detect a regression

Now the staged-rollout question teams guess wrong: **how big does a ring have to be to reliably catch a p99 latency regression?** Suppose the current build's crash rate is a baseline $p_0$ and a bad build raises it to $p_1$. You expose the bad build to $n$ devices in the ring; how large must $n$ be so you *detect* the elevated rate before you promote?

Model each device as a Bernoulli trial: it crashes with probability $p$ during the observation window. The observed crash count over $n$ devices is $\text{Binomial}(n, p)$, with mean $np$ and variance $np(1-p)$. You want to distinguish $p_1$ from $p_0$ with a one-sided test at significance $\alpha$ and power $1-\beta$. The standard normal-approximation sample size for two proportions gives

$$
n \;\approx\; \frac{\left(z_{1-\alpha}\sqrt{p_0(1-p_0)} + z_{1-\beta}\sqrt{p_1(1-p_1)}\right)^2}{(p_1 - p_0)^2},
$$

where $z_{1-\alpha}$ and $z_{1-\beta}$ are standard-normal quantiles. The key intuition: the required $n$ scales as $1/(p_1 - p_0)^2$ — **detecting a small regression costs quadratically more devices than detecting a large one.**

Plug in numbers. Baseline crash rate $p_0 = 0.001$ (0.1%), a "bad build" you want to catch at $p_1 = 0.003$ (0.3%), one-sided $\alpha = 0.05$ ($z = 1.645$) and power $0.80$ ($z = 0.84$):

$$
n \approx \frac{\left(1.645\sqrt{0.001 \cdot 0.999} + 0.84\sqrt{0.003 \cdot 0.997}\right)^2}{(0.002)^2} \approx \frac{(0.052 + 0.046)^2}{4\times 10^{-6}} \approx 2{,}400.
$$

So roughly **2,400 devices** must run the bad build before you can statistically distinguish a 0.3% crash rate from a 0.1% baseline at 80% power. On a 4-million-device fleet, a 1% canary is 40,000 devices — comfortably above 2,400, so the canary can catch this regression. But watch the quadratic: if you wanted to catch a subtler $p_1 = 0.0015$ (0.15%) regression, the denominator $(p_1-p_0)^2$ shrinks 16×, pushing $n$ toward ~40,000 — your entire 1% canary, with no headroom. Rare events need big rings or long windows; this formula tells you which.

```python
from scipy.stats import norm

def rollout_n(p0, p1, alpha=0.05, power=0.80):
    z_a = norm.ppf(1 - alpha)        # one-sided
    z_b = norm.ppf(power)
    num = (z_a * (p0*(1-p0))**0.5 + z_b * (p1*(1-p1))**0.5) ** 2
    return num / (p1 - p0) ** 2

print(round(rollout_n(0.001, 0.003)))    # ~2400 devices to catch 0.1% -> 0.3%
print(round(rollout_n(0.001, 0.0015)))   # ~38000 devices to catch a 0.05% bump
print(round(rollout_n(0.001, 0.005)))    # ~620 devices to catch a 0.1% -> 0.5% jump
```

The practical reading: **size your canary to the smallest regression you actually care about catching, not to a round percentage.** A "1% canary" is meaningless without knowing your baseline rate and your detectable effect; the formula converts a business requirement ("catch a doubling of crashes") into a device count, and the device count into a percentage of your fleet.

## The science: delta-update size

The third piece of math is the bandwidth budget. A naive OTA ships the whole artifact: for the wakeword model that is $S = 245{,}760$ bytes; for $N = 4\times 10^6$ devices that is

$$
\text{egress}_{\text{full}} = N \cdot S = 4\times 10^6 \times 245{,}760 \approx 9.8\times 10^{11}\ \text{bytes} \approx 980\ \text{GB}
$$

per rollout. But consecutive builds share most of their bytes. A binary delta encodes only the differences. If a fraction $f$ of the bytes changed and the diff compresses by factor $c$, the delta size is roughly $S_\Delta \approx f \cdot S / c$. For a fine-tuned successor where maybe 35% of the weight bytes moved and bsdiff+compression gives $c \approx 3.5$:

$$
S_\Delta \approx \frac{0.35 \times 245{,}760}{3.5} \approx 24{,}600\ \text{bytes},
$$

a **10× reduction** to ~98 GB of fleet egress. The manifest in Pillar 2 advertised a 38,912-byte delta against build 411 — same order of magnitude, the difference being how much of the network the optimization recipe actually perturbed. A pruning change that drops whole channels moves *more* bytes than a light fine-tune, so the delta size is itself a useful signal about how aggressive a recipe change was.

For a [large on-device LLM](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf) the stakes are different by orders of magnitude. A 4 GB GGUF artifact over 4 million devices is 16 PB of egress at full ship — completely infeasible. Here deltas are not an optimization, they are the *only* way to update at all, and you design the recipe so that updates touch as few weight blocks as possible (e.g. ship a LoRA-style adapter delta rather than re-quantizing the whole base). The size math is what forces that architectural decision.

#### Worked example: a staged rollout catches a regression at 10%

Walk the full incident, end to end (Figure 4). Build 412 of the wakeword model passes lab validation: 97.2% accuracy, p99 9.0 ms on the NPU tier. CI promotes it; the rollout starts.

![A six-event timeline of a staged rollout: a clean 1 percent canary, a 10 percent ring showing a 40 percent p99 regression, an automatic guardrail halt, a rollback OTA to the previous build, root-cause of an NPU op falling back to CPU, and a re-shipped fixed build resuming to 100 percent](/imgs/blogs/edge-mlops-4.png)

**Day 0 — 1% canary (40,000 NPU devices).** The device-side guardrails report up: crash rate 0.09%, p99 9.1 ms, confidence PSI 0.02. All clean. The aggregate p99 from a 40,000-device sample is tight enough (per the sample-size math, well above the ~2,400 needed to see a doubling of crashes) that "clean" means clean. Promote to 10%.

**Day 2 — 10% ring (400,000 devices).** Now the telemetry turns. p99 latency jumps to **12.9 ms — a 40% regression** — and the battery-delta proxy ticks up on a sub-slice. The crash rate is fine; the *confidence* PSI is fine. So this is not a data-drift problem and not a stability problem — it is an *operational* one, and the latency-plus-battery signature points straight at extra compute: an op is running somewhere it should not.

**Day 2 — auto-halt and rollback.** The 10% ring's p99 of 12.9 ms crosses the manifest's `max_p99_ms: 14`? Not quite — but the *delta* against the control arm (build 411 at 9.0 ms) is a 40% regression, which trips the relative guardrail. The rollout controller halts promotion automatically and publishes a manifest pinning `target_build: 411`. Because devices kept the previous good build on disk (the "keep N builds" rule), the 400,000 affected devices revert on next launch with **no download** — the 1% canary had to download 412, but the rollback is a local pointer swap. The fleet is back to healthy within the OTA propagation window, and critically the blast radius was 10%, not 100%, because the staged gate stopped it.

**Day 3 — root cause.** The recipe diff (from the registry) shows build 412 bumped the converter to LiteRT 2.16, which changed a depthwise-conv kernel such that on a *specific NPU firmware revision* present in ~25% of the flagship tier the op was unsupported and silently fell back to CPU — 4× slower for that op, hence the p99 and battery hit. This is the canonical "the NPU does not support this op and it falls back to CPU" failure the series keeps warning about, here caught by latency telemetry rather than a user complaint.

**Day 4 — fix and re-ship.** Build 413 pins the converter flag that keeps the op on the NPU for that firmware (or adds a tier split for it). It re-validates on the Pareto frontier — same accuracy, p99 back to 9.1 ms — re-enters the registry with a fresh hash, and rides a fresh staged rollout 1% → 10% → 100%. Total user-visible damage: a 40% latency regression on 400,000 devices for under two days, auto-rolled-back, versus the alternative of a global ship that degraded all four million for as long as it took a human to notice. That gap is the product.

## Pillar 5: the feedback loop and federated learning

Stitch the pillars together and you get the loop from Figure 1, running continuously:

1. **Telemetry in.** Devices upload privacy-safe aggregates: confidence histograms, output-rate, latency p50/p99, crash/ANR, battery delta — bucketed counts, never raw inputs.
2. **Detect.** The server computes PSI on the confidence histograms, watches the guardrail proxies, and fires an alert when a signal crosses threshold (the $0.25$ PSI line, a relative p99 regression, a crash-rate jump).
3. **Retrain / re-optimize.** A drift alert triggers a retrain on fresh data, then re-runs the optimization recipe — re-quantize, re-prune, re-distill — to land the new model back on the [accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier). You do not just retrain; you re-*optimize*, because a retrained fp32 model is not what ships.
4. **Re-validate.** The candidate is benchmarked on each target tier (on-target latency, not training-time FLOPs), and only promoted if it is Pareto-non-dominated against the current build — at least as accurate at equal-or-lower latency, or faster at equal accuracy.
5. **Staged OTA.** The winner enters the registry with a fresh hash, gets signed, and rides a staged rollout with the guardrails of Pillar 2, closing the loop.

### Where to get labels: federated learning and analytics

The loop's weak link is step 3: retraining needs data, and the data is on devices you cannot read. Two privacy-preserving channels close it:

- **Federated analytics.** The drift signals themselves — PSI, output-rate, histograms — *are* federated analytics: computed on-device, aggregated centrally, raw input never moves. This is what makes label-free monitoring possible.
- **Federated learning.** When you need to actually *update weights* on field data you cannot collect, [federated learning](/blog/machine-learning/edge-ai/on-device-and-federated-learning) trains on-device and uploads only gradient/weight updates (ideally with secure aggregation and differential privacy), which the server averages into a new global model. The output of federated training then re-enters the same optimization-and-OTA pipeline. Federated learning is the privacy-preserving way to feed the retrain step when "just collect the data" is off the table — which, for the on-device privacy-first models that justify edge deployment in the first place, it usually is.

The loop and the federated channel together are what let the wakeword model *adapt* to the car-codec shift instead of being permanently broken by it: detect via federated analytics, retrain via federated learning, re-optimize on the Pareto frontier, and OTA the fix — all without a single audio frame leaving a phone.

### Stress-testing the loop: where it breaks

A loop diagram is reassuring; production is not. Walk the ways this loop fails, because knowing the failure modes is how you design around them.

**The drift alert with no action.** A PSI page fires, but there is no fresh labeled data and no federated channel — the retrain step has no input. Now the monitor is a pager that only causes pain: it tells you the model is rotting and gives you no way to fix it. This is the most common edge-MLOps half-build. The rule: do not deploy a drift monitor until the retrain path behind it is real. A monitor whose alert has no downstream action is worse than no monitor, because it trains the team to ignore pages.

**Retraining on the drifted data makes it worse.** Suppose the field shift is adversarial or non-stationary — the inputs that lowered confidence are themselves bad (spam, attack traffic, a sensor fault). Naively retraining on whatever the field now produces can *bake the corruption in*. Federated learning amplifies this risk because you cannot inspect the data. Mitigations: weight federated updates by a trust signal, clip outlier updates (robust aggregation), and always re-validate the retrained-and-reoptimized candidate against a *clean, held-out* reference set before it can be promoted. The Pareto gate is your last line of defense here — a candidate that regressed on the clean held-out set never ships, no matter what the field telemetry said.

**Re-optimization moves the Pareto point.** You retrain, then re-quantize and re-prune with the same recipe — and discover the new weights quantize *worse* (a layer that was robust to int8 now has a heavier tail, so int8 costs more accuracy than before). The optimized candidate is dominated even though the fp32 retrain improved. This is why step 4 re-validates on-target, not in fp32: the thing that ships is the optimized artifact, and its Pareto point is what gates promotion. If re-optimization dominates the old build, ship; if not, you may need [mixed-precision or sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) to recover the lost accuracy before the candidate is shippable. The loop is not "retrain → ship"; it is "retrain → re-optimize → *prove* you are still on the frontier → ship."

**The OTA itself drifts the fleet apart.** Staged rollouts mean that at any moment your fleet is running *several* builds — 1% on the new one, 99% on the old, plus a long tail of devices that never updated (airplane mode, storage full, an OS too old to fetch). Your monitoring must be build-aware: a PSI computed across mixed builds is meaningless. Tag every telemetry payload with its `build`, segment all metrics by build, and accept that "the fleet runs one model" is a fiction — it runs a *distribution* over recent builds, and your dashboards must reflect that or they will mislead you. The long tail of never-updating devices is its own product decision: at some point you force-deprecate a build (refuse to serve its API, prompt an app update) because supporting an unbounded number of historical builds is unbounded cost.

The honest conclusion from stress-testing: the feedback loop is not an autopilot. It is an instrumented, gated, human-supervised system where every automated step (detect, retrain, re-optimize, roll out) has a gate a human designed and a failure mode a human anticipated. The automation buys you *speed and consistency*; it does not buy you the right to stop thinking.

## Device heterogeneity: one model per tier

Back to Assumption 3, because it threads through every pillar. A consumer fleet is not one device; it is a distribution over silicon, and the same logical model fans out into a per-tier set of optimized artifacts (Figure 7).

![A tree showing a 4.1 million device fleet splitting into NPU, GPU, and CPU tiers, each resolving to its own optimized artifact with distinct p99 latencies](/imgs/blogs/edge-mlops-7.png)

Concretely, one training run for the wakeword model yields three deployable builds:

- **NPU flagship tier (~38% of fleet):** int8 per-channel, built for the NPU delegate, p99 9 ms. The headline target.
- **GPU mid-range tier (~44%):** fp16, GPU delegate, p99 31 ms. int8 would be faster but a chunk of mid-range GPU delegates have flaky int8 op support, so fp16 is the safe robust choice for this tier.
- **CPU low-end tier (~18%):** int8 with the XNNPACK delegate, p99 74 ms. Slower, but it runs, and an 18% slice with no model at all is 18% of users with a broken feature.

Each build is a separate registry entry with its own recipe, hash, and metrics; each is gated separately in OTA (`device_tier` in the manifest); each is monitored separately (a drift alert on the CPU tier may not appear on the NPU tier, because they see different device populations and different inputs). This is why the registry's unit is the *per-target artifact*, not the model — heterogeneity is not an edge case, it is the default state of any real fleet, and an MLOps system that assumes one artifact will be wrong the first time a runtime update changes one tier's op support and not another's.

The operational cost is real: three artifacts to build, validate, monitor, and roll out is three times the surface area. The mitigation is to *minimize the number of tiers* — collapse the long tail of low-end devices into one robust CPU build rather than chasing per-chip optimization, and only split a tier when a measured failure (the LiteRT 2.16 op fallback) forces it. Tier proliferation is a tax; pay it only where the telemetry proves it buys you something.

## A cloud-vs-edge MLOps comparison

Pulling the contrasts into one table — this is the cheat sheet for someone coming from cloud MLOps:

| Concern | Cloud MLOps | Edge MLOps |
|---|---|---|
| Input visibility | Full — every request logged | None — privacy; monitor proxies only |
| Registry unit | Model checkpoint + metrics | (recipe, target, runtime, artifact-hash) tuple |
| Update mechanism | Redeploy container | Signed OTA artifact, no app release |
| Update latency | Seconds | Hours to days across the fleet |
| Rollback | `rollout undo`, instant | OTA (or local pinned build); inherits propagation latency |
| Hardware | Uniform, you chose it | Heterogeneous; one artifact per tier |
| Drift detection | On input distribution | On output/confidence proxies (PSI) |
| Labels for retrain | From production logs | Scarce; federated learning / human-in-loop |
| Bandwidth cost | Negligible (internal) | Real — delta updates mandatory at scale |
| Blast radius control | Canary deploy | Staged rollout sized by sample-size math |

And the registry-entry shape, as a reference table for what a single entry must carry:

| Field | Example | Why it is in the registry |
|---|---|---|
| Identity | `wakeword build 412` | Monotonic version, the primary key |
| Recipe | `int8 per-channel + 30% structured prune` | Reproduce the artifact bit-for-bit |
| Calibration hash | `4d1e…88, 512 samples` | PTQ ranges depend on it; pin for determinism |
| Converter | `litert==2.16.1` | A minor bump is a new recipe |
| Target + runtime | `Pixel 8 NPU, LiteRT 2.16` | Op support is runtime-version-specific |
| Artifact hash | `sha256 9f2a…e7` | End-to-end integrity; OTA verifies against it |
| On-target metrics | `97.2% acc, p99 9 ms, 240 KB` | The Pareto point you promote against |
| Provenance | `train a91c4e2, ds ww-2026.05` | Audit trail, regression bisection |

And the monitoring-signals table — signal, what it catches, privacy posture — so the on-call engineer knows which alert means what:

| Signal | Catches | Privacy posture | Label-free |
|---|---|---|---|
| Confidence PSI | Data drift, partial concept drift | Aggregate histogram only | Yes |
| Output-class rate | Class-mix shift, runaway class | Counts only | Yes |
| Latency p50/p99 | Op fallback, thermal throttle | Timing only | Yes |
| Crash / ANR rate | OOM, bad delegate, bad artifact | Stack hash only | Yes |
| Battery / energy delta | Runaway inference, 4× fallback | mWh aggregate only | Yes |
| Federated label trickle | True accuracy, deep concept drift | DP + secure aggregation | No (the exception) |

## A privacy-preserving telemetry payload

To make the privacy posture concrete, here is what a device actually uploads. Note there is no raw input anywhere — only bucketed counts, percentiles, and rates. This is the payload the PSI monitor and the rollout controller consume.

```json
{
  "model": "wakeword",
  "build": 412,
  "device_tier": "npu_flagship",
  "rollout_ring": "early",
  "experiment": { "id": "prune-v413", "arm": "control" },
  "window_utc": "2026-06-12T00:00:00Z/PT24H",
  "inference_count": 1843,
  "confidence_histogram": {
    "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "counts": [12, 19, 31, 44, 71, 132, 218, 341, 502, 473]
  },
  "output_class_rate": { "wake": 0.018, "noise": 0.982 },
  "latency_ms": { "p50": 6.2, "p90": 8.1, "p99": 9.3 },
  "stability": { "crash_count": 0, "anr_count": 0, "fallback_to_cpu": 0 },
  "energy_mwh_per_1k_inf": 0.42
}
```

Three privacy properties make this safe to collect at fleet scale:

1. **No raw inputs, ever.** The most granular thing here is a confidence *bucket count*. You cannot reconstruct an audio frame from "44 inferences landed in confidence bin [0.3, 0.4)."
2. **Aggregated over a window.** A 24-hour window of 1,843 inferences as ten histogram bins is k-anonymous in practice — no single inference is identifiable.
3. **Composable on the server.** The server *sums* these payloads across the ring before computing PSI, so even the per-device histogram is an intermediate the analyst never has to inspect. This is federated analytics in its simplest form, and it is enough to power every monitoring signal in the table above.

For stronger guarantees on tiny populations you add **local differential privacy** — each device adds calibrated noise to its counts before upload, so even the aggregate cannot leak an individual — at the cost of needing a larger ring to see the same effect (the noise raises the detectable-effect floor, which loops back to the sample-size math). For most fleets, aggregate-and-window is sufficient; DP is the dial you turn when the population is small or the data is sensitive enough that even aggregates are scrutinized.

## When full edge MLOps is warranted — and when it is overkill

Everything above is *cost*. A registry, an OTA service, a CDN with deltas, a signing pipeline, a fleet-telemetry backend, a drift monitor, a staged-rollout controller — that is a platform, and platforms have an owner, an on-call rotation, and a maintenance bill. Do not build it reflexively. The decision turns on three axes (Figure 8).

![A decision tree branching from shipping a model to devices into a small-or-static path that ends at bundling the model in the app, a large-or-evolving path that ends at full edge MLOps, and a safety-or-regulated path that adds signing and reproducible recipes](/imgs/blogs/edge-mlops-8.png)

**Build full edge MLOps when:**

- **The fleet is large.** Past a few hundred thousand devices, a bad bundled model is a recall-scale event and a staged OTA is the only safe way to ship anything. The blast-radius control alone justifies the platform.
- **Updates are frequent.** If the model improves monthly (most do, as you collect data and re-optimize), chaining its cadence to app releases throttles your entire improvement loop. OTA decoupling pays for itself in iteration speed.
- **The domain is safety-critical or regulated.** Medical, automotive, finance. You need the audit trail (provenance + signed reproducible recipes), the guaranteed rollback, and the integrity chain. Here even a *small* fleet warrants the registry and signing pieces, because the requirement is reproducibility and auditability, not scale.

**Skip it — ship a bundled static model — when:**

- **The model is small, static, and rarely changes.** A barcode-orientation classifier that has worked for three years does not need a drift monitor and an OTA channel. Bundle it, ship it, save the platform cost.
- **The fleet is tiny or internal.** A model on fifty kiosks you control is closer to a managed deployment than a consumer fleet; you can update them on a schedule and watch them directly.
- **You have no way to retrain.** If a drift alert has no downstream action — no fresh data, no retrain pipeline — then the monitor is a pager that only causes pain. Build the monitor when you can act on it.

The **org/ownership** angle is the part teams underestimate. Edge MLOps is a *platform*, and a platform with no owner rots faster than the models it serves. Someone owns the registry schema, someone owns the OTA controller and its guardrail thresholds, someone is on-call for a drift page at 2 a.m. and knows whether to roll back or ride it out. If you cannot name those owners, you are not ready to build the platform — you are ready to bundle a static model and revisit when the fleet and the cadence justify the headcount. The most common edge-MLOps failure is not technical; it is a half-built platform nobody owns, with a drift monitor that pages into a void.

## Case studies and real numbers

A few grounded reference points from shipped systems and the literature; where a number is approximate or my characterization, I say so.

- **Mobile app staged rollouts (Google Play, App Store Connect).** Both stores ship first-party staged-rollout tooling: Play lets you release to a percentage of users and halt/resume; App Store Connect's phased release ramps over seven days with a pause control. The edge-MLOps pattern in this post applies that exact discipline to the *model artifact* rather than the app binary — same staged-ring idea, same halt-on-regression gate, just decoupled from store review. This is well-documented public tooling.

- **Firebase ML / Remote Config style model delivery.** The pattern of fetching a model artifact at runtime, gated by a server-controlled rollout percentage, is what Firebase ML Custom Model hosting and Remote Config conditional delivery implement in practice — a device asks the server which model version to use and downloads it out-of-band from the app release. This is the productized form of the OTA manifest in Pillar 2.

- **Federated analytics for keyboard models (Gboard).** Google's published work on federated learning and analytics for the Gboard keyboard is the canonical real-world example of the privacy-preserving loop in this post: next-word and emoji models trained on-device with secure aggregation, and *analytics* (e.g. discovering out-of-vocabulary words) computed federally — raw keystrokes never leave the device. The 2017-onward federated-learning papers from Google (McMahan et al., and the federated-analytics follow-ups) are the primary sources; treat the specific numbers as paper-reported, not my measurement.

- **PSI in production drift monitoring.** Population Stability Index originates in credit-risk scorecards and is the de-facto industry default for distribution-shift alerting, with the $0.10 / 0.25$ thresholds I used carried over from that practice. Open-source drift libraries (Evidently, NannyML, river) implement PSI and KL-based detectors directly; the contribution of this post is applying PSI to *on-device confidence histograms* (a label-free proxy) rather than to logged cloud inputs.

- **TFX / MLflow model registries.** The registry-as-source-of-truth design — versioned models, stage transitions, lineage — is established cloud-MLOps practice (MLflow Model Registry, TFX ML Metadata). The edge extension this post argues for is expanding the registry's unit from "checkpoint + metrics" to the full "recipe + per-target artifact + on-target metrics" tuple, because the [artifact you ship is not the model you trained](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact).

The honest summary across these: the *mechanisms* — staged rollout, OTA model delivery, federated analytics, PSI drift, versioned registries — are all shipped, documented technology. What this post assembles is the *edge-specific composition* of them, and the few specific latency/size numbers (9 ms p99, 240 KB, 980 GB egress) are illustrative engineering figures for the running wakeword example, not measurements of a named public system.

## Key takeaways

- **Shipping the model is the easy 10%.** Operating a fleet — versioning, updating, watching, retraining — is edge MLOps, and it is its own engineering discipline, not a cloud-MLOps footnote.
- **The registry's unit is the recipe, not the checkpoint.** Pin quantization scheme, pruning ratio, calibration-set hash, converter version, target, runtime, and the artifact's content hash. Reproducibility of an *optimized* artifact is harder than a checkpoint and you must check it in CI.
- **OTA decouples the model's cadence from the app's.** True OTA model delivery — signed, hashed, delta-encoded, tier-gated — lets you ship a fix this afternoon to 1% of devices with no store review. It is the single highest-leverage capability in the stack.
- **Staged rollout is your only blast-radius control.** Size the canary by the sample-size formula $n \propto 1/(p_1-p_0)^2$ to the smallest regression you care about, not to a round percentage. Catch it at 1%, roll back before it reaches 100%.
- **You cannot see the inputs, so monitor proxies.** Confidence PSI, output-rate, latency p99, crash rate, battery — all label-free, all privacy-safe aggregates. PSI on the confidence histogram is your primary drift signal; the $0.25$ threshold is the page.
- **Distinguish data drift from concept drift.** Proxies catch data drift and operational faults well, concept drift only partially. For high-stakes concept drift, supplement with a small labeled stream — do not oversell what confidence histograms can do.
- **Heterogeneity is the default.** One training run, one artifact per hardware tier, each versioned and monitored separately. Minimize tiers; split one only when telemetry proves a failure forces it.
- **Close the loop with federated learning.** Telemetry → detect → retrain → re-optimize on the Pareto frontier → re-validate → staged OTA. When the privacy that justified going on-device also forbids collecting data, federated learning feeds the retrain step.
- **Build the platform only when scale, cadence, or regulation justify it — and only with a named owner.** The most common failure is a half-built platform nobody owns, paging a drift alert into a void. Otherwise, bundle a static model and revisit later.

## Further reading

- **Within this series:** [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) (the four-lever frame whose artifacts you operate), [the edge deployment lifecycle](/blog/machine-learning/edge-ai/the-edge-deployment-lifecycle) (how the artifact is made), [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact), [mobile deployment end to end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end) (asset vs OTA delivery), [the accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier) (where you re-validate), [on-device and federated learning](/blog/machine-learning/edge-ai/on-device-and-federated-learning) (the privacy-preserving retrain channel), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- **Federated learning and analytics:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017) — the FedAvg paper; and the Google AI blog series on federated learning and federated analytics for Gboard, which document the privacy-preserving loop in production.
- **Drift detection:** the Population Stability Index literature from credit-risk scorecard practice; and open-source implementations in Evidently, NannyML, and river for PSI/KL distribution-shift monitoring.
- **Registries and staged rollout:** the MLflow Model Registry and TFX ML Metadata docs for registry-as-source-of-truth design; and the Google Play Console / Apple App Store Connect docs on staged and phased releases, whose ring-and-halt model maps directly onto OTA model rollouts.
- **On-device model delivery:** Firebase ML Custom Model hosting and Remote Config conditional delivery docs for productized runtime model fetching and percentage-gated rollout.

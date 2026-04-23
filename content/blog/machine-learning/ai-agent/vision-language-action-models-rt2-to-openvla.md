---
title: "Vision-Language-Action Models: From RT-2 to OpenVLA and pi0"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["VLA", "RT-2", "OpenVLA", "pi0", "Robotics", "Foundation Models"]
date: "2026-04-18"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "Deep dive into the VLA lineage: RT-1, RT-2, RT-X, OpenVLA, Octo, and pi0 / pi-0.5. Architectures, action tokenization tradeoffs, inference latency on Jetson and RTX, fine-tuning recipes for new embodiments, and the open problems blocking general-purpose robot foundation models."
---

# Vision-Language-Action Models: From RT-2 to OpenVLA and pi0

A Vision-Language-Action (VLA) model is a single neural network that takes images and a text instruction and emits **robot actions**. Five years ago this sentence would have been science fiction. Today it is a competitive category with four production-ready open models and a growing ecosystem of fine-tuning recipes. This article traces the VLA lineage and gives a practitioner's view of what actually works when you need to deploy one on a humanoid by next quarter.


## 1. Why VLA at All?

Classical robot learning had three separate problems:

1. **Perception** (what is in front of me?) - solved with ResNets / ViTs.
2. **Planning** (what should I do?) - solved with task planners, MCTS, or LLMs.
3. **Control** (how do I move?) - solved with MPC or learned policies.

Each stage had its own losses, its own data format, and its own failure modes. Every integration was a dissertation.

The VLA hypothesis: **fold all three into one transformer**, trained on large-scale robot data plus internet-scale vision-language data. If it works, you get:

- Transfer of web commonsense to physical tasks.
- Unified action abstraction across embodiments.
- A single model to serve, quantize, and fine-tune.

If it fails, you get a system with no debugging handles and a 300ms inference latency on your Jetson.

The honest answer is that as of 2026 it **partially works**: VLAs dominate short-horizon manipulation and language generalization, but still need classical stacks for long-horizon planning, precise insertion, and safe locomotion.


## 2. The Lineage

### 2.1 RT-1 (Google, 2022): The Template

RT-1 is the template every subsequent VLA refines. Architecture:

```
  +-------------+     +-------------+    +------------+
  | 6 RGB frames|---->| EfficientNet|    |            |
  | (300x300)   |     |   + FiLM    |--->|            |
  +-------------+     +-------------+    |            |
  +-------------+                        | Token-     |
  | instruction |-->  Universal          | Learner    |
  |  (text)     |     Sentence Encoder ->|  + TF dec  |---> 11 action tokens
  +-------------+                        +------------+       (6-dof arm, gripper,
                                                               base x/y/yaw, mode)
```

Key ideas already present:

- **Action discretization**: each dimension into 256 bins, predicted autoregressively as tokens.
- **Language-conditioned**: FiLM layers modulate vision features by text.
- **Token compression**: TokenLearner reduces 81 vision tokens to 8 per frame, for tractable inference.

Results: 97% seen task success, 76% unseen, trained on **130k** episodes over 744 skills at Google. Inference: 3 Hz on TPU.

### 2.2 RT-2 (Google DeepMind, 2023): VLM Co-training

RT-1 could not read a menu. RT-2 could. The insight: start from a pre-trained VLM (PaLI-X 55B or PaLM-E 12B) and co-fine-tune on robot trajectories **mixed with** the original VLM internet data.

The tokenization trick:

```
  VLM vocab: ["the", "cat", ..., "<tok_246>", "<tok_1093>", ...]
                                    ^
                                    |
                          repurpose rarely-used tokens
                          to encode discretized actions

  Training batch: 50% web VQA, 50% robot trajectories
  Loss: next-token prediction over the unified vocabulary
```

No new parameters. No new heads. Actions become just another language.

**Emergent generalization** findings:

- "Pick the object that is not a fruit" - RT-2 generalized from categories never in robot data.
- "Move the banana to the sum of 1 + 2" - arithmetic + visual counting transferred.
- "Pick up the extinct animal" - identified a toy dinosaur.

Inference latency: 55B model could only run at 1-3 Hz on a cloud TPU v4 pod. RT-2-X (using PaLM-E 12B) hit 5 Hz. Latency is why RT-2 never shipped on-robot at full size; distillations followed.

### 2.3 RT-X and Open X-Embodiment (2023)

The RT-X paper and Open X-Embodiment (OXE) dataset are arguably the most important VLA contributions of 2023. OXE aggregated **22 embodiments, 527 skills, 1M+ trajectories** from 34 labs into a unified Zarr format.

Action unification was non-trivial: Franka, UR5, Google Robot, xArm, Sawyer all have different joint limits, gripper ranges, and action conventions. OXE's approach:

- Relative end-effector deltas as a common substrate.
- Per-dataset action normalization stats (1st-99th percentile clip).
- Embodiment ID token prepended to the sequence.

Training RT-1 and RT-2 on OXE gave a **positive transfer on 9 of 9 embodiments** and unlocked the subsequent open-model wave.

### 2.4 OpenVLA (Stanford + Google + TRI, 2024)

OpenVLA is the first **open-weights 7B VLA** people actually fine-tune. Architecture:

```
  image (224)  -> DINOv2 + SigLIP dual encoder  -> 256 visual tokens
                                                            \
  text ("pick up the marker")  -> Llama-2 tokenizer -------->  Llama-2 7B
                                                                    |
                                                                    v
                                            next-token prediction over
                                            256-bin discretized 7-DoF actions
                                            (by overloading least-used Llama tokens)
```

Training: 970k OXE episodes, 27 epochs, ~21.5 days on 64x A100. Uses **Prismatic VLM** as the starting point.

**Open code, open weights, LoRA-ready.** This is why OpenVLA won mindshare: you can fine-tune on a single 4090 with LoRA rank 32 in under a day for a new embodiment.

Zero-shot OXE success rate: 48%. Fine-tuned on in-domain demos (~200 episodes), OpenVLA hits 80-90% on tasks the model has never seen.

### 2.5 Octo (Berkeley, 2024)

Octo takes a different architectural bet: **diffusion policy head** instead of discrete tokens.

```
  image + language  -->  small transformer (27M or 93M)  -->  action readout
                                                                    |
                                                                    v
                                                      diffusion head (DDPM, 10 steps)
                                                                    |
                                                                    v
                                               continuous action chunk (4 steps)
```

Why diffusion: smooth continuous actions, better for dexterous tasks; action chunks (predicting several future actions jointly) prevent single-step jitter.

Octo is smaller (93M vs 7B) and faster (~25 Hz on RTX 4090), but less capable at language generalization out-of-distribution because the text encoder is small.

### 2.6 pi0 and pi0.5 (Physical Intelligence, 2024-2025)

pi0 combines the best ideas: VLM backbone + flow-matching action head.

```
  image + language
        |
        v
  +-----------------------+
  |  PaliGemma 3B backbone|  (frozen or jointly trained)
  +-----------------------+
        |
        +---cross attention---+
                              v
  +---------------------------+
  |   Action Expert           |  (300M, flow matching)
  |   - predicts 50 future    |
  |     actions at 50 Hz      |
  +---------------------------+
        |
        v
  action chunk (50 steps, continuous)
```

Flow matching beats diffusion on inference speed (single-pass sampling vs 10-50 denoising steps). The action chunking means you only need to forward the big VLM ~1 Hz while the chunk is executed open-loop at 50 Hz - matching human reaction cadence.

pi0.5 (2025) extends pi0 with **hierarchical sub-task conditioning**: the VLM first emits a high-level sub-task ("open drawer"), then the action expert executes it. This pushes long-horizon performance substantially.


## 3. Action Tokenization: The Hidden Design Axis

This is the most underdiscussed design choice in VLAs.

### 3.1 Discrete bins (RT-1, RT-2, OpenVLA)

Each action dimension is quantized into 256 bins, each a token.

| Pros | Cons |
|---|---|
| Fits LLM training infra exactly | Quantization error accumulates |
| Can share softmax with vocab | No smoothness prior |
| Autoregressive = per-dim conditioning | Slow (N tokens per action) |

A 7-DoF action = 7 forward passes at inference. At 200 ms per pass, that's... bad. OpenVLA uses parallel decoding tricks to mitigate.

### 3.2 Continuous regression (naive)

Output a 7-dim vector directly with MSE loss. Simple, fast, **mode collapses** on multi-modal action distributions (e.g., "pick with either hand" becomes "pick with the average of both hands" = pick with neither).

### 3.3 Diffusion / DDPM (Octo, Diffusion Policy)

Reverse diffusion over action sequence. Handles multi-modality, smooth outputs. Cost: 10-50 denoising steps per action.

### 3.4 Flow matching (pi0, pi0.5)

One-pass continuous-time ODE integration. Similar quality to diffusion, 5-10x faster inference. The method of choice in 2025-2026.

### 3.5 Action chunking (ACT, pi0)

Predict H future actions jointly (H = 8-50). Key effect: reduces inference Hz requirement by factor H. pi0 runs VLM at 1 Hz but executes at 50 Hz thanks to H=50 chunks.

Chunking hurts reactivity: if the environment changes mid-chunk, the robot is committed to stale actions. Solutions:

- **Temporal ensembling** (ACT): at each timestep, average overlapping chunks from multiple forward passes.
- **Chunk truncation**: execute only first k < H steps, re-plan.

### 3.6 Comparison matrix

| Method | Smoothness | Multi-modal | Inference cost | Used by |
|---|---|---|---|---|
| Discrete bins | Low | Yes (autoregressive) | High (N passes) | RT-1/2, OpenVLA |
| MSE | Medium | No | Low | RT-1 early |
| Diffusion | High | Yes | High (N denoise) | Octo, DP |
| Flow matching | High | Yes | Medium (1-5 ODE) | pi0, pi0.5 |
| Chunking (orthogonal) | - | - | Amortized | ACT, pi0 |


## 4. Training Data: What Actually Matters

### 4.1 Open X-Embodiment (OXE)

The workhorse. 1M+ trajectories, 22 embodiments. Caveats:

- Heavy skew: ~half is Google RT-1 data (pick-and-place).
- Action conventions vary; normalization is mandatory.
- Camera placement inconsistent.
- Only ~10% has language annotations at episode-level.

### 4.2 DROID (2024)

564 hours of Franka Panda teleop, 86k episodes, **multiple camera views, diverse scenes, curated language annotations**. Best open dataset for dexterous manipulation with consistent embodiment.

### 4.3 BridgeData V2 (2023)

60k episodes on a WidowX, cheap embodiment, great for sim-to-real baselines.

### 4.4 Internal teleop data

Every serious VLA shop (Figure, 1X, PI, Tesla) has an internal teleop fleet producing 1k-10k hours. The competitive edge in 2026 is **data**, not architecture. Specifically:

- **Diverse scenes**: 1000 different kitchens beat 10000 episodes in the same kitchen.
- **High-quality language**: episode-level plus sub-step captions. Crowdsourced LLM re-labeling (GPT-4V over videos) is standard.
- **Failure data**: episodes labeled with what went wrong are 10x more valuable for robustness.

### 4.5 Data mixing recipes

Standard 2025 recipe:

```
70% robot data (OXE + internal)
20% web image-text (LAION subset, captions)
10% web video (action-describing)
```

The 30% non-robot data is crucial. Without it, the VLA forgets visual generalization and regresses on "pick the fruit that ripens in summer."


## 5. Inference Latency: Where Dreams Meet Silicon

This is where VLA papers become vague and engineers become miserable.

### 5.1 Target budgets

For smooth humanoid manipulation, target latencies:

| Layer | Target Hz | Per-action budget |
|---|---|---|
| Reactive control | 500-1000 Hz | 1-2 ms |
| Low-level policy | 50-200 Hz | 5-20 ms |
| VLA skill policy | 10-50 Hz | 20-100 ms |
| Planner / System-2 | 1-5 Hz | 200-1000 ms |

A 7B VLA at BF16 on a Jetson AGX Orin 64GB does **not** hit 50 Hz. Typical numbers:

| Model | Precision | Hardware | Throughput | Notes |
|---|---|---|---|---|
| OpenVLA 7B | BF16 | RTX 4090 | 6 Hz | No action chunking |
| OpenVLA 7B | INT4 AWQ | RTX 4090 | 15 Hz | Some accuracy loss |
| OpenVLA 7B | INT8 | Jetson AGX Orin | 2-3 Hz | Memory bandwidth bound |
| OpenVLA 7B w/ parallel decode | BF16 | RTX 4090 | ~25 Hz | OpenVLA-OFT technique |
| Octo 93M | FP16 | RTX 4090 | 25 Hz | Diffusion, 10 steps |
| pi0 3.3B | BF16 | RTX 4090 | 50 Hz effective (with chunks) | Real wall-clock |
| pi0 3.3B | INT8 | Jetson AGX Orin | 10-15 Hz effective | With chunking |

### 5.2 Latency optimization techniques

1. **Action chunking**: execute H steps open-loop, amortize VLM cost by H. Biggest single lever.
2. **KV-cache**: obvious for transformers, still missed in some research codebases.
3. **Continuous batching**: if you run multiple robots, share a server.
4. **Quantization**: INT8 and INT4 AWQ are table stakes. Watch for action accuracy regression (measure in normalized action MSE).
5. **Distillation**: distill a 7B VLA into a 1B model for deployment. Figure Helix's System-1 is effectively a distilled fast policy.
6. **Parallel action decoding**: predict all 7 action tokens with a single forward (OpenVLA-OFT, 2025).
7. **Speculative decoding**: cheap in text LLMs, complex in VLA because draft models must also understand vision; emerging in 2026.
8. **Vision token compression**: SigLIP outputs 256 tokens per image; token merging or Q-former reduces to 32-64 with small quality loss.

### 5.3 System-level latency

Raw model throughput is not the end. You also pay:

- Camera capture and ISP: 15-33 ms at 30-60 FPS.
- USB / GMSL transport: 2-5 ms.
- Preprocessing (resize, normalize): 2-5 ms.
- Action post-processing and safety check: 1-3 ms.
- Controller ingestion: 1-2 ms.

Your 50 Hz VLA becomes a 25 Hz closed loop in practice. Always measure end-to-end.


## 6. Fine-tuning a VLA for a New Embodiment

Suppose you have a new humanoid with a 28-DoF action space (torso + dual arms + fingers + head). Typical recipe:

### 6.1 Data collection

- **Teleoperation**: 50-200 hours across 20-50 tasks, 5-10 scenes each.
- **Annotation**: episode-level language + 2-3 sub-step captions per episode (GPT-4V auto-label, human audit).
- **Curation**: remove idle segments, failed episodes where you want to keep only intentional failures.

### 6.2 Action head adaptation

OpenVLA is 7-DoF native. For 28-DoF, either:

- **Factorize** the action into end-effector pose per arm (7+7) + torso (3) + fingers (16) - match to 7-DoF heads where possible.
- **Add new action tokens** by extending the tokenizer and initializing as average of existing action tokens.
- **Replace** the action head with a task-space IK target (hand pose + contact mode), delegating joint-level detail to a whole-body controller.

The **task-space** route is best for humanoids - the VLA reasons about hand goals, the WBC solves the 28-DoF redundancy.

### 6.3 LoRA recipe (OpenVLA)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(openvla_base, lora_config)

# Train 5-20 epochs on ~200 trajectories
# Batch size 16, LR 5e-4, warmup 100 steps
# On 1x RTX 4090, ~6-10 hours
```

Key hyperparameters:

- **Rank**: 32 is a strong default; 8 for small data, 64 for >1000 trajectories.
- **LR**: 5e-4 for LoRA, 1e-5 for full fine-tune.
- **Warmup**: 100-500 steps; VLAs are sensitive to early LR spikes.
- **Action normalization**: critical. Compute per-dataset 1st-99th percentile, clip, and re-scale to [-1, 1].

### 6.4 Evaluation protocol

Hold out:

- **Seen tasks, seen scenes**: sanity check.
- **Seen tasks, new scenes**: visual generalization.
- **New tasks, seen scenes**: language/skill generalization.
- **New tasks, new scenes**: the real test.

Run 20+ rollouts per cell, report success rate + CI. Video every failure; categorize (perception, grounding, motor, affordance).


## 7. Evaluation of VLAs

### 7.1 Core metrics

- **Task success rate**: binary completion. Coarse but decision-relevant.
- **Language-conditioned accuracy**: if you say "pick red," does it pick red vs blue? Needs paired instructions.
- **Compositional generalization**: new combinations of seen concepts.
- **Robustness**: success under lighting, occlusion, object pose perturbation.
- **Safety violations**: force threshold breaches, collision events.
- **Action MSE / normalized distance**: during training only; weakly correlated with downstream success.

### 7.2 Benchmarks

| Benchmark | Domain | Pros | Cons |
|---|---|---|---|
| SimplerEnv | Simulated OXE tasks | Cheap, reproducible | Sim-to-real gap |
| LIBERO | 130 lifelong tasks | Skill diversity | Single embodiment |
| BEHAVIOR-1K | Household in Omniverse | Long-horizon | Heavy sim |
| RoboArena (2024) | Real-world eval harness | Real robots | Limited scale |
| CALVIN | Long-horizon language | Multi-step | Tabletop only |

### 7.3 Common evaluation mistakes

1. **Reporting only the best seed.** VLAs have high run-to-run variance; report mean + 95% CI over 3+ seeds.
2. **Cherry-picked demos.** Always report failure rate alongside success videos.
3. **Unrealistic initial states.** If you always reset the mug to the same pose, you are testing memorization.
4. **No OOD language.** Test paraphrases, typos, and vague instructions.
5. **Ignoring latency.** A 99% success at 1 Hz is useless for a robot that must react in 100 ms.


## 8. Architecture Comparison Cheat Sheet

```
+------------+---------+----------+---------+---------+--------+----------+
|   Model    |  Year   |  Params  |Vision   | Text    |Action  |Hz (prod) |
+------------+---------+----------+---------+---------+--------+----------+
|  RT-1      |  2022   |   35M    |EffNet   |USE      |Discrete|    3     |
|  RT-2-X    |  2023   |  12B-55B |PaLI-X   |PaLM-E   |Discrete|   1-5    |
|  OpenVLA   |  2024   |   7B     |DINO+SigL|Llama-2  |Discrete|   6-15   |
|  Octo-Base |  2024   |   93M    |small TF |small TF |Diffusion|  25     |
|  pi0       |  2024   |  3.3B    |SigLIP   |PaliGemma|Flow mat|   50     |
|  pi0.5     |  2025   |  3.3B    |SigLIP   |PaliGemma|Flow+H  |   50     |
|  GR00T N1  |  2025   |  2B+     |SigLIP   |Eagle-2  |Diffusion|  30     |
+------------+---------+----------+---------+---------+--------+----------+
```

H = hierarchical sub-task conditioning; GR00T is NVIDIA's humanoid-first VLA.


## 9. Fine-tuning Pitfalls I've Hit in Production

- **Data leakage**: the same scene in train and eval because the teleop fleet worked in one room. Enforce scene-level splits.
- **Gripper inversion**: different datasets use "open = 0" vs "open = 1". Normalize early and often.
- **Image resolution**: OpenVLA uses 224x224; if you feed it 512x512, the language generalization quietly degrades because DINO was not trained at that resolution.
- **Language too short**: models trained on "pick up the block" fail when you say "please grab that blue cube on the left." Augment with LLM-generated paraphrases.
- **Action horizon mismatch**: if the base model was trained with chunk H=8 and you fine-tune with H=50, training loss looks fine but execution becomes erratic.
- **Safety regressions**: fine-tuning on contact-rich tasks can erase the VLA's learned soft-contact behaviors if your demos are too aggressive. Weight-reg back toward the base model.


## 10. Open Problems

1. **Long-horizon coherence.** VLAs beyond ~30 s of rollout start to drift. Memory modules and hierarchical policies (pi0.5, GR00T-H) are only partial fixes.

2. **Bi- and multi-manual coordination.** Most OXE data is single-arm. Training data for coordinated dual-arm (e.g., carrying a tray, opening a jar) is scarce and hard to generate.

3. **Sim-to-real for contact-rich tasks.** Rigid-body sims lie about friction, compliant contact, and cables. 2026 approaches: neural contact models, diffusion-based reality gap bridging, real-world fine-tuning on top of sim-trained VLA.

4. **Robust language grounding.** "The one behind the red thing" is still a 50-60% accuracy problem for most VLAs in cluttered scenes. Spatial reasoning is underlearned.

5. **Action space universality.** OXE normalization is a hack. A true universal action space (contact-aware, embodiment-free) remains unsolved.

6. **Safety and uncertainty.** VLAs rarely know when they don't know. Ensembling, evidential outputs, and conformal prediction are active research areas.

7. **Energy and thermal.** A 7B VLA at 15 Hz on-robot dissipates 30-60 W just for cognition. Humanoid thermal budgets are tight; quantized distilled models (1B-class) are a deployment necessity.

8. **Evaluation at scale.** Running 1000 real-robot rollouts to compare two checkpoints is cost-prohibitive. Sim evaluation correlates weakly with real. The field still lacks a cheap, reliable eval harness.


## 11. Senior-Level Takeaways

1. **The VLA bet is a data bet.** Architecture differences (discrete vs flow matching) change performance by 5-10%; a 3x larger, better-labeled dataset changes it by 30-50%.

2. **Flow matching + action chunking is the current sweet spot** for deployment: smooth, multi-modal, and amortizes VLM cost. OpenVLA-style discrete tokens are easier to hack on but slower.

3. **Quantize aggressively**, measure action-space regression, not just perplexity. INT8 typically costs <5% on task success; INT4 AWQ is viable for inference-bound deployments with ~10% regression that can be recovered with quant-aware fine-tuning.

4. **Split System-1 and System-2.** A 7B model for reasoning at 2-5 Hz plus a 100-300M model for motor execution at 50+ Hz beats a single 7B at 10 Hz for humanoid-class latency requirements.

5. **Fine-tune with LoRA first.** Full fine-tunes are expensive and destructive to web priors. Reserve full fine-tunes for substantial embodiment changes.

6. **Do not ship a VLA without a safety veto layer.** Force limits, workspace bounds, collision checks remain classical. VLAs output suggestions; the safety layer decides what executes.

7. **Evaluation must include OOD language.** If your benchmark only uses the 50 training instructions, you are measuring memorization.

8. **Latency kills.** Run end-to-end (camera to actuator) budgets before committing to an architecture. Your 50 Hz model probably becomes a 20 Hz closed loop after IO.


## References

- RT-1: Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale" (2022)
- RT-2: Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023)
- RT-X / OXE: Padalkar et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models" (2023)
- OpenVLA: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)
- OpenVLA-OFT: Kim et al. (2025)
- Octo: Octo Model Team, "Octo: An Open-Source Generalist Robot Policy" (2024)
- pi0: Black et al., "pi_0: A Vision-Language-Action Flow Model for General Robot Control" (2024)
- pi0.5: Physical Intelligence, "pi-0.5" technical report (2025)
- Diffusion Policy: Chi et al. (2023)
- ACT: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (2023)
- DROID: Khazatsky et al. (2024)
- BridgeData V2: Walke et al. (2023)
- LIBERO: Liu et al. (2023)
- BEHAVIOR-1K: Li et al. (2023)
- GR00T: NVIDIA (2024-2025)

The VLA field moved from "interesting paper" to "deployed fleet" in under three years. The next frontier is not bigger models; it is better data pipelines, tighter latency, and honest evaluation that proves the robot works in the hundredth kitchen, not the first one.

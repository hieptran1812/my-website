---
title: "GR00T N1: How NVIDIA Gives a Humanoid Robot a Fast-and-Slow Brain"
date: "2026-06-12"
publishDate: "2026-06-12"
description: "A deep dive into NVIDIA's GR00T N1 — the dual-system vision-language-action foundation model for humanoid robots, its flow-matching action head, cross-embodiment training, and the data pyramid that turns scarce robot data into a generalist policy."
tags: ["robotics", "gr00t", "nvidia", "vision-language-action", "vla", "humanoid", "diffusion-policy", "flow-matching", "foundation-model", "ai-agent", "imitation-learning", "cross-embodiment"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

## The robot's dilemma: it must act, but it has barely any data

Language models had the internet — trillions of tokens of text, free for the scraping. Vision models had billions of captioned images. Robots have almost nothing. There is no internet of robot trajectories; every example of a robot picking up a cup or opening a drawer has to be *collected*, usually by a human teleoperating the robot, one demonstration at a time. A few hundred hours of robot data is a large dataset, where a language model trains on the equivalent of millions of years of reading. This data scarcity is *the* central problem of robot learning, and it is why robots have not had their "foundation model moment" until recently.

NVIDIA's **GR00T N1** is a serious attempt at that moment — an open **vision-language-action (VLA)** foundation model for generalist humanoid robots. It tackles the data problem from two directions at once. First, an **architecture** that splits cognition into a slow, deliberate vision-language "System 2" that reasons about the task and a fast, reactive "System 1" diffusion transformer that generates fluid motor actions — the same fast-and-slow division Daniel Kahneman described in human cognition. Second, a **data pyramid** that stretches the scarce real-robot data with a huge base of human video and a middle layer of synthetic trajectories, using clever tricks to make action-less video trainable. The result is a 2.2B-parameter model that, on a real Fourier GR-1 humanoid, hits **76.8% average success** where a from-scratch diffusion policy gets 46.4% — and **42.6% vs 10.2%** when task data is scarce, which is exactly when a foundation model should help most.

This is the seventh and final post in a series reading NVIDIA's model reports for their reusable techniques. The first six spanned language, speech, and the world model ([Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), [Nemotron-4 340B](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment), [Llama-Nemotron](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models), [Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer), [Canary/Parakeet](/blog/machine-learning/deep-learning/canary-parakeet-fastconformer-asr), [Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer)); this one closes the loop with the part that *acts*. It draws on the [GR00T N1 report](https://arxiv.org/abs/2503.14734). The reusable techniques are the **dual-system architecture**, the **flow-matching action head**, **cross-embodiment training**, and the **data pyramid** with pseudo-action recovery. It pairs naturally with the [grounding-language-in-robot-actions](/blog/machine-learning/ai-agent/llm-to-control-grounding-language-in-robot-actions) discussion.

The mismatch the whole design resolves:

| Question | The naive assumption | What GR00T N1 shows |
|---|---|---|
| One model for perception and control? | One network does it all | No — split into slow reasoning + fast control |
| What rate should a robot policy run at? | One frequency | Two — 10 Hz for planning, 120 Hz for motor control |
| Is robot data enough to train a policy? | Collect more demos | No — stretch it with human video + synthetic data |
| Can action-less human video help? | No, it has no actions | Yes — recover pseudo-actions and use it |
| One model per robot? | Train per embodiment | No — cross-embodiment, one model, many bodies |
| How do you generate motor actions? | Regress them directly | Flow-matching: denoise an action chunk |

![Pipeline diagram of GR00T N1: a camera image and instruction feed System 2, an Eagle-2 vision-language model running at 10 Hz that reasons and plans, producing vision-language tokens that System 1, a diffusion transformer running at 120 Hz with flow-matching, cross-attends to generate a 16-action chunk in 63.9 milliseconds that drives the robot's motors across arm, bimanual, and humanoid embodiments](/imgs/blogs/gr00t-n1-vla-humanoid-robots-1.webp)

It is worth naming what makes robotics genuinely harder than the modalities earlier in this series, because it explains why GR00T N1 needs every trick it uses. A language model is judged on text it produces; a vision model on labels it predicts; both have abundant data and forgiving, offline evaluation. A robot is judged on whether it *physically accomplishes a task in the real world*, with scarce data, real-time latency constraints, multimodal valid actions, and an unforgiving reality that does not grade on a curve — the cup is either grasped or it falls. This combination — scarce data, hard real-time, multimodal actions, embodiment diversity, and zero tolerance for averaged-out mush — is why robot learning lagged the other modalities, and why GR00T N1's design is a stack of solutions to each: the data pyramid for scarcity, the dual system for latency, flow-matching for multimodality, cross-embodiment for diversity. Each technique in this post exists because one of those constraints would otherwise break a naive single-model approach. Keep the constraints in mind as we go; they are the *why* behind every design choice.

The diagram above is the mental model: an image and an instruction go into a **slow vision-language planner (System 2)**, which produces a plan that a **fast diffusion-transformer controller (System 1)** turns into motor actions in real time. The rest of this article walks each piece — the two systems and why they run at different speeds, the flow-matching action head, the cross-embodiment trick, and the data pyramid that makes it all trainable despite the data scarcity. The organizing idea:

> A robot needs to think slowly about *what* to do and act quickly on *how* to do it, and one model cannot do both well at one speed. GR00T N1 splits these into two systems running at two rates, and feeds them with a pyramid of data that stretches scarce robot demonstrations with abundant human video.

## 1. The dual-system architecture: think slow, act fast

The central design choice, and the most reusable, is splitting the model into two systems inspired by the fast-and-slow dichotomy of human cognition.

![Matrix comparing the two systems: System 2 is the vision-language model, its role is to reason and plan, it runs slowly at 10 Hz, and it is the Eagle-2 VLM with 1.34 billion parameters; System 1 is the diffusion transformer, its role is motor actions, it runs fast at 120 Hz, and it is a diffusion transformer](/imgs/blogs/gr00t-n1-vla-humanoid-robots-2.webp)

The two systems, by analogy to Kahneman's *Thinking, Fast and Slow*:

- **System 2 — the vision-language model (slow, deliberate).** Built on the **Eagle-2 VLM** (1.34B of the model's 2.2B parameters), System 2 takes the camera image (224×224, encoded to 64 tokens per frame) and the language instruction, and *reasons* about them — understanding the scene, parsing the instruction, planning what to do. It runs at **10 Hz**, because deliberate reasoning does not need to happen every millisecond. Its output is a set of **vision-language feature tokens** (extracted from layer 12 of the VLM) that summarize "here is the situation and here is the plan."
- **System 1 — the diffusion transformer (fast, reactive).** A **Diffusion Transformer (DiT)** using flow-matching, System 1 takes System 2's tokens (via cross-attention) and the robot's current proprioceptive state, and generates **motor actions** — the actual joint commands. It runs at **120 Hz**, because motor control must be fast and smooth to produce fluid movement. It is the "muscle memory" to System 2's "deliberation."

Why split them? Because **reasoning and control have opposite requirements**. Reasoning is expensive, high-level, and slow-changing — you do not re-plan "pick up the cup" every 8 milliseconds. Motor control is cheap, low-level, and fast-changing — you must adjust the joints continuously to move smoothly and react to perturbations. A single model running at one frequency would be forced to compromise: run it fast and the reasoning is too expensive to keep up; run it slow and the motor control is too jerky. Splitting lets each system run at its natural rate — slow reasoning, fast acting — which is exactly what the robot needs.

```python
class GR00TN1:
    """Dual-system VLA: slow VLM planner + fast diffusion-transformer controller."""
    def __init__(self):
        self.system2 = Eagle2VLM()        # 1.34B, runs at 10 Hz
        self.system1 = DiffusionTransformer()  # flow-matching, runs at 120 Hz

    def act(self, image, instruction, robot_state, plan_cache):
        if plan_cache.is_stale():                          # ~every 100 ms
            plan_cache.tokens = self.system2(image, instruction)  # slow re-plan
        # fast loop: generate actions from the cached plan every ~8 ms
        return self.system1(plan_cache.tokens, robot_state)   # 16-action chunk
```

### Second-order optimization: match the component's rate to its job

The reusable principle is that **different sub-tasks have different natural timescales, and forcing them to share one rate cripples the slow ones or starves the fast ones**. Reasoning is slow by nature; motor control is fast by nature. A system that respects this — running each component at its own rate — gets the best of both, where a monolithic single-rate system gets a compromise. This is a deep idea beyond robotics: any system with both high-level deliberation and low-level reaction (a game AI, a trading system, an autonomous vehicle) benefits from separating the slow planning loop from the fast control loop, so neither is bottlenecked by the other's timescale. GR00T N1 is the robotics instance of a general architectural pattern.

## 2. Slow planning, fast acting: the timing

The frequency split is worth seeing concretely, because it is the heart of why the dual system works.

![Hand-authored timing diagram: a System 2 row at 10 Hz shows three wide plan boxes at 0, 100, and 200 milliseconds, while a System 1 row at 120 Hz shows twelve small action boxes packed densely beneath them, with a note explaining that one slow plan feeds many fast action chunks so the robot reacts in real time between re-plans](/imgs/blogs/gr00t-n1-vla-humanoid-robots-3.webp)

The timing relationship is roughly **12-to-1**: System 2 produces a new plan every 100 ms (10 Hz), and in that window System 1 generates about twelve action chunks (120 Hz). System 1 **cross-attends the latest plan** from System 2 and keeps generating motor actions from it, so between System 2's deliberate re-plans, the robot stays continuously reactive. When System 2 finally updates the plan — because the scene changed, the task progressed, or new instruction arrived — System 1 picks up the new plan and continues.

This is exactly how skilled human action works. When you reach for a cup, your conscious mind (System 2) makes the high-level decision once ("grab the cup") and then your motor system (System 1) executes the reach with continuous, sub-conscious adjustments — you do not consciously deliberate each muscle twitch. The deliberate decision is slow and infrequent; the motor execution is fast and continuous. GR00T N1's 10 Hz / 120 Hz split is the engineering realization of this, and it is why the robot can both *understand a complex instruction* (System 2's job, done carefully) and *move fluidly in real time* (System 1's job, done fast).

The practical payoff is **responsiveness without re-thinking**. If the robot had to run the full 1.34B VLM at 120 Hz, it would be far too slow (the VLM runs at 10 Hz on an L40). By caching System 2's plan and running only the lightweight System 1 at 120 Hz, the robot gets real-time motor control while the expensive reasoning happens at a sustainable rate. The action sampling itself is fast — **63.9 ms for a 16-action chunk** on an L40 in bf16 — fast enough to keep the 120 Hz loop fed.

### Second-order optimization: cache the slow, recompute the fast

The pattern here — compute the expensive, slow-changing thing once and reuse it across many cheap, fast-changing computations — is a classic efficiency move that recurs across systems. System 2's plan is expensive and changes slowly, so it is *cached* and reused; System 1's actions are cheap and change fast, so they are *recomputed* continuously. This is the same logic as a [KV cache in LLMs](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) (cache the expensive past, compute the cheap next token) or [Cosmos's causal tokenizer](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer) (cache encoder states, stream new frames). The lesson is to identify the slow-changing expensive computation in your system and cache it, recomputing only the fast-changing cheap part — the ratio of cheap-to-expensive recomputation is where the efficiency lives.

## 3. The action head: flow-matching to motor commands

How does System 1 actually generate motor actions? Not by regressing them directly, but by **flow-matching** — a diffusion-like process that denoises noise into an action chunk.

![Pipeline diagram of the action head: a noise sample in action space goes through the diffusion transformer's flow-matching process over four Euler steps while cross-attending the plan, producing a 16-action chunk over horizon H equals 16, which is executed on the robot at 120 Hz in 63.9 milliseconds](/imgs/blogs/gr00t-n1-vla-humanoid-robots-4.webp)

The action generation works like a tiny, fast diffusion model over actions:

1. **Start from noise** in action space — a random sample the shape of an action chunk.
2. **Denoise via flow-matching** — the DiT iteratively refines the noise into a clean action sequence, conditioned (via cross-attention) on System 2's plan and the robot's state. This uses **K=4 denoising steps** with forward Euler integration — far fewer steps than image diffusion's dozens, because actions are lower-dimensional and speed is critical.
3. **Produce an action chunk** — the output is **H=16** actions (an "action horizon" of 16 future timesteps), not a single action. Predicting a chunk rather than one step at a time gives smoother, more coherent motion and amortizes the inference cost.
4. **Execute** — the chunk drives the robot's motors, generated in **63.9 ms** for all 16 actions.

Why flow-matching (a diffusion variant) instead of just regressing the action directly? Because robot actions are **multimodal** — for a given situation, there are often *several* valid ways to act (you could grasp the cup from the left or the right), and a model that regresses a single action averages these modes, producing a mushy compromise that grasps *neither* side (the classic failure of behavior-cloning with a mean-squared-error loss). A diffusion/flow-matching head models the *full distribution* of valid actions and samples one coherent mode, avoiding the averaging problem. This is why [diffusion policies](/blog/machine-learning/deep-learning/diffusion-models) and [flow-matching](/blog/machine-learning/deep-learning/flow-matching) have become the dominant action representation in modern robot learning — they handle action multimodality where regression fails.

```python
def sample_actions(dit, plan_tokens, robot_state, horizon=16, steps=4):
    """Flow-matching: denoise noise into a 16-action chunk, conditioned on the plan."""
    a = randn((horizon, ACTION_DIM))                  # start from noise in action space
    for k in range(steps):                            # K=4 forward-Euler steps
        v = dit(a, k, context=plan_tokens, state=robot_state)  # cross-attend the plan
        a = a + v * (1.0 / steps)                     # integrate the flow field
    return a                                          # 16 coherent future actions
```

### The flow-matching objective, briefly

It is worth a note on what flow-matching actually optimizes, because it differs subtly from standard diffusion. Where diffusion learns to predict and remove noise, flow-matching learns a *velocity field* — a function that, at each point along a path from noise to data, points in the direction the sample should move. Training minimizes the difference between the model's predicted velocity and the true velocity of a straight (or nearly straight) path connecting a noise sample to a data sample. At inference, you integrate this velocity field forward (the K=4 Euler steps) to flow the noise sample to a clean action. The advantage over standard diffusion for control is that the flow paths can be made nearly straight, so you need *fewer integration steps* to reach the target — which is exactly why GR00T N1 can get away with K=4 steps where image diffusion needs dozens. GR00T N1 uses a Beta(1.5, 1) distribution over the flow timestep to weight the training, emphasizing the parts of the path that matter most for action quality. The lesson is that flow-matching is diffusion's faster cousin for settings where step count is the binding constraint — straighter paths mean fewer steps mean lower latency, which is what real-time control demands.

### Why a chunk of 16, not one action

The action horizon H=16 — predicting sixteen future timesteps at once rather than one — is a design choice with real consequences worth drawing out. Predicting one action at a time has two problems: the motions are *incoherent* (each action is predicted independently of the next, so the trajectory can jitter), and the inference rate must equal the control rate (one forward pass per action, expensive at 120 Hz). Predicting a chunk of 16 fixes both: the sixteen actions are generated together as a coherent short trajectory, so the motion is smooth, and one inference produces sixteen actions, so the model runs at one-sixteenth the control rate. The trade is responsiveness to sudden change — a longer chunk reacts more slowly to surprises, because it commits to sixteen actions before re-planning — so H=16 is a balance between smoothness/efficiency (longer is better) and reactivity (shorter is better). The lesson is that action chunking trades reactivity for smoothness and efficiency, and the chunk length is a tuned balance, not an arbitrary number.

### Second-order optimization: model the distribution, do not average it

The deep lesson is that **for multimodal outputs, you must model the distribution, not regress the mean**. Robot actions, like many real-world outputs, have multiple valid modes, and a model trained to minimize squared error to the "correct" action will average across modes into something invalid. Generative action heads (diffusion, flow-matching) sample a single coherent mode instead. This generalizes far beyond robotics: any time the target is multimodal — multiple valid translations, multiple plausible futures, multiple correct grasps — a generative head beats a regressive one, because regression collapses the modes and generation preserves them. The shift from "regress the action" to "sample from the action distribution" is one of the most important ideas in modern imitation learning, and flow-matching with K=4 steps makes it fast enough for 120 Hz control.

## 4. The data pyramid: stretching scarce robot data

Architecture solves the *how*; the data pyramid solves the *what to learn from*, and it is GR00T N1's answer to the data-scarcity problem that opened this post.

![Hand-authored pyramid diagram: the narrow top layer is real robot data (88 hours of Fourier GR-1 teleoperation plus Open X-Embodiment and AgiBot-Alpha), the wider middle layer is synthetic data (780,000 simulation trajectories from DexMimicGen plus 827 hours of neural trajectories), and the widest base layer is human and web video (Ego4D, EPIC-KITCHENS, web VLM data with no action labels), with quantity decreasing and embodiment-specificity increasing from base to peak](/imgs/blogs/gr00t-n1-vla-humanoid-robots-5.webp)

The pyramid has three layers, and the shape is the point: **quantity decreases and embodiment-specificity increases as you go up**.

- **Base — human and web video (largest, least specific).** A huge quantity of human egocentric video (Ego4D, Ego-Exo4D, EPIC-KITCHENS, HOI4D, and more) plus the web data the VLM was pretrained on. This teaches the model how the physical world works, how humans manipulate objects, how tasks unfold — *broad knowledge*, but with **no robot action labels** (humans are not robots, and the video has no joint commands).
- **Middle — synthetic data (large, moderately specific).** Simulation trajectories — **780,000 of them, equivalent to 6,500 hours, generated in just 11 hours** via DexMimicGen — plus **827 hours of "neural trajectories"** generated from 88 hours of real data by fine-tuned image-to-video models (a ~10× augmentation, using the same video-generation approach as [Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer)). This is cheap, abundant, and reasonably robot-relevant.
- **Peak — real robot data (smallest, most specific).** The precious real demonstrations: **88 hours of teleoperated Fourier GR-1** data, plus the Open X-Embodiment collection (RT-1, Bridge-v2, DROID, and others) and AgiBot-Alpha's 140,000 trajectories from 100 robots. This is exactly the robot's embodiment and tasks, but there is very little of it.

The strategy is to learn **broad knowledge from the abundant base** (how the world and manipulation work, from human video), **transferable skills from the synthetic middle** (how tasks are done, cheaply and at scale), and **precise embodiment-specific control from the scarce peak** (exactly how *this* robot moves). The pyramid stretches the tiny real-robot dataset by surrounding it with orders of magnitude more human and synthetic data, so the model is not bottlenecked by the 88 hours of real demonstrations alone.

### The economics of each pyramid layer

It is worth comparing the *cost per hour* of data across the three layers, because the economics explain the pyramid's shape. Real robot teleoperation data costs roughly *one hour of human operator time per hour of data* — the most expensive, which is why there is least of it (88 hours). Simulation data costs *compute, not human time*, and runs faster than real time in parallel — 6,500 hours generated in 11 wall-clock hours, so vastly cheaper per hour, which is why there is more of it. Human video is *already collected* (it exists on the internet and in academic datasets) — essentially free to acquire, which is why it forms the largest base. The pyramid's shape is a direct consequence of these economics: the cheaper the data per hour, the more of it you use, and you reserve the expensive real data for the part only it can provide (exact embodiment-specific control). The lesson is that data strategy is fundamentally an economic optimization — you want maximum useful signal per dollar, so you lean heavily on the cheapest sources that teach *most* of what you need and spend the expensive data budget only where the cheap sources fall short. The pyramid is the optimal shape under those economics.

### Quality versus relevance across the layers

A subtle tension runs through the pyramid: the cheapest data (human video) is the *least* directly relevant (humans are not robots), while the most relevant data (real robot demos) is the *scarcest*. The synthetic middle bridges this — simulation and neural trajectories are both reasonably cheap *and* reasonably robot-relevant. So the three layers trade off relevance against quantity in a smooth gradient: maximally relevant but scarce at the peak, minimally relevant but abundant at the base, with the synthetic middle splitting the difference. The model learns *general* knowledge (which tolerates low relevance) from the base, *transferable* skills (which need moderate relevance) from the middle, and *specific* control (which needs maximal relevance) from the peak. The lesson is that not all training signal needs to be maximally on-distribution — broad knowledge can come from loosely-relevant abundant data, and you only need tightly-relevant data for the parts of the task that are genuinely embodiment-specific. Matching the relevance of the data to the specificity of what it teaches is what makes the pyramid efficient.

### Second-order optimization: substitute abundant data for scarce data where you can

The principle is that **when your ideal data is scarce, find abundant data that teaches *most* of what you need, and reserve the scarce data for the part only it can teach**. Real robot data is scarce but teaches embodiment-specific control; human video is abundant and teaches general manipulation knowledge; synthetic data is cheap and teaches task structure. By layering them, GR00T N1 learns the bulk of its capability from abundant sources and uses the scarce real data only for the final embodiment-specific tuning. This is the same logic as the [Nemotron-4 synthetic-data flywheel](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — substitute cheap, abundant (synthetic) data for expensive, scarce (human-labeled) data wherever the cheap data suffices. In robotics, where the scarce data is the binding constraint, this substitution is not just an optimization, it is what makes a foundation model possible at all.

## 5. Cross-embodiment: one model, many bodies

A robot foundation model is far more valuable if it works across *many* robots, not just one. GR00T N1 trains across embodiments — single-arm, bimanual, and humanoid — with a single shared model.

![Graph showing cross-embodiment training: single-arm, bimanual, and humanoid robot states all feed embodiment-specific MLPs that project them into a shared space, which feeds a shared GR00T transformer, whose output passes through embodiment-specific MLPs that decode into robot-specific motor commands](/imgs/blogs/gr00t-n1-vla-humanoid-robots-6.webp)

The challenge is that different robots have **different observation and action spaces** — a single-arm robot has 7 joints, a bimanual one has 14, a humanoid has dozens, and their sensors differ. A shared model cannot directly consume these incompatible inputs or produce these incompatible outputs. The solution is **embodiment-specific MLPs** that bridge each robot to a shared space:

- **State/action encoders** — small embodiment-specific MLPs project each robot's variable-dimension observations into a **shared embedding space** that the common transformer understands. The single-arm robot's 7-dim state and the humanoid's high-dim state both map into the same shared representation.
- **A shared transformer** — the bulk of the model (System 2 and System 1) is shared across all embodiments, learning general manipulation knowledge that transfers between robots.
- **Action decoders** — embodiment-specific MLPs convert the shared transformer's output back into **each robot's specific motor commands**.

So the *body-specific* parts (the input/output adapters) are small and per-robot, while the *brain* (the transformer) is large and shared. One model's knowledge transfers across all the robots it trains on — a skill learned on the bimanual robot can inform the humanoid — and adding a new robot means adding only small adapter MLPs, not retraining the whole model.

### Why cross-embodiment transfer works at all

A fair skeptic asks: why should a single-arm robot's data help a humanoid? They have different bodies, different joints, different reachable workspaces. The answer is that *most of what a manipulation policy needs to know is not about the specific joints*. It is about the world and the task: that objects are rigid or deformable, that a grasp must approach from an open side, that pouring requires tilting past a threshold, that a stacked object must be released gently. This knowledge is *embodiment-agnostic* — it holds regardless of which arm executes it — and a shared transformer trained across embodiments learns it from all of them at once. Only the *final motor mapping* (which joint angles realize "approach the grasp") is body-specific, and that is what the thin adapter MLPs handle. So the single-arm robot's data teaches the shared core about grasping-in-general, which transfers to the humanoid; only the last-mile actuation differs. The lesson is that capability in many domains decomposes into a large embodiment-agnostic part (worth sharing) and a small embodiment-specific part (worth isolating), and recognizing that decomposition is what makes cross-embodiment — and cross-anything — transfer possible. The bodies differ; the physics of manipulation does not.

### Second-order optimization: a shared core with thin per-instance adapters

The reusable pattern is a **shared general core wrapped in thin, instance-specific adapters**. The expensive, knowledge-bearing part (the transformer) is shared and trained on all embodiments together, so it benefits from all the data and transfers knowledge across robots; the cheap, instance-specific part (the MLP adapters) handles the per-robot differences. This is the same structure as adapter-based fine-tuning in LLMs (a shared base, small per-task adapters) and the [Parakeet/Canary shared encoder with swappable decoders](/blog/machine-learning/deep-learning/canary-parakeet-fastconformer-asr). The lesson is that when you must serve many instances (robots, tasks, languages) that share most of their structure, put the shared structure in a common core and isolate the differences in thin adapters — you amortize the expensive learning across all instances and pay only a small per-instance cost.

## 6. Making action-less human video trainable

The base of the data pyramid — human video — has a fundamental problem: it has **no action labels**. A video of a person chopping vegetables shows *what* happens but not the *joint commands* that produced it (humans do not emit robot actions). A policy needs action labels to learn from. GR00T N1's trick is to *recover* pseudo-actions from the action-less video.

![Before-and-after comparison: on the left, human video from Ego4D and EPIC-KITCHENS shows what to do but has no action labels and cannot train a policy directly; on the right, latent actions from a VQ-VAE plus an inverse-dynamics model infer pseudo-actions from frame pairs, making the video usable in the training pyramid](/imgs/blogs/gr00t-n1-vla-humanoid-robots-7.webp)

Two complementary techniques recover the missing actions:

- **Latent actions (VQ-VAE).** A VQ-VAE is trained to encode the *change* between consecutive video frames into a discrete latent code, and that code is treated as a **pseudo-action label** — a learned, abstract representation of "what action caused this frame-to-frame transition." The model can then train on human video as if it had action labels, learning the mapping from observation to (latent) action. (This is the "LAPA" — Latent Action Pretraining — approach.)
- **Inverse Dynamics Model (IDM).** An inverse dynamics model is trained to **infer the action that connects two states** — given frame $t$ and frame $t+1$, predict what action would produce that transition. Applied to video (especially the neural-trajectory synthetic data), the IDM labels each frame pair with a pseudo-action, again making action-less video trainable.

Both techniques solve the same problem — *no action labels* — by *inferring* the actions from the observed state transitions. The video shows the *effect* (the frame change); the recovered pseudo-action is the inferred *cause*. With pseudo-actions, the enormous base of human video becomes usable training data, which is what lets the pyramid's base be so large.

### Second-order optimization: infer the missing labels rather than collecting them

The principle is that **when you lack labels but have the inputs and outputs, you can often *infer* the labels rather than collecting them**. Human video has states (frames) but no actions; an inverse dynamics model infers the action from the state transition. This "label inference" turns unlabeled data into labeled data at the cost of training a small auxiliary model. It is a recurring move in data-scarce settings — the [Nemotron-4 reward model](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) infers quality labels, the [Minitron importance estimator](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) infers which weights matter — and here it infers actions from video. The lesson is that scarce labels are sometimes *recoverable* from abundant unlabeled data via a learned inference model, and recovering them can be far cheaper than collecting them, unlocking data you otherwise could not use.

## 7. Results: where the foundation model pays off

Does it work? The clearest evidence is the data-efficiency result on the real robot, which is exactly where a foundation model should help.

![Matrix comparing GR00T N1-2B and a from-scratch diffusion policy on the real Fourier GR-1 robot: with 10% of the task data GR00T N1 achieves 42.6% versus the diffusion policy's 10.2%, with full data 76.8% versus 46.4%, and zero-shot from pretraining 76.6% versus not applicable](/imgs/blogs/gr00t-n1-vla-humanoid-robots-8.webp)

The headline numbers on the real Fourier GR-1 humanoid:

| Setting | GR00T N1-2B | From-scratch Diffusion Policy |
|---|---|---|
| 10% of task data | **42.6%** | 10.2% |
| Full task data | **76.8%** | 46.4% |
| Zero-shot (pretrain only) | **76.6%** (bimanual) / 73.3% (novel objects) | n/a |

And in simulation, across embodiments:

| Benchmark | GR00T N1 | Diffusion Policy | BC-Transformer |
|---|---|---|---|
| RoboCasa Kitchen (24 tasks) | **32.1%** | 25.6% | 26.3% |
| DexMimicGen (9 tasks) | **66.5%** | 56.1% | 53.9% |
| GR-1 Tabletop (24 tasks) | **50.0%** | 32.7% | 16.1% |

The most important number is the **10% data result: 42.6% vs 10.2%**. With little task-specific data — the realistic regime for any new robot task — GR00T N1 vastly outperforms training from scratch, because the foundation model's pretraining (on the data pyramid) gives it a strong prior that the from-scratch policy lacks. This is the entire promise of a foundation model: it makes you *data-efficient* on new tasks, turning a few demonstrations into a competent policy because the model already knows how the world and manipulation work. The full-data gap (76.8% vs 46.4%) is large too, but the low-data gap is the one that matters for practical robot deployment, where collecting data is the bottleneck.

The **neural-trajectory co-training gains** (+4-9% on RoboCasa, +5.8% on the real robot) confirm the synthetic middle layer of the pyramid is pulling real weight — the image-to-video-generated trajectories, despite being synthetic, measurably improve the policy.

### The zero-shot result and what it means

The zero-shot numbers — 76.6% on coordinated bimanual tasks and 73.3% on novel objects, *with no task-specific fine-tuning at all* — deserve their own emphasis, because zero-shot is the strongest test of a foundation model. Zero-shot means the model performs the task using only its pretraining, never having seen a single demonstration of that specific task. Hitting 76.6% zero-shot says the pretraining on the data pyramid taught the model enough general manipulation capability that it can do new tasks out of the box. This is the robotics analogue of a language model answering a question it was never explicitly trained on — the capability emerges from broad pretraining rather than task-specific training. The lesson is that strong zero-shot performance is the clearest signal that a foundation model has learned *general* capability rather than memorized specific tasks, and 76.6% zero-shot on a real humanoid is a strong claim that GR00T N1's pretraining produced genuine generalist manipulation skill, not just a lookup table of trained tasks. Zero-shot is where you see whether the "foundation" is real.

### Why the gap shrinks with more data

A subtle but important pattern in the results: the GR00T-N1-vs-baseline gap is *largest* with little data (42.6% vs 10.2%, a 4× ratio) and *smaller* with full data (76.8% vs 46.4%, a 1.7× ratio). This is exactly what a foundation model should show. With abundant task data, even a from-scratch policy can learn the task reasonably well, so the pretrained prior matters less — the data does the work. With scarce task data, the from-scratch policy has too little to learn from and fails, while the pretrained model fills the gap with its prior, so the pretraining matters enormously. The gap shrinking as data grows is the *signature* of a useful prior: it helps most exactly when you have least data, and its marginal value declines as task data accumulates. The lesson is that you should expect — and look for — this shrinking-gap pattern when evaluating a foundation model, because it confirms the model is providing a *prior* (most valuable when data is scarce) rather than just being a generically better architecture (which would help equally at all data levels). The shape of the gap-versus-data curve tells you what kind of advantage you have.

### Second-order optimization: measure where the model helps most, not just the average

The lesson in reading these results is to look at **where the foundation model helps most, which is the low-data regime**. The full-data numbers (76.8% vs 46.4%) are good, but the 10%-data numbers (42.6% vs 10.2%) are the *story*, because they show the foundation model's prior is worth the equivalent of vastly more task data. A foundation model's value is not that it is better at everything equally — it is that it makes you *data-efficient*, and that value shows up most starkly when task data is scarce. The lesson is to evaluate a foundation model specifically in the low-data regime, because that is where its pretraining prior pays off and where real deployment lives — anyone can collect a lot of data for a demo, but the win is doing well with little.

## 8. Training and deployment

A note on the engineering scale, which connects to the rest of the series. GR00T N1's pretraining took **50,000 H100 GPU hours** on up to **1,024 GPUs**, orchestrated by NVIDIA's OSMO platform — a serious but not absurd budget (far less than [Cosmos's](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer) 10,000 GPUs for three months, because the model is smaller and the data, while diverse, is less raw-pixel-heavy). Critically, **fine-tuning runs on a single A6000 GPU** — once pretrained, adapting GR00T N1 to a new robot or task is cheap, which is the whole point of a foundation model: expensive to pretrain once, cheap to adapt many times. This is the [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation)/[Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer) "train once, derive many" pattern, applied to robot policies: the foundation model is the capital investment, and each robot deployment is a cheap fine-tune.

The model is **open** (GR00T-N1-2B is released), continuing the series-long pattern of open releases that seed an ecosystem — here, the physical-AI and humanoid-robotics community, which lacked a strong open VLA foundation model to build on.

### Second-order optimization: expensive to pretrain, cheap to adapt

The economic structure of GR00T N1 — 50,000 GPU-hours to pretrain, a single GPU to fine-tune — is the defining property of a useful foundation model, and it is worth naming as the series concludes. The pretraining is a one-time capital cost that produces a general prior; every downstream use (a new robot, a new task) is a cheap fine-tune off that prior. This asymmetry — expensive once, cheap many times — is what makes foundation models economically transformative: it amortizes the enormous pretraining cost across an unlimited number of cheap downstream adaptations. The lesson, common to every post in this series, is that the goal of a foundation model is to *move the cost from per-task to once*, so that the marginal cost of a new task collapses — and GR00T N1 achieves it for robot control, turning "train a policy for this robot task" from a from-scratch project into an afternoon's fine-tune.

## 9. Failure modes the design guards against

As across the series, the design is a set of safeguards against specific failure modes.

- **One model too slow for control or too dumb for reasoning.** Symptom: jerky motion or shallow reasoning. Cause: one model at one rate. Safeguard: dual-system, 10 Hz reasoning + 120 Hz control (§1, §2).
- **Averaged, invalid actions.** Symptom: the robot grasps "between" two valid grasps and fails. Cause: regressing multimodal actions. Safeguard: flow-matching action head models the action distribution (§3).
- **Too little robot data to train a policy.** Symptom: from-scratch policies overfit and fail. Cause: data scarcity. Safeguard: the data pyramid stretches scarce robot data with human and synthetic data (§4).
- **Action-less human video unusable.** Symptom: the abundant base data cannot train a policy. Cause: no action labels. Safeguard: latent actions and inverse dynamics recover pseudo-actions (§6).
- **A separate model per robot.** Symptom: every new robot needs a from-scratch model. Cause: incompatible observation/action spaces. Safeguard: cross-embodiment training with adapter MLPs (§5).
- **Expensive adaptation to new tasks.** Symptom: each task needs huge data and compute. Cause: no transferable prior. Safeguard: pretrain once on the pyramid, fine-tune cheaply (§7, §8).

The meta-lesson, as throughout: each design choice is a "don't" — don't run one rate, don't regress multimodal actions, don't rely on scarce data alone, don't waste action-less video, don't train per-robot — and the model is the disciplined accumulation of those don'ts.

## 10. Case studies from GR00T N1

### 1. The fast-and-slow split as the core idea

The dual-system architecture is the keystone, and it is a case study in importing a cognitive-science framework into model design. Kahneman's System 1 (fast, automatic) and System 2 (slow, deliberate) describe human cognition; GR00T N1 realizes them as a fast diffusion-transformer controller and a slow vision-language planner. The framework is not decoration — it captures a real engineering truth, that reasoning and control have opposite timescale requirements, and the architecture follows from taking that truth seriously. The lesson is that good cognitive-science metaphors can be genuine design guides when they reflect real computational structure, and the fast/slow split reflects the real fact that deliberation and reaction operate at different rates.

### 2. The 12-to-1 frequency ratio

The specific 10 Hz / 120 Hz frequencies are a case study in matching rate to function. 10 Hz is fast enough for plan updates (the scene does not change meaningfully faster than that for most manipulation) and slow enough that the 1.34B VLM can keep up. 120 Hz is the rate needed for smooth motor control. The 12-to-1 ratio means System 1 runs many times per System 2 update, staying reactive between re-plans. The lesson is that the *ratio* between the slow and fast loops matters — too small and the fast loop re-plans needlessly (wasting the expensive System 2); too large and the plan goes stale before it updates. Finding the right ratio (here ~12:1) is part of the design, and it follows from the natural timescales of perception (slow) and control (fast).

### 3. Flow-matching over regression for actions

The choice of a flow-matching action head over direct regression is a case study in the multimodality problem. Behavior cloning with an MSE loss averages multiple valid actions into an invalid mean — the robot that grasps neither side of the cup. Flow-matching samples one coherent mode from the action distribution. This is now the standard in robot learning precisely because the multimodality problem is pervasive. The lesson is that the *loss and output representation* matter as much as the architecture: an MSE-regressed action head and a flow-matching action head can sit on the *same* backbone and produce wildly different behavior, because one collapses modes and the other preserves them. Choosing the generative head is what makes the policy work.

### 4. K=4 denoising steps for real-time control

The use of only K=4 denoising steps (versus dozens for image diffusion) is a case study in adapting a technique to a latency constraint. Image diffusion can afford 50 steps because quality matters and latency is forgiving; robot control needs actions in milliseconds, so GR00T N1 uses just 4 Euler steps, trading a little action quality for the speed needed at 120 Hz. The action space is also lower-dimensional than image space, so fewer steps suffice. The lesson is that the same generative technique (diffusion/flow-matching) is tuned very differently for different latency budgets — 50 steps for offline image generation, 4 steps for real-time control — and knowing how few steps you can get away with is the key to using diffusion in a real-time loop.

### 5. 780k simulation trajectories in 11 hours

Generating 780,000 simulation trajectories (6,500 hours equivalent) in just 11 hours via DexMimicGen is a case study in how simulation breaks the data bottleneck. Real robot data is collected in real time (88 hours of teleoperation is 88 hours of human effort); simulation generates trajectories far faster than real time and in parallel, producing 6,500 hours of data in 11 wall-clock hours. The lesson is that simulation's superpower is *time compression and parallelism* — it manufactures the data that real collection cannot, which is why the synthetic middle layer of the pyramid can be so large. The catch is the sim-to-real gap (simulated data is not perfectly realistic), which the real-robot peak layer and the neural trajectories help bridge.

### 6. Neural trajectories: 10× augmentation via video generation

The 827 hours of "neural trajectories" — generated from 88 hours of real data via fine-tuned image-to-video models, a ~10× augmentation — is a case study in using a *generative world model* to augment robot data, directly connecting to [Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer). Take real robot demonstrations, and use a video-generation model to produce variations (different backgrounds, lighting, object positions), 10×-ing the dataset. This is the world-model-as-data-engine idea made concrete: a video generator produces realistic training data for a policy. The lesson is that generative models and policies form a virtuous loop — the world model generates data that trains the policy — and the +5.8% real-robot gain from co-training on neural trajectories proves the synthetic augmentation transfers to reality. The two NVIDIA reports (Cosmos and GR00T) are designed to compose this way.

### 7. Pseudo-actions from inverse dynamics

The inverse-dynamics-model recovery of pseudo-actions is a case study in unlocking the largest, cheapest data source (human video) for policy learning. Human video is abundant but action-less; an IDM infers the action from each state transition, making it trainable. The lesson is that the inverse problem (given the effect, infer the cause) is often solvable with a learned model, and solving it converts a vast unusable dataset into a usable one. This is the single technique that lets the pyramid's base be human-scale rather than robot-scale, which is the difference between a foundation model and a narrow policy — without pseudo-action recovery, the model could only learn from the scarce real-robot data, and there would be no "foundation" to speak of.

### 8. Cross-embodiment knowledge transfer

Training one model across single-arm, bimanual, and humanoid robots is a case study in knowledge transfer across bodies. A skill or representation learned on one embodiment informs the others, because the shared transformer sees all of them and learns embodiment-agnostic manipulation knowledge. The thin adapter MLPs handle the body-specific input/output differences. The lesson is that much of manipulation knowledge is *embodiment-agnostic* — how objects behave, how to approach a grasp, what a task requires — and only the final motor mapping is body-specific, so a shared model can transfer the agnostic knowledge across bodies while adapting the specific part cheaply. This is why a robot foundation model is more than a per-robot policy: it pools learning across all the bodies it sees.

### 9. The 42.6%-vs-10.2% data-efficiency gap

The low-data result — 42.6% vs a from-scratch policy's 10.2% on the real robot with 10% of the data — is the case study that justifies the whole foundation-model approach. A from-scratch diffusion policy with little data barely works (10.2%); GR00T N1, with the same little data but a strong pretrained prior, works four times better. This 4× data-efficiency multiplier is the entire value proposition: the pretraining on the data pyramid is worth the equivalent of vastly more task-specific data. The lesson is that the right way to measure a robot foundation model is *data efficiency on new tasks*, and a 4× multiplier in the low-data regime is transformative for real deployment, where every demonstration is expensive to collect.

### 10. Fine-tuning on a single GPU

The fact that GR00T N1 fine-tunes on a single A6000 is a case study in the practical accessibility a foundation model provides. Pretraining took 50,000 H100-hours — out of reach for most teams — but the *result* is a model any robotics lab can fine-tune on one GPU. This democratizes capability: a small team cannot pretrain a robot foundation model, but they can take the open GR00T N1 and adapt it to their robot cheaply. The lesson is that the open release plus cheap fine-tuning is what spreads the capability — the expensive pretraining is done once by a well-resourced lab, and the cheap adaptation is done many times by everyone else, which is exactly how foundation models have democratized language and vision and now aim to democratize robotics.

### 11. The Eagle-2 VLM as System 2

Building System 2 on the pretrained Eagle-2 VLM is a case study in standing on a strong perception-and-language foundation. Rather than training a vision-language reasoner from scratch, GR00T N1 reuses a VLM already pretrained on internet-scale image-text data, inheriting its visual understanding and language grounding. The robot-specific training then teaches it to *plan actions*, building on the VLM's existing world knowledge. The lesson is that the perception-and-reasoning part of a robot policy should reuse a pretrained VLM, because visual and linguistic understanding transfer from internet data to robotics — a robot that needs to "pick up the red cup" benefits from a VLM that already knows what red and cups are. The action part is what is robot-specific and must be learned; the perception part is largely inherited.

### 12. Action chunking for smooth motion

Predicting a 16-action chunk (horizon H=16) rather than one action at a time is a case study in temporal grouping for smooth control. Single-step action prediction produces jittery motion (each step is predicted independently and they do not cohere) and is slow (one inference per action); predicting a chunk of 16 future actions at once gives smoother, more coherent motion and amortizes inference. The lesson is that *chunked* prediction beats step-by-step for control — it lets the model plan a short coherent motion sequence rather than reacting one twitch at a time, and it cuts the inference frequency (one inference per 16 actions, not per action). Action chunking is a now-standard trick in modern robot policies, and GR00T N1's H=16 is a well-tuned instance.

### 13. The sim-to-real bridge

The combination of simulation data (middle pyramid), neural trajectories (Cosmos-style augmentation), and real robot data (peak) is a case study in bridging the sim-to-real gap from multiple directions. Pure simulation data suffers the sim-to-real gap (simulated physics and rendering are not perfectly real); neural trajectories add photorealistic variation from real seeds; real robot data anchors the model to reality. The three together bridge the gap better than any one alone. The lesson is that closing sim-to-real is not a single technique but a *layering* — cheap-but-unrealistic simulation for scale, realistic-but-derived neural augmentation for the middle, and scarce-but-real data for the anchor — and the layered combination is what makes the synthetic data actually transfer to the real robot.

### 14. The world's first open humanoid foundation model

GR00T N1's positioning as "the world's first open foundation model for generalist humanoid robots" is a strategic case study. By releasing it openly, NVIDIA seeds the humanoid-robotics ecosystem — which, like physical AI broadly, lacked a strong open foundation to build on — and, as with every open release in this series, the ecosystem runs on NVIDIA hardware and tooling (Isaac, OSMO, the simulators). The lesson, identical to the language-model open releases, is that a strong open foundation model is a platform play that catalyzes an entire field while anchoring it to your ecosystem. For humanoid robotics specifically, the bottleneck was the lack of a shared foundation, and GR00T N1 aims to be it.

### 15. Latent actions versus inverse dynamics

The use of *two* pseudo-action recovery methods — latent actions (VQ-VAE) and inverse dynamics (IDM) — for different data sources is a case study in matching the technique to the data. Latent actions (an abstract learned code for frame-to-frame change) suit the diverse, uncontrolled human video where you cannot define a concrete action space; inverse dynamics (inferring a concrete action from a state pair) suits the neural trajectories where the action space is better defined. The lesson is that "recover the missing labels" has multiple instantiations, and you pick the one that fits the data's structure — an abstract latent code for messy human video, a concrete inferred action for cleaner synthetic data. Using both lets the model exploit both data sources fully.

### 16. The whole series, ending in a body

A reflective final case study: GR00T N1 is where the series' techniques converge on an embodied agent, and seeing the convergence is the point. It uses a *VLM* (perception and language, like every multimodal model), a *diffusion/flow-matching* generative head (like [Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer)), *synthetic data* (like [Nemotron-4](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) and Cosmos), *cross-embodiment adapters* (the shared-core-thin-adapter pattern), *train-once-fine-tune-cheaply* economics (like [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation)), and *open release with an ecosystem* (like all of them). The lesson is that the foundation-model playbook this series has traced — from a language model to a humanoid robot — is a *coherent, transferable methodology*, and GR00T N1 is its application to the hardest target: an agent that must perceive, reason, and act in the physical world. The body is where all the pieces come together.

### 17. Co-training, not staged pretraining

A methodological case study worth drawing out: GR00T N1 *co-trains* on all three pyramid layers together (human video, synthetic, real robot) rather than pretraining on the base and then fine-tuning on the peak in strict stages. Co-training mixes the data sources in each training batch, so the model learns the broad knowledge and the specific control *jointly*, and the gradients from the abundant base regularize the learning from the scarce peak. The measured neural-trajectory co-training gains (+4-9%) come from exactly this mixing — the synthetic data, included in the training mix, lifts performance on the real tasks. The lesson is that for heterogeneous data sources of different quality and quantity, *co-training* (mixing them in each batch) often beats *staged training* (one source then another), because the abundant data continuously regularizes the learning from the scarce data rather than being forgotten once the staged fine-tune begins. The mix ratios become a tuning knob, but the joint training is what lets the pyramid's layers reinforce each other.

### 18. The proprioceptive state input

A detail easy to overlook: System 1 conditions not only on System 2's plan and the camera image but on the robot's **proprioceptive state** — its current joint positions and velocities. This matters because motor control is fundamentally about the *current* configuration of the body: the action to take depends on where the joints already are, not just on what the camera sees. A policy that ignored proprioception would have to infer the body's state from vision alone, which is unreliable (the camera may not see all the joints). Feeding proprioception directly gives the controller precise knowledge of the body it is controlling. The lesson is that embodied control needs *body-state* input, not just external perception — the robot must know where its own limbs are to command them — and proprioception is a cheap, reliable signal that vision cannot fully replace. It is a small input with an outsized effect on control precision.

### 19. Why the VLM features come from layer 12

A precise architectural case study: System 1 cross-attends to vision-language features from *layer 12* of the Eagle-2 VLM, not its final layer. Why a middle layer? Because the final layers of a VLM are specialized for its *original* objective (generating text), while middle layers carry richer, more general visual-semantic representations that are more useful for *conditioning a controller*. Extracting features from an intermediate layer is a common trick when repurposing a pretrained model for a new downstream use — the middle layers are where the general-purpose representation lives, before the model commits to its original output task. The lesson is that when you tap a pretrained network for features to feed a new head, the *last* layer is often not the best — the most transferable representation is usually in the middle, before task-specific specialization, and choosing the right layer to tap (here, 12) is a tuned decision that materially affects the downstream policy.

### 20. Real-time onboard the constraint that shapes everything

A unifying case study: the requirement that the policy run *in real time on the robot* is the constraint that shapes nearly every design choice, and recognizing it ties the whole architecture together. Real-time control at 120 Hz is why System 1 is a small fast model separate from the heavy VLM (§1); why flow-matching uses only K=4 steps (§3); why actions are chunked to amortize inference (§12 above); why the VLM runs at 10 Hz with cached plans (§2). Strip away the real-time constraint and a simpler single-model design might suffice; *with* it, the dual-system, few-step, chunked, cached architecture becomes necessary. The lesson is that a single hard constraint — here, real-time onboard control — can dictate an entire architecture, and reading a system backward from its binding constraint reveals why each piece is shaped as it is. GR00T N1 is, in large part, the answer to "how do you run a 2.2B reasoning-and-control model fast enough to drive a robot in real time," and every design choice serves that question.

### 21. The diffusion-policy baseline as the right comparison

Choosing a from-scratch diffusion policy as the headline baseline is a case study in fair comparison. The diffusion policy is itself a strong modern method (it shares the flow-matching/diffusion action head that makes GR00T N1's control good), so the comparison isolates the *one variable that matters*: the foundation-model pretraining. Both use a good action head; only GR00T N1 has the data-pyramid pretraining. So the 42.6%-vs-10.2% gap is attributable specifically to the pretraining, not to a better action representation. The lesson is that a meaningful ablation compares against a baseline that shares everything except the thing you are testing — here, a from-scratch policy with the *same* action head, so the gap measures the value of pretraining alone. A weaker baseline (say, an MSE-regression policy) would have conflated the action-head advantage with the pretraining advantage and overstated the latter. Good comparisons isolate one variable.

### 22. The humanoid as the hardest embodiment

Targeting humanoids specifically — not just arms — is a case study in choosing an ambitious target that forces generality. A humanoid has many more degrees of freedom than a single arm, bimanual coordination, whole-body balance considerations, and a human-like form factor meant to operate in human environments. Building a foundation model that handles humanoids forces the architecture to be general enough for the hardest case, and a model that works on a humanoid also works on the simpler arms (which are a subset of the capability). The strategic bet is that humanoids are where embodied AI is heading — general-purpose robots in human spaces — so a foundation model should target them. The lesson is that choosing the hardest representative target (humanoids) forces a generality that subsumes the easier targets, and aiming at the frontier embodiment future-proofs the model against the simpler ones it also covers.

### 23. Imitation learning, not reinforcement learning

A notable choice: GR00T N1 is trained primarily by *imitation* (learning from demonstrations, the human video, the teleoperation, the synthetic trajectories) rather than *reinforcement learning* (learning from trial-and-error reward in the environment). This is deliberate — RL on real robots is slow and dangerous (the robot must physically try and fail many times, risking damage), while imitation learns from existing demonstrations safely and efficiently. The data pyramid is fundamentally an *imitation* dataset (demonstrations of tasks being done), and the model learns to copy them. The lesson is that for real-world robotics, imitation from demonstrations is often the more practical path than RL, because trial-and-error in the physical world is expensive and risky, where demonstrations (especially the abundant human-video kind) are comparatively cheap and safe. The frontier increasingly combines the two (imitation to bootstrap, RL to refine), but GR00T N1's foundation is imitation, and the data pyramid is what makes imitation learning scale.

### 24. The 2.2B size as a deployment choice

GR00T N1-2B's modest 2.2B-parameter size (tiny next to the 340B and 405B models earlier in this series) is a case study in sizing for the deployment, not the benchmark. A robot policy must run *onboard*, in real time, on whatever compute the robot carries — not in a data center. So the model is sized to be runnable at 120 Hz on robot-grade hardware (the action sampling is 63.9 ms on an L40), which caps how large it can be. A 70B robot policy would be more capable on paper but unservable on a real robot at real-time rates. The lesson, echoing [Nemotron-4's single-DGX constraint](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment), is that the deployment target dictates the model size — for a robot, "what fits onboard and runs in real time" is the binding constraint, and the architecture (small fast System 1, cached slow System 2) is built to make a 2.2B model feel responsive within it. Bigger is not better when the model has to live on a robot.

### 25. OSMO and the orchestration of training

The use of NVIDIA's OSMO platform to orchestrate training across up to 1,024 GPUs is a small case study in the infrastructure that frontier training requires. Training on the heterogeneous data pyramid — human video, simulation, neural trajectories, real robot data, each with different formats and pipelines — across a thousand GPUs is a serious distributed-systems problem, and OSMO is the orchestration layer that makes it tractable. This is the unglamorous foundation under every headline result in this series: none of these models trains without an enormous, well-engineered infrastructure for data loading, distributed training, and checkpoint management. The lesson is that frontier model training is as much a *systems* achievement as a modeling one, and the orchestration infrastructure — invisible in the results but essential to producing them — is part of why a well-resourced lab can train these models and a small team cannot. The model gets the paper; the infrastructure gets the model trained.

## When to reach for a VLA / dual-system approach — and when not to

**Reach for it when:**

- **You are building a generalist robot policy.** The dual-system VLA is designed for robots that must follow language instructions and act across varied tasks.
- **You have scarce robot data.** The data pyramid and pseudo-action recovery are precisely for the data-scarce regime where from-scratch policies fail.
- **You need both reasoning and fast control.** The fast/slow split gives you instruction-following *and* fluid real-time motion.
- **You serve multiple robot embodiments.** Cross-embodiment training shares knowledge across bodies and adapts cheaply to new ones.
- **You can fine-tune from a foundation.** Starting from open GR00T N1 and fine-tuning is vastly cheaper and more data-efficient than training from scratch.

**Skip it (or look elsewhere) when:**

- **Your task is narrow and you have ample data.** For a single repetitive task with lots of demonstrations, a specialized policy may be simpler and sufficient.
- **You need only low-level control, no reasoning.** If there is no language instruction or high-level planning, the System 2 VLM is overhead.
- **You lack the deployment hardware.** Running a 2.2B dual-system model at 120 Hz needs real onboard compute; tiny embedded robots may not have it.
- **Your robot is wildly different from the training embodiments.** Cross-embodiment transfer has limits; a radically novel morphology may need substantial new data.

The one-sentence version:

> Give a robot a slow vision-language brain to reason about what to do and a fast diffusion-transformer brain to generate how to do it, train the pair on a pyramid that stretches scarce robot data with abundant human video and synthetic trajectories, and you get a generalist policy that is data-efficient enough to adapt to a new robot or task with a handful of demonstrations.

## Conclusion: one playbook, from a language model to a humanoid

This is the last post in the series, so it is worth stepping back over all seven. Read together, NVIDIA's reports trace a single engineering culture applying one playbook across radically different modalities. [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) showed how to derive a family of models from one by pruning and distillation. [Nemotron-4 340B](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) showed how to align a model on synthetic data judged by a reward model. [Llama-Nemotron](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models) showed how to search an architecture for the hardware and toggle reasoning on demand. [Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) showed how to swap most attention for state-space layers to serve long context cheaply. [Canary and Parakeet](/blog/machine-learning/deep-learning/canary-parakeet-fastconformer-asr) showed how to cut an ASR encoder's cost by slashing the frame rate. [Cosmos](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer) showed how to compress video and learn to predict the physical world. And GR00T N1 shows how to give a robot a body it can use.

The through-line is unmistakable. Every report **finds the dominant cost and attacks it structurally** — tokens, human labels, KV cache, frame rate, raw pixels, scarce robot data. Every one **trains an expensive foundation once and derives cheap specializations many times**. Every one **substitutes abundant data (synthetic, distilled, inferred) for scarce data** wherever possible. Every one **uses low precision and aggressive efficiency** to make the scale affordable. And every one **ships openly with guardrails** to seed an ecosystem. The modalities differ — text, speech, video, action — and the specific techniques differ, but the *playbook* is the same, and watching it transfer from a 15B language model to a humanoid robot is the real lesson of the series. Foundation models are not a language-model trick; they are a general methodology for turning a hard learning problem into an expensive-once, cheap-many-times asset, and NVIDIA's reports are a master class in applying that methodology wherever the data and the compute allow. GR00T N1, giving a robot a fast-and-slow brain trained on a pyramid of data, is where that methodology reaches all the way into the physical world.

If there is one idea to carry away from the whole series, it is this: the hard part of building a capable model is almost never the model — it is finding the dominant cost and the binding constraint, and designing around them. For a 15B language model it was retraining cost (answer: prune and distill). For a 340B model it was alignment data (answer: synthesize it and judge it with a reward model). For a reasoning model it was inference latency (answer: search the architecture and toggle the reasoning). For long context it was the KV cache (answer: swap attention for state-space layers). For speech it was the frame rate (answer: downsample hard). For video it was the sheer size of pixels (answer: tokenize and compress). And for a robot it was data scarcity and real-time control (answer: a data pyramid and a two-system brain). The models are different; the discipline is identical — name the constraint, attack it structurally, and amortize the expensive work across cheap downstream uses. That discipline, more than any single architecture, is what these seven reports teach, and it is what will still be useful long after the specific models are superseded. The particular numbers — 8% attention, 8× downsampling, K=4 denoising steps, 10 Hz versus 120 Hz — will date quickly; the *way of thinking* that produced them will not, because it is the way of thinking that turns a research idea into a system that ships, on whatever modality and whatever hardware the next problem brings. Whether you are compressing a model, aligning one, searching its architecture, swapping its primitives, cutting its frame rate, tokenizing its pixels, or teaching it to move a body, the question is always the same — what is the dominant cost, what is the binding constraint, and how do you design around them — and learning to ask it well is worth more than any one of the answers in these seven posts. Thanks for reading the series.

## Further reading

- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734) — the full report, with the architecture, data pyramid, and benchmark details.
- [Cosmos World Foundation Models](/blog/machine-learning/computer-vision/cosmos-world-foundation-models-tokenizer) — the world model that generates GR00T's neural-trajectory augmentation data; the two compose.
- [Grounding language in robot actions](/blog/machine-learning/ai-agent/llm-to-control-grounding-language-in-robot-actions) — the broader picture of connecting language models to robot control.
- [Flow matching](/blog/machine-learning/deep-learning/flow-matching) and [diffusion models](/blog/machine-learning/deep-learning/diffusion-models) — the generative foundations behind the action head.
- [Nemotron-4 340B synthetic data](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — the synthetic-data philosophy the data pyramid shares.
- [Minitron: pruning and distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) — the train-once-derive-many economics, where this series began.

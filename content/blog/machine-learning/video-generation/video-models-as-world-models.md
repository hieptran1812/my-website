---
title: "Video Models as World Models: Playable, Action-Conditioned Generation"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Turn a passive video generator into an interactive world model: formalize action-conditioned next-frame prediction, walk GameNGen and the Genie line, code a streaming rollout loop, and connect it to model-based RL and robotics."
tags:
  [
    "video-generation",
    "diffusion-models",
    "world-models",
    "action-conditioned",
    "genie",
    "gamengen",
    "model-based-rl",
    "video-diffusion",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/video-models-as-world-models-1.png"
---

The first time a video model genuinely surprised me, it was not because of how good a clip looked. It was because I could *play* it. I held the W key and a first-person view of a corridor crept forward; I tapped the mouse and the camera swung; I let go and the world held still. There was no game engine underneath. There were no polygons, no collision meshes, no hand-written physics. There was a neural network predicting, frame by frame, what I should see next given what I had just seen and which key I was pressing. It was, in the most literal sense, a model that had learned to *be* the game rather than to render one. And it was running at a frame rate I could actually steer.

That experience is the subject of this post, and it marks a genuine fork in the road for video generation. Everything in this series up to now has treated a video model as a function from a prompt to a clip — you ask for "a golden retriever running through a park," you wait, and a fixed five seconds comes back. That is *passive* generation. It is open-loop. The model commits to an entire trajectory before you see the first frame, and nothing you do after pressing render changes the outcome. A **world model** breaks that loop. It does not predict a whole clip; it predicts the *next observation given the current state and an action*, and then it takes its own prediction as the new state and asks for the next action. The output of one step becomes the input of the next, and a human or an agent supplies the action in between. The clip is no longer rendered. It is *driven*.

The formal object underneath that shift is one conditional distribution, and the whole post hangs on it. A passive video model learns `$p(o_{1:T})$` — the joint distribution over a sequence of observations (frames). A world model learns the action-conditioned next-step distribution

$$
p(o_{t+1} \mid o_{\le t},\, a_t),
$$

the probability of the next frame given the history of observations and the action `$a_t$` you just took. That single extra symbol — `$a_t$`, the action — is the entire difference between a video you watch and a world you inhabit. Conditioning on it is what makes the model controllable. Predicting one step at a time is what makes it interactive. And carrying a usable latent state across steps is what gives it the memory it needs to keep the world consistent while you turn around. Figure 1 is the shape of the whole idea: the current observation and the chosen action go into a dynamics model, which emits the next observation that immediately becomes the current observation for the following step.

![Diagram of an action-conditioned rollout where the current observation and a chosen action feed a dynamics model that predicts the next observation which feeds back in as the new current observation while a latent state carries memory](/imgs/blogs/video-models-as-world-models-1.png)

This is part of the [Video Generation, From First Principles to the Frontier](/blog/machine-learning/video-generation/why-video-generation-is-hard) series, and it is where the series turns from generation toward *simulation*. It sits directly downstream of [Sora and the world-simulator thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis), which asked whether scaling passive video generation gives you a world simulator for free; this post takes the deliberate path instead, where we *build in* the action-conditioning and interactivity that Sora only approximates. It also leans hard on [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout), because an interactive model is autoregressive by construction and inherits every drift problem from that post, and on [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation), because interactivity has a latency budget measured in tens of milliseconds. By the end you will be able to write down the world-model objective and contrast it precisely with unconditional video diffusion, explain how Genie learns a controllable action interface from *unlabeled* video, implement an action-conditioned next-frame diffusion step and a streaming rollout loop in PyTorch, read an honest comparison table of GameNGen, the Genie line, and Oasis, and reason about why action-conditioned generation can support *causal* learning that passive video cannot — and where it still falls apart.

## 1. What a world model actually is

The phrase "world model" is overloaded, so let me pin it down before we build anything. In the reinforcement-learning tradition, a world model is a learned approximation of the *environment's transition dynamics*: given the current state and an action, it predicts the next state (and usually the reward). It is the thing that lets an agent ask "what would happen if I did this?" without actually doing it. That is the sense we care about. A world model in our setting is a learned simulator: feed it where you are and what you do, and it tells you where you end up.

The contrast with a passive video model is sharpest when you write the two objectives side by side. An unconditional video diffusion model — the kind from the [from-image-diffusion-to-video-diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) post — trains a denoiser `$\epsilon_\theta$` to reverse a noising process applied to a *whole clip* of latent frames, with the standard noise-prediction loss

$$
\mathcal{L}_\text{passive} = \mathbb{E}_{x_0,\, \epsilon,\, t}\big[\, \lVert \epsilon - \epsilon_\theta(x_t, t, c) \rVert^2 \,\big],
$$

where `$x_0$` is the clean latent video, `$x_t$` is its noised version at diffusion step `$t$`, and `$c$` is a *static* condition like a text prompt that applies to the entire clip at once. Notice what is missing: there is no per-frame action, no notion of time-indexed control, and no requirement that the model produce frame `$t+1$` before it has seen frame `$t+2$`. The whole clip is denoised jointly. (If the diffusion details here feel hazy, the [math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm) and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) posts in the image series carry the full derivation, and I will not re-derive it.)

A world model changes three things at once, and all three matter:

1. **Action-conditioning.** The condition is no longer a single static `$c$` for the whole clip; it is a *time-indexed* action `$a_t$` that changes every frame. The model must learn how each specific action transforms the world, so the same observation must map to different next observations depending on what you do.
2. **Interactivity / streaming.** The model produces `$o_{t+1}$` and *stops*, waits for `$a_{t+1}$`, then produces `$o_{t+2}$`. It cannot denoise the future jointly because the future does not exist yet — it depends on actions you have not taken. This forces autoregressive, frame-by-frame generation with a hard latency budget.
3. **A usable latent state.** To keep the world consistent — so the chest you walked past is still there when you turn back — the model needs a state that *remembers*. In RL terms this is the belief state; in our terms it is the rolling context of past frames (and any compressed memory) that the model attends to when predicting the next one.

Put those three together and you get the world-model objective in its autoregressive form. Factor the action-conditioned joint distribution of a trajectory left to right:

$$
p(o_{1:T} \mid a_{1:T}) = p(o_1) \prod_{t=1}^{T-1} p(o_{t+1} \mid o_{\le t},\, a_{\le t}).
$$

Training maximizes the log-likelihood of real trajectories under this factorization, which in a diffusion model becomes a noise-prediction loss on the *next frame conditioned on past frames and the action*:

$$
\mathcal{L}_\text{world} = \mathbb{E}\big[\, \lVert \epsilon - \epsilon_\theta(o^{(t+1)}_{\tau},\, \tau,\, o_{\le t},\, a_t) \rVert^2 \,\big],
$$

where `$o^{(t+1)}_{\tau}$` is the noised latent of frame `$t+1$` at diffusion step `$\tau$`, and the conditioning now includes both the clean past frames `$o_{\le t}$` and the action `$a_t$`. This is the equation the whole post implements. Compare it line for line with `$\mathcal{L}_\text{passive}$` and the difference is exactly the three changes above: a time-indexed action in the conditioning, a next-frame target rather than a whole-clip target, and a dependence on the past frames as state.

### Why this is not just "I2V with extra steps"

A fair objection: image-to-video already conditions the future on a past frame, so is a world model just I2V rolled out repeatedly? No, and the difference is the action. I2V conditions on a frame and lets the model pick *some* plausible continuation; the model is free to invent the motion. A world model conditions on a frame *and a specific action* and must produce the continuation *that action implies* — and a *different* action from the same frame must produce a *different* continuation. That counterfactual structure ("if I press left I go left, if I press right I go right, from the very same starting frame") is what makes it a model of dynamics rather than a model of plausible video. We will see in Section 8 that this is precisely what lets a world model support causal learning, which passive rollout cannot.

### The POMDP framing and why state is a belief, not a frame

There is a more rigorous way to say all of this, and it is worth doing once because it explains a subtlety that trips people up: *the state of a world model is not a frame.* The environment a world model approximates is a partially observed Markov decision process (POMDP). The true environment has a hidden state `$s_t$` — the full configuration of the world, most of which is off-screen — that evolves under a transition law `$s_{t+1} \sim T(s_{t+1} \mid s_t, a_t)$`, and you only ever see a partial observation `$o_t \sim O(o_t \mid s_t)$` (the current frame, which shows you a slice of the world). The Markov property holds on the hidden state `$s_t$`, *not* on the observation `$o_t$` — the current frame alone does not determine the future, because the future depends on things you cannot see in this one frame (what is behind you, how fast you were moving, what you are carrying).

This is exactly why a world model must condition on a *window* of past observations and carry a latent state, rather than just the single last frame. The history `$o_{\le t}$` is a proxy for the unobservable `$s_t$`: by accumulating observations you build a *belief* about the hidden state, and the belief is what makes the next-frame prediction well-posed. Formally the model approximates the belief-state transition

$$
b_{t+1} = f_\theta(b_t,\, o_t,\, a_t), \qquad o_{t+1} \sim p_\theta(o_{t+1} \mid b_{t+1}),
$$

where `$b_t$` is the model's internal belief (its latent state) updated each step from the previous belief, the new observation, and the action. In our architecture `$b_t$` is realized as the rolling window of context latents the denoiser attends to. When people say a world model "needs memory," this is the precise statement: it needs a belief state rich enough to stand in for the hidden environment state, and a finite attention window is a *bounded, lossy* belief — which is exactly why long-horizon consistency (Section 9) is so hard. You cannot hold an unbounded world in a bounded belief. Everything downstream — the persistence problem, the memory research, the explicit-state direction — is an attempt to make the belief state hold more of the hidden state for longer.

## 2. Passive video vs the interactive loop

Let me make the contrast concrete with the running example I will carry through the post: a first-person navigation world, the kind Genie and GameNGen produce. In the passive framing, you would prompt "walk down a stone corridor toward a door," wait for a render, and get a fixed clip of a walk down a corridor. If the camera path is wrong, your only recourse is to re-prompt and re-render. You are a spectator. In the interactive framing, you *are* the camera. You press forward and the corridor approaches; you stop and it stops; you turn and a side passage you did not know was there comes into view. The crucial property is that the trajectory is *not predetermined* — it is co-authored, one action at a time, by you and the model. Figure 2 lays the two regimes side by side: a passive generator that takes a prompt and emits a whole clip with no mid-stream control, against a world model that emits one frame at a time conditioned on a live action with a low-latency budget.

![Before and after comparison contrasting passive video generation that emits a whole clip from a prompt with no mid-stream control against an action-conditioned world model that emits frames one at a time conditioned on each action at low latency](/imgs/blogs/video-models-as-world-models-2.png)

The engineering consequences of crossing from the left column to the right are larger than they look, and they are the reason world models are a distinct research program rather than a feature you flip on:

- **Latency becomes a hard constraint, not a nuisance.** A passive model that takes 90 seconds to render a 5-second clip is annoying. An interactive model that takes 90 seconds *per frame* is unusable — there is a human in the loop expecting the world to respond. Interactivity demands generation at interactive frame rates, which for a playable feel means roughly 10–30 frames per second, i.e. a budget of 30–100 milliseconds *per frame*, end to end. That budget reshapes every architectural choice and is the entire reason GameNGen and the efficient-video line obsess over step distillation and caching.
- **The future is undefined until the action arrives.** A passive model can denoise all `$T$` frames jointly because they are all conditioned on the same prompt. A world model literally cannot, because frame `$t+1$` depends on action `$a_t$`, which depends on what the human saw in frame `$t$`. You cannot precompute the future. This is the deep reason interactivity *forces* autoregression, which we formalize in Section 5.
- **Consistency must be maintained without seeing the whole clip.** A passive model has global context — every frame can attend to every other frame, so the dog stays the same dog because the model sees the whole sequence at once. A world model only ever sees the past. Keeping the world consistent across a turn-and-return is a *memory* problem, and it is the single hardest open problem in the field (Section 9).

So the move from passive to interactive is not a small conditioning tweak. It trades global context for streaming, trades batch latency for per-frame latency, and trades "render a plausible clip" for "respond correctly to an action right now." Everything else in this post is a consequence of that trade.

## 3. The anatomy of an interactive world model

What does the architecture actually look like? It reuses most of the video stack from the rest of this series and adds two pieces: a way to get the action *into* the model, and a way to run the model *one frame at a time, fast*. Figure 3 stacks the components: a frame encoder that maps pixels to a compact latent (the same [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) the series leans on), an action encoder that embeds the action, a latent dynamics model that is a DiT denoiser conditioned on both the context frames and the action, a bounded memory window of recent frames, and a fast decoder back to pixels.

![Stacked architecture diagram of an interactive world model showing a frame encoder, an action encoder, a latent dynamics denoiser conditioned on the action, a bounded memory window of recent frames, and a fast decoder that streams the next frame](/imgs/blogs/video-models-as-world-models-3.png)

Walk it top to bottom with the running corridor example. A frame `$o_t$` (the current view) is encoded by the 3D-VAE into a latent `$z_t$` — this is the same compression that makes video tractable everywhere in the series, typically `$8\times$` in each spatial dimension and `$4\times$` in time, so a `$256\times 256$` frame becomes a `$32\times 32$` latent grid. The action `$a_t$` (say "press W" or a continuous "look 12 degrees left") is embedded into a vector `$e_t = \text{ActionEmbed}(a_t)$`. The dynamics model is a denoiser that takes the noised latent of the *next* frame, the diffusion timestep, the recent context latents `$z_{t-k:t}$`, and the action embedding `$e_t$`, and predicts the noise — exactly the `$\mathcal{L}_\text{world}$` loss from Section 1. The memory window bounds how many past frames the dynamics model conditions on (more on why this is both necessary and a problem in Section 9). Finally the predicted clean latent `$z_{t+1}$` is decoded to pixels by a *fast* decoder, because the VAE decode is on the per-frame critical path and is very often the latency wall, as the [long-video post](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) showed.

The one design decision that matters most here is *where the action goes in*. The naive thing — encode the action and concatenate it to the output — barely works, because by the time you are at the output the model has already decided the frame. The action has to influence the *generation*, which means injecting `$e_t$` into the denoiser the way text conditioning is injected in a text-to-video model: as cross-attention keys/values, or added to the timestep embedding, or via adaptive layer norm (AdaLN) modulation. The strongest systems use AdaLN-style modulation or cross-attention so the action steers every block of the denoiser, not just the last layer. Get this wrong and the model ignores your input — it generates plausible video that has nothing to do with the key you pressed.

### Diffusion vs autoregressive token backbones

There are two live families for the dynamics model, and they sit on either side of a tradeoff worth naming. The **diffusion** family (GameNGen, Oasis, most of the open work) makes the dynamics model a few-step diffusion denoiser over latent frames — it inherits the visual quality of the diffusion stack and the step-distillation tricks that make it fast. The **autoregressive-token** family (Genie's lineage, and discrete-token world models generally) tokenizes frames with a VQ encoder and predicts the next frame's tokens with a transformer, like a language model over spacetime tokens. Tokens are natural for action-conditioning (the action is just another token in the sequence) and for the latent-action trick we will see in Section 4, but historically lagged diffusion on raw visual fidelity. The 2024–2026 frontier is blurring the two — diffusion models borrowing autoregressive rollout, token models borrowing diffusion decoders — but the mental split is still useful when you read a paper: ask whether the next frame is *denoised* or *decoded token by token*.

## 4. GameNGen: a neural network that runs DOOM

The cleanest existence proof that this works came from Google in 2024: **GameNGen**, a system that runs the classic game DOOM in real time, entirely inside a diffusion model, conditioned on the player's actions. No game engine. The neural network *is* the engine. You press fire, an enemy reacts; you open a door, the room beyond appears; you take damage, the HUD updates. It runs at roughly 20 frames per second on a single TPU, and in a human study the raters could barely tell short clips of the neural DOOM from the real DOOM. That is the headline, and it is worth sitting with: a diffusion model learned the *playable dynamics* of a game well enough to be mistaken for the game.

How? The recipe is a beautiful piece of engineering and every step exists to solve a specific problem. First, they generated training data by letting an RL agent *play* DOOM for a very long time, logging the frames *and the actions* — this is the crucial difference from passive video data, where you have frames but no actions. With (frame, action) pairs you can train action-conditioning directly. Second, the dynamics model is a latent diffusion model (a fine-tuned Stable Diffusion backbone) conditioned on a sequence of past frames *and* the past actions, trained to denoise the next frame. The action is injected into the denoiser, and the past frames provide the state. Third — and this is the trick that separates a demo from a usable thing — they fought *autoregressive drift* head-on.

### The drift fix: noise augmentation on the context

Here is the problem GameNGen had to solve, and it is the [long-video rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) drift problem in its most acute form. At training time, the model conditions on *real* past frames. At inference time, it conditions on its *own* generated past frames, which contain small errors. Those errors are out of the training distribution, the model misreads them, produces a slightly worse frame, conditions on *that*, and the quality spirals — the corridor melts into mush within seconds. The structural fix GameNGen used is to add noise to the context frames during training. By corrupting the conditioning frames with random Gaussian noise (and telling the model how much, via a noise-level embedding), you force the model to learn to *correct* imperfect history rather than trust it blindly. It is a vaccine: train on slightly-broken history so that at inference, when the history really is slightly broken, the model already knows how to clean it up and stay on the rails. This single technique is what turned the rollout from "melts in 3 seconds" to "stable for minutes," and you will see the same idea — robustifying against your own errors — everywhere in this field.

The formalization is clean. Without noise augmentation, define the per-step error `$\delta_t = \lVert \hat{o}_t - o_t^\star \rVert$` between the generated frame and the true frame. In naive rollout these errors compound roughly multiplicatively because each frame is conditioned on the last frame's error, so after `$n$` steps the error grows like `$\delta_n \approx \delta_1 (1 + \rho)^{n}$` for some amplification factor `$\rho > 0$` — exponential drift, the curse of autoregression. Noise augmentation trains the model to be a *contraction* on the error: it learns a map that *reduces* small input perturbations, pushing `$\rho < 0$` so that errors decay rather than amplify. That is the whole game. (The [long-video post](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) derives this exposure-bias view in detail; here it is the load-bearing trick.)

The mechanism is worth seeing in one more line, because it explains *why* corrupting the input teaches correction. During training you sample a corruption level `$\sigma \sim U[0, \sigma_\text{max}]$`, add `$\sigma \cdot \eta$` (with `$\eta \sim \mathcal{N}(0, I)$`) to each context frame, and pass `$\sigma$` to the model through its own embedding so the model *knows* how corrupted its history is. The objective then includes corrupted-context examples, so the denoiser's learned function `$g_\theta$` is optimized to map a noisy history back toward the correct next frame: `$g_\theta(o_{\le t} + \sigma\eta,\, a_t) \approx o_{t+1}^\star$`. At inference, the model's *own* generation errors look, distributionally, like a small `$\sigma$` corruption — exactly the regime it was trained to correct. So the model spends a little of its capacity on error-correction and gains rollout stability in return. The cost is a slight loss of sharpness (you trained partly on blurred history), which is a tradeoff GameNGen accepted and tuned. This is the same family of idea as scheduled sampling and self-forcing in sequence models: make training look like inference so the exposure gap closes.

### How the action actually steers the denoiser

It is easy to say "condition the denoiser on the action" and hand-wave the mechanism, so let me be precise, because getting it wrong is the most common reason a world model ignores your input. The action embedding `$e_t$` has to modulate the *computation* of the denoiser, not decorate its output. The three injection points that work, in rough order of strength:

1. **Adaptive layer norm (AdaLN-Zero) modulation.** This is the DiT-standard route. From `$e_t$` (fused with the timestep embedding) you regress per-block scale and shift parameters `$(\gamma, \beta)$` and a residual gate `$\alpha$`, and apply them inside every transformer block: `$\text{block}(x) = x + \alpha \cdot \text{Attn}(\gamma \cdot \text{LN}(x) + \beta)$`. Because `$\gamma, \beta, \alpha$` depend on the action, the action reshapes the activation statistics of *every* layer. This is the strongest, most-used route and it is what the code in Section 10 uses via the `tconds` signal.
2. **Cross-attention.** Treat `$e_t$` (or a short sequence of action tokens) as keys/values and let the frame tokens attend to them, the same way text conditioning enters a text-to-video model. Natural when the action is structured (a camera pose, a trajectory) rather than a single discrete id.
3. **Token concatenation.** Append the action embedding as an extra token in the input sequence. Cheapest, weakest — fine for autoregressive-token backbones where everything is a token anyway, but for diffusion the AdaLN route dominates.

The reason output-concatenation fails is causal in the computational sense: by the output layer the denoiser has already committed the frame's content across two dozen blocks, so adding the action at the end can only nudge it, not redirect it. The action must enter *early and everywhere*, which is what AdaLN modulation buys. If you build one of these and find the model produces lovely video that ignores your keypresses, this is the first thing to check.

#### Worked example: GameNGen's numbers

Take the GameNGen system as reported (treat these as approximate where I say so). The dynamics model is a fine-tuned Stable Diffusion v1.4 backbone, roughly **860M parameters** in the U-Net. It runs at about **20 frames per second on a single TPU-v5**, which means a per-frame budget around **50 milliseconds** — and to hit that they reduced the diffusion sampler to a very small number of denoising steps (on the order of 4) rather than the 50 a passive model would use, because at 50 steps per frame you would be at sub-1-fps and there would be no game to play. The conditioning context is about 64 past frames. In the human-evaluation study, raters distinguished short clips of simulated DOOM from real DOOM only slightly better than chance. The honest caveat, which GameNGen's own authors make: this is *one* game, with a *fixed* set of actions, and the model was trained on enormous amounts of play *of that specific game*. It is a proof that neural networks can simulate an interactive environment in real time — not a proof that they can do so for arbitrary worlds. That generalization is exactly what the Genie line went after.

## 5. Why interactivity forces streaming, causal generation

Step back from the systems and ask the structural question: *why* must an interactive world model generate frame by frame, autoregressively, with bounded latency? It is not a choice; it is forced by the information structure of interactivity, and the argument is short and worth internalizing.

Recall that a passive video diffusion model denoises all `$T$` frames *jointly* — every frame attends to every other frame across all diffusion steps, which is why passive models have such good global coherence. That joint denoising is only possible because the conditioning (the prompt) is fixed in advance. In an interactive setting the conditioning is the action sequence `$a_{1:T}$`, and here is the catch: `$a_t$` is chosen by the human (or agent) *after seeing* `$o_t$`. The action at time `$t$` is a function of the observation at time `$t$`, which the model must produce *before* it can know `$a_t$`, which it needs to produce `$o_{t+1}$`. You cannot denoise frame `$t+1$` jointly with frame `$t$` because frame `$t+1$`'s conditioning does not exist until frame `$t$` has been generated, shown to the human, and acted upon. The dependency chain is strictly causal in real time:

$$
o_t \;\rightarrow\; \text{(human sees } o_t \text{)} \;\rightarrow\; a_t \;\rightarrow\; o_{t+1} \;\rightarrow\; a_{t+1} \;\rightarrow\; \cdots
$$

This is the same constraint that makes a language model generate tokens one at a time during chat — it cannot generate your next message for you, so it cannot precompute past the point where your input is needed. Interactivity *is* a real-time causal dependency between output and input, and a real-time causal dependency forbids joint generation. Hence streaming. Hence autoregression. Hence the latency budget, because the human is waiting between every output and the action that follows it.

This has a sharp practical consequence that I want to state plainly because it is where so much interactive-video engineering goes: **the only way to be fast enough is to push the per-frame cost down to a handful of operations.** A passive model amortizes 50 denoising steps over a whole clip it renders offline. An interactive model pays its denoising steps *on the critical path, per frame, with a human waiting.* So everything from the [efficient and real-time generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) post becomes mandatory rather than optional: step distillation down to 1–4 steps, feature caching across frames (consecutive frames are extremely similar, so you can reuse most of the previous frame's intermediate activations), a deliberately lightweight VAE decoder, and a bounded memory window so attention does not grow with the length of the session. The interactive world model is the most demanding consumer of every efficiency trick the series has covered, all at once.

### The error-accumulation tax of going causal

Going causal is not free on the quality side either, and this is the bridge to the long-video post. Because the model conditions on its own outputs, it pays the exact error-accumulation tax we formalized in Section 4: small per-frame errors are fed back as conditioning and risk compounding. A passive model that denoises jointly does *not* pay this tax — it never conditions on its own imperfect output mid-clip. So interactivity buys you control and unbounded length at the price of drift, which is the same fundamental trade the [long-video and autoregressive rollout post](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) is built around, now sharpened to a per-frame, real-time setting. Every stabilization technique from that post — noise-augmented context, Diffusion Forcing's per-frame noise levels, self-forcing where the model trains on its own rollouts — is a tool for paying less of that tax. An interactive world model is, in a real sense, the long-video problem with a human in the loop and no second chances.

## 6. Genie: learning the action interface from unlabeled video

GameNGen needs *logged actions* — it works because someone recorded which key was pressed for every frame of DOOM. That is a severe limitation, because the vast majority of video on the internet has *no action labels*. You have billions of hours of someone walking through a city, a hand picking up a cup, a car driving down a road — frames, but no record of the steering input, the muscle command, the intent. If world models could only learn from action-labeled data, they would be stuck at the scale of a few games. DeepMind's **Genie** (2024) broke that ceiling with an idea that is, to me, the most beautiful in this whole area: **infer the actions.**

The insight is that between any two consecutive frames, *something changed*, and whatever caused that change is, effectively, the action — even if no one recorded it. So Genie trains a **latent action model** that looks at frame `$o_t$` and the *next* frame `$o_{t+1}$` and infers a small, *discrete* latent code `$\tilde{a}_t$` that best explains the transition, jointly with a **dynamics model** that takes `$o_t$` and `$\tilde{a}_t$` and predicts `$o_{t+1}$`. The latent action is quantized to a tiny vocabulary (on the order of 8 codes in the original Genie) and is forced through an information bottleneck, so it cannot just memorize the next frame — it has to capture the *controllable* part of the change, the part a player would command. Crucially, the latent action model only sees the future frame *at training time*. At inference, you throw it away and feed the dynamics model a latent action *you* choose — and because the codes learned to mean things like "move left," "move right," "jump," you get a controllable, playable world from a model trained on *video with no action labels at all*. Figure 4 shows the joint training: two real frames go into the latent action model, which produces a discrete code; that code plus the current frame go into the dynamics model, which is trained to reconstruct the next frame; at test time you pick the code yourself.

![Graph of Genie's latent action model where two consecutive unlabeled frames produce a discrete latent action code that feeds the dynamics model alongside the current frame to predict the next frame with a reconstruction loss while at test time a chosen code controls generation](/imgs/blogs/video-models-as-world-models-4.png)

Why does the discrete bottleneck do the work? Think about what the latent action model is being asked to do. It must compress the entire difference between `$o_t$` and `$o_{t+1}$` into one of, say, 8 codes. Eight codes cannot store "the door at pixel (143, 88) opened by 12 degrees while a shadow moved." They are far too small. So the model is forced to discover the *low-dimensional controllable structure* of the transitions — the handful of moves that recur across the dataset — and assign a code to each. In a platformer that comes out as left/right/jump/duck; in a navigation world it comes out as forward/back/turn. The bottleneck *is* the inductive bias toward a usable action space. This is the same logic by which a VQ-VAE's small codebook forces it to learn meaningful visual primitives, applied to *dynamics* instead of appearance. (If you want the change-of-variables and codebook machinery, the image series covers [VAEs from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).)

Quantify the bottleneck to see why it has to capture *control* specifically. A single video frame at the latent resolution carries thousands of bits of information; the difference between two consecutive frames carries less, but still hundreds of bits in a busy scene. A codebook of `$K = 8$` entries can carry at most `$\log_2 8 = 3$` bits per transition. Three bits cannot reconstruct the next frame — not remotely — so the latent action model *cannot* solve its task by smuggling the answer through the code. The only way the joint system reaches a low reconstruction loss is for the dynamics model to do the heavy lifting (it sees `$o_t$` in full) while the 3 bits of latent action select *which of a few possible continuations* the actual transition took. And the few continuations that recur often enough to be worth a code are precisely the *controllable* ones — the moves a player or agent repeatedly makes. The information bottleneck is doing causal disentanglement by starvation: it is too small to encode appearance, so it is forced to encode intent. Widen the codebook and you can recover finer control (Genie 2 and later use richer action interfaces), but you also risk the codes starting to leak appearance and losing their clean "this is a *move*" semantics. The size of the bottleneck is a dial between controllability and disentanglement, and it is one of the most important hyperparameters in the whole system.

#### Worked example: the latent-action bit budget

Put numbers on it. Suppose your latent frames are `$32 \times 32 \times 16$` channels, and the inter-frame *difference* in a typical navigation clip has, say, an estimated `$\approx 400$` bits of genuinely unpredictable content (the rest is predictable from `$o_t$` and the scene). A latent action codebook of `$K = 8$` supplies **3 bits**; `$K = 64$` supplies **6 bits**; `$K = 256$` supplies **8 bits**. Even the largest of these is a `$50\times$` to `$130\times$` compression of the unpredictable content, which is why the dynamics model — not the code — must reconstruct the frame, and why the code is forced onto the controllable axis. Now the controllability tradeoff is concrete: with 3 bits you can name at most 8 distinct moves (enough for a platformer: left, right, jump, duck, and a few combinations); to express "turn left *or* right, by a small *or* large amount, while moving *or* stationary" you need on the order of `$2 \times 2 \times 2 = 8$` to a few dozen codes, so a 3-bit space is already cramped for a 3D navigation world and a 6–8 bit space is the practical floor. This is exactly the progression from Genie 1's tiny action vocabulary to the richer interfaces of Genie 2 and 3 — more bits in the action channel buys finer control at the cost of harder disentanglement.

### Genie 2 and Genie 3: from seconds to minutes

Genie 1 was trained on 2D platformers at low resolution — a proof of the latent-action idea. **Genie 2** (DeepMind, late 2024) scaled it to *3D* worlds: from a single image prompt it generates a playable 3D environment you can navigate with keyboard and mouse, with emergent object interactions, lighting, and even a rough sense of physics, holding together for on the order of 10–20 seconds. **Genie 3** (2025) pushed two axes that matter enormously for usefulness: **horizon** and **memory**. It generates interactive worlds at 720p and around 24 frames per second, and — the headline capability — it maintains *consistency over minutes*, so a place you visited and walked away from is still there, recognizably, when you return. That persistence is the thing earlier models could not do, and it is the difference between a hallucination you steer and a *world* you can build a task in. Genie 3 also added prompted world events (you can ask the world to change while you are in it) and is explicitly framed by DeepMind as a step toward training *embodied agents* inside generated environments. Treat the exact specs as approximate from public communications, but the *direction* is unambiguous: each generation is more general (any prompt, not one game), longer-horizon (minutes, not seconds), and more consistent (persistent state, not amnesia).

## 7. The system landscape: GameNGen, Genie, Oasis

It helps to see the systems in one frame, because they are *not* interchangeable — each makes a different bet on generality versus horizon versus openness. Figure 5 is the comparison matrix: rows are the four systems, columns are their action space, the horizon they hold consistency over, their consistency level, and the use case they actually fit.

![Matrix comparing GameNGen, Genie 2, Genie 3, and Oasis across action space, horizon, consistency, and use case showing GameNGen narrow and real-time, Genie 3 general with minute-scale memory, and Oasis as the open self-hostable reproduction](/imgs/blogs/video-models-as-world-models-5.png)

A few things stand out when you read it across rows. **GameNGen** is the *narrow but real-time* corner: one game, logged actions, beautiful real-time fidelity, near-perfect consistency *because the world is small and fixed*. **Genie 2** is the *general but short* corner: any image prompt becomes a playable 3D world, but it drifts after 10–20 seconds. **Genie 3** is the current frontier: general *and* minute-scale consistent, which is the combination that makes it interesting for training agents. **Oasis** (Decart and Etched, 2024) is the *open* corner — an open interactive world model that plays like Minecraft, released with weights you can actually run, which is why it matters disproportionately to practitioners even though its consistency and horizon are below the closed frontier. Oasis is built on the same recipe: a diffusion dynamics model conditioned on logged Minecraft actions and past frames, optimized hard for real-time inference (it was demoed running interactively in a browser). If you want to *touch* an interactive world model rather than read about one, Oasis is the entry point.

Here is the same landscape as a table you can actually pull numbers from. I have marked approximate figures explicitly; do not quote the soft ones as gospel.

| System | Year | Backbone | Action space | Real-time? | Horizon (consistency) | What it's for |
| --- | --- | --- | --- | --- | --- | --- |
| GameNGen | 2024 | SD-1.4 diffusion (~860M) | Logged DOOM keys | Yes (~20 fps, TPU) | Seconds–minutes, one fixed game | Proof: neural net runs a game in real time |
| Genie 1 | 2024 | Autoregressive tokens (~11B) | **Learned latent** (≈8 codes) | No (offline) | Seconds, 2D platformers | Proof: control learned from unlabeled video |
| Genie 2 | 2024 | Latent diffusion | Learned latent + keyboard | Near (offline-ish) | ~10–20 s, 3D from one image | Playable 3D worlds from a prompt |
| Genie 3 | 2025 | Diffusion (undisclosed) | Nav + prompted events | Yes (~24 fps, 720p)* | **Minutes**, persistent state | Embodied-agent training environments |
| Oasis | 2024 | Diffusion (~500M–billions)* | Logged Minecraft keys | Yes (real-time demo) | Seconds (drifts) | **Open** interactive model you can run |

*Approximate from public communications/blog posts, not a peer-reviewed spec. Parameter counts and exact frame rates vary by report and configuration.

The throughline: every one of these is the same equation — `$p(o_{t+1} \mid o_{\le t}, a_t)$` — implemented with a different backbone, a different action interface (logged vs learned-latent), and a different amount of engineering thrown at horizon and latency. The research frontier is a race along three axes at once: **generality** (one game → any prompt), **horizon** (seconds → minutes), and **openness** (closed demo → runnable weights). No system wins all three yet.

## 8. The payoff: agents that learn inside a simulator

Why does any of this matter beyond a very impressive tech demo? Because a world model is the thing that lets an agent *learn without acting in the real world*. This is the connection to model-based reinforcement learning and robotics, and it is, in my view, the real reason world models are a frontier and not a curiosity.

The setup is this. Real experience is expensive and sometimes dangerous: a robot arm takes seconds per grasp and breaks if it drops things; a self-driving policy cannot learn lane-keeping by crashing a million times. But if you have a *learned simulator* of the environment — a world model — the agent can roll out *simulated* trajectories inside it: try an action, see the predicted next observation, try another, plan, all without touching reality. This is the heart of model-based RL, made famous by Ha and Schmidhuber's 2018 "World Models" paper (which trained a policy almost entirely inside a learned dream of a car-racing game) and the Dreamer line that followed. Figure 6 stacks the loop: collect scarce real experience, fit a world model to it, generate thousands of cheap simulated rollouts, train the policy inside the model, occasionally act in the real world to gather new data and correct the model, and the win is *sample efficiency* — far fewer real-world steps for the same policy quality.

![Stacked diagram of the learning-in-the-model loop where scarce real experience trains a world model, the world model produces many cheap simulated rollouts, the policy is trained inside the model, and occasional real actions collect new data, yielding sample efficiency](/imgs/blogs/video-models-as-world-models-6.png)

The phrase to hold onto is **"learning in the model."** The agent dreams, and it learns from the dream. The quality of what it learns is bounded by the fidelity of the dream — which is exactly why pushing video-quality world models (Genie 3's photorealistic, minute-long, consistent worlds) matters for robotics: the better and longer the simulated rollouts, the more of the policy you can learn before you ever touch a real robot. There is a direct line from "a diffusion model that runs DOOM" to "a diffusion model that runs a *kitchen* well enough to train a robot to load a dishwasher inside it." That line is the whole bet. If you want the embodied-policy side of this, the robotics posts on [vision-language-action models](/blog/machine-learning/ai-agent/vision-language-action-models-rt2-to-openvla) and [building effective robotic agents](/blog/machine-learning/ai-agent/building-effective-robotic-agents) cover the policy that *consumes* such a simulator; the world model is the environment that policy trains in.

### Why action-conditioning enables causal learning

Here is the deepest scientific point in the post, and it is why action-conditioning is not a cosmetic addition. A passive video model learns *correlations* in observation sequences: it learns that after a ball is thrown, frames of it arcing tend to follow. But it has no notion of *intervention* — it cannot answer "what if I had thrown it harder," because it never saw the action as a separate, manipulable variable. An action-conditioned model *does* separate the action out. By training on `$p(o_{t+1} \mid o_t, a_t)$` with the action varied independently, the model learns the effect of *doing* `$a_t$` — which is exactly the structure of a causal intervention `$p(o_{t+1} \mid o_t, \text{do}(a_t))$` in Pearl's sense, *provided the training data has enough action diversity from similar states*. The same starting frame with different actions producing different outcomes is the data signature of a causal effect, and it is precisely what action-conditioned datasets (logged play, robot teleoperation, exploratory agents) provide and passive video does not.

This is why a world model can support *planning* — search over action sequences to reach a goal — while a passive video model cannot. Planning requires asking counterfactual "what if I do this" questions, and only a model with the action as a manipulable input can answer them. Passive video can extrapolate; action-conditioned video can *plan*. That distinction — extrapolation versus intervention — is the line between a very good video generator and a usable world model, and it is the scientific core of why this whole research direction exists.

#### Worked example: the cost case for learning in the model

Make the sample-efficiency claim concrete with rough numbers. Suppose a real robot grasp attempt takes about **5 seconds** of wall-clock plus reset, and a model-free policy needs on the order of **1,000,000** environment interactions to reach competence on a manipulation task. That is `$5 \times 10^6$` seconds `$\approx$` **58 days** of continuous robot time, ignoring hardware wear and supervision — clearly infeasible. Now suppose a learned world model can roll out a simulated step in **2 milliseconds** on a GPU. The same million interactions inside the model take `$2 \times 10^3$` seconds `$\approx$` **33 minutes**, and you can run many in parallel. Even if the agent still needs, say, **20,000** *real* interactions to keep the world model accurate (about **28 hours** of robot time, feasible), you have cut real-world steps by `$\approx 50\times$`. The dream is not free — building and correcting the world model costs real data and the simulated dynamics are imperfect — but the asymmetry between **5 seconds** of reality and **2 milliseconds** inside the model is the entire economic argument for model-based RL, and a high-fidelity video world model is what makes the simulated rollouts realistic enough to transfer.

Here is the same tradeoff as a table, which is the honest accounting an engineer should do before committing to model-based training:

| Approach | Real-world steps | Wall-clock for those steps | Risk / failure mode |
| --- | --- | --- | --- |
| Model-free, real environment | ~1,000,000 | ~58 days robot time | Infeasible; hardware wear, danger |
| Model-free, hand-built simulator | 0 real (sim only) | hours (GPU) | Sim-to-real gap; engineering cost of the sim |
| Model-based, learned world model | ~20,000 (correction only) | ~28 hours robot time | Dream-to-real gap; policy exploits model errors |
| Simulated rollouts (the dream itself) | 0 | ~33 min per 1M simulated steps | Compounding model error past the horizon |

The crucial new risk in the learned-world-model row is the **dream-to-real gap**, which is the sim-to-real gap with a twist: a hand-built simulator is *wrong in known ways* (you wrote it, you know its physics is approximate), whereas a learned world model is *wrong in unknown ways* (it has subtle, data-dependent errors you did not author and cannot enumerate). A policy trained by reinforcement learning is an adversary against its environment — it will find and exploit any way to get reward, including exploiting a physics bug in the dream that does not exist in reality. This is the single most important caveat for learning inside the model: the better your policy optimizer, the more aggressively it will mine your world model's errors, and the larger the dream-to-real gap can become. The mitigations are the obvious ones — keep the world model honest with periodic real data (the flywheel in Figure 6), penalize the policy for visiting states where the world model is uncertain, and validate on real hardware before trusting a policy. None of this is solved; it is the active research frontier where the video-world-model line meets robotics.

## 9. The hard open problems

I want to be honest about how far this is from solved, because the demos are seductive and the gaps are real. Four problems stand between today's interactive world models and a genuine learned simulator, and they are the [physics-and-limits](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation) post in miniature.

**Long-horizon consistency.** This is the headline failure mode and the one Genie 3 made real progress on without solving. Because the model conditions only on a bounded window of past frames, anything that scrolled off that window is *gone* — and when you return to it, the model invents it fresh, usually differently. You walk down a corridor, turn around, and the door you came through is now a window, or the room has rearranged. Extending the memory window helps but costs latency (attention over more frames) and VRAM, and it never fully solves the problem because no finite window can remember an arbitrarily large world. The arithmetic is unforgiving: if your context window holds `$K$` frames and you generate at `$f$` frames per second, the model's memory horizon is exactly `$K/f$` seconds — at `$K = 64$` and `$f = 20$` that is 3.2 seconds. Anything you did more than three seconds ago is, to the model, as if it never happened. Genie 3's minute-scale consistency implies either a far larger effective window or — more likely — an explicit memory mechanism that summarizes the past into a compact persistent state rather than holding raw frames, because holding a full minute of raw context frames in attention at 720p would be ruinous on both latency and VRAM.

**Persistent state and memory.** Closely related but distinct: even within the horizon, the model has no *explicit* state — no symbolic record of "the player has the key," "the door is unlocked," "there are three enemies left." Everything is implicit in the pixels and the latent. So the world model cannot reliably enforce game logic or object permanence the way an engine with a state variable trivially can. The frontier research direction is giving world models an explicit, external memory (a persistent latent map, a 3D scene representation, a retrievable state) so that consistency does not depend on cramming the whole world into a rolling attention window. This is where the [camera-control and 4D generation](/blog/machine-learning/video-generation/camera-control-and-4d-generation) work connects — an explicit 3D/4D representation *is* a form of persistent state.

**True physics.** A world model trained on video learns the *statistics* of how things move, not the *laws*. It will get the common cases right (balls fall, water flows downhill) because they are well-represented in the data, and get the rare or precise cases wrong (exact conservation of volume, momentum transfer in a collision, what happens when you do something unusual) because it is pattern-matching, not simulating. This is the central thesis of the next post, [physics and the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation): visual plausibility is not physical correctness, and a model can be coherent frame-to-frame while violating gravity and conservation. For training agents this matters enormously — a policy that learns to exploit a physics bug in the dream will fail in reality (the sim-to-real gap, now between a *learned* sim and reality).

**Controllability.** The action interface is still crude. Logged-action models (GameNGen, Oasis) only support the actions that were logged. Learned-latent models (Genie) give you a discovered action space that may not map cleanly to the controls you want, and the codes can be entangled (one code does two things, or context changes what a code does). Fine-grained, predictable, composable control — "look exactly 15 degrees left while walking forward at half speed" — is not solved, and it is essential for both gaming and robotics.

A failure mode worth naming explicitly is **action–context entanglement**: the *same* action code can mean different things depending on the scene, because the dynamics model learned a context-dependent interpretation. In a platformer the "up" code might mean "jump" on the ground and "swim up" in water and "climb" on a ladder — which is sometimes what you want (context-sensitive controls feel natural) and sometimes a bug (you press up to jump and the model decides you are in water). Disentangling action from context so that controls are *predictable* is open work, and it interacts badly with the consistency problem: if the model is unsure what scene it is in (because the context drifted), it becomes unsure what your action means, and control degrades exactly when you can least afford it. There is also the inverse problem of **action coverage** — a learned-latent model can only produce continuations it saw in training, so if your dataset never shows someone doing a backflip, no code will produce one no matter how you ask. The action space is bounded by the demonstrated behaviors in the data, which is a hard limit that more parameters do not fix.

Figure 7 places these systems on a timeline so the *trajectory* of progress is visible: from Ha and Schmidhuber's tiny 2018 recurrent world model, through the Dreamer line of latent-rollout RL, to GameNGen's real-time DOOM, to the Genie foundation models stretching horizon and consistency generation over generation.

![Timeline of the world-model line from the 2018 World Models paper and the Dreamer line through GameNGen in 2024 to Genie 1, Genie 2, and Genie 3 each adding generality, horizon, and consistency](/imgs/blogs/video-models-as-world-models-7.png)

The shape of progress is clear and it is the same on every axis: each generation buys you more generality, longer horizon, and better consistency, but none of them has crossed from *pattern-matched simulation* to *causal simulation* — the thing that would make the dream physically trustworthy. That crossing is the open frontier.

## 10. Building it: action-conditioned next-frame diffusion in PyTorch

Enough concepts. Let me show you the actual mechanics, because the gap between "condition the denoiser on the action" and working code is where the understanding lives. I will build it up in three pieces: the action-conditioned denoising *step*, the *streaming rollout* loop that turns single steps into an interactive session, and a note on where a public model (Oasis-style) plugs in.

First, the core of any world model is a denoiser that takes the noised next-frame latent, the diffusion timestep, a window of clean context latents, and an action embedding, and predicts the noise. The action injection is the part that matters — here I inject it via the timestep-embedding path (the simplest robust choice) and also offer it as a cross-attention condition. This is the `$\mathcal{L}_\text{world}$` loss from Section 1 made concrete.

```python
import torch
import torch.nn as nn

class ActionConditionedDenoiser(nn.Module):
    """Predicts the noise on the next-frame latent, conditioned on a window of
    past latents and the action taken. This is the dynamics model p(o_{t+1}|o_<=t, a_t)."""

    def __init__(self, latent_ch=16, model_dim=1024, n_actions=8, ctx_frames=8):
        super().__init__()
        self.ctx_frames = ctx_frames
        # Embed the discrete action (or a continuous action via a small MLP).
        self.action_embed = nn.Embedding(n_actions, model_dim)
        # Standard sinusoidal-then-MLP timestep embedding.
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        # Patchify the noisy next-frame latent into tokens.
        self.patch = nn.Conv2d(latent_ch, model_dim, kernel_size=2, stride=2)
        # Encode the context window of clean past latents into memory tokens.
        self.ctx_proj = nn.Conv3d(latent_ch, model_dim, kernel_size=(1, 2, 2),
                                  stride=(1, 2, 2))
        self.blocks = nn.ModuleList([DiTBlock(model_dim) for _ in range(24)])
        self.head = nn.Conv2d(model_dim, latent_ch, kernel_size=1)

    def forward(self, noisy_next, t_emb, ctx_latents, action_idx):
        # noisy_next: (B, C, H, W) noised latent of frame t+1
        # ctx_latents: (B, C, K, H, W) the last K clean context frames
        # action_idx: (B,) discrete action id for a_t
        a = self.action_embed(action_idx)                 # (B, D)
        tconds = self.time_mlp(t_emb) + a                 # fuse action into the
                                                          # timestep/AdaLN signal
        x = self.patch(noisy_next).flatten(2).transpose(1, 2)   # (B, N, D) tokens
        mem = self.ctx_proj(ctx_latents).flatten(2).transpose(1, 2)  # (B, M, D)
        for blk in self.blocks:
            # AdaLN modulation from (time + action); cross-attend to context memory.
            x = blk(x, cond=tconds, memory=mem)
        x = x.transpose(1, 2).reshape(noisy_next.shape[0], -1, *noisy_next.shape[2:])
        return self.head(x)                               # predicted noise on frame t+1
```

The two lines that make this a *world* model rather than an I2V model are `tconds = self.time_mlp(t_emb) + a` (the action steers every AdaLN-modulated block) and the cross-attention to `mem` (the context window is the state). Swap the `nn.Embedding` for a small MLP and you have continuous actions (a camera delta, a joystick vector). That is the entire architectural delta from a standard [video DiT](/blog/machine-learning/video-generation/video-diffusion-transformers).

Now the streaming rollout loop — the thing that turns single denoising steps into an interactive session. This is where the latency budget and the autoregressive feedback both live. Note the few-step sampler (we cannot afford 50 steps per frame), the rolling context window, and the place where the action comes from a human or agent each step.

```python
import torch
from collections import deque

@torch.no_grad()
def interactive_rollout(denoiser, vae, scheduler, init_latent, get_action,
                        ctx_frames=8, num_steps=4, device="cuda"):
    """Stream frames one at a time, taking an action between each. This is the
    real-time loop: generate frame t+1, decode, show it, read the next action."""
    context = deque([init_latent], maxlen=ctx_frames)   # rolling state window
    frames = [vae.decode(init_latent).sample]           # show the first frame

    while True:                                          # one iteration == one frame
        action_idx = get_action()                       # human/agent picks a_t HERE,
        if action_idx is None:                           # after seeing the last frame
            break

        # Stack the current context window as (B, C, K, H, W).
        ctx = torch.stack(list(context), dim=2)
        # Start from pure noise for the next frame's latent.
        z = torch.randn_like(init_latent)
        scheduler.set_timesteps(num_steps)              # few-step sampler: 1-4 steps
        for t in scheduler.timesteps:
            t_emb = timestep_embedding(t, dim=1024).to(device)
            eps = denoiser(z, t_emb, ctx, action_idx)   # action-conditioned predict
            z = scheduler.step(eps, t, z).prev_sample   # one denoising step

        context.append(z)                               # feed our own output back in
        frame = vae.decode(z).sample                    # fast decode on the hot path
        frames.append(frame)
        yield frame                                     # stream it to the display
```

Three details in that loop carry the whole post. `get_action()` is called *inside* the loop, after the previous frame exists — that is the real-time causal dependency from Section 5 made literal; you cannot precompute past it. `num_steps=4` is the latency concession: a passive model uses 50, an interactive one cannot. And `context.append(z)` feeds the model's *own* output back as conditioning — that is the autoregressive feedback that causes drift, which is why a production version of this loop would add noise augmentation to the context (Section 4) and a much faster distilled decoder. This is roughly the skeleton GameNGen and Oasis run; the differences are in the speed of every component, not the structure.

### Where a public model plugs in

If you want to *run* this rather than train it, the practical entry point as of 2026 is an Oasis-style open interactive model. The pattern, using the kind of latent-diffusion video pipeline this series uses throughout, looks like the sketch below — load a checkpoint with a video VAE and an action-conditioned denoiser, then drive it with the rollout loop above. (Treat the exact class names as illustrative; open interactive world models are young and the APIs are not yet standardized in 🤗 `diffusers` the way `CogVideoXPipeline` or `StableVideoDiffusionPipeline` are.)

```python
import torch
# Illustrative: an Oasis-style interactive checkpoint with action conditioning.
# Real open world models currently ship as standalone repos, not unified pipelines.
from interactive_world import OasisWorldModel  # community package, not 🤗 diffusers

model = OasisWorldModel.from_pretrained(
    "decart/oasis-500m",            # open weights, Minecraft action space
    torch_dtype=torch.float16,
).to("cuda")
model.set_action_space("minecraft")  # logged-key action interface

# Map keyboard input to the model's logged action ids.
KEY_TO_ACTION = {"w": 0, "a": 1, "s": 2, "d": 3, "space": 4, "noop": 7}

def get_action():
    key = read_keypress_nonblocking()    # your input loop
    return KEY_TO_ACTION.get(key, KEY_TO_ACTION["noop"])

# Drive the streaming loop from the previous snippet.
for frame in interactive_rollout(model.denoiser, model.vae, model.scheduler,
                                 init_latent=model.encode(start_image),
                                 get_action=get_action):
    display(frame)                       # ~20 fps target on a 4090-class GPU
```

The honest state of the art for self-hosting: Oasis-class models run interactively on a single high-end consumer GPU (a 4090-class card) at playable but modest resolution and frame rate, with visible drift after a handful of seconds. The closed frontier (Genie 3) is far ahead on horizon and consistency but is not downloadable. So if you want to *learn* by building, Oasis; if you want the frontier, you read the DeepMind communications and wait.

## 11. Case studies: real numbers from the literature

Let me ground the whole post in concrete reported results, with the usual honesty about which figures are approximate.

**GameNGen (Valevski et al., Google, 2024).** The cleanest result. An SD-1.4-based diffusion model (~860M U-Net params) simulates DOOM at about **20 fps on a single TPU-v5**, conditioned on ~64 past frames and the logged player actions, using a small number of denoising steps per frame. In a human study, raters told 1.6-second and 3.2-second clips of simulated vs real DOOM apart only slightly above chance (reported around 58–60% accuracy, where 50% is chance). The decisive engineering finding was that **noise-augmenting the context frames** during training was what made multi-minute rollouts stable rather than collapsing in seconds — the drift fix from Section 4. The honest scope: one game, fixed action set, trained on a very large corpus of agent play of that game.

**Genie (Bruce et al., DeepMind, 2024).** An ~**11B-parameter** model (a spatiotemporal-transformer dynamics model plus the latent action model and a video tokenizer) trained on a large corpus of 2D platformer videos *with no action labels*. The headline result is qualitative but profound: the model learned a *consistent* latent action space (the ~8 codes map to recognizable moves like left/right/jump) purely from watching, and at test time those codes control a playable world. This is the existence proof that controllability can be *learned*, not just logged.

**Genie 2 and Genie 3 (DeepMind, 2024–2025).** Genie 2 generates playable 3D worlds from a single image with coherence on the order of 10–20 seconds. Genie 3 reportedly generates at **720p, ~24 fps**, with **consistency over several minutes** and persistent state (a visited location remains when you return), plus promptable world events — explicitly positioned as an environment for training embodied agents. These figures are from public communications, not a peer-reviewed paper, so treat the exact numbers as approximate; the *capability jump* in horizon and memory is the robust claim.

**Oasis (Decart and Etched, 2024).** An open diffusion-based interactive world model that plays like Minecraft, conditioned on logged keyboard/mouse actions, optimized hard for real-time inference and demoed running interactively in a browser. Its significance is *openness* — it is the model practitioners can actually download, run, and modify, even though its horizon and consistency trail the closed frontier and it drifts within seconds. It is the reference implementation for understanding how these systems are built.

**The model-based-RL lineage (Ha and Schmidhuber 2018; Hafner et al., Dreamer, 2020–2023).** The "World Models" paper trained a policy almost entirely inside a learned latent dream of a racing game and a Doom-like task, and the Dreamer line scaled latent-rollout RL to a wide range of control tasks, repeatedly showing that learning *inside the model* is dramatically more sample-efficient than model-free RL. This is the intellectual foundation the video-world-model line is now scaling up with far higher-fidelity simulators.

## 12. When to reach for a world model (and when not to)

A decisive section, because the seductive demos invite misuse. The action-conditioning and streaming machinery is *expensive* and *hard*, and most video tasks do not need it.

**Reach for a world model when:**

- **There is a human or agent in the loop choosing actions.** Gaming, interactive media, embodied-agent training, any "what if I do X" setting. If the output must respond to live input, you need action-conditioning and streaming — there is no shortcut.
- **You need counterfactual rollouts for planning.** If the point is to search over action sequences (model-based RL, robot planning, "predict the consequences before acting"), only an action-conditioned model can answer the intervention questions planning requires. Passive video cannot.
- **You can supply action-labeled data, or your domain has learnable latent actions.** GameNGen-style logged actions or Genie-style learned latent actions both work; if you have neither and your video has no controllable structure, the action interface will be meaningless.

**Do *not* reach for a world model when:**

- **You just want a fixed clip from a prompt.** If no one is steering it, action-conditioning is pure overhead — use a passive text-to-video or [image-to-video](/blog/machine-learning/video-generation/latent-video-diffusion-svd-and-animatediff) model and render offline at full quality with 50 sampler steps. Do not pay the few-step quality tax and the drift tax for interactivity you will never use.
- **You need long-range *guaranteed* consistency or correct physics.** If your application breaks when the door turns into a window or volume is not conserved, a learned world model will betray you. Use a real engine (a game engine, a physics simulator) where state and laws are explicit, and consider a video model only for *rendering* on top of engine state, not for *being* the engine.
- **Latency is not a constraint but quality is everything.** A film shot does not need to respond in 50 ms; it needs to be flawless. Interactive constraints actively *hurt* quality (few steps, fast decoder, bounded context). Don't accept them when you are rendering offline.

The one-line rule: **action-conditioning earns its enormous cost only when something is taking actions in the loop.** Otherwise it is a passive video model wearing an expensive, lossy costume.

## 13. A taxonomy of design choices

To leave you with a map rather than a list, Figure 8 organizes the design space along the two axes that actually separate these systems: how the *action interface* works (labeled actions you log, versus latent actions you infer from video), and what the *state space* is (predicting in pixels, which is faithful but slow, versus predicting in a compressed latent, which is fast and is what every fast system uses). The strongest frontier systems sit at "learned latent action + latent state": Genie's inferred action space paired with a 3D-VAE latent, which buys both label-free training and real-time speed.

![Tree of world-model design choices branching into an action-interface axis with labeled actions versus learned latent actions and a state-space axis with pixel state versus latent state, with learned-latent-plus-latent-state marked as the strongest combination](/imgs/blogs/video-models-as-world-models-8.png)

Reading the tree as a decision guide: if you *have* clean logged actions for your domain (a game with input logs, robot teleoperation), labeled actions are simpler and more controllable — start there. If your data is unlabeled video at scale, you need Genie's learned latent actions, accepting that the action space is *discovered* and may not match your desired controls exactly. On the state axis the choice is nearly forced: pixel-space dynamics (early GameNGen-style approaches) are too slow for real-time, so essentially every interactive system predicts in the [VAE latent](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) and decodes fast. The interesting cell is the open frontier: learned latent actions over a latent state, with an *explicit external memory* bolted on to solve the persistence problem — that is where I expect the next generation to live.

## Key takeaways

- A world model learns the action-conditioned next-step distribution `$p(o_{t+1} \mid o_{\le t}, a_t)$`, not the passive joint `$p(o_{1:T})$`. The single symbol `$a_t$` — a time-indexed action — is the entire difference between a video you watch and a world you drive.
- Interactivity *forces* streaming, autoregressive generation with a hard per-frame latency budget, because the next action depends on the current observation in real time and so the future cannot be denoised jointly the way a passive model denoises a clip.
- GameNGen proved a diffusion model can run an interactive game (DOOM) in real time; its decisive trick was **noise-augmenting the context frames** to make the model a contraction on its own errors, which is the cure for autoregressive drift.
- Genie's breakthrough is the **learned latent action**: a tiny discrete bottleneck inferred from consecutive *unlabeled* frames discovers a controllable action space, freeing world models from the need for action-labeled data. Genie 2 brought 3D worlds from a prompt; Genie 3 brought minute-scale consistency and persistent state.
- Action-conditioning is what supports *causal* learning and planning: varying the action from similar states teaches the effect of `$\text{do}(a_t)$`, which passive video — pure correlation — cannot learn. This is why agents can *plan* inside a world model but not inside a passive generator.
- The payoff is **learning inside the model**: an agent trains on cheap simulated rollouts inside the world model and touches reality only to keep the model honest, cutting real-world steps by roughly `$10$`–`$50\times$` when the dream is faithful enough to transfer.
- The hard open problems are long-horizon consistency, persistent/explicit state, true physics (statistics, not laws), and fine-grained controllability. Genie 3 advanced consistency and horizon but none of these is solved, and none of the systems has crossed from pattern-matched to causal simulation.
- Use a world model only when something is taking actions in the loop. For a fixed clip from a prompt, a passive [I2V/T2V](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) model rendered offline is strictly better; interactivity's few-step, fast-decoder, bounded-context constraints actively hurt quality.

## Further reading

- Valevski, Kastryulin, Lu, Lehmann, Sohl-Dickstein, Fleet, Bar — *Diffusion Models Are Real-Time Game Engines* (GameNGen), 2024. The DOOM-in-a-diffusion-model paper; read the noise-augmentation section closely.
- Bruce, Dennis, Edwards, et al. — *Genie: Generative Interactive Environments* (DeepMind), 2024. The latent-action idea that learns control from unlabeled video.
- DeepMind — *Genie 2* (Dec 2024) and *Genie 3* (2025) communications. The horizon-and-memory progression; treat specifics as approximate from blog posts, not papers.
- Decart and Etched — *Oasis: A Universe in a Transformer*, 2024. The open interactive world model you can actually run; the reference for self-hosting.
- Ha and Schmidhuber — *World Models*, 2018; Hafner et al. — the *Dreamer* line, 2020–2023. The model-based-RL foundation: learning a policy inside a learned latent dream.
- 🤗 `diffusers` documentation — video pipelines (`StableVideoDiffusionPipeline`, `CogVideoXPipeline`) for the passive-generation building blocks an interactive loop reuses.
- Within this series: [Sora and the world-simulator thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) (does passive scaling give a simulator for free?), [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) (the drift you inherit), [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) (the latency budget), [physics and the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation) (why the dream breaks physics), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).

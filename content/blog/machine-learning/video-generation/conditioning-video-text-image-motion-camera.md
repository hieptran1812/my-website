---
title: "Conditioning Video: Text, Image, Motion, and Camera Control"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The practitioner's full steering wheel for a video model — T2V vs I2V vs V2V, first/last-frame and keyframe anchors, motion strength and trajectory drag, and camera-pose control — with the exact injection mechanics and diffusers knobs for each."
tags:
  [
    "video-generation",
    "diffusion-models",
    "image-to-video",
    "text-to-video",
    "video-diffusion",
    "conditioning",
    "camera-control",
    "controlnet",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/conditioning-video-text-image-motion-camera-1.png"
---

You typed a careful prompt — "a golden retriever sprinting across a sunlit meadow, low camera, slow-motion, cinematic" — and your text-to-video model gave you back a dog. Wrong breed, wrong meadow, and the camera does whatever it feels like. So you switch to image-to-video: you hand it a single photo of *the actual dog you mean*, and ask it to make that dog run. Suddenly the appearance is locked, the fur is right, the lighting matches, and the only thing the model has to invent is the motion. The clip is dramatically better, and you did less work. That gap — between describing a video in words and *showing* the model where to start — is the single most important practical lever in this entire field, and most people discover it by accident.

This post is the steering wheel. A video model is not one button; it is a console of conditioning ports, and which one you reach for decides your output quality more than the model checkpoint does. There are three base modes — **text-to-video** (T2V, prompt only), **image-to-video** (I2V, animate a given first frame), and **video-to-video** (V2V, restyle an input clip). On top of those sit frame anchors — **first/last-frame** and **keyframe** interpolation — and motion specs: **motion strength** (SVD's motion bucket), **trajectory/drag** control (DragNUWA, Tora), and **camera-pose** control (MotionCtrl, CameraCtrl). Each one enters the denoiser through a specific door — channel concatenation, cross-attention tokens, a ControlNet-style branch, or an additive embedding — and the door determines how hard the control grips and how much it fights your prompt.

![Graph showing text, start frame, motion, and camera signals entering a spacetime denoiser through cross-attention and channel-concat ports before producing a clip](/imgs/blogs/conditioning-video-text-image-motion-camera-1.png)

By the end of this post you will be able to look at a generation task and pick the right conditioning recipe in seconds: when to hand the model a first frame instead of trusting the prompt, when a motion bucket is enough and when you need an actual trajectory, how camera-pose conditioning turns a list of camera positions into a controllable orbit, and exactly which 🤗 `diffusers` knobs (`num_frames`, `motion_bucket_id`, `fps`, `decode_chunk_size`, `guidance_scale`) wire each one up. We will also be honest about the central tension this whole post lives inside: **more control fights prompt fidelity, and the strongest controls are the ones that quietly override your words.** This is a refinement of the architecture we built when we learned to [add the time axis to a diffusion model](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) — the denoiser is the same; we are just feeding it more than noise and a prompt. And it sits on the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) whose latent space is where most of these signals actually get injected. If you have not read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line frame is: video is spatial generation times temporal coherence under a brutal compute budget, and conditioning is how we spend less of that budget guessing.

## 1. The mental model: what is given versus what is invented

Here is the whole post in one sentence. **A conditioning signal is a piece of the answer you hand the model so it does not have to guess that piece.** Everything else — every mode, every knob — is a variation on *which* piece you hand over.

Think of generation as filling in a $T \times H \times W$ block of latent video. In pure T2V, the model gets only the text and a seed of noise; it has to invent the appearance of every object, the layout of the scene, the lighting, *and* the motion. That is a lot of degrees of freedom, and the failure modes follow directly: the breed is wrong because nothing pinned it, the identity drifts across frames because the appearance was never anchored, the camera wanders because no viewpoint was specified. The model is doing its best on an underdetermined problem.

Now subtract a degree of freedom. Hand it the first frame. The appearance problem — the genuinely hard part, the part where a single wrong texture ruins the clip — is now *solved by construction*. The model no longer invents what the dog looks like; it reads it off the pixels you gave it. All that remains is the motion: how those pixels should evolve over $T$ frames. This is why I2V is consistently higher quality than T2V at the same model size and compute. It is not magic; it is one fewer thing to get wrong.

![Tree taxonomy of the conditioning menu sorting base modes, frame anchors, motion specs, and structure tracks into a four-family hierarchy](/imgs/blogs/conditioning-video-text-image-motion-camera-2.png)

The full menu, shown above, sorts into four families, and the family tells you *what is given*:

- **Base modes** — what raw material the model starts from. T2V (text only), I2V (text plus a start frame), V2V (text plus a whole input clip to restyle).
- **Frame anchors** — fixed points in time the output must pass through. First/last-frame, keyframes at chosen indices.
- **Motion specs** — how much things move and where. A global strength scalar (motion bucket), a per-object path (trajectory), or a viewpoint path (camera pose).
- **Structure tracks** — a per-frame structural signal the output must follow: a depth-map sequence, an OpenPose skeleton sequence, a Canny-edge sequence. This is ControlNet, extended along time.

Keep this taxonomy in your head as we go. The deeper you go down a branch, the more you have *specified* and the less the model gets to *invent* — which is exactly the trade we will spend the second half of the post measuring.

There is a second axis hiding inside this tree that is worth naming early, because it predicts a control's behavior better than its family does: **the temporal extent of the signal.** Some controls are *point* signals — they constrain a single instant (the first frame, a keyframe). Some are *interval* signals — they constrain a span (a motion bucket sets energy across the whole clip; a camera path constrains every frame). And some are *dense* signals — one value per frame per pixel (a depth-map sequence, a Plücker ray map). A point control leaves the model maximum freedom between the anchored instants; a dense control leaves it almost none. When you find yourself surprised that a control "isn't doing anything," it is almost always because you reached for a point control (a single first frame) when the task needed an interval or dense one (a trajectory, a camera path). The amount of *time* your control covers is as important as *what* it specifies.

A useful way to feel the difference: a point control is a pushpin, an interval control is a rubber band stretched across the clip, and a dense control is a rail the output rides on. You can combine them — pin the start frame, stretch a motion bucket over the whole thing, and lay a camera rail underneath — and the model resolves all three simultaneously because, as we will see, each enters through a different port and so none erases another.

#### Worked example: counting the degrees of freedom

Take our running clip — a 5-second, 720p, 24 fps dog-running shot, which after the VAE's $4\times8\times8$ compression is roughly a $13 \times 90 \times 160 \times 16$ latent (about 3.0 million latent scalars). In **T2V**, all 3.0M scalars are unknown; the text conditions them only loosely through cross-attention. In **I2V**, the first latent frame — $1 \times 90 \times 160 \times 16 \approx 230\text{k}$ scalars, about 7.7% of the tensor — is *given*, and, crucially, it determines the appearance statistics (colors, textures, identity) that the remaining 92.3% should match. The model's job collapses from "paint a world and move it" to "propagate this world forward." Empirically that is worth roughly **3–5 VBench subject-consistency points and a meaningful FVD drop** at fixed compute, which we will quantify in §7. The lesson: anchoring 7.7% of the tensor removes far more than 7.7% of the difficulty, because appearance is the load-bearing variable.

## 2. T2V, I2V, V2V — the three base modes

These three are the trunk of the tree, so let us be precise about each, including how the conditioning physically enters.

**Text-to-video (T2V).** The model receives a text embedding (T5-XXL in CogVideoX and Wan, CLIP in the older SVD-era stacks) and pure noise. The text enters via **cross-attention**: at each transformer block, the video tokens attend to the text tokens. This is a *soft* conditioning — the model is nudged toward the prompt but nothing forces it. That softness is why T2V is the most creative mode and also the least controllable. It is the right choice when you have no reference image and you genuinely want the model to compose the scene.

**Image-to-video (I2V).** The model receives text *and* a start frame. The start frame enters through two ports at once, which we will dissect in §3: a **channel-concatenated latent** (hard, pixel-level) and often a **CLIP image embedding** in cross-attention (soft, semantic). I2V is the workhorse of production video generation in 2026 because it decouples the two hard problems — you can iterate on the still frame with a fast, cheap, high-quality *image* model (Flux, SDXL, Midjourney) until the look is perfect, then animate exactly that. The appearance bug is fixed before the expensive video model ever runs.

**Video-to-video (V2V).** The model receives text and an *entire input clip*, and re-renders it in a new style or with edited content while preserving the original motion and structure. Mechanically this is usually SDEdit-style: encode the input clip to latents, add a partial amount of noise (not all the way to pure noise — say 60–80% of the schedule), then denoise from there with the new prompt. The partial noise level is the strength knob: more noise means more creative freedom and less fidelity to the source; less noise means a faithful restyle. V2V is how you take a rough 3D render or a phone clip and "skin" it into a polished style while keeping the choreography.

The strength parameter in V2V is worth dwelling on, because it is the clearest single dial in this whole post for the control-versus-freedom trade. If the diffusion schedule runs over timesteps $t \in [0, T]$ and you re-noise the encoded input to timestep $t_{\text{start}} = s \cdot T$ for a strength $s \in (0, 1)$, then you have erased the source's high-frequency detail (which is destroyed first by noise) while keeping its low-frequency structure (which survives longer). Small $s$ (say 0.3) keeps almost everything — you are barely restyling, mostly denoising. Large $s$ (say 0.8) keeps only the coarse layout and motion skeleton, giving the new prompt room to repaint surfaces, textures, and lighting. The motion — which lives in the *low-frequency temporal* structure — survives across a wide range of $s$, which is exactly why V2V preserves choreography while changing appearance. This is the same SDEdit logic the image world uses for img2img, applied along the time axis; if you internalized it for images, you already understand V2V.

A frequent V2V mistake is treating it like I2V with extra steps. It is not: I2V *adds* motion to a static frame, while V2V *preserves* motion from a dynamic source and changes the look. Reach for V2V when you already have the motion you want (from a real clip, a 3D render, or a rough animatic) and the problem is purely appearance; reach for I2V when you have the appearance you want and the problem is purely motion. They solve opposite halves of the same equation.

Here is the same prompt in all three modes, in 🤗 `diffusers`, so you can see how little the call signature changes and how much the *inputs* differ:

```python
import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video

prompt = (
    "A golden retriever sprinting across a sunlit meadow, low angle, "
    "slow motion, shallow depth of field, cinematic."
)

# --- T2V: prompt only ---
t2v = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
t2v.enable_model_cpu_offload()      # fit on a single 24 GB card
t2v.vae.enable_tiling()             # avoid OOM in the VAE decode
frames = t2v(
    prompt=prompt,
    num_frames=49,                  # ~6 s at 8 fps after interpolation
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(frames, "t2v.mp4", fps=8)

# --- I2V: same prompt, but we hand it the first frame ---
i2v = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
i2v.enable_model_cpu_offload()
i2v.vae.enable_tiling()
first_frame = load_image("retriever_hero_frame.png")   # made by a fast image model
frames = i2v(
    image=first_frame,             # <-- the only structural change
    prompt=prompt,
    num_frames=49,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(frames, "i2v.mp4", fps=8)
```

Notice that the I2V call is byte-for-byte the T2V call plus `image=first_frame` and a different checkpoint. That single argument routes the start frame into the concatenation port inside the pipeline. The model class changed (`CogVideoXImageToVideoPipeline`) because the I2V checkpoint has extra input channels in its first conv to accept the concatenated frame — more on that next.

The V2V mode rounds out the trio. It encodes a whole input clip, re-noises to a chosen strength, and denoises with the new prompt:

```python
import torch
from diffusers import CogVideoXVideoToVideoPipeline
from diffusers.utils import load_video, export_to_video

v2v = CogVideoXVideoToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
v2v.enable_model_cpu_offload()
v2v.vae.enable_tiling()

source = load_video("rough_animatic.mp4")      # the motion we want to keep
frames = v2v(
    video=source,                   # the entire clip is the condition
    prompt="the same scene rendered as a Pixar-style 3D animation, soft lighting",
    strength=0.7,                   # 0 = identical to source, 1 = ignore source pixels
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(frames, "v2v.mp4", fps=8)
```

The `strength=0.7` is the single most important knob here: it is the $s$ from the SDEdit argument above. Sweep it — 0.5 for a faithful restyle that keeps recognizable surfaces, 0.85 for an aggressive reimagining that keeps only motion and layout — and you will feel the control-versus-freedom dial directly in the output.

## 3. How a start frame is injected (the I2V mechanics)

This is the science block of the post, because the I2V vs T2V quality gap is not a vibe — it falls out of the injection mechanics. There are two standard ways a start frame reaches the denoiser, and good I2V models use both.

![Graph of a first frame routed through a VAE to a channel-concatenated latent and through CLIP to a cross-attention context, both feeding the DiT denoiser](/imgs/blogs/conditioning-video-text-image-motion-camera-5.png)

**Port one: latent channel concatenation (the hard path).** You encode the start frame with the same 3D-VAE the model trained on, producing a latent $z_0 \in \mathbb{R}^{1 \times h \times w \times c}$. You then broadcast or place it into the temporal slot and **concatenate it along the channel axis** with the noisy video latent $z_t \in \mathbb{R}^{T \times h \times w \times c}$ at every denoising step. Concretely, the model's first projection sees $[\,z_t \,;\, z_{\text{cond}}\,]$ with $2c$ channels (plus often a binary mask channel marking *which* frames are given). Because this signal is fed in at *every* step and at *every* spatial position, the model can copy appearance directly from it — it is a hard, pixel-grounded constraint. That is why the I2V checkpoint has different first-layer weights than the T2V checkpoint: the input convolution was widened to accept the extra channels.

Formally, where T2V denoises $\epsilon_\theta(z_t, t, c_\text{text})$, I2V denoises

$$
\epsilon_\theta\big(\,[\,z_t \,;\, z_{\text{cond}} \,;\, m\,],\; t,\; c_\text{text}\,\big),
$$

with $z_{\text{cond}}$ the (noise-free) encoded conditioning frames placed in their temporal slots, zeros elsewhere, and $m$ the mask channel. The conditioning latent carries *no noise* — it is the clean signal — so the network always has a perfect reference for the anchored frames and learns to make the unanchored frames consistent with it.

**Port two: CLIP image embedding (the soft path).** Separately, you pass the start frame through a CLIP image encoder and feed the resulting embedding into cross-attention, exactly where the text embedding goes. This gives the model a *semantic* summary of the frame — "this is a golden retriever, outdoors, warm light" — that generalizes better than raw pixels for guiding global content. SVD relies heavily on this path; it conditions on a CLIP embedding of the input image plus a low-noise version of the image concatenated in latent space.

**Why this makes I2V easier than T2V, provably.** Decompose the generation distribution. T2V must model $p(z_{0:T} \mid c_\text{text})$ — the joint over appearance *and* motion. I2V models $p(z_{1:T} \mid z_0, c_\text{text})$ — motion *conditioned on* appearance. The appearance variable is the high-entropy one: there are vastly more ways to render "a dog" than there are plausible ways for *one specific rendered dog* to move over six seconds. By the chain rule, $p(z_{0:T}) = p(z_0)\,p(z_{1:T}\mid z_0)$, and I2V hands the model $p(z_0)$ for free. The model's effective task entropy drops by exactly the entropy of the appearance term, which is the dominant one. That is the formal statement of "the hard part is given." The remaining motion-modeling task is lower entropy *and* better-posed — the conditioning frame acts like a strong, always-present prior that suppresses identity drift, because every frame is being pulled toward consistency with a fixed reference.

![Before-and-after comparison of T2V versus I2V on one prompt showing the appearance gamble removed and subject consistency rising](/imgs/blogs/conditioning-video-text-image-motion-camera-3.png)

The cost side is real but small: the I2V checkpoint is a few percent larger (the widened input conv), and you pay one VAE encode of the start frame up front (milliseconds). In exchange you remove the single largest source of video-generation failure. This is why, if you can supply a first frame, you almost always should — a recommendation we will sharpen in §9.

#### Worked example: the SVD I2V call and its motion knobs

Stable Video Diffusion is the cleanest place to see I2V plus a motion control in one call. SVD takes *only* an image (no text), conditions on its CLIP embedding and a noise-augmented latent, and exposes two motion knobs we cover properly in §4 — `motion_bucket_id` and `fps`:

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16, variant="fp16",
)
pipe.enable_model_cpu_offload()

image = load_image("retriever_hero_frame.png").resize((1024, 576))

frames = pipe(
    image,
    num_frames=25,                  # SVD-XT generates 25 frames
    decode_chunk_size=8,            # decode 8 frames at a time -> caps VAE VRAM
    motion_bucket_id=127,           # 0..255: higher = more motion
    fps=7,                          # conditioning fps; lower fps -> larger inter-frame motion
    noise_aug_strength=0.02,        # how much noise is added to the cond image
    generator=torch.manual_seed(42),
).frames[0]
export_to_video(frames, "svd.mp4", fps=7)
```

On an RTX 4090 (24 GB), this runs in roughly 35–50 seconds for the 25-frame clip with `decode_chunk_size=8`; drop `decode_chunk_size` to 4 if the VAE decode is your VRAM wall (the decode, not the denoiser, is usually what OOMs at the end of an SVD run — exactly the failure the [3D-VAE post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) warned about). `noise_aug_strength` is a subtle but important knob: a little noise on the conditioning image *decouples* it slightly, letting the model deviate enough to produce motion instead of a frozen still.

### The mask channel: how the model knows which frames are given

There is a detail in the concatenation port that turns I2V into the general frame-anchoring mechanism behind first-last and keyframe conditioning, and it is worth making explicit because it is the same trick everywhere. When you concatenate $z_{\text{cond}}$ with the noisy latent $z_t$, you also concatenate a **binary mask channel** $m \in \{0, 1\}^{T \times h \times w}$ that is 1 at the temporal slots that are conditioned and 0 elsewhere. Without the mask, the model cannot tell a conditioned frame (clean signal it should copy) from an unconditioned one (zeros it should fill in) — both look like "some latent values." The mask resolves the ambiguity: it says "trust slot 0, invent the rest."

This is why the very same I2V architecture handles *any* anchoring pattern. Set $m$ to mark only frame 0 and you have I2V. Mark frames 0 and $T{-}1$ and you have first-last interpolation. Mark frames 0, 12, 24 and you have keyframe conditioning. The denoiser never changed; you changed the mask. During training, the model sees random masking patterns — sometimes one anchor, sometimes several, sometimes none — and learns a single conditional $p(z \mid z_{\text{given}}, m, c_\text{text})$ that gracefully handles whatever subset of frames you supply at inference. This *masked training* is the quiet generalization that lets one checkpoint serve I2V, first-last, and keyframe modes from a single set of weights, and it is why Wan can offer first-last with no separate model.

### Stress test: when does I2V's advantage break down?

The I2V advantage is not unconditional. Push it and it cracks in predictable places. **Large motion between the anchor and the rest:** if the prompt demands the dog leave the frame entirely by second 3, the start frame's appearance constraint weakens as the subject exits — there is nothing left for the anchor to pin, and identity can drift on whatever re-enters. **A start frame that contradicts the prompt:** as noted, the pixels win, so an I2V call whose frame and text disagree produces a clip that follows the frame and ignores the contradicting words — surprising if you expected the prompt to lead. **A start frame outside the model's training distribution:** hand SVD a tightly-cropped macro shot or an abstract pattern and the motion prior, trained mostly on natural scenes, produces incoherent or near-static motion because it has no idea how *that* kind of image should move. The fix in each case is to align the controls — choose a start frame consistent with the motion and prompt you want, or escalate to an explicit trajectory/camera path when the motion is too large for a single anchor to govern. I2V is a strong prior, not a contract; treat it as one.

## 4. Motion control: strength, buckets, and LoRAs

I2V fixes the look but leaves a question the model answers somewhat arbitrarily: *how much* should things move? Two clips of the same dog can both be "correct" — one a gentle trot, one a full sprint. Motion control is how you steer that, and there are three levels of precision.

**Level one — a global strength scalar.** This is SVD's `motion_bucket_id` (0–255) and the functionally identical "motion strength" sliders in other stacks. During training, SVD computed a motion score for each training clip and bucketed it; at inference, you select a bucket and the model produces motion of that rough energy. It is *one scalar for the whole clip* — blunt but cheap and effective. Low buckets (under ~80) give subtle, near-static motion (good for portraits, product shots); high buckets (180+) give vigorous motion but raise the risk of artifacts and warping. The related `fps` knob interacts with it: a lower conditioning `fps` implies more motion *between* consecutive frames, so SVD treats low fps as "bigger jumps." You are choosing a point on a motion-energy axis, and the model fills in plausible motion at that energy.

The science here is worth a sentence, because it explains *why* a single scalar can control something as rich as motion. The motion score is essentially the mean optical-flow magnitude over the clip: if $\mathbf{u}_i(p)$ is the flow vector at pixel $p$ from frame $i$ to $i{+}1$, the per-clip score is roughly

$$
\text{score} = \frac{1}{(T-1)\,|P|} \sum_{i=1}^{T-1} \sum_{p \in P} \lVert \mathbf{u}_i(p) \rVert_2 .
$$

This is a *one-dimensional projection* of the entire motion field — it throws away direction and locality and keeps only total energy. That is exactly why the motion bucket is blunt: it conditions on the scalar the model was trained to associate with motion magnitude, so it can scale energy up or down, but it has no channel through which to say *which way* or *where*. To recover direction and locality you have to add dimensions to the conditioning signal — which is precisely what levels two and three do.

**Level two — motion LoRAs.** A motion LoRA is a small low-rank adapter trained on clips that share a *motion pattern* rather than a subject: zoom-in, orbit-left, dolly-out, "explosion," "timelapse." AnimateDiff popularized these, and the placement is the clever part: the LoRA attaches to the **temporal-attention layers** of a frozen image backbone, not the spatial ones. That matters because the temporal layers are precisely where the model decides how tokens at the same spatial position relate *across frames* — i.e. where motion lives. A low-rank update $\Delta W = BA$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, rank $r$ a handful) on those layers nudges the motion prior toward "zoom" or "orbit" while leaving the spatial appearance machinery untouched — so the *what* (content) comes from the prompt and the *how it moves* comes from the LoRA. They are factored cleanly because they live in different layers. That is the whole reason a single small adapter can impose a reusable motion style on arbitrary content. You load one with `peft` and scale it:

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
)
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, beta_schedule="linear", steps_offset=1
)

# A motion LoRA biases *how* it moves, not *what* moves:
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-in",
    adapter_name="zoom_in",
)
pipe.set_adapters(["zoom_in"], adapter_weights=[0.8])  # dial the strength
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

out = pipe(
    prompt="a golden retriever running across a meadow, cinematic, masterpiece",
    negative_prompt="low quality, worst quality, blurry",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.manual_seed(0),
)
export_to_gif(out.frames[0], "zoom_in.gif")
```

The `adapter_weights` scalar (here 0.8) is your dial: push it toward 1.0 for an aggressive zoom, back it off toward 0.4 for a subtle one. You can stack multiple motion LoRAs with separate weights, though they start to fight each other past two or three. Why do they fight? Each LoRA adds its own $\Delta W = BA$ to the same temporal-attention weights, and the updates are not orthogonal — a "zoom-in" and an "orbit-left" both want to reshape the same attention maps, and their sum is not "zoom while orbiting" but a muddier compromise the model was never trained on. Two compatible motions (a slow zoom plus a gentle handheld shake) compose fine; two strong, competing motions (zoom plus orbit at full weight) produce mush. The practical rule: stack at most two strong motion LoRAs, and back their weights off (0.5–0.6 each) when you do, so neither saturates the layer.

**Level three — trajectory / drag control.** This is the precise end: you specify the *path* an object (or several) should follow. DragNUWA lets you draw drag arrows on the start frame — "this point goes here over the clip" — and conditions generation on those trajectories. Tora encodes trajectories as a spatiotemporal signal injected into a DiT, so object paths survive the model's strong motion prior. The injection is typically a **ControlNet-style branch** or a dedicated trajectory encoder whose features are added into the denoiser, because a trajectory is a structured, spatially-localized signal — exactly the kind cross-attention handles poorly and concatenation handles well. The trade-off is the sharp one we will measure in §6: a hard trajectory can override the prompt and the model's physical priors, producing motion that hits the path but looks unnatural (an object sliding rather than walking).

The representation choice for a trajectory mirrors the camera-pose lesson exactly. A bare list of $(x, y)$ waypoints is a poor signal for the same reason raw extrinsics are: it is not spatially aligned with the latent grid. So trajectory methods rasterize the path into a **dense per-frame map** — a Gaussian "heat blob" at the object's intended position in each frame, or a sparse optical-flow field encoding the displacement — and inject *that* through the conditioning branch. Tora goes further and turns trajectories into a *motion-patch* signal compatible with the spacetime patches the DiT already uses, so the path lives in the same token space as the video. The principle underneath all three levels is now visible: **richer motion control means a higher-dimensional, more spatially-aligned conditioning signal.** A scalar (bucket) is 1-D and unaligned; a LoRA biases a whole layer; a trajectory or camera map is dense and grid-aligned. You climb that ladder only as far as your precision requirement forces you to, because every rung costs setup effort and prompt fidelity.

#### Worked example: choosing a motion knob

You are animating a product shot — a watch on a turntable — and you need exactly one smooth 360° rotation in 4 seconds. Walk the three levels. `motion_bucket_id` alone is hopeless: it sets *energy*, not *direction*, so you would get *some* motion of roughly the right magnitude but no guarantee of a clean orbit. A **rotate/orbit motion LoRA** at weight ~0.7 gets you most of the way — consistent rotational bias — but the exact degrees and centering drift run to run. For a precise, repeatable 360°, you want **camera-pose control** (next section): specify 49 extrinsics that sweep azimuth 0° → 360° around the watch, and the rotation becomes deterministic. The rule of thumb: scalar for *mood*, LoRA for *style of motion*, trajectory/camera for *exact path*. Pay only for the precision you need — the precise controls cost more setup and more prompt fidelity (§6).

## 5. Camera control: turning extrinsics into per-frame embeddings

Camera control deserves its own section because it is the one that feels most like a real camera operator's job and because its injection mechanism is the most elegant. The question it answers: not *what moves in the scene*, but *where the lens is, frame by frame* — pans, tilts, dollies, orbits, zooms.

The input is a sequence of **camera extrinsics**: for each of the $T$ frames, a rotation $R_i \in SO(3)$ and translation $t_i \in \mathbb{R}^3$ describing the camera's pose in world space. You could naively feed the raw 6 numbers per frame, but that is a poor signal — it does not tell each *pixel* anything about its viewing ray. The trick MotionCtrl and CameraCtrl use is to convert the pose into a **Plücker ray map**: for every pixel of every frame, compute the 6-D Plücker coordinate $(\mathbf{d}, \mathbf{m})$ of the ray through that pixel given the camera intrinsics and the frame's extrinsics, where $\mathbf{d}$ is the ray direction and $\mathbf{m} = \mathbf{o} \times \mathbf{d}$ is the moment (with $\mathbf{o}$ the camera origin). This produces a dense $T \times H \times W \times 6$ tensor — one ray per pixel per frame — that encodes the geometry in a form aligned with the latent grid.

Why Plücker and not the raw 6-DOF pose? Because of a spatial-alignment argument that is the crux of why this control grips so well. The denoiser is a convolutional/attention network over a spatial grid; it conditions effectively on signals that are *spatially laid out* the same way the latent is. A raw pose vector $(R_i, t_i)$ is a single 6-number tuple for the whole frame — it has no spatial structure, so to use it the network would have to broadcast it everywhere and *infer* per-pixel rays internally, which is hard and data-hungry. The Plücker map does that inference for free and hands the network a per-pixel signal already aligned with its grid: pixel $(x, y)$ in the conditioning tensor describes the exact ray through pixel $(x, y)$ of the output. The two-part $(\mathbf{d}, \mathbf{m})$ form is also *translation-aware* — the moment $\mathbf{m} = \mathbf{o} \times \mathbf{d}$ encodes where the ray sits in space, not just its direction, so a dolly (translation, same direction) and a pan (rotation) produce distinguishable maps. A bare direction field could not tell a dolly from a zoom; the Plücker moment can. This alignment is why camera control is the tightest, most reliable control in the menu when it is supported.

That ray map is then encoded by a small network and **added to (or concatenated with) the latent** at the input of the denoiser, so every video token knows its viewing ray. Because the signal is dense and per-frame, the model gets a precise, spatially-grounded viewpoint instruction — far tighter than a text phrase like "orbit left" ever could be.

![Stack diagram showing a camera trajectory becoming Plucker ray maps, then a pose embedding added to the latent for a viewpoint-aware denoiser](/imgs/blogs/conditioning-video-text-image-motion-camera-6.png)

The figure traces the flow end to end: extrinsics → Plücker rays → pose encoder → per-frame embedding → added to the latent → a viewpoint-aware denoiser.

A sketch of the wiring, in the spirit of a CameraCtrl-style adapter, makes the shape concrete:

```python
import torch
import torch.nn as nn

def plucker_embedding(extrinsics, intrinsics, H, W):
    """extrinsics: (T,4,4) world-from-camera. returns (T,6,H,W) Plucker rays."""
    T = extrinsics.shape[0]
    # pixel grid -> camera-space ray directions via inverse intrinsics
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).float()   # (3,H,W)
    K_inv = torch.inverse(intrinsics)                                 # (3,3)
    dirs_cam = torch.einsum("ij,jhw->ihw", K_inv, pix)                # (3,H,W)
    rays = []
    for i in range(T):
        R = extrinsics[i, :3, :3]                # (3,3)
        o = extrinsics[i, :3, 3]                 # (3,)  camera origin in world
        d = torch.einsum("ij,jhw->ihw", R, dirs_cam)       # world-space direction
        d = d / d.norm(dim=0, keepdim=True).clamp_min(1e-6)
        m = torch.cross(o[:, None, None].expand_as(d), d, dim=0)      # moment o x d
        rays.append(torch.cat([d, m], dim=0))    # (6,H,W)
    return torch.stack(rays, dim=0)              # (T,6,H,W)

class PoseEncoder(nn.Module):
    """Small conv net: Plucker rays -> per-frame feature added to the latent."""
    def __init__(self, latent_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, latent_ch, 3, padding=1),
        )

    def forward(self, plucker):                  # (T,6,H,W)
        feats = [self.net(plucker[i]) for i in range(plucker.shape[0])]
        return torch.stack(feats, dim=0)         # (T,latent_ch,H,W), add to latent

# usage sketch inside the denoiser forward:
# pose_feat = pose_encoder(plucker_embedding(extrinsics, K, h, w))
# z_in = noisy_latent + pose_feat            # additive injection per frame
# eps = denoiser(z_in, t, text_ctx)
```

The key design choices in that sketch are the ones that matter in practice: the ray map is computed once and reused across all denoising steps; the pose encoder is small (it is learning a projection, not the scene); and the injection is **additive per-frame**, which keeps camera control orthogonal to the text and image conditioning so they stack. In `diffusers`, camera control today usually arrives as a model-specific adapter or a fun-camera LoRA (e.g. for Wan) rather than a single universal flag, but the mechanism under every one of them is this Plücker-ray-to-additive-embedding path.

Because camera pose enters as a *dense, per-frame, geometry-aligned* signal, it grips tightly — this is the control with the best "do exactly what I said" behavior. It is also the one most likely to fight scene content when overspecified: ask for a fast orbit through a complex scene and the model may sacrifice object permanence to satisfy the geometry. That tension is the subject of §6.

There is a deeper limit worth being honest about. Camera control tells the model where the *lens* is; it does not give the model a *3D scene* to render from that lens. A real renderer with a 3D model would produce perfectly consistent novel views because it is sampling a fixed geometry. A video model conditioned on a camera path is still *inventing* the scene frame by frame, now constrained to render it from the specified viewpoints — which is far better than no constraint, but it is not true 3D consistency. Push a camera orbit far enough around an object and the model can hallucinate a back side that does not match the front, because it never built a coherent 3D representation; it only ever modeled "what a plausible next frame from roughly here looks like." This is exactly the boundary where camera-conditioned video starts to need genuine 4D/multi-view machinery, and it is where the [camera-control-and-4d-generation post](/blog/machine-learning/video-generation/camera-control-and-4d-generation) picks up. For ordinary shots — a pan, a modest orbit, a dolly — camera conditioning is excellent; for a full 360° walkaround that must be geometrically consistent, you are at the edge of what conditioning alone can buy, and you should know that going in.

## 6. The control versus prompt-fidelity tension

Here is the uncomfortable truth nobody puts on the marketing page: **every control you add steals capacity from the prompt.** The denoiser has a fixed budget of attention and representation, and when you concatenate a start frame, add a camera embedding, and inject a trajectory, those hard signals dominate the gradient of the prediction. The text — which only ever entered softly through cross-attention — gets outvoted.

![Before-and-after comparison showing light control preserving prompt adherence while stacked hard control buys geometry at the cost of the prompt](/imgs/blogs/conditioning-video-text-image-motion-camera-7.png)

The mechanism is direct. Cross-attention conditioning (text) is *additive and soft*: it shifts the distribution but the model can ignore parts of it. Channel-concatenation conditioning (start frame, camera Plücker, mask) is *hard*: it is part of the input the first convolution sees, so the model is structurally compelled to honor it. When both are present, the hard signal wins ties. Concretely, if your start frame shows a daytime meadow but your prompt says "at night," I2V will usually give you a daytime clip — the pixels beat the words. If your trajectory drags an object along a path that the prompt's described physics would forbid, the trajectory wins and you get the slide-not-walk artifact.

This is not a bug to be eliminated; it is a budget to be managed. The practical rules:

- **Reach for the softest control that does the job.** If a prompt phrase ("slow pan left") gets you close enough, do not add camera-pose conditioning — you will lose prompt nuance for precision you did not need.
- **When you must stack hard controls, make them consistent.** A start frame, camera path, and prompt that all agree produce a clean clip. Contradictions resolve in favor of the hard signal, silently.
- **Use guidance to rebalance.** Raising `guidance_scale` strengthens the text's pull (it is classifier-free guidance on the text condition — see the [CFG post](/blog/machine-learning/image-generation/classifier-free-guidance)), partially clawing back prompt adherence when hard controls are dominating. But too high and you get the usual oversaturation and motion artifacts; 6–8 is the typical sane band for video.
- **Decouple slightly when you need motion.** SVD's `noise_aug_strength` and V2V's partial-noise level both exist because a *perfectly* hard conditioning can freeze the output. A little noise on the conditioning signal gives the model room to actually animate.

The guidance lever deserves the actual equation, because it shows precisely *what* it rebalances. Classifier-free guidance forms the denoising prediction as

$$
\hat{\epsilon} = \epsilon_\theta(z_t, \varnothing) + w \big( \epsilon_\theta(z_t, c_\text{text}) - \epsilon_\theta(z_t, \varnothing) \big),
$$

where $w$ is `guidance_scale` and $\varnothing$ is the null (unconditional) text. The guided difference $\epsilon_\theta(z_t, c_\text{text}) - \epsilon_\theta(z_t, \varnothing)$ is the *direction the text wants to move the prediction*, and $w$ amplifies it. Crucially, in I2V the hard controls — the concatenated start frame, the camera map — are present in *both* the conditional and unconditional passes (you do not drop them; only the text is dropped for the null pass). So CFG amplifies only the *text's* contribution, leaving the hard signals at full strength in both terms where they cancel. That is exactly why turning up guidance claws back prompt adherence without weakening the image or camera anchor: the anchor is in the part CFG does not touch. It also explains the ceiling — past $w \approx 9$ you are amplifying the text direction so hard that you overshoot into oversaturation and jittery motion, the video analog of the blown-out colors CFG produces in images. Some video models additionally run **dual guidance** (separate scales for the text and the image condition), which gives you an independent dial for "how hard should the start frame bind" versus "how hard should the prompt bind" — the cleanest possible handle on the tension.

![Matrix mapping each conditioning type to what it controls, the injection mechanism, and a fidelity note about how hard it grips](/imgs/blogs/conditioning-video-text-image-motion-camera-4.png)

The matrix above is the reference card for this whole tension. Read the third column as "how hard does this grip, and does it fight your prompt?" Text is the softest and weakest grip; latent concatenation (I2V, first-last) is the hardest; camera pose is tight but geometry-only so it rarely fights *content*; a ControlNet trajectory branch is the one most prone to overriding the prompt's physics. The injection port — column two — is the actual cause of the behavior in column three. Once you internalize that the *door* determines the *grip*, you can predict any new control's behavior the moment you learn how it is injected.

### The negative-prompt port and the unconditional pass

There is one more port worth naming because it is easy to forget it exists: the **unconditional (null) pass** that CFG runs alongside the conditional one. The negative prompt rides this port. When you set `negative_prompt="blurry, low quality, warped hands, flicker"`, you are not adding a constraint to the conditional pass — you are *replacing the null condition* in the CFG equation, so the guided direction becomes $\epsilon_\theta(z_t, c_\text{pos}) - \epsilon_\theta(z_t, c_\text{neg})$, i.e. "move toward the prompt and *away* from the negative." For video this is more useful than for images because the most common video artifacts — flicker, warping, morphing — have names you can put in the negative prompt, and pushing away from them measurably cleans up the output. It is a soft control (it lives on the same cross-attention path as the prompt), so it loses ties to hard controls just like the positive prompt does, but within the soft budget it is one of the cheapest quality wins available. Two practitioner notes: keep the negative prompt short and artifact-focused (a long negative prompt dilutes the push), and remember it only exists when CFG is on — at `guidance_scale=1.0` there is no unconditional pass and the negative prompt does nothing at all.

The positive prompt itself has video-specific phrasing that helps the soft port land harder. Motion verbs ("sprinting," "swaying," "drifting") condition the temporal layers more usefully than static adjectives; explicit camera language ("low angle, slow dolly in") nudges the model even when you are *not* using hard camera control, and is the right first thing to try before reaching for Plücker conditioning. The principle from §9 holds: try the soft port first because it costs nothing in setup and nothing in the prompt-fidelity budget — escalate to a hard port only when the soft one misses the bar.

## 7. Frame anchors and structure tracks (ControlNet for video)

Two more families round out the menu, and both are "anchor the output to something given."

**First/last-frame and keyframe interpolation.** Instead of anchoring only the first frame (I2V), you anchor *both ends* — give frame 0 and frame $T{-}1$, and let the model interpolate the motion between them. Mechanically this is the same channel-concatenation port as I2V, but now $z_{\text{cond}}$ has non-zero entries in *two* temporal slots and the mask $m$ marks both. Wan 2.1 ships native first-last-frame conditioning; you supply two images and get a clip that smoothly travels from one to the other. Keyframe conditioning generalizes this to anchors at arbitrary indices — frame 0, 12, 24 — for finer control of a longer shot. The science is identical to I2V's chain-rule argument, now with two (or more) givens: the model interpolates $p(z_{1:T-2} \mid z_0, z_{T-1}, c_\text{text})$.

The risk is *stalling*, and it has a clean intuition. The model is sampling from the conditional above, and that distribution concentrates on motion *paths* that start at $z_0$ and end at $z_{T-1}$. Among all such paths, the lowest-energy ones — the ones the motion prior assigns highest probability — are the *shortest* interpolations between the endpoints. If $z_0$ and $z_{T-1}$ are nearly identical, the shortest path is "barely move, then snap into place at the end," and that is exactly the near-static, slightly-creepy in-between you get. The endpoints define the *boundary conditions* of the motion, and if the boundary conditions are close, the path between them is short by construction. The fix is to make the endpoints genuinely different — frame 0 a closed door, frame $T{-}1$ a fully open one — so the shortest path *is* the motion you want, or to add a motion knob that raises the energy floor and forces the model off the lazy straight-line interpolation.

This is also why keyframe conditioning at *several* indices is so useful for longer shots: each pair of adjacent keyframes is a short, well-posed interpolation, so you decompose a long, drift-prone generation into several short, anchored ones — a theme the [long-video post](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) develops into a full strategy. Keyframes are I2V's mask trick used as a *length* tool, not just a control tool.

```python
# First-last-frame conditioning, Wan-style (API shape; the two images are the anchors)
import torch
from diffusers import WanImageToVideoPipeline      # class name as exposed by diffusers
from diffusers.utils import load_image, export_to_video

pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

first = load_image("door_closed.png")
last  = load_image("door_open.png")

frames = pipe(
    image=first,
    last_image=last,                 # the second anchor -> both temporal slots fixed
    prompt="a wooden door slowly swinging open into a sunlit room",
    num_frames=49,
    guidance_scale=5.0,
    num_inference_steps=40,
).frames[0]
export_to_video(frames, "first_last.mp4", fps=16)
```

**Structure tracks — ControlNet, along time.** The image world has [ControlNet and structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control): condition on a depth map, an OpenPose skeleton, or Canny edges so the output follows that structure exactly. The video extension feeds a *sequence* of such maps — a depth-map video, a pose-skeleton video — through a ControlNet-style branch (a trainable copy of the encoder whose features are added into the denoiser at each block). This is how you make a generated character follow a real dancer's motion (drive with a pose sequence) or restyle a clip while preserving its 3D layout (drive with a depth sequence). It is the most precise structural control available and the most expensive to set up, because you need the per-frame conditioning signal in the first place (extracted from a reference clip, or hand-authored). It composes with text and image conditioning through the same logic as everything else — its own injection port (a ControlNet branch), added into the shared denoiser.

The video ControlNet has one failure mode the image version does not: **temporal flicker in the conditioning signal itself.** If you extract a depth-map sequence frame-by-frame with a *per-frame* depth estimator, the depth values jitter slightly between frames even when the scene is static, and the model faithfully reproduces that jitter as flicker in the output — the control is *too* faithful. The fixes mirror the rest of this post: smooth the conditioning sequence temporally before injecting it (a light temporal filter on the depth/pose track), or use a *video* depth estimator that enforces its own temporal consistency. This is a recurring theme — a control is only as coherent as the signal you feed it, and along the time axis you have to make the *signal* coherent, not just the output. The same caution applies to pose: a jittery keypoint tracker yields a twitchy character. Clean the driving signal first.

There is also a subtle interaction with the base mode. A structure track on top of T2V gives the model freedom in everything *except* the structure — it invents appearance and fine motion while honoring the skeleton or depth. A structure track on top of I2V additionally pins appearance, leaving the model almost no freedom: it must produce *this look* moving along *this structure*. That is the tightest joint constraint in the menu, and it is exactly what you want for faithful motion transfer (animate *this specific character* with *this specific dance*) — and exactly what you do *not* want when you wanted the model to bring its own creativity. The number of free degrees of freedom is the thing to track; structure-on-I2V leaves the fewest.

A quick `diffusers`-shaped sketch of driving a generation with a pose sequence through a video ControlNet-style branch:

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.models import SparseControlNetModel       # ControlNet for video frames
from diffusers.utils import export_to_gif, load_video

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16
)
controlnet = SparseControlNetModel.from_pretrained(
    "guoyww/animatediff-sparsectrl-scribble", torch_dtype=torch.float16
)
pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=adapter, controlnet=controlnet, torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# the driving structure: one conditioning frame per output frame (here, scribbles/edges)
cond_frames = load_video("pose_or_edge_sequence.mp4")
out = pipe(
    prompt="a dancer in a red dress, studio lighting, cinematic",
    num_frames=len(cond_frames),
    conditioning_frames=cond_frames,        # the per-frame structure track
    controlnet_conditioning_scale=0.8,      # how hard the structure binds
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.manual_seed(7),
)
export_to_gif(out.frames[0], "structure_driven.gif")
```

The `controlnet_conditioning_scale` (here 0.8) is the structure analog of every grip dial in this post: 1.0 binds the output rigidly to the driving structure, 0.4 lets it drift toward the prompt's own ideas. Sweep it to taste, and lower it if the structure track is noisy and you are seeing flicker.

The text side itself deserves one note: the prompt is conditioning too, and how it is encoded matters. Video models inherit the [text-encoder and prompt-conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) machinery from the image world — CogVideoX and Wan use a large T5 encoder precisely because richer text features give the cross-attention port more to work with, which partly offsets text's inherent softness against the hard controls.

## 8. Measuring the I2V-versus-T2V gap honestly

It is easy to *assert* that I2V beats T2V; it is harder to measure it without fooling yourself, and the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) is blunt about why. Here is a protocol that produces a number you can trust, plus the traps that produce numbers you cannot.

**Fix everything except the variable.** Use the *same prompt set* (say 200 prompts), the *same seeds*, the *same model family* (a T2V and I2V checkpoint from the same base, so the only difference is the conditioning port), the same resolution, frame count, step count, and scheduler. For the I2V arm, generate each start frame *from the same prompt* with a fixed image model so you are not smuggling in a better appearance by hand-picking frames. Discard a warm-up run (the first generation after model load is slower and sometimes numerically different due to lazy kernel compilation). Only now are the two arms comparable.

**FVD is the headline, and it is noisy.** Fréchet Video Distance is the Fréchet distance between I3D feature distributions of generated and reference clips: $\text{FVD} = \lVert \mu_g - \mu_r \rVert^2 + \text{Tr}\big(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2}\big)$, the same formula as FID but on a video feature extractor. It is sensitive to sample size — under ~500 clips the estimate of $\Sigma$ is unstable and FVD can swing by tens of points run to run — so report it with a fixed, adequately large sample and the same reference set for both arms. A *relative* FVD (I2V vs T2V on the identical reference) is far more trustworthy than an absolute number you compare across papers, because everyone's reference set and I3D checkpoint differ.

**VBench dimensions catch what FVD blurs.** FVD is a single scalar; the consistency win of I2V shows up most cleanly in VBench's *subject consistency* and *background consistency* dimensions (DINO/CLIP feature similarity across frames). The I2V arm should post a few points higher there — that is the appearance anchor doing its job. Watch *dynamic degree* though: I2V can score *lower* on dynamic degree precisely because the anchor stabilizes the scene, and a naive reading would call that "worse." It is not worse; it is the stability-versus-motion trade, and it is the single most-gamed axis in video eval. A model can top a leaderboard's "motion smoothness" by barely moving. Always read consistency and dynamic degree *together*, and prefer a human side-by-side when the trade is the whole question.

**The honest reporting unit.** For a production decision, the number that matters is not raw FVD but *quality per accepted clip per dollar*: generate $N$ clips, have a human (or a calibrated reward model) accept/reject on appearance and motion, and report accepted-clip-cost. That is the unit the §7 budgeting example used, and it is the one that captures I2V's real advantage — its rejection rate on appearance is far lower because you curated the stills cheaply upstream. A pure FVD comparison *understates* I2V's practical win, because it does not price the appearance rejections T2V forces on you.

#### Worked example: an honest 200-clip A/B

You run the protocol on a single A100 80GB: 200 prompts, 5 seeds each (1,000 clips per arm), CogVideoX-5B T2V vs CogVideoX-5B-I2V with stills from a fixed Flux call, 49 frames at 720×480, 50 steps, same scheduler, warm-up discarded. Representative outcome: I2V lands roughly **0.96 vs 0.92 VBench subject consistency**, a **relative FVD ~20% lower**, and a **human accept rate of ~93% vs ~64%** (T2V loses most of its rejections to wrong/unwanted appearance, not bad motion). I2V's *dynamic degree* comes in slightly lower — expected, and not a defect. Net: at ~95 s/clip on the A100, the I2V arm delivers the target quality at roughly two-thirds the effective cost once rejections are priced in. Report all of it — including the dynamic-degree dip — so nobody later "discovers" the trade and distrusts the whole comparison.

## 9. Case studies: real conditioning support in shipped models

Let us ground all of this in what actually ships, with honest notes on numbers. (Where I cite a figure I am confident in the order of magnitude; where the literature is fuzzy I say so.)

![Matrix of conditioning support across SVD, CogVideoX, Wan, and HunyuanVideo showing I2V and first-last as standard and camera and trajectory as add-ons](/imgs/blogs/conditioning-video-text-image-motion-camera-8.png)

**Stable Video Diffusion (Stability AI, 2023).** The reference open I2V model. Image-only conditioning (no text), via CLIP image embedding plus a noise-augmented concatenated latent, with `motion_bucket_id` and `fps` as the motion knobs. SVD-XT generates 25 frames at 576×1024. Its enduring lesson, from the technical report, was the value of *data curation and conditioning design* — the motion-bucket scheme came from scoring training clips by motion magnitude, which is the blueprint every later motion-strength knob copied. Camera and trajectory control reach SVD only through external adapters (MotionCtrl, DragNUWA were demonstrated on SVD-class backbones).

**CogVideoX (THUDM/Zhipu, 2024).** T2V and a dedicated I2V variant (`CogVideoX-5b-I2V`), T5-XXL text encoder, a 3D causal VAE, and a DiT denoiser. The 5B model generates 49 frames at 720×480; reported VBench numbers put it in the strong-open-model tier (overall VBench in the low-80s percent range in the CogVideoX report, competitive with closed models of its era). I2V is a separate checkpoint with widened input channels — the canonical example of the concatenation port we dissected in §3. Camera/trajectory control arrive as community ControlNets and Tora-style add-ons rather than native flags.

**Wan 2.1 (Alibaba, 2025).** The open model that made first-last-frame conditioning table stakes: native I2V *and* native first-last-frame interpolation, plus a thriving ecosystem of "fun" camera-control LoRAs. The 14B-720P variant is the quality flagship; a 1.3B variant runs on consumer cards. Wan's first-last support is the cleanest production example of the two-anchor mechanism in §7. Camera control is community/LoRA-driven but well-supported.

**HunyuanVideo / HunyuanVideo-1.5 (Tencent, 2024–2025).** A large open DiT (the original HunyuanVideo is ~13B) with strong T2V and I2V; the 1.5 line pushed efficiency hard, with reports of long generations (tens of seconds) feasible on a single high-end consumer GPU. Keyframe-style conditioning is supported; camera and trajectory remain more limited / add-on. Treat any specific second-count I quote here as approximate — the efficiency numbers move with the exact variant, resolution, and step count.

One cross-cutting observation from these four: the *base modes and frame anchors converged* while *camera and trajectory did not*. By 2026 every serious open model ships native I2V, and most ship first-last or keyframe support, because the masked-concatenation mechanism (§3) is cheap to add and the quality win is large and universal. Camera and trajectory control, by contrast, still arrive as separate adapters, LoRAs, or community projects — because they need a richer, geometry-aware conditioning signal and dedicated training data (clips paired with known camera paths or object trajectories), which is harder to assemble than "any clip, mask a frame." The practical consequence for you: assume I2V and first-last are available on whatever model you pick; verify camera and trajectory support explicitly, because it is the part most likely to be missing or experimental. This split is itself a prediction — as the geometry-aware datasets mature, expect native camera control to converge next, which the [camera-control-and-4d-generation post](/blog/machine-learning/video-generation/camera-control-and-4d-generation) takes up in depth.

Here is the conditioning landscape as a decision-grade table:

| Model | Base modes | Frame anchors | Motion control | Camera / trajectory | Text encoder |
|---|---|---|---|---|---|
| SVD-XT | I2V only | first frame | `motion_bucket_id`, `fps` | external adapters | CLIP image (no text) |
| CogVideoX-5B | T2V + I2V | first frame (I2V) | prompt / LoRA | ControlNet / Tora add-on | T5-XXL |
| Wan 2.1 | T2V + I2V | first + last (native) | LoRA, prompt | fun-camera LoRAs | umT5 |
| HunyuanVideo-1.5 | T2V + I2V | keyframe | prompt / LoRA | limited | bilingual LLM/T5 |

And the quality-and-cost picture for the *modes* themselves, on named hardware, with the honesty the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) demands (fixed seed, same prompt set, warm-up run discarded, FVD on a held-out reference set):

| Mode | What's given | VBench subj. consistency | Relative FVD | Failure mode | Typical cost (RTX 4090) |
|---|---|---|---|---|---|
| T2V | text | ~0.92 | baseline | wrong appearance, drift | ~90 s / 49-frame clip |
| I2V | text + first frame | ~0.96 | ~15–25% lower | endpoint drift if motion large | ~95 s (+1 VAE encode) |
| First-last | text + both ends | ~0.96 | ~15–25% lower | stalls if ends too similar | ~95 s (+2 VAE encodes) |
| V2V | text + input clip | ~0.97 | n/a (re-render) | over/under restyle by noise level | ~110 s (+full encode) |

The VBench and FVD numbers above are representative magnitudes from open-model evaluations and the framing of the CogVideoX/Wan reports, not a single controlled study — the *direction* (I2V beats T2V on consistency by a few points, with a meaningful FVD drop) is robust and reproducible; the exact decimals depend on your model, prompt set, and reference distribution. Measure your own with a fixed protocol before trusting any leaderboard delta.

#### Worked example: budgeting an I2V-versus-T2V production run

You need 500 six-second clips for a marketing batch, target ~0.95+ VBench subject consistency, on a single RTX 4090. **T2V path:** ~90 s/clip, but you will reject roughly 30–40% for wrong appearance and re-roll, so effective time is ~90 s ÷ 0.65 ≈ 138 s/clip → ~19 hours, plus the human cost of curating which appearances are acceptable. **I2V path:** generate 500 hero stills with a fast image model first (~3–5 s each on the same card, ~35 min total, with near-100% appearance acceptance because you curate stills cheaply), then animate at ~95 s/clip with a ~5% reject rate → 95 s ÷ 0.95 ≈ 100 s/clip → ~14 hours plus 35 minutes. I2V is faster *and* higher quality *and* gives you art-direction control over every appearance — because you moved the hard, high-rejection appearance decision onto a cheap image model and let the expensive video model do only the motion. This is the single most consequential conditioning decision in a real pipeline, and it is why "supply a first frame whenever you can" is the strongest rule in this post.

## 10. When to reach for each control (and when not to)

A decisive recommendation section, because the whole point of a steering wheel is knowing which way to turn.

**Default to I2V whenever you can produce a first frame.** It is faster, higher quality, and more controllable than T2V, for the provable reason in §3. The only time to prefer pure T2V is when you have no reference and genuinely want the model to compose the scene from scratch — exploratory ideation, or when the *act of composing* is the creative value. For any production task with a known look, generate the still first.

**Use first-last only when both endpoints matter.** If you care about where the clip *ends* (a door fully open, a logo fully assembled), anchor both. If you only care about the start, plain I2V is simpler and won't stall. Never use first-last with near-identical endpoints expecting motion — it will give you a near-freeze.

**Use the motion bucket for mood, not choreography.** `motion_bucket_id` / motion-strength is the right tool when you want "more energy" or "calmer," and the wrong tool when you need a *specific* motion. Don't fight it run-to-run hoping for a particular path; escalate to a motion LoRA or trajectory.

**Use motion LoRAs for a style of motion you'll reuse.** Orbit, zoom, dolly, shake — if you want the same *kind* of motion across many clips, a motion LoRA at weight 0.6–0.8 is cheap and consistent. Don't stack more than two or three; they fight.

**Use camera-pose control for exact, repeatable camera moves.** A precise 360° orbit, a scripted dolly, a multi-shot sequence with matching camera paths — this is where Plücker-ray conditioning earns its setup cost. Don't reach for it when "pan left" in the prompt suffices; you'll spend prompt fidelity you didn't need to.

**Use trajectory/drag control only when an object's path is the deliverable.** It is the most prompt-fighting control. Use it when "this object must follow this path" is a hard requirement (a ball into a goal, a car along a road), and accept that you may need to soften it or up the guidance to keep the rest of the scene natural.

**Use structure tracks (depth/pose ControlNet) for motion transfer and faithful restyle.** Driving a character with a pose sequence, or restyling a clip while preserving its 3D layout via depth — that's the use case. It's the most expensive to set up (you need the per-frame signal) and the most precise. Don't use it for free-form generation; it constrains the output to the driving structure by design.

**On stacking:** every hard control you add costs prompt fidelity. Reach for the softest control that meets the bar, keep stacked controls mutually consistent, and lean on `guidance_scale` (6–8) and a touch of conditioning noise (`noise_aug_strength`, partial V2V noise) to keep the output from freezing or ignoring your words.

#### Worked example: a fully-conditioned hero shot

Put the whole steering wheel together for one demanding shot: *a specific character, a scripted camera orbit, energetic motion, faithful to a storyboard*. Here is the recipe, in priority order, and what each port buys you. **(1) Lock appearance with I2V** — render the character's hero frame with a fast image model and pass it as `image=`; this fixes identity (the highest-value control, §3). **(2) Add the camera orbit** — supply 49 extrinsics sweeping azimuth 0°→360° as a Plücker map through the camera adapter; this makes the orbit deterministic and repeatable (§5), and because it is geometry-only it does not fight the character's appearance. **(3) Set motion energy** — a moderate motion bucket or a gentle motion LoRA so the character is alive within the orbiting frame, not frozen (§4). **(4) Keep the prompt in play** — write the prompt for the things *not* otherwise pinned (lighting mood, expression, background style) and run `guidance_scale` at ~7 so the text still lands against the hard controls (§6).

The ordering is the lesson: you spend your degrees of freedom from most-valuable to least, and you let each control own the variable it grips best — I2V owns appearance, camera owns viewpoint, the motion knob owns energy, the prompt owns the soft remainder. Because each enters through a different port (concat, Plücker-add, additive embedding, cross-attention), they compose instead of overwriting one another. If you find two controls fighting — the prompt says "static portrait" but the camera map orbits fast — that is the signal that two ports are issuing contradictory instructions, and the hard one (the camera) will win. Resolve it at the *spec* level (make the controls agree) rather than fighting it with guidance. This is the entire art of conditioning a video model: decide what must be true, assign each truth to the port that holds it most firmly, and keep the ports from contradicting each other. Everything in this post is in service of that one workflow — which the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) turns into a full production pipeline.

## 11. Key takeaways

- **Conditioning is the biggest practical lever in video generation** — bigger than the checkpoint. Picking the right mode beats upgrading the model.
- **I2V beats T2V whenever you can supply a first frame**, provably: it hands the model the high-entropy appearance term and leaves only the lower-entropy motion to invent. Generate the still with a cheap image model, animate exactly that.
- **The injection port determines the grip.** Cross-attention (text) is soft; channel concatenation (start frame, camera Plücker, first-last) is hard; a ControlNet branch (structure, trajectory) is hard and spatial. Learn how a control is injected and you can predict how it behaves.
- **A start frame enters twice** — as a concatenated clean latent (pixel-level appearance) and a CLIP embedding (semantics) — which is why I2V both locks the look and stays steerable.
- **Camera control is Plücker rays added per frame.** Extrinsics become a dense $T\times H\times W\times 6$ ray map, encoded and added to the latent, giving tight, geometry-aligned viewpoint control.
- **Motion control has three precisions:** a global scalar (motion bucket) for mood, a motion LoRA for a reusable style of motion, and trajectory/camera for an exact path. Pay only for the precision you need.
- **More control fights prompt fidelity.** Hard signals outvote the soft text path; keep stacked controls consistent, rebalance with guidance, and add a little conditioning noise when a hard control threatens to freeze the output.
- **Measure honestly.** I2V's few-point VBench-consistency win and FVD drop over T2V are robust in direction; the exact numbers depend on your model, prompt set, and reference distribution — fix a protocol before trusting a delta.

## 12. Further reading

- Blattmann et al., *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets*, 2023 — the reference open I2V model, motion-bucket conditioning, and the data-curation lesson.
- Yang et al. (THUDM), *CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer*, 2024 — T2V/I2V with a 3D causal VAE and T5 text conditioning.
- Wan team (Alibaba), *Wan: Open and Advanced Large-Scale Video Generative Models*, 2025 — native I2V and first-last-frame conditioning.
- Wang et al., *MotionCtrl: A Unified and Flexible Motion Controller for Video Generation*, 2023 — joint camera-pose and object-motion control.
- He et al., *CameraCtrl: Enabling Camera Control for Text-to-Video Generation*, 2024 — Plücker-embedding camera conditioning, the mechanism in §5.
- Yin et al., *DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory*, 2023 — drag/trajectory control; and Zhang et al., *Tora: Trajectory-oriented Diffusion Transformer*, 2024.
- 🤗 `diffusers` documentation — video pipelines (`StableVideoDiffusionPipeline`, `CogVideoXImageToVideoPipeline`, `WanImageToVideoPipeline`, `AnimateDiffPipeline`) and their conditioning arguments.
- Within this series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), and the forward-links to [latent video diffusion: SVD and AnimateDiff](/blog/machine-learning/video-generation/latent-video-diffusion-svd-and-animatediff), [camera control and 4D generation](/blog/machine-learning/video-generation/camera-control-and-4d-generation), [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). For the underlying image-side machinery, see [ControlNet and structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control), [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning), and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance).
